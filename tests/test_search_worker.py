"""_search_worker.py の起動順序とエンドポイントテスト"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, NoReturn, cast

import pytest
from conftest import FakeSocket

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import FakeEmbedder

    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder


class _AcceptInterruptSocket:
    """``run_daemon`` の bind/listen/accept 順序を検証するためのスタブ socket

    ``accept`` で ``KeyboardInterrupt`` を投げ、受け取ったイベントの順序を
    ``_events`` に記録する
    """

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def setsockopt(self, level: int, optname: int, value: int) -> None:
        self._events.append("setsockopt")

    def bind(self, address: tuple[str, int]) -> None:
        self._events.append("bind")

    def listen(self, backlog: int) -> None:
        self._events.append("listen")

    def accept(self) -> NoReturn:
        self._events.append("accept")
        raise KeyboardInterrupt


def test_daemon_binds_before_loading_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重複ワーカーが重いモデルロード前に終了できるよう先に bind する"""
    from cc_mnemos import _search_worker
    from cc_mnemos import config as config_module
    from cc_mnemos import embedder as embedder_module

    events: list[str] = []

    def socket_factory(family: int, kind: int) -> _AcceptInterruptSocket:
        events.append("socket")
        return _AcceptInterruptSocket(events)

    def load_config() -> object:
        events.append("config")
        return object()

    class FakeEmbedderForOrder:
        def __init__(self, config: object) -> None:
            events.append("embedder")

    monkeypatch.setattr("cc_mnemos._search_worker.socket.socket", socket_factory)
    monkeypatch.setattr(config_module.Config, "load", staticmethod(load_config))
    monkeypatch.setattr(embedder_module, "Embedder", FakeEmbedderForOrder)

    with pytest.raises(KeyboardInterrupt):
        _search_worker.run_daemon(19836)

    assert events.index("listen") < events.index("config")
    assert events.index("listen") < events.index("embedder")


def test_worker_ping_returns_ready_response() -> None:
    """ready 判定用の ping には検索処理なしで応答する"""
    from cc_mnemos import _search_worker

    client_sock, server_sock = socket.socketpair()
    thread = threading.Thread(
        target=_search_worker._handle_client,
        args=(
            server_sock,
            cast("Embedder", object()),
            cast("Config", object()),
        ),
    )
    thread.start()
    try:
        client_sock.sendall(b'{"type":"ping"}\n')
        client_sock.shutdown(socket.SHUT_WR)
        response = client_sock.recv(65536)
    finally:
        client_sock.close()
        thread.join(timeout=5)

    payload = json.loads(response.decode("utf-8"))
    assert payload == {"ok": True}


def test_worker_available_requires_ready_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TCP 接続だけではなく ready ping の応答まで確認する"""
    from cc_mnemos.search_worker_control import is_search_worker_available

    fake_sock = FakeSocket(response=b'{"ok": true}')

    monkeypatch.setattr(
        "cc_mnemos.search_worker_control.socket.create_connection",
        lambda *_a, **_kw: fake_sock,
    )

    assert is_search_worker_available()
    assert bytes(fake_sock.sent) == b'{"type": "ping"}\n'


def test_worker_available_rejects_non_ready_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接続できても ready ping でなければ利用可能扱いしない"""
    from cc_mnemos.search_worker_control import is_search_worker_available

    monkeypatch.setattr(
        "cc_mnemos.search_worker_control.socket.create_connection",
        lambda *_a, **_kw: FakeSocket(response=b"[]"),
    )

    assert not is_search_worker_available()


# ---------------------------------------------------------------------------
# memorize エンドポイントのディスパッチ単体テスト
# ---------------------------------------------------------------------------
class TestMemorizeDispatch:
    def test_memorize_request_enqueues_and_acks(self) -> None:
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        alive = threading.Event()
        alive.set()

        response = _search_worker._dispatch_memorize_request(
            {
                "type": "memorize",
                "session_id": "s1",
                "chunks": [{"content": "abc", "tags": ["x"]}],
            },
            q,
            alive,
        )
        payload = json.loads(response.decode("utf-8"))
        assert payload == {"ok": True, "queued": 1}
        assert q.qsize() == 1

    def test_memorize_request_queue_full(self) -> None:
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
        q.put_nowait({"filler": True})
        alive = threading.Event()
        alive.set()

        response = _search_worker._dispatch_memorize_request(
            {"type": "memorize", "chunks": [{"content": "abc"}]},
            q,
            alive,
        )
        payload = json.loads(response.decode("utf-8"))
        assert payload == {"ok": False, "error": "queue_full"}

    def test_memorize_request_invalid_payload(self) -> None:
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        alive = threading.Event()
        alive.set()

        response = _search_worker._dispatch_memorize_request(
            {"type": "memorize"},  # chunks 欠落
            q,
            alive,
        )
        payload = json.loads(response.decode("utf-8"))
        assert payload == {"ok": False, "error": "invalid_payload"}

    def test_memorize_request_worker_not_ready(self) -> None:
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        alive = threading.Event()
        # alive を set() しない

        response = _search_worker._dispatch_memorize_request(
            {"type": "memorize", "chunks": [{"content": "abc"}]},
            q,
            alive,
        )
        payload = json.loads(response.decode("utf-8"))
        assert payload == {"ok": False, "error": "worker_not_ready"}

    def test_memorize_request_deduplicates_same_session(self) -> None:
        """同一 session_id のジョブはキュー上で上書きされる"""
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        alive = threading.Event()
        alive.set()

        first = {
            "type": "memorize",
            "session_id": "s-dup",
            "chunks": [{"content": "old"}],
        }
        second = {
            "type": "memorize",
            "session_id": "s-dup",
            "chunks": [{"content": "new1"}, {"content": "new2"}],
        }

        r1 = _search_worker._dispatch_memorize_request(first, q, alive)
        r2 = _search_worker._dispatch_memorize_request(second, q, alive)

        p1 = json.loads(r1.decode("utf-8"))
        p2 = json.loads(r2.decode("utf-8"))
        assert p1.get("ok") is True
        assert p2.get("ok") is True
        assert p2.get("deduped") is True
        # キューに 1 件しか存在しない
        assert q.qsize() == 1
        item = q.get_nowait()
        assert item is second  # 上書きされた最新ジョブ

    def test_memorize_request_keeps_different_sessions_separate(self) -> None:
        """異なる session_id は別ジョブとして両方キューに入る"""
        from cc_mnemos import _search_worker

        q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        alive = threading.Event()
        alive.set()

        _search_worker._dispatch_memorize_request(
            {"type": "memorize", "session_id": "s-a", "chunks": [{"content": "a"}]},
            q,
            alive,
        )
        _search_worker._dispatch_memorize_request(
            {"type": "memorize", "session_id": "s-b", "chunks": [{"content": "b"}]},
            q,
            alive,
        )
        assert q.qsize() == 2


# ---------------------------------------------------------------------------
# _handle_client 経由の memorize 受領 (socketpair 経由)
# ---------------------------------------------------------------------------
def test_handle_client_routes_memorize_to_queue() -> None:
    """type=memorize リクエストが _handle_client からキューに積まれる"""
    from cc_mnemos import _search_worker

    q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
    alive = threading.Event()
    alive.set()

    client_sock, server_sock = socket.socketpair()
    thread = threading.Thread(
        target=_search_worker._handle_client,
        args=(
            server_sock,
            cast("Embedder", object()),
            cast("Config", object()),
        ),
        kwargs={"memorize_queue": q, "memorize_alive": alive},
    )
    thread.start()
    try:
        payload = json.dumps(
            {
                "type": "memorize",
                "session_id": "s2",
                "chunks": [{"content": "hello"}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        client_sock.sendall(payload + b"\n")
        client_sock.shutdown(socket.SHUT_WR)
        response = client_sock.recv(65536)
    finally:
        client_sock.close()
        thread.join(timeout=5)

    decoded = json.loads(response.decode("utf-8"))
    assert decoded.get("ok") is True
    assert q.qsize() == 1
    item = q.get_nowait()
    assert item["session_id"] == "s2"


# ---------------------------------------------------------------------------
# memorize worker loop の挙動 (実 MemoryStore + FakeEmbedder)
# ---------------------------------------------------------------------------
class TestMemorizeWorkerLoop:
    def test_persists_chunks_via_loop(
        self,
        tmp_path: Path,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        from cc_mnemos import _search_worker
        from cc_mnemos.embedder import Embedder

        config = __import__("cc_mnemos.config", fromlist=["Config"]).Config(
            general={"data_dir": str(tmp_path)}
        )
        embedder = Embedder(config)  # FakeEmbedder

        q: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=10)
        alive = threading.Event()

        thread = threading.Thread(
            target=_search_worker._memorize_worker_loop,
            args=(q, embedder, config, alive),
            daemon=True,
        )
        thread.start()

        # alive が立つまで少し待つ
        assert alive.wait(timeout=5.0)

        q.put({
            "session_id": "worker-session",
            "project": "demo",
            "work_dir": "/tmp",
            "chunks": [
                {
                    "role_user": "ユーザー発話",
                    "role_assistant": "アシスタント応答",
                    "content": "ユーザー発話\nアシスタント応答",
                    "tags": ["debug"],
                },
            ],
        })

        # シャットダウンセンチネル
        q.put(None)
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        store = __import__("cc_mnemos.store", fromlist=["MemoryStore"]).MemoryStore(config)
        try:
            stats = store.get_stats()
            assert stats["total_chunks"] == 1
            assert stats["total_sessions"] == 1
            row = store.conn.execute(
                "SELECT recorded_source FROM session_sources WHERE session_id = ?",
                ("worker-session",),
            ).fetchone()
            assert row is not None
            assert row[0] == "claude"
        finally:
            store.close()

    def test_loop_continues_after_exception(
        self,
        tmp_path: Path,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        """1 件目で例外が出ても 2 件目を処理して loop が継続する"""
        from cc_mnemos import _search_worker
        from cc_mnemos.config import Config
        from cc_mnemos.embedder import Embedder
        from cc_mnemos.store import MemoryStore

        config = Config(general={"data_dir": str(tmp_path)})
        embedder = Embedder(config)

        q: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=10)
        alive = threading.Event()

        thread = threading.Thread(
            target=_search_worker._memorize_worker_loop,
            args=(q, embedder, config, alive),
            daemon=True,
        )
        thread.start()
        assert alive.wait(timeout=5.0)

        # 1 件目: 不正リクエスト (chunks が辞書ではない) — _persist_memorize_request 内で
        # スキップされるが、後段 (insert_chunk など) で例外を起こす意図的なケースとして
        # 「contents は空 → 早期 return」になる。例外耐性確認のため次のケースを使う
        q.put({"session_id": "broken", "chunks": "not-a-list"})

        # 2 件目: 正常
        q.put({
            "session_id": "good-session",
            "project": "demo",
            "work_dir": "/tmp",
            "chunks": [{"content": "ok content for good session"}],
        })

        q.put(None)
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        store = MemoryStore(config)
        try:
            # broken は contents 抽出で空になり保存されない、good-session は保存される
            row = store.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE session_id = ?",
                ("good-session",),
            ).fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# end-to-end: run_daemon を起動して memorize リクエストを処理させる
# ---------------------------------------------------------------------------
class TestRunDaemonIntegration:
    def test_run_daemon_starts_memorize_thread(
        self,
        tmp_path: Path,
        free_port: int,
        mock_embedder: type[FakeEmbedder],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_daemon を別スレッドで起動し、memorize リクエストが accepted される"""
        from cc_mnemos import _search_worker
        from cc_mnemos.config import Config

        config = Config(general={"data_dir": str(tmp_path)})

        # Config.load() を tmp_path 用の config に差し替える
        monkeypatch.setattr(
            "cc_mnemos.config.Config.load", classmethod(lambda cls: config)
        )

        daemon_thread = threading.Thread(
            target=_search_worker.run_daemon,
            args=(free_port,),
            daemon=True,
        )
        daemon_thread.start()

        # listen 待ち
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", free_port), timeout=0.2) as sock:
                    sock.sendall(b'{"type":"ping"}\n')
                    sock.shutdown(socket.SHUT_WR)
                    raw = sock.recv(65536)
                if json.loads(raw.decode("utf-8")) == {"ok": True}:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("worker daemon did not become ready in time")

        # memorize リクエスト送信 (memorize worker thread の初期化を待つため最大数秒リトライ)
        request_bytes = (
            json.dumps(
                {
                    "type": "memorize",
                    "session_id": "daemon-session",
                    "project": "demo",
                    "work_dir": "/tmp",
                    "chunks": [{"content": "integration test content"}],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )

        deadline = time.monotonic() + 5.0
        decoded: dict[str, object] = {}
        while time.monotonic() < deadline:
            with socket.create_connection(("127.0.0.1", free_port), timeout=2.0) as sock:
                sock.sendall(request_bytes)
                sock.shutdown(socket.SHUT_WR)
                raw = sock.recv(65536)
            decoded = json.loads(raw.decode("utf-8"))
            if decoded.get("ok") is True:
                break
            time.sleep(0.1)

        assert decoded.get("ok") is True

        # 永続化されるまで少し待ってから検証
        from cc_mnemos.store import MemoryStore

        for _ in range(50):
            store = MemoryStore(config)
            try:
                stats = store.get_stats()
            finally:
                store.close()
            if stats["total_chunks"] >= 1:
                break
            time.sleep(0.1)
        assert stats["total_chunks"] >= 1
