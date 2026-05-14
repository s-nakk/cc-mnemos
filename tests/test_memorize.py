"""memorize モジュールのテスト

memorize は worker daemon にチャンクと事前計算したタグを投げ、
ack を受け取るだけの軽量 TCP クライアント。worker が起動できない／
通信失敗時には in-process フォールバックで保存処理を継続する
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conftest import FakeSocket

from cc_mnemos.config import Config
from cc_mnemos.memorize import _extract_session_started_at, run_memorize
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from conftest import FakeEmbedder, TranscriptFactory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestMemorizePipeline:
    """既存パイプライン挙動の回帰テスト（worker 不在 → in-process フォールバック経路）"""

    def test_full_pipeline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # worker を立てずに in-process フォールバックを検証する
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: False,
        )

        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-001",
            "transcript_path": str(FIXTURES_DIR / "sample_transcript.jsonl"),
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }
        run_memorize(hook_input, config)
        store = MemoryStore(config)
        stats = store.get_stats()
        assert stats["total_chunks"] >= 2
        source_row = store.conn.execute(
            """
            SELECT recorded_source
            FROM session_sources
            WHERE session_id = ?
            """,
            ("test-session-001",),
        ).fetchone()
        assert source_row is not None
        assert source_row[0] == "claude"
        store.close()

    def test_skips_when_stop_hook_active(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-002",
            "transcript_path": str(FIXTURES_DIR / "sample_transcript.jsonl"),
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }
        run_memorize(hook_input, config)
        store = MemoryStore(config)
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        store.close()

    def test_handles_missing_transcript(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-003",
            "transcript_path": "/nonexistent/path.jsonl",
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }
        run_memorize(hook_input, config)  # クラッシュしないこと

    def test_started_at_from_transcript_is_persisted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """transcript JSONL の最初のメッセージ timestamp が sessions.started_at に保存される

        worker 不在 → in-process フォールバック経路で、JSONL 先頭の timestamp が
        会話開始時刻として正しく伝搬することを確認する (Item 1)
        """
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: False,
        )

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join([
                json.dumps({
                    "type": "user",
                    "timestamp": "2026-01-15T08:30:00Z",
                    "message": {"content": "border-radius の設定方法を教えて"},
                }, ensure_ascii=False),
                json.dumps({
                    "type": "assistant",
                    "timestamp": "2026-01-15T08:30:05Z",
                    "message": {"content": "CSS の border-radius で角丸を設定できます"},
                }, ensure_ascii=False),
            ]) + "\n",
            encoding="utf-8",
        )

        config = Config(general={"data_dir": str(tmp_path / "data")})
        hook_input = {
            "session_id": "test-session-started-at",
            "transcript_path": str(transcript_path),
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }
        run_memorize(hook_input, config)

        store = MemoryStore(config)
        try:
            row = store.conn.execute(
                "SELECT started_at FROM sessions WHERE session_id = ?",
                ("test-session-started-at",),
            ).fetchone()
            assert row is not None
            assert row[0] == "2026-01-15T08:30:00+00:00"
        finally:
            store.close()


class TestExtractSessionStartedAt:
    """_extract_session_started_at の単体テスト"""

    def test_extracts_first_timestamp(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            "\n".join([
                json.dumps({"type": "user", "timestamp": "2026-03-25T01:23:45Z"}),
                json.dumps({"type": "assistant", "timestamp": "2026-03-25T01:23:50Z"}),
            ]) + "\n",
            encoding="utf-8",
        )
        assert _extract_session_started_at(transcript) == "2026-03-25T01:23:45+00:00"

    def test_skips_lines_without_timestamp(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            "\n".join([
                json.dumps({"type": "user"}),
                json.dumps({"type": "assistant", "timestamp": "2026-03-25T02:00:00+09:00"}),
            ]) + "\n",
            encoding="utf-8",
        )
        assert _extract_session_started_at(transcript) == "2026-03-25T02:00:00+09:00"

    def test_returns_none_when_no_timestamp(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n",
            encoding="utf-8",
        )
        assert _extract_session_started_at(transcript) is None

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert _extract_session_started_at(tmp_path / "missing.jsonl") is None

    def test_tolerates_malformed_json_lines(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            "\n".join([
                "not-json",
                json.dumps({"type": "user", "timestamp": "2026-03-25T03:00:00Z"}),
            ]) + "\n",
            encoding="utf-8",
        )
        assert _extract_session_started_at(transcript) == "2026-03-25T03:00:00+00:00"

    def test_lone_z_timestamp_is_returned_as_is(self, tmp_path: Path) -> None:
        """``"Z"`` 一文字の不正な timestamp に対して ``"+00:00"`` を返さない

        Z だけの値に対して ``ts[:-1]`` を適用すると ``""`` になり、結果が
        ``"+00:00"`` という壊れた timestamp になってしまう。長さガードで
        Z 正規化を回避し、生の値を返すことで「壊れた timestamp は壊れたまま」
        後段に渡す
        """
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "timestamp": "Z"}) + "\n",
            encoding="utf-8",
        )
        assert _extract_session_started_at(transcript) == "Z"


# ---------------------------------------------------------------------------
# stop_hook_active / transcript 不在では worker への接続も試みない
# ---------------------------------------------------------------------------
class TestEarlyReturn:
    def test_stop_hook_active_does_not_contact_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_create_connection(*_a: object, **_kw: object) -> FakeSocket:
            raise AssertionError("stop_hook_active 時は worker に接続してはいけない")

        def fake_ensure(**_: object) -> bool:
            raise AssertionError("stop_hook_active 時は ensure_worker も呼んではいけない")

        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection", fake_create_connection
        )
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker", fake_ensure
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "stop_hook_active": True,
                "transcript_path": str(tmp_path / "x.jsonl"),
            },
            config,
        )

    def test_missing_transcript_does_not_contact_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*_a: object, **_kw: object) -> FakeSocket:
            raise AssertionError("transcript 不在時は worker に接続してはいけない")

        def fail_ensure(**_: object) -> bool:
            raise AssertionError("transcript 不在時は ensure_worker も呼んではいけない")

        monkeypatch.setattr("cc_mnemos.memorize.socket.create_connection", boom)
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker", fail_ensure
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "transcript_path": str(tmp_path / "missing.jsonl"),
                "session_id": "x",
            },
            config,
        )


# ---------------------------------------------------------------------------
# worker への送信 — payload の中身と ack 取得
# ---------------------------------------------------------------------------
class TestWorkerHappyPath:
    def test_sends_memorize_payload_to_worker(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )

        fake_socket = FakeSocket(response=json.dumps({"ok": True, "queued": 2}).encode())
        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection",
            lambda *_a, **_kw: fake_socket,
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "test-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )

        sent_bytes = bytes(fake_socket.sent).rstrip(b"\n")
        payload = json.loads(sent_bytes.decode("utf-8"))
        assert payload["type"] == "memorize"
        assert payload["session_id"] == "test-session"
        assert payload["work_dir"] == str(tmp_path)
        assert isinstance(payload["chunks"], list)
        assert len(payload["chunks"]) >= 1
        first = payload["chunks"][0]
        assert "role_user" in first
        assert "role_assistant" in first
        assert "content" in first
        assert isinstance(first["tags"], list)

    def test_generates_session_id_when_missing(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )
        fake_socket = FakeSocket(response=json.dumps({"ok": True, "queued": 1}).encode())
        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection",
            lambda *_a, **_kw: fake_socket,
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {"transcript_path": str(transcript), "cwd": str(tmp_path)},
            config,
        )

        payload = json.loads(bytes(fake_socket.sent).rstrip(b"\n").decode("utf-8"))
        assert isinstance(payload["session_id"], str)
        assert len(payload["session_id"]) > 0

    def test_does_not_persist_when_worker_acks(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """worker が ack を返した場合は in-process フォールバックを走らせない"""
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )
        fake_socket = FakeSocket(response=json.dumps({"ok": True, "queued": 1}).encode())
        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection",
            lambda *_a, **_kw: fake_socket,
        )

        # Embedder が呼ばれたらフォールバックが走った証拠
        def embedder_should_not_be_called(*_a: object, **_kw: object) -> object:
            raise AssertionError("worker ack 取得時は Embedder を構築してはいけない")

        monkeypatch.setattr(
            "cc_mnemos.embedder.Embedder", embedder_should_not_be_called
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "ack-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )


# ---------------------------------------------------------------------------
# worker 不在・通信失敗 → in-process フォールバック
# ---------------------------------------------------------------------------
class TestFallback:
    def test_worker_unavailable_falls_back_to_in_process(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: False,
        )

        def fake_create_connection(*_a: object, **_kw: object) -> FakeSocket:
            raise AssertionError("worker 不在時は socket 接続を試みてはいけない")

        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection", fake_create_connection
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "fallback-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )

        store = MemoryStore(config)
        try:
            stats = store.get_stats()
            assert stats["total_chunks"] >= 1
            assert stats["total_sessions"] >= 1
        finally:
            store.close()

    def test_ack_timeout_falls_back_in_process(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )
        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection",
            lambda *_a, **_kw: FakeSocket(recv_error=TimeoutError("ack timeout")),
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "timeout-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )

        store = MemoryStore(config)
        try:
            assert store.get_stats()["total_chunks"] >= 1
        finally:
            store.close()

    def test_queue_full_response_falls_back_in_process(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )
        full = json.dumps({"ok": False, "error": "queue_full"}).encode()
        monkeypatch.setattr(
            "cc_mnemos.memorize.socket.create_connection",
            lambda *_a, **_kw: FakeSocket(response=full),
        )

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "queue-full-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )

        store = MemoryStore(config)
        try:
            assert store.get_stats()["total_chunks"] >= 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 例外耐性 — hook を壊さない
# ---------------------------------------------------------------------------
class TestExceptionResilience:
    def test_socket_error_does_not_raise(
        self,
        tmp_path: Path,
        transcript_factory: TranscriptFactory,
        monkeypatch: pytest.MonkeyPatch,
        mock_embedder: type[FakeEmbedder],
    ) -> None:
        transcript = transcript_factory()
        monkeypatch.setattr(
            "cc_mnemos.memorize.search_worker_control.ensure_worker",
            lambda **_: True,
        )

        def boom(*_a: object, **_kw: object) -> FakeSocket:
            raise OSError("connection refused")

        monkeypatch.setattr("cc_mnemos.memorize.socket.create_connection", boom)

        config = Config(general={"data_dir": str(tmp_path)})
        run_memorize(
            {
                "session_id": "boom-session",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            },
            config,
        )
