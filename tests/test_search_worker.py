"""_search_worker.py の起動順序テスト"""

from __future__ import annotations

import json
import socket
import threading
from types import TracebackType
from typing import TYPE_CHECKING, Literal, NoReturn, cast

import pytest

if TYPE_CHECKING:
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder


class _FakeSocket:
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


class _FakeReadySocket:
    def __init__(self, response: bytes, sent: list[bytes]) -> None:
        self._response = response
        self._sent = sent

    def __enter__(self) -> _FakeReadySocket:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def settimeout(self, timeout_seconds: float) -> None:
        return

    def sendall(self, data: bytes) -> None:
        self._sent.append(data)

    def shutdown(self, how: int) -> None:
        return

    def recv(self, bufsize: int) -> bytes:
        return self._response


def test_daemon_binds_before_loading_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重複ワーカーが重いモデルロード前に終了できるよう先に bind する"""
    from cc_mnemos import _search_worker
    from cc_mnemos import config as config_module
    from cc_mnemos import embedder as embedder_module

    events: list[str] = []

    def socket_factory(family: int, kind: int) -> _FakeSocket:
        events.append("socket")
        return _FakeSocket(events)

    def load_config() -> object:
        events.append("config")
        return object()

    class FakeEmbedder:
        def __init__(self, config: object) -> None:
            events.append("embedder")

    monkeypatch.setattr("cc_mnemos._search_worker.socket.socket", socket_factory)
    monkeypatch.setattr(config_module.Config, "load", staticmethod(load_config))
    monkeypatch.setattr(embedder_module, "Embedder", FakeEmbedder)

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

    sent: list[bytes] = []

    def create_connection(
        address: tuple[str, int],
        timeout: float,
    ) -> _FakeReadySocket:
        return _FakeReadySocket(b'{"ok": true}', sent)

    monkeypatch.setattr(
        "cc_mnemos.search_worker_control.socket.create_connection",
        create_connection,
    )

    assert is_search_worker_available()
    assert sent == [b'{"type": "ping"}\n']


def test_worker_available_rejects_non_ready_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接続できても ready ping でなければ利用可能扱いしない"""
    from cc_mnemos.search_worker_control import is_search_worker_available

    def create_connection(
        address: tuple[str, int],
        timeout: float,
    ) -> _FakeReadySocket:
        return _FakeReadySocket(b"[]", [])

    monkeypatch.setattr(
        "cc_mnemos.search_worker_control.socket.create_connection",
        create_connection,
    )

    assert not is_search_worker_available()
