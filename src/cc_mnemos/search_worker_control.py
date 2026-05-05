"""search worker プロセスの起動補助"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

WORKER_HOST = "127.0.0.1"
WORKER_PORT = 19836
WORKER_CONNECT_TIMEOUT_SECONDS = 1.0
WORKER_PING_REQUEST = {"type": "ping"}
WORKER_PING_RESPONSE = {"ok": True}


def is_search_worker_listening(
    *,
    port: int = WORKER_PORT,
    timeout_seconds: float = WORKER_CONNECT_TIMEOUT_SECONDS,
) -> bool:
    """search worker の TCP 待受状態を確認する

    Args:
        port: 確認対象ポート
        timeout_seconds: 接続確認のタイムアウト秒数

    Returns:
        接続できる場合は True
    """
    try:
        with socket.create_connection((WORKER_HOST, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def is_search_worker_available(
    *,
    port: int = WORKER_PORT,
    timeout_seconds: float = WORKER_CONNECT_TIMEOUT_SECONDS,
) -> bool:
    """search worker が検索可能な状態か確認する

    Args:
        port: 確認対象ポート
        timeout_seconds: ping 応答待ちのタイムアウト秒数

    Returns:
        ready ping に応答する場合は True
    """
    try:
        with socket.create_connection((WORKER_HOST, port), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            request = json.dumps(WORKER_PING_REQUEST).encode("utf-8") + b"\n"
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(65536)
        payload: object = json.loads(response.decode("utf-8"))
        return payload == WORKER_PING_RESPONSE
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def get_search_worker_path() -> Path:
    """search worker スクリプトのパスを返す

    Returns:
        _search_worker.py の絶対パス
    """
    return Path(__file__).parent / "_search_worker.py"


def start_search_worker_process(
    *,
    port: int = WORKER_PORT,
    python_executable: str | None = None,
) -> subprocess.Popen[bytes]:
    """search worker プロセスを起動する

    Args:
        port: search worker が待ち受けるポート
        python_executable: 起動に使う Python 実行ファイル

    Returns:
        起動したプロセス
    """
    python = python_executable if python_executable is not None else sys.executable
    worker = str(get_search_worker_path())
    return subprocess.Popen(
        [python, worker, "--daemon", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
