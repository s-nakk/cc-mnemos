"""search worker プロセスの起動補助"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WORKER_HOST = "127.0.0.1"
WORKER_PORT = 19836
WORKER_CONNECT_TIMEOUT_SECONDS = 1.0
WORKER_PING_REQUEST = {"type": "ping"}
WORKER_PING_RESPONSE = {"ok": True}

# ensure_worker のデフォルト値
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_STARTUP_POLL_SECONDS = 0.5


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


def ensure_worker(
    *,
    port: int = WORKER_PORT,
    startup_timeout_seconds: float | None = None,
    poll_interval_seconds: float | None = None,
) -> bool:
    """search worker が ready 状態になるまで待機し、必要なら起動する

    呼び出し元（``cc_mnemos.server`` の MCP ハンドラ、``cc_mnemos.memorize``
    の hook クライアント等）の双方から共通利用できるよう、プロセスローカル
    なキャッシュフラグは持たない。呼び出し側で短寿命のフラグを管理する想定

    Args:
        port: worker が待ち受けるポート
        startup_timeout_seconds: 起動完了を待つ最大秒数 (``None`` ならモジュール定数を使用)
        poll_interval_seconds: ready 判定のポーリング間隔 (``None`` ならモジュール定数を使用)

    Returns:
        ready が確認できた場合は True
    """
    # モジュール定数を実行時に解決することで、テストから monkeypatch で短縮可能にする
    timeout_seconds = (
        startup_timeout_seconds
        if startup_timeout_seconds is not None
        else DEFAULT_STARTUP_TIMEOUT_SECONDS
    )
    poll_seconds = (
        poll_interval_seconds
        if poll_interval_seconds is not None
        else DEFAULT_STARTUP_POLL_SECONDS
    )

    if is_search_worker_available():
        return True

    # 待受中ならロード完了を待つ。未起動の場合だけ新規起動する
    if not is_search_worker_listening():
        try:
            start_search_worker_process(port=port)
        except Exception:  # noqa: BLE001
            logger.exception("search worker の起動に失敗しました")
            return False

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_search_worker_available():
            return True
        time.sleep(poll_seconds)

    logger.error(
        "search worker が %.1f 秒以内に起動しませんでした",
        timeout_seconds,
    )
    return False
