"""memorize hook クライアント — 会話ログを worker daemon に投げる軽量実装

Stop hook から起動された軽量プロセス側のロジック。重い embedding 計算と
SQLite 書き込みは常駐 worker daemon に逐次処理させ、このプロセスは
チャンク分割とタグ付け (どちらも regex のみで軽量) を済ませてから
TCP で payload を送信し、ack を受け取って即終了する

worker が起動できない／通信が失敗した場合は in-process フォールバックで
従来通り Embedder と MemoryStore を直接開いて保存する
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cc_mnemos import chunker, project, search_worker_control, tagger
from cc_mnemos._memorize_persistence import RECORDED_SOURCE, persist_chunks
from cc_mnemos.search_worker_control import WORKER_HOST, WORKER_PORT

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

# TCP 通信タイムアウト
_CONNECT_TIMEOUT_SECONDS = 2.0
_ACK_TIMEOUT_SECONDS = 5.0

# worker 起動待ちの上限。Stop hook のデフォルト timeout (30 秒) 以内で
# ack 取得 + フォールバックまで完了できるよう余裕を取って 20 秒に絞る
_WORKER_STARTUP_TIMEOUT_SECONDS = 20.0


def _extract_session_started_at(transcript_path: Path) -> str | None:
    """transcript JSONL の先頭付近から会話開始時刻 (timestamp) を取得する

    Claude Code の transcript は 1 行 1 メッセージの JSONL で、各行に
    ``timestamp`` フィールドが入っている。最初に timestamp を持つ行の値を
    会話開始時刻として返す。取得できなかった場合は ``None``
    """
    try:
        with transcript_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                ts = entry.get("timestamp")
                if isinstance(ts, str) and ts:
                    if ts.endswith("Z") and len(ts) > 1:
                        return f"{ts[:-1]}+00:00"
                    return ts
    except OSError:
        return None
    return None


def run_memorize(hook_input: Mapping[str, object], config: Config) -> None:
    """memorize hook のエントリポイント

    1. stop_hook_active チェック (無限ループ防止)
    2. transcript ファイル検証
    3. チャンク分割 (embedder 不要)
    4. タグ付け (embedder 不要、regex のみ)
    5. worker への TCP 送信を試行
    6. 失敗時は in-process フォールバックで永続化

    例外は全て握り潰してログだけ残し、hook プロセスを壊さない
    """
    try:
        _run_memorize_impl(hook_input, config)
    except Exception:
        logger.exception("memorize で予期しないエラーが発生しました")


def _run_memorize_impl(hook_input: Mapping[str, object], config: Config) -> None:
    if hook_input.get("stop_hook_active", False):
        logger.info("stop_hook_active が True のためスキップします")
        return

    transcript_path = Path(str(hook_input.get("transcript_path", ""))).expanduser()
    if not transcript_path.exists():
        logger.warning("トランスクリプトが見つかりません: %s", transcript_path)
        return

    chunks = chunker.chunk_transcript(
        transcript_path,
        max_chars=config.max_chunk_chars,
        min_chars=config.min_chunk_chars,
    )
    if not chunks:
        logger.info("チャンクが生成されませんでした")
        return

    cwd = str(hook_input.get("cwd", "."))
    project_name = project.infer_project_name(cwd, config)
    session_id = str(hook_input.get("session_id") or uuid.uuid4().hex)
    started_at = _extract_session_started_at(transcript_path) or datetime.now(
        tz=timezone.utc
    ).isoformat()

    tag_rules = config.tag_rules
    chunk_payload: list[dict[str, Any]] = []
    for c in chunks:
        tags = tagger.assign_tags(
            c.content,
            tag_rules,
            keyword_text=c.role_user,
        )
        chunk_payload.append({
            "role_user": c.role_user,
            "role_assistant": c.role_assistant,
            "content": c.content,
            "tags": tags,
        })

    payload: dict[str, object] = {
        "type": "memorize",
        "session_id": session_id,
        "project": project_name,
        "work_dir": cwd,
        "started_at": started_at,
        "chunks": chunk_payload,
    }

    if _try_send_to_worker(payload):
        logger.info(
            "セッション %s: %d チャンクを worker に投入しました",
            session_id,
            len(chunk_payload),
        )
        return

    _persist_in_process(
        session_id, project_name, cwd, chunk_payload, config, started_at
    )


def _try_send_to_worker(payload: dict[str, object]) -> bool:
    """worker daemon に memorize payload を投げて ack を待つ

    Returns:
        ack が ``{"ok": true, ...}`` で返ってきた場合のみ True
    """
    if not search_worker_control.ensure_worker(
        startup_timeout_seconds=_WORKER_STARTUP_TIMEOUT_SECONDS,
    ):
        return False

    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"

    try:
        with socket.create_connection(
            (WORKER_HOST, WORKER_PORT),
            timeout=_CONNECT_TIMEOUT_SECONDS,
        ) as sock:
            sock.settimeout(_ACK_TIMEOUT_SECONDS)
            sock.sendall(request_bytes)
            sock.shutdown(socket.SHUT_WR)

            buffer = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
    except (OSError, TimeoutError) as exc:
        logger.warning("worker への送信に失敗しました: %s", exc)
        return False

    try:
        response = json.loads(buffer.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("worker からの応答を JSON として解釈できませんでした")
        return False

    if isinstance(response, dict) and response.get("ok") is True:
        return True

    logger.warning("worker が memorize を受け付けませんでした: %s", response)
    return False


def _persist_in_process(
    session_id: str,
    project_name: str,
    cwd: str,
    chunk_payload: list[dict[str, Any]],
    config: Config,
    started_at: str,
) -> None:
    """worker 不在時の in-process フォールバック

    Embedder と MemoryStore をこのプロセス内で構築して保存する
    最終手段であり、頻発するようなら worker daemon の常駐状態を疑う
    """
    from cc_mnemos.embedder import Embedder
    from cc_mnemos.store import MemoryStore

    logger.info("worker 不在のため in-process フォールバックで保存します")

    embedder = Embedder(config)
    store = MemoryStore(config)
    try:
        persist_chunks(
            session_id=session_id,
            project_name=project_name,
            work_dir=cwd,
            chunk_records=chunk_payload,
            embedder=embedder,
            store=store,
            recorded_source=RECORDED_SOURCE,
            started_at=started_at,
        )
        logger.info(
            "in-process フォールバックでセッション %s に %d チャンクを保存しました",
            session_id,
            len(chunk_payload),
        )
    finally:
        store.close()
