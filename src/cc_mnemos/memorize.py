"""保存パイプライン — 会話ログを分割・タグ付け・埋め込み・保存する"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cc_mnemos import chunker, project, tagger
from cc_mnemos.embedder import Embedder
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)


def run_memorize(hook_input: dict[str, object], config: Config) -> None:
    """会話ログの保存パイプラインを実行する

    1. stop_hook_active チェック (無限ループ防止)
    2. トランスクリプトファイルの検証
    3. チャンク分割
    4. 埋め込みベクトル生成
    5. タグ付け
    6. SQLite へ永続化

    Args:
        hook_input: フック入力 (session_id, transcript_path, cwd など)
        config: アプリケーション設定
    """
    try:
        _run_memorize_impl(hook_input, config)
    except Exception:
        logger.exception("memorize パイプラインでエラーが発生しました")


def _run_memorize_impl(hook_input: dict[str, object], config: Config) -> None:
    """パイプラインの内部実装"""
    # 1. 無限ループ防止: stop_hook_active が True なら即座にリターン
    if hook_input.get("stop_hook_active", False):
        logger.info("stop_hook_active が True のためスキップします")
        return

    # 2. トランスクリプトファイルの検証
    transcript_path_str = str(hook_input.get("transcript_path", ""))
    transcript_path = Path(transcript_path_str).expanduser()
    if not transcript_path.exists():
        logger.warning("トランスクリプトが見つかりません: %s", transcript_path)
        return

    # 3. チャンク分割
    chunks = chunker.chunk_transcript(
        transcript_path,
        max_chars=config.max_chunk_chars,
        min_chars=config.min_chunk_chars,
    )
    if not chunks:
        logger.info("チャンクが生成されませんでした")
        return

    # 4. プロジェクト名の推定
    cwd = str(hook_input.get("cwd", "."))
    project_name = project.infer_project_name(cwd, config)

    # 5. 埋め込みベクトル生成
    embedder = Embedder(config)
    embeddings = embedder.encode_documents([c.content for c in chunks])

    # 6. タグ付け (キーワードはrole_userで判定、embeddingフォールバックは
    #    現在のprototype文では精度が出ないため無効化)
    tag_rules = config.tag_rules
    chunk_tags_list: list[list[str]] = []
    for c in chunks:
        tags = tagger.assign_tags(
            c.content,
            tag_rules,
            keyword_text=c.role_user,
        )
        chunk_tags_list.append(tags)

    # 7. SQLite へ永続化
    now = datetime.now(tz=timezone.utc).isoformat()
    session_id = str(hook_input.get("session_id", str(uuid.uuid4())))

    store = MemoryStore(config)
    try:
        store.insert_session(
            session_id=session_id,
            project=project_name,
            work_dir=cwd,
            started_at=now,
        )

        for i, c in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_data: dict[str, str | int] = {
                "id": chunk_id,
                "session_id": session_id,
                "role_user": c.role_user,
                "role_assistant": c.role_assistant,
                "content": c.content,
                "tags": json.dumps(chunk_tags_list[i]),
                "created_at": now,
                "token_count": len(c.content),
            }
            store.insert_chunk(chunk_data, embeddings[i])

        logger.info(
            "セッション %s: %d チャンクを保存しました",
            session_id,
            len(chunks),
        )
    finally:
        store.close()
