"""memorize の永続化パイプライン (worker 側と in-process フォールバックの共通実装)

embedding 計算と SQLite への書き込みは worker daemon の memorize ワーカースレッドと、
worker 不在時の in-process フォールバックの両方から実行されるため、両者の挙動を
ずらさないよう共通化している。``chunk_id`` の生成方式や ``recorded_source`` の値、
トランザクションの順序が片方だけ変更されると重複排除と整合性検査が壊れる
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mnemos.embedder import Embedder
    from cc_mnemos.store import MemoryStore

logger = logging.getLogger(__name__)

# memorize hook の記録元タグ。worker 経由でも in-process でも常にこの値を保存する
RECORDED_SOURCE = "claude"


def make_chunk_id(session_id: str, index: int, content: str) -> str:
    """セッション内で安定かつ衝突しないチャンク ID を生成する

    ``index`` を含めることで「同一セッション内で同一テキストのチャンクが複数」
    というケースでも一意性が保たれる。`session_id` と `index` のみではなく
    `content` も含めるのは、後段で `delete_session_chunks` → 再 insert する際に
    同じ位置の更新コンテンツが旧 ID と区別されるようにするため
    """
    return hashlib.sha256(
        f"{session_id}:{index}:{content}".encode()
    ).hexdigest()


def persist_chunks(
    *,
    session_id: str,
    project_name: str,
    work_dir: str,
    chunk_records: list[dict[str, object]],
    embedder: Embedder,
    store: MemoryStore,
    recorded_source: str = RECORDED_SOURCE,
) -> None:
    """事前計算済みのチャンクを embedding 化して SQLite に永続化する

    ``chunk_records`` は ``{"role_user", "role_assistant", "content", "tags"}``
    を持つ辞書のリスト。``tags`` は文字列リストで、JSON 文字列としては保存しない
    生形式。本関数内で ``json.dumps`` する

    Args:
        session_id: セッション識別子。既存のチャンクは upsert で置き換える
        project_name: プロジェクト名
        work_dir: 元の作業ディレクトリ
        chunk_records: 保存するチャンクのリスト
        embedder: 埋め込み計算に使うインスタンス
        store: 書き込み先の MemoryStore
        recorded_source: ``session_sources.recorded_source`` に入れる値
    """
    contents = [str(record["content"]) for record in chunk_records]
    if not contents:
        return

    embeddings = embedder.encode_documents(contents)
    now = datetime.now(tz=timezone.utc).isoformat()

    with store.transaction():
        store.delete_session_chunks(session_id, commit=False)
        store.insert_session(
            session_id=session_id,
            project=project_name,
            work_dir=work_dir,
            started_at=now,
            recorded_source=recorded_source,
            commit=False,
        )
        for index, record in enumerate(chunk_records):
            content_str = str(record["content"])
            tags_raw = record.get("tags", [])
            tags_list = list(tags_raw) if isinstance(tags_raw, list) else []
            chunk_data: dict[str, str | int] = {
                "id": make_chunk_id(session_id, index, content_str),
                "session_id": session_id,
                "role_user": str(record.get("role_user", "")),
                "role_assistant": str(record.get("role_assistant", "")),
                "content": content_str,
                "tags": json.dumps(tags_list),
                "created_at": now,
                "token_count": len(content_str),
            }
            store.insert_chunk(chunk_data, embeddings[index], commit=False)
