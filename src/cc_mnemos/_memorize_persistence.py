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
    """事前計算済みのチャンクを差分更新で SQLite に永続化する

    ``chunk_records`` は ``{"role_user", "role_assistant", "content", "tags"}``
    を持つ辞書のリスト。``tags`` は文字列リストで、JSON 文字列としては保存しない
    生形式。本関数内で ``json.dumps`` する

    Stop hook はターン毎に発火し、``chunker.chunk_transcript`` は会話全体を再パース
    して同じ chunk を毎回返す。`chunk_id` は ``make_chunk_id(session_id, index,
    content)`` で決定論的に決まるため、既に DB にある ID を再 embed する必要はない
    差分更新の流れ:
      1. 新 chunk_id 一覧を計算
      2. DB 上の既存 chunk_id 集合を取得
      3. ``existing - new`` を削除
      4. ``new - existing`` の content だけ embed + insert (既存 chunk は触らない)

    これにより、ターン進行に伴う 1 ジョブあたりの embed 計算量はおおむね
    その 1 ターンで増えた新規 chunk 数 (典型的には 0〜1 件) で頭打ちになる

    Args:
        session_id: セッション識別子
        project_name: プロジェクト名
        work_dir: 元の作業ディレクトリ
        chunk_records: 保存するチャンクのリスト
        embedder: 埋め込み計算に使うインスタンス
        store: 書き込み先の MemoryStore
        recorded_source: ``session_sources.recorded_source`` に入れる値
    """
    if not chunk_records:
        return

    new_ids: list[str] = [
        make_chunk_id(session_id, index, str(record["content"]))
        for index, record in enumerate(chunk_records)
    ]

    existing_ids = store.get_session_chunk_ids(session_id)
    new_id_set = set(new_ids)
    to_delete = list(existing_ids - new_id_set)
    to_insert_indices = [
        index for index, chunk_id in enumerate(new_ids) if chunk_id not in existing_ids
    ]

    contents_to_embed = [
        str(chunk_records[index]["content"]) for index in to_insert_indices
    ]
    new_embeddings = (
        embedder.encode_documents(contents_to_embed) if contents_to_embed else None
    )

    now = datetime.now(tz=timezone.utc).isoformat()

    with store.transaction():
        if to_delete:
            store.delete_chunks_by_ids(to_delete, commit=False)
        store.insert_session(
            session_id=session_id,
            project=project_name,
            work_dir=work_dir,
            started_at=now,
            recorded_source=recorded_source,
            commit=False,
        )
        if new_embeddings is not None:
            for offset, original_index in enumerate(to_insert_indices):
                record = chunk_records[original_index]
                content_str = str(record["content"])
                tags_raw = record.get("tags", [])
                tags_list = list(tags_raw) if isinstance(tags_raw, list) else []
                chunk_data: dict[str, str | int] = {
                    "id": new_ids[original_index],
                    "session_id": session_id,
                    "role_user": str(record.get("role_user", "")),
                    "role_assistant": str(record.get("role_assistant", "")),
                    "content": content_str,
                    "tags": json.dumps(tags_list),
                    "created_at": now,
                    "token_count": len(content_str),
                }
                store.insert_chunk(chunk_data, new_embeddings[offset], commit=False)

    skipped = len(existing_ids & new_id_set)
    logger.debug(
        "persist_chunks session=%s skip=%d delete=%d insert=%d",
        session_id,
        skipped,
        len(to_delete),
        len(to_insert_indices),
    )
