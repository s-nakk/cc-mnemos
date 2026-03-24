"""既存セッション履歴の一括インポート"""

from __future__ import annotations

import gc
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cc_mnemos.chunker import chunk_transcript
from cc_mnemos.store import MemoryStore
from cc_mnemos.tagger import assign_tags

if TYPE_CHECKING:
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder

logger = logging.getLogger(__name__)


def _resolve_cwd(project_dir_name: str) -> str:
    """プロジェクトディレクトリ名からcwdを復元

    Args:
        project_dir_name: プロジェクトディレクトリ名

    Returns:
        復元されたcwdパス文字列
    """
    cwd = project_dir_name.replace("--", "/").replace("-", "/")
    if len(cwd) >= 2 and cwd[1] == "/":
        cwd = cwd[0].upper() + ":" + cwd[1:]
    return cwd


def _infer_project(cwd: str) -> str:
    """cwdパスからプロジェクト名を推定（subprocess不使用）"""
    parts = [p for p in cwd.replace("\\", "/").split("/") if p]
    for i, part in enumerate(parts):
        if part.lower() == "projects" and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-1] if parts else "unknown"


def import_history(
    config: Config,
    embedder: Embedder | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """既存のClaude Codeセッション履歴を一括インポート

    Args:
        config: アプリケーション設定
        embedder: 共有Embedderインスタンス (None の場合は内部で生成)
        verbose: 進捗表示を行うかどうか

    Returns:
        ``{"imported": int, "skipped": int, "errors": int}``
    """
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        if verbose:
            print("Claude Code projects directory not found")
        return {"imported": 0, "skipped": 0, "errors": 0}

    store = MemoryStore(config)

    existing = set(
        row[0]
        for row in store.conn.execute("SELECT session_id FROM sessions").fetchall()
    )

    all_files: list[Path] = []
    for jsonl in projects_dir.rglob("*.jsonl"):
        if "subagents" in jsonl.parts:
            continue
        if jsonl.stem in existing:
            continue
        all_files.append(jsonl)

    if not all_files:
        if verbose:
            print("No new sessions to import")
        store.close()
        return {"imported": 0, "skipped": len(existing), "errors": 0}

    if verbose:
        print(f"Importing {len(all_files)} sessions (skipping {len(existing)} already imported)...")

    # Embedderを1回だけロード
    if embedder is None:
        from cc_mnemos.embedder import Embedder as EmbedderClass
        embedder = EmbedderClass(config)

    imported = 0
    errors = 0
    start = time.time()

    for i, jsonl in enumerate(sorted(all_files)):
        try:
            rel = jsonl.relative_to(projects_dir)
            project_dir_name = rel.parts[0]
            cwd = _resolve_cwd(project_dir_name)

            chunks = chunk_transcript(jsonl, config.max_chunk_tokens, config.min_chunk_tokens)
            if not chunks:
                continue

            project = _infer_project(cwd)
            now_str = datetime.now(tz=timezone.utc).isoformat()

            store.insert_session(
                session_id=jsonl.stem,
                project=project,
                work_dir=cwd,
                started_at=now_str,
                ended_at=now_str,
            )

            # バッチエンコード
            embeddings = embedder.encode_documents([c.content for c in chunks])

            for idx, chunk in enumerate(chunks):
                tags = assign_tags(chunk.role_user, config.tag_rules)
                chunk_data: dict[str, str | int] = {
                    "id": str(uuid.uuid4()),
                    "session_id": jsonl.stem,
                    "role_user": chunk.role_user,
                    "role_assistant": chunk.role_assistant,
                    "content": chunk.content,
                    "tags": json.dumps(tags),
                    "created_at": now_str,
                    "token_count": len(chunk.content.split()),
                }
                store.insert_chunk(chunk_data, embeddings[idx])

            # Embeddingテンソルを明示的に解放
            del embeddings
            imported += 1
        except Exception:
            errors += 1
            if verbose and errors <= 5:
                logger.exception("  ERROR %s", jsonl.name[:30])

        # 50セッションごとにGC実行してメモリ解放
        if (i + 1) % 50 == 0:
            gc.collect()

        if verbose:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(all_files) - i - 1) / rate if rate > 0 else 0
            bar_width = 30
            progress = (i + 1) / len(all_files)
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  {bar} {i + 1}/{len(all_files)} "
                f"({imported} saved, {errors} err) "
                f"~{remaining:.0f}s left",
                end="", flush=True,
            )

    store.close()
    elapsed = time.time() - start

    # 最終GC
    gc.collect()

    if verbose:
        print()
        print(f"Done: {imported} sessions imported, {errors} errors ({elapsed:.0f}s)")

    return {"imported": imported, "skipped": len(existing), "errors": errors}
