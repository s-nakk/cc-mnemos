"""既存セッション履歴の一括インポート"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cc_mnemos.chunker import Chunk, chunk_transcript
from cc_mnemos.codex_history import NormalizedMessage, load_codex_sessions
from cc_mnemos.project import infer_project_name
from cc_mnemos.store import MemoryStore
from cc_mnemos.tagger import assign_tags

if TYPE_CHECKING:
    from cc_mnemos.config import Config, TagRule
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
    """cwdパスからプロジェクト名を推定

    git remote origin URLを試行し、取得できなければパスベースで推定する
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            name = url.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[:-4]
            if name:
                return name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    parts = [p for p in cwd.replace("\\", "/").split("/") if p]
    for i, part in enumerate(parts):
        if part.lower() == "projects" and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-1] if parts else "unknown"


def _read_session_metadata(path: Path) -> tuple[str | None, str | None]:
    """JSONL先頭付近から cwd と timestamp を取得する"""
    with open(path, encoding="utf-8", errors="ignore") as f:
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
            cwd = entry.get("cwd")
            timestamp = entry.get("timestamp")
            cwd_value = str(cwd) if isinstance(cwd, str) and cwd else None
            timestamp_value = str(timestamp) if isinstance(timestamp, str) and timestamp else None
            if cwd_value is not None or timestamp_value is not None:
                return cwd_value, timestamp_value
    return None, None


def _normalize_timestamp(timestamp: str | None) -> str | None:
    """datetime.fromisoformat が扱いやすい ISO 文字列へ正規化する"""
    if timestamp is None:
        return None
    if timestamp.endswith("Z"):
        return f"{timestamp[:-1]}+00:00"
    return timestamp


def import_history(
    config: Config,
    embedder: Embedder | None = None,
    verbose: bool = True,
    device: str | None = None,
    agent: str = "claude",
) -> dict[str, int]:
    """既存セッション履歴を一括インポート

    Args:
        config: アプリケーション設定
        embedder: 共有Embedderインスタンス (None の場合は内部で生成)
        verbose: 進捗表示を行うかどうか

    Returns:
        ``{"imported": int, "skipped": int, "errors": int}``
    """
    if agent == "codex":
        return _import_codex_history(
            config,
            embedder=embedder,
            verbose=verbose,
            device=device,
        )

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
        embedder = EmbedderClass(config, device=device)

    imported = 0
    errors = 0
    start = time.time()

    tag_rules = config.tag_rules
    project_name_cache: dict[str, str] = {}

    for i, jsonl in enumerate(sorted(all_files)):
        try:
            rel = jsonl.relative_to(projects_dir)
            project_dir_name = rel.parts[0]
            metadata_cwd, metadata_timestamp = _read_session_metadata(jsonl)
            cwd = metadata_cwd if metadata_cwd is not None else _resolve_cwd(project_dir_name)

            chunks = chunk_transcript(jsonl, config.max_chunk_chars, config.min_chunk_chars)
            if not chunks:
                continue

            project = project_name_cache.get(cwd)
            if project is None:
                if metadata_cwd is not None:
                    project = infer_project_name(cwd, config)
                else:
                    project = _infer_project(cwd)
                project_name_cache[cwd] = project

            normalized_timestamp = _normalize_timestamp(metadata_timestamp)
            if normalized_timestamp is not None:
                session_time = normalized_timestamp
            else:
                mtime = jsonl.stat().st_mtime
                session_time = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            with store.transaction():
                store.insert_session(
                    session_id=jsonl.stem,
                    project=project,
                    work_dir=cwd,
                    started_at=session_time,
                    ended_at=session_time,
                    recorded_source="claude",
                    commit=False,
                )

                embeddings = embedder.encode_documents([c.content for c in chunks])

                for idx, chunk in enumerate(chunks):
                    tags = assign_tags(
                        chunk.content,
                        tag_rules,
                        keyword_text=chunk.role_user,
                    )
                    # 決定論的ID: 同一セッション+同一コンテンツは常に同じID
                    chunk_id = hashlib.sha256(
                        f"{jsonl.stem}:{chunk.content}".encode()
                    ).hexdigest()
                    chunk_data: dict[str, str | int] = {
                        "id": chunk_id,
                        "session_id": jsonl.stem,
                        "role_user": chunk.role_user,
                        "role_assistant": chunk.role_assistant,
                        "content": chunk.content,
                        "tags": json.dumps(tags),
                        "created_at": session_time,
                        "token_count": len(chunk.content),
                    }
                    store.insert_chunk(chunk_data, embeddings[idx], commit=False)

            # Embeddingテンソルを明示的に解放
            del embeddings
            imported += 1
        except Exception:
            errors += 1
            if verbose and errors <= 5:
                logger.exception("  ERROR %s", jsonl.name[:30])

        # 50セッションごとにGC + CUDAキャッシュ解放
        if (i + 1) % 50 == 0:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

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


def _messages_to_chunks(
    messages: list[NormalizedMessage],
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    normalized_entries = [
        {"type": "user" if message.role == "user" else "assistant", "content": message.text}
        for message in messages
    ]

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".jsonl", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        for entry in normalized_entries:
            temp_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    try:
        chunks = chunk_transcript(temp_path, max_chars=max_chars, min_chars=min_chars)
    finally:
        temp_path.unlink(missing_ok=True)

    if chunks:
        return chunks

    user_text = "\n".join(message.text for message in messages if message.role == "user").strip()
    if len(user_text) < min_chars:
        return []

    return [Chunk(role_user=user_text, role_assistant="", content=user_text)]


def _insert_chunks_for_session(
    store: MemoryStore,
    *,
    session_id: str,
    project: str,
    cwd: str,
    session_time: str,
    chunks: list[Chunk],
    embedder: Embedder,
    tag_rules: dict[str, TagRule],
    recorded_source: str,
) -> None:
    with store.transaction():
        store.insert_session(
            session_id=session_id,
            project=project,
            work_dir=cwd,
            started_at=session_time,
            ended_at=session_time,
            recorded_source=recorded_source,
            commit=False,
        )

        embeddings = embedder.encode_documents([chunk.content for chunk in chunks])
        for idx, chunk in enumerate(chunks):
            tags = assign_tags(
                chunk.content,
                tag_rules,
                keyword_text=chunk.role_user,
            )
            chunk_id = hashlib.sha256(
                f"{session_id}:{chunk.content}".encode()
            ).hexdigest()
            chunk_data: dict[str, str | int] = {
                "id": chunk_id,
                "session_id": session_id,
                "role_user": chunk.role_user,
                "role_assistant": chunk.role_assistant,
                "content": chunk.content,
                "tags": json.dumps(tags),
                "created_at": session_time,
                "token_count": len(chunk.content),
            }
            store.insert_chunk(chunk_data, embeddings[idx], commit=False)


def _import_codex_history(
    config: Config,
    embedder: Embedder | None = None,
    verbose: bool = True,
    device: str | None = None,
) -> dict[str, int]:
    codex_dir = Path.home() / ".codex"
    if not codex_dir.exists():
        if verbose:
            print("Codex directory not found")
        return {"imported": 0, "skipped": 0, "errors": 0}

    sessions = load_codex_sessions(codex_dir)
    if not sessions:
        if verbose:
            print("No new Codex sessions to import")
        return {"imported": 0, "skipped": 0, "errors": 0}

    store = MemoryStore(config)
    existing = set(
        row[0]
        for row in store.conn.execute("SELECT session_id FROM sessions").fetchall()
    )
    pending_sessions = [
        session for session in sessions if session.session_id not in existing
    ]
    if not pending_sessions:
        if verbose:
            print("No new sessions to import")
        store.close()
        return {"imported": 0, "skipped": len(existing), "errors": 0}

    if embedder is None:
        from cc_mnemos.embedder import Embedder as EmbedderClass

        embedder = EmbedderClass(config, device=device)

    imported = 0
    errors = 0
    tag_rules = config.tag_rules

    for session in pending_sessions:
        try:
            chunks = _messages_to_chunks(
                session.messages,
                max_chars=config.max_chunk_chars,
                min_chars=config.min_chunk_chars,
            )
            if not chunks:
                continue

            cwd = session.cwd
            project = _infer_project(cwd) if cwd else "codex"
            _insert_chunks_for_session(
                store,
                session_id=session.session_id,
                project=project,
                cwd=cwd,
                session_time=session.timestamp,
                chunks=chunks,
                embedder=embedder,
                tag_rules=tag_rules,
                recorded_source="codex",
            )
            imported += 1
        except Exception:
            errors += 1
            if verbose and errors <= 5:
                logger.exception("  ERROR %s", session.session_id[:30])

    store.close()
    return {"imported": imported, "skipped": len(existing), "errors": errors}
