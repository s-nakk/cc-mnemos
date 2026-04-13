"""Codex セッション履歴の正規化"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class NormalizedMessage:
    role: str
    text: str


@dataclass
class NormalizedSession:
    session_id: str
    cwd: str
    timestamp: str
    messages: list[NormalizedMessage]


def _normalize_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return datetime.now(tz=timezone.utc).isoformat()
    if timestamp.endswith("Z"):
        return f"{timestamp[:-1]}+00:00"
    return timestamp


def _extract_text(content: object) -> str:
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type not in ("input_text", "output_text", "text"):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def _load_codex_session_file(path: Path) -> NormalizedSession | None:
    session_id = path.stem
    cwd = ""
    timestamp: str | None = None
    messages: list[NormalizedMessage] = []

    with open(path, encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type")
            payload = entry.get("payload")

            if entry_type == "session_meta" and isinstance(payload, dict):
                payload_id = payload.get("id")
                payload_cwd = payload.get("cwd")
                payload_timestamp = payload.get("timestamp")
                if isinstance(payload_id, str) and payload_id:
                    session_id = payload_id
                if isinstance(payload_cwd, str):
                    cwd = payload_cwd
                if isinstance(payload_timestamp, str):
                    timestamp = payload_timestamp
                continue

            if entry_type != "response_item" or not isinstance(payload, dict):
                continue
            if payload.get("type") != "message":
                continue

            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue

            text = _extract_text(payload.get("content"))
            if not text:
                continue

            messages.append(NormalizedMessage(role=str(role), text=text))

    if not messages:
        return None

    return NormalizedSession(
        session_id=session_id,
        cwd=cwd,
        timestamp=_normalize_timestamp(timestamp),
        messages=messages,
    )


def _load_codex_history_file(path: Path) -> list[NormalizedSession]:
    grouped_messages: dict[str, list[NormalizedMessage]] = defaultdict(list)
    timestamps: dict[str, str] = {}

    with open(path, encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            session_id = entry.get("session_id")
            text = entry.get("text")
            ts_value = entry.get("ts")
            if not isinstance(session_id, str) or not isinstance(text, str) or not text.strip():
                continue

            grouped_messages[session_id].append(NormalizedMessage(role="user", text=text.strip()))
            if isinstance(ts_value, int):
                timestamps[session_id] = datetime.fromtimestamp(
                    ts_value, tz=timezone.utc
                ).isoformat()

    sessions: list[NormalizedSession] = []
    for session_id, messages in grouped_messages.items():
        sessions.append(
            NormalizedSession(
                session_id=session_id,
                cwd="",
                timestamp=timestamps.get(session_id, _normalize_timestamp(None)),
                messages=messages,
            )
        )
    return sessions


def load_codex_sessions(codex_dir: Path) -> list[NormalizedSession]:
    """Codex ディレクトリからセッション一覧を読み込む"""
    sessions: list[NormalizedSession] = []
    seen_ids: set[str] = set()

    sessions_dir = codex_dir / "sessions"
    if sessions_dir.exists():
        for session_file in sorted(sessions_dir.rglob("*.jsonl")):
            normalized = _load_codex_session_file(session_file)
            if normalized is None or normalized.session_id in seen_ids:
                continue
            sessions.append(normalized)
            seen_ids.add(normalized.session_id)

    history_path = codex_dir / "history.jsonl"
    if history_path.exists():
        for normalized in _load_codex_history_file(history_path):
            if normalized.session_id in seen_ids:
                continue
            sessions.append(normalized)
            seen_ids.add(normalized.session_id)

    return sessions
