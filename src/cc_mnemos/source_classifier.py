"""セッションの記録元を判定する補助モジュール"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

RecordedSource = Literal["claude", "codex"]
SourceClassification = Literal["claude", "codex", "unknown"]
SourceConfidence = Literal["high", "medium", "low"]


class SourceClassificationResult(TypedDict):
    source_classification: SourceClassification
    source_classification_confidence: SourceConfidence
    source_classification_reason: list[str]


def discover_claude_session_ids(home_dir: Path) -> set[str]:
    """Claude Code のローカル履歴から session_id の集合を返す"""
    projects_dir = home_dir / ".claude" / "projects"
    if not projects_dir.exists():
        return set()

    session_ids: set[str] = set()
    for jsonl_path in projects_dir.rglob("*.jsonl"):
        if "subagents" in jsonl_path.parts:
            continue
        session_ids.add(jsonl_path.stem)
    return session_ids


def discover_codex_session_ids(home_dir: Path) -> set[str]:
    """Codex のローカル履歴から session_id の集合を返す"""
    codex_dir = home_dir / ".codex"
    if not codex_dir.exists():
        return set()

    session_ids: set[str] = set()

    sessions_dir = codex_dir / "sessions"
    if sessions_dir.exists():
        for session_file in sessions_dir.rglob("*.jsonl"):
            session_ids.add(session_file.stem)
            try:
                with open(session_file, encoding="utf-8", errors="ignore") as file_obj:
                    for line in file_obj:
                        raw = line.strip()
                        if not raw:
                            continue
                        entry = json.loads(raw)
                        if not isinstance(entry, dict):
                            continue
                        if entry.get("type") != "session_meta":
                            continue
                        payload = entry.get("payload")
                        if not isinstance(payload, dict):
                            continue
                        session_id = payload.get("id")
                        if isinstance(session_id, str) and session_id:
                            session_ids.add(session_id)
                            break
            except OSError:
                continue
            except json.JSONDecodeError:
                continue

    history_path = codex_dir / "history.jsonl"
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8", errors="ignore") as file_obj:
                for line in file_obj:
                    raw = line.strip()
                    if not raw:
                        continue
                    entry = json.loads(raw)
                    if not isinstance(entry, dict):
                        continue
                    session_id = entry.get("session_id")
                    if isinstance(session_id, str) and session_id:
                        session_ids.add(session_id)
        except OSError:
            return session_ids
        except json.JSONDecodeError:
            return session_ids

    return session_ids


def classify_session_source(
    *,
    session_id: str,
    work_dir: str,
    claude_session_ids: set[str],
    codex_session_ids: set[str],
) -> SourceClassificationResult:
    """既存セッションの記録元を推定する"""
    matched_claude = session_id in claude_session_ids
    matched_codex = session_id in codex_session_ids

    if matched_claude and not matched_codex:
        return {
            "source_classification": "claude",
            "source_classification_confidence": "high",
            "source_classification_reason": [
                "session_id matched a transcript under ~/.claude/projects",
            ],
        }

    if matched_codex and not matched_claude:
        return {
            "source_classification": "codex",
            "source_classification_confidence": "high",
            "source_classification_reason": [
                "session_id matched a session under ~/.codex history",
            ],
        }

    if not work_dir.strip():
        return {
            "source_classification": "codex",
            "source_classification_confidence": "medium",
            "source_classification_reason": [
                "empty work_dir matches legacy Codex history import fallback",
            ],
        }

    if matched_claude and matched_codex:
        return {
            "source_classification": "unknown",
            "source_classification_confidence": "low",
            "source_classification_reason": [
                "session_id matched both ~/.claude and ~/.codex history",
            ],
        }

    return {
        "source_classification": "unknown",
        "source_classification_confidence": "low",
        "source_classification_reason": [
            "no local Claude/Codex history match found",
        ],
    }
