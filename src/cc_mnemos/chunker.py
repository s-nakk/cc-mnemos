"""会話JSONL → Q&Aチャンク分割"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKIP_TYPES = {"tool_use", "tool_result", "system"}


@dataclass
class Chunk:
    """Q&Aペアのチャンク"""

    role_user: str
    role_assistant: str
    content: str  # role_user + "\n" + role_assistant


def parse_transcript(path: Path) -> list[dict[str, str]]:
    """JONLファイルをパースし、ツール呼び出しを除外したメッセージリストを返す"""
    messages: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg: dict[str, str] = json.loads(line)
            msg_type = msg.get("type", msg.get("role", ""))
            if msg_type not in SKIP_TYPES:
                messages.append(msg)
    return messages


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """トークン数を制限する（空白分割による近似）"""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def chunk_transcript(
    path: Path,
    max_tokens: int = 2000,
    min_tokens: int = 1,
) -> list[Chunk]:
    """会話JONLをQ&Aペアに分割する"""
    messages = parse_transcript(path)
    chunks: list[Chunk] = []
    user_buffer: str = ""
    for msg in messages:
        msg_type = msg.get("type", msg.get("role", ""))
        text = msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        if msg_type in ("human", "user"):
            user_buffer = text.strip()
        elif msg_type == "assistant" and user_buffer:
            assistant_text = _truncate_tokens(text.strip(), max_tokens)
            content = f"{user_buffer}\n{assistant_text}"
            total_tokens = len(content.split())
            if total_tokens >= min_tokens:
                chunks.append(
                    Chunk(
                        role_user=user_buffer,
                        role_assistant=assistant_text,
                        content=content,
                    ),
                )
            user_buffer = ""
    return chunks
