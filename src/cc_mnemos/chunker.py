"""会話JSONL → Q&Aチャンク分割"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 処理対象のメッセージタイプ
ALLOWED_TYPES = {"user", "human", "assistant"}


@dataclass
class Chunk:
    """Q&Aペアのチャンク"""

    role_user: str
    role_assistant: str
    content: str  # role_user + "\n" + role_assistant


def _extract_text(msg: dict) -> str:
    """メッセージからテキストを抽出する

    Claude Code JONLの実際のフォーマットに対応:
    - user: msg["message"]["content"] (文字列 or リスト)
    - assistant: msg["message"]["content"] (パーツリスト, textタイプのみ抽出)
    - 旧フォーマット: msg["content"] (文字列 or リスト)
    """
    # 新フォーマット: msg["message"]["content"]
    message = msg.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        # 旧フォーマット / テストフィクスチャ: msg["content"]
        content = msg.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # パーツリストからtextタイプのみ抽出
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") not in ("thinking", "tool_use", "tool_result"):
                    # 未知のタイプでもtextフィールドがあれば抽出
                    text_val = part.get("text", "")
                    if text_val:
                        texts.append(text_val)
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()

    return ""


def parse_transcript(path: Path) -> list[dict]:
    """JONLファイルをパースし、user/assistantメッセージのみ抽出する"""
    messages: list[dict] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type", "")
            if msg_type in ALLOWED_TYPES:
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
        msg_type = msg.get("type", "")
        text = _extract_text(msg)

        if msg_type in ("human", "user"):
            if text:
                user_buffer = text
        elif msg_type == "assistant" and user_buffer:
            if text:
                assistant_text = _truncate_tokens(text, max_tokens)
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
