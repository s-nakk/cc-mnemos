"""会話JSONL → Q&Aチャンク分割"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 処理対象のメッセージタイプ
ALLOWED_TYPES = {"user", "human", "assistant"}

# システムテキストのフィルタ（チャンクに含めるべきでない接頭辞）
_NOISE_PREFIXES = (
    "Base directory for this skill:",
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<system-reminder>",
    "ARGUMENTS:",
)


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
    # 事前フィルタ: JSONパース前に文字列レベルで不要な行をスキップ
    # 会話に必要な type は "user", "human", "assistant" のみ
    _prefixes = ('"type":"user"', '"type":"assistant"', '"type":"human"',
                 '"type": "user"', '"type": "assistant"', '"type": "human"')
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 文字列レベルで会話メッセージ行だけをJSONパースする
            if not any(p in line for p in _prefixes):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type", "")
            if msg_type in ALLOWED_TYPES:
                messages.append(msg)
    return messages


MAX_CHARS = 1500  # 日本語テキストの文字数上限（≒500-1000モデルトークン）


def _truncate(text: str) -> str:
    """テキストを文字数で制限する"""
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS]


def chunk_transcript(
    path: Path,
    max_tokens: int = 2000,  # 後方互換（未使用、文字数ベースに移行済み）
    min_tokens: int = 1,     # 後方互換（未使用）
) -> list[Chunk]:
    """会話JONLをQ&Aペアに分割する"""
    messages = parse_transcript(path)
    chunks: list[Chunk] = []
    user_buffer: str = ""
    for msg in messages:
        msg_type = msg.get("type", "")
        text = _extract_text(msg)

        if msg_type in ("human", "user"):
            if text and not any(text.lstrip().startswith(p) for p in _NOISE_PREFIXES):
                user_buffer = _truncate(text)
        elif msg_type == "assistant" and user_buffer:
            # テキストが空の場合はスキップ（thinking-onlyパーツ等）
            # user_bufferは保持し、次のassistantメッセージでペアにする
            if not text:
                continue
            assistant_text = _truncate(text)
            content = f"{user_buffer}\n{assistant_text}"
            if len(content) >= 20:
                chunks.append(
                    Chunk(
                        role_user=user_buffer,
                        role_assistant=assistant_text,
                        content=content,
                    ),
                )
            user_buffer = ""
    return chunks
