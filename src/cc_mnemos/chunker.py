"""会話JSONL → Q&Aチャンク分割"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TranscriptMessage = dict[str, object]

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

# システムタグ除去パターン
_SYSTEM_TAGS_RE = re.compile(
    r"<local-command-caveat>.*?</local-command-caveat>\s*"
    r"|<local-command-stdout>.*?</local-command-stdout>\s*",
    re.DOTALL,
)

# 文脈なし短文パターン（追撃発話）
_PHATIC_RE = re.compile(
    r"^(はい|うん|ok|OK|yes|no|了解|ありがとう|A|B|C|D|それで|続けて|進めて|hai)$",
    re.IGNORECASE,
)

# 短文の文字数閾値
_SHORT_MSG_CHARS = 10


@dataclass
class Chunk:
    """Q&Aペアのチャンク"""

    role_user: str
    role_assistant: str
    content: str  # role_user + "\n" + role_assistant


def _extract_text(msg: TranscriptMessage) -> str:
    """メッセージからテキストを抽出する

    Claude Code JONLの実際のフォーマットに対応:
    - user: msg["message"]["content"] (文字列 or リスト)
    - assistant: msg["message"]["content"] (パーツリスト, textタイプのみ抽出)
    - 旧フォーマット: msg["content"] (文字列 or リスト)
    """
    # 新フォーマット: msg["message"]["content"] / 旧フォーマット: msg["content"]
    message = msg.get("message")
    content = message.get("content", "") if isinstance(message, dict) else msg.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # パーツリストからtextタイプのみ抽出
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    text_value = part.get("text", "")
                    if isinstance(text_value, str):
                        texts.append(text_value)
                elif part_type not in ("thinking", "tool_use", "tool_result"):
                    # 未知のタイプでもtextフィールドがあれば抽出
                    text_val = part.get("text", "")
                    if isinstance(text_val, str) and text_val:
                        texts.append(text_val)
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()

    return ""


def parse_transcript(path: Path) -> list[TranscriptMessage]:
    """JONLファイルをパースし、user/assistantメッセージのみ抽出する"""
    messages: list[TranscriptMessage] = []
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
            if not isinstance(msg, dict):
                continue
            normalized_msg = {str(key): value for key, value in msg.items()}
            msg_type = normalized_msg.get("type", "")
            if isinstance(msg_type, str) and msg_type in ALLOWED_TYPES:
                messages.append(normalized_msg)
    return messages


MAX_CHARS = 1500  # 日本語テキストの文字数上限（≒500-1000モデルトークン）


def _truncate(text: str, limit: int = MAX_CHARS) -> str:
    """テキストを文字数で制限する"""
    if len(text) <= limit:
        return text
    return text[:limit]


def _is_short_phatic(text: str) -> bool:
    """文脈なしの短い追撃発話かどうか判定する"""
    stripped = text.strip()
    if _PHATIC_RE.match(stripped):
        return True
    return len(stripped) <= _SHORT_MSG_CHARS


def chunk_transcript(
    path: Path,
    max_chars: int = MAX_CHARS,
    min_chars: int = 20,
) -> list[Chunk]:
    """会話JONLをQ&Aペアに分割する

    連続するuser発話はリストとして蓄積し、assistant応答時にまとめてペアにする。
    短い追撃発話（「はい」「A」等）は直前チャンクにマージする
    """
    messages = parse_transcript(path)
    chunks: list[Chunk] = []
    user_parts: list[str] = []  # 連続user発話を蓄積
    prev_chunk: Chunk | None = None  # 直前のチャンク（短文マージ用）

    for msg in messages:
        msg_type_obj = msg.get("type", "")
        msg_type = msg_type_obj if isinstance(msg_type_obj, str) else ""
        text = _extract_text(msg)

        if msg_type in ("human", "user"):
            # システムタグ(local-command-caveat等)を除去して実際のユーザー発話を抽出
            text = _SYSTEM_TAGS_RE.sub("", text).strip()
            if text and not any(text.lstrip().startswith(p) for p in _NOISE_PREFIXES):
                user_parts.append(_truncate(text, max_chars))
        elif msg_type == "assistant" and user_parts:
            if not text:
                continue

            assistant_text = _truncate(text, max_chars)
            user_combined = "\n".join(user_parts)

            # 短い追撃発話は直前チャンクにマージ
            if _is_short_phatic(user_combined) and prev_chunk is not None:
                user_with_context = f"{prev_chunk.role_user}\n{user_combined}"
                user_with_context = _truncate(user_with_context, max_chars)
                content = f"{user_with_context}\n{assistant_text}"
                if len(content) >= min_chars:
                    new_chunk = Chunk(
                        role_user=user_with_context,
                        role_assistant=assistant_text,
                        content=content,
                    )
                    chunks.append(new_chunk)
                    prev_chunk = new_chunk
            else:
                content = f"{user_combined}\n{assistant_text}"
                if len(content) >= min_chars:
                    new_chunk = Chunk(
                        role_user=user_combined,
                        role_assistant=assistant_text,
                        content=content,
                    )
                    chunks.append(new_chunk)
                    prev_chunk = new_chunk

            user_parts = []  # リセット
    return chunks
