from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cc_mnemos.chunker import (
    MAX_CHARS,
    _is_short_phatic,
    chunk_transcript,
    parse_transcript,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParseTranscript:
    def test_parse_jsonl_file(self) -> None:
        messages = parse_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        assert len(messages) > 0
        assert all("type" in m for m in messages)

    def test_parse_filters_tool_messages(self) -> None:
        messages = parse_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        types = {m["type"] for m in messages}
        assert "tool_use" not in types
        assert "tool_result" not in types


class TestChunkTranscript:
    def test_creates_qa_pairs(self) -> None:
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.role_user != ""
            assert chunk.role_assistant != ""

    def test_truncates_long_responses(self) -> None:
        """MAX_CHARS以下に文字数が制限される"""
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        for chunk in chunks:
            assert len(chunk.role_assistant) <= MAX_CHARS
            assert len(chunk.role_user) <= MAX_CHARS

    def test_content_is_combined(self) -> None:
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        for chunk in chunks:
            assert chunk.role_user in chunk.content
            assert chunk.role_assistant in chunk.content


class TestConsecutiveUserMessages:
    """連続user発話の蓄積テスト"""

    def test_consecutive_user_messages_accumulated(self) -> None:
        """連続するuser発話がまとめてchunkに含まれる"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            msg_u1 = {"type": "user", "message": {"content": "最初の質問"}}
            msg_u2 = {"type": "user", "message": {"content": "補足情報です"}}
            msg_a = {"type": "assistant", "message": {
                "content": "回答します。両方の情報を踏まえて説明します。"
            }}
            lines = [json.dumps(m) for m in [msg_u1, msg_u2, msg_a]]
            f.write("\n".join(lines))
            path = Path(f.name)

        chunks = chunk_transcript(path)
        assert len(chunks) == 1
        # 両方のuser発話がrole_userに含まれている
        assert "最初の質問" in chunks[0].role_user
        assert "補足情報です" in chunks[0].role_user


class TestShortMessageMerge:
    """短文マージのテスト"""

    def test_is_short_phatic(self) -> None:
        assert _is_short_phatic("はい") is True
        assert _is_short_phatic("A") is True
        assert _is_short_phatic("ok") is True
        assert _is_short_phatic("了解") is True
        assert _is_short_phatic("yes") is True
        assert _is_short_phatic("UIデザインのレイアウトを変更したい") is False

    def test_short_phatic_gets_context(self) -> None:
        """短い追撃発話に直前チャンクの文脈が前置される"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            msgs = [
                {"type": "user", "message": {
                    "content": "border-radiusの設定方法を教えて"
                }},
                {"type": "assistant", "message": {
                    "content": "CSSのborder-radiusプロパティで角丸を"
                    "設定できます。例: border-radius: 8px;"
                }},
                {"type": "user", "message": {"content": "はい"}},
                {"type": "assistant", "message": {
                    "content": "他にもpxだけでなく%でも指定可能です。"
                }},
            ]
            lines = [json.dumps(m) for m in msgs]
            f.write("\n".join(lines))
            path = Path(f.name)

        chunks = chunk_transcript(path)
        assert len(chunks) == 2
        # 2つ目のチャンクに直前の文脈が含まれている
        assert "border-radius" in chunks[1].role_user
