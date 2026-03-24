from __future__ import annotations

from pathlib import Path

from cc_mnemos.chunker import chunk_transcript, parse_transcript

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

    def test_filters_trivial_exchanges(self) -> None:
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl", min_tokens=10)
        for chunk in chunks:
            total_tokens = len(chunk.content.split())
            assert total_tokens >= 10

    def test_truncates_long_responses(self) -> None:
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl", max_tokens=50)
        for chunk in chunks:
            assistant_tokens = len(chunk.role_assistant.split())
            assert assistant_tokens <= 50

    def test_content_is_combined(self) -> None:
        chunks = chunk_transcript(FIXTURES_DIR / "sample_transcript.jsonl")
        for chunk in chunks:
            assert chunk.role_user in chunk.content
            assert chunk.role_assistant in chunk.content
