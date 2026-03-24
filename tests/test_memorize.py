from __future__ import annotations

from pathlib import Path

from cc_mnemos.config import Config
from cc_mnemos.memorize import run_memorize
from cc_mnemos.store import MemoryStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestMemorizePipeline:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-001",
            "transcript_path": str(FIXTURES_DIR / "sample_transcript.jsonl"),
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }
        run_memorize(hook_input, config)
        store = MemoryStore(config)
        stats = store.get_stats()
        assert stats["total_chunks"] >= 2
        store.close()

    def test_skips_when_stop_hook_active(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-002",
            "transcript_path": str(FIXTURES_DIR / "sample_transcript.jsonl"),
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }
        run_memorize(hook_input, config)
        store = MemoryStore(config)
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        store.close()

    def test_handles_missing_transcript(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        hook_input = {
            "session_id": "test-session-003",
            "transcript_path": "/nonexistent/path.jsonl",
            "cwd": str(tmp_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }
        run_memorize(hook_input, config)  # Should not crash
