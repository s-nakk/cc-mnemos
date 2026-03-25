"""prompt_inject モジュールのテスト"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from conftest import make_chunk, make_session_id

from cc_mnemos.config import Config
from cc_mnemos.prompt_inject import _format_injection, run_prompt_inject
from cc_mnemos.store import MemoryStore


class TestPromptInject:
    def test_injects_relevant_memory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """関連する記憶がある場合はstdoutに出力される"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        chunk = make_chunk(
            session_id,
            role_user="border-radiusの設定方法を教えて",
            role_assistant="CSSのborder-radiusで角丸を設定できます",
        )
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        store.close()

        hook_input = {
            "user_prompt": "border-radius 設定方法",
            "cwd": "/tmp",
            "session_id": "test-session",
        }
        run_prompt_inject(hook_input, config)
        captured = capsys.readouterr()
        assert "border-radius" in captured.out

    def test_silent_when_no_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """関連する記憶がない場合は何も出力しない"""
        config = Config(general={"data_dir": str(tmp_path)})
        # 空のDBを初期化
        store = MemoryStore(config)
        store.close()

        hook_input = {
            "user_prompt": "まったく関係ない質問です",
            "cwd": "/tmp",
            "session_id": "test-session",
        }
        run_prompt_inject(hook_input, config)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_silent_on_short_prompt(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """短すぎるプロンプトでは検索しない"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        store.close()

        hook_input = {
            "user_prompt": "はい",
            "cwd": "/tmp",
            "session_id": "test-session",
        }
        run_prompt_inject(hook_input, config)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_crash_on_exception(self, tmp_path: Path) -> None:
        """例外が発生してもクラッシュしない"""
        config = Config(general={"data_dir": str(tmp_path / "nonexistent")})
        hook_input = {
            "user_prompt": "テスト",
            "cwd": "/tmp",
            "session_id": "test-session",
        }
        # DBが存在しなくてもクラッシュしない
        run_prompt_inject(hook_input, config)


class TestFormatInjection:
    def test_formats_results(self) -> None:
        results = [
            {
                "tags": json.dumps(["ui-ux"]),
                "content": "border-radiusの設定について",
            },
        ]
        output = _format_injection(results, "test-project")
        assert "[cc-mnemos]" in output
        assert "test-project" in output
        assert "border-radius" in output
        assert "ui-ux" in output
