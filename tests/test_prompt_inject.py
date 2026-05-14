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


@pytest.fixture
def _force_fts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """テスト中は worker daemon に接続させず必ず FTS fallback 経路を通す

    開発機で worker daemon (port 19836) が常駐していると、tmp_path の検証用 DB
    ではなく本物の DB を見に行くため、テストが非決定論的になる。`_query_worker`
    を None 固定でスタブ化することで fallback 経路に強制誘導する
    """
    monkeypatch.setattr(
        "cc_mnemos.prompt_inject._query_worker",
        lambda *args, **kwargs: None,
    )


class TestPromptInject:
    def test_injects_relevant_memory(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        _force_fts_fallback: None,
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
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        _force_fts_fallback: None,
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
    def test_formats_results_fts(self) -> None:
        results = [
            {
                "tags": json.dumps(["ui-ux"]),
                "content": "border-radiusの設定について",
            },
        ]
        output = _format_injection(results, "test-project")
        assert "[cc-mnemos:FTS]" in output
        assert "1 件" in output
        assert "test-project" in output
        assert "border-radius" in output
        assert "ui-ux" in output
        assert "回答に活用した場合は" in output

    def test_formats_results_hybrid(self) -> None:
        results = [
            {
                "tags": json.dumps(["architecture"]),
                "content": "APIルート設計について",
            },
        ]
        output = _format_injection(results, "my-project", search_method="hybrid")
        assert "[cc-mnemos:hybrid]" in output
        assert "my-project" in output

    def test_formats_qa_style(self) -> None:
        """role_user/role_assistantがある場合はQ&A形式で表示される"""
        results = [
            {
                "tags": json.dumps(["coding-style"]),
                "content": "dummy",
                "role_user": "border-radiusの設定方法を教えて",
                "role_assistant": "CSSのborder-radiusで角丸を設定できます",
            },
        ]
        output = _format_injection(results, "test-project")
        assert "Q: border-radius" in output
        assert "A: CSSのborder-radius" in output

    def test_empty_tags_shows_general(self) -> None:
        results = [
            {
                "tags": json.dumps([]),
                "content": "何かのコンテンツがここにあります",
            },
        ]
        output = _format_injection(results, "test-project")
        assert "[general]" in output
