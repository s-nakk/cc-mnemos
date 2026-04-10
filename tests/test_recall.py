"""recall モジュールのテスト"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from conftest import make_chunk, make_session_id

from cc_mnemos.config import Config
from cc_mnemos.recall import format_recall_output, run_recall
from cc_mnemos.store import MemoryStore


class TestFormatRecallOutput:
    """format_recall_output の単体テスト"""

    def test_with_recent_and_cross_project(self) -> None:
        """プロジェクト別・横断チャンクが両方あるケース"""
        recent = [
            {
                "role_user": "ボタンの色を変えたい",
                "role_assistant": "primaryカラーを#3B82F6にしましょう",
                "created_at": "2026-03-20T10:00:00+00:00",
                "tags": '["ui-ux"]',
            },
        ]
        cross = [
            {
                "role_user": "ESLintの設定は？",
                "role_assistant": "flat configを使いましょう",
                "created_at": "2026-03-18T10:00:00+00:00",
                "tags": '["coding-style"]',
            },
        ]
        output = format_recall_output("my-app", recent, cross)
        assert "直近の記憶（my-app）" in output
        assert "ボタンの色を変えたい" in output
        assert "primaryカラーを#3B82F6にしましょう" in output
        assert "よく参照される知見" in output
        assert "flat configを使いましょう" in output
        assert "search_memory" in output

    def test_header_contains_project_name(self) -> None:
        """ヘッダーにプロジェクト名が含まれる"""
        output = format_recall_output("test-proj", [], [])
        assert "直近の記憶（test-proj）" in output

    def test_empty_db_returns_reminder_only(self) -> None:
        """空DBの場合はリマインダーのみ出力される"""
        output = format_recall_output("empty-proj", [], [])
        assert "search_memory" in output
        # ヘッダーは存在するがチャンク行は無い
        assert "直近の記憶（empty-proj）" in output

    def test_only_recent_chunks(self) -> None:
        """プロジェクト別チャンクのみのケース"""
        recent = [
            {
                "role_user": "質問A",
                "role_assistant": "回答A",
                "created_at": "2026-03-20T10:00:00+00:00",
                "tags": '["general"]',
            },
        ]
        output = format_recall_output("proj", recent, [])
        assert "質問A" in output
        assert "回答A" in output
        assert "search_memory" in output

    def test_only_cross_project_chunks(self) -> None:
        """横断チャンクのみのケース"""
        cross = [
            {
                "role_user": "横断質問",
                "role_assistant": "横断回答",
                "created_at": "2026-03-19T10:00:00+00:00",
                "tags": '["architecture"]',
            },
        ]
        output = format_recall_output("proj", [], cross)
        assert "横断回答" in output
        assert "search_memory" in output

    def test_recent_section_has_no_date_prefix(self) -> None:
        """直近セクションに日付プレフィックスが付かない

        日付はセッション間で変動しプロンプトキャッシュを壊すため、recent には出力しない
        """
        recent = [
            {
                "role_user": "質問",
                "role_assistant": "回答",
                "created_at": "2026-03-20T10:00:00+00:00",
                "tags": '["general"]',
            },
        ]
        output = format_recall_output("proj", recent, [])
        assert "2026-03-20" not in output
        assert "- 質問 → 回答" in output

    def test_long_entries_are_truncated(self) -> None:
        """長大なエントリはトランケートされる"""
        long_user = "あ" * 500
        long_assistant = "い" * 1000
        recent = [
            {
                "role_user": long_user,
                "role_assistant": long_assistant,
                "created_at": "2026-03-20T10:00:00+00:00",
                "tags": '["general"]',
            },
        ]
        output = format_recall_output("proj", recent, [])
        assert long_user not in output
        assert long_assistant not in output
        assert "..." in output


class TestRunRecall:
    """run_recall の統合テスト"""

    def _populate_store(
        self, store: MemoryStore, project_name: str, count: int = 3
    ) -> None:
        """テストDB にチャンクを投入するヘルパー"""
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project=project_name,
            work_dir="/tmp/test",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        for i in range(count):
            chunk = make_chunk(
                session_id,
                role_user=f"質問{i}",
                role_assistant=f"回答{i}",
            )
            embedding = np.random.rand(768).astype(np.float32)
            store.insert_chunk(chunk, embedding)

    def test_run_recall_with_populated_db(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """データがあるDBでrun_recallを実行し標準出力に結果が出る"""
        config = Config(
            general={"data_dir": str(tmp_path)},
            projects={str(tmp_path): "test-project"},
        )
        store = MemoryStore(config)
        self._populate_store(store, "test-project", count=3)
        store.close()

        hook_input = {"cwd": str(tmp_path)}
        run_recall(hook_input, config)

        captured = capsys.readouterr()
        assert "直近の記憶（test-project）" in captured.out
        assert "search_memory" in captured.out

    def test_run_recall_empty_db(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """空DBでもエラーにならずリマインダーが出力される"""
        config = Config(
            general={"data_dir": str(tmp_path)},
            projects={str(tmp_path): "empty-project"},
        )
        # DB初期化のみ (データ投入なし)
        store = MemoryStore(config)
        store.close()

        hook_input = {"cwd": str(tmp_path)}
        run_recall(hook_input, config)

        captured = capsys.readouterr()
        assert "search_memory" in captured.out

    def test_run_recall_exception_produces_no_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """例外発生時は何も出力しない (セッション開始をブロックしない)"""
        config = Config(general={"data_dir": str(tmp_path)})

        def boom(*args: object, **kwargs: object) -> None:
            msg = "DB接続失敗"
            raise RuntimeError(msg)

        monkeypatch.setattr("cc_mnemos.recall.MemoryStore", boom)

        hook_input = {"cwd": str(tmp_path)}
        run_recall(hook_input, config)

        captured = capsys.readouterr()
        assert captured.out == ""
