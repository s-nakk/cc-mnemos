"""batch_import モジュールのテスト"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cc_mnemos.batch_import import _read_session_metadata, _resolve_cwd, import_history
from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    pass

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_RNG = np.random.default_rng(42)


def _mock_encode_documents(texts: list[str]) -> np.ndarray:
    """テスト用: 入力テキスト数に応じたダミー埋め込みを返す"""
    return _RNG.random((len(texts), 768)).astype(np.float32)


def _mock_encode_topic(text: str) -> np.ndarray:
    """テスト用: 単一トピックのダミー埋め込みを返す"""
    return _RNG.random(768).astype(np.float32)


class TestResolveCwd:
    """_resolve_cwd のユニットテスト"""

    def test_windows_drive_path(self) -> None:
        assert _resolve_cwd("c--projects-Morfee") == "C:/projects/Morfee"

    def test_windows_drive_uppercase(self) -> None:
        assert _resolve_cwd("C--projects-Resitoly") == "C:/projects/Resitoly"

    def test_deep_path(self) -> None:
        assert _resolve_cwd("d--work-my-app") == "D:/work/my/app"

    def test_single_segment(self) -> None:
        # ドライブレター+セパレータなしの場合
        result = _resolve_cwd("myproject")
        assert result == "myproject"


class TestReadSessionMetadata:
    def test_reads_cwd_and_timestamp_from_jsonl(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "cwd": "C:\\projects\\cc-mnemos",
                            "timestamp": "2026-03-25T00:00:00Z",
                            "message": {"content": "cc-mnemos の保存仕様について確認したいです"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"content": "保存仕様は SQLite ベースで管理されています"},
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        cwd, timestamp = _read_session_metadata(transcript)

        assert cwd == "C:\\projects\\cc-mnemos"
        assert timestamp == "2026-03-25T00:00:00Z"


class TestImportHistory:
    """import_history の統合テスト"""

    def test_no_projects_dir(self, tmp_path: Path) -> None:
        """projects ディレクトリが存在しない場合"""
        config = Config(general={"data_dir": str(tmp_path)})
        with patch("cc_mnemos.batch_import.Path.home") as mock_home:
            mock_home.return_value = tmp_path / "fakehome"
            result = import_history(config, verbose=False)
        assert result == {"imported": 0, "skipped": 0, "errors": 0}

    def test_import_single_session(self, tmp_path: Path) -> None:
        """1セッションのインポートが正常に動作する"""
        config = Config(general={"data_dir": str(tmp_path / "data")})

        # ~/.claude/projects/<project>/<session>.jsonl を模擬
        fake_home = tmp_path / "fakehome"
        projects_dir = fake_home / ".claude" / "projects" / "c--projects-TestApp"
        projects_dir.mkdir(parents=True)
        src = FIXTURES_DIR / "sample_transcript.jsonl"
        shutil.copy(src, projects_dir / "session-001.jsonl")

        # Embedder をモックして高速化
        mock_embedder = MagicMock()
        mock_embedder.encode_documents.side_effect = _mock_encode_documents
        mock_embedder.encode_topic.side_effect = _mock_encode_topic

        with (
            patch("cc_mnemos.batch_import.Path.home") as mock_home,
        ):
            mock_home.return_value = fake_home
            result = import_history(config, embedder=mock_embedder, verbose=False)

        assert result["imported"] == 1
        assert result["errors"] == 0

        # DB にセッションとチャンクが保存されていることを確認
        store = MemoryStore(config)
        stats = store.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_chunks"] >= 1
        store.close()

    def test_skip_already_imported(self, tmp_path: Path) -> None:
        """既にインポート済みのセッションはスキップされる"""
        config = Config(general={"data_dir": str(tmp_path / "data")})

        # 先にセッションをDBに登録
        store = MemoryStore(config)
        store.insert_session(
            session_id="session-001",
            project="TestApp",
            work_dir="C:/projects/TestApp",
            started_at="2025-01-01T00:00:00+00:00",
        )
        store.close()

        # ~/.claude/projects/<project>/session-001.jsonl を配置
        fake_home = tmp_path / "fakehome"
        projects_dir = fake_home / ".claude" / "projects" / "c--projects-TestApp"
        projects_dir.mkdir(parents=True)
        shutil.copy(
            FIXTURES_DIR / "sample_transcript.jsonl",
            projects_dir / "session-001.jsonl",
        )

        with patch("cc_mnemos.batch_import.Path.home") as mock_home:
            mock_home.return_value = fake_home
            result = import_history(config, verbose=False)

        assert result["skipped"] == 1
        assert result["imported"] == 0

    def test_skip_subagents(self, tmp_path: Path) -> None:
        """subagents ディレクトリ内のファイルは除外される"""
        config = Config(general={"data_dir": str(tmp_path / "data")})

        fake_home = tmp_path / "fakehome"
        subagent_dir = (
            fake_home / ".claude" / "projects" / "c--projects-TestApp" / "subagents"
        )
        subagent_dir.mkdir(parents=True)
        shutil.copy(
            FIXTURES_DIR / "sample_transcript.jsonl",
            subagent_dir / "sub-session.jsonl",
        )

        with patch("cc_mnemos.batch_import.Path.home") as mock_home:
            mock_home.return_value = fake_home
            result = import_history(config, verbose=False)

        assert result["imported"] == 0

    def test_progress_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """verbose=True 時にメッセージが出力される"""
        config = Config(general={"data_dir": str(tmp_path / "data")})

        fake_home = tmp_path / "fakehome"
        projects_dir = fake_home / ".claude" / "projects" / "c--projects-TestApp"
        projects_dir.mkdir(parents=True)
        shutil.copy(
            FIXTURES_DIR / "sample_transcript.jsonl",
            projects_dir / "session-progress.jsonl",
        )

        mock_embedder = MagicMock()
        mock_embedder.encode_documents.side_effect = _mock_encode_documents
        mock_embedder.encode_topic.side_effect = _mock_encode_topic

        with (
            patch("cc_mnemos.batch_import.Path.home") as mock_home,
        ):
            mock_home.return_value = fake_home
            import_history(config, embedder=mock_embedder, verbose=True)

        captured = capsys.readouterr()
        assert "Importing 1 sessions" in captured.out
        assert "Done:" in captured.out

    def test_prefers_jsonl_metadata_over_directory_name(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path / "data")})

        fake_home = tmp_path / "fakehome"
        projects_dir = fake_home / ".claude" / "projects" / "C--projects-cc-mnemos"
        projects_dir.mkdir(parents=True)
        session_path = projects_dir / "session-001.jsonl"
        session_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "cwd": "C:\\projects\\cc-mnemos",
                            "timestamp": "2026-03-25T00:00:00Z",
                            "message": {"content": "cc-mnemos の保存仕様について確認したいです"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"content": "保存仕様は SQLite ベースで管理されています"},
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        mock_embedder = MagicMock()
        mock_embedder.encode_documents.side_effect = _mock_encode_documents

        with patch("cc_mnemos.batch_import.Path.home") as mock_home:
            mock_home.return_value = fake_home
            result = import_history(config, embedder=mock_embedder, verbose=False)

        assert result["imported"] == 1

        store = MemoryStore(config)
        row = store.conn.execute(
            "SELECT project, work_dir, started_at FROM sessions WHERE session_id = ?",
            ("session-001",),
        ).fetchone()
        store.close()

        assert row is not None
        assert row[0] == "cc-mnemos"
        assert row[1] == "C:\\projects\\cc-mnemos"
        assert row[2] == "2026-03-25T00:00:00+00:00"
