"""server.py の内部ヘルパー関数テスト"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from conftest import make_chunk, make_session_id

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore


def _populate_test_db(config: Config) -> None:
    """テスト用DBにセッションとチャンクを投入する"""
    store = MemoryStore(config)

    # プロジェクトAに2セッション・3チャンク
    sid_a1 = make_session_id()
    store.insert_session(
        session_id=sid_a1,
        project="project-alpha",
        work_dir="/tmp/alpha",
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    for _ in range(2):
        chunk = make_chunk(sid_a1, role_user="alphaの質問")
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)

    sid_a2 = make_session_id()
    store.insert_session(
        session_id=sid_a2,
        project="project-alpha",
        work_dir="/tmp/alpha",
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    chunk = make_chunk(sid_a2, role_user="alphaの追加質問")
    embedding = np.random.rand(768).astype(np.float32)
    store.insert_chunk(chunk, embedding)

    # プロジェクトBに1セッション・1チャンク
    sid_b = make_session_id()
    store.insert_session(
        session_id=sid_b,
        project="project-beta",
        work_dir="/tmp/beta",
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    chunk = make_chunk(sid_b, role_user="betaの質問")
    embedding = np.random.rand(768).astype(np.float32)
    store.insert_chunk(chunk, embedding)

    store.close()


class TestGetStats:
    def test_returns_correct_totals(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        _populate_test_db(config)

        from cc_mnemos.server import _get_stats

        stats = _get_stats(config=config)
        assert stats["total_chunks"] == 4
        assert stats["total_sessions"] == 3

    def test_returns_by_project_breakdown(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        _populate_test_db(config)

        from cc_mnemos.server import _get_stats

        stats = _get_stats(config=config)
        by_project = stats["by_project"]
        assert by_project["project-alpha"] == 3
        assert by_project["project-beta"] == 1

    def test_empty_db_returns_zeros(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})

        from cc_mnemos.server import _get_stats

        stats = _get_stats(config=config)
        assert stats["total_chunks"] == 0
        assert stats["total_sessions"] == 0
        assert stats["by_project"] == {}


class TestListProjects:
    def test_returns_all_projects(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        _populate_test_db(config)

        from cc_mnemos.server import _list_projects

        projects = _list_projects(config=config)
        assert "project-alpha" in projects
        assert "project-beta" in projects
        assert len(projects) == 2

    def test_returns_sorted_order(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})
        _populate_test_db(config)

        from cc_mnemos.server import _list_projects

        projects = _list_projects(config=config)
        assert projects == sorted(projects)

    def test_empty_db_returns_empty_list(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path)})

        from cc_mnemos.server import _list_projects

        projects = _list_projects(config=config)
        assert projects == []
