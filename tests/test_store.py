from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from conftest import make_chunk, make_session_id

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore


class TestSchemaInit:
    def test_tables_created(self, store: MemoryStore) -> None:
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "sessions" in table_names
        assert "session_sources" in table_names
        assert "chunks" in table_names
        assert "schema_version" in table_names

    def test_wal_mode(self, store: MemoryStore) -> None:
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestInsertAndQuery:
    def test_insert_session_and_chunks(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test-project",
            work_dir="/tmp/test",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
            recorded_source="claude",
        )
        chunk = make_chunk(session_id)
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 1

        source_row = store.conn.execute(
            """
            SELECT recorded_source
            FROM session_sources
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        assert source_row is not None
        assert source_row[0] == "claude"

    def test_insert_session_preserves_started_at_on_conflict(
        self, store: MemoryStore
    ) -> None:
        """同一 session_id に対する 2 回目以降の insert_session で started_at が
        上書きされないことを保証する (issue #2 リグレッション防止)
        """
        session_id = make_session_id()
        initial_started_at = "2026-01-01T00:00:00+00:00"
        later_started_at = "2026-05-14T12:00:00+00:00"

        store.insert_session(
            session_id=session_id,
            project="test-project",
            work_dir="/tmp/test",
            started_at=initial_started_at,
        )

        store.insert_session(
            session_id=session_id,
            project="test-project",
            work_dir="/tmp/test",
            started_at=later_started_at,
            ended_at="2026-05-14T13:00:00+00:00",
            summary="updated summary",
        )

        row = store.conn.execute(
            "SELECT started_at, ended_at, summary FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == initial_started_at, "started_at は初回値のまま維持される必要がある"
        assert row[1] == "2026-05-14T13:00:00+00:00"
        assert row[2] == "updated summary"

    def test_fts_search(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
            recorded_source="codex",
        )
        chunk = make_chunk(session_id, role_user="border-radiusの設定")
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        results = store.fts_search("border-radius", limit=5)
        assert len(results) >= 1
        assert results[0]["effective_source"] == "codex"

    def test_backfill_classifies_existing_sessions_from_agent_history(self, tmp_path: Path) -> None:
        config = Config(general={"data_dir": str(tmp_path / "data")})
        store = MemoryStore(config)

        claude_session_id = "claude-session-001"
        codex_session_id = "session-codex-001"
        started_at = datetime.now(tz=timezone.utc).isoformat()

        store.insert_session(
            session_id=claude_session_id,
            project="claude-project",
            work_dir="C:/projects/claude-project",
            started_at=started_at,
        )
        store.insert_session(
            session_id=codex_session_id,
            project="codex-project",
            work_dir="C:/projects/codex-project",
            started_at=started_at,
        )

        fake_home = tmp_path / "fakehome"
        claude_projects_dir = fake_home / ".claude" / "projects" / "c--projects-TestApp"
        claude_projects_dir.mkdir(parents=True)
        (claude_projects_dir / f"{claude_session_id}.jsonl").write_text(
            json.dumps({"type": "user", "message": {"content": "Claude transcript"}}) + "\n",
            encoding="utf-8",
        )

        codex_sessions_dir = fake_home / ".codex" / "sessions" / "2026" / "04" / "14"
        codex_sessions_dir.mkdir(parents=True)
        (codex_sessions_dir / "rollout-001.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session_meta",
                            "payload": {
                                "id": codex_session_id,
                                "timestamp": "2026-04-14T10:00:00.000Z",
                                "cwd": "C:\\projects\\CodexApp",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Codex transcript"}],
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        classified = store.backfill_session_sources(home_dir=fake_home)

        assert classified == 2

        rows = store.conn.execute(
            """
            SELECT session_id, source_classification, source_classification_confidence
            FROM session_sources
            ORDER BY session_id
            """
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (claude_session_id, "claude", "high"),
            (codex_session_id, "codex", "high"),
        ]


class TestHybridSearch:
    def test_rrf_scoring(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        for i in range(3):
            chunk = make_chunk(session_id, role_user=f"テスト質問{i}")
            embedding = np.random.rand(768).astype(np.float32)
            store.insert_chunk(chunk, embedding)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = store.hybrid_search(
            query_text="テスト", query_embedding=query_embedding, limit=10
        )
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tag_filter(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        chunk_ui = make_chunk(session_id, role_user="UI質問", tags=["ui-ux"])
        chunk_debug = make_chunk(session_id, role_user="デバッグ質問", tags=["debug"])
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk_ui, embedding)
        store.insert_chunk(chunk_debug, embedding)

        results = store.hybrid_search(
            query_text="質問", query_embedding=embedding, tags=["ui-ux"], limit=10
        )
        for r in results:
            assert "ui-ux" in json.loads(str(r["tags"]))

    def test_time_decay(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        now = datetime.now(tz=timezone.utc)
        old_date = (now - timedelta(days=60)).isoformat()
        new_date = now.isoformat()
        chunk_old = make_chunk(session_id, role_user="古い質問", created_at=old_date)
        chunk_new = make_chunk(session_id, role_user="新しい質問", created_at=new_date)
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk_old, embedding)
        store.insert_chunk(chunk_new, embedding)
        results = store.hybrid_search(
            query_text="質問", query_embedding=embedding, limit=10
        )
        assert len(results) == 2

    def test_project_filter_retrieves_relevant_result_beyond_unfiltered_top_k(
        self, store: MemoryStore
    ) -> None:
        query_embedding = np.ones(768, dtype=np.float32)
        now = datetime.now(tz=timezone.utc).isoformat()

        for i in range(20):
            session_id = make_session_id()
            store.insert_session(
                session_id=session_id,
                project="other-project",
                work_dir="/tmp/other",
                started_at=now,
            )
            chunk = make_chunk(session_id, role_user=f"共通クエリ{i}")
            store.insert_chunk(chunk, query_embedding)

        target_session = make_session_id()
        store.insert_session(
            session_id=target_session,
            project="target-project",
            work_dir="/tmp/target",
            started_at=now,
        )
        target_embedding = np.concatenate(
            [np.ones(10, dtype=np.float32), np.zeros(758, dtype=np.float32)]
        )
        target_chunk = make_chunk(target_session, role_user="共通クエリ target")
        store.insert_chunk(target_chunk, target_embedding)

        results = store.hybrid_search(
            query_text="共通クエリ",
            query_embedding=query_embedding,
            project="target-project",
            limit=1,
        )

        assert len(results) == 1
        assert results[0]["session_id"] == target_session

    def test_tag_filter_retrieves_relevant_result_beyond_unfiltered_top_k(
        self, store: MemoryStore
    ) -> None:
        query_embedding = np.ones(768, dtype=np.float32)
        now = datetime.now(tz=timezone.utc).isoformat()

        for i in range(20):
            session_id = make_session_id()
            store.insert_session(
                session_id=session_id,
                project="shared-project",
                work_dir="/tmp/shared",
                started_at=now,
            )
            chunk = make_chunk(session_id, role_user=f"タグ検索{i}", tags=["general"])
            store.insert_chunk(chunk, query_embedding)

        tagged_session = make_session_id()
        store.insert_session(
            session_id=tagged_session,
            project="shared-project",
            work_dir="/tmp/shared",
            started_at=now,
        )
        tagged_embedding = np.concatenate(
            [np.ones(10, dtype=np.float32), np.zeros(758, dtype=np.float32)]
        )
        tagged_chunk = make_chunk(
            tagged_session,
            role_user="タグ検索 target",
            tags=["ui-ux"],
        )
        store.insert_chunk(tagged_chunk, tagged_embedding)

        results = store.hybrid_search(
            query_text="タグ検索",
            query_embedding=query_embedding,
            tags=["ui-ux"],
            limit=1,
        )

        assert len(results) == 1
        assert "ui-ux" in json.loads(str(results[0]["tags"]))


class TestStats:
    def test_get_stats(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test-project",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        chunk = make_chunk(session_id, tags=["ui-ux"])
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        stats = store.get_stats()
        assert stats["total_chunks"] == 1
        assert "test-project" in stats["by_project"]
