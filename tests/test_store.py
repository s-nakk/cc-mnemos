from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
from conftest import make_chunk, make_session_id

from cc_mnemos.store import MemoryStore


class TestSchemaInit:
    def test_tables_created(self, store: MemoryStore) -> None:
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "sessions" in table_names
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
        )
        chunk = make_chunk(session_id)
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 1

    def test_fts_search(self, store: MemoryStore) -> None:
        session_id = make_session_id()
        store.insert_session(
            session_id=session_id,
            project="test",
            work_dir="/tmp",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        chunk = make_chunk(session_id, role_user="border-radiusの設定")
        embedding = np.random.rand(768).astype(np.float32)
        store.insert_chunk(chunk, embedding)
        results = store.fts_search("border-radius", limit=5)
        assert len(results) >= 1


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
