"""_memorize_persistence の差分更新挙動のテスト

persist_chunks は「セッション全置換」ではなく「既存 chunk_id を温存しつつ
新規分だけ embed + insert する」差分更新方式で動作する。chunk_id は
``make_chunk_id(session_id, index, content)`` で決定論的に決まるため、
同じ transcript を 2 回連続で memorize しても embed 計算は走らない
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeEmbedder

from cc_mnemos._memorize_persistence import make_chunk_id, persist_chunks
from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore


def _build_record(role_user: str, role_assistant: str) -> dict[str, object]:
    return {
        "role_user": role_user,
        "role_assistant": role_assistant,
        "content": f"{role_user}\n{role_assistant}",
        "tags": ["general"],
    }


def _records_from_pairs(pairs: list[tuple[str, str]]) -> list[dict[str, object]]:
    return [_build_record(u, a) for u, a in pairs]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """FakeEmbedder インスタンスを直接生成して返す"""
    return FakeEmbedder()


def _encode_calls(embedder: FakeEmbedder) -> list[list[str]]:
    """encode_documents の呼び出しごとの texts を返す"""
    return [
        list(call[1])  # type: ignore[arg-type]
        for call in embedder.calls
        if call[0] == "encode_documents"
    ]


class TestIncrementalEmbedding:
    """ターン進行に伴う embed 回数の単調増加を抑える差分更新の挙動"""

    def test_initial_run_embeds_all_chunks(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """初回呼び出しでは全チャンクが 1 回の encode_documents でまとめて embed される

        N+1 退行防止: チャンク数を増やしても ``encode_documents`` の呼び出し回数は
        1 回に固定される (`testing.md` の batch 化チェック)
        """
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            records = _records_from_pairs([
                ("Q1", "A1"),
                ("Q2", "A2"),
                ("Q3", "A3"),
            ])
            persist_chunks(
                session_id="sess-init",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=records,
                embedder=fake_embedder,
                store=store,
            )

            calls = _encode_calls(fake_embedder)
            # 1 回の呼び出しで全 3 件をまとめて投げる (batch 化担保)
            assert len(calls) == 1
            assert calls[0] == ["Q1\nA1", "Q2\nA2", "Q3\nA3"]

            count = store.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE session_id = ?", ("sess-init",)
            ).fetchone()[0]
            assert count == 3
        finally:
            store.close()

    def test_repeat_with_same_chunks_skips_embedding(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """同じ chunk_records で 2 回呼ぶと、2 回目は encode_documents が走らない"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            records = _records_from_pairs([("Q1", "A1"), ("Q2", "A2")])

            persist_chunks(
                session_id="sess-repeat",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=records,
                embedder=fake_embedder,
                store=store,
            )
            persist_chunks(
                session_id="sess-repeat",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=records,
                embedder=fake_embedder,
                store=store,
            )

            calls = _encode_calls(fake_embedder)
            # 初回 1 回のみ、2 回目は 0 件
            assert len(calls) == 1
            assert calls[0] == ["Q1\nA1", "Q2\nA2"]

            count = store.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE session_id = ?", ("sess-repeat",)
            ).fetchone()[0]
            assert count == 2
        finally:
            store.close()

    def test_appended_chunk_embeds_only_new_one(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """末尾に 1 件追加された場合、encode_documents には新規 1 件だけ渡る"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            initial = _records_from_pairs([("Q1", "A1"), ("Q2", "A2")])
            persist_chunks(
                session_id="sess-append",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=initial,
                embedder=fake_embedder,
                store=store,
            )

            extended = _records_from_pairs([
                ("Q1", "A1"),
                ("Q2", "A2"),
                ("Q3", "A3"),
            ])
            persist_chunks(
                session_id="sess-append",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=extended,
                embedder=fake_embedder,
                store=store,
            )

            calls = _encode_calls(fake_embedder)
            assert len(calls) == 2
            assert calls[1] == ["Q3\nA3"]

            count = store.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE session_id = ?", ("sess-append",)
            ).fetchone()[0]
            assert count == 3
        finally:
            store.close()

    def test_removed_chunk_is_deleted(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """新しい chunk_records に含まれない既存 chunk は DB から消える

        併せて以下の整合性も担保する:
          - 「削除のみ・挿入なし」パスで session が upsert され続けること
          - ``chunk_vec_map`` に孤立行が残らないこと
        """
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            persist_chunks(
                session_id="sess-remove",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=_records_from_pairs([("Q1", "A1"), ("Q2", "A2")]),
                embedder=fake_embedder,
                store=store,
            )

            persist_chunks(
                session_id="sess-remove",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=_records_from_pairs([("Q1", "A1")]),
                embedder=fake_embedder,
                store=store,
            )

            calls = _encode_calls(fake_embedder)
            assert len(calls) == 1
            assert calls[0] == ["Q1\nA1", "Q2\nA2"]

            ids = [
                row[0]
                for row in store.conn.execute(
                    "SELECT id FROM chunks WHERE session_id = ?", ("sess-remove",)
                ).fetchall()
            ]
            assert len(ids) == 1
            assert ids[0] == make_chunk_id("sess-remove", 0, "Q1\nA1")

            # 削除のみのターンでも session は upsert され続けている
            session_row = store.conn.execute(
                "SELECT project FROM sessions WHERE session_id = ?", ("sess-remove",)
            ).fetchone()
            assert session_row is not None
            assert session_row[0] == "proj"

            # chunk_vec_map に消えた chunk の孤立エントリが残っていない
            orphan_count = store.conn.execute(
                """
                SELECT COUNT(*) FROM chunk_vec_map
                WHERE chunk_id NOT IN (SELECT id FROM chunks)
                """
            ).fetchone()[0]
            assert orphan_count == 0
        finally:
            store.close()

    def test_changed_content_replaces_chunk(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """同じ index で content が変わると、旧 chunk_id は削除され新 chunk_id が insert される"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            persist_chunks(
                session_id="sess-change",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=_records_from_pairs([("Q1", "A1"), ("Q2", "A2")]),
                embedder=fake_embedder,
                store=store,
            )

            # index=1 の content が "Q2\nA2-edited" に変わるケース
            persist_chunks(
                session_id="sess-change",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=_records_from_pairs([
                    ("Q1", "A1"),
                    ("Q2", "A2-edited"),
                ]),
                embedder=fake_embedder,
                store=store,
            )

            calls = _encode_calls(fake_embedder)
            # 2 回目は 1 件だけ新規 embed
            assert len(calls) == 2
            assert calls[1] == ["Q2\nA2-edited"]

            ids = {
                row[0]
                for row in store.conn.execute(
                    "SELECT id FROM chunks WHERE session_id = ?", ("sess-change",)
                ).fetchall()
            }
            assert make_chunk_id("sess-change", 0, "Q1\nA1") in ids
            assert make_chunk_id("sess-change", 1, "Q2\nA2-edited") in ids
            # 旧 chunk_id は除去されている
            assert make_chunk_id("sess-change", 1, "Q2\nA2") not in ids
        finally:
            store.close()

    def test_empty_chunk_records_is_noop(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """空入力では embed もセッション作成も行わない"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            persist_chunks(
                session_id="sess-empty",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=[],
                embedder=fake_embedder,
                store=store,
            )

            assert _encode_calls(fake_embedder) == []
            row = store.conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?", ("sess-empty",)
            ).fetchone()
            assert row[0] == 0
        finally:
            store.close()

    def test_recorded_source_is_set_on_first_run(
        self,
        tmp_path: Path,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """初回 persist で session_sources.recorded_source が設定される"""
        config = Config(general={"data_dir": str(tmp_path)})
        store = MemoryStore(config)
        try:
            persist_chunks(
                session_id="sess-source",
                project_name="proj",
                work_dir=str(tmp_path),
                chunk_records=_records_from_pairs([("Q1", "A1")]),
                embedder=fake_embedder,
                store=store,
            )

            row = store.conn.execute(
                """
                SELECT recorded_source
                FROM session_sources
                WHERE session_id = ?
                """,
                ("sess-source",),
            ).fetchone()
            assert row is not None
            assert row[0] == "claude"
        finally:
            store.close()
