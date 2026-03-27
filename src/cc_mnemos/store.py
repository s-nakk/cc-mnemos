"""SQLiteストレージ + ハイブリッド検索モジュール

FTS5全文検索とベクトル検索(sqlite-vec or numpyフォールバック)を組み合わせた
RRFベースのハイブリッド検索を提供する
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# スキーマ定義SQL
# ---------------------------------------------------------------------------
_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project    TEXT NOT NULL,
    work_dir   TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    summary    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    role_user     TEXT,
    role_assistant TEXT,
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    token_count   INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB
);

CREATE TABLE IF NOT EXISTS chunk_vec_map (
    rowid_int  INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id   TEXT NOT NULL UNIQUE REFERENCES chunks(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_session
    ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at
    ON chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_tags
    ON chunks(tags);
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    content,
    tokenize='trigram'
);
"""

_CREATE_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(id, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
    DELETE FROM chunks_fts WHERE id = old.id;
    INSERT INTO chunks_fts(id, content) VALUES (new.id, new.content);
END;
"""


class MemoryStore:
    """SQLiteベースのメモリストア

    sqlite-vecが利用可能な場合はネイティブベクトル検索を使用し、
    利用不可の場合はnumpyによるフォールバックで動作する
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._use_sqlite_vec = False

        # データディレクトリ作成
        config.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite接続
        self.conn = sqlite3.connect(str(config.db_path))
        self.conn.row_factory = sqlite3.Row

        # sqlite-vec拡張の読み込みを試行
        self._try_load_sqlite_vec()

        # スキーマ初期化
        self._init_schema()

    def _try_load_sqlite_vec(self) -> None:
        """sqlite-vec拡張の読み込みを試行する

        読み込みに失敗した場合はnumpyフォールバックを使用する
        """
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._use_sqlite_vec = True
            logger.info("sqlite-vec拡張を読み込みました")
        except Exception:  # noqa: BLE001
            self._use_sqlite_vec = False
            logger.info("sqlite-vec拡張が利用不可のためnumpyフォールバックを使用します")

    def _init_schema(self) -> None:
        """データベーススキーマを初期化する"""
        # WALモード + busy_timeout + 外部キー制約
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")

        # テーブル作成
        self.conn.executescript(_CREATE_TABLES)

        # FTS5テーブル作成
        self.conn.executescript(_CREATE_FTS)

        # FTSトリガー作成
        self.conn.executescript(_CREATE_FTS_TRIGGERS)

        # schema_versionの初期化
        row = self.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            self.conn.execute(
                "INSERT INTO schema_version(version) VALUES(?)", (SCHEMA_VERSION,)
            )

        # sqlite-vecテーブル作成
        if self._use_sqlite_vec:
            dim = self._config.embedding_dimension
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
                f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )

        # マイグレーション
        self._migrate()

        self.conn.commit()

    def _migrate(self) -> None:
        """スキーマバージョンに基づいてマイグレーションを実行する"""
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            return
        current = int(row[0])
        if current >= SCHEMA_VERSION:
            return

        # v1 → v2: vec_chunksをcosine距離で再構築
        if current < 2 and self._use_sqlite_vec:
            logger.info("マイグレーション v1→v2: vec_chunksをcosine距離で再構築します")
            dim = self._config.embedding_dimension
            self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self.conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks "
                f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )
            # chunk_vec_mapをリセットして再構築
            self.conn.execute("DELETE FROM chunk_vec_map")
            rows = self.conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
            for r in rows:
                chunk_id = r[0]
                emb_blob = r[1]
                if emb_blob is None:
                    continue
                self.conn.execute(
                    "INSERT INTO chunk_vec_map(chunk_id) VALUES(?)", (chunk_id,)
                )
                int_rowid = self.conn.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
                self.conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?)",
                    (int_rowid, emb_blob),
                )

        self.conn.execute(
            "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
        )
        logger.info("スキーマをバージョン %d に更新しました", SCHEMA_VERSION)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """複数操作を1トランザクションで実行する"""
        try:
            self.conn.execute("BEGIN")
            yield
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def delete_session_chunks(self, session_id: str, *, commit: bool = True) -> int:
        """セッションの既存チャンクをすべて削除する

        再インジェスト時に古いチャンクを除去してから新しいチャンクを挿入するために使用する。
        chunks, chunk_vec_map, vec_chunks, chunks_fts を整合的にクリーンアップする

        Args:
            session_id: 対象セッションID
            commit: 自動コミットするか

        Returns:
            削除したチャンク数
        """
        chunk_rows = self.conn.execute(
            "SELECT id FROM chunks WHERE session_id = ?", (session_id,)
        ).fetchall()

        if not chunk_rows:
            return 0

        chunk_ids = [row[0] for row in chunk_rows]

        placeholders = ",".join("?" for _ in chunk_ids)

        # vec_chunks + chunk_vec_map のバッチクリーンアップ
        if self._use_sqlite_vec:
            map_rows = self.conn.execute(
                f"SELECT rowid_int FROM chunk_vec_map WHERE chunk_id IN ({placeholders})",  # noqa: S608
                chunk_ids,
            ).fetchall()
            if map_rows:
                rowid_phs = ",".join("?" for _ in map_rows)
                rowids = [r[0] for r in map_rows]
                self.conn.execute(
                    f"DELETE FROM vec_chunks WHERE rowid IN ({rowid_phs})",  # noqa: S608
                    rowids,
                )

        # chunk_vec_map 削除
        self.conn.execute(
            f"DELETE FROM chunk_vec_map WHERE chunk_id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        )

        # chunks 削除 (FTSトリガーが chunks_fts も自動削除)
        deleted = self.conn.execute(
            "DELETE FROM chunks WHERE session_id = ?", (session_id,)
        ).rowcount

        if commit:
            self.conn.commit()
        return deleted

    def insert_session(
        self,
        *,
        session_id: str,
        project: str,
        work_dir: str,
        started_at: str,
        ended_at: str | None = None,
        summary: str | None = None,
        commit: bool = True,
    ) -> None:
        """セッションレコードを挿入する

        Args:
            session_id: セッションID
            project: プロジェクト名
            work_dir: 作業ディレクトリ
            started_at: 開始日時(ISO 8601)
            ended_at: 終了日時(ISO 8601)
            summary: セッション要約
        """
        self.conn.execute(
            """
            INSERT INTO sessions(
                session_id, project, work_dir, started_at, ended_at, summary
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                project=excluded.project,
                work_dir=excluded.work_dir,
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                summary=excluded.summary
            """,
            (session_id, project, work_dir, started_at, ended_at, summary),
        )
        if commit:
            self.conn.commit()

    def insert_chunk(
        self,
        chunk: dict[str, str | int],
        embedding: np.ndarray,
        commit: bool = True,
    ) -> None:
        """チャンクレコードと埋め込みベクトルを挿入する

        Args:
            chunk: チャンクデータ辞書
            embedding: 埋め込みベクトル (float32)
        """
        embedding_bytes = embedding.astype(np.float32).tobytes()
        chunk_id = str(chunk["id"])

        # INSERT OR REPLACEする前にFTSトリガーの重複を防ぐため既存行を確認
        existing = self.conn.execute(
            "SELECT id FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()

        if existing:
            # 既存レコードの更新
            self.conn.execute(
                """
                UPDATE chunks SET session_id=?, role_user=?, role_assistant=?,
                    content=?, tags=?, created_at=?, token_count=?, embedding=?
                WHERE id=?
                """,
                (
                    chunk["session_id"],
                    chunk.get("role_user"),
                    chunk.get("role_assistant"),
                    chunk["content"],
                    chunk.get("tags", "[]"),
                    chunk["created_at"],
                    chunk.get("token_count", 0),
                    embedding_bytes,
                    chunk_id,
                ),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO chunks(id, session_id, role_user, role_assistant,
                                   content, tags, created_at, token_count, embedding)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk["session_id"],
                    chunk.get("role_user"),
                    chunk.get("role_assistant"),
                    chunk["content"],
                    chunk.get("tags", "[]"),
                    chunk["created_at"],
                    chunk.get("token_count", 0),
                    embedding_bytes,
                ),
            )

        # sqlite-vec仮想テーブルへの挿入(整数rowid経由のマッピング)
        if self._use_sqlite_vec:
            # マッピング行の取得または作成
            map_row = self.conn.execute(
                "SELECT rowid_int FROM chunk_vec_map WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if map_row:
                int_rowid = map_row[0]
                self.conn.execute(
                    "UPDATE vec_chunks SET embedding = ? WHERE rowid = ?",
                    (embedding_bytes, int_rowid),
                )
            else:
                self.conn.execute(
                    "INSERT INTO chunk_vec_map(chunk_id) VALUES(?)",
                    (chunk_id,),
                )
                int_rowid = self.conn.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
                self.conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?)",
                    (int_rowid, embedding_bytes),
                )

        if commit:
            self.conn.commit()

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> list[dict[str, str | int | float]]:
        """FTS5全文検索を実行する

        trigramで3文字以上のトークンはFTS5 MATCH(AND結合)、
        3文字未満の短語はLIKEフォールバックで検索する。
        AND結合で0件の場合はOR結合にフォールバックする

        Args:
            query: 検索クエリ文字列
            limit: 結果の最大件数

        Returns:
            マッチしたチャンクのリスト
        """
        long_tokens, short_tokens = self._split_query_tokens(query)

        results: list[dict[str, str | int | float]] = []
        fetch_limit = limit if tags is None else self._count_candidate_chunks(project=project)
        if fetch_limit <= 0:
            return []

        # FTS5検索 (3文字以上のトークン)
        if long_tokens:
            # AND結合で検索
            and_query = " AND ".join(f'"{t}"' for t in long_tokens)
            rows = self._run_fts_query(and_query, fetch_limit=fetch_limit, project=project)
            results = [dict(row) for row in rows]

            # AND結合で0件の場合はOR結合にフォールバック
            if not results and len(long_tokens) > 1:
                or_query = " OR ".join(f'"{t}"' for t in long_tokens)
                rows = self._run_fts_query(or_query, fetch_limit=fetch_limit, project=project)
                results = [dict(row) for row in rows]

        # 短語のLIKEフォールバック (3文字未満)
        if short_tokens:
            seen_ids = {str(r["id"]) for r in results}
            for token in short_tokens:
                rows = self._run_like_query(
                    token,
                    fetch_limit=fetch_limit,
                    project=project,
                )
                for row in rows:
                    d = dict(row)
                    if str(d["id"]) not in seen_ids:
                        results.append(d)
                        seen_ids.add(str(d["id"]))

        filtered = self._filter_results(results, tags=tags)
        return filtered[:limit]

    @staticmethod
    def _split_query_tokens(query: str) -> tuple[list[str], list[str]]:
        """クエリをFTS5用トークンと短語に分割する

        Args:
            query: 生のクエリ文字列

        Returns:
            (3文字以上のトークンリスト, 3文字未満の短語リスト)
        """
        special_chars = set('"*():^{}[]!&|~@#$%')
        cleaned = ""
        for ch in query:
            if ch in special_chars:
                cleaned += " "
            else:
                cleaned += ch

        tokens = cleaned.split()
        long_tokens: list[str] = []
        short_tokens: list[str] = []
        for t in tokens:
            if not t:
                continue
            if len(t) >= 3:
                long_tokens.append(t)
            else:
                short_tokens.append(t)
        return long_tokens, short_tokens

    def vector_search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int = 10,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> list[dict[str, str | int | float]]:
        """ベクトル類似度検索を実行する

        Args:
            query_embedding: クエリの埋め込みベクトル
            limit: 結果の最大件数

        Returns:
            類似度の高いチャンクのリスト (distance付き)
        """
        if self._use_sqlite_vec and tags is None and project is None:
            return self._sqlite_vec_search(query_embedding, limit=limit)
        return self._numpy_vector_search(
            query_embedding,
            limit=limit,
            tags=tags,
            project=project,
        )

    def _sqlite_vec_search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int = 10,
    ) -> list[dict[str, str | int | float]]:
        """sqlite-vecによるベクトル検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            limit: 結果の最大件数

        Returns:
            類似度の高いチャンクのリスト
        """
        query_bytes = query_embedding.astype(np.float32).tobytes()
        # vec0 KNNクエリはWHERE句に k=? が必要
        rows = self.conn.execute(
            """
            SELECT c.*, sub.distance
            FROM (
                SELECT rowid, distance
                FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            ) sub
            JOIN chunk_vec_map m ON m.rowid_int = sub.rowid
            JOIN chunks c ON c.id = m.chunk_id
            ORDER BY sub.distance
            """,
            (query_bytes, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _numpy_vector_search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int = 10,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> list[dict[str, str | int | float]]:
        """numpyによるコサイン類似度ベクトル検索(フォールバック)

        Args:
            query_embedding: クエリの埋め込みベクトル
            limit: 結果の最大件数

        Returns:
            類似度の高いチャンクのリスト (distance付き)
        """
        query = """
            SELECT c.id, c.session_id, c.role_user, c.role_assistant,
                   c.content, c.tags, c.created_at, c.token_count, c.embedding
            FROM chunks c
            JOIN sessions s ON s.session_id = c.session_id
            WHERE c.embedding IS NOT NULL
        """
        params: list[str] = []
        if project is not None:
            query += " AND s.project = ?"
            params.append(project)
        rows = self.conn.execute(query, params).fetchall()

        if not rows:
            return []

        dim = self._config.embedding_dimension
        query_vec = query_embedding.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normalized = query_vec / query_norm

        scored: list[tuple[float, dict[str, str | int | float]]] = []
        for row in rows:
            row_data = dict(row)
            if not self._matches_tags(row_data, tags):
                continue
            emb_blob = row_data.get("embedding")
            if emb_blob is None:
                continue
            vec = np.frombuffer(bytes(emb_blob), dtype=np.float32)
            if len(vec) != dim:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            cosine_sim = float(np.dot(query_normalized, vec / vec_norm))
            # コサイン距離 (1 - similarity) に変換してsqlite-vecと互換にする
            distance = 1.0 - cosine_sim
            row_data["distance"] = distance
            row_data.pop("embedding", None)
            scored.append((distance, row_data))

        # 距離昇順(= 類似度降順)でソート
        scored.sort(key=lambda x: x[0])
        return [row_data for _, row_data in scored[:limit]]

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: np.ndarray,
        tags: list[str] | None = None,
        project: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, str | int | float]]:
        """FTS5 + ベクトル検索のRRFハイブリッド検索を実行する

        RRF(Reciprocal Rank Fusion)でスコアを統合し、
        時間減衰を適用した結果を返す

        Args:
            query_text: テキストクエリ
            query_embedding: クエリの埋め込みベクトル
            tags: タグフィルタ(指定時はいずれかを含むチャンクのみ)
            project: プロジェクトフィルタ
            limit: 結果の最大件数

        Returns:
            スコア降順でソートされたチャンクのリスト
        """
        k = self._config.rrf_k
        half_life = self._config.time_decay_half_life_days
        fts_weight = self._config.fts_weight
        vector_weight = self._config.vector_weight

        # 検索用にlimitを拡張(フィルタ後に絞るため)
        expanded_limit = limit * 5

        # FTS検索
        fts_results = self.fts_search(
            query_text,
            limit=expanded_limit,
            tags=tags,
            project=project,
        )

        # ベクトル検索
        vec_results = self.vector_search(
            query_embedding,
            limit=expanded_limit,
            tags=tags,
            project=project,
        )

        # 重み付きRRFスコア計算
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict[str, str | int | float]] = {}

        for rank, result in enumerate(fts_results):
            chunk_id = str(result["id"])
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + fts_weight / (k + rank + 1)
            chunk_data[chunk_id] = result

        for rank, result in enumerate(vec_results):
            chunk_id = str(result["id"])
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0) + vector_weight / (k + rank + 1)
            )
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # 時間減衰の適用
        now = datetime.now(tz=timezone.utc)
        for chunk_id, base_score in rrf_scores.items():
            data = chunk_data[chunk_id]
            created_at_str = str(data.get("created_at", ""))
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                created_at = now

            age_days = (now - created_at).total_seconds() / 86400.0
            # 指数減衰: decay = 0.5^(age_days / half_life)
            decay = math.pow(0.5, age_days / half_life)
            rrf_scores[chunk_id] = base_score * decay

        filtered_ids = list(rrf_scores)

        # スコア降順ソート
        filtered_ids.sort(key=lambda cid: rrf_scores[cid], reverse=True)

        # 結果構築
        results: list[dict[str, str | int | float]] = []
        for chunk_id in filtered_ids[:limit]:
            data = dict(chunk_data[chunk_id])
            data["score"] = rrf_scores[chunk_id]
            # embeddingバイナリは結果から除外
            data.pop("embedding", None)
            results.append(data)

        return results

    def _run_fts_query(
        self,
        match_query: str,
        *,
        fetch_limit: int,
        project: str | None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT c.*, rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.id
            JOIN sessions s ON s.session_id = c.session_id
            WHERE chunks_fts MATCH ?
        """
        params: list[str | int] = [match_query]
        if project is not None:
            query += " AND s.project = ?"
            params.append(project)
        query += " ORDER BY rank LIMIT ?"
        params.append(fetch_limit)
        return self.conn.execute(query, params).fetchall()

    def _run_like_query(
        self,
        token: str,
        *,
        fetch_limit: int,
        project: str | None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT c.*, 0 as rank
            FROM chunks c
            JOIN sessions s ON s.session_id = c.session_id
            WHERE c.content LIKE ?
        """
        params: list[str | int] = [f"%{token}%"]
        if project is not None:
            query += " AND s.project = ?"
            params.append(project)
        query += " LIMIT ?"
        params.append(fetch_limit)
        return self.conn.execute(query, params).fetchall()

    def _count_candidate_chunks(self, *, project: str | None) -> int:
        if project is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                JOIN sessions s ON s.session_id = c.session_id
                WHERE s.project = ?
                """,
                (project,),
            ).fetchone()[0]
        )

    @staticmethod
    def _filter_results(
        results: list[dict[str, str | int | float]],
        *,
        tags: list[str] | None,
    ) -> list[dict[str, str | int | float]]:
        if tags is None:
            return results
        return [result for result in results if MemoryStore._matches_tags(result, tags)]

    @staticmethod
    def _matches_tags(
        result: dict[str, str | int | float],
        tags: list[str] | None,
    ) -> bool:
        if tags is None:
            return True
        tags_raw = str(result.get("tags", "[]"))
        try:
            result_tags = json.loads(tags_raw)
        except json.JSONDecodeError:
            return False
        return any(tag in result_tags for tag in tags)

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """ストレージの統計情報を取得する

        Returns:
            チャンク数・セッション数・プロジェクト別統計を含む辞書
        """
        total_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_sessions = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        by_project_rows = self.conn.execute(
            """
            SELECT s.project, COUNT(c.id) as chunk_count
            FROM chunks c
            JOIN sessions s ON c.session_id = s.session_id
            GROUP BY s.project
            """
        ).fetchall()
        by_project: dict[str, int] = {row[0]: row[1] for row in by_project_rows}

        return {
            "total_chunks": total_chunks,
            "total_sessions": total_sessions,
            "by_project": by_project,
        }

    def list_projects(self) -> list[str]:
        """登録されているプロジェクト一覧を返す

        Returns:
            プロジェクト名のリスト
        """
        rows = self.conn.execute(
            "SELECT DISTINCT project FROM sessions ORDER BY project"
        ).fetchall()
        return [row[0] for row in rows]

    def get_recent_chunks(
        self,
        *,
        limit: int = 10,
        project: str | None = None,
    ) -> list[dict[str, str | int]]:
        """最近のチャンクを取得する

        Args:
            limit: 結果の最大件数
            project: プロジェクトフィルタ

        Returns:
            チャンクの辞書リスト(新しい順)
        """
        if project:
            rows = self.conn.execute(
                """
                SELECT c.id, c.session_id, c.role_user, c.role_assistant,
                       c.content, c.tags, c.created_at, c.token_count
                FROM chunks c
                JOIN sessions s ON c.session_id = s.session_id
                WHERE s.project = ?
                ORDER BY c.created_at DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, session_id, role_user, role_assistant,
                       content, tags, created_at, token_count
                FROM chunks
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_tagged_chunks(
        self,
        *,
        limit: int = 5,
        exclude_project: str | None = None,
        exclude_tags: list[str] | None = None,
    ) -> list[dict[str, str | int]]:
        """特定タグが付いたチャンクを横断的に取得する

        exclude_tagsのみのチャンクを除外し、実際にカテゴリ分類された
        知見のみを返す。プロジェクト除外もSQL側で処理する

        Args:
            limit: 結果の最大件数
            exclude_project: 除外するプロジェクト名
            exclude_tags: 除外するタグ名リスト

        Returns:
            チャンクの辞書リスト(新しい順)
        """
        if exclude_tags is None:
            exclude_tags = ["general"]

        # exclude_tagsのみのチャンクを除外するWHERE句を構築
        # JSON配列を文字列比較で除外（例: '["general"]'）
        exclude_patterns = [
            json.dumps([t]) for t in exclude_tags
        ]
        placeholders = ",".join("?" for _ in exclude_patterns)

        params: list[str | int] = list(exclude_patterns)

        query = f"""
            SELECT c.id, c.session_id, c.role_user, c.role_assistant,
                   c.content, c.tags, c.created_at, c.token_count
            FROM chunks c
            JOIN sessions s ON c.session_id = s.session_id
            WHERE c.tags NOT IN ({placeholders})
        """

        if exclude_project:
            query += " AND s.project != ?"
            params.append(exclude_project)

        query += " ORDER BY c.created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def deduplicate_chunks(self) -> int:
        """重複チャンクを削除する(同一session_id + contentの重複を除去)

        各(session_id, content)グループで最初のチャンクのみ残し、
        残りを削除する。関連する chunk_vec_map, vec_chunks, chunks_fts も整合的にクリーンアップする

        Returns:
            削除したチャンク数
        """
        # 重複チャンクのIDを取得 (各グループの最小IDのみ残す)
        dup_rows = self.conn.execute(
            """
            SELECT id FROM chunks
            WHERE id NOT IN (
                SELECT MIN(id) FROM chunks GROUP BY session_id, content
            )
            """
        ).fetchall()

        if not dup_rows:
            return 0

        dup_ids = [row[0] for row in dup_rows]
        total = len(dup_ids)

        # バッチ処理 (SQLiteパラメータ上限対策)
        batch_size = 500
        for start in range(0, total, batch_size):
            batch = dup_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)

            # vec_chunks バッチクリーンアップ
            if self._use_sqlite_vec:
                map_rows = self.conn.execute(
                    f"SELECT rowid_int FROM chunk_vec_map WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    batch,
                ).fetchall()
                if map_rows:
                    rowid_phs = ",".join("?" for _ in map_rows)
                    rowids = [r[0] for r in map_rows]
                    self.conn.execute(
                        f"DELETE FROM vec_chunks WHERE rowid IN ({rowid_phs})",  # noqa: S608
                        rowids,
                    )

            # chunk_vec_map 削除
            self.conn.execute(
                f"DELETE FROM chunk_vec_map WHERE chunk_id IN ({placeholders})",  # noqa: S608
                batch,
            )

            # chunks 削除 (FTSトリガーが chunks_fts も自動削除)
            self.conn.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
                batch,
            )

        self.conn.commit()
        return total

    def normalize_project_names(self) -> dict[str, str]:
        """大文字小文字が異なる重複プロジェクト名を統一する

        同一名の大文字小文字バリアントが存在する場合、
        最もチャンク数の多いバリアントに統一する

        Returns:
            統一された名前のマッピング {旧名: 新名}
        """
        rows = self.conn.execute(
            """
            SELECT s.project, COUNT(c.id) as cnt
            FROM sessions s
            LEFT JOIN chunks c ON s.session_id = c.session_id
            GROUP BY s.project
            ORDER BY cnt DESC
            """
        ).fetchall()

        # 大文字小文字を無視してグループ化
        groups: dict[str, list[tuple[str, int]]] = {}
        for row in rows:
            name = row[0]
            count = row[1]
            key = name.lower()
            if key not in groups:
                groups[key] = []
            groups[key].append((name, count))

        renames: dict[str, str] = {}
        for variants in groups.values():
            if len(variants) <= 1:
                continue
            # 最もチャンクが多いバリアントを正規名とする
            canonical = max(variants, key=lambda v: v[1])[0]
            for name, _ in variants:
                if name != canonical:
                    renames[name] = canonical
                    self.conn.execute(
                        "UPDATE sessions SET project = ? WHERE project = ?",
                        (canonical, name),
                    )

        if renames:
            self.conn.commit()
        return renames

    def close(self) -> None:
        """データベース接続を閉じる"""
        self.conn.close()
