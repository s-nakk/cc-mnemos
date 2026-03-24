"""SQLiteストレージ + ハイブリッド検索モジュール

FTS5全文検索とベクトル検索(sqlite-vec or numpyフォールバック)を組み合わせた
RRFベースのハイブリッド検索を提供する
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

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
    tokenize='unicode61'
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
        # WALモード + busy_timeout
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")

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
                f"USING vec0(embedding float[{dim}])"
            )

        self.conn.commit()

    def insert_session(
        self,
        *,
        session_id: str,
        project: str,
        work_dir: str,
        started_at: str,
        ended_at: str | None = None,
        summary: str | None = None,
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
            INSERT OR REPLACE INTO sessions(
                session_id, project, work_dir, started_at, ended_at, summary
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (session_id, project, work_dir, started_at, ended_at, summary),
        )
        self.conn.commit()

    def insert_chunk(
        self,
        chunk: dict[str, str | int],
        embedding: np.ndarray,
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

        self.conn.commit()

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, str | int | float]]:
        """FTS5全文検索を実行する

        Args:
            query: 検索クエリ文字列
            limit: 結果の最大件数

        Returns:
            マッチしたチャンクのリスト
        """
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        rows = self.conn.execute(
            """
            SELECT c.*, rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (sanitized, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """FTS5クエリ用に文字列をサニタイズする

        Args:
            query: 生のクエリ文字列

        Returns:
            FTS5に安全なクエリ文字列
        """
        # 特殊文字およびハイフンをスペースに置換し、各トークンをダブルクォートで囲む
        # unicode61トークナイザはハイフンで分割するため、ハイフンもセパレータとして扱う
        special_chars = set('"*():^{}[]!&|~@#$%-')
        cleaned = ""
        for ch in query:
            if ch in special_chars:
                cleaned += " "
            else:
                cleaned += ch

        tokens = cleaned.split()
        if not tokens:
            return ""

        # 各トークンをダブルクォートで囲み、OR結合
        quoted = [f'"{token}"' for token in tokens if token]
        return " OR ".join(quoted)

    def vector_search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int = 10,
    ) -> list[dict[str, str | int | float]]:
        """ベクトル類似度検索を実行する

        Args:
            query_embedding: クエリの埋め込みベクトル
            limit: 結果の最大件数

        Returns:
            類似度の高いチャンクのリスト (distance付き)
        """
        if self._use_sqlite_vec:
            return self._sqlite_vec_search(query_embedding, limit=limit)
        return self._numpy_vector_search(query_embedding, limit=limit)

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
    ) -> list[dict[str, str | int | float]]:
        """numpyによるコサイン類似度ベクトル検索(フォールバック)

        Args:
            query_embedding: クエリの埋め込みベクトル
            limit: 結果の最大件数

        Returns:
            類似度の高いチャンクのリスト (distance付き)
        """
        rows = self.conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        dim = self._config.embedding_dimension
        query_vec = query_embedding.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normalized = query_vec / query_norm

        scored: list[tuple[str, float]] = []
        for row in rows:
            chunk_id = row[0]
            emb_blob = row[1]
            if emb_blob is None:
                continue
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if len(vec) != dim:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            cosine_sim = float(np.dot(query_normalized, vec / vec_norm))
            # コサイン距離 (1 - similarity) に変換してsqlite-vecと互換にする
            distance = 1.0 - cosine_sim
            scored.append((chunk_id, distance))

        # 距離昇順(= 類似度降順)でソート
        scored.sort(key=lambda x: x[1])
        top_ids = scored[:limit]

        results: list[dict[str, str | int | float]] = []
        for chunk_id, distance in top_ids:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["distance"] = distance
                results.append(d)

        return results

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

        # 検索用にlimitを拡張(フィルタ後に絞るため)
        expanded_limit = limit * 5

        # FTS検索
        fts_results = self.fts_search(query_text, limit=expanded_limit)

        # ベクトル検索
        vec_results = self.vector_search(query_embedding, limit=expanded_limit)

        # RRFスコア計算
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict[str, str | int | float]] = {}

        for rank, result in enumerate(fts_results):
            chunk_id = str(result["id"])
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_data[chunk_id] = result

        for rank, result in enumerate(vec_results):
            chunk_id = str(result["id"])
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
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

        # フィルタリング
        filtered_ids: list[str] = []
        for chunk_id in rrf_scores:
            data = chunk_data[chunk_id]

            # タグフィルタ
            if tags:
                chunk_tags_str = str(data.get("tags", "[]"))
                try:
                    chunk_tags = json.loads(chunk_tags_str)
                except json.JSONDecodeError:
                    chunk_tags = []
                if not any(t in chunk_tags for t in tags):
                    continue

            # プロジェクトフィルタ
            if project:
                session_id = str(data.get("session_id", ""))
                session = self.conn.execute(
                    "SELECT project FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None or session[0] != project:
                    continue

            filtered_ids.append(chunk_id)

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

    def close(self) -> None:
        """データベース接続を閉じる"""
        self.conn.close()
