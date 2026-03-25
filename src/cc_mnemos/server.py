"""FastMCP サーバー — MCP ツールを提供する

search_memory, get_memory_stats, list_projects の3ツールを公開し、
テスト用の内部ヘルパー (_get_stats, _list_projects) も提供する
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from cc_mnemos.embedder import Embedder

logger = logging.getLogger(__name__)

mcp = FastMCP("cc-mnemos")

# ---------------------------------------------------------------------------
# グローバル Config / Embedder (遅延ロード・キャッシュ)
# ---------------------------------------------------------------------------
_global_config: Config | None = None
_global_embedder: Embedder | None = None


def _load_config() -> Config:
    """グローバル設定を遅延ロードして返す"""
    global _global_config  # noqa: PLW0603
    if _global_config is None:
        _global_config = Config.load()
    return _global_config


def _load_embedder() -> Embedder:
    """グローバルEmbedderを遅延ロードして返す（モデルは1回だけロード）"""
    global _global_embedder  # noqa: PLW0603
    if _global_embedder is None:
        from cc_mnemos.embedder import Embedder as EmbedderClass

        _global_embedder = EmbedderClass(_load_config())
    return _global_embedder


# ---------------------------------------------------------------------------
# テスト用内部ヘルパー
# ---------------------------------------------------------------------------
def _get_stats(config: Config | None = None) -> dict[str, int | dict[str, int]]:
    """統計情報を取得する内部ヘルパー

    Args:
        config: 設定オブジェクト (テスト用。Noneの場合はグローバル設定を使用)

    Returns:
        チャンク数・セッション数・プロジェクト別統計を含む辞書
    """
    cfg = config if config is not None else _load_config()
    store = MemoryStore(cfg)
    try:
        return store.get_stats()
    finally:
        store.close()


def _list_projects(config: Config | None = None) -> list[str]:
    """プロジェクト一覧を取得する内部ヘルパー

    Args:
        config: 設定オブジェクト (テスト用。Noneの場合はグローバル設定を使用)

    Returns:
        プロジェクト名のソート済みリスト
    """
    cfg = config if config is not None else _load_config()
    store = MemoryStore(cfg)
    try:
        return store.list_projects()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# MCP ツール
# ---------------------------------------------------------------------------
@mcp.tool()
def search_memory(
    query: str,
    tags: list[str] | None = None,
    project: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """過去の会話記憶をハイブリッド検索で取得する

    以下の場合に必ず呼び出すこと:
    - ユーザーが過去の決定を参照した時
    - 新しい画面・コンポーネントの作成時
    - コーディングスタイルや規約に関する判断時
    - 過去に類似の問題を扱った可能性がある時
    - ユーザーが「覚えてる？」「記憶を確認」と言った時

    Args:
        query: 検索クエリ文字列
        tags: タグフィルタ (いずれかを含むチャンクのみ)
        project: プロジェクトフィルタ
        limit: 結果の最大件数

    Returns:
        スコア降順でソートされたチャンクのリスト
    """
    cfg = _load_config()
    embedder = _load_embedder()
    store = MemoryStore(cfg)
    try:
        query_embedding = embedder.encode_query(query)
        results = store.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            tags=tags,
            project=project,
            limit=limit,
        )
        return results
    finally:
        store.close()


@mcp.tool()
def get_memory_stats() -> dict:
    """記憶の統計情報を返す

    Returns:
        チャンク数・セッション数・プロジェクト別統計を含む辞書
    """
    return _get_stats()


@mcp.tool()
def list_projects() -> list[str]:
    """記憶に存在するプロジェクト一覧を返す

    Returns:
        プロジェクト名のソート済みリスト
    """
    return _list_projects()


# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------
def run_server() -> None:
    """MCPサーバーをstdioトランスポートで起動する"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
