"""MCP サーバー — 低レベルmcp SDKで実装

FastMCPのstdioトランスポートではtools/call時にハングする問題があるため、
mcp.server.Server + mcp.server.stdio.stdio_server を直接使用する
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from functools import partial
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from cc_mnemos.embedder import Embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# グローバル Config / Embedder (遅延ロード・キャッシュ、スレッドセーフ)
# ---------------------------------------------------------------------------
_global_config: Config | None = None
_global_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def _load_config() -> Config:
    """グローバル設定を遅延ロードして返す"""
    global _global_config  # noqa: PLW0603
    if _global_config is None:
        _global_config = Config.load()
    return _global_config


def _load_embedder() -> Embedder:
    """グローバルEmbedderを遅延ロードして返す（スレッドセーフ、モデルは1回だけロード）"""
    global _global_embedder  # noqa: PLW0603
    if _global_embedder is not None:
        return _global_embedder
    with _embedder_lock:
        if _global_embedder is None:
            from cc_mnemos.embedder import Embedder as EmbedderClass

            _global_embedder = EmbedderClass(_load_config())
        return _global_embedder


# ---------------------------------------------------------------------------
# テスト用内部ヘルパー (同期)
# ---------------------------------------------------------------------------
def _get_stats(config: Config | None = None) -> dict[str, int | dict[str, int]]:
    """統計情報を取得する内部ヘルパー"""
    cfg = config if config is not None else _load_config()
    store = MemoryStore(cfg)
    try:
        return store.get_stats()
    finally:
        store.close()


def _list_projects(config: Config | None = None) -> list[str]:
    """プロジェクト一覧を取得する内部ヘルパー"""
    cfg = config if config is not None else _load_config()
    store = MemoryStore(cfg)
    try:
        return store.list_projects()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 同期検索の実装 (スレッドプールで実行される)
# ---------------------------------------------------------------------------
def _search_memory_sync(
    query: str,
    tags: list[str] | None = None,
    project: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """ハイブリッド検索の同期実装"""
    cfg = _load_config()
    embedder = _load_embedder()
    store = MemoryStore(cfg)
    try:
        query_embedding = embedder.encode_query(query)
        return store.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            tags=tags,
            project=project,
            limit=limit,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# MCP サーバー定義
# ---------------------------------------------------------------------------
server = Server("cc-mnemos")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """利用可能なツール一覧を返す"""
    return [
        Tool(
            name="search_memory",
            description=(
                "過去の会話記憶をハイブリッド検索で取得する\n\n"
                "以下の場合に必ず呼び出すこと:\n"
                "- ユーザーが過去の決定を参照した時\n"
                "- 新しい画面・コンポーネントの作成時\n"
                "- コーディングスタイルや規約に関する判断時\n"
                "- 過去に類似の問題を扱った可能性がある時\n"
                "- ユーザーが「覚えてる？」「記憶を確認」と言った時"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ文字列"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "タグフィルタ",
                    },
                    "project": {"type": "string", "description": "プロジェクトフィルタ"},
                    "limit": {
                        "type": "integer",
                        "description": "結果の最大件数",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memory_stats",
            description="記憶の統計情報を返す",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_projects",
            description="記憶に存在するプロジェクト一覧を返す",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """ツール呼び出しを処理する"""
    loop = asyncio.get_running_loop()

    if name == "search_memory":
        results = await loop.run_in_executor(
            None,
            partial(
                _search_memory_sync,
                arguments.get("query", ""),
                arguments.get("tags"),
                arguments.get("project"),
                arguments.get("limit", 10),
            ),
        )
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]

    if name == "get_memory_stats":
        stats = await loop.run_in_executor(None, _get_stats)
        return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False))]

    if name == "list_projects":
        projects = await loop.run_in_executor(None, _list_projects)
        return [TextContent(type="text", text=json.dumps(projects, ensure_ascii=False))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------
async def _run_server_async() -> None:
    """MCPサーバーを非同期で起動する"""
    import threading

    threading.Thread(target=_load_embedder, daemon=True).start()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_server() -> None:
    """MCPサーバーをstdioトランスポートで起動する"""
    asyncio.run(_run_server_async())


if __name__ == "__main__":
    run_server()
