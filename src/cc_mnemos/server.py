"""MCP サーバー — 低レベルmcp SDKで実装

FastMCPのstdioトランスポートではtools/call時にハングする問題があるため、
mcp.server.Server + mcp.server.stdio.stdio_server を直接使用する
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from functools import partial
from typing import Protocol, TypeVar, cast

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from cc_mnemos import search_worker_control
from cc_mnemos.config import Config
from cc_mnemos.search_worker_control import (
    WORKER_HOST,
    WORKER_PORT,
)
from cc_mnemos.store import MemoryStore, StoreStats

logger = logging.getLogger(__name__)
JsonObject = dict[str, object]
HandlerT = TypeVar("HandlerT", bound=Callable[..., object])


class MCPServerProtocol(Protocol):
    def list_tools(self) -> Callable[[HandlerT], HandlerT]:
        ...

    def call_tool(self) -> Callable[[HandlerT], HandlerT]:
        ...

    async def run(
        self,
        read_stream: object,
        write_stream: object,
        initialization_options: object,
    ) -> None:
        ...

    def create_initialization_options(self) -> object:
        ...

# ---------------------------------------------------------------------------
# グローバル Config (遅延ロード)
# ---------------------------------------------------------------------------
_global_config: Config | None = None


def _load_config() -> Config:
    """グローバル設定を遅延ロードして返す"""
    global _global_config  # noqa: PLW0603
    if _global_config is None:
        _global_config = Config.load()
    return _global_config


# ---------------------------------------------------------------------------
# テスト用内部ヘルパー (同期)
# ---------------------------------------------------------------------------
def _get_stats(config: Config | None = None) -> StoreStats:
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


def _decode_search_results(payload: str) -> list[JsonObject]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        return []
    results: list[JsonObject] = []
    for item in raw:
        if isinstance(item, dict):
            results.append({str(key): value for key, value in item.items()})
    return results


def _coerce_tags(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [str(tag) for tag in value]


def _coerce_project(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_limit(value: object, default: int = 10) -> int:
    return value if isinstance(value, int) else default


# ---------------------------------------------------------------------------
# 同期検索の実装 (スレッドプールで実行される)
# ---------------------------------------------------------------------------
_WORKER_REQUEST_TIMEOUT_SECONDS = 30.0
_WORKER_COMMUNICATION_ATTEMPTS = 2
_worker_started = False


def _ensure_worker() -> bool:
    """検索ワーカーデーモンが起動していなければ起動する

    プロセスローカルなキャッシュフラグだけ管理し、起動と ready 確認は
    ``search_worker_control.ensure_worker`` に委譲する
    """
    global _worker_started  # noqa: PLW0603
    if _worker_started and search_worker_control.is_search_worker_available():
        return True

    _worker_started = False
    if search_worker_control.ensure_worker():
        _worker_started = True
        return True
    return False


def _search_memory_sync(
    query: str,
    tags: list[str] | None = None,
    project: str | None = None,
    limit: int = 10,
) -> list[JsonObject]:
    """デーモンワーカー経由でハイブリッド検索を実行する"""
    global _worker_started  # noqa: PLW0603
    import socket

    if not _ensure_worker():
        return []

    request = json.dumps({
        "query": query,
        "tags": tags,
        "project": project,
        "limit": limit,
    })

    for attempt in range(_WORKER_COMMUNICATION_ATTEMPTS):
        try:
            with socket.create_connection(
                (WORKER_HOST, WORKER_PORT),
                timeout=_WORKER_REQUEST_TIMEOUT_SECONDS,
            ) as sock:
                sock.sendall(request.encode("utf-8") + b"\n")
                # レスポンスを読み取り
                chunks: list[bytes] = []
                while True:
                    data = sock.recv(65536)
                    if not data:
                        break
                    chunks.append(data)
                response = b"".join(chunks).decode("utf-8")
                return _decode_search_results(response)
        except Exception:  # noqa: BLE001
            if attempt < _WORKER_COMMUNICATION_ATTEMPTS - 1:
                logger.warning("search worker communication failed, checking readiness")
                _worker_started = False
                if not _ensure_worker():
                    break
            else:
                logger.exception("search worker communication failed after retry")
    return []


# ---------------------------------------------------------------------------
# MCP サーバー定義
# ---------------------------------------------------------------------------
server: MCPServerProtocol = cast(MCPServerProtocol, Server("cc-mnemos"))


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
async def handle_call_tool(name: str, arguments: JsonObject) -> list[TextContent]:
    """ツール呼び出しを処理する"""
    loop = asyncio.get_running_loop()

    if name == "search_memory":
        query = arguments.get("query", "")
        query_text = query if isinstance(query, str) else ""
        # ワーカー起動をスレッドプールで実行（ブロッキング処理をイベントループから分離）
        results = await loop.run_in_executor(
            None,
            partial(
                _search_memory_sync,
                query_text,
                _coerce_tags(arguments.get("tags")),
                _coerce_project(arguments.get("project")),
                _coerce_limit(arguments.get("limit", 10)),
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
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_server() -> None:
    """MCPサーバーをstdioトランスポートで起動する"""
    asyncio.run(_run_server_async())


if __name__ == "__main__":
    run_server()
