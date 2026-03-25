"""search_memory用のデーモンワーカー

MCPサーバーのasyncioイベントループ内でtorch/sentence-transformersを
ロードするとstdioトランスポートがブロックされるため、
このワーカーを別プロセスで常駐させ、ソケットでリクエストを受け付ける

起動方法:
    python _search_worker.py --daemon PORT
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder


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


def _handle_client(
    conn: socket.socket,
    embedder: Embedder,
    config: Config,
) -> None:
    """1クライアントのリクエストを処理する"""
    from cc_mnemos.store import MemoryStore

    try:
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        request = json.loads(data.decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request payload must be an object")
        query = request.get("query")
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        store = MemoryStore(config)
        try:
            qe = embedder.encode_query(query)
            results = store.hybrid_search(
                query_text=query,
                query_embedding=qe,
                tags=_coerce_tags(request.get("tags")),
                project=_coerce_project(request.get("project")),
                limit=_coerce_limit(request.get("limit", 10)),
            )
            response = json.dumps(results, ensure_ascii=False)
        finally:
            store.close()

        conn.sendall(response.encode("utf-8"))
    except Exception:
        logger.exception("Worker request handling failed")
        conn.sendall(b"[]")
    finally:
        conn.close()


def run_daemon(port: int) -> None:
    """デーモンモードで起動し、ソケットでリクエストを待ち受ける"""
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder

    config = Config.load()
    embedder = Embedder(config)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)

    while True:
        conn, _ = srv.accept()
        t = threading.Thread(
            target=_handle_client,
            args=(conn, embedder, config),
            daemon=True,
        )
        t.start()


def main() -> None:
    """エントリポイント"""
    if len(sys.argv) >= 3 and sys.argv[1] == "--daemon":
        port = int(sys.argv[2])
        run_daemon(port)
    elif len(sys.argv) >= 2:
        # レガシー: 単発実行モード
        args = json.loads(sys.argv[1])
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        from cc_mnemos.config import Config
        from cc_mnemos.embedder import Embedder
        from cc_mnemos.store import MemoryStore

        cfg = Config.load()
        embedder = Embedder(cfg)
        store = MemoryStore(cfg)
        try:
            query = args.get("query")
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            qe = embedder.encode_query(query)
            results = store.hybrid_search(
                query_text=query,
                query_embedding=qe,
                tags=_coerce_tags(args.get("tags")),
                project=_coerce_project(args.get("project")),
                limit=_coerce_limit(args.get("limit", 10)),
            )
            print(json.dumps(results, ensure_ascii=False))
        finally:
            store.close()


if __name__ == "__main__":
    main()
