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

logger = logging.getLogger(__name__)


def _handle_client(
    conn: socket.socket,
    embedder: object,
    config: object,
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
        store = MemoryStore(config)  # type: ignore[arg-type]
        try:
            qe = embedder.encode_query(request["query"])  # type: ignore[union-attr]
            results = store.hybrid_search(
                query_text=request["query"],
                query_embedding=qe,
                tags=request.get("tags"),
                project=request.get("project"),
                limit=request.get("limit", 10),
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
        from cc_mnemos.config import Config
        from cc_mnemos.embedder import Embedder
        from cc_mnemos.store import MemoryStore

        cfg = Config.load()
        embedder = Embedder(cfg)
        store = MemoryStore(cfg)
        try:
            qe = embedder.encode_query(args["query"])
            results = store.hybrid_search(
                query_text=args["query"],
                query_embedding=qe,
                tags=args.get("tags"),
                project=args.get("project"),
                limit=args.get("limit", 10),
            )
            print(json.dumps(results, ensure_ascii=False))
        finally:
            store.close()


if __name__ == "__main__":
    main()
