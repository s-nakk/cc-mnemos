"""search / memorize 用のデーモンワーカー

MCP サーバーの asyncio イベントループ内で torch / sentence-transformers を
ロードすると stdio トランスポートがブロックされるため、このワーカーを
別プロセスで常駐させてソケットでリクエストを受け付ける
search リクエストはスレッドごとに並列に処理し、memorize リクエストは
専用キューに積んで単一ワーカースレッドで逐次処理する

起動方法:
    python _search_worker.py --daemon PORT
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import sys
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder
    from cc_mnemos.store import MemoryStore


# memorize キューの上限。これを超える未処理ジョブが溜まった場合は
# クライアントに queue_full を返してフォールバックを促す
MEMORIZE_QUEUE_MAXSIZE = 200

# memorize worker thread が新しいジョブを待つ最大時間 (秒)
# 定期的に wake してシャットダウン要求を検出できるようにする
_MEMORIZE_POLL_TIMEOUT_SECONDS = 1.0


# ---------------------------------------------------------------------------
# 引数の coercion
# ---------------------------------------------------------------------------
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
# 検索リクエスト処理 (従来通り thread-per-connection で並列処理)
# ---------------------------------------------------------------------------
def _handle_search_request(
    request: dict[str, object],
    embedder: Embedder,
    config: Config,
) -> bytes:
    """検索リクエストを処理して JSON バイト列を返す"""
    from cc_mnemos.store import MemoryStore

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
        return json.dumps(results, ensure_ascii=False).encode("utf-8")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# memorize リクエスト処理 (キュー投入のみ、実処理は memorize worker thread)
# ---------------------------------------------------------------------------
def _dispatch_memorize_request(
    request: dict[str, object],
    memorize_queue: queue.Queue[dict[str, object] | None] | None,
    memorize_alive: threading.Event | None,
) -> bytes:
    """memorize リクエストをキューに積み、ack バイト列を返す

    同一 ``session_id`` の未処理ジョブが既にキューに居る場合、それを上書きして
    再投入数を 1 件に抑える。memorize は差分更新で「現在の transcript の chunk
    集合」を最終状態として持つだけなので、途中の古いジョブを skip しても最新
    ジョブの処理結果に追いつく。Stop hook が連発するシナリオで RAM が線形に
    膨らむのを防ぐための重要な最適化

    Args:
        request: クライアントが送ってきたペイロード
        memorize_queue: memorize ジョブ用のキュー (未起動状態 ``None`` ならエラー)
        memorize_alive: memorize worker thread の生存フラグ
    """
    if memorize_queue is None or memorize_alive is None or not memorize_alive.is_set():
        return json.dumps({"ok": False, "error": "worker_not_ready"}).encode("utf-8")

    chunks = request.get("chunks")
    if not isinstance(chunks, list):
        return json.dumps({"ok": False, "error": "invalid_payload"}).encode("utf-8")

    session_id_raw = request.get("session_id")
    new_session_id = str(session_id_raw) if session_id_raw else ""

    # 同一 session_id の未処理ジョブがあれば上書き (CPython の Queue 内部 deque を触る)
    if new_session_id:
        with memorize_queue.mutex:
            internal_deque = memorize_queue.queue
            for index, item in enumerate(internal_deque):
                if (
                    isinstance(item, dict)
                    and str(item.get("session_id", "")) == new_session_id
                ):
                    internal_deque[index] = request
                    return json.dumps(
                        {"ok": True, "queued": len(chunks), "deduped": True},
                    ).encode("utf-8")

    try:
        memorize_queue.put_nowait(request)
    except queue.Full:
        return json.dumps({"ok": False, "error": "queue_full"}).encode("utf-8")

    return json.dumps({"ok": True, "queued": len(chunks)}).encode("utf-8")


# ---------------------------------------------------------------------------
# memorize ワーカーループ
# ---------------------------------------------------------------------------
def _memorize_worker_loop(
    memorize_queue: queue.Queue[dict[str, object] | None],
    embedder: Embedder,
    config: Config,
    memorize_alive: threading.Event,
) -> None:
    """memorize ジョブを 1 スレッドで逐次処理する

    キューから ``None`` を受け取った場合はシャットダウンしてループを抜ける
    例外は 1 ジョブごとに握り潰してループを継続する
    """
    from cc_mnemos.store import MemoryStore

    try:
        store = MemoryStore(config)
    except Exception:
        logger.exception("memorize ワーカースレッドの MemoryStore 初期化に失敗")
        return

    memorize_alive.set()
    try:
        while True:
            try:
                job = memorize_queue.get(timeout=_MEMORIZE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                _persist_memorize_request(job, embedder, store)
            except Exception:
                logger.exception("memorize ジョブの処理中にエラーが発生しました")
            finally:
                memorize_queue.task_done()
    finally:
        memorize_alive.clear()
        store.close()


def _persist_memorize_request(
    request: dict[str, object],
    embedder: Embedder,
    store: MemoryStore,
) -> None:
    """memorize リクエストを embedding + SQLite に永続化する"""
    from cc_mnemos._memorize_persistence import persist_chunks

    session_id_raw = request.get("session_id")
    session_id = str(session_id_raw) if session_id_raw else ""
    if not session_id:
        import uuid

        session_id = uuid.uuid4().hex

    project_name = str(request.get("project", ""))
    work_dir = str(request.get("work_dir", ""))
    chunks_raw = request.get("chunks", [])
    if not isinstance(chunks_raw, list):
        return

    chunk_records: list[dict[str, object]] = []
    for raw in chunks_raw:
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content", ""))
        if not content:
            continue
        tags_raw = raw.get("tags", [])
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
        chunk_records.append({
            "role_user": str(raw.get("role_user", "")),
            "role_assistant": str(raw.get("role_assistant", "")),
            "content": content,
            "tags": tags,
        })

    if not chunk_records:
        return

    persist_chunks(
        session_id=session_id,
        project_name=project_name,
        work_dir=work_dir,
        chunk_records=chunk_records,
        embedder=embedder,
        store=store,
    )


# ---------------------------------------------------------------------------
# クライアントハンドリング
# ---------------------------------------------------------------------------
def _handle_client(
    conn: socket.socket,
    embedder: Embedder,
    config: Config,
    *,
    memorize_queue: queue.Queue[dict[str, object] | None] | None = None,
    memorize_alive: threading.Event | None = None,
) -> None:
    """1 クライアントのリクエストを処理する

    type=ping → ready ping 応答 (検索処理なし)
    type=memorize → キューに積んで ack 返却 (実処理は memorize worker thread)
    その他 (type=search または type 未指定) → ハイブリッド検索を即時実行
    """
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

        req_type = request.get("type")
        if req_type == "ping":
            conn.sendall(json.dumps({"ok": True}).encode("utf-8"))
            return
        if req_type == "memorize":
            conn.sendall(_dispatch_memorize_request(request, memorize_queue, memorize_alive))
            return

        # 既存クライアントは type を付けずに query を送ってくる
        conn.sendall(_handle_search_request(request, embedder, config))
    except Exception:
        logger.exception("Worker request handling failed")
        conn.sendall(b"[]")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# デーモン起動
# ---------------------------------------------------------------------------
def run_daemon(port: int) -> None:
    """デーモンモードで起動し、ソケットでリクエストを待ち受ける

    重い model ロードに入る前に bind/listen を済ませることで、重複起動した
    ワーカーはアドレス重複でただちに終了できる
    """
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        # Windows では SO_REUSEADDR が既存リスナーへの重複 bind を許可してしまう
        # SO_EXCLUSIVEADDRUSE で排他的 bind にし、ゾンビワーカーの蓄積を防止
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)

    config = Config.load()
    embedder = Embedder(config)

    # memorize 用の専用キューとワーカースレッドを起動
    memorize_queue: queue.Queue[dict[str, object] | None] = queue.Queue(
        maxsize=MEMORIZE_QUEUE_MAXSIZE
    )
    memorize_alive = threading.Event()
    threading.Thread(
        target=_memorize_worker_loop,
        args=(memorize_queue, embedder, config, memorize_alive),
        daemon=True,
        name="cc-mnemos-memorize-worker",
    ).start()

    while True:
        conn, _ = srv.accept()
        threading.Thread(
            target=_handle_client,
            args=(conn, embedder, config),
            kwargs={
                "memorize_queue": memorize_queue,
                "memorize_alive": memorize_alive,
            },
            daemon=True,
        ).start()


def main() -> None:
    """エントリポイント"""
    if len(sys.argv) >= 3 and sys.argv[1] == "--daemon":
        port = int(sys.argv[2])
        run_daemon(port)
    elif len(sys.argv) >= 2:
        # レガシー: 単発実行モード (検索のみ)
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
