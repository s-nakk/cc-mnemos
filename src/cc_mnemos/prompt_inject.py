"""プロンプト時記憶注入 — UserPromptSubmit hookで関連記憶を自動検索・注入する

search workerデーモンが利用可能な場合はhybrid search（FTS + ベクトル検索）を使用し、
利用不可の場合はFTS-only検索にフォールバックする
"""

from __future__ import annotations

import json
import logging
import socket
import sys
from typing import TYPE_CHECKING

from cc_mnemos import project
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

# 結果が少なすぎる場合は注入しない（ノイズ防止）
_MIN_CONTENT_LEN = 30

# 検索対象から除外する短いプロンプト
_MIN_QUERY_LEN = 5

# search workerデーモンのポート
_WORKER_PORT = 19836

# workerへの接続タイムアウト（秒） — hookの5秒制限内に収める
_WORKER_TIMEOUT = 3.0

# LIKE検索フォールバック用のストップワード
_STOP_WORDS = frozenset({
    "の", "は", "を", "に", "が", "で", "と", "も", "か", "て", "し",
    "です", "ます", "する", "した", "して", "ない", "ある", "いる",
    "どう", "どの", "この", "その", "それ", "これ", "ください",
    "教えて", "方法", "方", "やり方", "について",
})


def run_prompt_inject(hook_input: dict[str, object], config: Config) -> None:
    """ユーザープロンプトに関連する記憶をFTS検索して注入する

    例外が発生した場合は何も出力しない（プロンプト送信をブロックしない）

    Args:
        hook_input: フック入力 (user_prompt, cwdなど)
        config: アプリケーション設定
    """
    try:
        _run_prompt_inject_impl(hook_input, config)
    except Exception:  # noqa: BLE001
        logger.debug(
            "prompt_inject で例外が発生しましたが、プロンプト送信をブロックしません",
            exc_info=True,
        )


def _query_worker(
    query: str,
    project_name: str | None = None,
    limit: int = 3,
) -> list[dict[str, str | int | float]] | None:
    """search workerデーモンにhybrid searchリクエストを送信する

    workerが利用不可の場合はNoneを返す

    Args:
        query: 検索クエリ
        project_name: プロジェクトフィルタ
        limit: 結果の最大件数

    Returns:
        検索結果のリスト、またはworker不在時はNone
    """
    request: dict[str, str | int] = {"query": query, "limit": limit}
    if project_name:
        request["project"] = project_name

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_WORKER_TIMEOUT)
        sock.connect(("127.0.0.1", _WORKER_PORT))
        sock.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        sock.shutdown(socket.SHUT_WR)

        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk

        sock.close()
        results = json.loads(data.decode("utf-8"))
        if isinstance(results, list) and len(results) > 0:
            return results
        return None
    except (OSError, json.JSONDecodeError, ValueError):
        logger.debug("search workerデーモンに接続できませんでした", exc_info=True)
        return None


def _fts_fallback(
    store: MemoryStore,
    user_prompt: str,
) -> list[dict[str, str | int | float]]:
    """FTS検索 + LIKE検索フォールバック

    Args:
        store: MemoryStore インスタンス
        user_prompt: ユーザーの入力テキスト

    Returns:
        検索結果のリスト
    """
    results: list[dict[str, str | int | float]] = store.fts_search(
        user_prompt, limit=3
    )

    # FTSで見つからない場合、主要キーワードでLIKE検索フォールバック
    if not results:
        keywords = _extract_keywords(user_prompt)
        seen_ids: set[str] = set()
        for kw in keywords[:3]:
            rows = store.conn.execute(
                "SELECT * FROM chunks WHERE content LIKE ? LIMIT 3",
                (f"%{kw}%",),
            ).fetchall()
            for row in rows:
                d = dict(row)
                cid = str(d["id"])
                if cid not in seen_ids:
                    results.append(d)
                    seen_ids.add(cid)
            if len(results) >= 3:
                break

    return results


def _run_prompt_inject_impl(hook_input: dict[str, object], config: Config) -> None:
    """記憶注入の内部実装"""
    user_prompt = str(hook_input.get("user_prompt", ""))
    if len(user_prompt) < _MIN_QUERY_LEN:
        return

    cwd = str(hook_input.get("cwd", "."))
    project_name = project.infer_project_name(cwd, config)

    # 1. search workerデーモン経由でhybrid search（FTS + ベクトル）を試行
    search_method = "hybrid"
    results = _query_worker(user_prompt, project_name, limit=3)

    # 2. worker不在の場合はFTS-only検索にフォールバック
    if results is None:
        search_method = "fts"
        store = MemoryStore(config)
        try:
            results = _fts_fallback(store, user_prompt)
        finally:
            store.close()

    if not results:
        return

    # 有意な結果のみフィルタ + content重複排除
    seen_contents: set[str] = set()
    meaningful: list[dict[str, str | int | float]] = []
    for r in results:
        content = str(r.get("content", ""))
        if len(content) < _MIN_CONTENT_LEN:
            continue
        # 先頭200文字で重複判定（同一会話の類似チャンクを排除）
        content_key = content[:200]
        if content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        meaningful.append(r)
    if not meaningful:
        return

    output = _format_injection(meaningful, project_name, search_method)
    sys.stdout.write(output)


def _extract_keywords(text: str) -> list[str]:
    """テキストからLIKE検索用のキーワードを抽出する

    英数字トークンと、4文字以上の日本語部分文字列を抽出する

    Args:
        text: 入力テキスト

    Returns:
        検索用キーワードのリスト（長い順）
    """
    import re

    # 英数字+ハイフン/アンダースコアのトークン
    ascii_tokens = re.findall(r"[a-zA-Z0-9_-]{3,}", text)
    # 日本語文字列（ひらがな・カタカナ・漢字の連続）から4文字以上を抽出
    ja_tokens = re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{4,}", text)

    keywords = [t for t in ascii_tokens + ja_tokens if t not in _STOP_WORDS]
    # 長いキーワードを優先（より具体的）
    keywords.sort(key=len, reverse=True)
    return keywords


def _format_injection(
    results: list[dict[str, str | int | float]],
    project_name: str,
    search_method: str = "fts",
) -> str:
    """検索結果を注入用テキストにフォーマットする

    Args:
        results: 検索結果
        project_name: 現在のプロジェクト名
        search_method: 使用した検索方法 ("hybrid" or "fts")

    Returns:
        注入用テキスト
    """
    method_label = "hybrid" if search_method == "hybrid" else "FTS"
    lines: list[str] = []
    lines.append(
        f"[cc-mnemos:{method_label}] "
        f"過去の関連記憶が {len(results)} 件見つかりました ({project_name})"
    )
    lines.append(
        "回答に活用した場合は「過去の記憶によると〜」等で参照元を明示してください"
    )
    lines.append("")

    for r in results:
        tags_str = str(r.get("tags", "[]"))
        try:
            tags = json.loads(tags_str)
        except json.JSONDecodeError:
            tags = []
        tags_label = ", ".join(str(t) for t in tags) if tags else "general"

        # role_user/role_assistantがあればQ&A形式で表示
        role_user = str(r.get("role_user", ""))
        role_assistant = str(r.get("role_assistant", ""))
        if role_user and role_assistant:
            user_preview = role_user[:100] + ("..." if len(role_user) > 100 else "")
            asst_preview = (
                role_assistant[:200]
                + ("..." if len(role_assistant) > 200 else "")
            )
            lines.append(f"- [{tags_label}] Q: {user_preview}")
            lines.append(f"  A: {asst_preview}")
        else:
            content = str(r.get("content", ""))
            preview = content[:300] + ("..." if len(content) > 300 else "")
            lines.append(f"- [{tags_label}] {preview}")

    return "\n".join(lines)
