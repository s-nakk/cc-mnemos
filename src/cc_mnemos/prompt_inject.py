"""プロンプト時記憶注入 — UserPromptSubmit hookで関連記憶を自動検索・注入する

FTS-only検索（embeddingモデル不要）で高速に動作し、
関連する記憶があればstdoutに出力してコンテキストに注入する
"""

from __future__ import annotations

import json
import logging
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


def _run_prompt_inject_impl(hook_input: dict[str, object], config: Config) -> None:
    """記憶注入の内部実装"""
    user_prompt = str(hook_input.get("user_prompt", ""))
    if len(user_prompt) < _MIN_QUERY_LEN:
        return

    cwd = str(hook_input.get("cwd", "."))
    project_name = project.infer_project_name(cwd, config)

    store = MemoryStore(config)
    try:
        results = store.fts_search(user_prompt, limit=3)

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
    finally:
        store.close()

    if not results:
        return

    # 有意な結果のみフィルタ
    meaningful = [
        r for r in results
        if len(str(r.get("content", ""))) >= _MIN_CONTENT_LEN
    ]
    if not meaningful:
        return

    # プロジェクトフィルタ: 同一プロジェクトの結果を優先表示
    output = _format_injection(meaningful, project_name)
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
) -> str:
    """検索結果を注入用テキストにフォーマットする

    Args:
        results: FTS検索結果
        project_name: 現在のプロジェクト名

    Returns:
        注入用テキスト
    """
    lines: list[str] = []
    lines.append(f"[cc-mnemos] 関連する過去の記憶 ({project_name}):")

    for r in results:
        tags_str = str(r.get("tags", "[]"))
        try:
            tags = json.loads(tags_str)
        except json.JSONDecodeError:
            tags = []
        tags_label = ", ".join(str(t) for t in tags)
        content = str(r.get("content", ""))
        # 長すぎる場合は先頭300文字に切り詰め
        preview = content[:300] + ("..." if len(content) > 300 else "")
        lines.append(f"- [{tags_label}] {preview}")

    return "\n".join(lines)
