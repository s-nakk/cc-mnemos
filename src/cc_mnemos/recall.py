"""記憶注入モジュール — セッション開始時に過去の記憶をstdoutへ出力する"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from cc_mnemos import project
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

_REMINDER = (
    "---\n"
    "新しい画面やコンポーネントの実装時、スタイル判断時は\n"
    "search_memory で過去のデザイン判断やコーディングパターンを確認してください"
)


def format_recall_output(
    project_name: str,
    recent_chunks: list[dict[str, str | int]],
    cross_project_chunks: list[dict[str, str | int]],
) -> str:
    """記憶出力をフォーマットする

    Args:
        project_name: プロジェクト名
        recent_chunks: プロジェクト固有の直近チャンク
        cross_project_chunks: プロジェクト横断の知見チャンク

    Returns:
        フォーマット済みの記憶テキスト
    """
    lines: list[str] = []

    # プロジェクト別の直近記憶セクション
    lines.append(f"## 直近の記憶（{project_name}）")
    for chunk in recent_chunks:
        date = _extract_date(str(chunk.get("created_at", "")))
        user_msg = str(chunk.get("role_user", ""))
        assistant_msg = str(chunk.get("role_assistant", ""))
        lines.append(f"- [{date}] {user_msg} → {assistant_msg}")

    lines.append("")

    # 横断知見セクション（content重複排除）
    lines.append("## よく参照される知見")
    seen_contents: set[str] = set()
    for chunk in cross_project_chunks:
        tags_str = str(chunk.get("tags", "[]"))
        try:
            tags = json.loads(tags_str)
        except json.JSONDecodeError:
            tags = []
        tags_label = ", ".join(str(t) for t in tags)
        assistant_msg = str(chunk.get("role_assistant", ""))
        content_key = assistant_msg[:200]
        if content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        lines.append(f"- [{tags_label}] {assistant_msg}")

    lines.append("")
    lines.append(_REMINDER)

    return "\n".join(lines)


def _extract_date(iso_str: str) -> str:
    """ISO 8601文字列から日付部分 (YYYY-MM-DD) を抽出する

    Args:
        iso_str: ISO 8601形式の日時文字列

    Returns:
        日付文字列 (パース失敗時は元の文字列をそのまま返す)
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return iso_str


def run_recall(hook_input: dict[str, object], config: Config) -> None:
    """セッション開始時の記憶注入を実行する

    hookのstdinから情報を取得し、DBを検索して結果をstdoutへ出力する
    例外が発生した場合は何も出力しない (セッション開始をブロックしない)

    Args:
        hook_input: フック入力 (cwdなど)
        config: アプリケーション設定
    """
    try:
        _run_recall_impl(hook_input, config)
    except Exception:  # noqa: BLE001
        logger.debug(
            "recall で例外が発生しましたが、セッション開始をブロックしません",
            exc_info=True,
        )


def _run_recall_impl(hook_input: dict[str, object], config: Config) -> None:
    """記憶注入の内部実装

    Args:
        hook_input: フック入力
        config: アプリケーション設定
    """
    # 1. cwdを取得
    cwd = str(hook_input.get("cwd", "."))

    # 2. プロジェクト名を推定
    project_name = project.infer_project_name(cwd, config)

    # 3. MemoryStoreを開いてチャンクを取得
    store = MemoryStore(config)
    try:
        # プロジェクト固有の直近チャンク
        recent_chunks = store.get_recent_chunks(project=project_name, limit=5)

        # プロジェクト横断の知見（generalタグのみは除外、現プロジェクトも除外）
        cross_project_chunks = store.get_tagged_chunks(
            limit=5, exclude_project=project_name
        )
    finally:
        store.close()

    # 4. フォーマットして標準出力へ
    output = format_recall_output(project_name, recent_chunks, cross_project_chunks)
    sys.stdout.write(output)
