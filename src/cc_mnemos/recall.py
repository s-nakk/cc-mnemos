"""記憶注入モジュール — セッション開始時に過去の記憶をstdoutへ出力する"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
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

# recall 出力のトランケート上限（プロンプトキャッシュの破壊面積を抑えるため）
_USER_MSG_MAX_LENGTH = 150
_ASSISTANT_MSG_MAX_LENGTH = 250


def _truncate(text: str, limit: int) -> str:
    """長文を limit 文字まで縮める。改行は空白に潰す

    Args:
        text: 対象の文字列
        limit: 最大文字数

    Returns:
        limit を超える場合は末尾に "..." を付与した短縮版を返す
    """
    collapsed = text.strip().replace("\n", " ")
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def format_recall_output(
    project_name: str,
    recent_chunks: Sequence[Mapping[str, object]],
    cross_project_chunks: Sequence[Mapping[str, object]],
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
    # 日付プレフィックスは入れない（日次で変動しプロンプトキャッシュを壊すため）
    lines.append(f"## 直近の記憶（{project_name}）")
    for chunk in recent_chunks:
        user_msg = _truncate(str(chunk.get("role_user", "")), _USER_MSG_MAX_LENGTH)
        assistant_msg = _truncate(
            str(chunk.get("role_assistant", "")), _ASSISTANT_MSG_MAX_LENGTH
        )
        lines.append(f"- {user_msg} → {assistant_msg}")

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
        raw_assistant = str(chunk.get("role_assistant", ""))
        content_key = raw_assistant[:200]
        if content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        assistant_msg = _truncate(raw_assistant, _ASSISTANT_MSG_MAX_LENGTH)
        lines.append(f"- [{tags_label}] {assistant_msg}")

    lines.append("")
    lines.append(_REMINDER)

    return "\n".join(lines)


def run_recall(hook_input: Mapping[str, object], config: Config) -> None:
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


def _run_recall_impl(hook_input: Mapping[str, object], config: Config) -> None:
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

    # 5. MCP search_memory用のデーモンワーカーをバックグラウンドで起動
    import subprocess
    import threading
    from pathlib import Path

    def _start_worker() -> None:
        try:
            python = sys.executable
            worker = str(Path(__file__).parent / "_search_worker.py")
            subprocess.Popen(
                [python, worker, "--daemon", "19836"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_start_worker, daemon=True).start()

