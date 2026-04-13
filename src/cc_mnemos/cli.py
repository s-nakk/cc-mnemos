"""CLI エントリポイント — argparse ベースのサブコマンド群

エントリポイント: ``cc-mnemos = "cc_mnemos.cli:main"``
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

from cc_mnemos.batch_import import import_history

if TYPE_CHECKING:
    from cc_mnemos.config import Config

logger = logging.getLogger(__name__)

InitTarget = Literal["auto", "claude", "codex", "all"]


class _TomlModule(Protocol):
    def loads(self, data: str, /) -> dict[str, object]:
        ...


def _load_toml_module() -> _TomlModule:
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    return cast(_TomlModule, import_module(module_name))


_TOML_MODULE = _load_toml_module()


HookCommand = TypedDict(
    "HookCommand",
    {"type": str, "command": str, "timeout": int, "async": bool},
    total=False,
)


class HookEntry(TypedDict):
    matcher: str
    hooks: list[HookCommand]

# ---------------------------------------------------------------------------
# CLAUDE.md に追記するテキスト
# ---------------------------------------------------------------------------
_CLAUDE_MD_SECTION = """
## 記憶検索ルール（cc-mnemos）

以下の場面では `search_memory` MCPツールを呼び出すこと:
1. **UI/UX実装時**: 新しい画面・コンポーネント作成前に、過去のデザイン判断を検索
2. **スタイル判断時**: コーディングスタイルや命名規則に迷った場合
3. **技術選定時**: ライブラリ・パターンの選定前に、過去の採用理由を確認
4. **過去参照の発言時**: 「前に」「以前」「覚えてる」「before」「previously」
5. **プロジェクト横断時**: 別プロジェクトで得た知見が活きそうな場面
"""

_CODEX_AGENTS_SECTION = """

## 記憶検索ルール（cc-mnemos）

以下の場面では `search_memory` MCPツールを呼び出すこと:
1. **UI/UX実装時**: 新しい画面・コンポーネント作成前に、過去のデザイン判断を検索
2. **スタイル判断時**: コーディングスタイルや命名規則に迷った場合
3. **技術選定時**: ライブラリ・パターンの選定前に、過去の採用理由を確認
4. **過去参照の発言時**: 「前に」「以前」「覚えてる」「before」「previously」
5. **プロジェクト横断時**: 別プロジェクトで得た知見が活きそうな場面
"""


# ---------------------------------------------------------------------------
# settings.json に追加する設定
# ---------------------------------------------------------------------------
_HOOKS_CONFIG: dict[str, list[HookEntry]] = {
    "Stop": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cc-mnemos memorize",
                    "timeout": 30,
                    "async": True,
                },
            ],
        },
    ],
    "SessionStart": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cc-mnemos recall",
                    "timeout": 10,
                },
            ],
        },
    ],
    "UserPromptSubmit": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "cc-mnemos prompt-inject",
                    "timeout": 5,
                },
            ],
        },
    ],
}

_MCP_SERVER_CONFIG: dict[str, object] = {
    "type": "stdio",
    "command": "cc-mnemos",
    "args": ["server"],
    "env": {"PYTHONUNBUFFERED": "1"},
}


# ---------------------------------------------------------------------------
# init コマンド
# ---------------------------------------------------------------------------
def run_init(
    settings_path: Path | None = None,
    claude_md_path: Path | None = None,
    mcp_config_path: Path | None = None,
    *,
    target: InitTarget = "auto",
    home_dir: Path | None = None,
    codex_dir: Path | None = None,
    import_history_enabled: bool = False,
    config: Config | None = None,
    device: str | None = None,
) -> None:
    """対象エージェント向けに hooks / MCP / ルール設定を登録する

    Args:
        settings_path: Claude Code settings.json のパス
            (None の場合は ~/.claude/settings.json)
        claude_md_path: CLAUDE.md のパス (None の場合は ~/.claude/CLAUDE.md)
        mcp_config_path: Claude Code user config のパス
            (None の場合は ~/.claude.json)
    """
    resolved_home = home_dir if home_dir is not None else Path.home()
    selected_targets = _resolve_init_targets(target=target, home_dir=resolved_home)

    completed_targets: list[str] = []
    if "claude" in selected_targets:
        resolved_settings = settings_path or resolved_home / ".claude" / "settings.json"
        resolved_claude_md = claude_md_path or resolved_home / ".claude" / "CLAUDE.md"
        resolved_mcp_config = mcp_config_path or resolved_home / ".claude.json"
        _update_settings(resolved_settings)
        _update_mcp_config(resolved_mcp_config)
        _update_claude_md(resolved_claude_md)
        completed_targets.append("claude")

    if "codex" in selected_targets:
        resolved_codex_dir = codex_dir or resolved_home / ".codex"
        _update_codex_config(resolved_codex_dir / "config.toml")
        _update_codex_agents(resolved_codex_dir / "AGENTS.md")
        completed_targets.append("codex")

    if import_history_enabled:
        from cc_mnemos.config import Config

        cfg = config if config is not None else Config.load()
        for selected_target in completed_targets:
            import_history(cfg, device=device, agent=selected_target, verbose=True)

    joined = ", ".join(completed_targets)
    print(f"init 完了: {joined} を更新しました")


def _resolve_init_targets(target: InitTarget, home_dir: Path) -> list[str]:
    if target == "claude":
        return ["claude"]
    if target == "codex":
        return ["codex"]
    if target == "all":
        return ["claude", "codex"]

    detected_targets: list[str] = []
    if (home_dir / ".claude").exists():
        detected_targets.append("claude")
    if (home_dir / ".codex").exists():
        detected_targets.append("codex")
    if not detected_targets:
        raise ValueError("Claude/Codex の設定ディレクトリが見つかりません")
    return detected_targets


def _resolve_command_path() -> str:
    """cc-mnemos コマンドのフルパスを解決する

    .venv 内のexeが存在すればフルパスを返し、PATHに依存しない実行を保証する
    """
    import shutil

    # 1. shutil.which で探す（PATHに含まれている場合）
    found = shutil.which("cc-mnemos")
    if found:
        return found

    # 2. 現在のPythonと同じ環境の Scripts/bin を探す
    scripts_dir = Path(sys.executable).parent
    for name in ("cc-mnemos.exe", "cc-mnemos"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    # 3. フォールバック: そのまま返す（ユーザーがPATHに追加する前提）
    return "cc-mnemos"


def _normalize_path(path: str) -> str:
    """Windowsパスをフォワードスラッシュに正規化する

    Claude Codeのhookはシェル経由で実行されるため、
    バックスラッシュだと問題を起こす場合がある
    """
    return path.replace("\\", "/")


def _update_settings(settings_path: Path) -> None:
    """settings.json にフック設定をマージする"""
    # 親ディレクトリ作成
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # cc-mnemos のフルパスを解決（フォワードスラッシュに正規化）
    cmd_path = _normalize_path(_resolve_command_path())

    # 既存設定の読み込み
    if settings_path.exists():
        raw = settings_path.read_text(encoding="utf-8")
        settings: dict[str, object] = json.loads(raw) if raw.strip() else {}
    else:
        settings = {}

    # hooks のマージ (既存キーを上書きしない)
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks

    # コマンドパスを実際のパスに置換したhook設定を生成
    import copy
    resolved_hooks = copy.deepcopy(_HOOKS_CONFIG)
    for entries in resolved_hooks.values():
        for entry in entries:
            for hook in entry["hooks"]:
                cmd = hook.get("command", "")
                if isinstance(cmd, str):
                    hook["command"] = cmd.replace("cc-mnemos", cmd_path, 1)

    for event_name, event_entries in resolved_hooks.items():
        existing_entries = hooks.get(event_name, [])
        if not isinstance(existing_entries, list):
            existing_entries = []
        hooks[event_name] = existing_entries

        for event_entry in event_entries:
            matcher = event_entry.get("matcher", "")
            target_entry: dict[str, object] | None = None
            for existing_entry in existing_entries:
                if (
                    isinstance(existing_entry, dict)
                    and existing_entry.get("matcher", "") == matcher
                    and isinstance(existing_entry.get("hooks"), list)
                ):
                    target_entry = existing_entry
                    break
            if target_entry is None:
                target_entry = {"matcher": matcher, "hooks": []}
                existing_entries.append(target_entry)

            target_hooks = target_entry.get("hooks", [])
            if not isinstance(target_hooks, list):
                target_hooks = []
                target_entry["hooks"] = target_hooks

            for desired_hook in event_entry["hooks"]:
                desired_command = desired_hook.get("command", "")
                if not isinstance(desired_command, str):
                    continue

                hook_replaced = False
                for index, existing_hook in enumerate(target_hooks):
                    if not isinstance(existing_hook, dict):
                        continue
                    existing_command = existing_hook.get("command", "")
                    if (
                        isinstance(existing_command, str)
                        and "cc-mnemos" in existing_command
                        and existing_command.split()[-1] == desired_command.split()[-1]
                    ):
                        target_hooks[index] = desired_hook
                        hook_replaced = True
                        break

                if not hook_replaced:
                    target_hooks.append(desired_hook)

    # 書き戻し
    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _update_mcp_config(mcp_config_path: Path) -> None:
    """Claude Code user config に MCP サーバー設定をマージする"""
    mcp_config_path.parent.mkdir(parents=True, exist_ok=True)

    if mcp_config_path.exists():
        raw = mcp_config_path.read_text(encoding="utf-8")
        config: dict[str, object] = json.loads(raw) if raw.strip() else {}
    else:
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        config["mcpServers"] = mcp_servers

    import copy

    resolved_mcp_config = copy.deepcopy(_MCP_SERVER_CONFIG)
    resolved_mcp_config["command"] = _normalize_path(_resolve_command_path())
    mcp_servers["cc-mnemos"] = resolved_mcp_config

    mcp_config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _update_claude_md(claude_md_path: Path) -> None:
    """CLAUDE.md に記憶検索ルールを追記する (重複チェック付き)"""
    claude_md_path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if claude_md_path.exists():
        existing = claude_md_path.read_text(encoding="utf-8")

    # 既に cc-mnemos の記述がある場合はスキップ
    if "cc-mnemos" in existing:
        return

    with open(claude_md_path, "a", encoding="utf-8") as f:
        f.write(_CLAUDE_MD_SECTION)


def _update_codex_config(config_path: Path) -> None:
    """Codex config.toml に MCP サーバー設定を追記する"""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    if existing.strip():
        _TOML_MODULE.loads(existing)
    if "[mcp_servers.cc-mnemos]" in existing:
        return

    command = _normalize_path(_resolve_command_path())
    block = (
        "\n[mcp_servers.cc-mnemos]\n"
        f'command = "{command}"\n'
        'args = ["server"]\n'
    )
    new_content = f"{existing.rstrip()}{block}" if existing.strip() else block.lstrip("\n")
    config_path.write_text(new_content, encoding="utf-8")


def _update_codex_agents(agents_path: Path) -> None:
    """Codex 用 AGENTS.md に記憶検索ルールを追記する"""
    agents_path.parent.mkdir(parents=True, exist_ok=True)

    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""
    if "cc-mnemos" in existing:
        return

    new_content = f"{existing.rstrip()}{_CODEX_AGENTS_SECTION}" if existing.strip() else _CODEX_AGENTS_SECTION.lstrip("\n")
    agents_path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# setup コマンド
# ---------------------------------------------------------------------------
def run_setup(config: None = None) -> None:
    """モデルダウンロード + DB 初期化を実行する

    Args:
        config: アプリケーション設定 (None の場合はデフォルト設定を使用)
    """
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder
    from cc_mnemos.store import MemoryStore

    cfg = config if config is not None else Config.load()

    # 1. データディレクトリ作成
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"データディレクトリ: {cfg.data_dir}")

    # 2. DB 初期化
    store = MemoryStore(cfg)
    store.close()
    print(f"DB 初期化完了: {cfg.db_path}")

    # 3. Embedding モデルのダウンロード
    print(f"モデルをダウンロード中: {cfg.embedding_model} ...")
    Embedder(cfg)
    print("モデルのダウンロード完了")

    print("setup 完了")


# ---------------------------------------------------------------------------
# サブコマンドハンドラ
# ---------------------------------------------------------------------------
def _handle_memorize(args: argparse.Namespace) -> None:
    """memorize サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.memorize import run_memorize

    hook_input: dict[str, object] = json.load(sys.stdin)
    cfg = Config.load()
    run_memorize(hook_input, cfg)


def _handle_recall(args: argparse.Namespace) -> None:
    """recall サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.recall import run_recall

    hook_input: dict[str, object] = json.load(sys.stdin)
    cfg = Config.load()
    run_recall(hook_input, cfg)


def _handle_prompt_inject(args: argparse.Namespace) -> None:
    """prompt-inject サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.prompt_inject import run_prompt_inject

    hook_input: dict[str, object] = json.load(sys.stdin)
    cfg = Config.load()
    run_prompt_inject(hook_input, cfg)


def _handle_server(args: argparse.Namespace) -> None:
    """server サブコマンドのハンドラ"""
    from cc_mnemos.server import run_server

    run_server()


def _handle_init(args: argparse.Namespace) -> None:
    """init サブコマンドのハンドラ"""
    run_init(
        target=args.target,
        import_history_enabled=args.import_history,
        device="cpu" if args.cpu else None,
    )


def _handle_rebuild(args: argparse.Namespace) -> None:
    """rebuild サブコマンドのハンドラ"""
    from cc_mnemos.batch_import import import_history
    from cc_mnemos.config import Config
    from cc_mnemos.store import MemoryStore

    cfg = Config.load()
    store = MemoryStore(cfg)

    stats = store.get_stats()
    total = stats["total_chunks"]
    sessions = stats["total_sessions"]
    print(f"既存データ: {sessions} sessions, {total} chunks")

    if not args.yes:
        answer = input("DBをクリアして全セッションを再インポートしますか？ [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("キャンセルしました")
            store.close()
            return

    if store._use_sqlite_vec:
        store.conn.execute("DELETE FROM vec_chunks")
    store.conn.execute("DELETE FROM chunk_vec_map")
    store.conn.execute("DELETE FROM chunks")
    store.conn.execute("DELETE FROM sessions")
    store.conn.commit()
    store.close()
    print("DBをクリアしました")

    device = "cpu" if args.cpu else None
    import_history(cfg, device=device)


def _handle_deduplicate(args: argparse.Namespace) -> None:
    """deduplicate サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.store import MemoryStore

    cfg = Config.load()
    store = MemoryStore(cfg)
    try:
        stats_before = store.get_stats()
        print(f"修正前: {stats_before['total_chunks']} chunks, "
              f"{stats_before['total_sessions']} sessions")

        # 1. プロジェクト名の正規化
        renames = store.normalize_project_names()
        if renames:
            print(f"\nプロジェクト名を統一しました:")
            for old, new in renames.items():
                print(f"  {old} → {new}")
        else:
            print("\nプロジェクト名の不整合はありません")

        # 2. 重複チャンクの削除
        removed = store.deduplicate_chunks()
        print(f"\n重複チャンクを {removed} 件削除しました")

        stats_after = store.get_stats()
        print(f"\n修正後: {stats_after['total_chunks']} chunks, "
              f"{stats_after['total_sessions']} sessions")

        total_before = stats_before["total_chunks"]
        total_after = stats_after["total_chunks"]
        reduction = total_before - total_after
        if total_before > 0:
            pct = reduction / total_before * 100
            print(f"削減率: {pct:.1f}% ({reduction} 件削除)")
    finally:
        store.close()


def _handle_setup(args: argparse.Namespace) -> None:
    """setup サブコマンドのハンドラ"""
    run_setup()


def _handle_stats(args: argparse.Namespace) -> None:
    """stats サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.store import MemoryStore

    cfg = Config.load()
    store = MemoryStore(cfg)
    try:
        stats = store.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    finally:
        store.close()


def _handle_search(args: argparse.Namespace) -> None:
    """search サブコマンドのハンドラ"""
    from cc_mnemos.config import Config
    from cc_mnemos.embedder import Embedder
    from cc_mnemos.store import MemoryStore

    cfg = Config.load()
    embedder = Embedder(cfg)
    store = MemoryStore(cfg)
    try:
        query_embedding = embedder.encode_query(args.query)
        results = store.hybrid_search(
            query_text=args.query,
            query_embedding=query_embedding,
            limit=args.limit,
        )

        if not results:
            print("結果なし")
            return

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            tags_str = str(result.get("tags", "[]"))
            content = str(result.get("content", ""))
            created_at = str(result.get("created_at", ""))
            # 長いコンテンツは先頭200文字に切り詰め
            preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"[{i}] score={score:.4f} tags={tags_str} ({created_at})")
            print(f"    {preview}")
            print()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI のメインエントリポイント"""
    parser = argparse.ArgumentParser(
        prog="cc-mnemos",
        description="Long-term memory system for Claude Code and Codex",
    )
    subparsers = parser.add_subparsers(dest="command")

    # memorize
    sub_memorize = subparsers.add_parser(
        "memorize",
        help="会話ログを記憶に保存する (hook stdin から JSON を読み取り)",
    )
    sub_memorize.set_defaults(handler=_handle_memorize)

    # recall
    sub_recall = subparsers.add_parser(
        "recall",
        help="セッション開始時に過去の記憶を注入する (hook stdin から JSON を読み取り)",
    )
    sub_recall.set_defaults(handler=_handle_recall)

    # prompt-inject
    sub_prompt_inject = subparsers.add_parser(
        "prompt-inject",
        help="ユーザー発話に関連する記憶を検索・注入する (UserPromptSubmit hook)",
    )
    sub_prompt_inject.set_defaults(handler=_handle_prompt_inject)

    # server
    sub_server = subparsers.add_parser(
        "server",
        help="MCP サーバーを起動する",
    )
    sub_server.set_defaults(handler=_handle_server)

    # init
    sub_init = subparsers.add_parser(
        "init",
        help="Claude Code / Codex 向けの hooks / MCP / ルール設定を登録する",
    )
    sub_init.add_argument(
        "--target",
        choices=("auto", "claude", "codex", "all"),
        default="auto",
        help="設定対象のエージェント (デフォルト: auto)",
    )
    sub_init.add_argument(
        "--import-history",
        action="store_true",
        help="検出した対象エージェントの既存セッション履歴を一括インポートする",
    )
    sub_init.add_argument(
        "--cpu",
        action="store_true",
        help="GPU を使用せず CPU でEmbedding生成する",
    )
    sub_init.set_defaults(handler=_handle_init)

    # setup
    sub_setup = subparsers.add_parser(
        "setup",
        help="モデルダウンロード + DB 初期化を実行する",
    )
    sub_setup.set_defaults(handler=_handle_setup)

    # rebuild
    sub_rebuild = subparsers.add_parser(
        "rebuild",
        help="DB をクリアして全セッションを再インポートする",
    )
    sub_rebuild.add_argument(
        "-y", "--yes",
        action="store_true",
        help="確認プロンプトをスキップする",
    )
    sub_rebuild.add_argument(
        "--cpu",
        action="store_true",
        help="GPU を使用せず CPU でEmbedding生成する",
    )
    sub_rebuild.set_defaults(handler=_handle_rebuild)

    # deduplicate
    sub_deduplicate = subparsers.add_parser(
        "deduplicate",
        help="重複チャンクの削除とプロジェクト名の正規化を実行する",
    )
    sub_deduplicate.set_defaults(handler=_handle_deduplicate)

    # stats
    sub_stats = subparsers.add_parser(
        "stats",
        help="記憶の統計情報を表示する",
    )
    sub_stats.set_defaults(handler=_handle_stats)

    # search
    sub_search = subparsers.add_parser(
        "search",
        help="CLI からデバッグ用検索を実行する",
    )
    sub_search.add_argument("query", help="検索クエリ")
    sub_search.add_argument(
        "--limit",
        type=int,
        default=10,
        help="結果の最大件数 (デフォルト: 10)",
    )
    sub_search.set_defaults(handler=_handle_search)

    args = parser.parse_args()

    if not hasattr(args, "handler"):
        parser.print_help()
        sys.exit(1)

    try:
        args.handler(args)
    except Exception:
        logger.exception("コマンド実行中にエラーが発生しました")
        # エラーログをファイルにも出力
        logging.basicConfig(
            filename="error.log",
            level=logging.ERROR,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logger.error("コマンド '%s' が失敗しました", args.command, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
