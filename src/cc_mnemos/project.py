"""プロジェクト名推定"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cc_mnemos.config import Config


def _get_git_remote(cwd: str) -> str | None:
    """git remote origin URLからリポジトリ名を取得"""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        url = result.stdout.strip()
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def infer_project_name(cwd: str, config: Config) -> str:
    """cwdからプロジェクト名を推定

    推定の優先順位:
    1. Config の projects マッピングに前方一致するパスがあればその名前
    2. git remote origin URL からリポジトリ名を取得
    3. ディレクトリのbasename
    4. "unknown" (ルートディレクトリなどbasenameが空の場合)
    """
    cwd_normalized = cwd.replace("\\", "/")
    for path_prefix, name in config.project_mapping.items():
        prefix_normalized = path_prefix.replace("\\", "/")
        if cwd_normalized.startswith(prefix_normalized):
            return name

    git_name = _get_git_remote(cwd)
    if git_name:
        return git_name

    basename = Path(cwd).name
    return basename if basename else "unknown"
