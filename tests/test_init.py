"""initコマンドのテスト"""

from __future__ import annotations

import json
from pathlib import Path

from cc_mnemos.cli import run_init


class TestInit:
    def test_creates_hooks_in_settings(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text("{}")
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("# Guidelines\n")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        settings = json.loads(settings_path.read_text())
        mcp_config = json.loads(mcp_config_path.read_text())
        assert "hooks" in settings
        assert "Stop" in settings["hooks"]
        assert "SessionStart" in settings["hooks"]
        assert "cc-mnemos" in mcp_config.get("mcpServers", {})

    def test_appends_to_claude_md(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text("{}")
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("# Guidelines\n")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        content = claude_md_path.read_text()
        assert "cc-mnemos" in content
        assert "search_memory" in content

    def test_preserves_existing_settings(self, tmp_path: Path) -> None:
        existing_settings = {"env": {"FOO": "bar"}}
        existing_mcp_config = {"mcpServers": {"other": {"command": "test"}}}
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(existing_settings))
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text(json.dumps(existing_mcp_config))
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        settings = json.loads(settings_path.read_text())
        mcp_config = json.loads(mcp_config_path.read_text())
        assert settings["env"]["FOO"] == "bar"
        assert "other" in mcp_config["mcpServers"]
        assert "cc-mnemos" in mcp_config["mcpServers"]

    def test_skips_claude_md_if_already_present(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text("{}")
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("# Guidelines\ncc-mnemos already here\n")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        content = claude_md_path.read_text()
        # search_memory は追記されていないはず
        assert content.count("search_memory") == 0

    def test_creates_settings_file_if_missing(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        mcp_config_path = tmp_path / ".claude.json"
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        assert settings_path.exists()
        assert mcp_config_path.exists()
        settings = json.loads(settings_path.read_text())
        mcp_config = json.loads(mcp_config_path.read_text())
        assert "hooks" in settings
        assert "cc-mnemos" in mcp_config["mcpServers"]

    def test_creates_claude_md_if_missing(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text("{}")
        claude_md_path = tmp_path / "CLAUDE.md"
        # ファイルを作成しない

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        assert claude_md_path.exists()
        content = claude_md_path.read_text()
        assert "cc-mnemos" in content
