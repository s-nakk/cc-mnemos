"""initコマンドのテスト"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cc_mnemos.cli import _resolve_command_path, run_init


class TestInit:
    def test_resolve_command_path_prefers_current_python_environment(self, tmp_path: Path) -> None:
        scripts_dir = tmp_path / ".venv" / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "python.exe").write_text("", encoding="utf-8")
        (scripts_dir / "cc-mnemos.exe").write_text("", encoding="utf-8")

        with (
            patch("cc_mnemos.cli.sys.executable", str(scripts_dir / "python.exe")),
            patch("shutil.which", return_value="C:\\legacy\\cc-mnemos.exe"),
        ):
            resolved = _resolve_command_path()

        assert resolved == str(scripts_dir / "cc-mnemos.exe")

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

    def test_merges_existing_hooks_in_same_event(self, tmp_path: Path) -> None:
        existing_settings = {
            "hooks": {
                "Stop": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "custom-stop-command",
                                "timeout": 15,
                            }
                        ],
                    }
                ]
            }
        }
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(existing_settings))
        mcp_config_path = tmp_path / ".claude.json"
        mcp_config_path.write_text("{}")
        claude_md_path = tmp_path / "CLAUDE.md"
        claude_md_path.write_text("")

        run_init(
            settings_path=settings_path,
            claude_md_path=claude_md_path,
            mcp_config_path=mcp_config_path,
        )

        settings = json.loads(settings_path.read_text())
        stop_hooks = settings["hooks"]["Stop"][0]["hooks"]
        commands = [hook["command"] for hook in stop_hooks]
        assert "custom-stop-command" in commands
        assert any("cc-mnemos" in command and command.endswith(" memorize") for command in commands)

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

    def test_target_codex_updates_codex_files(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            '[mcp_servers.context7]\nurl = "https://example.com"\n',
            encoding="utf-8",
        )
        agents_path = codex_dir / "AGENTS.md"
        agents_path.write_text("# Rules\n", encoding="utf-8")

        run_init(target="codex", codex_dir=codex_dir)

        config_text = config_path.read_text(encoding="utf-8")
        agents_text = agents_path.read_text(encoding="utf-8")
        assert '[mcp_servers.cc-mnemos]' in config_text
        assert 'command = ' in config_text
        assert '[mcp_servers.context7]' in config_text
        assert "search_memory" in agents_text

    def test_target_auto_updates_detected_targets(self, tmp_path: Path) -> None:
        home_dir = tmp_path / "home"
        claude_dir = home_dir / ".claude"
        claude_dir.mkdir(parents=True)
        codex_dir = home_dir / ".codex"
        codex_dir.mkdir()
        (home_dir / ".claude.json").write_text("{}", encoding="utf-8")
        (codex_dir / "config.toml").write_text("", encoding="utf-8")

        run_init(home_dir=home_dir, target="auto")

        assert (claude_dir / "settings.json").exists()
        assert (home_dir / ".claude.json").exists()
        assert "cc-mnemos" in (codex_dir / "config.toml").read_text(encoding="utf-8")

    def test_codex_agents_update_is_idempotent(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        agents_path = codex_dir / "AGENTS.md"
        agents_path.write_text("# Rules\n", encoding="utf-8")

        run_init(target="codex", codex_dir=codex_dir)
        run_init(target="codex", codex_dir=codex_dir)

        content = agents_path.read_text(encoding="utf-8")
        assert content.count("cc-mnemos") == 1

    def test_target_codex_updates_existing_cc_mnemos_mcp_server(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    '[mcp_servers.context7]',
                    'url = "https://example.com"',
                    '',
                    '[mcp_servers.cc-mnemos]',
                    'command = "C:/legacy/cc-mnemos.exe"',
                    'args = ["server"]',
                    '',
                ]
            ),
            encoding="utf-8",
        )

        with patch(
            "cc_mnemos.cli._resolve_command_path",
            return_value="C:\\projects\\cc-mnemos\\.venv\\Scripts\\cc-mnemos.exe",
        ):
            run_init(target="codex", codex_dir=codex_dir)

        config_text = config_path.read_text(encoding="utf-8")
        assert '[mcp_servers.context7]' in config_text
        assert 'command = "C:/projects/cc-mnemos/.venv/Scripts/cc-mnemos.exe"' in config_text
        assert 'command = "C:/legacy/cc-mnemos.exe"' not in config_text

    def test_init_import_history_runs_selected_target(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "config.toml").write_text("", encoding="utf-8")

        with patch("cc_mnemos.cli.import_history") as import_history_mock:
            import_history_mock.return_value = {"imported": 0, "skipped": 0, "errors": 0}
            run_init(
                target="codex",
                codex_dir=codex_dir,
                import_history_enabled=True,
            )

        import_history_mock.assert_called_once()
        assert import_history_mock.call_args.kwargs["agent"] == "codex"
