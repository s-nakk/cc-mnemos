"""CLIエントリポイントのテスト"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from cc_mnemos.cli import _handle_memorize
from cc_mnemos.config import Config


class TestCLIHelp:
    def test_main_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cc_mnemos.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "memorize" in result.stdout
        assert "recall" in result.stdout
        assert "server" in result.stdout
        assert "init" in result.stdout
        assert "setup" in result.stdout
        assert "Codex" in result.stdout

    def test_init_help_mentions_target(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cc_mnemos.cli", "init", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--target" in result.stdout
        assert "codex" in result.stdout


# ---------------------------------------------------------------------------
# _handle_memorize — hook を壊さない例外耐性
# ---------------------------------------------------------------------------
class TestHandleMemorize:
    def test_swallows_invalid_stdin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """stdin が JSON として不正でも例外を上げず終了する"""

        def fake_load(_config_cls) -> Config:
            return Config(general={"data_dir": str(tmp_path)})

        monkeypatch.setattr(
            "cc_mnemos.config.Config.load", classmethod(fake_load)
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))

        # 例外を上げないこと
        _handle_memorize(object())  # type: ignore[arg-type]

    def test_swallows_run_memorize_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """run_memorize が例外を投げても hook プロセスは壊れない"""

        def fake_load(_config_cls) -> Config:
            return Config(general={"data_dir": str(tmp_path)})

        def boom(_hook_input: object, _config: object) -> None:
            raise RuntimeError("intentional failure")

        monkeypatch.setattr(
            "cc_mnemos.config.Config.load", classmethod(fake_load)
        )
        monkeypatch.setattr("cc_mnemos.memorize.run_memorize", boom)
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"session_id": "x"}'))

        _handle_memorize(object())  # type: ignore[arg-type]
