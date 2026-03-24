"""CLIエントリポイントのテスト"""

from __future__ import annotations

import subprocess
import sys


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
