from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cc_mnemos.config import Config
from cc_mnemos.project import infer_project_name


class TestInferProjectName:
    def test_config_override(self) -> None:
        config = Config(projects={"C:/projects/resitoly": "resitoly"})
        result = infer_project_name("C:/projects/resitoly", config)
        assert result == "resitoly"

    def test_config_override_subdirectory(self) -> None:
        config = Config(projects={"C:/projects/resitoly": "resitoly"})
        result = infer_project_name("C:/projects/resitoly/frontend", config)
        assert result == "resitoly"

    def test_git_remote(self, tmp_path: Path) -> None:
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/user/my-repo.git"],
            cwd=tmp_path,
            capture_output=True,
        )
        config = Config()
        result = infer_project_name(str(tmp_path), config)
        assert result == "my-repo"

    def test_directory_basename_fallback(self) -> None:
        config = Config()
        with patch("cc_mnemos.project._get_git_remote", return_value=None):
            result = infer_project_name("C:/projects/my-project", config)
            assert result == "my-project"

    def test_unknown_fallback(self) -> None:
        config = Config()
        with patch("cc_mnemos.project._get_git_remote", return_value=None):
            result = infer_project_name("/", config)
            assert isinstance(result, str)
            assert len(result) > 0
