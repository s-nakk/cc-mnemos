import os
from pathlib import Path
from unittest.mock import patch

from cc_mnemos.config import Config, get_data_dir


class TestGetDataDir:
    def test_default_linux(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch.dict(os.environ, {}, clear=True),
            patch("pathlib.Path.home", return_value=Path("/home/testuser")),
        ):
            result = get_data_dir()
            assert "cc-mnemos" in str(result)
            assert ".local/share/cc-mnemos" in str(result).replace("\\", "/")

    def test_default_windows(self) -> None:
        with patch("sys.platform", "win32"), patch.dict(
            os.environ, {"LOCALAPPDATA": "C:\\Users\\test\\AppData\\Local"}, clear=True
        ):
            result = get_data_dir()
            assert "cc-mnemos" in str(result)

    def test_override_via_config(self) -> None:
        config = Config(general={"data_dir": "/custom/path"})
        assert config.data_dir == Path("/custom/path")


class TestConfig:
    def test_defaults(self) -> None:
        config = Config()
        assert config.embedding_model == "cl-nagoya/ruri-v3-310m"
        assert config.embedding_dimension == 768
        assert config.rrf_k == 60
        assert config.time_decay_half_life_days == 180
        assert config.fts_weight == 2.0
        assert config.vector_weight == 0.75
        assert config.max_chunk_chars == 1500
        assert config.min_chunk_chars == 20
        assert config.default_search_limit == 10

    def test_from_toml(self, tmp_path: Path) -> None:
        toml_content = '[embedding]\nmodel = "custom-model"\ndimension = 384\n'
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        config = Config.from_file(config_file)
        assert config.embedding_model == "custom-model"
        assert config.embedding_dimension == 384

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"CC_MNEMOS_DATA_DIR": "/env/path"}):
            config = Config()
            assert config.data_dir == Path("/env/path")

    def test_tag_rules_default(self) -> None:
        config = Config()
        assert "ui-ux" in config.tag_rules
        assert "coding-style" in config.tag_rules

    def test_custom_tag_from_toml(self, tmp_path: Path) -> None:
        toml_content = (
            '[tags.my-tag]\nkeywords = ["foo", "bar"]\n'
            'threshold = 2\nprototype = "test"\n'
        )
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        config = Config.from_file(config_file)
        assert "my-tag" in config.tag_rules

    def test_project_mapping(self, tmp_path: Path) -> None:
        toml_content = '[projects]\n"C:/projects/resitoly" = "resitoly"\n'
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        config = Config.from_file(config_file)
        assert config.project_mapping["C:/projects/resitoly"] == "resitoly"
