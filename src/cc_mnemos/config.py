"""設定モジュール — TOML / 環境変数 / デフォルト値を統合して提供する"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, Protocol, cast


class _TomlModule(Protocol):
    def load(self, fp: BinaryIO, /) -> dict[str, object]:
        ...


def _load_toml_module() -> _TomlModule:
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    return cast(_TomlModule, import_module(module_name))


_TOML_MODULE = _load_toml_module()


# ---------------------------------------------------------------------------
# TagRule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TagRule:
    """1つのタグに対するキーワードマッチングルール"""

    keywords: list[str]
    threshold: int
    prototype: str


# ---------------------------------------------------------------------------
# デフォルトタグルール (日英バイリンガル — 正規表現パターン)
# ---------------------------------------------------------------------------
DEFAULT_TAG_RULES: dict[str, TagRule] = {
    "ui-ux": TagRule(
        keywords=[
            r"デザイン|design",
            r"レイアウト|layout",
            r"カラー|色[味彩]|color",
            r"フォント|font",
            r"マージン|パディング|margin|padding",
            r"UI|UX|アクセシビリティ|accessibility",
            r"ボタン|モーダル|ナビ|サイドバー|カード",
            r"見た目|見栄え|余白|角丸|border-radius",
            r"css|CSS|スタイル|style",
            r"rounded|radius|corner",
            r"tailwind|Tailwind",
            r"token|theme|テーマ",
        ],
        threshold=2,
        prototype="UI design layout color font spacing component appearance CSS style",
    ),
    "coding-style": TagRule(
        keywords=[
            r"命名規則|naming",
            r"lint|eslint|prettier|ruff",
            r"インデント|フォーマット|format",
            r"コーディング規約|style\s*guide",
            r"型定義|type[Ss]cript|type\s*hint",
        ],
        threshold=1,
        prototype="coding style naming convention format lint type definition",
    ),
    "architecture": TagRule(
        keywords=[
            r"設計|architect",
            r"パターン|pattern",
            r"DB|データベース|database|schema",
            r"API設計|endpoint|REST|GraphQL",
            r"状態管理|state\s*manage",
        ],
        threshold=2,
        prototype="architecture design pattern database API state management",
    ),
    "debug": TagRule(
        keywords=[
            r"バグ|bug",
            r"エラー|error|exception",
            r"修正|fix",
            r"デバッグ|debug",
            r"スタックトレース|stack\s*trace|traceback",
        ],
        threshold=2,
        prototype="bug fix error debug stack trace exception",
    ),
    "config": TagRule(
        keywords=[
            r"環境変数|env",
            r"設定|config|settings",
            r"package\.json|pyproject|Cargo\.toml",
            r"ビルド|build|webpack|vite",
        ],
        threshold=2,
        prototype="configuration environment variable build settings",
    ),
    "decision": TagRule(
        keywords=[
            r"採用し[たて]|chose|chosen|selected",
            r"選[んび]|picked",
            r"決め[たて]|decided",
            r"やめ[たて]|abandoned|rejected",
            r"不採用|比較し[たて]|compared",
            r"理由[はとで]|because|reason",
            r"トレードオフ|trade-?off",
        ],
        threshold=2,
        prototype="technical decision comparison trade-off adoption reason",
    ),
}


# ---------------------------------------------------------------------------
# プラットフォーム依存パス解決
# ---------------------------------------------------------------------------
def get_data_dir() -> Path:
    """データディレクトリのデフォルトパスを返す

    優先順位:
    1. 環境変数 CC_MNEMOS_DATA_DIR
    2. Windows: %LOCALAPPDATA%/cc-mnemos
    3. Linux/macOS: $XDG_DATA_HOME/cc-mnemos (デフォルト ~/.local/share/cc-mnemos)
    """
    env_override = os.environ.get("CC_MNEMOS_DATA_DIR")
    if env_override:
        return Path(env_override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "")
        if base:
            return Path(base) / "cc-mnemos"
        return Path.home() / "AppData" / "Local" / "cc-mnemos"

    # Linux / macOS — XDG 準拠
    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    if xdg_data:
        return Path(xdg_data) / "cc-mnemos"
    return Path.home() / ".local" / "share" / "cc-mnemos"


def get_config_path() -> Path:
    """設定ファイルのデフォルトパスを返す

    優先順位:
    1. 環境変数 CC_MNEMOS_CONFIG
    2. Windows: %APPDATA%/cc-mnemos/config.toml
    3. Linux/macOS: $XDG_CONFIG_HOME/cc-mnemos/config.toml
       (デフォルト ~/.config/cc-mnemos/config.toml)
    """
    env_override = os.environ.get("CC_MNEMOS_CONFIG")
    if env_override:
        return Path(env_override)

    if sys.platform == "win32":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(base) / "cc-mnemos" / "config.toml"

    # Linux / macOS — XDG 準拠
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg_config:
        return Path(xdg_config) / "cc-mnemos" / "config.toml"
    return Path.home() / ".config" / "cc-mnemos" / "config.toml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config:
    """アプリケーション全体の設定を保持するクラス

    内部的に生の TOML セクション辞書 (_raw) を保持し、
    @property 経由でデフォルト値付きアクセスを提供する

    生成方法:
    - ``Config()`` — すべてデフォルト値
    - ``Config(embedding={...}, general={...})`` — TOML セクション辞書から構築
    - ``Config.from_file(path)`` — TOML ファイルから読み込み
    - ``Config.load()`` — デフォルトパスから読み込み (ファイルが無ければデフォルト)
    """

    def __init__(self, **raw_sections: dict[str, object]) -> None:
        """TOML セクション辞書からConfigを組み立てる

        Args:
            **raw_sections: ``embedding``, ``search``, ``chunking``,
                ``general``, ``tags``, ``projects`` などのセクション辞書
        """
        self._raw: dict[str, dict[str, object]] = {
            section_name: dict(section_values)
            for section_name, section_values in raw_sections.items()
        }

        # 環境変数オーバーライド: data_dir
        env_data_dir = os.environ.get("CC_MNEMOS_DATA_DIR")
        if env_data_dir:
            general = dict(self._section("general"))
            general["data_dir"] = env_data_dir
            self._raw["general"] = general

    def _section(self, name: str) -> dict[str, object]:
        return self._raw.get(name, {})

    def _get_str(self, section: str, key: str, default: str) -> str:
        return str(self._section(section).get(key, default))

    def _get_int(self, section: str, key: str, default: int) -> int:
        value = self._section(section).get(key, default)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str)):
            return int(value)
        return default

    def _get_float(self, section: str, key: str, default: float) -> float:
        value = self._section(section).get(key, default)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, str)):
            return float(value)
        return default

    @staticmethod
    def _as_str_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    # --- Embedding ---
    @property
    def embedding_model(self) -> str:
        """埋め込みモデル名を返す"""
        return self._get_str("embedding", "model", "cl-nagoya/ruri-v3-310m")

    @property
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        return self._get_int("embedding", "dimension", 768)

    @property
    def embedding_batch_size(self) -> int:
        """埋め込みバッチサイズを返す"""
        return self._get_int("embedding", "batch_size", 32)

    # --- Search ---
    @property
    def rrf_k(self) -> int:
        """RRF の k パラメータを返す"""
        return self._get_int("search", "rrf_k", 60)

    @property
    def time_decay_half_life_days(self) -> int:
        """時間減衰の半減期(日)を返す"""
        return self._get_int("search", "time_decay_half_life_days", 180)

    @property
    def fts_weight(self) -> float:
        """RRFにおけるFTSスコアの重みを返す"""
        return self._get_float("search", "fts_weight", 2.0)

    @property
    def vector_weight(self) -> float:
        """RRFにおけるベクトルスコアの重みを返す"""
        return self._get_float("search", "vector_weight", 0.75)

    @property
    def default_search_limit(self) -> int:
        """検索結果のデフォルト上限を返す"""
        return self._get_int("search", "default_search_limit", 10)

    # --- Chunking ---
    @property
    def max_chunk_chars(self) -> int:
        """チャンクの最大文字数を返す"""
        return self._get_int("chunking", "max_chunk_chars", 1500)

    @property
    def min_chunk_chars(self) -> int:
        """チャンクの最小文字数を返す"""
        return self._get_int("chunking", "min_chunk_chars", 20)

    # --- Maintenance ---
    @property
    def max_chunk_age_days(self) -> int:
        """チャンクの最大保持日数を返す"""
        return self._get_int("maintenance", "max_chunk_age_days", 365)

    @property
    def max_db_size_mb(self) -> int:
        """データベースの最大サイズ(MB)を返す"""
        return self._get_int("maintenance", "max_db_size_mb", 500)

    @property
    def vacuum_interval_days(self) -> int:
        """VACUUM の実行間隔(日)を返す"""
        return self._get_int("maintenance", "vacuum_interval_days", 30)

    # --- Misc ---
    @property
    def log_level(self) -> str:
        """ログレベルを返す"""
        return self._get_str("general", "log_level", "INFO")

    @property
    def project_mapping(self) -> dict[str, str]:
        """プロジェクトパスとプロジェクト名のマッピングを返す"""
        return {str(k): str(v) for k, v in self._section("projects").items()}

    # --- Paths ---
    @property
    def data_dir(self) -> Path:
        """データディレクトリのパスを返す"""
        general = self._section("general")
        if isinstance(general, dict) and "data_dir" in general:
            return Path(str(general["data_dir"]))
        return get_data_dir()

    @property
    def db_path(self) -> Path:
        """SQLite データベースファイルのパスを返す"""
        general = self._section("general")
        if isinstance(general, dict) and "db_path" in general:
            return Path(str(general["db_path"]))
        return self.data_dir / "memories.db"

    # --- Tags ---
    @property
    def tag_rules(self) -> dict[str, TagRule]:
        """タグルールの辞書を返す"""
        rules = dict(DEFAULT_TAG_RULES)
        for tag_name, tag_def in self._section("tags").items():
            if isinstance(tag_def, Mapping):
                rules[str(tag_name)] = TagRule(
                    keywords=self._as_str_list(tag_def.get("keywords", [])),
                    threshold=int(tag_def.get("threshold", 1)),
                    prototype=str(tag_def.get("prototype", "")),
                )
        return rules

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """TOML ファイルから Config を読み込む

        Args:
            path: 設定ファイルのパス

        Returns:
            読み込んだ設定を反映した Config インスタンス
        """
        with open(path, "rb") as f:
            loaded = _TOML_MODULE.load(f)
        sections = {
            str(section_name): dict(section_values)
            for section_name, section_values in loaded.items()
            if isinstance(section_values, Mapping)
        }
        return cls(**sections)

    @classmethod
    def load(cls) -> Config:
        """デフォルトパスから設定を読み込む

        設定ファイルが存在しない場合はデフォルト値で Config を返す

        Returns:
            Config インスタンス
        """
        config_path = get_config_path()
        if config_path.exists():
            return cls.from_file(config_path)
        return cls()
