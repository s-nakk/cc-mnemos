"""設定モジュール — TOML / 環境変数 / デフォルト値を統合して提供する"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10


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
# デフォルトタグルール (日英バイリンガル)
# ---------------------------------------------------------------------------
DEFAULT_TAG_RULES: dict[str, TagRule] = {
    "ui-ux": TagRule(
        keywords=[
            "UI",
            "UX",
            "デザイン",
            "design",
            "レイアウト",
            "layout",
            "コンポーネント",
            "component",
            "スタイル",
            "style",
            "CSS",
            "Tailwind",
            "アクセシビリティ",
            "accessibility",
        ],
        threshold=2,
        prototype="UIデザインやUXに関するメモリ",
    ),
    "coding-style": TagRule(
        keywords=[
            "命名",
            "naming",
            "フォーマット",
            "format",
            "lint",
            "ruff",
            "prettier",
            "コーディング規約",
            "coding standard",
            "convention",
            "インデント",
            "indent",
        ],
        threshold=2,
        prototype="コーディングスタイルや規約に関するメモリ",
    ),
    "architecture": TagRule(
        keywords=[
            "アーキテクチャ",
            "architecture",
            "設計",
            "design pattern",
            "モジュール",
            "module",
            "レイヤー",
            "layer",
            "依存",
            "dependency",
            "DI",
            "クリーンアーキテクチャ",
            "clean architecture",
        ],
        threshold=2,
        prototype="ソフトウェアアーキテクチャや設計に関するメモリ",
    ),
    "debug": TagRule(
        keywords=[
            "デバッグ",
            "debug",
            "バグ",
            "bug",
            "エラー",
            "error",
            "例外",
            "exception",
            "トラブルシュート",
            "troubleshoot",
            "ログ",
            "log",
        ],
        threshold=2,
        prototype="デバッグやトラブルシューティングに関するメモリ",
    ),
    "config": TagRule(
        keywords=[
            "設定",
            "config",
            "環境変数",
            "env",
            ".env",
            "toml",
            "yaml",
            "json",
            "設定ファイル",
            "configuration",
        ],
        threshold=2,
        prototype="設定や環境構築に関するメモリ",
    ),
    "decision": TagRule(
        keywords=[
            "決定",
            "decision",
            "採用",
            "adopt",
            "理由",
            "reason",
            "トレードオフ",
            "trade-off",
            "比較",
            "compare",
            "選定",
            "選択",
            "select",
        ],
        threshold=2,
        prototype="技術的意思決定に関するメモリ",
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
    2. Windows: %LOCALAPPDATA%/cc-mnemos/config.toml
    3. Linux/macOS: $XDG_CONFIG_HOME/cc-mnemos/config.toml
       (デフォルト ~/.config/cc-mnemos/config.toml)
    """
    env_override = os.environ.get("CC_MNEMOS_CONFIG")
    if env_override:
        return Path(env_override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "")
        if base:
            return Path(base) / "cc-mnemos" / "config.toml"
        return Path.home() / "AppData" / "Local" / "cc-mnemos" / "config.toml"

    # Linux / macOS — XDG 準拠
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg_config:
        return Path(xdg_config) / "cc-mnemos" / "config.toml"
    return Path.home() / ".config" / "cc-mnemos" / "config.toml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """アプリケーション全体の設定を保持するデータクラス

    生成方法:
    - ``Config()`` — すべてデフォルト値
    - ``Config(embedding={...}, general={...})`` — TOML セクション辞書から構築
    - ``Config.from_file(path)`` — TOML ファイルから読み込み
    - ``Config.load()`` — デフォルトパスから読み込み (ファイルが無ければデフォルト)
    """

    # --- Embedding ---
    embedding_model: str = "cl-nagoya/ruri-v3-310m"
    embedding_dimension: int = 768

    # --- Search ---
    rrf_k: int = 60
    time_decay_half_life_days: int = 30
    default_search_limit: int = 10

    # --- Chunking ---
    max_chunk_tokens: int = 2000
    min_chunk_tokens: int = 20

    # --- Paths ---
    _data_dir: Path | None = field(default=None, repr=False)
    _db_path: Path | None = field(default=None, repr=False)

    # --- Embedding (extended) ---
    embedding_batch_size: int = 32

    # --- Maintenance ---
    max_chunk_age_days: int = 365
    max_db_size_mb: int = 500
    vacuum_interval_days: int = 30

    # --- Misc ---
    log_level: str = "INFO"
    project_mapping: dict[str, str] = field(default_factory=dict)
    tag_rules: dict[str, TagRule] = field(default_factory=lambda: dict(DEFAULT_TAG_RULES))

    def __init__(self, **raw_sections: dict[str, object]) -> None:
        """TOML セクション辞書からConfigを組み立てる

        Args:
            **raw_sections: ``embedding``, ``search``, ``chunking``,
                ``general``, ``tags``, ``projects`` などのセクション辞書
        """
        # デフォルト値で初期化
        self.embedding_model = "cl-nagoya/ruri-v3-310m"
        self.embedding_dimension = 768
        self.rrf_k = 60
        self.time_decay_half_life_days = 30
        self.default_search_limit = 10
        self.max_chunk_tokens = 2000
        self.min_chunk_tokens = 20
        self.embedding_batch_size = 32
        self.max_chunk_age_days = 365
        self.max_db_size_mb = 500
        self.vacuum_interval_days = 30
        self.log_level = "INFO"
        self.project_mapping = {}
        self.tag_rules = dict(DEFAULT_TAG_RULES)
        self._data_dir = None
        self._db_path = None

        # --- embedding セクション ---
        embedding = raw_sections.get("embedding", {})
        if isinstance(embedding, dict):
            if "model" in embedding:
                self.embedding_model = str(embedding["model"])
            if "dimension" in embedding:
                self.embedding_dimension = int(embedding["dimension"])  # type: ignore[arg-type]
            if "batch_size" in embedding:
                self.embedding_batch_size = int(embedding["batch_size"])  # type: ignore[arg-type]

        # --- maintenance セクション ---
        maintenance = raw_sections.get("maintenance", {})
        if isinstance(maintenance, dict):
            if "max_chunk_age_days" in maintenance:
                self.max_chunk_age_days = int(maintenance["max_chunk_age_days"])  # type: ignore[arg-type]
            if "max_db_size_mb" in maintenance:
                self.max_db_size_mb = int(maintenance["max_db_size_mb"])  # type: ignore[arg-type]
            if "vacuum_interval_days" in maintenance:
                self.vacuum_interval_days = int(maintenance["vacuum_interval_days"])  # type: ignore[arg-type]

        # --- search セクション ---
        search = raw_sections.get("search", {})
        if isinstance(search, dict):
            if "rrf_k" in search:
                self.rrf_k = int(search["rrf_k"])  # type: ignore[arg-type]
            if "time_decay_half_life_days" in search:
                self.time_decay_half_life_days = int(search["time_decay_half_life_days"])  # type: ignore[arg-type]
            if "default_search_limit" in search:
                self.default_search_limit = int(search["default_search_limit"])  # type: ignore[arg-type]

        # --- chunking セクション ---
        chunking = raw_sections.get("chunking", {})
        if isinstance(chunking, dict):
            if "max_chunk_tokens" in chunking:
                self.max_chunk_tokens = int(chunking["max_chunk_tokens"])  # type: ignore[arg-type]
            if "min_chunk_tokens" in chunking:
                self.min_chunk_tokens = int(chunking["min_chunk_tokens"])  # type: ignore[arg-type]

        # --- general セクション ---
        general = raw_sections.get("general", {})
        if isinstance(general, dict):
            if "data_dir" in general:
                self._data_dir = Path(str(general["data_dir"]))
            if "db_path" in general:
                self._db_path = Path(str(general["db_path"]))
            if "log_level" in general:
                self.log_level = str(general["log_level"])

        # --- 環境変数オーバーライド ---
        env_data_dir = os.environ.get("CC_MNEMOS_DATA_DIR")
        if env_data_dir:
            self._data_dir = Path(env_data_dir)

        # --- tags セクション ---
        tags = raw_sections.get("tags", {})
        if isinstance(tags, dict):
            for tag_name, tag_def in tags.items():
                if isinstance(tag_def, dict):
                    self.tag_rules[str(tag_name)] = TagRule(
                        keywords=list(tag_def.get("keywords", [])),
                        threshold=int(tag_def.get("threshold", 2)),  # type: ignore[arg-type]
                        prototype=str(tag_def.get("prototype", "")),
                    )

        # --- projects セクション ---
        projects = raw_sections.get("projects", {})
        if isinstance(projects, dict):
            self.project_mapping = {str(k): str(v) for k, v in projects.items()}

    @property
    def data_dir(self) -> Path:
        """データディレクトリのパスを返す"""
        if self._data_dir is not None:
            return self._data_dir
        return get_data_dir()

    @property
    def db_path(self) -> Path:
        """SQLite データベースファイルのパスを返す"""
        if self._db_path is not None:
            return self._db_path
        return self.data_dir / "memories.db"

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """TOML ファイルから Config を読み込む

        Args:
            path: 設定ファイルのパス

        Returns:
            読み込んだ設定を反映した Config インスタンス
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)

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
