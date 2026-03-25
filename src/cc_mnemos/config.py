"""設定モジュール — TOML / 環境変数 / デフォルト値を統合して提供する"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
        self._raw: dict[str, object] = dict(raw_sections)

        # 環境変数オーバーライド: data_dir
        env_data_dir = os.environ.get("CC_MNEMOS_DATA_DIR")
        if env_data_dir:
            general = dict(self._raw.get("general", {}))  # type: ignore[arg-type]
            general["data_dir"] = env_data_dir
            self._raw["general"] = general

    # --- Embedding ---
    @property
    def embedding_model(self) -> str:
        """埋め込みモデル名を返す"""
        return str(self._raw.get("embedding", {}).get("model", "cl-nagoya/ruri-v3-310m"))  # type: ignore[union-attr]

    @property
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        return int(self._raw.get("embedding", {}).get("dimension", 768))  # type: ignore[union-attr]

    @property
    def embedding_batch_size(self) -> int:
        """埋め込みバッチサイズを返す"""
        return int(self._raw.get("embedding", {}).get("batch_size", 32))  # type: ignore[union-attr]

    # --- Search ---
    @property
    def rrf_k(self) -> int:
        """RRF の k パラメータを返す"""
        return int(self._raw.get("search", {}).get("rrf_k", 60))  # type: ignore[union-attr]

    @property
    def time_decay_half_life_days(self) -> int:
        """時間減衰の半減期(日)を返す"""
        return int(self._raw.get("search", {}).get("time_decay_half_life_days", 180))  # type: ignore[union-attr]

    @property
    def fts_weight(self) -> float:
        """RRFにおけるFTSスコアの重みを返す"""
        return float(self._raw.get("search", {}).get("fts_weight", 2.0))  # type: ignore[union-attr]

    @property
    def vector_weight(self) -> float:
        """RRFにおけるベクトルスコアの重みを返す"""
        return float(self._raw.get("search", {}).get("vector_weight", 0.75))  # type: ignore[union-attr]

    @property
    def default_search_limit(self) -> int:
        """検索結果のデフォルト上限を返す"""
        return int(self._raw.get("search", {}).get("default_search_limit", 10))  # type: ignore[union-attr]

    # --- Chunking ---
    @property
    def max_chunk_tokens(self) -> int:
        """チャンクの最大トークン数を返す"""
        return int(self._raw.get("chunking", {}).get("max_chunk_tokens", 2000))  # type: ignore[union-attr]

    @property
    def min_chunk_tokens(self) -> int:
        """チャンクの最小トークン数を返す"""
        return int(self._raw.get("chunking", {}).get("min_chunk_tokens", 20))  # type: ignore[union-attr]

    # --- Maintenance ---
    @property
    def max_chunk_age_days(self) -> int:
        """チャンクの最大保持日数を返す"""
        return int(self._raw.get("maintenance", {}).get("max_chunk_age_days", 365))  # type: ignore[union-attr]

    @property
    def max_db_size_mb(self) -> int:
        """データベースの最大サイズ(MB)を返す"""
        return int(self._raw.get("maintenance", {}).get("max_db_size_mb", 500))  # type: ignore[union-attr]

    @property
    def vacuum_interval_days(self) -> int:
        """VACUUM の実行間隔(日)を返す"""
        return int(self._raw.get("maintenance", {}).get("vacuum_interval_days", 30))  # type: ignore[union-attr]

    # --- Misc ---
    @property
    def log_level(self) -> str:
        """ログレベルを返す"""
        return str(self._raw.get("general", {}).get("log_level", "INFO"))  # type: ignore[union-attr]

    @property
    def project_mapping(self) -> dict[str, str]:
        """プロジェクトパスとプロジェクト名のマッピングを返す"""
        raw = self._raw.get("projects", {})
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {}

    # --- Paths ---
    @property
    def data_dir(self) -> Path:
        """データディレクトリのパスを返す"""
        general = self._raw.get("general", {})
        if isinstance(general, dict) and "data_dir" in general:
            return Path(str(general["data_dir"]))
        return get_data_dir()

    @property
    def db_path(self) -> Path:
        """SQLite データベースファイルのパスを返す"""
        general = self._raw.get("general", {})
        if isinstance(general, dict) and "db_path" in general:
            return Path(str(general["db_path"]))
        return self.data_dir / "memories.db"

    # --- Tags ---
    @property
    def tag_rules(self) -> dict[str, TagRule]:
        """タグルールの辞書を返す"""
        rules = dict(DEFAULT_TAG_RULES)
        raw_tags = self._raw.get("tags", {})
        if isinstance(raw_tags, dict):
            for tag_name, tag_def in raw_tags.items():
                if isinstance(tag_def, dict):
                    rules[str(tag_name)] = TagRule(
                        keywords=list(tag_def.get("keywords", [])),
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
