# cc-mnemos

Long-term memory system for Claude Code — automatically saves and retrieves conversation context across sessions.

[日本語](#日本語)

## Overview

cc-mnemos gives Claude Code persistent memory by:

- Auto-saving conversation chunks when sessions end (via Claude Code Hooks)
- Auto-injecting relevant memories when sessions start
- On-demand search through an MCP server tool during conversations

All data stays local — no external APIs, no cloud storage. Embeddings are generated on-device using [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m), a Japanese-optimized sentence transformer.

## Features

- Hybrid search: FTS5 trigram full-text search + vector similarity search, fused with weighted Reciprocal Rank Fusion (RRF)
- Auto-tagging: 2-stage tagging (keyword regex + embedding cosine similarity) with categories like `ui-ux`, `coding-style`, `architecture`, `debug`, `config`, `decision`
- Smart chunking: Q&A pair extraction with consecutive message accumulation and short phatic reply merging
- Cross-project knowledge: memories from all projects are searchable, enabling knowledge transfer
- GPU acceleration: CUDA / MPS auto-detection for fast embedding generation
- Zero config: works out of the box with sensible defaults, fully customizable via TOML

## Requirements

- Python 3.10+
- Claude Code
- (Recommended) NVIDIA GPU with CUDA for faster embedding

## Installation

```bash
pip install cc-mnemos
```

## Quick Start

```bash
# 1. Download the embedding model and initialize the database
cc-mnemos setup

# 2. Register hooks and MCP server in Claude Code settings
cc-mnemos init

# 3. (Optional) Import existing Claude Code session history
cc-mnemos init --import-history
```

After setup, cc-mnemos works automatically:
- Session end → conversation is chunked, tagged, embedded, and stored
- Session start → recent and relevant memories are injected into context
- During conversation → Claude can call `search_memory` via MCP when needed

## Architecture

```
Claude Code
  ├── Hook: Stop              → cc-mnemos memorize       (auto-save)
  ├── Hook: SessionStart      → cc-mnemos recall          (auto-inject)
  ├── Hook: UserPromptSubmit  → cc-mnemos prompt-inject   (per-prompt inject)
  └── MCP Server              → cc-mnemos server          (on-demand search)

Pipeline:
  JSONL transcript
    → Chunker (Q&A pair extraction)
    → Tagger (keyword regex matching)
    → Embedder (Ruri v3, GPU-accelerated)
    → Store (SQLite + FTS5 trigram + sqlite-vec)
```

## CLI Commands

| Command                           | Description                                  |
|-----------------------------------|----------------------------------------------|
| `cc-mnemos setup`                 | Download embedding model + initialize DB     |
| `cc-mnemos init`                  | Register hooks and MCP server in Claude Code |
| `cc-mnemos init --import-history` | Also import existing session history         |
| `cc-mnemos stats`                 | Show memory statistics                       |
| `cc-mnemos search "query"`        | Search memories from CLI                     |
| `cc-mnemos server`                | Start the MCP server                         |

## MCP Tools

| Tool               | Description                                   |
|--------------------|-----------------------------------------------|
| `search_memory`    | Hybrid search over past conversation memories |
| `get_memory_stats` | Return chunk/session/project statistics       |
| `list_projects`    | List all projects with stored memories        |

## Configuration

Create `~/.config/cc-mnemos/config.toml` (Linux/macOS) or `%APPDATA%/cc-mnemos/config.toml` (Windows):

```toml
[embedding]
model = "cl-nagoya/ruri-v3-310m"
dimension = 768
batch_size = 32

[search]
rrf_k = 60
fts_weight = 2.0
vector_weight = 0.75
time_decay_half_life_days = 180

[tags.my-custom-tag]
keywords = ["pattern1", "pattern2"]
threshold = 2
prototype = "description for embedding similarity"
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

# 日本語

## 概要

cc-mnemos は Claude Code に長期記憶を与えるシステムです。

- セッション終了時に会話を 自動保存 (Claude Code Hooks)
- セッション開始時に関連する記憶を 自動注入
- 会話中に MCP ツールで オンデマンド検索

すべてのデータはローカルに保存されます。外部 API やクラウドストレージは使用しません。埋め込みベクトルは日本語に最適化された [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) でデバイス上で生成します。

## 主な機能

- ハイブリッド検索: FTS5 trigram 全文検索 + ベクトル類似度検索を重み付き RRF で統合
- 自動タグ付け: キーワード正規表現 + 埋め込みコサイン類似度の 2 段階タグ付け (`ui-ux`, `coding-style`, `architecture`, `debug`, `config`, `decision`)
- スマートチャンキング: Q&A ペア抽出、連続発話の蓄積、短い追撃発話のマージ
- プロジェクト横断検索: 全プロジェクトの記憶を横断検索し、知見を活用
- GPU 高速化: CUDA / MPS 自動検出で高速な埋め込み生成
- ゼロコンフィグ: デフォルト設定でそのまま動作、TOML で完全カスタマイズ可能

## 必要要件

- Python 3.10 以上
- Claude Code
- (推奨) CUDA 対応 NVIDIA GPU

## インストール

```bash
pip install cc-mnemos
```

## セットアップ

```bash
# 1. 埋め込みモデルのダウンロードと DB 初期化
cc-mnemos setup

# 2. Claude Code の settings.json にフックと MCP サーバーを登録
cc-mnemos init

# 3. (任意) 既存のセッション履歴をインポート
cc-mnemos init --import-history
```

セットアップ後は自動で動作します:
- セッション終了 → 会話がチャンク分割・タグ付け・ベクトル化されて保存
- セッション開始 → 直近の記憶とよく参照される知見が自動注入
- 会話中 → Claude が必要に応じて `search_memory` を MCP 経由で呼び出し

## アーキテクチャ

```
Claude Code
  ├── Hook: Stop              → cc-mnemos memorize       (自動保存)
  ├── Hook: SessionStart      → cc-mnemos recall          (自動注入)
  ├── Hook: UserPromptSubmit  → cc-mnemos prompt-inject   (発話ごと注入)
  └── MCP Server              → cc-mnemos server          (オンデマンド検索)

パイプライン:
  JSONL トランスクリプト
    → Chunker (Q&A ペア抽出)
    → Tagger (キーワード正規表現マッチング)
    → Embedder (Ruri v3, GPU 高速化)
    → Store (SQLite + FTS5 trigram + sqlite-vec)
```

## CLI コマンド

| コマンド                          | 説明                                       |
|-----------------------------------|--------------------------------------------|
| `cc-mnemos setup`                 | 埋め込みモデルのダウンロード + DB 初期化   |
| `cc-mnemos init`                  | フックと MCP サーバーを Claude Code に登録 |
| `cc-mnemos init --import-history` | 既存セッション履歴もインポート             |
| `cc-mnemos stats`                 | 記憶の統計情報を表示                       |
| `cc-mnemos search "クエリ"`       | CLI から記憶を検索                         |
| `cc-mnemos server`                | MCP サーバーを起動                         |

## 設定

`~/.config/cc-mnemos/config.toml` (Linux/macOS) または `%APPDATA%/cc-mnemos/config.toml` (Windows) に配置:

```toml
[embedding]
model = "cl-nagoya/ruri-v3-310m"  # 埋め込みモデル
dimension = 768
batch_size = 32

[search]
rrf_k = 60
fts_weight = 2.0          # FTS スコアの重み
vector_weight = 0.75       # ベクトルスコアの重み
time_decay_half_life_days = 180  # 時間減衰の半減期(日)

[tags.my-custom-tag]
keywords = ["パターン1", "pattern2"]
threshold = 2
prototype = "埋め込み類似度用の説明文"
```

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照してください。
