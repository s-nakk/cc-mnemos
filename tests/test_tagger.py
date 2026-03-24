from __future__ import annotations

import numpy as np

from cc_mnemos.config import Config
from cc_mnemos.tagger import assign_tags, tag_by_embedding, tag_by_keywords


class TestKeywordTagging:
    def test_ui_ux_tag(self) -> None:
        config = Config()
        text = "ボタンのデザインとレイアウトを変更したい"
        tags = tag_by_keywords(text, config.tag_rules)
        assert "ui-ux" in tags

    def test_threshold_prevents_single_match(self) -> None:
        config = Config()
        text = "デザインについて考えています"
        tags = tag_by_keywords(text, config.tag_rules)
        assert "ui-ux" not in tags

    def test_coding_style_tag(self) -> None:
        config = Config()
        text = "lint設定を確認したい"
        tags = tag_by_keywords(text, config.tag_rules)
        assert "coding-style" in tags

    def test_multiple_tags(self) -> None:
        config = Config()
        text = "UIデザインのレイアウトを採用した理由を比較した結果"
        tags = tag_by_keywords(text, config.tag_rules)
        assert "ui-ux" in tags
        assert "decision" in tags

    def test_no_match_returns_empty(self) -> None:
        config = Config()
        text = "今日は天気がいい"
        tags = tag_by_keywords(text, config.tag_rules)
        assert tags == []


class TestEmbeddingFallback:
    def test_assigns_tag_above_threshold(self) -> None:
        chunk_emb = np.ones(768, dtype=np.float32)
        prototype_embs = {
            "ui-ux": np.ones(768, dtype=np.float32) * 0.9,
            "debug": np.zeros(768, dtype=np.float32),
        }
        tags = tag_by_embedding(chunk_emb, prototype_embs, threshold=0.5)
        assert "ui-ux" in tags
        assert "debug" not in tags

    def test_no_match_returns_general(self) -> None:
        chunk_emb = np.ones(768, dtype=np.float32)
        prototype_embs = {"ui-ux": -np.ones(768, dtype=np.float32)}
        tags = tag_by_embedding(chunk_emb, prototype_embs, threshold=0.5)
        assert tags == ["general"]


class TestAssignTags:
    def test_uses_keywords_when_available(self) -> None:
        config = Config()
        text = "UIデザインのレイアウトを変更"
        tags = assign_tags(text, config.tag_rules)
        assert "ui-ux" in tags

    def test_returns_general_when_no_keywords_and_no_embedding(self) -> None:
        config = Config()
        text = "今日は天気がいい"
        tags = assign_tags(text, config.tag_rules)
        assert tags == ["general"]
