from __future__ import annotations

import numpy as np
import pytest

from cc_mnemos.config import Config
from cc_mnemos.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    config = Config()
    return Embedder(config)


class TestEmbedder:
    def test_encode_single(self, embedder: Embedder) -> None:
        result = embedder.encode(["テスト文章"])
        assert result.shape == (1, 768)
        assert result.dtype == np.float32

    def test_encode_batch(self, embedder: Embedder) -> None:
        texts = ["テスト文章1", "テスト文章2", "テスト文章3"]
        result = embedder.encode(texts)
        assert result.shape == (3, 768)

    def test_encode_with_query_prefix(self, embedder: Embedder) -> None:
        result = embedder.encode_query("検索テスト")
        assert result.shape == (768,)

    def test_encode_with_document_prefix(self, embedder: Embedder) -> None:
        result = embedder.encode_document("ドキュメントテスト")
        assert result.shape == (768,)

    def test_query_and_document_differ(self, embedder: Embedder) -> None:
        text = "同じテキスト"
        q = embedder.encode_query(text)
        d = embedder.encode_document(text)
        assert not np.allclose(q, d)
