"""Ruri v3 Embedding（差し替え可能）"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from cc_mnemos.config import Config

QUERY_PREFIX = "検索クエリ: "
DOCUMENT_PREFIX = "検索文書: "


class Embedder:
    """sentence-transformersベースのEmbedder"""

    def __init__(self, config: Config) -> None:
        self._model = SentenceTransformer(config.embedding_model)
        self._batch_size = config.embedding_batch_size
        self._dimension = config.embedding_dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """テキストリストをバッチエンコード"""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """検索クエリ用プレフィックスを付けてエンコード"""
        result = self.encode([f"{QUERY_PREFIX}{text}"])
        return result[0]

    def encode_document(self, text: str) -> np.ndarray:
        """ドキュメント用プレフィックスを付けてエンコード"""
        result = self.encode([f"{DOCUMENT_PREFIX}{text}"])
        return result[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """複数ドキュメントをバッチエンコード（プレフィックス付き）"""
        prefixed = [f"{DOCUMENT_PREFIX}{t}" for t in texts]
        return self.encode(prefixed)
