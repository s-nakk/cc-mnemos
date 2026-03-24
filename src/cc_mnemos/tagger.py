"""2段階ルールベースタグ付け"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cc_mnemos.config import TagRule


def tag_by_keywords(text: str, tag_rules: dict[str, TagRule]) -> list[str]:
    """Stage 1: キーワードマッチングでタグを付与

    テキスト中に出現するキーワード(正規表現パターン)の数が
    閾値以上のタグを返す

    Args:
        text: タグ付け対象のテキスト
        tag_rules: タグ名とルールのマッピング

    Returns:
        マッチしたタグ名のリスト
    """
    matched_tags: list[str] = []
    for tag_name, rule in tag_rules.items():
        match_count = sum(
            1 for pattern in rule.keywords if re.search(pattern, text, re.IGNORECASE)
        )
        if match_count >= rule.threshold:
            matched_tags.append(tag_name)
    return matched_tags


def tag_by_embedding(
    chunk_embedding: np.ndarray,
    prototype_embeddings: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> list[str]:
    """Stage 2: Embeddingコサイン類似度でタグを付与

    チャンクの埋め込みベクトルと各タグのプロトタイプ埋め込みの
    コサイン類似度を計算し、閾値以上のタグを返す

    Args:
        chunk_embedding: チャンクの埋め込みベクトル
        prototype_embeddings: タグ名とプロトタイプ埋め込みのマッピング
        threshold: コサイン類似度の閾値

    Returns:
        マッチしたタグ名のリスト (マッチなしの場合は ``["general"]``)
    """
    matched_tags: list[str] = []
    chunk_norm = float(np.linalg.norm(chunk_embedding))
    if chunk_norm == 0:
        return ["general"]
    for tag_name, proto_emb in prototype_embeddings.items():
        proto_norm = float(np.linalg.norm(proto_emb))
        if proto_norm == 0:
            continue
        similarity = float(
            np.dot(chunk_embedding, proto_emb) / (chunk_norm * proto_norm)
        )
        if similarity >= threshold:
            matched_tags.append(tag_name)
    return matched_tags if matched_tags else ["general"]


# 短文閾値: この文字数未満ならkeyword+embedding両方を併用
SHORT_TEXT_THRESHOLD = 40


def assign_tags(
    text: str,
    tag_rules: dict[str, TagRule],
    chunk_embedding: np.ndarray | None = None,
    prototype_embeddings: dict[str, np.ndarray] | None = None,
    embedding_threshold: float = 0.5,
) -> list[str]:
    """2段階タグ付けを実行

    Stage 1 でキーワードマッチングを試み、マッチしなければ
    Stage 2 で埋め込み類似度を使用する。
    短文(40文字未満)の場合はkeyword+embeddingの結果をunionする。
    いずれもマッチしなければ ``["general"]`` を返す

    Args:
        text: タグ付け対象のテキスト
        tag_rules: タグ名とルールのマッピング
        chunk_embedding: チャンクの埋め込みベクトル (省略可)
        prototype_embeddings: タグ名とプロトタイプ埋め込みのマッピング (省略可)
        embedding_threshold: コサイン類似度の閾値

    Returns:
        付与されたタグ名のリスト
    """
    keyword_tags = tag_by_keywords(text, tag_rules)
    has_embeddings = chunk_embedding is not None and prototype_embeddings is not None

    # 短文は keyword + embedding を union して精度を上げる
    if len(text) < SHORT_TEXT_THRESHOLD and has_embeddings:
        emb_tags = tag_by_embedding(
            chunk_embedding, prototype_embeddings, embedding_threshold  # type: ignore[arg-type]
        )
        # "general" のみの場合は実質マッチなし
        emb_real = [t for t in emb_tags if t != "general"]
        combined = list(dict.fromkeys(keyword_tags + emb_real))
        return combined if combined else ["general"]

    # 通常テキスト: keyword優先、なければembeddingフォールバック
    if keyword_tags:
        return keyword_tags
    if has_embeddings:
        return tag_by_embedding(
            chunk_embedding, prototype_embeddings, embedding_threshold  # type: ignore[arg-type]
        )
    return ["general"]
