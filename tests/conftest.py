from __future__ import annotations

import json
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    TranscriptFactory = Callable[[list[tuple[str, str]] | None, str], Path]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """テスト用設定（tmp_pathをデータディレクトリに使用）"""
    return Config(general={"data_dir": str(tmp_path)})


@pytest.fixture
def store(config: Config) -> Generator[MemoryStore, None, None]:
    """テスト用MemoryStore"""
    s = MemoryStore(config)
    yield s
    s.close()


def make_session_id() -> str:
    return str(uuid.uuid4())


def make_chunk(
    session_id: str,
    role_user: str = "テスト質問",
    role_assistant: str = "テスト回答",
    tags: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, str | int]:
    if tags is None:
        tags = ["general"]
    if created_at is None:
        created_at = datetime.now(tz=timezone.utc).isoformat()
    content = f"{role_user}\n{role_assistant}"
    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "role_user": role_user,
        "role_assistant": role_assistant,
        "content": content,
        "tags": json.dumps(tags),
        "created_at": created_at,
        "token_count": len(content.split()),
    }


# ---------------------------------------------------------------------------
# 埋め込みモデルをロードしないフェイク Embedder
#   テストで Ruri v3 を実ダウンロードするのを避けるための fixture
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """テスト用の軽量 Embedder スタンドイン

    実モデルをロードせず、与えられた件数に応じてゼロベクトルを返す
    呼び出し履歴を ``calls`` 属性に記録し、テストで参照できる
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls: list[tuple[str, object]] = []
        FakeEmbedder.instances.append(self)

    instances: list[FakeEmbedder] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.calls.append(("encode", list(texts)))
        return np.zeros((len(texts), 768), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.calls.append(("encode_query", text))
        return np.zeros(768, dtype=np.float32)

    def encode_document(self, text: str) -> np.ndarray:
        self.calls.append(("encode_document", text))
        return np.zeros(768, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(("encode_documents", list(texts)))
        return np.zeros((len(texts), 768), dtype=np.float32)

    def encode_topic(self, text: str) -> np.ndarray:
        self.calls.append(("encode_topic", text))
        return np.zeros(768, dtype=np.float32)

    def encode_topics(self, texts: list[str]) -> np.ndarray:
        self.calls.append(("encode_topics", list(texts)))
        return np.zeros((len(texts), 768), dtype=np.float32)


@pytest.fixture
def mock_embedder(monkeypatch: pytest.MonkeyPatch) -> type[FakeEmbedder]:
    """``Embedder`` 実装を ``FakeEmbedder`` に差し替える

    Ruri v3 のダウンロードを避けて軽量にテストを回すための fixture
    呼び出し履歴は ``FakeEmbedder.instances[-1].calls`` で参照できる
    """
    FakeEmbedder.instances = []
    monkeypatch.setattr("cc_mnemos.embedder.Embedder", FakeEmbedder)
    return FakeEmbedder


# ---------------------------------------------------------------------------
# トランスクリプト JSONL のヘルパー
# ---------------------------------------------------------------------------
def make_transcript(
    path: Path,
    pairs: list[tuple[str, str]] | None = None,
) -> Path:
    """テスト用のトランスクリプト JSONL を作成する

    Args:
        path: 出力ファイルパス
        pairs: (user発話, assistant応答) のリスト
            省略時はサンプルの 2 ペアを書き込む

    Returns:
        作成したファイルのパス
    """
    if pairs is None:
        pairs = [
            (
                "border-radius の設定方法を教えて",
                "CSS の border-radius で角丸を設定できます",
            ),
            (
                "ESLint と Prettier の使い分けは？",
                "ESLint はバグ検出、Prettier はフォーマットを担当します",
            ),
        ]

    lines: list[str] = []
    for user_text, assistant_text in pairs:
        lines.append(json.dumps({
            "type": "user",
            "message": {"content": user_text},
        }, ensure_ascii=False))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": assistant_text},
        }, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def transcript_factory(tmp_path: Path) -> TranscriptFactory:
    """トランスクリプト JSONL を返す factory fixture"""

    def _make(
        pairs: list[tuple[str, str]] | None = None,
        name: str = "transcript.jsonl",
    ) -> Path:
        return make_transcript(tmp_path / name, pairs)

    return _make


# ---------------------------------------------------------------------------
# 偽 socket — TCP I/O を伴うコードを単体テストするためのスタブ
# ---------------------------------------------------------------------------
class FakeSocket:
    """``socket.create_connection`` の戻り値に差し込めるテスト用 socket

    送信されたバイト列を ``sent`` に蓄積し、事前設定した ``response`` を
    ``recv()`` で返す。コンテキストマネージャとしても使える
    既存テスト用に ``settimeout`` を受け取れる旧 ``_FakeReadySocket`` の
    置き換えもこのクラスで賄える
    """

    def __init__(
        self,
        response: bytes = b"",
        *,
        connect_error: BaseException | None = None,
        recv_error: BaseException | None = None,
        send_error: BaseException | None = None,
    ) -> None:
        self.sent = bytearray()
        self._response = response
        self._connect_error = connect_error
        self._recv_error = recv_error
        self._send_error = send_error
        self._timeout: float | None = None
        self.closed = False
        if connect_error is not None:
            raise connect_error

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    def sendall(self, data: bytes) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.extend(data)

    def shutdown(self, _how: int) -> None:
        return None

    def recv(self, _bufsize: int) -> bytes:
        if self._recv_error is not None:
            raise self._recv_error
        if self._response:
            data, self._response = self._response, b""
            return data
        return b""

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@pytest.fixture
def free_port() -> int:
    """空き TCP ポートを 1 つ確保して返す"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
