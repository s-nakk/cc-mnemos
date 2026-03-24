from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cc_mnemos.config import Config
from cc_mnemos.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Generator


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
