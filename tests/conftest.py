"""pytest の共通設定。

``@slow`` は PyMC を要するテストに付ける。PyMC は既定の依存に入っていないので
（``uv sync --extra slow`` が要る）、入っていない環境では自動的に飛ばす。CI の既定の
ジョブが PyMC のビルドで10分待つことのないようにするための切り分けである。
"""

from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items) -> None:
    """PyMC が無い環境では ``@slow`` を自動で skip する。"""
    if importlib.util.find_spec("pymc") is not None:
        return
    skip = pytest.mark.skip(reason="PyMC が無い（uv sync --extra slow が要る）")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def seed() -> int:
    """テストで使う既定のシード。値そのものに意味はないが、動かさないことに意味がある。"""
    return 0
