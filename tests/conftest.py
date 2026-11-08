from __future__ import annotations

import shutil

import pytest

from reliable_tool_agent.world import build_database


@pytest.fixture(scope="session")
def shop_db(tmp_path_factory):
    """A freshly generated shop database, shared by all tests (never written to)."""
    return build_database(tmp_path_factory.mktemp("db") / "shop.db")


@pytest.fixture
def writable_db(shop_db, tmp_path):
    """A private copy of the shop database, for tests that write to it."""
    copy = tmp_path / "shop.db"
    shutil.copyfile(shop_db, copy)
    return copy
