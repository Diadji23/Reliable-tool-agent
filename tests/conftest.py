from __future__ import annotations

import pytest

from reliable_tool_agent.world import build_database


@pytest.fixture(scope="session")
def shop_db(tmp_path_factory):
    """A freshly generated shop database, shared by all tests (never written to)."""
    return build_database(tmp_path_factory.mktemp("db") / "shop.db")
