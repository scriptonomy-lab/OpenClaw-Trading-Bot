"""Integration tests need a real key. Without one they are skipped, never failed."""

from pathlib import Path

import pytest

from bot.config import data_mode

HERE = Path(__file__).parent
# Decided at import time. The shared autouse fixture clears EODHD_MODE inside each
# test, so checking later would miss an EODHD_MODE=fake set for this run.
NO_LIVE_KEY = data_mode() == "fake"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every test in this folder when there is no live key."""
    skip = pytest.mark.skip(reason="No live EODHD key (fake mode or no key): needs real data.")
    for item in items:
        if HERE in Path(str(item.fspath)).parents and NO_LIVE_KEY:
            item.add_marker(skip)
