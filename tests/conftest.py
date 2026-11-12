"""Shared fixtures: cached EODHD JSON read through the client in offline mode."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot.data import eodhd_client as client

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "eodhd"


class FixtureLoader:
    """Reads raw rows from tests/fixtures/eodhd through the client wrappers, offline."""

    def _kwargs(self) -> dict[str, Any]:
        return {"cache_dir": FIXTURE_DIR, "offline": True, "api_token": "demo"}

    def eod(self, ticker: str) -> list[dict[str, Any]]:
        """Daily bars for 2020-08-17 to 2020-09-15."""
        from datetime import date

        return client.get_eod_prices(ticker, date(2020, 8, 17), date(2020, 9, 15), **self._kwargs())

    def splits(self, ticker: str) -> list[dict[str, Any]]:
        """All splits for the ticker."""
        return client.get_splits(ticker, **self._kwargs())

    def dividends(self, ticker: str) -> list[dict[str, Any]]:
        """Dividends 2019-01-01 to 2021-12-31."""
        from datetime import date

        return client.get_dividends(ticker, date(2019, 1, 1), date(2021, 12, 31), **self._kwargs())

    def news_2016(self) -> list[dict[str, Any]]:
        """AAPL articles for 2016 (one article)."""
        from datetime import date

        return client.get_news("AAPL", date(2016, 1, 1), date(2016, 12, 31), **self._kwargs())

    def news_2023(self) -> list[dict[str, Any]]:
        """AAPL articles 2023-01-03 to 2023-01-05 (40 rows, a truncated page)."""
        from datetime import date

        return client.get_news(
            "AAPL",
            date(2023, 1, 3),
            date(2023, 1, 5),
            page_size=40,
            max_pages=1,
            allow_truncated=True,
            **self._kwargs(),
        )

    def sentiments_2023(self) -> list[dict[str, Any]]:
        """Daily sentiment rows for AAPL 2023-01-03 to 2023-01-10."""
        from datetime import date

        return client.get_daily_sentiment(
            "AAPL", date(2023, 1, 3), date(2023, 1, 10), **self._kwargs()
        )

    def earnings(self) -> list[dict[str, Any]]:
        """AAPL earnings rows 2019 to 2021."""
        from datetime import date

        return client.get_earnings("AAPL", date(2019, 1, 1), date(2021, 12, 31), **self._kwargs())


@pytest.fixture(autouse=True)
def clear_eodhd_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without EODHD_MODE, so exporting it for the shell does not leak in.

    CLAUDE.md tells people to export ``EODHD_MODE=fake``. Client tests that
    expect the real code path would fail with it set. A test that needs the
    fake world sets the variable itself (review L8).
    """
    monkeypatch.delenv("EODHD_MODE", raising=False)
    # The fake world uses SIM tickers, which are not in the real frozen sample
    # (D33). Tests use the full universe unless a test sets sample mode itself,
    # so fake-mode tests never load an empty universe and pass by accident.
    monkeypatch.setenv("EODHD_UNIVERSE", "full")


@pytest.fixture
def fixtures() -> FixtureLoader:
    """Loader for cached EODHD fixtures."""
    return FixtureLoader()


@pytest.fixture
def trading_days() -> pd.DatetimeIndex:
    """Weekdays 2023-01-02 to 2023-03-31 as a stand-in trading calendar."""
    return pd.bdate_range("2023-01-02", "2023-03-31")


# End-to-end files that load the fake market or run the CLI. The pre-commit
# hook skips them (-m "not slow"); the pre-push hook runs everything.
SLOW_TEST_FILES = {
    "test_cli_commands.py",
    "test_results_summary.py",
    "test_paper_update.py",
    "test_lookahead.py",
    "test_planted_backtest.py",
    "test_backtest_review_fixes.py",
    "test_approval_concurrency.py",
    "test_session_data.py",
    "test_verify_live.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in a slow file with the slow marker."""
    for item in items:
        if item.path.name in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
