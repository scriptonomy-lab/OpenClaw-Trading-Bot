"""Record small EODHD demo-token fixtures for offline tests.

Uses the public "demo" token, so it only touches AAPL and TSLA. Nothing dated
2024 or later is recorded. Run with:
    uv run python scripts/record_fixtures.py
"""

import json
from datetime import date
from pathlib import Path

from bot.data import eodhd_client as client

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eodhd"
DEMO = "demo"


def record() -> None:
    """Call each wrapper once per fixture so the cache fills under FIXTURE_DIR."""
    opts = {"cache_dir": FIXTURE_DIR, "api_token": DEMO}
    split_start, split_end = date(2020, 8, 17), date(2020, 9, 15)
    for ticker in ("AAPL", "TSLA"):
        client.get_eod_prices(ticker, split_start, split_end, **opts)
        client.get_splits(ticker, **opts)
    client.get_dividends("AAPL", date(2019, 1, 1), date(2021, 12, 31), **opts)
    # Thin coverage year, then a busy window capped at one small page.
    client.get_news("AAPL", date(2016, 1, 1), date(2016, 12, 31), **opts)
    # allow_truncated: this busy window is deliberately cut to one page of 40
    # rows to keep the fixture small. It is NOT a full window.
    client.get_news(
        "AAPL",
        date(2023, 1, 3),
        date(2023, 1, 5),
        page_size=40,
        max_pages=1,
        allow_truncated=True,
        **opts,
    )
    client.get_daily_sentiment("AAPL", date(2023, 1, 3), date(2023, 1, 10), **opts)
    client.get_earnings("AAPL", date(2019, 1, 1), date(2021, 12, 31), **opts)


def strip_article_text() -> None:
    """Replace article bodies in news fixtures with a stub.

    The tests only need dates, symbols and sentiment. Full article text is
    third-party content that should not live in a public repo.
    """
    for file in (FIXTURE_DIR / "news").glob("*.json"):
        rows = json.loads(file.read_text())
        for row in rows:
            row["content"] = "(article text removed from test fixture)"
        file.write_text(json.dumps(rows))


def report() -> None:
    """Print each recorded file and its size."""
    total = 0
    for file in sorted(FIXTURE_DIR.rglob("*.json")):
        size = file.stat().st_size
        total += size
        print(f"{size:>8}  {file.relative_to(FIXTURE_DIR)}")
    print(f"{total:>8}  total")


if __name__ == "__main__":
    record()
    strip_article_text()
    report()
