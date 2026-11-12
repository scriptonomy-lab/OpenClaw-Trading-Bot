"""Estimate the EODHD API calls for the full download. Makes no network calls.

By default it prints a realistic point estimate for the chosen universe
(sample or full, see `--universe`); `--wide` prints the old low/high table.

Costs used: EOD, splits, dividends and calendar requests cost 1 call. News
costs 5 per request plus 5 per ticker. Fundamentals are assumed to cost 10
(verify with the real key). Sentiment is assumed to cost 5 per request
(verify with the real key). Run with:
    uv run python scripts/estimate_api_calls.py [--tickers 750]
"""

import argparse
import json
import math
from dataclasses import dataclass

from bot.config import CACHE_DIR
from bot.universe import UniverseMode, load_sample, universe_label, universe_mode

CALL_CAP = 30_000  # D33: stop above this many calls
# News is the only uncertain part. The live 24-ticker run (D26) used about 1,100
# calls in total, with very busy names such as AAPL and TSLA included. Taking
# off the other per-ticker calls and the shared lists leaves the news share.
MEASURED_TICKERS, MEASURED_TOTAL_CALLS = 24, 1_100

FIRST_YEAR, LAST_YEAR = 2015, 2023
YEARS = LAST_YEAR - FIRST_YEAR + 1
NEWS_PAGE_LIMIT = 1000
NEWS_PAGINATION_CAP = 10_000  # offset + limit, so split a window beyond this
NEWS_PER_REQUEST, NEWS_PER_TICKER = 5, 5
FUNDAMENTALS_COST = 10  # ASSUMPTION: verify with the real key
SENTIMENT_COST = 5  # ASSUMPTION: verify with the real key
DEFAULT_TICKERS = 750

# Articles per ticker per year, (low, high). Coverage is thin before 2021: the
# live sample jumped about 35 times between 2020 and 2021, so 2021-2023 is the
# realistic dense-news span (D4 update). The download still covers 2015-2023, and
# the 2019-2023 era keeps the wide range as a cost upper bound. High is a heavily
# covered large cap.
ARTICLES_PER_YEAR = {"2015-2018": (5, 100), "2019-2023": (50, 1500)}
YEARS_IN_ERA = {"2015-2018": 4, "2019-2023": 5}

MEMBERSHIP_CACHE = (
    CACHE_DIR / "eodhd" / "fundamentals" / "GSPC.INDX" / "filter=HistoricalTickerComponents.json"
)


@dataclass(frozen=True)
class Line:
    """One row of the estimate table."""

    name: str
    low: int
    high: int
    note: str


def cached_universe_size() -> int | None:
    """Count distinct codes in the cached membership history, or None if absent."""
    if not MEMBERSHIP_CACHE.exists():
        return None
    data = json.loads(MEMBERSHIP_CACHE.read_text())
    rows = list(data.values()) if isinstance(data, dict) else data
    return len({row["Code"] for row in rows if row.get("Code")})


def news_calls(tickers: int) -> tuple[int, int]:
    """Return (low, high) news calls.

    Low: one date window per ticker for all years, paged by offset.
    High: one window per ticker-year (needed when a year can pass the
    10000 offset cap), at least one request each.
    """
    low_articles = sum(ARTICLES_PER_YEAR[e][0] * YEARS_IN_ERA[e] for e in YEARS_IN_ERA)
    high_per_year = {e: ARTICLES_PER_YEAR[e][1] for e in YEARS_IN_ERA}
    low_requests = max(1, math.ceil(low_articles / NEWS_PAGE_LIMIT))
    high_requests = sum(
        YEARS_IN_ERA[e] * max(1, math.ceil(high_per_year[e] / NEWS_PAGE_LIMIT))
        for e in YEARS_IN_ERA
    )
    fixed = tickers * NEWS_PER_TICKER
    return (
        tickers * low_requests * NEWS_PER_REQUEST + fixed,
        tickers * high_requests * NEWS_PER_REQUEST + fixed,
    )


def per_ticker_other_calls() -> int:
    """Calls per ticker for prices, splits, dividends, sentiment and earnings (not news)."""
    return 3 + SENTIMENT_COST + 1


def membership_calls() -> int:
    """History and current lists at the assumed 10 each, delisted 1, renames 1 per year."""
    return 2 * FUNDAMENTALS_COST + 1 + YEARS


def news_calls_per_ticker() -> int:
    """News calls per ticker backed out of the measured sample run (see module top)."""
    other = MEASURED_TICKERS * per_ticker_other_calls() + membership_calls()
    return math.ceil((MEASURED_TOTAL_CALLS - other) / MEASURED_TICKERS)


def realistic_lines(tickers: int, *, prices_only: bool = False) -> list[Line]:
    """One-figure estimate per item. SPY adds one request each to prices, splits and dividends."""
    lines = [
        Line("Prices (EOD)", tickers + 1, tickers + 1, "1 per ticker plus SPY"),
        Line("Splits", tickers + 1, tickers + 1, "1 per ticker plus SPY"),
        Line("Dividends", tickers + 1, tickers + 1, "1 per ticker plus SPY"),
    ]
    if not prices_only:
        per_news = news_calls_per_ticker()
        lines += [
            Line(
                "News",
                tickers * per_news,
                tickers * per_news,
                f"{per_news} per ticker, measured on the 24-ticker live run",
            ),
            Line("Sentiment", tickers * SENTIMENT_COST, tickers * SENTIMENT_COST, "5 per ticker"),
            Line("Earnings", tickers, tickers, "1 per ticker"),
        ]
    lines.append(Line("Membership + lists", membership_calls(), membership_calls(), "once"))
    return lines


def sample_size_and_label(mode: UniverseMode) -> tuple[int, str]:
    """Ticker count and a label for the chosen universe, without any network call."""
    if mode == "sample":
        return len(load_sample().tickers), universe_label("sample")
    cached = cached_universe_size()
    return (cached or load_sample().population), universe_label("full")


def print_realistic(mode: UniverseMode, *, prices_only: bool) -> int:
    """Print the realistic estimate and return its total."""
    tickers, label = sample_size_and_label(mode)
    lines = realistic_lines(tickers, prices_only=prices_only)
    total = sum(x.high for x in lines)
    kind = "prices only" if prices_only else "all data"
    print(f"Universe mode: {mode}. {label}")
    print(f"Estimate for {tickers} tickers plus SPY, {kind}:")
    for line in lines:
        print(f"  {line.name:<22}{line.high:>9,}  {line.note}")
    print(f"  {'TOTAL':<22}{total:>9,}  cap {CALL_CAP:,}")
    return total


def build_lines(tickers: int) -> list[Line]:
    """Build the estimate rows."""
    news_low, news_high = news_calls(tickers)
    quarters = YEARS * 4
    return [
        Line("Prices (EOD)", tickers, tickers, "1 request per ticker, full range"),
        Line("Splits + dividends", 2 * tickers, 2 * tickers, "2 requests per ticker"),
        Line("News", news_low, news_high, "5 per request + 5 per ticker; see assumptions"),
        Line("Sentiment", tickers * SENTIMENT_COST, tickers * SENTIMENT_COST, "assumed 5/request"),
        Line(
            "Earnings",
            quarters,
            tickers,
            f"bulk calendar by quarter ({quarters} requests, if symbols is optional) "
            "vs 1 per ticker",
        ),
        Line(
            "Membership + symbol lists",
            2 * FUNDAMENTALS_COST + 1 + YEARS,
            2 * FUNDAMENTALS_COST + 1 + YEARS,
            "history + current at assumed 10 each, delisted 1, renames 1 per year",
        ),
    ]


def print_table(lines: list[Line], tickers: int, source: str) -> None:
    """Print the table, totals and the assumptions."""
    print(f"Universe: {tickers} tickers ({source}). Years {FIRST_YEAR}-{LAST_YEAR}.\n")
    print(f"{'Item':<28}{'Low':>10}{'High':>10}  Note")
    for line in lines:
        print(f"{line.name:<28}{line.low:>10,}{line.high:>10,}  {line.note}")
    print(f"{'TOTAL':<28}{sum(x.low for x in lines):>10,}{sum(x.high for x in lines):>10,}")
    print("\nAssumptions to verify with the real key:")
    print(f"- Fundamentals cost {FUNDAMENTALS_COST} calls per request; sentiment {SENTIMENT_COST}.")
    print("- Articles per ticker-year (low, high):")
    for era, (low, high) in ARTICLES_PER_YEAR.items():
        print(f"    {era}: {low} to {high}")
    print(f"- News pages hold {NEWS_PAGE_LIMIT}; offset+limit is capped at {NEWS_PAGINATION_CAP}.")
    print("  A window with more articles must be split into smaller date ranges.")
    print("- News low case uses one window per ticker; high case one window per ticker-year.")
    print("- Earnings: the low figure assumes the calendar works without symbols.")
    print("- Re-runs cost nothing for cached responses.")


def main() -> None:
    """Parse arguments and print the estimate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--tickers", type=int, default=None, help="universe size override")
    parser.add_argument("--universe", choices=["sample", "full"], default=None)
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--wide", action="store_true", help="old low/high table")
    args = parser.parse_args()
    if not args.wide and args.tickers is None:
        print_realistic(args.universe or universe_mode(), prices_only=args.prices_only)
        return
    cached = cached_universe_size()
    if args.tickers is not None:
        tickers, source = args.tickers, "from --tickers"
    elif cached is not None:
        tickers, source = cached, "counted from cached membership history"
    else:
        tickers, source = DEFAULT_TICKERS, "default estimate, no cached membership yet"
    print_table(build_lines(tickers), tickers, source)


if __name__ == "__main__":
    main()
