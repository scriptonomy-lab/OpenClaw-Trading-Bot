"""Download all data for the backtest universe, up to 2023-12-31 only.

The universe comes from the S&P 500 membership history (every name that was
ever a member from 2015 to 2023), never from today's components. Everything
goes through the client wrappers, so caching and the holdout guard apply.
Works in both modes (fake and live). Examples:

    uv run python scripts/download_data.py --dry-run
    uv run python scripts/download_data.py --universe sample
    uv run python scripts/download_data.py --universe full --prices-only
    uv run python scripts/download_data.py --limit-tickers 5

The universe is the frozen sample unless --universe full (or EODHD_UNIVERSE=full)
is given (D33). The estimated calls are printed first, and the run stops before
starting if they are above 30,000. While it runs, it reads the day's call count
from GET /user every 10 tickers and stops cleanly if this run used more than
30,000. Re-run the same command to resume: cached responses cost nothing.
`--prices-only` fetches prices, splits, dividends and membership only.
"""

import argparse
import importlib.util
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

from bot.config import BACKTEST_END, MEMBERSHIP_START, data_mode
from bot.data import eodhd_client as client
from bot.data.membership import RawMembershipRow, build_membership_history
from bot.universe import UniverseMode, load_sample, universe_label, universe_mode

PRICE_START = date(2014, 1, 1)  # one year before membership, for look-back
NEWS_START = MEMBERSHIP_START
BENCHMARK = "SPY"
CALL_CAP = 30_000  # D33: stop before starting above this, and stop mid-run past it
BUDGET_CHECK_EVERY = 10  # tickers between reads of GET /user (free)


class EstimateTooHighError(RuntimeError):
    """The estimated calls are above the cap, so nothing was downloaded."""


class BudgetExceededError(RuntimeError):
    """The calls used by this run passed the cap. The cache keeps all progress."""


def load_estimator() -> Any:
    """Load scripts/estimate_api_calls.py (scripts are not a package)."""
    path = Path(__file__).resolve().parent / "estimate_api_calls.py"
    spec = importlib.util.spec_from_file_location("estimate_api_calls_for_download", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BudgetGuard:
    """Counts the calls this run has used, from ``apiRequests`` on GET /user.

    The counter resets each day, so a drop means a new day: what was used
    before the drop is kept and counting restarts from zero.
    """

    def __init__(self, reader: Callable[[], int], cap: int = CALL_CAP) -> None:
        self.reader, self.cap = reader, cap
        self.baseline = self.last = reader()
        self.carried = 0

    @property
    def used(self) -> int:
        """Calls used by this run so far, as of the last check."""
        return self.carried + self.last - self.baseline

    def check(self) -> int:
        """Read the counter and raise BudgetExceededError past the cap."""
        now = self.reader()
        if now < self.last:
            self.carried += self.last - self.baseline
            self.baseline = 0
        self.last = now
        if self.used > self.cap:
            raise BudgetExceededError(
                f"This run has used about {self.used:,} calls, above the cap of {self.cap:,}."
            )
        return self.used


def load_universe(*, offline: bool = False, mode: UniverseMode | None = None) -> list[str]:
    """Sorted tickers (no suffix) to download; in sample mode only the frozen sample."""
    rows = cast(list[RawMembershipRow], client.get_index_history(offline=offline))
    history = build_membership_history(rows)
    symbols = history.universe(MEMBERSHIP_START, BACKTEST_END)
    tickers = sorted(symbol.removesuffix(".US") for symbol in symbols)
    if (mode or universe_mode()) == "sample":
        wanted = set(load_sample().tickers)
        tickers = [ticker for ticker in tickers if ticker in wanted]
    return tickers


def split_news_windows(ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    """Fetch news, halving the date window whenever it is too big for one query."""
    try:
        return client.get_news(ticker, start, end)
    except client.NewsWindowTooLargeError:
        if (end - start).days < 1:
            raise
    middle = date.fromordinal((start.toordinal() + end.toordinal()) // 2)
    first = split_news_windows(ticker, start, middle)
    second = split_news_windows(ticker, date.fromordinal(middle.toordinal() + 1), end)
    return first + second


def download_ticker(
    ticker: str,
    rows: Counter[str],
    news_unavailable: list[str] | None = None,
    *,
    prices_only: bool = False,
) -> bool:
    """Download every category for one ticker and count the rows.

    Returns False when the ticker has no price data (empty /eod), in which
    case nothing else is downloaded for it.
    """
    try:
        price_rows = client.get_eod_prices(ticker, PRICE_START, BACKTEST_END)
    except client.EodhdEmptyResponseError:
        return False
    rows["prices"] += len(price_rows)
    if prices_only:
        rows["splits"] += len(client.get_splits(ticker))
        rows["dividends"] += len(client.get_dividends(ticker, PRICE_START, BACKTEST_END))
        return True
    download_other_categories(ticker, rows, news_unavailable)
    return True


def download_other_categories(
    ticker: str, rows: Counter[str], news_unavailable: list[str] | None = None
) -> None:
    """Download splits, dividends, news, sentiment and earnings for one ticker.

    A symbol that EODHD rejects for news (reused codes such as APC_old) is
    recorded as news unavailable, not as a failure. Its news is unknown (D26).
    """
    rows["splits"] += len(client.get_splits(ticker))
    rows["dividends"] += len(client.get_dividends(ticker, PRICE_START, BACKTEST_END))
    if ticker == BENCHMARK:
        return
    try:
        rows["news"] += len(split_news_windows(ticker, NEWS_START, BACKTEST_END))
        rows["sentiment"] += len(client.get_daily_sentiment(ticker, NEWS_START, BACKTEST_END))
    except client.NewsSymbolUnsupportedError:
        if news_unavailable is not None:
            news_unavailable.append(ticker)
    rows["earnings"] += len(client.get_earnings(ticker, PRICE_START, BACKTEST_END))


def download_reference_lists() -> None:
    """Fetch the lists that are not per ticker."""
    client.get_current_components()
    client.get_delisted_symbols()
    client.get_symbol_changes(MEMBERSHIP_START, BACKTEST_END)


def print_plan(tickers: int, mode: UniverseMode, *, prices_only: bool, cap: int = CALL_CAP) -> int:
    """Print the estimated API calls for this universe and return the total."""
    print(f"Data mode: {data_mode()}. {universe_label(mode)}")
    lines = load_estimator().realistic_lines(tickers, prices_only=prices_only)
    total = int(sum(line.high for line in lines))
    kind = "prices only" if prices_only else "all data"
    print(f"Estimated API calls for {tickers} tickers plus {BENCHMARK} ({kind}):")
    for line in lines:
        print(f"  {line.name:<22}{line.high:>9,}  {line.note}")
    print(f"  {'TOTAL':<22}{total:>9,}  (cap {cap:,})")
    print("News is estimated from the live 24-ticker run (about 1,100 calls in all).")
    return total


def run_dry(limit: int | None, mode: UniverseMode, *, prices_only: bool) -> None:
    """Print the plan without downloading prices or news."""
    try:
        count = len(load_universe(offline=True, mode=mode))
    except client.OfflineCacheMissError:
        if data_mode() == "live":
            # The sample size is known without any call. For the full index use its population.
            sample = load_sample()
            count = len(sample.tickers) if mode == "sample" else sample.population
        else:
            count = len(load_universe(mode=mode))
    total = print_plan(min(count, limit) if limit else count, mode, prices_only=prices_only)
    if total > CALL_CAP:
        print(f"This is above the cap of {CALL_CAP:,}. A real run would stop before starting.")
    if data_mode() == "live":
        print(f"API calls used today so far (free check): {client.api_requests_used():,}.")


def pick_sample(universe: list[str], wanted: list[str]) -> list[str]:
    """Keep the wanted tickers that really were members in 2015-2023.

    A sample is for checking that the live pipeline works, not for results:
    a backtest on it is not the S&P 500 and must not be reported as one.
    """
    missing = sorted(set(wanted) - set(universe))
    if missing:
        print(f"Not members in 2015-2023, skipped: {missing}")
    return [ticker for ticker in wanted if ticker in universe]


def run_download(
    limit: int | None,
    sample: list[str] | None = None,
    *,
    mode: UniverseMode | None = None,
    prices_only: bool = False,
    cap: int = CALL_CAP,
) -> int:
    """Run the download. Returns the number of tickers that failed.

    Raises EstimateTooHighError before any ticker if the estimate is above
    ``cap``, and BudgetExceededError if the calls used by this run pass it.
    """
    started = time.time()
    mode = mode or universe_mode()
    # An explicit --tickers list is checked against the whole index history.
    tickers = load_universe(mode="full" if sample else mode)
    if sample:
        tickers = pick_sample(tickers, sample)
    elif limit:
        tickers = tickers[:limit]
    total = print_plan(len(tickers), mode, prices_only=prices_only, cap=cap)
    if total > cap:
        raise EstimateTooHighError(
            f"Estimated {total:,} calls, above the cap of {cap:,}. Nothing was downloaded."
        )
    if not prices_only:
        download_reference_lists()
    guard = BudgetGuard(client.api_requests_used, cap)
    names = [*tickers, BENCHMARK]
    rows: Counter[str] = Counter()
    failed: list[str] = []
    no_price_data: list[str] = []
    news_unavailable: list[str] = []
    for position, ticker in enumerate(names, start=1):
        try:
            if not download_ticker(ticker, rows, news_unavailable, prices_only=prices_only):
                no_price_data.append(ticker)
                print(f"  NO PRICE DATA {ticker}", file=sys.stderr)
        except client.EodhdPlanLimitError:
            raise
        except client.EodhdError as exc:
            failed.append(ticker)
            print(f"  FAILED {ticker}: {exc}", file=sys.stderr)
        if position % BUDGET_CHECK_EVERY == 0 or position == len(names):
            used = guard.check()
            print(
                f"  {position}/{len(names)} tickers done, {time.time() - started:.1f}s, "
                f"{used:,} calls used by this run"
            )
    print(f"Data mode: {data_mode()}. Tickers: {len(tickers)} plus {BENCHMARK}.")
    print("Rows: " + ", ".join(f"{k}={v}" for k, v in sorted(rows.items())))
    print(f"No price data: {no_price_data or 'none'}.")
    print(f"News unavailable (unknown, not zero): {news_unavailable or 'none'}.")
    print(f"Failed: {failed or 'none'}. Total time {time.time() - started:.1f}s")
    return len(failed)


def main() -> None:
    """Parse flags and run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--limit-tickers", type=int, default=None, help="cost test on N tickers")
    parser.add_argument("--dry-run", action="store_true", help="print planned request counts only")
    parser.add_argument(
        "--tickers", default="", help="comma-separated sample of members, e.g. AAPL,AET,BRK-B"
    )
    parser.add_argument(
        "--universe",
        choices=["sample", "full"],
        default=None,
        help="sample (default, or EODHD_UNIVERSE) or the full historical index",
    )
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="prices, splits, dividends and membership only (no news, sentiment or earnings)",
    )
    args = parser.parse_args()
    mode: UniverseMode = args.universe or universe_mode()
    sample = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if args.dry_run:
        run_dry(len(sample) or args.limit_tickers, mode, prices_only=args.prices_only)
        return
    try:
        failed = run_download(
            args.limit_tickers, sample or None, mode=mode, prices_only=args.prices_only
        )
    except EstimateTooHighError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        sys.exit(2)
    except BudgetExceededError as error:
        print(f"STOPPED: {error} Progress is cached; re-run to resume.", file=sys.stderr)
        sys.exit(3)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
