"""Load cached data through the client wrappers into one MarketData bundle.

This is the only place that turns raw cached rows into ``MarketData``. The
backtest, the holdout run and paper trading all call ``load_market_data``.
The client wrappers enforce the holdout guard, so asking for an ``end`` on or
after 2024-01-01 raises unless ``approved=True`` and a holdout or paper
session is open (see bot/data/holdout.py).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd

from bot.config import MEMBERSHIP_START
from bot.data import eodhd_client as client
from bot.data.earnings import (
    EARNINGS_VISIBILITY_TRADING_DAYS,
    build_earnings_frame,
    parse_earnings_rows,
)
from bot.data.holdout import assert_before_holdout, is_paper_session
from bot.data.membership import (
    MembershipHistory,
    RawMembershipRow,
    build_membership_history,
    to_symbol,
)
from bot.data.news import (
    COMPANY_NEWS_RULE,
    NEWS_FRAME_COLUMNS,
    CompanyNewsRule,
    align_daily_sentiment,
    build_news_frame,
    parse_news_rows,
)
from bot.data.prices import (
    RawDividendRow,
    RawEodRow,
    RawSplitRow,
    build_price_frame,
    infer_splits_from_adjusted_ratio,
    merge_split_sources,
    parse_split_rows,
)
from bot.strategy.blocks import MarketData, build_market_data
from bot.universe import load_sample, universe_label, universe_mode

# Prices start a year before the membership history, for look-backs. The
# download script and the cache use the same start.
PRICE_START = date(2014, 1, 1)


MAX_SKIPPED_SHARE = 0.02

# Paper sessions read the earnings calendar past today (H1). It lists scheduled
# reports, so this is not lookahead. The reach is the visibility horizon plus
# a margin, in trading days, converted to calendar days with room for holidays.
EARNINGS_MARGIN_TRADING_DAYS = 20
EARNINGS_FORWARD_CALENDAR_DAYS = (
    (EARNINGS_VISIBILITY_TRADING_DAYS + EARNINGS_MARGIN_TRADING_DAYS) * 7 + 4
) // 5 + 3


class TooManySkippedError(RuntimeError):
    """Raised when more than 2% of the universe has no usable price data."""


@dataclass(frozen=True)
class LoadReport:
    """What a load did: universe size and every symbol skipped, with the reason."""

    universe_size: int
    skipped: tuple[tuple[str, str], ...]
    universe_mode: str = "full"
    universe_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output."""
        return {
            "universe_mode": self.universe_mode,
            "universe_label": self.universe_label,
            "universe_size": self.universe_size,
            "skipped_count": len(self.skipped),
            "skipped_symbols": [{"symbol": s, "reason": r} for s, r in self.skipped],
        }


def load_market_data(
    start: date,
    end: date,
    *,
    approved: bool = False,
    news_rule: CompanyNewsRule = COMPANY_NEWS_RULE,
) -> MarketData:
    """Build MarketData for the whole universe from cached rows between ``start`` and ``end``.

    See ``load_market_report`` for the version that also returns the skipped symbols.
    """
    return load_market_report(start, end, approved=approved, news_rule=news_rule)[0]


def load_market_report(
    start: date,
    end: date,
    *,
    approved: bool = False,
    news_rule: CompanyNewsRule = COMPANY_NEWS_RULE,
) -> tuple[MarketData, LoadReport]:
    """Like ``load_market_data`` but also returns the symbols that were skipped (M4).

    ``news_rule`` says which tagged articles count as company news (D31). The
    default is the locked 2-symbol rule (COMPANY_NEWS_RULE). Both news blocks
    read the same filtered set.

    In sample mode (the default, D33) only the frozen sample tickers and SPY are
    loaded, and membership is cut to the sample, so the 2% guard is measured
    against the tickers actually requested. ``EODHD_UNIVERSE=full`` loads all.

    Prices, dividends and earnings start at ``start``. News starts at
    ``start`` too, but never before MEMBERSHIP_START. A symbol whose price
    request is empty or fails is skipped and listed, never dropped silently.
    Raises TooManySkippedError when more than 2% of the universe is skipped,
    and HoldoutAccessError when ``end`` is in the holdout and no approved
    session is open.
    """
    assert_before_holdout(end, approved=approved)
    rows = cast(list[RawMembershipRow], client.get_index_history(approved=approved))
    mode = universe_mode()
    history = build_membership_history(rows, approved=approved)
    if mode == "sample":
        history = restrict_to_sample(history)
    company_names = company_names_by_code(rows)
    spy_frame = _price_frame("SPY", None, start, end, approved)
    days = pd.DatetimeIndex(spy_frame.index)
    universe = history.universe(max(start, MEMBERSHIP_START), end, approved=approved)
    prices: dict[str, pd.DataFrame] = {}
    news: dict[str, pd.DataFrame] = {}
    earnings: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for ticker in sorted(symbol.removesuffix(".US") for symbol in universe):
        symbol = f"{ticker}.US"
        try:
            prices[symbol] = _price_frame(ticker, days, start, end, approved)
        except client.EodhdPlanLimitError:
            raise
        except client.EodhdError as error:
            skipped.append((symbol, type(error).__name__))
            continue
        names = company_names.get(ticker, ())
        news[symbol] = _news_frame(
            ticker, symbol, days, start, end, approved, rule=news_rule, names=names
        )
        earnings[symbol] = _earnings_frame(ticker, days, start, end, approved)
    report = LoadReport(len(universe), tuple(skipped), mode, universe_label(mode))
    if len(skipped) > MAX_SKIPPED_SHARE * max(len(universe), 1):
        first = ", ".join(symbol for symbol, _ in skipped[:10])
        raise TooManySkippedError(
            f"{len(skipped)} of {len(universe)} symbols have no usable price data "
            f"(more than 2%). First ones: {first}."
        )
    market = build_market_data(prices, news, earnings, spy_frame, history, approved=approved)
    return market, report


def restrict_to_sample(history: MembershipHistory) -> MembershipHistory:
    """Keep only the spells of sample tickers (D33).

    On each date the tradable universe is then the sample intersected with that
    day's real members, because the membership table has columns for the sample only.
    """
    wanted = {to_symbol(ticker) for ticker in load_sample().tickers}
    return MembershipHistory(spells=tuple(s for s in history.spells if s.symbol in wanted))


def _price_frame(
    ticker: str, days: pd.DatetimeIndex | None, start: date, end: date, approved: bool
) -> pd.DataFrame:
    """One canonical price frame from cached raw rows, with reported and inferred splits."""
    eod = cast(list[RawEodRow], client.get_eod_prices(ticker, start, end, approved=approved))
    reported = parse_split_rows(
        cast(list[RawSplitRow], client.get_splits(ticker, approved=approved))
    )
    dividends = cast(
        list[RawDividendRow], client.get_dividends(ticker, start, end, approved=approved)
    )
    index = pd.DatetimeIndex([pd.Timestamp(row["date"]) for row in eod])
    inferred = infer_splits_from_adjusted_ratio(eod)
    splits = merge_split_sources(reported, inferred, index if days is None else days)
    return build_price_frame(eod, splits, dividends, approved=approved)


def company_names_by_code(rows: list[RawMembershipRow]) -> dict[str, tuple[str, ...]]:
    """Every distinct membership ``Name`` per ``Code``, for the company-news title test.

    A code with two spells (AMD, TER) may carry two spellings, so all are kept.
    """
    names: dict[str, list[str]] = {}
    for row in rows:
        known = names.setdefault(row["Code"], [])
        name = (row.get("Name") or "").strip()
        if name and name not in known:
            known.append(name)
    return {code: tuple(found) for code, found in names.items()}


MAX_NEWS_SPLIT_DEPTH = 6


def _cached_news_halves(
    ticker: str, first: date, end: date, approved: bool, depth: int = 0
) -> list[dict[str, Any]]:
    """Read news from the cache only, joining halves. Raises on a cache miss."""
    try:
        return client.get_news(ticker, first, end, approved=approved, offline=True)
    except client.OfflineCacheMissError:
        if depth >= MAX_NEWS_SPLIT_DEPTH or (end - first).days < 1:
            raise
    middle = date.fromordinal((first.toordinal() + end.toordinal()) // 2)
    left = _cached_news_halves(ticker, first, middle, approved, depth + 1)
    right = _cached_news_halves(ticker, middle + timedelta(days=1), end, approved, depth + 1)
    return left + right


def _news_rows(ticker: str, first: date, end: date, approved: bool) -> list[dict[str, Any]]:
    """News rows for one window, rebuilt from cached halves when the download split it.

    Very busy tickers do not fit one news query, so the download script stores them in
    halves (see split_news_windows in scripts/download_data.py). The cache is tried
    first, so a split ticker needs no network call. If the cache cannot rebuild the
    window, one normal call is made, as before.
    """
    try:
        return _cached_news_halves(ticker, first, end, approved)
    except client.OfflineCacheMissError:
        return client.get_news(ticker, first, end, approved=approved)


def _news_frame(
    ticker: str,
    symbol: str,
    days: pd.DatetimeIndex,
    start: date,
    end: date,
    approved: bool,
    *,
    rule: CompanyNewsRule = COMPANY_NEWS_RULE,
    names: tuple[str, ...] = (),
) -> pd.DataFrame:
    """The daily news frame for one stock, keeping only articles ``rule`` accepts."""
    first = max(start, MEMBERSHIP_START)
    try:
        rows = _news_rows(ticker, first, end, approved)
        daily = client.get_daily_sentiment(ticker, first, end, approved=approved)
    except client.NewsSymbolUnsupportedError:
        return unknown_news_frame(days)
    articles = parse_news_rows(cast(Any, rows), symbol, rule, names)
    return build_news_frame(articles, align_daily_sentiment(cast(Any, daily), days), days)


def unknown_news_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    """All-NaN news frame for a symbol EODHD has no news for (D26).

    Unknown is not zero: every news block reads NaN as False, so no news-based
    block can fire for the symbol, including no_news.
    """
    frame = pd.DataFrame(float("nan"), index=days, columns=list(NEWS_FRAME_COLUMNS))
    frame.index.name = "date"
    return frame


def _earnings_frame(
    ticker: str, days: pd.DatetimeIndex, start: date, end: date, approved: bool
) -> pd.DataFrame:
    """The daily earnings frame for one stock."""
    reach = end + timedelta(days=EARNINGS_FORWARD_CALENDAR_DAYS) if is_paper_session() else end
    rows = client.get_earnings(ticker, start, reach, approved=approved)
    return build_earnings_frame(parse_earnings_rows(cast(Any, rows), days, approved=approved), days)
