"""Shared builders for the block, engine, metrics and planted-case tests.

Two kinds of data are built here:

- Tiny hand-built markets, where every price is typed in so that expected
  values can be worked out on paper.
- The fake EODHD world (``EODHD_MODE=fake``), loaded once as raw rows and
  turned into ``MarketData`` the same way the real pipeline would, with an
  optional cut so the strict lookahead test can rebuild from rows known at D.

The toy builders create ``MarketData`` directly (not through
``build_market_data``) so that engine and block tests do not depend on it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, run_backtest
from bot.config import BACKTEST_END, MEMBERSHIP_START
from bot.data import eodhd_client as client
from bot.data.earnings import (
    EarningsEvent,
    build_earnings_frame,
    extend_trading_days,
    parse_earnings_rows,
)
from bot.data.membership import (
    MembershipHistory,
    MembershipSpell,
    RawMembershipRow,
    build_membership_history,
)
from bot.data.news import (
    RawDailySentimentRow,
    RawNewsRow,
    align_daily_sentiment,
    build_news_frame,
    decision_datetime,
    open_datetime,
    parse_news_rows,
)
from bot.data.prices import (
    PRICE_COLUMNS,
    RawDividendRow,
    RawEodRow,
    RawSplitRow,
    build_price_frame,
    parse_split_rows,
    total_return_index,
    wide_panel,
)
from bot.strategy.blocks import MarketData
from bot.strategy.schema import BacktestWindow, StrategyDefinition, strategy_from_json

FAKE_START = date(2014, 1, 1)
FAKE_END = date(2023, 12, 31)
SPY = "SPY"

# --------------------------------------------------------------- toy markets


def toy_days(count: int, start: str = "2023-03-01") -> pd.DatetimeIndex:
    """``count`` consecutive weekdays. The engine only needs a calendar, not real holidays."""
    return pd.bdate_range(start, periods=count, name="date")


@dataclass(frozen=True)
class Bar:
    """One raw (as traded) daily bar."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 1_000.0


def flat_bar(price: float, volume: float = 1_000.0) -> Bar:
    """A bar where open, high, low and close are all ``price``."""
    return Bar(price, price, price, price, volume)


def raw_frame(
    days: pd.DatetimeIndex,
    bars: dict[int, Bar],
    *,
    dividends: dict[int, float] | None = None,
    splits: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Canonical price frame built by the real ``build_price_frame`` from raw bars.

    ``bars`` maps a day number (position in ``days``) to the raw bar. A day
    without a bar is a day with no row. ``dividends`` and ``splits`` map a day
    number to the raw cash per share and to the split factor with that ex-date.
    """
    rows = [
        cast(
            RawEodRow,
            {
                "date": days[n].date().isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "adjusted_close": bar.close,
                "volume": bar.volume,
            },
        )
        for n, bar in sorted(bars.items())
    ]
    split_rows = [
        cast(RawSplitRow, {"date": days[n].date().isoformat(), "split": f"{factor:.6f}/1.000000"})
        for n, factor in (splits or {}).items()
    ]
    dividend_rows = [
        cast(
            RawDividendRow,
            {"date": days[n].date().isoformat(), "unadjustedValue": amount, "value": amount},
        )
        for n, amount in (dividends or {}).items()
    ]
    return build_price_frame(rows, parse_split_rows(split_rows), dividend_rows)


def assemble_market(
    days: pd.DatetimeIndex,
    frames: dict[str, pd.DataFrame],
    *,
    spy_frame: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    article_count: pd.DataFrame | None = None,
    mean_polarity: pd.DataFrame | None = None,
    open_article_count: pd.DataFrame | None = None,
    open_mean_polarity: pd.DataFrame | None = None,
    days_to_next: pd.DataFrame | None = None,
    days_since_last: pd.DataFrame | None = None,
) -> MarketData:
    """Make a ``MarketData`` straight from canonical frames. Nothing is forward filled.

    A missing membership table means "every stock is a member on every day".
    A missing news or earnings table means "no stock has that data".
    """
    spy = (
        spy_frame
        if spy_frame is not None
        else raw_frame(days, {n: flat_bar(400.0) for n in range(len(days))})
    )

    def table(column: str) -> pd.DataFrame:
        panel = wide_panel(frames, column).reindex(days)
        panel.index.name = "date"
        return panel

    index_table = pd.DataFrame({s: total_return_index(f) for s, f in frames.items()}).reindex(days)
    index_table.index.name = "date"
    members = (
        membership
        if membership is not None
        else pd.DataFrame(True, index=days, columns=sorted(frames), dtype=bool)
    )

    def optional(value: pd.DataFrame | None) -> pd.DataFrame:
        if value is None:
            empty = pd.DataFrame(index=days)
            empty.index.name = "date"
            return empty
        return value.reindex(days)

    return MarketData(
        trading_days=days,
        open=table("open"),
        high=table("high"),
        low=table("low"),
        close=table("close"),
        raw_close=table("raw_close"),
        dividend=table("dividend"),
        split_factor=table("split_factor"),
        total_return_index=index_table,
        article_count=optional(article_count),
        mean_polarity=optional(mean_polarity),
        open_article_count=optional(open_article_count),
        open_mean_polarity=optional(open_mean_polarity),
        days_to_next_earnings=optional(days_to_next),
        days_since_last_earnings=optional(days_since_last),
        spy=spy,
        membership=members,
        volume_by_symbol={s: f["volume"].reindex(days) for s, f in frames.items()},
    )


def engine_market(
    count: int,
    stocks: dict[str, dict[int, Bar]],
    *,
    dividends: dict[str, dict[int, float]] | None = None,
    splits: dict[str, dict[int, float]] | None = None,
    membership: pd.DataFrame | None = None,
) -> MarketData:
    """A toy market of ``count`` days with a flat SPY, for engine tests."""
    days = toy_days(count)
    frames = {
        symbol: raw_frame(
            days,
            bars,
            dividends=(dividends or {}).get(symbol),
            splits=(splits or {}).get(symbol),
        )
        for symbol, bars in stocks.items()
    }
    return assemble_market(days, frames, membership=membership)


def round_trip_market() -> MarketData:
    """One stock, one trade: buy 100 shares at a raw open of 100, sell them at 110.

    Day 0 close 100. Day 1 opens 105 (a 5% gap up, the signal). Day 2 opens 100
    (the fill: equity 100,000 / 10 positions = 10,000 = 100 shares). Days 3 and
    4 close at 102 and 107. Day 5 opens 110, where holding_days 3 sells.
    """
    return engine_market(
        8,
        {
            "RT.US": {
                0: flat_bar(100.0),
                1: Bar(105.0, 106.0, 104.0, 105.0),
                2: Bar(100.0, 101.0, 99.0, 100.0),
                3: Bar(101.0, 102.0, 100.0, 102.0),
                4: Bar(103.0, 107.0, 102.0, 107.0),
                5: Bar(110.0, 111.0, 109.0, 110.0),
                6: flat_bar(110.0),
                7: flat_bar(110.0),
            }
        },
    )


def gap_up_strategy(
    *,
    holding_days: int | None = None,
    stop_loss_pct: float | None = None,
    profit_target_pct: float | None = None,
    max_positions: int = 10,
    min_gap_pct: float = 4,
) -> StrategyDefinition:
    """A strategy whose only entry rule is a gap up, so a signal is set by typing an open price."""
    exit_rules = {
        key: value
        for key, value in {
            "holding_days": holding_days,
            "stop_loss_pct": stop_loss_pct,
            "profit_target_pct": profit_target_pct,
        }.items()
        if value is not None
    }
    import json

    return strategy_from_json(
        json.dumps(
            {
                "name": "Toy gap up",
                "description_plain_english": "Buy a gap up of a few percent for the engine tests.",
                "entry": [{"block": "gap", "direction": "up", "min_gap_pct": min_gap_pct}],
                "exit": exit_rules,
                "sizing": {"max_positions": max_positions},
            }
        )
    )


def run_toy(
    strategy: StrategyDefinition,
    market: MarketData,
    *,
    first_day: int = 0,
    last_day: int | None = None,
    bps: float = 0.0,
) -> BacktestResult:
    """Run ``run_backtest`` over day numbers ``first_day`` to ``last_day`` of a toy market."""
    days = market.trading_days
    end = days[len(days) - 1 if last_day is None else last_day].date()
    window = BacktestWindow("primary", days[first_day].date(), end, None)
    return run_backtest(strategy, market, window, CostModel(bps_per_side=bps))


def day_of(market: MarketData, number: int) -> date:
    """The calendar date of day number ``number`` in a toy market."""
    return market.trading_days[number].date()


# ---------------------------------------------------------- the fake world


@dataclass
class RawWorld:
    """Raw EODHD rows for a set of tickers, as the client returns them in fake mode."""

    codes: tuple[str, ...]
    membership_rows: list[RawMembershipRow]
    eod: dict[str, list[dict[str, Any]]]
    splits: dict[str, list[dict[str, Any]]]
    dividends: dict[str, list[dict[str, Any]]]
    news: dict[str, list[dict[str, Any]]]
    sentiments: dict[str, list[dict[str, Any]]]
    earnings: dict[str, list[dict[str, Any]]]
    history: MembershipHistory = field(default_factory=lambda: MembershipHistory(spells=()))


def symbol_of(code: str) -> str:
    """``SIM001`` becomes ``SIM001.US``, the name used in membership and in the tables."""
    return f"{code}.US"


@cache
def fake_universe_codes() -> tuple[str, ...]:
    """Every code that was in the fake index at some point from 2015 to 2023, sorted."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EODHD_MODE", "fake")
        patch.setattr(client, "FAKE_CACHE_DIR", Path(tempfile.mkdtemp(prefix="fake-cache-")))
        rows = cast(list[RawMembershipRow], client.get_index_history())
    history = build_membership_history(rows)
    symbols = history.universe(MEMBERSHIP_START, BACKTEST_END)
    return tuple(sorted(symbol.removesuffix(".US") for symbol in symbols))


@cache
def load_raw_world(codes: tuple[str, ...]) -> RawWorld:
    """Load raw fake rows for ``codes`` and SPY through the client wrappers, once per session.

    Uses a throwaway cache folder so the real ``data/cache`` is never touched.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EODHD_MODE", "fake")
        patch.setattr(client, "FAKE_CACHE_DIR", Path(tempfile.mkdtemp(prefix="fake-cache-")))
        membership_rows = cast(list[RawMembershipRow], client.get_index_history())
        every = (*codes, SPY)
        eod = {c: client.get_eod_prices(c, FAKE_START, FAKE_END) for c in every}
        splits = {c: client.get_splits(c) for c in every}
        dividends = {c: client.get_dividends(c, FAKE_START, FAKE_END) for c in every}
        news = {c: client.get_news(c, FAKE_START, FAKE_END) for c in codes}
        sentiments = {c: client.get_daily_sentiment(c, FAKE_START, FAKE_END) for c in codes}
        earnings = {c: client.get_earnings(c, FAKE_START, FAKE_END) for c in codes}
    wanted = {symbol_of(c) for c in codes}
    full = build_membership_history(membership_rows)
    history = MembershipHistory(spells=tuple(s for s in full.spells if s.symbol in wanted))
    return RawWorld(
        codes, membership_rows, eod, splits, dividends, news, sentiments, earnings, history
    )


def _on_or_before(
    rows: Iterable[dict[str, Any]], key: str, cut: date | None
) -> list[dict[str, Any]]:
    """Rows whose ``key`` date is on or before ``cut`` (all rows when ``cut`` is None)."""
    if cut is None:
        return list(rows)
    return [row for row in rows if date.fromisoformat(row[key][:10]) <= cut]


# Volume on the decision day in an open-time rebuild is multiplied by this, so a
# block that read today's volume at the open would see a different number.
SCRAMBLED_VOLUME_FACTOR = 1000.0


def _scramble_after_open(rows: list[dict[str, Any]], day: date) -> list[dict[str, Any]]:
    """Overwrite what is not known at 09:30 on ``day``: its high, low, close and volume.

    The row is kept (``build_price_frame`` drops invalid rows), with high, low
    and close set to the open, which is always a valid bar.
    """
    changed = []
    for row in rows:
        if row["date"] == day.isoformat():
            opened = float(row["open"])
            row = dict(row, high=opened, low=opened, close=opened, adjusted_close=opened)
            row["volume"] = float(row["volume"]) * SCRAMBLED_VOLUME_FACTOR
        changed.append(row)
    return changed


def _price_frame(
    world: RawWorld, code: str, cut: date | None, at_open: bool = False
) -> pd.DataFrame:
    """The canonical price frame for one code from rows dated on or before ``cut``.

    With ``at_open``, the row of ``cut`` keeps only what is known at its open.
    """
    eod_rows = _on_or_before(world.eod[code], "date", cut)
    if at_open and cut is not None:
        eod_rows = _scramble_after_open(eod_rows, cut)
    eod = cast(list[RawEodRow], eod_rows)
    splits = parse_split_rows(
        cast(list[RawSplitRow], _on_or_before(world.splits[code], "date", cut))
    )
    dividends = cast(list[RawDividendRow], _on_or_before(world.dividends[code], "date", cut))
    return build_price_frame(eod, splits, dividends)


def _news_frame(
    world: RawWorld, code: str, days: pd.DatetimeIndex, cut: date | None, at_open: bool = False
) -> pd.DataFrame:
    """Daily news frame. With a cut: articles before 16:00 New York on D, sentiment before D.

    With ``at_open`` the article cut is 09:30 New York on D instead.
    """
    rows = world.news[code]
    sentiment_rows = world.sentiments[code]
    if cut is not None:
        limit = open_datetime(cut) if at_open else decision_datetime(cut)
        rows = [r for r in rows if datetime.fromisoformat(r["date"]) < limit]
        sentiment_rows = [r for r in sentiment_rows if date.fromisoformat(r["date"]) < cut]
    articles = parse_news_rows(cast(list[RawNewsRow], rows), symbol_of(code))
    sentiment = align_daily_sentiment(cast(list[RawDailySentimentRow], sentiment_rows), days)
    return build_news_frame(articles, sentiment, days)


def _earnings_events(
    world: RawWorld, code: str, full_days: pd.DatetimeIndex, cut: date | None, visible: int
) -> list[EarningsEvent]:
    """Events for one code. With a cut: only those reacting within ``visible`` days after D.

    This is the designed exception (D15): the next-earnings date may be seen up
    to ten trading days ahead, and placing an after-market report needs the
    full calendar.
    """
    events = parse_earnings_rows(cast(Any, world.earnings[code]), full_days)
    if cut is None:
        return events
    extended = extend_trading_days(full_days, visible + 3)
    position = int(extended.searchsorted(pd.Timestamp(cut), side="left"))
    last_visible = extended[position + visible].date()
    return [e for e in events if e.earliest_reaction_session <= last_visible]


def build_world_market(
    world: RawWorld,
    *,
    cut: date | None = None,
    earnings_visibility: int = 10,
    at_open: bool = False,
) -> MarketData:
    """Build ``MarketData`` from raw rows, optionally using only what was known at ``cut``.

    ``cut`` None is the full pre-holdout history. A date applies the cuts of
    the strict lookahead test: prices, splits and dividends on or before D,
    news published before 16:00 New York on D, daily sentiment dated before D,
    earnings with a reaction session at most ``earnings_visibility`` trading
    days after D, and the membership table up to D.

    ``at_open`` moves the cut to 09:30 New York on D, for same-open entries:
    D's high, low and close are overwritten with its open, D's volume is
    multiplied by ``SCRAMBLED_VOLUME_FACTOR`` (for every stock and SPY), and
    only news published before 09:30 on D is kept.
    """
    from bot.strategy.blocks import build_market_data

    spy_frame = _price_frame(world, SPY, cut, at_open)
    days = spy_frame.index
    full_days = _price_frame(world, SPY, None).index
    price_frames: dict[str, pd.DataFrame] = {}
    news_frames: dict[str, pd.DataFrame] = {}
    earnings_frames: dict[str, pd.DataFrame] = {}
    for code in world.codes:
        frame = _price_frame(world, code, cut, at_open)
        if frame.empty:
            continue
        symbol = symbol_of(code)
        price_frames[symbol] = frame
        news_frames[symbol] = _news_frame(world, code, days, cut, at_open)
        events = _earnings_events(world, code, full_days, cut, earnings_visibility)
        earnings_frames[symbol] = build_earnings_frame(events, full_days).loc[days]
    return build_market_data(price_frames, news_frames, earnings_frames, spy_frame, world.history)


@cache
def planted_market() -> tuple[RawWorld, MarketData]:
    """The planted tickers plus a few neighbours, over the full pre-holdout history."""
    codes = ("SIM001", "SIM002", "SIM003", "SIM004", "SIM007", "SIM008")
    world = load_raw_world(codes)
    return world, build_world_market(world)


def spell_membership(days: pd.DatetimeIndex, spells: list[MembershipSpell]) -> pd.DataFrame:
    """A bool membership table for a toy market from explicit spells."""
    history = MembershipHistory(spells=tuple(spells))
    from bot.data.membership import membership_matrix

    return membership_matrix(history, days)


def raw_open_on(world: RawWorld, code: str, day: str) -> float:
    """The raw EODHD open of ``code`` on ``day`` (ISO date), for hand checks of ledger prices."""
    for row in world.eod[code]:
        if row["date"] == day:
            return float(row["open"])
    raise KeyError(f"{code} has no row on {day}")


def raw_close_on(world: RawWorld, code: str, day: str) -> float:
    """The raw EODHD close of ``code`` on ``day`` (ISO date)."""
    for row in world.eod[code]:
        if row["date"] == day:
            return float(row["close"])
    raise KeyError(f"{code} has no row on {day}")


__all__ = [
    "PRICE_COLUMNS",
    "Bar",
    "RawWorld",
    "assemble_market",
    "build_world_market",
    "day_of",
    "engine_market",
    "fake_universe_codes",
    "flat_bar",
    "gap_up_strategy",
    "load_raw_world",
    "planted_market",
    "raw_close_on",
    "raw_frame",
    "raw_open_on",
    "round_trip_market",
    "run_toy",
    "spell_membership",
    "symbol_of",
    "toy_days",
]
