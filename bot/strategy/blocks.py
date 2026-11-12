"""Building blocks: one small pure function per entry condition.

Interface stubs for Phase 2. See docs/design/strategy_schema.md (what each
block means) and docs/design/backtest.md (how they are used).

Every block has the same shape:

    block_function(data: DecisionData, params) -> pd.Series[bool]

The Series is indexed by ``data.members`` (the S&P 500 members on the
decision date, sorted) and says, for each stock, whether the condition is true
at the close of the decision date. We return one cross-section per day, not
one bool per stock, because two things need the whole cross-section at once:
the SPY filter (one answer for every stock) and the slot tie-break when more
stocks qualify than there are free slots. The same call works unchanged in
paper trading, which only ever has data up to today.

Rules shared by all blocks:

- A block only sees ``DecisionData``, which holds nothing dated after the
  decision date. ``slice_to_decision_date`` is the only way to build one.
- Missing data gives False, never True. A stock with too little history, a
  NaN value or no news or earnings data at all does not pass.
- Blocks are scale-free: they use returns, ratios, moving averages and RSI of
  one series, never an absolute split-adjusted price level. That is what lets
  the strict lookahead test pass when a split falls between D and 2023.
- Percent values are in percent units: 3 means 3%.

Each block has a matching ``measure_*`` function that returns the raw number
(for example the gap in percent). The block compares it with the threshold.
The measurements are also stored with each trade, so "why did you buy AMD?"
can be answered from numbers Python computed, not from the agent.

The lookahead test (Phase 2):

For a sample of decision dates D and every block type, compute signals two
ways and require them to be equal:

1. Full history: build MarketData from all pre-holdout raw rows, then
   ``slice_to_decision_date(market, D)``.
2. Rebuilt at D: build MarketData again from raw rows cut at D, then slice at D.
   Per data_layer.md (section "Building blocks and the lookahead test"), this
   rebuild is what catches a block that depends on the split anchor.

How each source is cut for the rebuild:

- Prices, splits, dividends: rows dated on or before D.
- News: articles published before ``decision_datetime(D)`` (16:00 New York on
  D), not "dated on or before D".
- Daily sentiment: rows dated before D (a value dated X is first used the
  trading day after X).
- Earnings: by design ``days_to_next`` can see a reaction session up to
  EARNINGS_VISIBILITY_TRADING_DAYS ahead, and ``reaction_session`` needs the
  full trading calendar to place an after-market report. So the rebuild keeps
  events whose reaction session is at most 10 trading days after D and uses the
  full pre-holdout calendar for alignment. This is the one documented place
  where a block sees past D, and the test checks it sees no further.
- Membership: the membership matrix row for D only.

Same-open entries (``entry_timing`` "same_open"):

A same-open strategy decides at 09:30 New York on D and buys at D's open.
``slice_to_open`` builds an ``OpenDecisionData`` for that moment. From row D
it reads only the open, the dividend (the ex-date is known in advance), the
membership row and the open-window news counts (articles from 16:00 on D-1 up
to 09:30 on D). Everything else comes from the data cut at the close of D-1.
So gap is measured at D's open, news up to D's open, and every other block
(n-day return, moving average, RSI, relative volume, earnings, SPY filter) as
of the close of D-1. The lookahead test for this mode rebuilds the data at D
with D's high, low, close and volume overwritten and news after 09:30 removed,
and requires the same signals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from bot.data.membership import MembershipHistory, membership_matrix
from bot.data.prices import total_return_index
from bot.strategy.schema import (
    DaysSinceEarningsCondition,
    DaysToEarningsCondition,
    EntryCondition,
    GapCondition,
    NDayReturnCondition,
    NewsSentimentCondition,
    NoNewsCondition,
    PriceVsMovingAverageCondition,
    RelativeVolumeCondition,
    RsiCondition,
    SpyFilterCondition,
    StrategyDefinition,
    rank_condition,
)

# RSI is computed over a fixed trailing window, so its value does not depend
# on where the loaded history starts (paper trading may load less history
# than the backtest). After 250 days the seed's weight is below 1e-7.
RSI_WINDOW_DAYS = 250

SignalValues = dict[str, float | bool | None]


@dataclass(frozen=True)
class MarketData:
    """Everything the blocks and the engine may read, for all dates.

    Every frame is a date x symbol table indexed by trading date (the SPY
    calendar), tz-naive, sorted, before the holdout. Nothing is forward filled:
    a stock with no row on a day is NaN that day.

    Price tables come from the canonical price frames in bot/data/prices.py:
    ``open``, ``high``, ``low``, ``close`` (split-adjusted), ``raw_close``,
    ``dividend`` (adjusted basis, ex-date row), ``split_factor`` and
    ``total_return_index`` (growth of 1.0 from ``total_return``, per stock).
    News tables come from the daily news frames in bot/data/news.py. The
    ``open_*`` news tables cover only the part of D's window before 09:30 on D,
    for same-open strategies. Earnings
    tables come from the earnings frames in bot/data/earnings.py:
    ``days_to_next_earnings`` counts to the earliest possible reaction session
    and ``days_since_last_earnings`` from the latest possible one (review M2).
    Each errs toward "earnings are close", which is the safe side for both.

    ``volume_by_symbol`` is deliberately not a date x symbol table. EODHD
    volume is on today's share basis, which includes splits after 2023, so it
    is only valid as a ratio within one stock (D14, review M1). Keeping it as
    one Series per stock means no block can compare volume across stocks or
    turn it into dollar volume. Only ``measure_relative_volume`` reads it.

    ``spy`` is
    SPY's own canonical price frame. ``membership`` is the bool table from
    ``membership_matrix``.
    """

    trading_days: pd.DatetimeIndex
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    raw_close: pd.DataFrame
    dividend: pd.DataFrame
    split_factor: pd.DataFrame
    total_return_index: pd.DataFrame
    article_count: pd.DataFrame
    mean_polarity: pd.DataFrame
    open_article_count: pd.DataFrame
    open_mean_polarity: pd.DataFrame
    days_to_next_earnings: pd.DataFrame
    days_since_last_earnings: pd.DataFrame
    spy: pd.DataFrame
    membership: pd.DataFrame
    volume_by_symbol: dict[str, pd.Series]


@dataclass(frozen=True)
class DecisionData:
    """Market data cut at the close of ``decision_date``. All a block or the engine sees that day.

    ``market`` has no row after ``decision_date``. ``members`` are the S&P 500
    members on ``decision_date``, sorted. Every block returns a Series on this
    index.
    """

    decision_date: date
    members: pd.Index
    market: MarketData


@dataclass(frozen=True)
class OpenDecisionData:
    """What is known at 09:30 New York on ``decision_date``, for same-open entries.

    ``previous`` is the data cut at the close of the previous trading day, with
    ``members`` set to today's members. The Series hold today's open and
    dividend (adjusted basis, same anchor as ``previous``) and today's
    open-window news, each on ``members``. Nothing else from today is here.
    """

    decision_date: date
    members: pd.Index
    previous: DecisionData
    open_today: pd.Series
    dividend_today: pd.Series
    open_article_count: pd.Series
    open_mean_polarity: pd.Series


def build_market_data(
    price_frames: dict[str, pd.DataFrame],
    news_frames: dict[str, pd.DataFrame],
    earnings_frames: dict[str, pd.DataFrame],
    spy_frame: pd.DataFrame,
    membership: MembershipHistory,
    *,
    approved: bool = False,
) -> MarketData:
    """Turn per-symbol frames from bot/data into the date x symbol tables of MarketData.

    The calendar is the SPY index. Symbols without news or earnings frames get
    no column in those tables, which the blocks treat as "no data" (False).
    """
    days = pd.DatetimeIndex(spy_frame.index)
    return MarketData(
        trading_days=days,
        open=_panel(price_frames, "open", days),
        high=_panel(price_frames, "high", days),
        low=_panel(price_frames, "low", days),
        close=_panel(price_frames, "close", days),
        raw_close=_panel(price_frames, "raw_close", days),
        dividend=_panel(price_frames, "dividend", days),
        split_factor=_panel(price_frames, "split_factor", days),
        total_return_index=_total_return_table(price_frames, days),
        article_count=_panel(news_frames, "article_count", days),
        mean_polarity=_panel(news_frames, "mean_polarity", days),
        open_article_count=_panel(news_frames, "open_article_count", days),
        open_mean_polarity=_panel(news_frames, "open_mean_polarity", days),
        days_to_next_earnings=_panel(earnings_frames, "days_to_next", days),
        days_since_last_earnings=_panel(earnings_frames, "days_since_last", days),
        spy=spy_frame.reindex(days),
        membership=membership_matrix(membership, days, approved=approved),
        volume_by_symbol={
            symbol: frame["volume"].reindex(days).astype("float64")
            for symbol, frame in price_frames.items()
        },
    )


def _total_return_table(
    price_frames: dict[str, pd.DataFrame], days: pd.DatetimeIndex
) -> pd.DataFrame:
    """Growth of 1.0 per stock from its own ``total_return``, on the SPY calendar."""
    columns = {symbol: total_return_index(price_frames[symbol]) for symbol in sorted(price_frames)}
    table = pd.DataFrame(columns, index=days).astype("float64")
    table.index.name = "date"
    return table


def slice_to_decision_date(market: MarketData, decision_date: date) -> DecisionData:
    """Cut every table in ``market`` to rows dated on or before ``decision_date``.

    This is the bundle version of ``bot.data.prices.slice_to_decision_date``,
    which cuts one frame. Raises ValueError if ``decision_date`` is not a
    trading day in ``market.trading_days``.
    """
    stamp = pd.Timestamp(decision_date)
    position = int(market.trading_days.searchsorted(stamp, side="left"))
    if position >= len(market.trading_days) or market.trading_days[position] != stamp:
        raise ValueError(f"{decision_date} is not a trading day in the market data.")
    keep = position + 1
    cut = MarketData(
        trading_days=market.trading_days[:keep],
        open=market.open.iloc[:keep],
        high=market.high.iloc[:keep],
        low=market.low.iloc[:keep],
        close=market.close.iloc[:keep],
        raw_close=market.raw_close.iloc[:keep],
        dividend=market.dividend.iloc[:keep],
        split_factor=market.split_factor.iloc[:keep],
        total_return_index=market.total_return_index.iloc[:keep],
        article_count=market.article_count.iloc[:keep],
        mean_polarity=market.mean_polarity.iloc[:keep],
        open_article_count=market.open_article_count.iloc[:keep],
        open_mean_polarity=market.open_mean_polarity.iloc[:keep],
        days_to_next_earnings=market.days_to_next_earnings.iloc[:keep],
        days_since_last_earnings=market.days_since_last_earnings.iloc[:keep],
        spy=market.spy.iloc[:keep],
        membership=market.membership.iloc[:keep],
        volume_by_symbol={s: v.iloc[:keep] for s, v in market.volume_by_symbol.items()},
    )
    today = market.membership.iloc[position]
    members = pd.Index(sorted(today.index[today.to_numpy(dtype=bool)]))
    return DecisionData(decision_date=decision_date, members=members, market=cut)


def slice_to_open(market: MarketData, decision_date: date) -> OpenDecisionData:
    """What a same-open decision on ``decision_date`` may see (see OpenDecisionData).

    ``market`` may include rows up to and after ``decision_date``; only the
    open, dividend, membership and open-window news of that day are read, plus
    everything up to the previous close. Raises ValueError if the day is not a
    trading day or is the first one.
    """
    today = slice_to_decision_date(market, decision_date)
    days = today.market.trading_days
    if len(days) < 2:
        raise ValueError(f"{decision_date} has no previous trading day in the market data.")
    previous = slice_to_decision_date(market, days[-2].date())
    members = today.members
    cut = today.market

    def row(frame: pd.DataFrame) -> pd.Series:
        return _row_from_end(frame, members)

    return OpenDecisionData(
        decision_date=decision_date,
        members=members,
        previous=replace(previous, members=members),
        open_today=row(cut.open),
        dividend_today=row(cut.dividend),
        open_article_count=row(cut.open_article_count),
        open_mean_polarity=row(cut.open_mean_polarity),
    )


# ------------------------------------------------------------------ helpers

# Inclusive bounds get a tiny tolerance so that float noise (for example a gap
# of -2.9999999999999996 against a 3% threshold) cannot flip a result.
BOUND_TOLERANCE = 1e-9


def _nan_series(members: pd.Index) -> pd.Series[float]:
    """A float Series of NaN on ``members``: the answer when data is missing."""
    return pd.Series(np.nan, index=members, dtype="float64")


def _row_from_end(frame: pd.DataFrame, members: pd.Index, back: int = 0) -> pd.Series[float]:
    """The row ``back`` rows before the last one, on ``members``. NaN if there is no such row."""
    if len(frame) <= back:
        return _nan_series(members)
    row = frame.iloc[len(frame) - 1 - back]
    return row.reindex(members).astype("float64").rename(None)


def _tail_window(frame: pd.DataFrame, rows: int, members: pd.Index) -> pd.DataFrame | None:
    """The last ``rows`` rows for ``members``, or None when the table is shorter."""
    if len(frame) < rows:
        return None
    return frame.iloc[len(frame) - rows :].reindex(columns=members)


def _at_least(values: pd.Series[float], threshold: float) -> pd.Series[bool]:
    """values >= threshold, with NaN giving False."""
    return (values >= threshold - BOUND_TOLERANCE).astype(bool)


def _at_most(values: pd.Series[float], threshold: float) -> pd.Series[bool]:
    """values <= threshold, with NaN giving False."""
    return (values <= threshold + BOUND_TOLERANCE).astype(bool)


def _compare(values: pd.Series[float], comparison: str, threshold: float) -> pd.Series[bool]:
    """Apply "at_least" or "at_most"."""
    if comparison == "at_least":
        return _at_least(values, threshold)
    return _at_most(values, threshold)


def _panel(frames: dict[str, pd.DataFrame], column: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """One column of many per-symbol frames as a float date x symbol table on ``days``."""
    columns = {symbol: frames[symbol][column] for symbol in sorted(frames)}
    table = pd.DataFrame(columns, index=days).astype("float64")
    table.index.name = "date"
    return table


def _wilder_average(values: np.ndarray, period: int) -> float:
    """Wilder's smoothed average: simple-mean seed, then avg = (avg * (p - 1) + x) / p.

    Written in closed form so no Python loop over days is needed.
    """
    seed = float(values[:period].mean())
    later = values[period:]
    decay = (period - 1) / period
    weights = decay ** np.arange(len(later) - 1, -1, -1)
    return seed * decay ** len(later) + float(later @ weights) / period


def _rsi_from_closes(closes: np.ndarray, period: int) -> float:
    """Wilder's RSI of one stock's closes (oldest first). NaN with fewer than period + 1."""
    changes = np.diff(closes)
    if len(changes) < period:
        return float("nan")
    average_gain = _wilder_average(np.maximum(changes, 0.0), period)
    average_loss = _wilder_average(np.maximum(-changes, 0.0), period)
    if average_loss == 0.0:
        return 50.0 if average_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


def _volume_ratio(series: pd.Series[float] | None, lookback: int) -> float:
    """Last volume over the mean of the ``lookback`` volumes before it. NaN if any is missing."""
    if series is None or len(series) <= lookback:
        return float("nan")
    values = series.to_numpy(dtype="float64")[-(lookback + 1) :]
    if np.isnan(values).any():
        return float("nan")
    baseline = float(values[:-1].mean())
    if baseline <= 0.0:
        return float("nan")
    return float(values[-1]) / baseline


def _members_only(data: DecisionData, symbol: str) -> DecisionData:
    """The same data, with the cross-section reduced to one stock."""
    return replace(data, members=pd.Index([symbol]))


def _clean(value: float) -> float | None:
    """A plain float, or None for NaN so the value stores as JSON."""
    return None if np.isnan(value) else float(value)


def _has_column(table: pd.DataFrame, members: pd.Index) -> pd.Series[bool]:
    """True for members that have a column in ``table``."""
    return pd.Series(members.isin(table.columns), index=members, dtype=bool)


# ------------------------------------------------------------- measurements


def measure_gap_pct(data: DecisionData) -> pd.Series[float]:
    """Today's open versus yesterday's close in percent, with any dividend added back.

    ``(open[D] + dividend[D]) / close[D-1] * 100 - 100``. The open of D is
    known at the close of D, when the decision is made. Adding back the
    dividend stops an ex-dividend day from looking like a gap down. NaN when
    either price is missing.
    """
    market, members = data.market, data.members
    open_today = _row_from_end(market.open, members)
    dividend_today = _row_from_end(market.dividend, members)
    previous_close = _row_from_end(market.close, members, back=1)
    return _gap_pct(open_today, dividend_today, previous_close)


def measure_gap_pct_at_open(data: OpenDecisionData) -> pd.Series[float]:
    """The same gap as ``measure_gap_pct``, from what is known at 09:30 on D."""
    previous_close = _row_from_end(data.previous.market.close, data.members)
    return _gap_pct(data.open_today, data.dividend_today, previous_close)


def _gap_pct(
    open_today: pd.Series[float], dividend_today: pd.Series[float], previous_close: pd.Series[float]
) -> pd.Series[float]:
    """(open + dividend) / previous close, in percent away from 0."""
    return (open_today + dividend_today) / previous_close * 100.0 - 100.0


def measure_n_day_return_pct(data: DecisionData, days: int) -> pd.Series[float]:
    """Total return in percent from the close ``days`` trading days ago to today's close.

    Uses ``total_return_index``, so dividends count. NaN when either end is missing.
    """
    index = data.market.total_return_index
    now = _row_from_end(index, data.members)
    then = _row_from_end(index, data.members, back=days)
    return (now / then - 1.0) * 100.0


def measure_close_vs_ma_pct(data: DecisionData, ma_days: int) -> pd.Series[float]:
    """How far today's close is above (+) or below (-) its simple moving average, in percent.

    The average covers the last ``ma_days`` closes including today. NaN unless
    all ``ma_days`` closes are present.
    """
    window = _tail_window(data.market.close, ma_days, data.members)
    if window is None:
        return _nan_series(data.members)
    average = window.mean().where(window.notna().all())
    return (window.iloc[-1] / average - 1.0) * 100.0


def measure_rsi(data: DecisionData, period: int) -> pd.Series[float]:
    """Wilder's RSI (0 to 100) of the close.

    Formula, over the closes present in the last RSI_WINDOW_DAYS + 1 rows
    ending today (a missing day, such as a halt, is skipped):
    change = close - previous close; gain = max(change, 0); loss = max(-change, 0).
    Seed avg_gain and avg_loss with the simple mean of the first ``period``
    values, then for each later day: avg = (previous avg * (period - 1) +
    today's value) / period. RSI = 100 - 100 / (1 + avg_gain / avg_loss).
    avg_loss 0 with avg_gain > 0 gives 100. Both 0 gives 50. NaN with fewer
    than ``period`` + 1 closes or when today's close is missing. A stock with
    under 251 rows uses all it has, which is the same in a backtest and in
    paper trading, as long as paper trading loads at least 251 days.
    """
    if len(data.market.close) == 0:
        return _nan_series(data.members)
    recent = data.market.close.iloc[-(RSI_WINDOW_DAYS + 1) :].reindex(columns=data.members)
    table = recent.to_numpy(dtype="float64")
    values = []
    for column in range(table.shape[1]):
        closes = table[:, column]
        if np.isnan(closes[-1]):
            values.append(float("nan"))
        else:
            values.append(_rsi_from_closes(closes[~np.isnan(closes)], period))
    return pd.Series(values, index=data.members, dtype="float64")


def measure_relative_volume(data: DecisionData, lookback_days: int) -> pd.Series[float]:
    """Today's volume divided by the mean volume of the previous ``lookback_days`` days.

    Today is excluded from the average, so a spike does not dilute its own
    baseline and "2 times normal volume" means what it says. Each stock's
    ratio is computed from its own ``volume_by_symbol`` series only, so the
    share basis cancels (D14). The result is comparable across stocks because
    it has no units. NaN when any baseline day is missing or the baseline
    mean is 0.
    """
    by_symbol = data.market.volume_by_symbol
    values = [_volume_ratio(by_symbol.get(symbol), lookback_days) for symbol in data.members]
    return pd.Series(values, index=data.members, dtype="float64")


def measure_news_count(data: DecisionData, lookback_sessions: int) -> pd.Series[float]:
    """Articles in the last ``lookback_sessions`` news windows, ending today.

    A window runs from 16:00 New York on the previous trading day up to 16:00
    on its decision day (D15). 1 session means "since the previous close".
    NaN for a stock with no news column.
    """
    window = _tail_window(data.market.article_count, lookback_sessions, data.members)
    if window is None:
        return _nan_series(data.members)
    return window.sum(skipna=False)


def measure_news_polarity(data: DecisionData, lookback_sessions: int) -> pd.Series[float]:
    """Average article polarity (-1 to 1) over the last ``lookback_sessions`` windows.

    Each day's ``mean_polarity`` is weighted by its ``article_count``. This is
    exact unless some articles had no polarity score, in which case it is a
    close approximation. NaN when there are no articles.
    """
    counts = _tail_window(data.market.article_count, lookback_sessions, data.members)
    polarity = _tail_window(data.market.mean_polarity, lookback_sessions, data.members)
    if counts is None or polarity is None:
        return _nan_series(data.members)
    return _weighted_polarity(counts, polarity)


def measure_news_count_at_open(data: OpenDecisionData, lookback_sessions: int) -> pd.Series[float]:
    """Articles from 16:00 New York ``lookback_sessions`` trading days ago up to 09:30 on D.

    That is today's open window plus the ``lookback_sessions`` - 1 full close
    windows before it. 1 session means "from the previous close to today's open".
    """
    window = _open_news_window(data, lookback_sessions, "count")
    if window is None:
        return _nan_series(data.members)
    return window.sum(skipna=False)


def measure_news_polarity_at_open(
    data: OpenDecisionData, lookback_sessions: int
) -> pd.Series[float]:
    """Average polarity over the same windows as ``measure_news_count_at_open``."""
    counts = _open_news_window(data, lookback_sessions, "count")
    polarity = _open_news_window(data, lookback_sessions, "polarity")
    if counts is None or polarity is None:
        return _nan_series(data.members)
    return _weighted_polarity(counts, polarity)


def _open_news_window(
    data: OpenDecisionData, lookback_sessions: int, kind: str
) -> pd.DataFrame | None:
    """Rows of the earlier close windows, then today's open window, for counts or polarity."""
    market = data.previous.market
    table = market.article_count if kind == "count" else market.mean_polarity
    today = data.open_article_count if kind == "count" else data.open_mean_polarity
    earlier = _tail_window(table, lookback_sessions - 1, data.members)
    if earlier is None:
        return None
    rows = [earlier.reset_index(drop=True), today.to_frame().T.reset_index(drop=True)]
    return pd.concat(rows, ignore_index=True).astype("float64")


def _weighted_polarity(counts: pd.DataFrame, polarity: pd.DataFrame) -> pd.Series[float]:
    """Each window's mean polarity weighted by its article count. NaN without articles."""
    weighted = (polarity * counts).sum(min_count=1)
    total = counts.where(polarity.notna()).sum(min_count=1)
    return weighted / total.where(total > 0)


def measure_days_to_next_earnings(data: DecisionData) -> pd.Series[float]:
    """Trading days to the earliest possible next earnings reaction session.

    Earliest: an unknown before/after-market flag counts as the report date,
    so "avoid earnings" avoids it. NaN when no report is visible within
    EARNINGS_VISIBILITY_TRADING_DAYS.
    """
    return _row_from_end(data.market.days_to_next_earnings, data.members)


def measure_days_since_last_earnings(data: DecisionData) -> pd.Series[float]:
    """Trading days since the latest possible last earnings reaction session. 0 means today.

    Latest: an unknown flag counts as the next trading day, so a stock is never
    treated as having reacted before it could have (review M2). NaN if none.
    A future earnings-surprise block (actual vs estimate) must use this
    session too.
    """
    return _row_from_end(data.market.days_since_last_earnings, data.members)


def measure_spy_vs_ma_pct(data: DecisionData, ma_days: int) -> float:
    """How far SPY's close is above (+) or below (-) its ``ma_days`` simple average, in percent."""
    closes = data.market.spy["close"]
    if len(closes) < ma_days:
        return float("nan")
    window = closes.iloc[len(closes) - ma_days :]
    if window.isna().any():
        return float("nan")
    return float((window.iloc[-1] / window.mean() - 1.0) * 100.0)


# ------------------------------------------------------------------- blocks


def gap_block(data: DecisionData, params: GapCondition) -> pd.Series[bool]:
    """Down: gap <= -min_gap_pct. Up: gap >= min_gap_pct. Both ends inclusive."""
    return _gap_passes(measure_gap_pct(data), params)


def _gap_passes(gap: pd.Series[float], params: GapCondition) -> pd.Series[bool]:
    """The gap test, shared by close-time and open-time evaluation."""
    if params.direction == "down":
        return _at_most(gap, -params.min_gap_pct)
    return _at_least(gap, params.min_gap_pct)


def n_day_return_block(data: DecisionData, params: NDayReturnCondition) -> pd.Series[bool]:
    """The ``days``-day total return is at least or at most ``return_pct``."""
    value = measure_n_day_return_pct(data, params.days)
    return _compare(value, params.comparison, params.return_pct)


def price_vs_moving_average_block(
    data: DecisionData, params: PriceVsMovingAverageCondition
) -> pd.Series[bool]:
    """Above: close > average. Below: close < average. Equal passes neither."""
    distance = measure_close_vs_ma_pct(data, params.ma_days)
    if params.position == "above":
        return (distance > 0).astype(bool)
    return (distance < 0).astype(bool)


def rsi_block(data: DecisionData, params: RsiCondition) -> pd.Series[bool]:
    """The RSI is at least or at most ``level``."""
    return _compare(measure_rsi(data, params.period), params.comparison, params.level)


def relative_volume_block(data: DecisionData, params: RelativeVolumeCondition) -> pd.Series[bool]:
    """Relative volume is at least or at most ``ratio``."""
    ratio = measure_relative_volume(data, params.lookback_days)
    return _compare(ratio, params.comparison, params.ratio)


def no_news_block(data: DecisionData, params: NoNewsCondition) -> pd.Series[bool]:
    """True when the article count over the lookback is exactly 0.

    A stock with no news column is False: no data is not the same as no news.
    Before 2021 coverage is thin (D4 update), which backtest_windows handles.
    """
    return (measure_news_count(data, params.lookback_sessions) == 0).astype(bool)


def news_sentiment_block(data: DecisionData, params: NewsSentimentCondition) -> pd.Series[bool]:
    """At least ``min_articles`` articles, and average polarity at least or at most ``polarity``."""
    count = measure_news_count(data, params.lookback_sessions)
    polarity = measure_news_polarity(data, params.lookback_sessions)
    return _sentiment_passes(count, polarity, params)


def _sentiment_passes(
    count: pd.Series[float], polarity: pd.Series[float], params: NewsSentimentCondition
) -> pd.Series[bool]:
    """The sentiment test, shared by close-time and open-time evaluation."""
    enough = (count >= params.min_articles).astype(bool)
    return enough & _compare(polarity, params.comparison, params.polarity)


def days_to_earnings_block(data: DecisionData, params: DaysToEarningsCondition) -> pd.Series[bool]:
    """Within: days to next <= ``days``. Not within: no visible report, or days to next > ``days``.

    A stock with no earnings column is False for both, so "avoid earnings"
    never passes a stock we know nothing about.
    """
    days = measure_days_to_next_earnings(data)
    known = _has_column(data.market.days_to_next_earnings, data.members)
    if params.relation == "within":
        return (days <= params.days).astype(bool)
    return known & (days.isna() | (days > params.days))


def days_since_earnings_block(
    data: DecisionData, params: DaysSinceEarningsCondition
) -> pd.Series[bool]:
    """Within: days since last <= ``days``. Not within: none yet, or days since > ``days``.

    A stock with no earnings column is False for both.
    """
    days = measure_days_since_last_earnings(data)
    known = _has_column(data.market.days_since_last_earnings, data.members)
    if params.relation == "within":
        return (days <= params.days).astype(bool)
    return known & (days.isna() | (days > params.days))


def spy_filter_block(data: DecisionData, params: SpyFilterCondition) -> pd.Series[bool]:
    """The same answer for every member: SPY's close above or below its average.

    False for everyone when SPY lacks ``ma_days`` of history.
    """
    distance = measure_spy_vs_ma_pct(data, params.ma_days)
    if np.isnan(distance):
        passes = False
    elif params.position == "above":
        passes = distance > 0
    else:
        passes = distance < 0
    return pd.Series(passes, index=data.members, dtype=bool)


# --------------------------------------------------------------- combining


def evaluate_condition(data: DecisionData, condition: EntryCondition) -> pd.Series[bool]:
    """Call the block function that matches ``condition.block``."""
    match condition:
        case GapCondition():
            return gap_block(data, condition)
        case NDayReturnCondition():
            return n_day_return_block(data, condition)
        case PriceVsMovingAverageCondition():
            return price_vs_moving_average_block(data, condition)
        case RsiCondition():
            return rsi_block(data, condition)
        case RelativeVolumeCondition():
            return relative_volume_block(data, condition)
        case NoNewsCondition():
            return no_news_block(data, condition)
        case NewsSentimentCondition():
            return news_sentiment_block(data, condition)
        case DaysToEarningsCondition():
            return days_to_earnings_block(data, condition)
        case DaysSinceEarningsCondition():
            return days_since_earnings_block(data, condition)
        case SpyFilterCondition():
            return spy_filter_block(data, condition)
    raise ValueError(f"Unknown entry condition: {condition!r}")


def entry_signals(data: DecisionData, strategy: StrategyDefinition) -> pd.Series[bool]:
    """AND of every entry condition, over ``data.members``. True means "buy at the next open"."""
    signals = pd.Series(True, index=data.members, dtype=bool)
    for condition in strategy.entry:
        signals = signals & evaluate_condition(data, condition)
    return signals


def signal_values(data: DecisionData, strategy: StrategyDefinition, symbol: str) -> SignalValues:
    """The measured value behind each condition for one stock, for example {"gap_pct": -4.2}.

    Stored with each trade so questions such as "why did you buy AMD?" are
    answered from numbers Python computed.
    """
    single = _members_only(data, symbol)
    values: SignalValues = {}
    for condition in strategy.entry:
        values.update(_condition_values(single, condition, symbol))
    return values


def _condition_values(data: DecisionData, condition: EntryCondition, symbol: str) -> SignalValues:
    """The measured number or numbers behind one condition for one stock."""
    match condition:
        case GapCondition():
            return {"gap_pct": _clean(measure_gap_pct(data)[symbol])}
        case NDayReturnCondition():
            key = f"return_{condition.days}d_pct"
            return {key: _clean(measure_n_day_return_pct(data, condition.days)[symbol])}
        case PriceVsMovingAverageCondition():
            key = f"close_vs_ma{condition.ma_days}_pct"
            return {key: _clean(measure_close_vs_ma_pct(data, condition.ma_days)[symbol])}
        case RsiCondition():
            return {f"rsi{condition.period}": _clean(measure_rsi(data, condition.period)[symbol])}
        case RelativeVolumeCondition():
            key = f"relative_volume_{condition.lookback_days}d"
            return {key: _clean(measure_relative_volume(data, condition.lookback_days)[symbol])}
        case NoNewsCondition():
            return _news_values(data, condition.lookback_sessions, symbol, with_polarity=False)
        case NewsSentimentCondition():
            return _news_values(data, condition.lookback_sessions, symbol, with_polarity=True)
        case DaysToEarningsCondition():
            return {"days_to_next_earnings": _clean(measure_days_to_next_earnings(data)[symbol])}
        case DaysSinceEarningsCondition():
            return {"days_since_earnings": _clean(measure_days_since_last_earnings(data)[symbol])}
        case SpyFilterCondition():
            key = f"spy_vs_ma{condition.ma_days}_pct"
            return {key: _clean(measure_spy_vs_ma_pct(data, condition.ma_days))}
    raise ValueError(f"Unknown entry condition: {condition!r}")


def _news_values(
    data: DecisionData, sessions: int, symbol: str, *, with_polarity: bool
) -> SignalValues:
    """Article count, and optionally average polarity, for one stock."""
    values: SignalValues = {
        f"news_count_{sessions}": _clean(measure_news_count(data, sessions)[symbol])
    }
    if with_polarity:
        values[f"news_polarity_{sessions}"] = _clean(measure_news_polarity(data, sessions)[symbol])
    return values


# ------------------------------------------------- same-open evaluation


def evaluate_condition_at_open(
    data: OpenDecisionData, condition: EntryCondition
) -> pd.Series[bool]:
    """One condition for a same-open decision: gap and news at D's open, the rest at D-1's close."""
    match condition:
        case GapCondition():
            return _gap_passes(measure_gap_pct_at_open(data), condition)
        case NoNewsCondition():
            count = measure_news_count_at_open(data, condition.lookback_sessions)
            return (count == 0).astype(bool)
        case NewsSentimentCondition():
            count = measure_news_count_at_open(data, condition.lookback_sessions)
            polarity = measure_news_polarity_at_open(data, condition.lookback_sessions)
            return _sentiment_passes(count, polarity, condition)
        case _:
            return evaluate_condition(data.previous, condition)


def entry_signals_at_open(data: OpenDecisionData, strategy: StrategyDefinition) -> pd.Series[bool]:
    """AND of every entry condition at 09:30 on D. True means "buy at this open"."""
    signals = pd.Series(True, index=data.members, dtype=bool)
    for condition in strategy.entry:
        signals = signals & evaluate_condition_at_open(data, condition)
    return signals


def signal_values_at_open(
    data: OpenDecisionData, strategy: StrategyDefinition, symbol: str
) -> SignalValues:
    """Like ``signal_values``, with the numbers a same-open decision actually used."""
    single = _open_members_only(data, symbol)
    values: SignalValues = {}
    for condition in strategy.entry:
        match condition:
            case GapCondition():
                values["gap_pct"] = _clean(measure_gap_pct_at_open(single)[symbol])
            case NoNewsCondition() | NewsSentimentCondition():
                values.update(_open_news_values(single, condition, symbol))
            case _:
                values.update(_condition_values(single.previous, condition, symbol))
    return values


def _open_news_values(
    data: OpenDecisionData, condition: NoNewsCondition | NewsSentimentCondition, symbol: str
) -> SignalValues:
    """Article count, and for sentiment the average polarity, up to today's open."""
    sessions = condition.lookback_sessions
    count = measure_news_count_at_open(data, sessions)[symbol]
    values: SignalValues = {f"news_count_{sessions}": _clean(count)}
    if isinstance(condition, NewsSentimentCondition):
        polarity = measure_news_polarity_at_open(data, sessions)[symbol]
        values[f"news_polarity_{sessions}"] = _clean(polarity)
    return values


def _open_members_only(data: OpenDecisionData, symbol: str) -> OpenDecisionData:
    """The same open-time data, with the cross-section reduced to one stock."""
    return _open_members_subset(data, pd.Index([symbol]))


# ---------------------------------------------------------------- ranking


def rank_scores(
    data: DecisionData | OpenDecisionData, strategy: StrategyDefinition, symbols: list[str]
) -> pd.Series[float]:
    """The ``rank_by`` number for ``symbols``, turned so that higher is bought first.

    largest_gap uses the size of the gap in either direction. lowest_rsi is
    negated so the lowest RSI scores highest. Open-time data gives the gap and
    sentiment at D's open and the rest at D-1's close, like the signals.
    Raises ValueError when the strategy has no ``rank_by``.
    """
    condition = rank_condition(strategy)
    if condition is None:
        raise ValueError("The strategy has no rank_by.")
    index = pd.Index(symbols)
    if isinstance(data, OpenDecisionData):
        opened = _open_members_subset(data, index)
        close_view = opened.previous
    else:
        opened = None
        close_view = replace(data, members=index)
    match condition:
        case GapCondition():
            gap = measure_gap_pct(close_view) if opened is None else measure_gap_pct_at_open(opened)
            return gap.abs()
        case NewsSentimentCondition():
            sessions = condition.lookback_sessions
            if opened is None:
                return measure_news_polarity(close_view, sessions)
            return measure_news_polarity_at_open(opened, sessions)
        case _:
            return _close_rank_score(close_view, condition)


def _close_rank_score(data: DecisionData, condition: EntryCondition) -> pd.Series[float]:
    """Rank numbers for the blocks that are always read at a close."""
    match condition:
        case NDayReturnCondition():
            return measure_n_day_return_pct(data, condition.days)
        case RelativeVolumeCondition():
            return measure_relative_volume(data, condition.lookback_days)
        case RsiCondition():
            return -measure_rsi(data, condition.period)
    raise ValueError(f"No rank number for block {condition.block!r}.")


def _open_members_subset(data: OpenDecisionData, index: pd.Index) -> OpenDecisionData:
    """The open-time data for several stocks."""
    return OpenDecisionData(
        decision_date=data.decision_date,
        members=index,
        previous=replace(data.previous, members=index),
        open_today=data.open_today.reindex(index),
        dividend_today=data.dividend_today.reindex(index),
        open_article_count=data.open_article_count.reindex(index),
        open_mean_polarity=data.open_mean_polarity.reindex(index),
    )
