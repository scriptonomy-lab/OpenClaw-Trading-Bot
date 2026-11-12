"""Daily prices: split-adjusted OHLC, volume and a total-return series.

Interface stubs for Phase 1. See docs/design/data_layer.md for the reasoning.

Canonical daily price frame, one per symbol:

- Index: ``pd.DatetimeIndex`` named ``date``, tz-naive, one row per trading
  day, sorted ascending, no duplicates. All dates are before HOLDOUT_START
  unless an approved holdout session is open.
- ``open``, ``high``, ``low``, ``close`` (float64): split-adjusted by us,
  anchored on the last split before HOLDOUT_START.
- ``raw_close`` (float64): EODHD's unadjusted close. Only for absolute price
  filters such as "price above $5" and for checks.
- ``volume`` (float64): EODHD volume as delivered. It is already
  split-adjusted, so we never adjust it again. Its basis is "today", which can
  include splits after 2023, so only use it in ratios within one series.
- ``dividend`` (float64): cash per share with its ex-date on this row, on the
  same split basis as ``close``. 0.0 on other days.
- ``split_factor`` (float64): factor of a split whose ex-date is this row,
  1.0 on other days.
- ``total_return`` (float64): daily return including dividends,
  ``(close + dividend) / previous close - 1``. NaN on the first row.

EODHD's ``adjusted_close`` is not in the frame. It is computed as of today and
so carries splits and dividends from the holdout years.
"""

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict

import pandas as pd

from bot.data.holdout import assert_before_holdout

PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "raw_close",
    "volume",
    "dividend",
    "split_factor",
    "total_return",
)

# Split ratios we accept when inferring a split from the adjusted/close ratio.
COMMON_SPLIT_FACTORS: tuple[float, ...] = (
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    20.0,
    1 / 2,
    1 / 3,
    1 / 4,
    1 / 5,
    1 / 8,
    1 / 10,
    1 / 15,
    1 / 20,
)


class RawEodRow(TypedDict):
    """One row of ``GET /api/eod/{SYMBOL}``. Prices are raw, volume is split-adjusted."""

    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float


class RawSplitRow(TypedDict):
    """One row of ``GET /api/splits/{SYMBOL}``. ``split`` is "new/old", e.g. "4.000000/1.000000"."""

    date: str
    split: str


class RawDividendRow(TypedDict, total=False):
    """One row of ``GET /api/div/{SYMBOL}``. We use ``unadjustedValue``, not ``value``."""

    date: str
    declarationDate: str | None
    recordDate: str | None
    paymentDate: str | None
    period: str | None
    value: float
    unadjustedValue: float
    currency: str


@dataclass(frozen=True)
class SplitEvent:
    """A split taking effect on ``ex_date``.

    ``factor`` is new shares per old share. A 4-for-1 split has factor 4.0, a
    1-for-10 reverse split has factor 0.1. Rows strictly before ``ex_date`` are
    divided by the factor. The ex-date row itself is already on the new basis.
    """

    ex_date: date
    factor: float
    source: Literal["eodhd", "inferred"]


def parse_split_ratio(split: str) -> float:
    """Turn an EODHD "new/old" string into a factor: "4.000000/1.000000" gives 4.0.

    Raises ValueError for a malformed string or a non-positive part.
    """
    parts = split.split("/")
    if len(parts) != 2:
        raise ValueError(f"Split must look like 'new/old', got {split!r}.")
    new_shares = float(parts[0])
    old_shares = float(parts[1])
    if new_shares <= 0 or old_shares <= 0:
        raise ValueError(f"Split parts must be positive, got {split!r}.")
    return new_shares / old_shares


def parse_split_rows(rows: list[RawSplitRow]) -> list[SplitEvent]:
    """Turn splits endpoint rows into SplitEvents, sorted by ex-date."""
    events = [
        SplitEvent(
            ex_date=date.fromisoformat(row["date"]),
            factor=parse_split_ratio(row["split"]),
            source="eodhd",
        )
        for row in rows
    ]
    return sorted(events, key=lambda event: event.ex_date)


def infer_splits_from_adjusted_ratio(
    rows: list[RawEodRow], tolerance: float = 0.05
) -> list[SplitEvent]:
    """Find splits from jumps in ``adjusted_close / close`` (D3).

    Dividends move this ratio by a small step. A split moves it by the split
    factor between two consecutive rows. A jump counts as a split only if it is
    within ``tolerance`` (relative) of a value in COMMON_SPLIT_FACTORS. Only
    consecutive pre-holdout rows are compared, so a constant post-2023 factor
    in ``adjusted_close`` has no effect.
    """
    ordered = sorted(rows, key=lambda row: row["date"])
    found: list[SplitEvent] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous["close"] <= 0 or current["close"] <= 0:
            continue
        previous_ratio = previous["adjusted_close"] / previous["close"]
        current_ratio = current["adjusted_close"] / current["close"]
        if previous_ratio <= 0 or current_ratio <= 0:
            continue
        factor = _match_common_factor(current_ratio / previous_ratio, tolerance)
        if factor is not None:
            ex_date = date.fromisoformat(current["date"])
            found.append(SplitEvent(ex_date=ex_date, factor=factor, source="inferred"))
    return found


def merge_split_sources(
    reported: list[SplitEvent],
    inferred: list[SplitEvent],
    trading_days: pd.DatetimeIndex,
    max_gap_trading_days: int = 2,
) -> list[SplitEvent]:
    """Combine reported and inferred splits.

    A reported split always wins. An inferred split is added only when no
    reported split lies within ``max_gap_trading_days`` trading days of it.
    The gap is counted in positions of ``trading_days`` (the price row dates),
    not calendar days: a reported split on Thursday and an inferred one on the
    next Monday are 2 trading days apart and are one event.
    """
    merged = list(reported)
    for candidate in inferred:
        spot = _trading_position(trading_days, candidate.ex_date)
        gaps = [abs(spot - _trading_position(trading_days, e.ex_date)) for e in reported]
        if not gaps or min(gaps) > max_gap_trading_days:
            merged.append(candidate)
    return sorted(merged, key=lambda event: event.ex_date)


def _trading_position(trading_days: pd.DatetimeIndex, day: date) -> int:
    """Position of the first trading day on or after ``day`` (len if none)."""
    return int(trading_days.searchsorted(pd.Timestamp(day), side="left"))


def cumulative_split_factors(
    dates: pd.DatetimeIndex, splits: list[SplitEvent], *, approved: bool = False
) -> pd.Series:
    """For each date, the product of factors of all splits with ex-date after it.

    Dividing a raw price by this value puts it on the basis of the latest split
    in ``splits``. Dates on or after the last ex-date get 1.0. A split dated on
    or after HOLDOUT_START raises HoldoutAccessError unless approved (L5).
    """
    for event in splits:
        assert_before_holdout(event.ex_date, approved=approved)
    factors = pd.Series(1.0, index=dates, dtype="float64")
    for event in splits:
        # Rows strictly before the ex-date are on the old share basis.
        factors[dates < pd.Timestamp(event.ex_date)] *= event.factor
    return factors


def invalid_price_rows(rows: list[RawEodRow]) -> list[date]:
    """Dates of rows with a bad raw price: any of open, high, low, close at or below
    zero or not a number, or a high below the low (review M1).

    ``build_price_frame`` drops these rows, so they count as missing days. The
    data-quality report lists them.
    """
    bad: list[date] = []
    for row in sorted(rows, key=lambda r: r["date"]):
        if not _row_has_valid_prices(row):
            bad.append(date.fromisoformat(row["date"]))
    return bad


def _row_has_valid_prices(row: RawEodRow) -> bool:
    """True when open, high, low and close are all above zero and high is not below low."""
    try:
        values = [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])]
    except (TypeError, ValueError, KeyError):
        return False
    if any(math.isnan(value) or value <= 0.0 for value in values):
        return False
    return values[1] >= values[2]


def build_price_frame(
    eod_rows: list[RawEodRow],
    splits: list[SplitEvent],
    dividends: list[RawDividendRow],
    *,
    approved: bool = False,
) -> pd.DataFrame:
    """Build the canonical daily price frame (see module docstring) for one symbol.

    Divides raw open, high, low and close by the cumulative split factor. Leaves
    volume untouched. Divides ``unadjustedValue`` by the same factor so prices
    and dividends share one basis. Calls the holdout guard on the last date and
    on every split date. ``approved`` is passed to the guard (D12).

    Rows with a zero, negative or NaN price, or a high below the low, are
    dropped (review M1). To every later step they are days with no row, so a
    bad tick can never fill an order or fire a stop. The holdout guard still
    looks at every input row. ``invalid_price_rows`` lists the dropped dates.
    """
    if not eod_rows:
        return _empty_price_frame()
    assert_before_holdout(
        max(date.fromisoformat(row["date"]) for row in eod_rows), approved=approved
    )
    valid_rows = [row for row in eod_rows if _row_has_valid_prices(row)]
    if not valid_rows:
        return _empty_price_frame()
    ordered = sorted(valid_rows, key=lambda row: row["date"])
    index = pd.DatetimeIndex([pd.Timestamp(row["date"]) for row in ordered], name="date")
    cumulative = cumulative_split_factors(index, splits, approved=approved)
    frame = pd.DataFrame(index=index)
    for column in ("open", "high", "low", "close"):
        raw_values = pd.Series([float(row[column]) for row in ordered], index=index)
        frame[column] = raw_values / cumulative
    frame["raw_close"] = [float(row["close"]) for row in ordered]
    frame["volume"] = [float(row["volume"]) for row in ordered]
    frame["dividend"] = _dividend_column(dividends, index, cumulative)
    frame["split_factor"] = _split_factor_column(splits, index)
    previous_close = frame["close"].shift(1)
    frame["total_return"] = (frame["close"] + frame["dividend"]) / previous_close - 1.0
    return frame[list(PRICE_COLUMNS)].astype("float64")


def large_unexplained_moves(frame: pd.DataFrame, max_gap: float = 0.4) -> list[date]:
    """Return every date where the adjusted close moves by more than ``max_gap``.

    This lists all large one-day moves in the split-adjusted close, not only
    ex-dates. A leftover jump usually means a missing or wrong split, but a real
    crash or spike also shows up. A person reads the list. Used by the
    data-quality report, not by strategies.
    """
    if len(frame) < 2:
        return []
    moves = frame["close"].pct_change().abs()
    flagged = moves[moves > max_gap]
    return [timestamp.date() for timestamp in flagged.index]


# Deprecated name from before L4. Kept until tests/test_fake_eodhd.py uses the new one.
check_splits_against_prices = large_unexplained_moves


def dividends_outside_frame(dividends: list[RawDividendRow], index: pd.DatetimeIndex) -> int:
    """Count dividends that ``build_price_frame`` drops: before the first row or after the last."""
    if len(index) == 0:
        return len(dividends)
    first, last = index[0].date(), index[-1].date()
    return sum(1 for row in dividends if not first <= date.fromisoformat(row["date"]) <= last)


def relative_volume(frame: pd.DataFrame, lookback: int) -> pd.Series:
    """Today's volume divided by the mean of the previous ``lookback`` days.

    Today is excluded from the mean. NaN until ``lookback`` earlier rows exist.
    A ratio inside one stock is safe because EODHD volume is on today's share
    basis, a constant factor per stock (D14). Never compare raw volume across
    stocks.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    previous_mean = frame["volume"].shift(1).rolling(lookback, min_periods=lookback).mean()
    return (frame["volume"] / previous_mean).rename("relative_volume")


def total_return_index(frame: pd.DataFrame) -> pd.Series:
    """Cumulative growth of 1.0 from ``total_return``. The first row is 1.0."""
    return (1.0 + frame["total_return"].fillna(0.0)).cumprod()


def slice_to_decision_date(frame: pd.DataFrame, decision_date: date) -> pd.DataFrame:
    """Return the rows dated on or before ``decision_date``.

    Building blocks only ever receive frames cut this way.
    """
    cutoff = pd.Timestamp(decision_date)
    return frame.loc[frame.index <= cutoff]


def wide_panel(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """One column from many symbol frames as a date x symbol table.

    Days where a symbol has no row are NaN. Nothing is forward filled. The
    ``volume`` column is refused: it is on today's share basis, which differs
    across stocks by splits after 2023 (D14). Use ``relative_volume`` per stock.
    """
    if column == "volume":
        raise ValueError(
            "wide_panel refuses 'volume': it is not comparable across stocks (D14). "
            "Use relative_volume(frame, lookback) per stock instead."
        )
    if not frames:
        return pd.DataFrame()
    panel = pd.DataFrame({symbol: frame[column] for symbol, frame in frames.items()})
    panel.index.name = "date"
    return panel.sort_index()


def _match_common_factor(jump: float, tolerance: float) -> float | None:
    """Return the closest common split factor within ``tolerance`` of ``jump``, else None."""
    best: float | None = None
    best_error = tolerance
    for factor in COMMON_SPLIT_FACTORS:
        error = abs(jump / factor - 1.0)
        if error <= best_error:
            best = factor
            best_error = error
    return best


def _empty_price_frame() -> pd.DataFrame:
    """A frame with the canonical columns and no rows."""
    index = pd.DatetimeIndex([], name="date")
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in PRICE_COLUMNS}, index=index)


def _first_row_on_or_after(index: pd.DatetimeIndex, day: date) -> int:
    """Position of the first row dated on or after ``day``. Equals len(index) if none."""
    return int(index.searchsorted(pd.Timestamp(day), side="left"))


def _dividend_column(
    dividends: list[RawDividendRow], index: pd.DatetimeIndex, cumulative: pd.Series
) -> pd.Series:
    """Dividends per row on the split basis of ``close``. 0.0 where none.

    Uses ``unadjustedValue``. A dividend dated on a non-trading day moves to the
    next row. One outside the date range of the rows is ignored.
    """
    totals = [0.0] * len(index)
    for row in dividends:
        ex_date = date.fromisoformat(row["date"])
        position = _first_row_on_or_after(index, ex_date)
        # Dividends before the first row or after the last row are outside this frame.
        if ex_date >= index[0].date() and position < len(index):
            totals[position] += float(row["unadjustedValue"]) / float(cumulative.iloc[position])
    return pd.Series(totals, index=index, dtype="float64")


def _split_factor_column(splits: list[SplitEvent], index: pd.DatetimeIndex) -> pd.Series:
    """The split factor on the row of each ex-date (the next row if it is not a trading day)."""
    values = [1.0] * len(index)
    for event in splits:
        position = _first_row_on_or_after(index, event.ex_date)
        # A split before the first row or after the last row has no row to mark.
        if 0 < position < len(index) or (position == 0 and index[0].date() == event.ex_date):
            values[position] *= event.factor
    return pd.Series(values, index=index, dtype="float64")
