"""Earnings dates, aligned to the first market session that can react to them.

Interface stubs for Phase 1. See docs/design/data_layer.md for the reasoning.

``report_date`` is the announcement day. ``date`` in the raw row is the fiscal
period end and is not used for timing.

Reaction sessions (the sessions whose prices can reflect the result). Each
event has an earliest and a latest one:

- ``BeforeMarket``: both are the report_date session.
- ``AfterMarket``: both are the next trading day after report_date.
- Missing or unknown flag: earliest is the report_date session, latest is the
  next trading day after it. We cannot tell which is right.

Use the earliest session for "days to next earnings" and "avoid earnings"
rules, so an unknown report is treated as close. Use the latest session for
"days since last earnings" and for ``actual`` and ``estimate``, so a number is
never used before it could be known.

If report_date is not a trading day, both sessions are the next trading day.

The calendar passed in ends at the last known session. Events whose session
falls after it are kept: the calendar is extended forward with weekdays minus
NYSE holidays (``extend_trading_days``), so "days to next" works on the last
day, which matters for paper trading.

Known limitation: the historical calendar holds the date the report actually
happened. We cannot rebuild when that date was first announced. Companies
usually confirm dates a few weeks ahead, so "days to next earnings" is only
filled in when the next reaction session is within
EARNINGS_VISIBILITY_TRADING_DAYS of the decision date. Beyond that it is
unknown (None or NaN). The horizon is provisional.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd

from bot.data.holdout import drop_holdout_rows

# How far ahead a scheduled report date counts as known. Provisional.
EARNINGS_VISIBILITY_TRADING_DAYS = 10

MarketTiming = Literal["BeforeMarket", "AfterMarket", "Unknown"]


class RawEarningsRow(TypedDict, total=False):
    """One row of ``earnings`` in ``GET /api/calendar/earnings``."""

    code: str
    report_date: str
    date: str
    before_after_market: str | None
    currency: str | None
    actual: float | None
    estimate: float | None
    difference: float | None
    percent: float | None


@dataclass(frozen=True)
class EarningsEvent:
    """One earnings report for one symbol.

    ``earliest_reaction_session`` is for "next earnings" and avoid rules. For
    unknown timing it is the report date. ``latest_reaction_session`` is for
    "days since last" and ``actual`` / ``estimate``, which may only be used from
    that session on. For unknown timing it is the next trading day. For known
    timing the two are equal.
    """

    symbol: str
    report_date: date
    timing: MarketTiming
    earliest_reaction_session: date
    latest_reaction_session: date
    fiscal_period_end: date | None
    actual: float | None
    estimate: float | None

    @property
    def reaction_session(self) -> date:
        """Deprecated alias of ``latest_reaction_session``. Use the explicit fields."""
        return self.latest_reaction_session


def parse_timing(value: str | None) -> MarketTiming:
    """Map the raw ``before_after_market`` value to a MarketTiming. Unrecognized gives Unknown."""
    if value == "BeforeMarket":
        return "BeforeMarket"
    if value == "AfterMarket":
        return "AfterMarket"
    return "Unknown"


def reaction_sessions(
    report_date: date, timing: MarketTiming, trading_days: pd.DatetimeIndex
) -> tuple[date, date]:
    """The (earliest, latest) trading session that can react to the report.

    See the module docstring. ``trading_days`` must reach past both sessions,
    so extend it first with ``extend_trading_days``. Raises ValueError if not.
    """
    stamp = pd.Timestamp(report_date)
    on_or_after = int(trading_days.searchsorted(stamp, side="left"))
    strictly_after = int(trading_days.searchsorted(stamp, side="right"))
    earliest_position = strictly_after if timing == "AfterMarket" else on_or_after
    latest_position = on_or_after if timing == "BeforeMarket" else strictly_after
    if max(earliest_position, latest_position) >= len(trading_days):
        raise ValueError(f"No trading day after {report_date} in the calendar.")
    return trading_days[earliest_position].date(), trading_days[latest_position].date()


def reaction_session(
    report_date: date, timing: MarketTiming, trading_days: pd.DatetimeIndex
) -> date:
    """Deprecated: the LATEST reaction session. Use ``reaction_sessions``.

    Kept so older callers keep working. It is the safe choice for ``actual``
    and ``estimate`` but too late for "avoid earnings" rules.
    """
    return reaction_sessions(report_date, timing, trading_days)[1]


def nyse_holidays(year: int) -> set[date]:
    """NYSE full-day closures for ``year``: the standard rules, not special closures.

    New Year's Day, Martin Luther King Day (3rd Monday of January),
    Presidents Day (3rd Monday of February), Good Friday, Memorial Day (last
    Monday of May), Juneteenth (from 2022), Independence Day, Labor Day (1st
    Monday of September), Thanksgiving (4th Thursday of November) and
    Christmas. A holiday on Saturday is observed the Friday before and one on
    Sunday the Monday after, except New Year's Day on a Saturday (not
    observed). One-off closures such as a day of national mourning are not
    included. The list only extends the calendar a few days past the last
    known session.
    """
    fixed = [date(year, 7, 4), date(year, 12, 25)]
    if year >= 2022:
        fixed.append(date(year, 6, 19))
    days = {_observed(day) for day in fixed}
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        days.add(_observed(new_year))
    days |= {
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_monday(year, 5),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
    }
    return days


def extend_trading_days(trading_days: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Append ``n`` future sessions: weekdays after the last day, minus NYSE holidays.

    Keep ``n`` small (a few weeks). Holidays are covered for three years.
    """
    if n <= 0 or len(trading_days) == 0:
        return trading_days
    last = trading_days[-1].date()
    extra: list[pd.Timestamp] = []
    day = last
    holidays: set[date] = set()
    for year in range(last.year, last.year + 3):
        holidays |= nyse_holidays(year)
    while len(extra) < n:
        day += timedelta(days=1)
        if day.weekday() < 5 and day not in holidays:
            extra.append(pd.Timestamp(day))
    extended = pd.DatetimeIndex(trading_days.append(pd.DatetimeIndex(extra)))
    extended.name = trading_days.name
    return extended


def parse_earnings_rows(
    rows: list[RawEarningsRow],
    trading_days: pd.DatetimeIndex,
    *,
    approved: bool = False,
) -> list[EarningsEvent]:
    """Turn raw rows into EarningsEvents sorted by earliest reaction session.

    Rows with a report_date on or after HOLDOUT_START are dropped by the holdout
    guard unless ``approved`` (D12). A session after the last day of
    ``trading_days`` is found on a calendar extended by weekdays minus NYSE
    holidays. An event still beyond that is dropped: it is further out than the
    visibility horizon, so no decision can use it.
    """
    events: list[EarningsEvent] = []
    kept = cast(
        list[RawEarningsRow],
        drop_holdout_rows(rows, "report_date", approved=approved),  # type: ignore[arg-type]
    )
    extended = extend_trading_days(trading_days, EARNINGS_VISIBILITY_TRADING_DAYS + 3)
    for row in kept:
        report_date = date.fromisoformat(row["report_date"])
        timing = parse_timing(row.get("before_after_market"))
        try:
            sessions = reaction_sessions(report_date, timing, extended)
        except ValueError:
            continue
        events.append(_event_from_row(row, report_date, timing, sessions))
    return sorted(events, key=lambda event: (event.earliest_reaction_session, event.report_date))


def days_to_next_earnings(
    events: list[EarningsEvent], decision_date: date, trading_days: pd.DatetimeIndex
) -> int | None:
    """Trading days from ``decision_date`` to the next possible reaction session after it.

    Uses the earliest session, and for an unknown-timing report also its latest
    session, so a report dated today with unknown timing still counts as
    tomorrow. 0 is never returned: a session on the decision date has already
    happened by the close. Returns None when the next session is more than
    EARNINGS_VISIBILITY_TRADING_DAYS away or there is no next event. Sessions
    past the end of ``trading_days`` are found on an extended calendar.
    """
    extended = _extended(trading_days)
    today = _position_of(decision_date, trading_days)
    positions = _next_positions(events, extended)
    later = positions[positions > today]
    if len(later) == 0:
        return None
    distance = int(later[0]) - today
    if distance > EARNINGS_VISIBILITY_TRADING_DAYS:
        return None
    return distance


def days_since_last_earnings(
    events: list[EarningsEvent], decision_date: date, trading_days: pd.DatetimeIndex
) -> int | None:
    """Trading days from the last latest-reaction session on or before ``decision_date``.

    0 means that session is the decision date. None if there is none yet.
    """
    extended = _extended(trading_days)
    today = _position_of(decision_date, trading_days)
    positions = _positions([event.latest_reaction_session for event in events], extended)
    earlier = positions[positions <= today]
    if len(earlier) == 0:
        return None
    return today - int(earlier[-1])


def build_earnings_frame(
    events: list[EarningsEvent], trading_days: pd.DatetimeIndex
) -> pd.DataFrame:
    """A daily frame for one symbol, indexed by ``trading_days`` (named ``date``).

    Columns: ``days_to_next`` (float64, NaN when unknown; from the earliest
    reaction session) and ``days_since_last`` (float64, NaN when none yet; from
    the latest). Float so NaN fits.
    """
    extended = _extended(trading_days)
    next_positions = _next_positions(events, extended)
    last_positions = _positions([e.latest_reaction_session for e in events], extended)
    days_to_next = np.full(len(trading_days), np.nan)
    days_since_last = np.full(len(trading_days), np.nan)
    for today in range(len(trading_days)):
        after = int(np.searchsorted(next_positions, today, side="right"))
        if after < len(next_positions):
            distance = int(next_positions[after]) - today
            if distance <= EARNINGS_VISIBILITY_TRADING_DAYS:
                days_to_next[today] = distance
        done = int(np.searchsorted(last_positions, today, side="right"))
        if done > 0:
            days_since_last[today] = today - int(last_positions[done - 1])
    frame = pd.DataFrame(
        {"days_to_next": days_to_next, "days_since_last": days_since_last}, index=trading_days
    )
    frame.index.name = "date"
    return frame


def _event_from_row(
    row: RawEarningsRow, report_date: date, timing: MarketTiming, sessions: tuple[date, date]
) -> EarningsEvent:
    """Build one EarningsEvent. The fiscal period end is kept but never used for timing."""
    period_end = row.get("date")
    return EarningsEvent(
        symbol=row.get("code", ""),
        report_date=report_date,
        timing=timing,
        earliest_reaction_session=sessions[0],
        latest_reaction_session=sessions[1],
        fiscal_period_end=date.fromisoformat(period_end) if period_end else None,
        actual=row.get("actual"),
        estimate=row.get("estimate"),
    )


def _extended(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The calendar plus enough future sessions to place events inside the visibility horizon."""
    return extend_trading_days(trading_days, EARNINGS_VISIBILITY_TRADING_DAYS + 3)


def _position_of(day: date, trading_days: pd.DatetimeIndex) -> int:
    """Position of ``day`` in the calendar. Raises ValueError if it is not a trading day."""
    stamp = pd.Timestamp(day)
    position = int(trading_days.searchsorted(stamp, side="left"))
    if position >= len(trading_days) or trading_days[position] != stamp:
        raise ValueError(f"{day} is not a trading day.")
    return position


def _positions(sessions: list[date], trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Sorted calendar positions of the given sessions."""
    positions = [int(trading_days.searchsorted(pd.Timestamp(day), side="left")) for day in sessions]
    return np.array(sorted(positions), dtype="int64")


def _next_positions(events: list[EarningsEvent], trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Sorted positions of every session that could be the next reaction.

    Both the earliest and latest session of each event, so an unknown-timing
    report dated today still shows up as tomorrow.
    """
    days = [e.earliest_reaction_session for e in events]
    days += [e.latest_reaction_session for e in events]
    return np.unique(_positions(days, trading_days))


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    j, k = c // 4, c % 4
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * j - k - h + m + 32) % 7
    month = (h - m + r + 90) // 25
    return date(year, month, (h - m + r + month + 19) % 32)


def _observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday (Monday is 0) of a month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_monday(year: int, month: int) -> date:
    """The last Monday of a month."""
    following = date(year + (month == 12), month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=last.weekday())
