"""Earnings timing tests."""

import json
from datetime import date
from typing import Any

import pandas as pd

from bot.data.earnings import (
    EARNINGS_VISIBILITY_TRADING_DAYS,
    EarningsEvent,
    build_earnings_frame,
    days_since_last_earnings,
    days_to_next_earnings,
    parse_earnings_rows,
    parse_timing,
    reaction_sessions,
)

# Weekdays only. Wed 2023-01-04, Fri 2023-01-06, Mon 2023-01-09 are all sessions.
DAYS = pd.bdate_range("2023-01-02", "2023-03-31")


def event(reaction: date, report: date | None = None) -> EarningsEvent:
    return EarningsEvent(
        "AAPL.US", report or reaction, "BeforeMarket", reaction, reaction, None, None, None
    )


def test_visibility_horizon_is_ten() -> None:
    assert EARNINGS_VISIBILITY_TRADING_DAYS == 10


def test_parse_timing() -> None:
    assert parse_timing("BeforeMarket") == "BeforeMarket"
    assert parse_timing("AfterMarket") == "AfterMarket"
    assert parse_timing(None) == "Unknown"
    assert parse_timing("") == "Unknown"
    assert parse_timing("SomethingElse") == "Unknown"


def test_reaction_sessions_before_after_unknown() -> None:
    wed = date(2023, 1, 4)
    assert reaction_sessions(wed, "BeforeMarket", DAYS) == (wed, wed)
    assert reaction_sessions(wed, "AfterMarket", DAYS) == (date(2023, 1, 5), date(2023, 1, 5))
    # Unknown: earliest is the report day, latest the next trading day.
    assert reaction_sessions(wed, "Unknown", DAYS) == (wed, date(2023, 1, 5))


def test_reaction_sessions_after_market_friday_is_monday() -> None:
    monday = date(2023, 1, 9)
    assert reaction_sessions(date(2023, 1, 6), "AfterMarket", DAYS) == (monday, monday)
    assert reaction_sessions(date(2023, 1, 6), "Unknown", DAYS) == (date(2023, 1, 6), monday)


def test_reaction_sessions_on_non_trading_day_is_next_trading_day() -> None:
    saturday = date(2023, 1, 7)
    monday = date(2023, 1, 9)
    assert reaction_sessions(saturday, "BeforeMarket", DAYS) == (monday, monday)
    assert reaction_sessions(saturday, "Unknown", DAYS) == (monday, monday)


def test_parse_earnings_rows_from_fixture(fixtures: Any) -> None:
    rows = fixtures.earnings()
    days = pd.bdate_range("2018-12-31", "2023-12-29")
    events = parse_earnings_rows(rows, days)
    assert len(events) == len(rows)
    assert [e.earliest_reaction_session for e in events] == sorted(
        e.earliest_reaction_session for e in events
    )
    first = events[0]
    # Reported 2019-01-29 after the close, so the first reaction is the 30th.
    assert first.symbol == "AAPL.US"
    assert first.report_date == date(2019, 1, 29)
    assert first.timing == "AfterMarket"
    assert first.earliest_reaction_session == date(2019, 1, 30)
    assert first.latest_reaction_session == date(2019, 1, 30)
    assert first.fiscal_period_end == date(2018, 12, 31)
    assert first.actual == 1.05
    assert first.estimate == 1.04


def test_parse_earnings_rows_drops_holdout_report_dates() -> None:
    rows: Any = [
        {"code": "AAPL.US", "report_date": "2023-10-30", "date": "2023-09-30"},
        {"code": "AAPL.US", "report_date": "2024-02-01", "date": "2023-12-31"},
    ]
    days = pd.bdate_range("2023-10-02", "2023-12-29")
    events = parse_earnings_rows(rows, days)
    assert [e.report_date for e in events] == [date(2023, 10, 30)]
    assert events[0].timing == "Unknown"
    assert events[0].earliest_reaction_session == date(2023, 10, 30)
    assert events[0].latest_reaction_session == date(2023, 10, 31)
    assert json.dumps(rows)  # rows are plain data and untouched


def test_days_to_next_counts_trading_days() -> None:
    events = [event(date(2023, 1, 20))]
    assert days_to_next_earnings(events, date(2023, 1, 18), DAYS) == 2
    assert days_to_next_earnings(events, date(2023, 1, 19), DAYS) == 1


def test_days_to_next_is_never_zero() -> None:
    events = [event(date(2023, 1, 20))]
    got = days_to_next_earnings(events, date(2023, 1, 20), DAYS)
    assert got != 0
    assert got is None  # no later event is known


def test_days_to_next_after_market_report_on_decision_day_is_one() -> None:
    # Reported Jan 10 after the close, so the reaction session is Jan 11.
    events = [
        EarningsEvent(
            "AAPL.US",
            date(2023, 1, 10),
            "AfterMarket",
            date(2023, 1, 11),
            date(2023, 1, 11),
            None,
            None,
            None,
        )
    ]
    assert days_to_next_earnings(events, date(2023, 1, 10), DAYS) == 1


def test_days_to_next_only_within_ten_trading_days() -> None:
    events = [event(date(2023, 1, 20))]
    # Jan 6 to Jan 20 is exactly 10 trading days. Jan 5 is 11.
    assert days_to_next_earnings(events, date(2023, 1, 6), DAYS) == 10
    assert days_to_next_earnings(events, date(2023, 1, 5), DAYS) is None


def test_days_to_next_uses_nearest_future_event() -> None:
    events = [event(date(2023, 1, 12)), event(date(2023, 1, 20))]
    assert days_to_next_earnings(events, date(2023, 1, 11), DAYS) == 1
    assert days_to_next_earnings(events, date(2023, 1, 12), DAYS) == 6  # Jan 20 is 6 days on


def test_days_since_last() -> None:
    events = [event(date(2023, 1, 20))]
    assert days_since_last_earnings(events, date(2023, 1, 20), DAYS) == 0
    assert days_since_last_earnings(events, date(2023, 1, 25), DAYS) == 3
    assert days_since_last_earnings(events, date(2023, 1, 19), DAYS) is None


def test_days_since_last_ignores_event_not_yet_reacted() -> None:
    events = [
        event(date(2023, 1, 5)),
        EarningsEvent(
            "AAPL.US",
            date(2023, 1, 10),
            "AfterMarket",
            date(2023, 1, 11),
            date(2023, 1, 11),
            None,
            None,
            None,
        ),
    ]
    # On Jan 10 the after-market report has not reacted yet, so the last one is Jan 5.
    assert days_since_last_earnings(events, date(2023, 1, 10), DAYS) == 3
    assert days_since_last_earnings(events, date(2023, 1, 11), DAYS) == 0


def test_build_earnings_frame() -> None:
    events = [event(date(2023, 1, 20))]
    frame = build_earnings_frame(events, DAYS)
    assert frame.index.name == "date"
    assert list(frame.index) == list(DAYS)
    assert list(frame.columns) == ["days_to_next", "days_since_last"]
    assert all(str(t) == "float64" for t in frame.dtypes)
    assert frame.loc[pd.Timestamp("2023-01-18"), "days_to_next"] == 2.0
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-18"), "days_since_last"])
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-05"), "days_to_next"])  # 11 days away
    assert frame.loc[pd.Timestamp("2023-01-06"), "days_to_next"] == 10.0
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-20"), "days_to_next"])
    assert frame.loc[pd.Timestamp("2023-01-20"), "days_since_last"] == 0.0
    assert frame.loc[pd.Timestamp("2023-01-25"), "days_since_last"] == 3.0
