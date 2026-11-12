"""Tests for the Phase 1 review fixes (M1 to M4, M7, L4, L5, L7 to L10)."""

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from bot.data.earnings import (
    EarningsEvent,
    build_earnings_frame,
    days_since_last_earnings,
    days_to_next_earnings,
    extend_trading_days,
    nyse_holidays,
    parse_earnings_rows,
)
from bot.data.holdout import HoldoutAccessError
from bot.data.membership import (
    MembershipHistory,
    MembershipSpell,
    membership_matrix,
)
from bot.data.news import NewsArticle, coverage_by_year, parse_news_rows
from bot.data.prices import (
    SplitEvent,
    build_price_frame,
    cumulative_split_factors,
    dividends_outside_frame,
    infer_splits_from_adjusted_ratio,
    merge_split_sources,
    parse_split_rows,
    relative_volume,
    wide_panel,
)

SYMBOL = "AAPL.US"


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def rows_for(days: list[str], closes: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "date": d,
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adjusted_close": c,
            "volume": 1000,
        }
        for d, c in zip(days, closes, strict=True)
    ]


# M1: volume


def test_wide_panel_refuses_volume() -> None:
    frame = pd.DataFrame({"volume": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"]))
    with pytest.raises(ValueError, match="relative_volume"):
        wide_panel({"A.US": frame}, "volume")


def test_relative_volume_excludes_today_and_needs_lookback() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({"volume": [100.0, 200.0, 300.0, 600.0, 100.0]}, index=index)
    got = relative_volume(frame, 3)
    assert got.iloc[:3].isna().all()
    # Day 4: 600 / mean(100, 200, 300) = 3.0. Day 5: 100 / mean(200, 300, 600) = 0.0909...
    assert got.iloc[3] == pytest.approx(3.0)
    assert got.iloc[4] == pytest.approx(100 / (1100 / 3))


def test_relative_volume_is_scale_free() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    small = pd.DataFrame({"volume": [1.0, 2.0, 3.0, 4.0]}, index=index)
    big = small * 4
    assert list(relative_volume(small, 2).dropna()) == pytest.approx(
        list(relative_volume(big, 2).dropna())
    )


# M2 and M3: earnings


def test_unknown_timing_with_actual_uses_both_sessions() -> None:
    days = pd.bdate_range("2023-01-02", "2023-02-28")
    raw: Any = [
        {
            "code": SYMBOL,
            "report_date": "2023-01-18",
            "date": "2022-12-31",
            "actual": 1.5,
            "estimate": 1.4,
        }
    ]
    (event,) = parse_earnings_rows(raw, days)
    assert event.timing == "Unknown"
    assert event.earliest_reaction_session == date(2023, 1, 18)
    assert event.latest_reaction_session == date(2023, 1, 19)
    assert event.actual == 1.5
    # days_to_next uses the earliest session. days_since_last waits for the latest.
    assert days_to_next_earnings([event], date(2023, 1, 17), days) == 1
    assert days_since_last_earnings([event], date(2023, 1, 18), days) is None
    assert days_since_last_earnings([event], date(2023, 1, 19), days) == 0


def test_unknown_report_dated_today_still_counts_as_tomorrow() -> None:
    days = pd.bdate_range("2023-01-02", "2023-02-28")
    raw: Any = [{"code": SYMBOL, "report_date": "2023-01-18", "date": "2022-12-31"}]
    events = parse_earnings_rows(raw, days)
    assert days_to_next_earnings(events, date(2023, 1, 18), days) == 1
    frame = build_earnings_frame(events, days)
    assert frame.loc[ts("2023-01-18"), "days_to_next"] == 1.0
    assert frame.loc[ts("2023-01-17"), "days_to_next"] == 1.0
    assert pd.isna(frame.loc[ts("2023-01-18"), "days_since_last"])
    assert frame.loc[ts("2023-01-19"), "days_since_last"] == 0.0


def test_events_after_calendar_end_are_kept() -> None:
    # The review scenario: calendar ends 2023-10-25. AfterMarket 10-25 and
    # BeforeMarket 10-26 both react on 10-26.
    days = pd.bdate_range("2023-09-01", "2023-10-25")
    raw: Any = [
        {"code": SYMBOL, "report_date": "2023-10-25", "before_after_market": "AfterMarket"},
        {"code": SYMBOL, "report_date": "2023-10-26", "before_after_market": "BeforeMarket"},
    ]
    events = parse_earnings_rows(raw, days)
    assert len(events) == 2
    assert days_to_next_earnings(events, date(2023, 10, 25), days) == 1
    frame = build_earnings_frame(events, days)
    assert frame.loc[ts("2023-10-25"), "days_to_next"] == 1.0


def test_event_two_sessions_past_calendar_end() -> None:
    days = pd.bdate_range("2023-09-01", "2023-10-25")
    raw: Any = [
        {"code": SYMBOL, "report_date": "2023-10-27", "before_after_market": "BeforeMarket"}
    ]
    events = parse_earnings_rows(raw, days)
    assert days_to_next_earnings(events, date(2023, 10, 25), days) == 2


def test_extend_trading_days_skips_weekends_and_holidays() -> None:
    days = pd.DatetimeIndex(["2023-11-21", "2023-11-22"])
    got = extend_trading_days(days, 3)
    # Thanksgiving Thu 2023-11-23 is closed, so the next sessions are Fri, Mon, Tue.
    assert [d.date() for d in got[2:]] == [
        date(2023, 11, 24),
        date(2023, 11, 27),
        date(2023, 11, 28),
    ]
    assert extend_trading_days(days, 0) is days


def test_nyse_holidays_2023() -> None:
    assert nyse_holidays(2023) == {
        date(2023, 1, 2),  # New Year's Day on a Sunday, observed Monday
        date(2023, 1, 16),
        date(2023, 2, 20),
        date(2023, 4, 7),  # Good Friday
        date(2023, 5, 29),
        date(2023, 6, 19),
        date(2023, 7, 4),
        date(2023, 9, 4),
        date(2023, 11, 23),
        date(2023, 12, 25),
    }
    assert date(2021, 12, 24) in nyse_holidays(2021)  # Christmas on a Saturday
    assert date(2022, 12, 30) not in nyse_holidays(2022)  # New Year's on a Saturday is not observed
    assert date(2021, 6, 18) not in nyse_holidays(2021)  # Juneteenth starts in 2022


# M4: merge by trading days


def test_merge_aapl_reported_and_inferred_same_split(fixtures: Any) -> None:
    eod = fixtures.eod("AAPL")
    days = pd.DatetimeIndex([ts(r["date"]) for r in eod])
    inferred = infer_splits_from_adjusted_ratio(eod)
    assert [e.ex_date for e in inferred] == [date(2020, 8, 31)]
    assert inferred[0].factor == pytest.approx(4.0, rel=0.01)
    reported = parse_split_rows(fixtures.splits("AAPL"))
    assert merge_split_sources(reported, inferred, days) == reported
    # Without the reported split the inferred one carries the adjustment.
    frame = build_price_frame(eod, merge_split_sources([], inferred, days), [])
    assert frame.loc[ts("2020-08-28"), "close"] == pytest.approx(499.23 / inferred[0].factor)


def test_merge_review_scenario_thursday_to_monday() -> None:
    days = pd.bdate_range("2020-08-17", "2020-09-15")
    reported = [SplitEvent(date(2020, 8, 27), 4.0, "eodhd")]
    inferred = [SplitEvent(date(2020, 8, 31), 4.0, "inferred")]
    assert merge_split_sources(reported, inferred, days) == reported


# Split inference on real fixture rows


def test_infer_tsla_five_for_one_from_fixture(fixtures: Any) -> None:
    inferred = infer_splits_from_adjusted_ratio(fixtures.eod("TSLA"))
    assert [e.ex_date for e in inferred] == [date(2020, 8, 31)]
    assert inferred[0].factor == pytest.approx(5.0, rel=0.01)


# Split edge cases through build_price_frame


def test_reverse_split_through_build_price_frame() -> None:
    rows = rows_for(["2020-03-02", "2020-03-03", "2020-03-04"], [2.0, 2.0, 20.0])
    frame = build_price_frame(rows, [SplitEvent(date(2020, 3, 4), 0.1, "eodhd")], [])
    assert list(frame["close"]) == pytest.approx([20.0, 20.0, 20.0])
    assert frame.loc[ts("2020-03-04"), "total_return"] == pytest.approx(0.0)
    assert frame.loc[ts("2020-03-04"), "split_factor"] == pytest.approx(0.1)
    assert list(frame["raw_close"]) == [2.0, 2.0, 20.0]


def test_split_on_non_trading_day_through_build_price_frame() -> None:
    # Saturday 2020-03-07 split: rows through Friday are divided, Monday is new basis.
    rows = rows_for(["2020-03-05", "2020-03-06", "2020-03-09"], [100.0, 100.0, 25.0])
    frame = build_price_frame(rows, [SplitEvent(date(2020, 3, 7), 4.0, "eodhd")], [])
    assert list(frame["close"]) == pytest.approx([25.0, 25.0, 25.0])
    assert frame.loc[ts("2020-03-09"), "total_return"] == pytest.approx(0.0)
    assert frame.loc[ts("2020-03-09"), "split_factor"] == 4.0


def test_dividend_on_non_trading_day_moves_to_next_row() -> None:
    rows = rows_for(["2020-03-06", "2020-03-09", "2020-03-10"], [100.0, 100.0, 100.0])
    div: Any = [{"date": "2020-03-07", "value": 1.0, "unadjustedValue": 1.0}]
    frame = build_price_frame(rows, [], div)
    assert list(frame["dividend"]) == [0.0, 1.0, 0.0]
    assert frame.loc[ts("2020-03-09"), "total_return"] == pytest.approx(0.01)


def test_dividends_outside_frame_are_counted() -> None:
    index = pd.DatetimeIndex(["2020-03-06", "2020-03-09"])
    div: Any = [
        {"date": "2020-03-01", "unadjustedValue": 1.0},
        {"date": "2020-03-07", "unadjustedValue": 1.0},
        {"date": "2020-03-20", "unadjustedValue": 1.0},
    ]
    assert dividends_outside_frame(div, index) == 2
    assert dividends_outside_frame(div, pd.DatetimeIndex([])) == 3


# L5 and M7: holdout


def test_holdout_dated_split_raises_unless_approved() -> None:
    index = pd.DatetimeIndex(["2023-12-28", "2023-12-29"])
    late = [SplitEvent(date(2024, 3, 1), 2.0, "eodhd")]
    with pytest.raises(HoldoutAccessError):
        cumulative_split_factors(index, late)
    rows = rows_for(["2023-12-28", "2023-12-29"], [10.0, 10.0])
    with pytest.raises(HoldoutAccessError):
        build_price_frame(rows, late, [])
    # approved=True alone still raises: no session is open (D12).
    with pytest.raises(HoldoutAccessError):
        build_price_frame(rows, late, [], approved=True)


def test_approved_flag_is_passed_to_every_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_guard(day: date, *, approved: bool = False) -> None:
        seen.append(approved)

    monkeypatch.setattr("bot.data.prices.assert_before_holdout", fake_guard)
    monkeypatch.setattr("bot.data.membership.assert_before_holdout", fake_guard)
    rows = rows_for(["2024-01-02"], [10.0])
    build_price_frame(rows, [], [], approved=True)
    history = MembershipHistory((MembershipSpell(SYMBOL, date(2020, 1, 1), None),))
    history.members_on(date(2024, 1, 2), approved=True)
    history.universe(date(2020, 1, 1), date(2024, 1, 2), approved=True)
    membership_matrix(history, pd.DatetimeIndex(["2024-01-02"]), approved=True)
    assert seen == [True, True, True, True]


def test_parse_earnings_rows_approved_needs_a_session() -> None:
    days = pd.bdate_range("2023-12-01", "2024-02-29")
    raw: Any = [
        {"code": SYMBOL, "report_date": "2024-01-30", "before_after_market": "BeforeMarket"}
    ]
    assert parse_earnings_rows(raw, days) == []
    assert parse_earnings_rows(raw, days, approved=True) == []  # no open session


# L10: dedupe


def yahoo_row(date_text: str, link: str, title: str, polarity: float) -> Any:
    return {
        "date": date_text,
        "title": title,
        "content": "c",
        "link": link,
        "symbols": [SYMBOL],
        "tags": [],
        "sentiment": {"polarity": polarity, "neg": 0.0, "neu": 1.0, "pos": 0.0},
    }


def test_syndicated_copy_on_two_yahoo_domains_counts_once() -> None:
    rows = [
        yahoo_row(
            "2023-01-05T15:00:00+00:00",
            "https://finance.yahoo.com/news/apple-beats-1",
            "Apple Beats Estimates!",
            0.9,
        ),
        yahoo_row(
            "2023-01-05T15:00:00+00:00",
            "https://uk.finance.yahoo.com/news/apple-beats-1",
            "  apple beats   estimates ",
            0.1,
        ),
    ]
    got = parse_news_rows(rows, SYMBOL)
    assert len(got) == 1
    # Same timestamp, so the copy seen first is kept.
    assert got[0].link == "https://finance.yahoo.com/news/apple-beats-1"
    assert got[0].polarity == 0.9


def test_duplicate_keeps_earliest_timestamp_even_if_seen_later() -> None:
    rows = [
        yahoo_row("2023-01-05T16:00:00+00:00", "same", "A story", 0.2),
        yahoo_row("2023-01-05T15:00:00+00:00", "same", "A story", 0.8),
    ]
    got = parse_news_rows(rows, SYMBOL)
    assert len(got) == 1
    assert got[0].polarity == 0.8
    assert got[0].published_at == datetime(2023, 1, 5, 15, 0, tzinfo=UTC)


def test_same_title_at_different_times_is_not_a_duplicate() -> None:
    rows = [
        yahoo_row("2023-01-05T15:00:00+00:00", "l1", "Daily update", 0.2),
        yahoo_row("2023-01-06T15:00:00+00:00", "l2", "Daily update", 0.3),
    ]
    assert len(parse_news_rows(rows, SYMBOL)) == 2


def test_blank_titles_do_not_collapse() -> None:
    rows = [
        yahoo_row("2023-01-05T15:00:00+00:00", "l1", "", 0.2),
        yahoo_row("2023-01-05T15:00:00+00:00", "l2", "", 0.3),
    ]
    assert len(parse_news_rows(rows, SYMBOL)) == 2


# L9: coverage by member ticker-year


def test_coverage_by_year_counts_only_member_ticker_years() -> None:
    history = MembershipHistory(
        (
            MembershipSpell("A.US", date(2015, 1, 1), None),
            MembershipSpell("B.US", date(2016, 6, 1), date(2017, 3, 1)),
            MembershipSpell("C.US", date(2017, 1, 1), None),
        )
    )

    def series(counts: dict[str, int]) -> pd.Series:
        return pd.Series(list(counts.values()), index=pd.DatetimeIndex(list(counts)), dtype="int64")

    counts = {
        "A.US": series({"2015-03-02": 10, "2016-03-02": 20, "2017-03-02": 30}),
        "B.US": series({"2016-07-01": 4, "2017-02-01": 6}),
        # C.US has no articles at all: it counts as 0 in 2017, and is absent in 2015 and 2016.
    }
    table = coverage_by_year(counts, history, range(2015, 2018))
    assert list(table["member_tickers"]) == [1, 2, 3]
    assert list(table["median"]) == [10.0, 12.0, 6.0]
    # 2016 members A=20, B=4: p10 = 4 + 0.1 * 16 = 5.6. 2017 members 30, 6, 0: p10 = 0 + 0.2 * 6.
    assert table.loc[2016, "p10"] == pytest.approx(5.6)
    assert table.loc[2017, "p10"] == pytest.approx(1.2)
    assert table.loc[2017, "share_zero"] == pytest.approx(1 / 3)
    assert table.index.name == "year"


def test_coverage_by_year_refuses_holdout_years() -> None:
    history = MembershipHistory((MembershipSpell("A.US", date(2015, 1, 1), None),))
    with pytest.raises(HoldoutAccessError):
        coverage_by_year({}, history, [2024])


def test_news_article_dataclass_still_importable() -> None:
    assert NewsArticle.__name__ == "NewsArticle"


def test_earnings_event_fields() -> None:
    event = EarningsEvent(
        SYMBOL, date(2023, 1, 18), "Unknown", date(2023, 1, 18), date(2023, 1, 19), None, 1.0, 1.1
    )
    assert event.earliest_reaction_session < event.latest_reaction_session
