"""News timing tests. Decision time is 16:00 New York, window [prev close, this close)."""

from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from bot.data.news import (
    NEWS_FRAME_COLUMNS,
    NewsArticle,
    align_daily_sentiment,
    build_news_frame,
    decision_datetime,
    news_window,
    no_news_days,
    parse_news_rows,
    summarize_window,
)

NY = ZoneInfo("America/New_York")
SYMBOL = "AAPL.US"


def article(utc_text: str, link: str, polarity: float | None = 0.0) -> NewsArticle:
    moment = datetime.fromisoformat(utc_text).astimezone(UTC)
    return NewsArticle(SYMBOL, moment, polarity, "title", link)


def raw(date_text: str, link: str, symbols: list[str], polarity: float = 0.5) -> Any:
    return {
        "date": date_text,
        "title": "t",
        "content": "c",
        "link": link,
        "symbols": symbols,
        "tags": [],
        "sentiment": {"polarity": polarity, "neg": 0.0, "neu": 1.0, "pos": 0.0},
    }


DAYS = pd.DatetimeIndex(
    ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10"]
)


def count_for(articles: list[NewsArticle], day: str) -> int:
    return summarize_window(articles, date.fromisoformat(day), DAYS).article_count


def test_decision_datetime_is_four_pm_new_york() -> None:
    winter = decision_datetime(date(2023, 1, 5))
    assert winter == datetime(2023, 1, 5, 16, tzinfo=NY)
    assert winter.utcoffset() is not None
    summer = decision_datetime(date(2023, 7, 5))
    assert summer.astimezone(UTC).hour == 20
    assert winter.astimezone(UTC).hour == 21


def test_1559_counts_for_same_day_1600_for_next_trading_day() -> None:
    before = article("2023-01-05T20:59:00+00:00", "a")  # 15:59 EST
    at_close = article("2023-01-05T21:00:00+00:00", "b")  # 16:00 EST
    assert count_for([before], "2023-01-05") == 1
    assert count_for([before], "2023-01-06") == 0
    assert count_for([at_close], "2023-01-05") == 0
    assert count_for([at_close], "2023-01-06") == 1


def test_weekend_window_friday_close_to_monday_close() -> None:
    friday_before = article("2023-01-06T20:59:00+00:00", "fri-before")
    friday_after = article("2023-01-06T21:00:00+00:00", "fri-after")
    saturday = article("2023-01-07T15:00:00+00:00", "sat")
    monday_before = article("2023-01-09T20:59:00+00:00", "mon-before")
    monday_after = article("2023-01-09T21:00:00+00:00", "mon-after")
    items = [friday_before, friday_after, saturday, monday_before, monday_after]
    assert count_for(items, "2023-01-06") == 1  # friday_before only
    assert count_for(items, "2023-01-09") == 3  # friday_after, saturday, monday_before
    assert count_for(items, "2023-01-10") == 1  # monday_after


def test_news_window_bounds() -> None:
    start, end = news_window(date(2023, 1, 9), DAYS)
    assert start == datetime(2023, 1, 6, 16, tzinfo=NY)
    assert end == datetime(2023, 1, 9, 16, tzinfo=NY)


def test_news_window_errors() -> None:
    with pytest.raises(ValueError):
        news_window(date(2023, 1, 7), DAYS)  # Saturday
    with pytest.raises(ValueError):
        news_window(date(2023, 1, 3), DAYS)  # first day, no previous close


def test_dst_start_window_uses_new_york_offsets() -> None:
    days = pd.DatetimeIndex(["2023-03-09", "2023-03-10", "2023-03-13", "2023-03-14"])
    start, end = news_window(date(2023, 3, 13), days)
    # Friday close is EST (21:00 UTC). Monday close is EDT (20:00 UTC).
    assert start.astimezone(UTC) == datetime(2023, 3, 10, 21, tzinfo=UTC)
    assert end.astimezone(UTC) == datetime(2023, 3, 13, 20, tzinfo=UTC)
    friday_1559 = article("2023-03-10T20:59:00+00:00", "a")
    friday_1600 = article("2023-03-10T21:00:00+00:00", "b")
    monday_1559 = article("2023-03-13T19:59:00+00:00", "c")
    monday_1600 = article("2023-03-13T20:00:00+00:00", "d")
    items = [friday_1559, friday_1600, monday_1559, monday_1600]
    monday = summarize_window(items, date(2023, 3, 13), days)
    assert monday.article_count == 2  # friday_1600 and monday_1559
    friday = summarize_window(items, date(2023, 3, 10), days)
    assert friday.article_count == 1  # friday_1559
    tuesday = summarize_window(items, date(2023, 3, 14), days)
    assert tuesday.article_count == 1  # monday_1600


def test_dst_end_window_uses_new_york_offsets() -> None:
    days = pd.DatetimeIndex(["2023-11-02", "2023-11-03", "2023-11-06", "2023-11-07"])
    start, end = news_window(date(2023, 11, 6), days)
    # Friday close is EDT (20:00 UTC). Monday close is EST (21:00 UTC).
    assert start.astimezone(UTC) == datetime(2023, 11, 3, 20, tzinfo=UTC)
    assert end.astimezone(UTC) == datetime(2023, 11, 6, 21, tzinfo=UTC)
    friday_1559 = article("2023-11-03T19:59:00+00:00", "a")
    friday_1600 = article("2023-11-03T20:00:00+00:00", "b")
    monday_1559 = article("2023-11-06T20:59:00+00:00", "c")
    monday_1600 = article("2023-11-06T21:00:00+00:00", "d")
    items = [friday_1559, friday_1600, monday_1559, monday_1600]
    assert summarize_window(items, date(2023, 11, 6), days).article_count == 2
    assert summarize_window(items, date(2023, 11, 3), days).article_count == 1
    assert summarize_window(items, date(2023, 11, 7), days).article_count == 1


def test_parse_news_rows_keeps_only_exact_us_symbol() -> None:
    rows = [
        raw("2023-01-05T15:00:00+00:00", "l1", ["AAPL.US", "MSFT.US"]),
        raw("2023-01-05T15:01:00+00:00", "l2", ["AAPL.MX", "AAPL34.SA"]),
        raw("2023-01-05T15:02:00+00:00", "l3", ["MSFT.US"]),
        raw("2023-01-05T15:03:00+00:00", "l4", ["AAPL.US"]),
    ]
    got = parse_news_rows(rows, SYMBOL)
    assert sorted(a.link for a in got) == ["l1", "l4"]
    assert all(a.symbol == SYMBOL for a in got)
    assert all(a.published_at.tzinfo is not None for a in got)


def test_parse_news_rows_dedupes_by_link() -> None:
    rows = [
        raw("2023-01-05T15:00:00+00:00", "same", [SYMBOL], 0.2),
        raw("2023-01-05T16:00:00+00:00", "same", [SYMBOL], 0.9),
        raw("2023-01-05T17:00:00+00:00", "other", [SYMBOL], 0.4),
    ]
    got = parse_news_rows(rows, SYMBOL)
    assert len(got) == 2
    assert sorted(a.link for a in got) == ["other", "same"]


def test_parse_news_rows_naive_timestamp_raises() -> None:
    with pytest.raises(ValueError):
        parse_news_rows([raw("2023-01-05T15:00:00", "x", [SYMBOL])], SYMBOL)


def test_parse_news_rows_reads_polarity() -> None:
    got = parse_news_rows([raw("2023-01-05T15:00:00+00:00", "x", [SYMBOL], 0.25)], SYMBOL)
    assert got[0].polarity == 0.25


def test_summarize_window_mean_polarity_and_empty() -> None:
    items = [
        article("2023-01-05T15:00:00+00:00", "a", 0.2),
        article("2023-01-05T16:00:00+00:00", "b", 0.6),
    ]
    summary = summarize_window(items, date(2023, 1, 5), DAYS)
    assert summary.article_count == 2
    assert summary.mean_polarity == pytest.approx(0.4)
    assert summary.symbol == SYMBOL
    assert summary.decision_date == date(2023, 1, 5)
    empty = summarize_window(items, date(2023, 1, 10), DAYS)
    assert empty.article_count == 0
    assert empty.mean_polarity is None


def test_realistic_counts_from_2023_fixture(fixtures: Any) -> None:
    days = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
    articles = parse_news_rows(fixtures.news_2023(), SYMBOL)
    assert len(articles) == 39  # 40 rows; one Yahoo copy has the same title and time (L10)
    # Counted with plain datetime comparison on the 40 cached articles:
    # Jan 4 window 8 articles (mean polarity 0.15), Jan 5 window 30 (mean 0.3466),
    # Jan 6 window 1 (the 17:59 New York article on Jan 5).
    jan4 = summarize_window(articles, date(2023, 1, 4), days)
    jan5 = summarize_window(articles, date(2023, 1, 5), days)
    jan6 = summarize_window(articles, date(2023, 1, 6), days)
    assert (jan4.article_count, jan5.article_count, jan6.article_count) == (8, 30, 1)
    assert jan4.mean_polarity == pytest.approx(0.15)
    assert jan5.mean_polarity == pytest.approx(0.3466)


def test_realistic_count_from_2016_fixture(fixtures: Any) -> None:
    days = pd.DatetimeIndex(["2016-02-17", "2016-02-18", "2016-02-19", "2016-02-22"])
    articles = parse_news_rows(fixtures.news_2016(), SYMBOL)
    assert len(articles) == 1
    # Published 2016-02-19 10:05 New York, so it counts for Friday the 19th.
    assert summarize_window(articles, date(2016, 2, 19), days).article_count == 1
    assert summarize_window(articles, date(2016, 2, 22), days).article_count == 0
    assert summarize_window(articles, date(2016, 2, 19), days).mean_polarity == pytest.approx(0.994)


def test_align_daily_sentiment_uses_next_trading_day(fixtures: Any) -> None:
    days = pd.DatetimeIndex(
        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10"]
    )
    series = align_daily_sentiment(fixtures.sentiments_2023(), days)
    assert str(series.dtype) == "float64"
    assert list(series.index) == list(days)
    assert pd.isna(series[pd.Timestamp("2023-01-03")])  # nothing dated before Jan 3
    assert series[pd.Timestamp("2023-01-04")] == pytest.approx(0.1977)  # dated Jan 3
    assert series[pd.Timestamp("2023-01-05")] == pytest.approx(0.1409)  # dated Jan 4
    assert series[pd.Timestamp("2023-01-06")] == pytest.approx(0.3639)  # dated Jan 5
    assert series[pd.Timestamp("2023-01-10")] == pytest.approx(0.3944)  # dated Jan 9
    # Monday (Jan 9) is the count-weighted mean of Fri (20 x 0.6063), Sat (8 x 0.633)
    # and Sun (5 x 0.689) = (12.126 + 5.064 + 3.445) / 33 = 0.625303..., never its own 0.3944.
    assert series[pd.Timestamp("2023-01-09")] == pytest.approx(20.635 / 33)


def test_build_news_frame(fixtures: Any) -> None:
    days = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
    articles = parse_news_rows(fixtures.news_2023(), SYMBOL)
    sentiment = pd.Series([0.1, 0.2, 0.3, float("nan")], index=days, dtype="float64")
    frame = build_news_frame(articles, sentiment, days)
    assert tuple(frame.columns) == NEWS_FRAME_COLUMNS
    assert frame.index.name == "date"
    assert str(frame["article_count"].dtype) == "float64"
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-03"), "article_count"])
    assert str(frame["mean_polarity"].dtype) == "float64"
    assert frame.loc[pd.Timestamp("2023-01-05"), "article_count"] == 30
    assert frame.loc[pd.Timestamp("2023-01-05"), "mean_polarity"] == pytest.approx(0.3466)
    assert frame.loc[pd.Timestamp("2023-01-05"), "daily_sentiment"] == pytest.approx(0.3)
    assert frame.loc[pd.Timestamp("2023-01-06"), "article_count"] == 1
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-06"), "daily_sentiment"])


def test_build_news_frame_no_articles_gives_zero_after_first_day_and_nan_on_first() -> None:
    days = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-05"])
    sentiment = pd.Series(float("nan"), index=days, dtype="float64")
    frame = build_news_frame([], sentiment, days)
    # The first day has no earlier close: unknown, not "no news" (L7).
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-03"), "article_count"])
    assert not no_news_days(frame).iloc[0]
    assert no_news_days(frame).iloc[2]
    assert frame.loc[pd.Timestamp("2023-01-05"), "article_count"] == 0
    assert pd.isna(frame.loc[pd.Timestamp("2023-01-05"), "mean_polarity"])
