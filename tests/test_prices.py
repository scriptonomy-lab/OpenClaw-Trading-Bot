"""Price tests: splits, volume, dividends, total return. All from cached fixtures."""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot.data import eodhd_client as client
from bot.data.holdout import HoldoutAccessError
from bot.data.prices import (
    PRICE_COLUMNS,
    SplitEvent,
    build_price_frame,
    cumulative_split_factors,
    infer_splits_from_adjusted_ratio,
    large_unexplained_moves,
    merge_split_sources,
    parse_split_ratio,
    parse_split_rows,
    slice_to_decision_date,
    total_return_index,
    wide_panel,
)

AAPL_SPLIT = SplitEvent(date(2020, 8, 31), 4.0, "eodhd")


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def aapl_frame(fixtures: Any) -> pd.DataFrame:
    splits = parse_split_rows(fixtures.splits("AAPL"))
    return build_price_frame(fixtures.eod("AAPL"), splits, [])


def test_parse_split_ratio() -> None:
    assert parse_split_ratio("4.000000/1.000000") == 4.0
    assert parse_split_ratio("1.000000/10.000000") == pytest.approx(0.1)
    assert parse_split_ratio("3/2") == pytest.approx(1.5)


@pytest.mark.parametrize("bad", ["", "4", "abc/1", "4/0", "0/1", "-2/1", "4/1/1"])
def test_parse_split_ratio_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_split_ratio(bad)


def test_parse_split_rows_sorted_with_source(fixtures: Any) -> None:
    events = parse_split_rows(fixtures.splits("AAPL"))
    assert [e.ex_date for e in events] == sorted(e.ex_date for e in events)
    assert events[-1] == SplitEvent(date(2020, 8, 31), 4.0, "eodhd")
    assert events[-2] == SplitEvent(date(2014, 6, 9), 7.0, "eodhd")


def test_client_never_returns_splits_after_2023(tmp_path: Path) -> None:
    rows = [
        {"date": "2020-08-31", "split": "4.000000/1.000000"},
        {"date": "2024-03-01", "split": "2.000000/1.000000"},
    ]
    path = client.cache_path("/splits/AAPL.US", {"to": "2023-12-31"}, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(rows))
    got = client.get_splits("AAPL", cache_dir=tmp_path, offline=True, api_token="x")
    assert [r["date"] for r in got] == ["2020-08-31"]


def test_cumulative_split_factors_hand_computed() -> None:
    dates = pd.DatetimeIndex(["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-09", "2020-01-10"])
    splits = [
        SplitEvent(date(2020, 1, 5), 2.0, "eodhd"),
        SplitEvent(date(2020, 1, 10), 3.0, "eodhd"),
    ]
    got = cumulative_split_factors(dates, splits)
    # Before both splits: 2 * 3 = 6. On the first ex-date only the 3:1 is still ahead.
    assert list(got) == [6.0, 6.0, 3.0, 3.0, 1.0]


def test_aapl_split_adjustment(fixtures: Any) -> None:
    frame = aapl_frame(fixtures)
    raw = {r["date"]: r for r in fixtures.eod("AAPL")}
    # 499.23 / 4 = 124.8075 on 2020-08-28. Not EODHD's adjusted_close of 120.9557.
    assert frame.loc[ts("2020-08-28"), "close"] == pytest.approx(124.8075)
    assert frame.loc[ts("2020-08-28"), "open"] == pytest.approx(504.05 / 4)
    assert frame.loc[ts("2020-08-28"), "high"] == pytest.approx(505.77 / 4)
    assert frame.loc[ts("2020-08-28"), "low"] == pytest.approx(498.31 / 4)
    assert frame.loc[ts("2020-08-28"), "raw_close"] == 499.23
    assert frame.loc[ts("2020-08-17"), "close"] == pytest.approx(458.43 / 4)
    # Rows on and after the ex-date are unchanged.
    for day in ("2020-08-31", "2020-09-01", "2020-09-15"):
        for column in ("open", "high", "low", "close"):
            assert frame.loc[ts(day), column] == pytest.approx(raw[day][column])
    assert frame.loc[ts("2020-08-31"), "split_factor"] == 4.0
    assert (frame["split_factor"].drop(ts("2020-08-31")) == 1.0).all()


def test_aapl_no_artificial_jump_across_split(fixtures: Any) -> None:
    frame = aapl_frame(fixtures)
    adjusted_move = frame.loc[ts("2020-08-31"), "close"] / frame.loc[ts("2020-08-28"), "close"] - 1
    raw_move = (
        frame.loc[ts("2020-08-31"), "raw_close"] / frame.loc[ts("2020-08-28"), "raw_close"] - 1
    )
    # 129.04 * 4 / 499.23 - 1 = 0.0339, while raw is 129.04 / 499.23 - 1 = -0.7415.
    assert adjusted_move == pytest.approx(516.16 / 499.23 - 1)
    assert abs(adjusted_move) < 0.05
    assert raw_move == pytest.approx(129.04 / 499.23 - 1)
    assert raw_move < -0.7


def test_tsla_five_for_one(fixtures: Any) -> None:
    splits = [SplitEvent(date(2020, 8, 31), 5.0, "eodhd")]
    frame = build_price_frame(fixtures.eod("TSLA"), splits, [])
    assert frame.loc[ts("2020-08-28"), "close"] == pytest.approx(2213.4 / 5)
    assert frame.loc[ts("2020-08-31"), "close"] == pytest.approx(498.3201)
    adjusted_move = frame.loc[ts("2020-08-31"), "close"] / frame.loc[ts("2020-08-28"), "close"] - 1
    assert adjusted_move == pytest.approx(498.3201 * 5 / 2213.4 - 1)  # about +0.126
    assert 498.3201 / 2213.4 - 1 < -0.7


def test_tsla_two_splits_compound(fixtures: Any) -> None:
    # The 2022 3:1 split is after this window, so these rows are divided by 5 * 3 = 15
    # before 2020-08-31 and by 3 afterwards. EODHD's adjusted_close agrees here.
    splits = parse_split_rows(fixtures.splits("TSLA"))
    frame = build_price_frame(fixtures.eod("TSLA"), splits, [])
    assert frame.loc[ts("2020-08-28"), "close"] == pytest.approx(2213.4 / 15)
    assert frame.loc[ts("2020-08-31"), "close"] == pytest.approx(498.3201 / 3)


def test_volume_is_not_adjusted(fixtures: Any) -> None:
    rows = fixtures.eod("AAPL")
    frame = aapl_frame(fixtures)
    assert list(frame["volume"]) == [float(r["volume"]) for r in rows]


def test_frame_shape_and_dtypes(fixtures: Any) -> None:
    frame = aapl_frame(fixtures)
    assert tuple(frame.columns) == PRICE_COLUMNS
    assert frame.index.name == "date"
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 21
    assert all(str(t) == "float64" for t in frame.dtypes)
    assert "adjusted_close" not in frame.columns


def test_total_return_without_dividend(fixtures: Any) -> None:
    frame = aapl_frame(fixtures)
    assert pd.isna(frame["total_return"].iloc[0])
    assert frame.loc[ts("2020-08-31"), "total_return"] == pytest.approx(516.16 / 499.23 - 1)
    assert (frame["dividend"] == 0.0).all()


def _synthetic_rows(days: list[str], closes: list[float]) -> list[dict[str, Any]]:
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


def test_dividend_before_split_is_divided_by_four(fixtures: Any) -> None:
    dividends = [r for r in fixtures.dividends("AAPL") if r["date"] == "2020-02-07"]
    assert dividends[0]["unadjustedValue"] == 0.77
    rows = _synthetic_rows(["2020-02-05", "2020-02-06", "2020-02-07"], [320.0, 440.0, 444.0])
    frame = build_price_frame(rows, [AAPL_SPLIT], dividends)
    # Factor 4: close 444 / 4 = 111, previous 440 / 4 = 110, dividend 0.77 / 4 = 0.1925.
    assert frame.loc[ts("2020-02-07"), "dividend"] == pytest.approx(0.1925)
    assert frame.loc[ts("2020-02-07"), "total_return"] == pytest.approx((111 + 0.1925) / 110 - 1)
    assert frame.loc[ts("2020-02-06"), "dividend"] == 0.0


def test_dividend_after_split_is_unchanged(fixtures: Any) -> None:
    dividends = [r for r in fixtures.dividends("AAPL") if r["date"] == "2020-11-06"]
    assert dividends[0]["unadjustedValue"] == 0.205
    rows = _synthetic_rows(["2020-11-04", "2020-11-05", "2020-11-06"], [118.0, 120.0, 119.0])
    frame = build_price_frame(rows, [AAPL_SPLIT], dividends)
    assert frame.loc[ts("2020-11-06"), "dividend"] == pytest.approx(0.205)
    # (119 + 0.205) / 120 - 1 = -0.006625
    assert frame.loc[ts("2020-11-06"), "total_return"] == pytest.approx(-0.006625)


def test_dividends_use_unadjusted_value_not_value() -> None:
    # value is on today's basis. Only unadjustedValue may be used.
    div = [{"date": "2020-02-07", "value": 9.99, "unadjustedValue": 0.80}]
    rows = _synthetic_rows(["2020-02-06", "2020-02-07"], [400.0, 400.0])
    frame = build_price_frame(rows, [AAPL_SPLIT], div)  # type: ignore[arg-type]
    assert frame.loc[ts("2020-02-07"), "dividend"] == pytest.approx(0.20)


def test_build_price_frame_refuses_holdout_dates() -> None:
    rows = _synthetic_rows(["2023-12-29", "2024-01-02"], [100.0, 101.0])
    with pytest.raises(HoldoutAccessError):
        build_price_frame(rows, [], [])


def _ratio_rows(ratios: list[float], raws: list[float]) -> list[dict[str, Any]]:
    days = pd.bdate_range("2019-03-01", periods=len(ratios))
    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": r,
            "high": r,
            "low": r,
            "close": r,
            "adjusted_close": r * k,
            "volume": 1,
        }
        for d, k, r in zip(days, ratios, raws, strict=True)
    ]


def test_infer_two_for_one_ignores_dividend_steps() -> None:
    # Ratio steps: 0.502 / 0.5 = 1.004 (a dividend, ignored) and 1.004 / 0.502 = 2.0 (a split).
    rows = _ratio_rows([0.5, 0.5, 0.502, 0.502, 1.004, 1.004], [100, 100, 100, 100, 50, 50])
    events = infer_splits_from_adjusted_ratio(rows)  # type: ignore[arg-type]
    assert len(events) == 1
    assert events[0].ex_date == date(2019, 3, 7)
    assert events[0].factor == pytest.approx(2.0, rel=0.01)
    assert events[0].source == "inferred"


def test_infer_reverse_split() -> None:
    rows = _ratio_rows([1.0, 1.0, 0.1, 0.1], [10, 10, 100, 100])
    events = infer_splits_from_adjusted_ratio(rows)  # type: ignore[arg-type]
    assert len(events) == 1
    assert events[0].factor == pytest.approx(0.1, rel=0.01)


def test_infer_ignores_jump_that_is_not_a_common_ratio() -> None:
    rows = _ratio_rows([1.0, 1.0, 1.3, 1.3], [10, 10, 10, 10])
    assert infer_splits_from_adjusted_ratio(rows) == []  # type: ignore[arg-type]


def test_infer_constant_later_factor_has_no_effect() -> None:
    plain = _ratio_rows([0.5, 0.5, 1.0, 1.0], [100, 100, 50, 50])
    scaled = _ratio_rows([0.25, 0.25, 0.5, 0.5], [100, 100, 50, 50])
    one = infer_splits_from_adjusted_ratio(plain)  # type: ignore[arg-type]
    two = infer_splits_from_adjusted_ratio(scaled)  # type: ignore[arg-type]
    assert [e.ex_date for e in one] == [e.ex_date for e in two]
    assert len(one) == 1


WEEKDAYS = pd.bdate_range("2018-01-01", "2020-12-31")


def test_merge_reported_split_wins_within_two_trading_days() -> None:
    reported = [SplitEvent(date(2020, 8, 31), 4.0, "eodhd")]
    inferred = [SplitEvent(date(2020, 9, 2), 4.0, "inferred")]
    assert merge_split_sources(reported, inferred, WEEKDAYS) == reported


def test_merge_keeps_distant_inferred_split_sorted() -> None:
    reported = [SplitEvent(date(2020, 8, 31), 4.0, "eodhd")]
    far = SplitEvent(date(2018, 5, 1), 2.0, "inferred")
    merged = merge_split_sources(reported, [far], WEEKDAYS)
    assert merged == [far, reported[0]]


@pytest.mark.parametrize(
    ("reported_day", "inferred_day"),
    [
        (date(2020, 8, 28), date(2020, 8, 31)),  # Friday to Monday: 3 calendar days, 1 trading day
        (date(2020, 8, 27), date(2020, 8, 31)),  # Thursday to Monday: 4 calendar, 2 trading
    ],
)
def test_merge_counts_trading_days_not_calendar_days(
    reported_day: date, inferred_day: date
) -> None:
    reported = [SplitEvent(reported_day, 4.0, "eodhd")]
    inferred = [SplitEvent(inferred_day, 4.0, "inferred")]
    assert merge_split_sources(reported, inferred, WEEKDAYS) == reported


def test_merge_adds_inferred_split_three_trading_days_away() -> None:
    # Wednesday to Monday is 5 calendar days and 3 trading days: a separate event.
    reported = [SplitEvent(date(2020, 8, 26), 4.0, "eodhd")]
    inferred = [SplitEvent(date(2020, 8, 31), 4.0, "inferred")]
    assert len(merge_split_sources(reported, inferred, WEEKDAYS)) == 2


def test_large_unexplained_moves(fixtures: Any) -> None:
    good = aapl_frame(fixtures)
    assert large_unexplained_moves(good) == []
    missing = build_price_frame(fixtures.eod("AAPL"), [], [])
    assert large_unexplained_moves(missing) == [date(2020, 8, 31)]


def test_total_return_index_hand_computed() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"], name="date")
    frame = pd.DataFrame(0.0, index=index, columns=list(PRICE_COLUMNS))
    frame["total_return"] = [float("nan"), 0.1, -0.1]
    got = total_return_index(frame)
    assert list(got) == pytest.approx([1.0, 1.1, 0.99])


def test_slice_to_decision_date_is_inclusive(fixtures: Any) -> None:
    frame = aapl_frame(fixtures)
    cut = slice_to_decision_date(frame, date(2020, 8, 31))
    assert cut.index[-1] == ts("2020-08-31")
    assert len(cut) == 11
    # A weekend decision date keeps the Friday as the last row.
    assert slice_to_decision_date(frame, date(2020, 8, 30)).index[-1] == ts("2020-08-28")


def test_wide_panel_does_not_forward_fill() -> None:
    one = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    two = pd.DataFrame(
        {"close": [10.0, 30.0]}, index=pd.DatetimeIndex(["2020-01-02", "2020-01-06"])
    )
    panel = wide_panel({"A.US": one, "B.US": two}, "close")
    assert list(panel.columns) == ["A.US", "B.US"]
    assert pd.isna(panel.loc[ts("2020-01-03"), "B.US"])
    assert panel.loc[ts("2020-01-06"), "B.US"] == 30.0
