"""Tests for the fake EODHD transport and the planted cases.

Each planted case has a known answer, so these tests also show that the data
layer (client, membership, prices, news, earnings) handles it correctly.
"""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from bot import config
from bot.data import eodhd_client as client
from bot.data import fake_eodhd as fake
from bot.data.earnings import parse_earnings_rows, reaction_session
from bot.data.holdout import HoldoutAccessError
from bot.data.membership import RawMembershipRow, build_membership_history
from bot.data.news import parse_news_rows, summarize_window
from bot.data.prices import (
    RawEodRow,
    RawSplitRow,
    build_price_frame,
    check_splits_against_prices,
    parse_split_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eodhd"
START, END = date(2014, 1, 1), date(2023, 12, 31)


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Force fake mode and point the default fake cache at a temp folder."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", tmp_path / "fake")
    yield


def d(text: str) -> date:
    return date.fromisoformat(text)


def membership() -> Any:
    rows = cast(list[RawMembershipRow], client.get_index_history())
    return build_membership_history(rows)


def prices(ticker: str) -> pd.DataFrame:
    eod = cast(list[RawEodRow], client.get_eod_prices(ticker, START, END))
    splits = parse_split_rows(cast(list[RawSplitRow], client.get_splits(ticker)))
    return build_price_frame(eod, splits, client.get_dividends(ticker, START, END))  # type: ignore[arg-type]


def test_same_seed_same_output() -> None:
    first = json.dumps(fake.fake_shape_examples(), sort_keys=True)
    fake._world.cache_clear()
    fake._news_year.cache_clear()
    assert json.dumps(fake.fake_shape_examples(), sort_keys=True) == first


def shape(value: Any) -> Any:
    """Key structure with value types replaced by type names."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__ if value is not None else "NoneType"


def keys_of(row: dict[str, Any]) -> set[str]:
    return set(row)


def test_shapes_match_recorded_fixtures_key_for_key() -> None:
    examples = fake.fake_shape_examples()
    eod = json.loads((FIXTURES / "eod/AAPL.US/from=2020-08-17_to=2020-09-15.json").read_text())
    assert keys_of(examples["eod"][0]) == keys_of(eod[0])
    div = json.loads((FIXTURES / "div/AAPL.US/from=2019-01-01_to=2021-12-31.json").read_text())
    assert keys_of(examples["div"][0]) == keys_of(div[0])
    splits = json.loads((FIXTURES / "splits/AAPL.US/to=2023-12-31.json").read_text())
    assert keys_of(examples["splits"][0]) == keys_of(splits[0])
    news_dir = FIXTURES / "news"
    news = json.loads(next(news_dir.glob("from=2023*")).read_text())
    assert keys_of(examples["news"][0]) == keys_of(news[0])
    assert keys_of(examples["news"][0]["sentiment"]) == keys_of(news[0]["sentiment"])
    sent = json.loads(next((FIXTURES / "sentiments").iterdir()).read_text())
    assert keys_of(next(iter(examples["sentiments"].values()))[0]) == keys_of(
        next(iter(sent.values()))[0]
    )
    cal = json.loads(next((FIXTURES / "calendar/earnings").iterdir()).read_text())
    assert keys_of(examples["calendar_earnings"]) == keys_of(cal)
    assert keys_of(examples["calendar_earnings"]["earnings"][0]) == keys_of(cal["earnings"][0])


def test_news_timestamps_and_symbols() -> None:
    articles = client.get_news("SIM001", date(2023, 1, 1), date(2023, 3, 31))
    assert articles
    assert all(a["date"].endswith("+00:00") for a in articles)
    assert any(
        any(s.startswith("SIM001.") and s != "SIM001.US" for s in a["symbols"]) for a in articles
    )
    new_york_hours = {pd.Timestamp(a["date"]).tz_convert("America/New_York").hour for a in articles}
    assert any(hour >= 16 for hour in new_york_hours)
    injected = [a for a in articles if "Ignore previous instructions" in a["title"]]
    everything = client.get_news("SIM002", date(2020, 1, 1), date(2023, 12, 31))
    assert injected or any("Ignore previous instructions" in a["title"] for a in everything)


def test_news_is_thin_before_2019() -> None:
    def count(year: int) -> int:
        return sum(
            len(client.get_news(f"SIM{n:03d}", date(year, 1, 1), date(year, 12, 31)))
            for n in range(1, 21)
        )

    assert count(2016) < count(2019) < count(2022)


def test_ohlc_consistent_and_raw_across_split() -> None:
    planted = fake.PLANTED["split_before_holdout"]
    rows = client.get_eod_prices(planted["ticker"], START, END)
    for row in rows:
        assert row["high"] >= max(row["open"], row["close"])
        assert row["low"] <= min(row["open"], row["close"])
    by_date = {r["date"]: r for r in rows}
    ex = d(planted["date"])
    index = [r["date"] for r in rows].index(ex.isoformat())
    before, after = rows[index - 1], rows[index]
    assert before["close"] / after["close"] > 2.5  # raw price drops about 4x
    assert before["volume"] / after["volume"] < 6  # volume has no 4x jump
    assert by_date[before["date"]]["adjusted_close"] < before["close"] / 2


def test_split_adjusted_continuity_and_no_flags() -> None:
    frame = prices(fake.PLANTED["split_before_holdout"]["ticker"])
    ex = pd.Timestamp(fake.PLANTED["split_before_holdout"]["date"])
    move = frame["close"].pct_change().loc[ex]
    assert abs(move) < 0.25
    assert check_splits_against_prices(frame) == []


def test_post_holdout_split_is_in_adjusted_close_only() -> None:
    ticker = fake.PLANTED["split_after_holdout"]["ticker"]
    eod = client.get_eod_prices(ticker, START, END)
    assert all(row["date"] < "2024-01-01" for row in eod)
    assert all(row["adjusted_close"] < row["close"] * 0.8 for row in eod[-50:])
    assert client.get_splits(ticker) == []  # the 2025 split is clipped by the guard


def test_reverse_split_and_dividend_fields() -> None:
    splits = client.get_splits("SIM030")
    assert splits == [{"date": "2020-05-11", "split": "1.000000/10.000000"}]
    dividends = client.get_dividends("SIM002", START, END)
    assert dividends
    assert dividends[0]["unadjustedValue"] >= dividends[0]["value"]


def test_removed_still_trading() -> None:
    spec = fake.PLANTED["removed_still_trading"]
    history = membership()
    symbol = f"{spec['ticker']}.US"
    assert symbol in history.members_on(d(spec["last_member_day"]))
    assert symbol not in history.members_on(d(spec["first_non_member_day"]))
    rows = client.get_eod_prices(spec["ticker"], d("2018-06-01"), d("2018-12-31"))
    assert rows[-1]["date"] > spec["first_non_member_day"]


def test_delisted_series_ends() -> None:
    spec = fake.PLANTED["delisted"]
    rows = client.get_eod_prices(spec["ticker"], START, END)
    assert rows[-1]["date"] == spec["last_trading_day"]
    raw = client._fetch_json("/fundamentals/GSPC.INDX", {"filter": "HistoricalTickerComponents"})
    flags = [r for r in raw.values() if r["Code"] == spec["ticker"]]
    assert flags and flags[0]["IsDelisted"] == 1 and flags[0]["IsActiveNow"] == 0
    listed = {r["Code"] for r in client.get_delisted_symbols()}
    assert spec["ticker"] in listed


def test_membership_has_holdout_rows_rejoin_and_stable_size() -> None:
    # The client clips the holdout, so read the fake transport directly.
    url = client.BASE_URL + "/fundamentals/GSPC.INDX"
    query = {"filter": "HistoricalTickerComponents", "api_token": fake.FAKE_PAID_TOKEN}
    raw = list(fake.FakeEodhdTransport().get(url, params=query).json().values())
    assert any(r["EndDate"] and r["EndDate"] >= "2024-01-01" for r in raw)
    assert any(r["StartDate"] >= "2024-01-01" for r in raw)
    codes = [r["Code"] for r in raw]
    assert len(codes) != len(set(codes))  # a rejoining name has two rows
    history = membership()
    sizes = {
        len(history.members_on(day)) for day in (d("2016-06-01"), d("2020-06-01"), d("2023-06-01"))
    }
    assert all(55 <= size <= 65 for size in sizes)
    holdout_names = {r["Code"] for r in raw if r["StartDate"] >= "2024-01-01"}
    assert all(
        f"{c}.US" not in history.universe(config.MEMBERSHIP_START, END) for c in holdout_names
    )


def test_gap_down_without_news_has_zero_articles_in_window() -> None:
    spec = fake.PLANTED["gap_down_no_news"]
    assert _gap(spec["ticker"], spec["date"]) == pytest.approx(spec["gap"], abs=1e-3)
    assert _window_count(spec["ticker"], spec["date"]) == 0
    assert (
        _window_count(spec["ticker"], "2022-02-15") > 0
        or _window_count(spec["ticker"], "2022-02-16") > 0
    )


def test_gap_down_with_news_has_articles_in_window() -> None:
    spec = fake.PLANTED["gap_down_with_news"]
    assert _gap(spec["ticker"], spec["date"]) == pytest.approx(spec["gap"], abs=1e-3)
    assert _window_count(spec["ticker"], spec["date"]) >= spec["min_articles"]


def _gap(ticker: str, day: str) -> float:
    rows = {r["date"]: r for r in client.get_eod_prices(ticker, START, END)}
    dates = sorted(rows)
    previous = dates[dates.index(day) - 1]
    return float(rows[day]["open"] / rows[previous]["close"] - 1)


def _window_count(ticker: str, day: str) -> int:
    calendar = pd.DatetimeIndex(
        [pd.Timestamp(r["date"]) for r in client.get_eod_prices("SPY", START, END)]
    )
    rows = client.get_news(ticker, d("2022-01-01"), d("2022-12-31"))
    articles = parse_news_rows(rows, f"{ticker}.US")  # type: ignore[arg-type]
    return summarize_window(articles, d(day), calendar).article_count


def test_earnings_after_market_reaction_session() -> None:
    spec = fake.PLANTED["earnings_after_market"]
    calendar = pd.DatetimeIndex(
        [pd.Timestamp(r["date"]) for r in client.get_eod_prices("SPY", START, END)]
    )
    rows = client.get_earnings(spec["ticker"], d("2021-01-01"), d("2021-12-31"))
    row = next(r for r in rows if r["report_date"] == spec["report_date"])
    assert row["before_after_market"] == "AfterMarket"
    assert row["date"] == spec["period_end"]
    events = parse_earnings_rows(rows, calendar)  # type: ignore[arg-type]
    assert d(spec["reaction_session"]) in [e.reaction_session for e in events]
    assert reaction_session(d(spec["report_date"]), "AfterMarket", calendar) == d(
        spec["reaction_session"]
    )
    timings = {r["before_after_market"] for r in client.get_earnings("SIM010", START, END)}
    assert {"BeforeMarket", "AfterMarket"} <= timings


def test_known_price_gaps_exist() -> None:
    dates = {r["date"] for r in client.get_eod_prices("SIM020", START, END)}
    assert "2017-03-14" not in dates


def test_holdout_guard_still_raises_in_fake_mode() -> None:
    with pytest.raises(HoldoutAccessError):
        client.get_eod_prices("SIM001", START, date(2024, 1, 5))
    with pytest.raises(HoldoutAccessError):
        client.get_news("SIM001", START, date(2024, 1, 5))
    assert all(r["date"] < "2024-01-01" for r in client.get_eod_prices("SIM001", START, END))


def test_fake_cache_is_separate_from_real(tmp_path: Path) -> None:
    client.get_splits("SIM003")
    assert (tmp_path / "fake" / "splits" / "SIM003.US" / "to=2023-12-31.json").exists()
    assert client.FAKE_CACHE_DIR != client.DEFAULT_CACHE_DIR
    assert "fake" in client.FAKE_CACHE_DIR.parts


def test_fake_plans_and_errors() -> None:
    transport = fake.FakeEodhdTransport()
    base = client.BASE_URL
    bad = transport.get(base + "/user", params={"api_token": fake.FAKE_BAD_TOKEN})
    assert bad.status_code == 401
    forbidden = transport.get(
        base + "/calendar/earnings", params={"api_token": fake.FAKE_FREE_TOKEN}
    )
    assert forbidden.status_code == 403
    paid_user = transport.get(base + "/user", params={"api_token": fake.FAKE_PAID_TOKEN}).json()
    assert paid_user["subscriptionType"] == "simulated"
    old = {"from": "2020-01-01", "to": "2020-02-01", "api_token": fake.FAKE_FREE_TOKEN}
    assert transport.get(base + "/eod/SIM001.US", params=old).json() == [
        {"warning": fake.FREE_PLAN_WARNING}
    ]


def test_free_plan_warning_goes_through_client_checks(tmp_path: Path) -> None:
    with pytest.raises(client.EodhdPlanLimitError):
        client._fetch_json(
            "/eod/SIM001.US",
            {"from": "2020-01-01", "to": "2020-02-01"},
            cache_dir=tmp_path,
            api_token="fake-free",
        )
    assert not list(tmp_path.rglob("*.json"))
    with pytest.raises(client.EodhdHttpError) as excinfo:
        client._fetch_json("/user", {}, cache_dir=tmp_path, api_token="fake-bad")
    assert excinfo.value.status_code == 401
    with pytest.raises(client.EodhdHttpError):
        client._fetch_json(
            "/fundamentals/GSPC.INDX",
            {"filter": "Components"},
            cache_dir=tmp_path,
            api_token="fake-free",
        )


def test_unverified_shapes_are_listed() -> None:
    assert len(fake.UNVERIFIED_SHAPES) >= 5
    text = " ".join(fake.UNVERIFIED_SHAPES)
    assert "HistoricalTickerComponents" in text and "symbol-change-history" in text
    assert set(fake.fake_shape_examples()) == set(fake.ENDPOINT_KEYS)


def test_data_mode_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EODHD_MODE", raising=False)
    monkeypatch.setenv("EODHD_API_KEY", "")
    assert config.data_mode() == "fake"
    monkeypatch.setenv("EODHD_API_KEY", "abc")
    assert config.data_mode() == "live"
    monkeypatch.setenv("EODHD_MODE", "fake")
    assert config.data_mode() == "fake"
    monkeypatch.setenv("EODHD_MODE", "live")
    monkeypatch.setenv("EODHD_API_KEY", "")
    assert config.data_mode() == "live"
    monkeypatch.setenv("EODHD_MODE", "oops")
    with pytest.raises(ValueError):
        config.data_mode()


def test_output_dir_and_title(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config.output_dir().parts[-2:] == ("outputs", "simulated")
    assert config.report_title("Report").startswith(config.SIMULATED_LABEL)
    monkeypatch.setenv("EODHD_MODE", "live")
    assert config.output_dir().parts[-2:] == ("outputs", "live")
    assert config.report_title("Report") == "Report"
