"""The loader rebuilds news for busy tickers from the halves the download stored."""

from datetime import date, timedelta
from typing import Any

import pytest

from bot.data import eodhd_client as client
from bot.data import market

FIRST = date(2015, 1, 1)
END = date(2015, 1, 31)


def fake_get_news(cached: dict[tuple[date, date], list[dict[str, Any]]], calls: list[str]) -> Any:
    """A get_news stand-in that serves only the windows in ``cached``."""

    def get_news(
        ticker: str, start: date, end: date, *, approved: bool = False, offline: bool = False
    ) -> list[dict[str, Any]]:
        if (start, end) in cached:
            return cached[(start, end)]
        if offline:
            raise client.OfflineCacheMissError("not cached")
        calls.append("network")
        return [{"date": "network"}]

    return get_news


def test_split_halves_are_joined_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    middle = date.fromordinal((FIRST.toordinal() + END.toordinal()) // 2)
    cached = {
        (FIRST, middle): [{"date": "a"}],
        (middle + timedelta(days=1), END): [{"date": "b"}],
    }
    calls: list[str] = []
    monkeypatch.setattr(client, "get_news", fake_get_news(cached, calls))
    rows = market._news_rows("AAPL", FIRST, END, False)
    assert rows == [{"date": "a"}, {"date": "b"}]
    assert calls == []


def test_whole_window_in_cache_is_used_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(client, "get_news", fake_get_news({(FIRST, END): [{"date": "x"}]}, calls))
    assert market._news_rows("AAPL", FIRST, END, False) == [{"date": "x"}]
    assert calls == []


def test_uncached_window_falls_back_to_one_normal_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(client, "get_news", fake_get_news({}, calls))
    assert market._news_rows("AAPL", FIRST, END, False) == [{"date": "network"}]
    assert calls == ["network"]
