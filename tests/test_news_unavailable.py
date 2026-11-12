"""Reused-ticker codes (APC_old) have no news: unknown, never zero (D26). Fake mode, temp cache."""

import importlib.util
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot.data import eodhd_client as client
from bot.data import market
from bot.data.fake_eodhd import FakeEodhdTransport
from bot.strategy.schema import strategy_from_json
from tests.backtest_helpers import (
    Bar,
    assemble_market,
    flat_bar,
    raw_frame,
    run_toy,
    toy_days,
)

OLD = "SIM005_old"
START, END = date(2023, 1, 1), date(2023, 3, 31)


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def count_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    original = client._http_get

    def spy(url: str, query: dict[str, Any], token: str) -> Any:
        seen.append(f"{url}|{query.get('s', '')}")
        return original(url, query, token)

    monkeypatch.setattr(client, "_http_get", spy)
    return seen


def load_downloader() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data_news_unavailable", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fake_returns_422_for_old_codes_on_news_and_sentiments() -> None:
    transport = FakeEodhdTransport()
    for path in ("/api/news", "/api/sentiments"):
        response = transport.get(f"https://x{path}", params={"s": f"{OLD}.US", "api_token": "x"})
        assert response.status_code == 422
        assert "valid ticker symbol" in response.json()["errors"]["s"][0]


def test_client_raises_a_clear_error_and_caches_a_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = count_calls(monkeypatch)
    with pytest.raises(client.NewsSymbolUnsupportedError) as caught:
        client.get_news(OLD, START, END)
    assert isinstance(caught.value, client.EodhdHttpError)
    assert caught.value.status_code == 422
    assert client.is_news_unavailable(OLD)
    first = len(calls)
    with pytest.raises(client.NewsSymbolUnsupportedError):
        client.get_news(OLD, START, END)
    with pytest.raises(client.NewsSymbolUnsupportedError):
        client.get_daily_sentiment(OLD, START, END)
    assert len(calls) == first  # reruns cost nothing


def test_marker_is_not_an_empty_news_list(fake_mode: Path) -> None:
    with pytest.raises(client.NewsSymbolUnsupportedError):
        client.get_news(OLD, START, END)
    news_dir = fake_mode / "news"
    assert not news_dir.exists() or not list(news_dir.glob("*.json"))
    # Offline with the marker: still unavailable, never [].
    with pytest.raises(client.NewsSymbolUnsupportedError):
        client.get_news(OLD, START, END, offline=True)


def test_ordinary_tickers_are_not_marked() -> None:
    assert client.get_news("SIM005", START, END) is not None
    assert not client.is_news_unavailable("SIM005")


def test_download_marks_unavailable_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "PRICE_START", date(2022, 1, 1))
    monkeypatch.setattr(downloader, "NEWS_START", START)
    monkeypatch.setattr(downloader, "BACKTEST_END", END)
    rows: Counter[str] = Counter()
    unavailable: list[str] = []
    assert downloader.download_ticker(OLD, rows, unavailable) is True
    assert unavailable == [OLD]
    assert rows["prices"] > 0 and rows["news"] == 0 and rows["sentiment"] == 0
    calls = count_calls(monkeypatch)
    again: list[str] = []
    downloader.download_ticker(OLD, Counter(), again)
    assert again == [OLD]
    assert [c for c in calls if "/news" in c or "/sentiments" in c] == []


def test_loader_gives_nan_news_for_unavailable_symbol() -> None:
    days = pd.bdate_range("2023-01-03", periods=10)
    frame = market._news_frame(OLD, f"{OLD}.US", days, START, END, False)
    assert len(frame) == len(days)
    assert frame.isna().all().all()
    normal = market._news_frame("SIM005", "SIM005.US", days, START, END, False)
    assert normal["article_count"].notna().any()


def test_gap_down_no_news_never_enters_a_news_unavailable_symbol() -> None:
    days = toy_days(8)
    gap = {
        0: flat_bar(100.0),
        1: Bar(95.0, 96.0, 94.0, 95.0),
        **{n: flat_bar(95.0) for n in range(2, 8)},
    }
    frames = {s: raw_frame(days, gap) for s in ("KNOWN.US", "OLD.US")}
    zero = pd.DataFrame({"KNOWN.US": 0.0, "OLD.US": float("nan")}, index=days)
    unknown = market.unknown_news_frame(days)["article_count"]
    assert unknown.isna().all()
    count = zero.copy()
    count["OLD.US"] = unknown
    stock = assemble_market(days, frames, article_count=count, mean_polarity=count)
    strategy = strategy_from_json(
        json.dumps(
            {
                "name": "Gap down quiet",
                "description_plain_english": "Gap down with no news.",
                "entry": [
                    {"block": "gap", "direction": "down", "min_gap_pct": 3},
                    {"block": "no_news", "lookback_sessions": 1},
                ],
                "exit": {"holding_days": 2},
                "sizing": {"max_positions": 10},
            }
        )
    )
    symbols = {t.symbol for t in run_toy(strategy, stock).trades}
    assert symbols == {"KNOWN.US"}
