"""Tests for the phase 1 review fixes in the EODHD client (H1, L1, L2, M5, M6, L11)."""

import importlib.util
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from bot.data import eodhd_client as client
from bot.data import holdout
from bot.data.holdout import HoldoutAccessError, HoldoutApproval, holdout_session

SECRET = "SECRET-TOKEN-123"


class _Response:
    status_code = 200

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def json(self) -> Any:
        return self.payload


def _serve(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[str, dict[str, Any]], Any]
) -> list[dict[str, Any]]:
    """Replace requests.get. Returns the list of query dicts it was called with."""
    seen: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _Response:
        seen.append({"url": url, **params})
        return _Response(answer(url, params))

    monkeypatch.setattr(client.requests, "get", fake_get)
    return seen


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network was used")

    monkeypatch.setattr(client.requests, "get", fail)


@pytest.fixture
def open_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let holdout_session open for any strategy."""
    monkeypatch.setattr(
        holdout,
        "find_approval",
        lambda sid: HoldoutApproval(sid, datetime(2025, 1, 1, tzinfo=UTC), False),
    )
    monkeypatch.setattr(holdout, "mark_approval_used", lambda sid: None)


# H1 ------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["from", "to", "effective"])
def test_fetch_json_refuses_2024_param_outside_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str
) -> None:
    _no_network(monkeypatch)
    with pytest.raises(HoldoutAccessError):
        client._fetch_json("/x", {key: "2024-01-01"}, cache_dir=tmp_path, api_token="x")
    assert not list(tmp_path.rglob("*.json"))


def test_fetch_json_is_the_only_private_door() -> None:
    assert not hasattr(client, "fetch_json")


def test_splits_request_is_date_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _serve(monkeypatch, lambda url, params: [{"date": "2020-08-31", "split": "4/1"}])
    client.get_splits("AAPL", cache_dir=tmp_path, api_token="x")
    assert seen[0]["to"] == "2023-12-31"
    assert (tmp_path / "splits" / "AAPL.US" / "to=2023-12-31.json").exists()


def test_splits_in_session_asks_for_full_history_in_holdout_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, open_session: None
) -> None:
    rows = [{"date": "2020-08-31", "split": "4/1"}, {"date": "2025-03-03", "split": "2/1"}]
    seen = _serve(monkeypatch, lambda url, params: rows)
    base = tmp_path / "eodhd"
    with holdout_session("s1"):
        got = client.get_splits("AAPL", approved=True, cache_dir=base, api_token="x")
        assert len(got) == 2
        assert "to" not in seen[0]
        assert list((tmp_path / "eodhd-holdout").rglob("*.json"))
    assert not base.exists()


def test_session_cache_dir_removed_after_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, open_session: None
) -> None:
    _serve(monkeypatch, lambda url, params: [{"date": "2024-02-01", "close": 1}])
    base = tmp_path / "eodhd"
    with holdout_session("s1"):
        client.get_eod_prices(
            "AAPL", date(2024, 1, 1), date(2024, 3, 1), approved=True, cache_dir=base, api_token="x"
        )
        assert (tmp_path / "eodhd-holdout").exists()
    assert not (tmp_path / "eodhd-holdout").exists()
    assert not base.exists()


def test_session_cache_dir_removed_after_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, open_session: None
) -> None:
    _serve(monkeypatch, lambda url, params: [{"a": 1}])
    with pytest.raises(RuntimeError), holdout_session("s1"):
        client._fetch_json("/x", {"from": "2024-01-01"}, cache_dir=tmp_path / "e", api_token="x")
        raise RuntimeError("boom")
    assert not (tmp_path / "e-holdout").exists()


def test_index_history_clips_start_and_end_before_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = {
        "0": {"Code": "A", "StartDate": "2015-01-01", "EndDate": "2024-05-05"},
        "1": {"Code": "B", "StartDate": "2024-02-02", "EndDate": None},
        "2": {"Code": "C", "StartDate": "2016-01-01", "EndDate": "2020-01-01"},
    }
    _serve(monkeypatch, lambda url, params: raw)
    rows = client.get_index_history(cache_dir=tmp_path, api_token="x")
    assert [r["Code"] for r in rows] == ["A", "C"]
    assert rows[0]["EndDate"] is None
    assert rows[1]["EndDate"] == "2020-01-01"
    text = "".join(f.read_text() for f in tmp_path.rglob("*.json"))
    assert "2024" not in text


def test_index_history_clips_old_unclipped_cache_on_read(tmp_path: Path) -> None:
    file = client.cache_path(
        f"/fundamentals/{client.INDEX_SYMBOL}", {"filter": "HistoricalTickerComponents"}, tmp_path
    )
    file.parent.mkdir(parents=True)
    file.write_text(
        json.dumps({"0": {"Code": "A", "StartDate": "2015-01-01", "EndDate": "2025-01-01"}})
    )
    rows = client.get_index_history(cache_dir=tmp_path, offline=True, api_token="x")
    assert rows[0]["EndDate"] is None


# M5 / L11 ------------------------------------------------------------------


def _article(index: int) -> dict[str, Any]:
    return {"date": f"2023-01-03T10:{index:02d}:00+00:00", "title": f"t{index}"}


def _paged_news(total: int) -> Callable[[str, dict[str, Any]], Any]:
    def answer(url: str, params: dict[str, Any]) -> Any:
        start = params["offset"]
        return [_article(i) for i in range(start, min(total, start + params["limit"]))]

    return answer


def test_news_pages_are_joined_and_cached_as_one_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _serve(monkeypatch, _paged_news(5))
    rows = client.get_news(
        "AAPL", date(2023, 1, 3), date(2023, 1, 4), page_size=2, cache_dir=tmp_path, api_token="x"
    )
    assert [r["title"] for r in rows] == [f"t{i}" for i in range(5)]
    assert [s["offset"] for s in seen] == [0, 2, 4]
    files = list((tmp_path / "news").glob("*.json"))
    assert len(files) == 1
    assert "offset" not in files[0].name
    _no_network(monkeypatch)
    again = client.get_news(
        "AAPL", date(2023, 1, 3), date(2023, 1, 4), page_size=2, cache_dir=tmp_path, api_token="x"
    )
    assert again == rows


def test_news_offset_cap_raises_and_caches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(client, "NEWS_MAX_WINDOW", 4)
    _serve(monkeypatch, _paged_news(100))
    with pytest.raises(client.NewsWindowTooLargeError):
        client.get_news(
            "AAPL",
            date(2023, 1, 3),
            date(2023, 1, 4),
            page_size=2,
            cache_dir=tmp_path,
            api_token="x",
        )
    assert not list(tmp_path.rglob("*.json"))


def test_news_max_pages_on_full_page_raises_unless_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, _paged_news(100))
    window = (date(2023, 1, 3), date(2023, 1, 4))
    with pytest.raises(client.NewsTruncatedError):
        client.get_news(
            "AAPL", *window, page_size=2, max_pages=2, cache_dir=tmp_path, api_token="x"
        )
    rows = client.get_news(
        "AAPL",
        *window,
        page_size=2,
        max_pages=2,
        allow_truncated=True,
        cache_dir=tmp_path,
        api_token="x",
    )
    assert len(rows) == 4
    # A capped window never answers for a full one.
    full = client.cache_path(
        "/news", {"s": "AAPL.US", "from": "2023-01-03", "to": "2023-01-04"}, tmp_path
    )
    assert not full.exists()


def test_news_max_pages_not_full_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, _paged_news(3))
    rows = client.get_news(
        "AAPL",
        date(2023, 1, 3),
        date(2023, 1, 4),
        page_size=2,
        max_pages=5,
        cache_dir=tmp_path,
        api_token="x",
    )
    assert len(rows) == 3


# M6 ------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["errors", "error", "message"])
def test_error_body_raises_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str
) -> None:
    _serve(monkeypatch, lambda url, params: {key: "boom"})
    with pytest.raises(client.EodhdBadResponseError):
        client.get_earnings(
            "AAPL", date(2020, 1, 1), date(2020, 12, 31), cache_dir=tmp_path, api_token="x"
        )
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    ("call", "payload"),
    [
        (
            lambda d: client.get_daily_sentiment(
                "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=d, api_token="x"
            ),
            [],
        ),
        (
            lambda d: client.get_daily_sentiment(
                "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=d, api_token="x"
            ),
            {"OTHER.US": []},
        ),
        (
            lambda d: client.get_earnings(
                "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=d, api_token="x"
            ),
            [],
        ),
        (
            lambda d: client.get_earnings(
                "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=d, api_token="x"
            ),
            {"earnings": None},
        ),
        (lambda d: client.get_splits("AAPL", cache_dir=d, api_token="x"), {"a": 1}),
        (
            lambda d: client.get_dividends(
                "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=d, api_token="x"
            ),
            {"a": 1},
        ),
        (lambda d: client.get_delisted_symbols(cache_dir=d, api_token="x"), {"a": 1}),
        (lambda d: client.get_index_history(cache_dir=d, api_token="x"), []),
    ],
)
def test_wrong_shape_raises_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, call: Callable[[Path], Any], payload: Any
) -> None:
    _serve(monkeypatch, lambda url, params: payload)
    with pytest.raises(client.EodhdBadResponseError):
        call(tmp_path)
    assert not list(tmp_path.rglob("*.json"))


def test_empty_eod_raises_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, lambda url, params: [])
    with pytest.raises(client.EodhdEmptyResponseError):
        client.get_eod_prices(
            "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=tmp_path, api_token="x"
        )
    assert not list(tmp_path.rglob("*.json"))


def test_empty_splits_and_dividends_are_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, lambda url, params: [])
    assert client.get_splits("AAPL", cache_dir=tmp_path, api_token="x") == []
    assert (
        client.get_dividends(
            "AAPL", date(2020, 1, 1), date(2020, 2, 1), cache_dir=tmp_path, api_token="x"
        )
        == []
    )


def _load_downloader() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_downloader_records_empty_eod_as_no_price_data(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = _load_downloader()

    def empty(*args: Any, **kwargs: Any) -> None:
        raise client.EodhdEmptyResponseError("empty")

    monkeypatch.setattr(downloader.client, "get_eod_prices", empty)
    rows: Counter[str] = Counter()
    assert downloader.download_ticker("GONE", rows) is False
    assert not rows


# L1 ------------------------------------------------------------------------


def test_network_error_hides_token_everywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(url: str, params: dict[str, Any], timeout: float) -> None:
        raise requests.ConnectionError(f"failed {url}?api_token={params['api_token']}")

    monkeypatch.setattr(client.requests, "get", boom)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    with pytest.raises(client.EodhdError) as info:
        client._fetch_json("/x", {}, cache_dir=tmp_path, api_token=SECRET)
    error = info.value
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert error.__context__ is None
    assert error.__cause__ is None


def test_non_json_error_has_no_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Bad:
        status_code = 200

        def json(self) -> None:
            raise ValueError(SECRET)

    monkeypatch.setattr(client.requests, "get", lambda *a, **k: Bad())
    with pytest.raises(client.EodhdError) as info:
        client._fetch_json("/x", {}, cache_dir=tmp_path, api_token=SECRET)
    assert info.value.__context__ is None
    assert SECRET not in repr(info.value)


def test_urllib3_logger_is_not_debug() -> None:
    assert logging.getLogger("urllib3").getEffectiveLevel() >= logging.WARNING
