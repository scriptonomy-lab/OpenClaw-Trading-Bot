"""Tests of the EODHD client mechanics: cache keys, offline mode, no-network reads."""

import json
from pathlib import Path
from typing import Any

import pytest

from bot.data import eodhd_client as client


def test_cache_path_excludes_token_and_is_deterministic(tmp_path: Path) -> None:
    one = client.cache_path(
        "/eod/AAPL.US", {"to": "b", "from": "a", "api_token": "SECRET"}, tmp_path
    )
    two = client.cache_path("/eod/AAPL.US", {"from": "a", "to": "b"}, tmp_path)
    assert one == two
    assert "SECRET" not in str(one)
    assert one.name == "from=a_to=b.json"


def test_offline_cache_miss_raises(tmp_path: Path) -> None:
    with pytest.raises(client.OfflineCacheMissError):
        client._fetch_json("/splits/AAPL.US", {}, cache_dir=tmp_path, offline=True)


def test_cached_read_makes_no_network_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = client.cache_path("/splits/AAPL.US", {}, tmp_path)
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps([{"date": "2020-08-31", "split": "4.000000/1.000000"}]))

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network call made")

    monkeypatch.setattr(client.requests, "get", fail)
    data = client._fetch_json("/splits/AAPL.US", {}, cache_dir=tmp_path)
    assert data[0]["split"] == "4.000000/1.000000"


def test_token_never_written_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self) -> list[int]:
            return [1]

    monkeypatch.setattr(client.requests, "get", lambda *a, **k: FakeResponse())
    client._fetch_json("/x", {"a": 1}, cache_dir=tmp_path, api_token="SECRET")
    for file in tmp_path.rglob("*"):
        assert "SECRET" not in str(file)
        if file.is_file():
            assert "SECRET" not in file.read_text()


def test_403_error_mentions_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Forbidden:
        status_code = 403

    monkeypatch.setattr(client.requests, "get", lambda *a, **k: Forbidden())
    with pytest.raises(client.EodhdHttpError, match="plan or token") as info:
        client._fetch_json("/x", {}, cache_dir=tmp_path, api_token="demo")
    assert info.value.status_code == 403


def test_plan_warning_raises_and_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free plan returns 200 with a warning row. That must not be cached as data."""

    class FakeResponse:
        status_code = 200

        def json(self) -> list[dict[str, str]]:
            return [{"warning": "Data is limited by one year as you have free subscription"}]

    monkeypatch.setattr(client.requests, "get", lambda *args, **kwargs: FakeResponse())
    params = {"from": "2020-08-25", "to": "2020-09-02"}
    with pytest.raises(client.EodhdPlanLimitError):
        client._fetch_json("/eod/AAPL.US", params, cache_dir=tmp_path, api_token="x")
    assert not client.cache_path("/eod/AAPL.US", params, tmp_path).exists()
