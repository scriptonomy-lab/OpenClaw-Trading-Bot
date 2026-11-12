"""Sample universe (D33): loader, download script, call estimate and budget guard. Fake mode."""

import importlib.util
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import bot.universe as universe
from bot.data import eodhd_client as client
from bot.data import market

SAMPLE = ["SIM001", "SIM002", "SIM003"]
START, END = date(2022, 1, 1), date(2022, 3, 31)
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def write_sample(path: Path, tickers: list[str], *, sha: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "seed": 1,
                "population": 80,
                "strata": {"active": len(tickers)},
                "tickers": tickers,
                "sha256": sha or universe.tickers_hash(tickers),
            }
        )
    )
    return path


_real_load_sample = universe.load_sample


@pytest.fixture(autouse=True)
def fake_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", tmp_path / "cache")
    path = write_sample(tmp_path / "sample.json", SAMPLE)
    monkeypatch.setattr(universe, "SAMPLE_PATH", path)
    # load_sample's default argument was bound at import, so wrap it with the temp path.
    monkeypatch.setattr(universe, "load_sample", lambda p=path: _real_load_sample(p))
    monkeypatch.setattr(market, "load_sample", universe.load_sample)


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_universe_test", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_mode_loads_only_sample_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EODHD_UNIVERSE", "sample")
    data, report = market.load_market_report(START, END)
    symbols = {s for s in data.close.columns}
    assert symbols <= {f"{t}.US" for t in SAMPLE}
    assert symbols
    # Membership columns are the sample too: tradable = sample and that day's members.
    assert set(data.membership.columns) <= {f"{t}.US" for t in SAMPLE}
    assert report.universe_mode == "sample"
    assert "Random sample of 3 of 80" in report.universe_label
    assert report.universe_size <= len(SAMPLE)
    assert report.to_dict()["universe_mode"] == "sample"


def test_full_mode_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EODHD_UNIVERSE", "full")
    data, report = market.load_market_report(START, END)
    assert len(data.close.columns) > len(SAMPLE)
    assert report.universe_mode == "full"
    assert report.universe_label.startswith("Full universe")


def test_sample_membership_is_sample_intersect_real_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_UNIVERSE", "full")
    full, _ = market.load_market_report(START, END)
    monkeypatch.setenv("EODHD_UNIVERSE", "sample")
    sample, _ = market.load_market_report(START, END)
    columns = list(sample.membership.columns)
    assert columns
    assert sample.membership[columns].equals(full.membership[columns])


def test_hash_mismatch_refuses_to_load(tmp_path: Path) -> None:
    path = write_sample(tmp_path / "bad.json", SAMPLE, sha="0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        _real_load_sample(path)


def test_prices_only_fetches_no_news(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = load_script("download_data")
    seen: list[str] = []
    original = client._http_get

    def spy(url: str, query: dict[str, Any], token: str) -> Any:
        seen.append(url)
        return original(url, query, token)

    monkeypatch.setattr(client, "_http_get", spy)
    rows: Counter[str] = Counter()
    assert downloader.download_ticker("SIM001", rows, prices_only=True)
    joined = " ".join(seen)
    assert "/eod/" in joined
    for forbidden in ("/news", "/sentiments", "/calendar"):
        assert forbidden not in joined


def test_estimate_is_shown_and_stops_above_the_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloader = load_script("download_data")
    monkeypatch.setattr(downloader, "load_universe", lambda **k: ["SIM001", "SIM002"])
    with pytest.raises(downloader.EstimateTooHighError):
        downloader.run_download(None, mode="sample", cap=10)
    assert "Estimated API calls" in capsys.readouterr().out


def test_budget_guard_stops_above_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = load_script("download_data")
    used = iter([100, 150, 400])
    guard = downloader.BudgetGuard(lambda: next(used), cap=200)
    assert guard.check() == 50
    with pytest.raises(downloader.BudgetExceededError):
        guard.check()


def test_budget_guard_survives_a_daily_reset() -> None:
    downloader = load_script("download_data")
    readings = iter([900, 950, 20])
    guard = downloader.BudgetGuard(lambda: next(readings), cap=1000)
    guard.check()
    assert guard.check() == 70


def test_run_stops_cleanly_when_budget_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = load_script("download_data")
    monkeypatch.setattr(downloader, "load_universe", lambda **k: ["SIM001", "SIM002"])
    monkeypatch.setattr(downloader, "BUDGET_CHECK_EVERY", 1)
    readings = iter([0, 5, 50_000, 50_000])
    monkeypatch.setattr(client, "api_requests_used", lambda **k: next(readings))
    with pytest.raises(downloader.BudgetExceededError):
        downloader.run_download(None, mode="sample", prices_only=True)


def test_realistic_estimates() -> None:
    estimate = load_script("estimate_api_calls")
    sample = sum(x.high for x in estimate.realistic_lines(200))
    full_prices = sum(x.high for x in estimate.realistic_lines(741, prices_only=True))
    assert 5_000 < sample < estimate.CALL_CAP
    assert full_prices < sample
    names = {x.name for x in estimate.realistic_lines(10, prices_only=True)}
    assert "News" not in names


def test_fake_mode_defaults_to_full_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake tickers are not in the real sample, so fake mode must not default to it."""
    from bot.universe import universe_mode

    monkeypatch.delenv("EODHD_UNIVERSE", raising=False)
    monkeypatch.setenv("EODHD_MODE", "fake")
    assert universe_mode() == "full"
    monkeypatch.setenv("EODHD_UNIVERSE", "sample")
    assert universe_mode() == "sample"
