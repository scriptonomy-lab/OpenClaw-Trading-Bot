"""Fast tests for the go-live command. No network: everything uses the fake transport."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bot import verify_live as vl
from bot.data.fake_eodhd import FAKE_BAD_TOKEN, FAKE_FREE_TOKEN, FAKE_PAID_TOKEN


def make_run(
    tmp_path: Path, *, dry_run: bool = True, token: str = FAKE_PAID_TOKEN, **kw: Any
) -> vl.Run:
    """A Run that writes under tmp_path and uses the fake transport."""
    return vl.Run(
        dry_run=dry_run,
        assume_yes=kw.get("assume_yes", False),
        token=token,
        stamp="test",
        out_dir=tmp_path / "verify",
        input_fn=kw.get("input_fn", input),
    )


def kinds(diffs: list[vl.ShapeDiff]) -> list[str]:
    return [d.kind for d in diffs]


# ------------------------------------------------------------------ shape diff


def test_diff_match_ignores_values_and_int_float() -> None:
    assert vl.diff_shape({"a": 1, "b": "x"}, {"a": 2.5, "b": "y"}) == []


def test_diff_missing_key() -> None:
    diffs = vl.diff_shape({"a": 1, "b": 2}, {"a": 1})
    assert kinds(diffs) == ["missing_key"]
    assert diffs[0].path == "$.b"


def test_diff_extra_key() -> None:
    diffs = vl.diff_shape({"a": 1}, {"a": 1, "z": 3})
    assert kinds(diffs) == ["extra_key"]
    assert diffs[0].path == "$.z"


def test_diff_type_change() -> None:
    diffs = vl.diff_shape({"a": 1}, {"a": "1"})
    assert kinds(diffs) == ["type_change"]
    assert "expected number, got str" in diffs[0].detail


def test_diff_bool_is_not_a_number() -> None:
    assert kinds(vl.diff_shape({"a": 1}, {"a": True})) == ["type_change"]


def test_diff_nested_dict_and_first_list_record() -> None:
    expected = {"rows": [{"id": 1, "meta": {"k": "v"}}]}
    actual = {"rows": [{"id": 1, "meta": {"k": 5}}, {"other": 1}]}
    diffs = vl.diff_shape(expected, actual)
    assert [d.path for d in diffs] == ["$.rows[0].meta.k"]
    assert kinds(diffs) == ["type_change"]


def test_diff_null_difference_is_a_warning_not_fatal() -> None:
    diffs = vl.diff_shape({"end": None}, {"end": "2020-01-01"})
    assert kinds(diffs) == ["null_difference"]
    assert not diffs[0].fatal


def test_diff_empty_real_list_is_fatal() -> None:
    diffs = vl.diff_shape([{"a": 1}], [])
    assert kinds(diffs) == ["empty_real"]
    assert diffs[0].fatal


def test_compare_keyed_collection_ignores_outer_keys() -> None:
    expected = {"0": {"Code": "A"}, "1": {"Code": "B"}}
    assert vl.compare_endpoint("fundamentals_history", expected, 200, {"17": {"Code": "Z"}}) == []
    as_list = vl.compare_endpoint("fundamentals_history", expected, 200, [{"Code": "Z"}])
    assert kinds(as_list) == ["type_change"]


def test_compare_http_error() -> None:
    diffs = vl.compare_endpoint("eod", [{"a": 1}], 403, {"message": "no"})
    assert kinds(diffs) == ["http_error"]


def test_fakes_match_themselves_on_every_endpoint(tmp_path: Path) -> None:
    results = vl.check_endpoints(make_run(tmp_path))
    assert [r.spec.key for r in results if r.diffs] == []


def test_shape_mismatch_fails_loudly_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = vl.api_get

    def changed(run: vl.Run, path: str, params: dict[str, Any]) -> tuple[int, Any]:
        status, body = real_get(run, path, params)
        if path == "/user":
            body = {**body, "surprise": 1}
        return status, body

    monkeypatch.setattr(vl, "api_get", changed)
    run = make_run(tmp_path)
    with pytest.raises(vl.VerifyError, match="differ from the fakes"):
        vl.step_shapes(run)
    report = (run.out_dir / "shapes.md").read_text()
    assert "$.surprise" in report and "extra_key" in report


def test_verified_lines_only_for_clean_endpoints() -> None:
    spec = vl.EndpointSpec("fundamentals_components", "/x", {})
    clean = vl.EndpointResult(spec, 200, ())
    lines = vl.verified_lines([clean], vl.date(2026, 11, 2))
    assert any(
        "verified live on 2026-11-02" in line and "filter=Components" in line for line in lines
    )
    assert any(
        "STILL UNVERIFIED" in line and "HistoricalTickerComponents" in line for line in lines
    )


# ------------------------------------------------------------------ plan


def test_plan_paid_prints_limits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vl.step_plan(make_run(tmp_path))
    out = capsys.readouterr().out
    assert "Daily limit: 100000" in out and "Calls used today" in out


def test_plan_free_stops(tmp_path: Path) -> None:
    with pytest.raises(vl.VerifyError, match="FREE plan"):
        vl.step_plan(make_run(tmp_path, token=FAKE_FREE_TOKEN))


def test_plan_401_stops(tmp_path: Path) -> None:
    with pytest.raises(vl.VerifyError, match="HTTP 401"):
        vl.step_plan(make_run(tmp_path, token=FAKE_BAD_TOKEN))


def test_api_get_refuses_holdout_dates(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="holdout"):
        vl.api_get(make_run(tmp_path), "/eod/SIM001.US", {"from": "2023-12-01", "to": "2024-01-05"})


# ------------------------------------------------------------------ confirmation gate


def live_run(tmp_path: Path, answer: str | None, *, yes: bool = False) -> vl.Run:
    def ask(prompt: str) -> str:
        assert "Type DOWNLOAD to start the full download" in prompt
        if answer is None:
            raise EOFError
        return answer

    return make_run(tmp_path, dry_run=False, assume_yes=yes, input_fn=ask)


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    calls: list[int | None] = []
    monkeypatch.setattr(vl, "run_full_download", lambda limit: calls.append(limit) or 0)
    return calls


@pytest.mark.parametrize("answer", ["", "download", "yes", "DOWNLOAD now", None])
def test_no_download_without_the_word(
    tmp_path: Path, downloads: list[int | None], answer: str | None
) -> None:
    with pytest.raises(vl.VerifyError):
        vl.step_download(live_run(tmp_path, answer))
    assert downloads == []


def test_download_runs_after_the_word(tmp_path: Path, downloads: list[int | None]) -> None:
    vl.step_download(live_run(tmp_path, "DOWNLOAD"))
    assert downloads == [None]


def test_yes_flag_skips_the_prompt(tmp_path: Path, downloads: list[int | None]) -> None:
    vl.step_download(live_run(tmp_path, None, yes=True))
    assert downloads == [None]


def test_dry_run_download_is_limited_to_ten_tickers(
    tmp_path: Path, downloads: list[int | None]
) -> None:
    vl.step_download(make_run(tmp_path))
    assert downloads == [10]


def test_failed_tickers_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vl, "run_full_download", lambda limit: 3)
    with pytest.raises(vl.VerifyError, match="3 ticker"):
        vl.step_download(live_run(tmp_path, "DOWNLOAD"))


# ------------------------------------------------------------------ command line


def test_refuses_fake_mode_without_dry_run_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(vl, "api_get", lambda *a, **k: pytest.fail("must not call EODHD"))
    assert vl.main([], output_root=tmp_path) == 2
    assert "Refusing to run" in capsys.readouterr().err
    assert not (tmp_path / "live").exists()


def test_a_failing_step_stops_the_run_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EODHD_MODE", "fake")
    later: list[str] = []

    def boom(run: vl.Run) -> None:
        raise vl.VerifyError("nope")

    monkeypatch.setattr(vl, "steps", lambda: [("a", boom), ("b", lambda run: later.append("b"))])
    assert vl.main(["--dry-run-fake"], output_root=tmp_path) == 1
    assert later == []


def test_dry_run_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every step on the fake transport. The pytest child process is skipped here."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(vl, "step_integration", lambda run: print("  (integration skipped)"))

    def no_prompt(prompt: str) -> str:
        raise AssertionError("a dry run must not ask for confirmation")

    code = vl.main(["--dry-run-fake"], output_root=tmp_path, input_fn=no_prompt)
    out = capsys.readouterr().out
    assert code == 0, out
    folders = list((tmp_path / "simulated" / "verify").iterdir())
    assert len(folders) == 1
    assert (folders[0] / "shapes.md").exists()
    trades = (folders[0] / "trade_checks.md").read_text()
    assert "SIMULATED DATA" in trades and "MISMATCH" not in trades
    assert "Result: all fields match." in trades
    assert not (tmp_path / "live").exists()


# ------------------------------------------------------------------ hand check helpers


def test_count_news_window_is_half_open_and_deduplicated() -> None:
    def article(moment: str, link: str, symbols: list[str]) -> dict[str, Any]:
        return {"date": moment, "link": link, "symbols": symbols}

    rows = [
        article("2023-01-03T21:00:00+00:00", "a", ["X.US"]),  # 16:00 New York: included
        article("2023-01-03T21:00:00+00:00", "a", ["X.US"]),  # same link: once
        article("2023-01-04T20:59:59+00:00", "b", ["X.US"]),  # just before the end: included
        article("2023-01-04T21:00:00+00:00", "c", ["X.US"]),  # at the end: excluded
        article("2023-01-03T20:59:59+00:00", "d", ["X.US"]),  # before the start: excluded
        article("2023-01-04T10:00:00+00:00", "e", ["X.MX"]),  # other listing: excluded
    ]
    start, end = vl.close_time("2023-01-03"), vl.close_time("2023-01-04")
    assert vl.count_news(rows, "X.US", start, end) == 2


def test_split_ratio_between_is_exclusive_then_inclusive() -> None:
    splits = [("2020-08-31", 4.0), ("2021-01-04", 2.0)]
    assert vl.split_ratio_between(splits, "2020-08-31", "2021-01-04") == 2.0
    assert vl.split_ratio_between(splits, "2020-08-28", "2021-01-04") == 8.0


def test_hand_exit_holding_period_and_stop_and_target() -> None:
    days = ["d0", "d1", "d2", "d3", "d4"]
    bars = {d: {"open": 100.0, "high": 101.0, "low": 99.0} for d in days}
    holding = {"reason": "holding_period", "holding_days": 3}
    assert vl.hand_exit(bars, days, 1, 100.0, holding, []) == ("d4", 100.0)
    bars["d2"] = {"open": 98.0, "high": 99.0, "low": 89.0}  # gaps through a 5% stop at 95
    stop = {"reason": "stop_loss", "stop_pct": 5.0, "target_pct": 10.0}
    assert vl.hand_exit(bars, days, 1, 100.0, stop, []) == ("d2", 95.0)
    bars["d2"] = {"open": 100.0, "high": 111.0, "low": 99.0}
    target = {"reason": "profit_target", "stop_pct": 5.0, "target_pct": 10.0}
    assert vl.hand_exit(bars, days, 1, 100.0, target, []) == ("d2", pytest.approx(110.0))


def test_compare_fields_flags_a_wrong_ledger_value() -> None:
    ledger = {"pnl": 10.0, "entry_date": "2020-01-02", "signal_values": {"gap_pct": -4.0}}
    hand = {"pnl": 10.0000001, "entry_date": "2020-01-03", "gap_pct": -4.0}
    rows = {name: same for name, _, _, same in vl.compare_fields(ledger, hand)}
    assert rows == {"pnl": True, "entry_date": False, "gap_pct": True}


def trade(symbol: str, day: str, reason: str = "holding_period", dividends: float = 0.0) -> Any:
    return SimpleNamespace(
        symbol=symbol, signal_date=day, exit_reason=reason, dividends=dividends, flags=()
    )


def test_pick_trades_one_per_group_and_filler() -> None:
    trades = [trade(f"P{i}", f"2020-01-{i + 1:02d}") for i in range(5)]
    trades += [trade("D", "2020-02-01", dividends=1.0), trade("S", "2020-03-01", "stop_loss")]
    picked = vl.pick_trades(trades, lambda t: True)
    assert [label for label, _ in picked] == [
        "ordinary exit",
        "stop or target",
        "dividend or split",
    ]
    only_plain = vl.pick_trades(trades[:5], lambda t: True)
    assert len(only_plain) == 3 and len({t.symbol for _, t in only_plain}) == 3
    with pytest.raises(vl.VerifyError):
        vl.pick_trades(trades[:2], lambda t: True)


def test_holding_period_exit_with_a_stop_touch_is_a_mismatch() -> None:
    days = ["d0", "d1", "d2", "d3", "d4"]
    bars = {d: {"open": 100.0, "high": 101.0, "low": 99.0} for d in days}
    rules = {"reason": "holding_period", "holding_days": 3, "stop_pct": 5.0, "target_pct": 10.0}
    assert vl.hand_exit(bars, days, 1, 100.0, rules, []) == ("d4", 100.0)
    bars["d3"] = {"open": 100.0, "high": 101.0, "low": 90.0}
    with pytest.raises(vl.VerifyError, match="touched a stop or target on d3"):
        vl.hand_exit(bars, days, 1, 100.0, rules, [])


def test_integration_counts_need_a_pass_and_no_skips() -> None:
    assert vl.parse_pytest_counts("x\n13 passed in 4.1s") == {"passed": 13}
    assert vl.parse_pytest_counts("1 passed, 12 skipped in 0.3s")["skipped"] == 12
    vl.check_integration_counts("13 passed in 4.1s")
    with pytest.raises(vl.VerifyError, match="skipped"):
        vl.check_integration_counts("1 passed, 12 skipped in 0.3s")
    with pytest.raises(vl.VerifyError, match="No integration test passed"):
        vl.check_integration_counts("13 skipped in 0.3s")
    with pytest.raises(vl.VerifyError, match="No integration test passed"):
        vl.check_integration_counts("")


def test_empty_fake_reference_is_unverified_not_verified() -> None:
    diffs = vl.diff_shape([], [{"a": 1}])
    assert kinds(diffs) == ["empty_reference"]
    assert not diffs[0].fatal
    keyed = vl.compare_endpoint("fundamentals_components", {}, 200, {"0": {"a": 1}})
    assert kinds(keyed) == ["empty_reference"]
    spec = vl.EndpointSpec("fundamentals_components", "/x", {})
    lines = vl.verified_lines([vl.EndpointResult(spec, 200, tuple(keyed))], vl.date(2026, 11, 2))
    assert not any("verified live on" in line for line in lines)
