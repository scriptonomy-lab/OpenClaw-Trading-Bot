"""CLI commands in fake mode with a temporary output folder, database and cache (review Phase 5)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from bot import cli, commands, storage, telegram_send
from bot.backtest import charts
from bot.backtest.engine import PortfolioState, state_to_json
from bot.data import eodhd_client as client
from bot.strategy.schema import strategy_from_json, strategy_id, strategy_to_json
from bot.universe import SAMPLE_LIMITATIONS, universe_label

STRATEGY = {
    "name": "Gap down test",
    "description_plain_english": "Buy stocks whose open is 3% or more below the previous close.",
    "version": 1,
    "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 3}],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}
TODAY = "2024-02-29"


@pytest.fixture(scope="module")
def fake_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One fake-data cache for the module, so the slow first load happens once."""
    return tmp_path_factory.mktemp("fake_cache")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_cache: Path) -> dict[str, Any]:
    """Fake mode, with the database, output folder and fake cache in a temp folder."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setenv("OPENCLAW_BOT_TODAY", TODAY)
    monkeypatch.setattr(storage, "db_path", lambda: tmp_path / "bot.sqlite")
    monkeypatch.setattr(commands, "output_dir", lambda: tmp_path)
    monkeypatch.setattr(charts, "output_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", fake_cache)
    monkeypatch.delenv("OPENCLAW_CHAT_ID", raising=False)
    code_file = tmp_path / "code.txt"
    monkeypatch.setenv(commands.TEST_CODE_FILE_ENV, str(code_file))
    sent: list[str] = []

    def fake_text(text: str) -> telegram_send.SendResult:
        sent.append(text)
        return telegram_send.SendResult(0, 0, "not sent", [])

    monkeypatch.setattr(telegram_send, "send_text", fake_text)
    monkeypatch.setattr(
        telegram_send, "send_photo", lambda path, caption: telegram_send.SendResult(0, 0, "n", [])
    )
    return {"tmp": tmp_path, "code_file": code_file, "sent": sent}


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """Run the CLI and return the exit code and stdout."""
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    """Run a JSON command. Stdout must be exactly one JSON object."""
    _, out = run(capsys, *argv)
    parsed: dict[str, Any] = json.loads(out)
    return parsed


def validated_id(capsys: pytest.CaptureFixture[str]) -> str:
    """Validate the test strategy through the CLI and return its id."""
    out = run_json(capsys, "validate", json.dumps(STRATEGY))
    assert out["ok"] is True
    return str(out["strategy_id"])


# --------------------------------------------------------------- error mapping


def test_bad_arguments_become_a_json_error(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "backtest")
    assert code == 1
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "Bad command line" in parsed["errors"][0]


def test_unknown_strategy_id_is_a_plain_error(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "backtest", "strat_" + "0" * 16)
    assert code == 1
    assert "has not been validated" in json.loads(out)["errors"][0]
    code, out = run(capsys, "report", "not-an-id")
    assert code == 1
    assert "not a strategy id" in json.loads(out)["errors"][0]


def test_holdout_guard_and_other_errors_are_mapped(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.data.holdout import HoldoutAccessError

    def guard() -> dict[str, Any]:
        raise HoldoutAccessError("sealed")

    monkeypatch.setattr(commands, "data_status_command", guard)
    code, out = run(capsys, "data-status")
    assert code == 1
    assert json.loads(out)["errors"] == ["Holdout guard: sealed"]

    def broken() -> dict[str, Any]:
        raise ValueError("oops")

    monkeypatch.setattr(commands, "data_status_command", broken)
    code, out = run(capsys, "data-status")
    assert code == 1
    assert "ValueError" in json.loads(out)["errors"][0]


def test_invalid_strategy_is_reported_not_raised(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "validate", json.dumps({"name": "x"}))
    assert code == 1
    assert json.loads(out)["ok"] is False


# ------------------------------------------------------------ simple commands


def test_data_status_has_holdout_count(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = run_json(capsys, "data-status")
    assert out["ok"] is True
    assert out["data_mode"] == "fake"
    assert out["holdout_runs_so_far"] == 0
    assert "sealed" in out["holdout"]


def test_paper_status_and_summary_before_any_update(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = run_json(capsys, "paper-status")
    assert out["paper_trading_started"] is False
    code, text = run(capsys, "daily-summary")
    assert code == 0
    assert text.strip() == "NO_REPLY"


def test_why_with_nothing_stored_is_an_error(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "why", "AAPL")
    assert code == 1
    assert "No stored entry" in json.loads(out)["errors"][0]
    code, out = run(capsys, "why", "aapl")
    assert "does not look like a ticker" in json.loads(out)["errors"][0]
    code, out = run(capsys, "why", "AAPL", "--date", "yesterday")
    assert "must look like" in json.loads(out)["errors"][0]


# --------------------------------------------------------- backtest and report


def test_backtest_then_report_and_why(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid = validated_id(capsys)
    out = run_json(capsys, "backtest", sid)
    assert out["ok"] is True, out.get("errors")
    assert out["windows"][0]["label"] == "primary"
    assert out["data_quality"] is not None
    # The fake world runs on the full (SIM) universe (conftest), so no limitations note.
    assert out["universe_mode"] == "full"
    assert out["universe"] == universe_label("full")
    assert out["limitations"] == []
    assert out["benchmark_labels"]["equal_weight_sp500"] == "Equal-weight, full universe"
    saved = run_json(capsys, "report", sid)
    assert saved["universe"] == out["universe"]
    assert saved["limitations"] == out["limitations"]
    assert saved["windows"] == out["windows"]
    assert saved["holdout_runs_so_far"] == 0
    assert "holdout" not in saved
    trades = storage.list_trades(sid, "backtest")
    if trades:
        ticker = str(trades[0]["symbol"]).removesuffix(".US")
        why = run_json(capsys, "why", ticker)
        assert why["ok"] is True
        assert why["strategy_id"] == sid
        assert why["source"] == "backtest"


def test_report_in_sample_mode_quotes_the_sample_and_limitations(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored run keeps the universe it used; sample runs carry the limitations note."""
    sid = validated_id(capsys)
    run = {
        "window_label": "primary",
        "window_start": "2016-01-04",
        "window_end": "2023-12-29",
        "cost_bps": 5.0,
        "metrics": {
            "total_return_pct": 1.0, "cagr_pct": 0.1, "max_drawdown_pct": -2.0, "sharpe": 0.3,
            "win_rate_pct": 50.0, "average_trade_return_pct": 0.1, "closed_trades": 12,
            "open_at_end": 0, "data_ended_trades": 0, "too_few_trades": True,
            "versions_tried": 1, "universe_mode": "sample",
            "benchmarks": {"spy": {}, "equal_weight_sp500": {}},
        },
        "chart_paths": [],
    }  # fmt: skip
    storage.replace_backtest(sid, "backtest", [run], [], None)
    out = run_json(capsys, "report", sid)
    assert out["universe"] == universe_label("sample")
    assert out["limitations"] == [SAMPLE_LIMITATIONS]
    window = out["windows"][0]
    assert window["too_few_trades"] is True and "Too few trades" in window["note"]
    assert window["by_cost_bps"]["5"]["too_few_trades"] is True


def test_why_never_offers_holdout_trades(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid = validated_id(capsys)
    row = {
        "symbol": "AAPL.US",
        "signal_date": "2024-01-10",
        "entry_date": "2024-01-11",
        "exit_date": "2024-01-15",
        "exit_reason": "holding_period",
        "signal_values": {"gap_pct": -4.0},
        "window_label": "holdout",
    }
    storage.replace_backtest(sid, "holdout", [], [row], None)
    code, out = run(capsys, "why", "AAPL")
    assert code == 1
    assert "No stored entry" in json.loads(out)["errors"][0]


# ------------------------------------------------------- real approval, end to end


def test_full_approval_path_runs_holdout_once_and_starts_paper(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid = validated_id(capsys)
    run_json(capsys, "backtest", sid)
    _, asked = run(capsys, "approve", sid)
    code = Path(world["code_file"]).read_text()
    assert code not in asked
    assert json.loads(asked)["status"] == "code_sent"
    assert world["sent"][0].startswith(sid)

    code_exit, approved_text = run(capsys, "approve", sid, "--code", code)
    assert code_exit == 0
    assert code not in approved_text
    approved = json.loads(approved_text)
    assert approved["paper_trading_started"] is True
    assert approved["paper_start_day"] == TODAY
    assert approved["holdout"]["holdout_runs_so_far"] == 1
    assert approved["holdout"]["universe"] == universe_label("full")
    assert approved["holdout"]["limitations"] == []
    assert approved["simulated_approval"] is True
    assert "live holdout record is untouched" in approved["approval_note"]
    assert json.loads(asked)["simulated_approval"] is True
    assert universe_label("full") in world["sent"][0]
    assert storage.get_paper_run(sid) is not None
    assert storage.count_used_approvals() == 1

    code_exit, again = run(capsys, "approve", sid, "--code", code)
    assert code_exit == 1
    assert "already approved" in json.loads(again)["errors"][0]

    report = run_json(capsys, "report", sid)
    assert report["holdout"]["holdout_runs_so_far"] == 1
    status = run_json(capsys, "paper-status")
    assert status["paper_trading_started"] is True
    assert status["portfolios"][0]["strategy_id"] == sid
    assert run_json(capsys, "data-status")["holdout_runs_so_far"] == 1
    assert not (world["tmp"] / commands.HOLDOUT_LOG).exists()


def test_outputs_never_hold_the_code_or_news_text(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid = validated_id(capsys)
    texts = [run(capsys, "backtest", sid)[1], run(capsys, "approve", sid)[1]]
    code = Path(world["code_file"]).read_text()
    texts.append(run(capsys, "approve", sid, "--code", code)[1])
    texts += [run(capsys, name)[1] for name in ("paper-status", "data-status", "daily-summary")]
    for text in texts:
        assert code not in text
        assert "headline" not in text.lower()
        assert "content" not in json.loads(text) if text.startswith("{") else True


# ------------------------------------------------------------- daily summary


def seed_paper_day(sid: str, day: str, equity: float, trades: list[dict[str, Any]]) -> None:
    """Save one paper day directly, as paper_update would."""
    state = PortfolioState(cash=equity, positions={}, pending=(), last_date=date.fromisoformat(day))
    ledger = {"cash": equity, "holdings_value": 0.0, "equity": equity, "open_positions": 0}
    storage.save_paper_day(sid, day, state_to_json(state), ledger, trades)


def sold(symbol: str, exit_day: str) -> dict[str, Any]:
    """A closed paper trade row with the fields the summary reads."""
    return {
        "symbol": symbol,
        "signal_date": "2024-02-01",
        "entry_date": "2024-02-02",
        "exit_date": exit_day,
        "exit_reason": "holding_period",
        "signal_values": {},
        "return_pct": 1.5,
    }


def test_daily_summary_catches_up_missed_days(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy = strategy_from_json(json.dumps(STRATEGY))
    sid = strategy_id(strategy)
    storage.save_strategy(sid, strategy_to_json(strategy), None)
    storage.start_paper_run(sid, "2024-02-26")

    def summary_on(day: str) -> str:
        monkeypatch.setattr(commands, "paper_today", lambda: date.fromisoformat(day))
        return commands.daily_summary_text()

    seed_paper_day(sid, "2024-02-26", 100_000.0, [])
    first = summary_on("2024-02-26")
    assert "since start" in first
    # The cron job missed two days, then ran once and stepped both.
    seed_paper_day(sid, "2024-02-27", 100_500.0, [sold("AAA.US", "2024-02-27")])
    seed_paper_day(sid, "2024-02-28", 101_000.0, [sold("BBB.US", "2024-02-28")])
    seed_paper_day(sid, "2024-02-29", 102_000.0, [])
    catch_up = summary_on("2024-02-29")
    assert "Sold AAA on 2024-02-27" in catch_up
    assert "Sold BBB on 2024-02-28" in catch_up
    assert "since 2024-02-26 2.0%" in catch_up
    # Running it again on the same day gives the same summary, not an empty one.
    assert summary_on("2024-02-29") == catch_up
    # The next day only lists new trades.
    seed_paper_day(sid, "2024-03-01", 102_000.0, [sold("CCC.US", "2024-03-01")])
    next_day = summary_on("2024-03-01")
    assert "Sold CCC" in next_day
    assert "AAA" not in next_day and "BBB" not in next_day


def test_paper_status_lists_every_portfolio(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    ids = []
    for gap in (3, 4):
        raw = {**STRATEGY, "entry": [{"block": "gap", "direction": "down", "min_gap_pct": gap}]}
        strategy = strategy_from_json(json.dumps(raw))
        sid = strategy_id(strategy)
        storage.save_strategy(sid, strategy_to_json(strategy), None)
        storage.start_paper_run(sid, "2024-02-26")
        seed_paper_day(sid, "2024-02-26", 100_000.0, [])
        ids.append(sid)
    out = run_json(capsys, "paper-status")
    assert sorted(p["strategy_id"] for p in out["portfolios"]) == sorted(ids)
