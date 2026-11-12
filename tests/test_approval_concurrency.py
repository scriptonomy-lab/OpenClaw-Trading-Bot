"""Concurrent approval attempts: the lockout and the single holdout run (review M1, M3)."""

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from bot import commands, storage, telegram_send
from bot.data import holdout
from bot.strategy.schema import strategy_from_json, strategy_id, strategy_to_json

STRATEGY = {
    "name": "Gap down test",
    "description_plain_english": "Buy stocks whose open is 3% or more below the previous close.",
    "version": 1,
    "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 3}],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}
THREADS = 20


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A temporary database with one backtested strategy and a fake Telegram."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(storage, "db_path", lambda: tmp_path / "bot.sqlite")
    monkeypatch.setattr(commands, "output_dir", lambda: tmp_path)
    code_file = tmp_path / "code.txt"
    monkeypatch.setenv(commands.TEST_CODE_FILE_ENV, str(code_file))
    monkeypatch.setattr(
        telegram_send, "send_text", lambda text: telegram_send.SendResult(0, 0, "not sent", [])
    )
    strategy = strategy_from_json(json.dumps(STRATEGY))
    sid = strategy_id(strategy)
    storage.save_strategy(sid, strategy_to_json(strategy), None)
    run = {
        "window_label": "primary",
        "window_start": "2016-01-01",
        "window_end": "2023-12-31",
        "cost_bps": 5.0,
        "metrics": {},
        "chart_paths": [],
    }
    storage.replace_backtest(sid, "backtest", [run], [], None)
    commands.approve_command(sid)
    return {"sid": sid, "strategy": strategy, "code": code_file.read_text()}


def run_parallel(count: int, work: Any) -> list[str]:
    """Run ``work(i)`` in ``count`` threads that start together. Returns each outcome text."""
    barrier = threading.Barrier(count)
    outcomes: list[str] = [""] * count

    def target(index: int) -> None:
        barrier.wait()
        try:
            outcomes[index] = "OK:" + str(work(index))
        except (commands.CommandError, holdout.HoldoutAccessError) as error:
            outcomes[index] = str(error)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def test_parallel_wrong_guesses_check_at_most_three(setup: dict[str, Any]) -> None:
    guesses = [f"{i:06d}" for i in range(THREADS)]
    guesses = [g if g != setup["code"] else "999999" for g in guesses]
    outcomes = run_parallel(THREADS, lambda i: commands._check_code(setup["sid"], guesses[i]))
    checked = [o for o in outcomes if o.startswith("Wrong code")]
    assert len(checked) == 3
    assert sum("locked" in o and not o.startswith("Wrong") for o in outcomes) == THREADS - 3
    row = storage.get_approval_code(setup["sid"])
    assert row is not None
    assert row["failed_attempts"] == 3


def test_only_one_parallel_check_of_the_right_code_wins(setup: dict[str, Any]) -> None:
    outcomes = run_parallel(THREADS, lambda i: commands._check_code(setup["sid"], setup["code"]))
    assert sum(o.startswith("OK:") for o in outcomes) == 1


def test_two_correct_approvals_run_the_holdout_once(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []
    started = threading.Event()

    def slow_run(sid: str, strategy: Any, today: date) -> dict[str, Any]:
        with holdout.holdout_session(sid):
            runs.append(sid)
            started.wait(0.2)
        return {"ok": True}

    monkeypatch.setattr(commands, "_run_holdout", slow_run)
    monkeypatch.setattr(commands, "start_paper", lambda sid, today: None)
    monkeypatch.setattr(commands, "paper_today", lambda: date(2024, 2, 1))
    outcomes = run_parallel(
        4, lambda i: commands._approve_with_valid_code(setup["sid"], setup["strategy"])
    )
    assert len(runs) == 1
    assert sum(o.startswith("OK:") for o in outcomes) == 1
    assert storage.count_used_approvals() == 1


def test_claiming_the_run_twice_fails_the_second_time(setup: dict[str, Any]) -> None:
    assert storage.record_approval(setup["sid"]) is True
    assert storage.record_approval(setup["sid"]) is False
    assert storage.set_approval_used(setup["sid"]) is True
    assert storage.set_approval_used(setup["sid"]) is False


def test_recorded_but_unused_approval_can_be_resumed(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the approval row but before the holdout must not lock the strategy out."""
    storage.record_approval(setup["sid"])
    monkeypatch.setattr(commands, "paper_today", lambda: date(2024, 2, 1))
    monkeypatch.setattr(commands, "start_paper", lambda sid, today: None)

    def run(sid: str, strategy: Any, today: date) -> dict[str, Any]:
        with holdout.holdout_session(sid):
            pass
        return {"ok": True}

    monkeypatch.setattr(commands, "_run_holdout", run)
    commands._check_code(setup["sid"], setup["code"])
    out = commands._approve_with_valid_code(setup["sid"], setup["strategy"])
    assert out["paper_trading_started"] is True
    with pytest.raises(commands.CommandError, match="already approved"):
        commands.approve_command(setup["sid"])


def test_holdout_failure_message_is_fixed_and_logged(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(sid: str, strategy: Any, today: date) -> dict[str, Any]:
        raise RuntimeError("secret detail 12345")

    monkeypatch.setattr(commands, "_run_holdout", boom)
    monkeypatch.setattr(commands, "paper_today", lambda: date(2024, 2, 1))
    with pytest.raises(commands.CommandError) as caught:
        commands._approve_with_valid_code(setup["sid"], setup["strategy"])
    assert "secret detail" not in str(caught.value)
    assert "RuntimeError" not in str(caught.value)
    log = (tmp_path / commands.HOLDOUT_LOG).read_text()
    assert "secret detail 12345" in log
