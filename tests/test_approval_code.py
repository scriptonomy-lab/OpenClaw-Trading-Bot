"""The one-time approval code flow (D20). No network, no real holdout run."""

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from bot import cli, commands, storage, telegram_send
from bot.strategy.schema import (
    describe_rules,
    strategy_from_json,
    strategy_id,
    strategy_to_json,
)

STRATEGY = {
    "name": "Gap down test",
    "description_plain_english": "Buy stocks whose open is 3% or more below the previous close.",
    "version": 1,
    "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 3}],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A temporary database with one backtested strategy and a fake Telegram."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(storage, "db_path", lambda: tmp_path / "bot.sqlite")
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
    monkeypatch.setattr(commands, "_approve_with_valid_code", lambda s, st: {"ok": True})
    return {"sid": sid, "code_file": code_file}


def ask_for_code(setup: dict[str, Any]) -> str:
    """Request a code and read it from the test file."""
    out = commands.approve_command(setup["sid"])
    assert out["status"] == "code_sent"
    return Path(setup["code_file"]).read_text()


def test_code_never_in_stdout(setup: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["approve", setup["sid"]]) == 0
    printed = capsys.readouterr().out
    code = Path(setup["code_file"]).read_text()
    assert len(code) == 6
    assert code not in printed
    assert json.loads(printed)["status"] == "code_sent"


def test_right_code_works_once_only(setup: dict[str, Any]) -> None:
    code = ask_for_code(setup)
    assert commands.approve_command(setup["sid"], code) == {"ok": True}
    with pytest.raises(commands.CommandError, match="No unused"):
        commands.approve_command(setup["sid"], code)


def test_wrong_code_refused(setup: dict[str, Any]) -> None:
    code = ask_for_code(setup)
    wrong = "000000" if code != "000000" else "111111"
    with pytest.raises(commands.CommandError, match="Wrong code"):
        commands.approve_command(setup["sid"], wrong)


def test_expired_code_refused(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    code = ask_for_code(setup)
    later = commands.now() + timedelta(minutes=11)
    monkeypatch.setattr(commands, "now", lambda: later)
    with pytest.raises(commands.CommandError, match="expired"):
        commands.approve_command(setup["sid"], code)


def test_lockout_after_three_wrong_codes(setup: dict[str, Any]) -> None:
    code = ask_for_code(setup)
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(3):
        with pytest.raises(commands.CommandError, match="Wrong code"):
            commands.approve_command(setup["sid"], wrong)
    with pytest.raises(commands.CommandError, match="locked"):
        commands.approve_command(setup["sid"], code)
    with pytest.raises(commands.CommandError, match="locked"):
        commands.approve_command(setup["sid"])


def test_live_mode_refuses_test_code_file(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EODHD_MODE", "live")
    with pytest.raises(commands.CommandError, match="live mode"):
        commands.approve_command(setup["sid"])
    assert not Path(setup["code_file"]).exists()


def test_no_telegram_and_no_test_file_fails(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(commands.TEST_CODE_FILE_ENV)
    with pytest.raises(commands.CommandError, match="needs Telegram"):
        commands.approve_command(setup["sid"])
    row = storage.get_approval_code(setup["sid"])
    assert row is not None
    assert row["code_hash"] == ""
    assert row["used"] == 1


def test_failed_send_does_not_reset_lockout(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    code = ask_for_code(setup)
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(2):
        with pytest.raises(commands.CommandError, match="Wrong code"):
            commands.approve_command(setup["sid"], wrong)
    monkeypatch.delenv(commands.TEST_CODE_FILE_ENV)
    later = commands.now() + timedelta(minutes=11)
    monkeypatch.setattr(commands, "now", lambda: later)
    with pytest.raises(commands.CommandError, match="needs Telegram"):
        commands.approve_command(setup["sid"])
    assert storage.get_approval_code(setup["sid"])["failed_attempts"] == 2  # type: ignore[index]
    monkeypatch.setenv(commands.TEST_CODE_FILE_ENV, str(setup["code_file"]))
    new_code = ask_for_code(setup)
    wrong = "000000" if new_code != "000000" else "111111"
    with pytest.raises(commands.CommandError, match="Wrong code"):
        commands.approve_command(setup["sid"], wrong)
    with pytest.raises(commands.CommandError, match="locked"):
        commands.approve_command(setup["sid"], new_code)


def test_hash_is_keyed_not_plain_sha256(setup: dict[str, Any]) -> None:
    code = ask_for_code(setup)
    row = storage.get_approval_code(setup["sid"])
    assert row is not None
    plain = hashlib.sha256(f"{setup['sid']}|{code}".encode()).hexdigest()
    assert row["code_hash"] != plain
    assert row["code_hash"] == commands.hash_code(setup["sid"], code)


def test_hash_uses_the_env_secret(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A project .env value wins over the environment (bot.config.secret), so
    # point the config at an empty .env to test the environment value alone.
    from bot import config

    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("APPROVAL_CODE_SECRET", "one")
    first = commands.hash_code(setup["sid"], "123456")
    monkeypatch.setenv("APPROVAL_CODE_SECRET", "two")
    assert commands.hash_code(setup["sid"], "123456") != first


def test_live_mode_refuses_a_code_without_the_secret(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot import config

    monkeypatch.setattr(config, "ENV_FILE", Path("/nonexistent/.env"))
    monkeypatch.setenv("EODHD_MODE", "live")
    monkeypatch.delenv(commands.TEST_CODE_FILE_ENV)
    monkeypatch.delenv("APPROVAL_CODE_SECRET", raising=False)
    monkeypatch.setattr(commands, "secret", lambda name: "")
    with pytest.raises(commands.CommandError, match="APPROVAL_CODE_SECRET"):
        commands.approve_command(setup["sid"])
    assert storage.get_approval_code(setup["sid"]) is None


def test_one_code_per_ten_minutes(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ask_for_code(setup)
    with pytest.raises(commands.CommandError, match="already sent"):
        commands.approve_command(setup["sid"])
    later = commands.now() + timedelta(minutes=11)
    monkeypatch.setattr(commands, "now", lambda: later)
    assert commands.approve_command(setup["sid"])["status"] == "code_sent"


def test_message_starts_with_id_and_has_the_full_rules() -> None:
    rules = "Rule line. " * 100
    text = commands.approval_message("strat_0123456789abcdef", "Name", rules, "654321")
    assert text.startswith("strat_0123456789abcdef")
    assert rules in text
    assert text.rstrip().endswith("expires in 10 minutes.")
    assert "654321" in text


def test_long_rules_are_cut_but_the_code_stays() -> None:
    text = commands.approval_message("strat_0123456789abcdef", "Name", "x" * 9000, "654321")
    assert len(text) + len("SIMULATED DATA: ") <= 4096
    assert "654321" in text


def test_sent_text_has_the_full_rules(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []

    def fake_send(text: str) -> telegram_send.SendResult:
        sent.append(text)
        return telegram_send.SendResult(1, 1, "sent", [])

    monkeypatch.setattr(telegram_send, "send_text", fake_send)
    monkeypatch.delenv(commands.TEST_CODE_FILE_ENV)
    commands.approve_command(setup["sid"])
    rules = describe_rules(strategy_from_json(json.dumps(STRATEGY)))
    assert sent[0].startswith(setup["sid"])
    assert rules in sent[0]
