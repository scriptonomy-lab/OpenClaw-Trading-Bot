"""The --data simulated|live flag (D34) and the universe fields on outputs (D33).

Every test sets EODHD_MODE through monkeypatch first, so the flag's change to
the process environment is undone afterwards. Outputs go to a temp folder.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bot import cli, commands, config, reporting, storage
from bot.universe import SAMPLE_LIMITATIONS, universe_label

STRATEGY = {
    "name": "Gap down test",
    "description_plain_english": "Buy stocks whose open is 3% or more below the previous close.",
    "version": 1,
    "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 3}],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}


@pytest.fixture
def tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Outputs under a temp project root, in live mode unless a test overrides it."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("EODHD_MODE", "live")
    monkeypatch.setenv("EODHD_UNIVERSE", "sample")
    monkeypatch.delenv("OPENCLAW_CHAT_ID", raising=False)
    return tmp_path


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    """Run the CLI and parse its one JSON object."""
    code = cli.main(list(argv))
    parsed: dict[str, Any] = json.loads(capsys.readouterr().out)
    return code, parsed


# ------------------------------------------------------------ the override


def test_override_maps_flag_values_to_modes(tmp_root: Path) -> None:
    config.apply_data_override("simulated")
    assert config.data_mode() == "fake"
    assert config.output_dir() == tmp_root / "outputs" / "simulated"
    config.apply_data_override("live")
    assert config.data_mode() == "live"
    assert config.output_dir() == tmp_root / "outputs" / "live"


def test_no_flag_leaves_the_mode_alone(tmp_root: Path) -> None:
    config.apply_data_override(None)
    assert config.data_mode() == "live"


@pytest.mark.parametrize("bad", ["fake", "LIVE", "", "simulated;id"])
def test_override_refuses_other_values(tmp_root: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        config.apply_data_override(bad)


# ---------------------------------------------------------------- the parser


@pytest.mark.parametrize("name", cli.AGENT_COMMANDS)
def test_every_agent_command_takes_the_flag(name: str) -> None:
    extra = {
        "validate": ["{}"],
        "backtest": ["strat_0000000000000000"],
        "report": ["strat_0000000000000000"],
        "approve": ["strat_0000000000000000"],
        "why": ["AAPL"],
    }.get(name, [])
    args = cli.build_parser().parse_args([name, *extra, "--data", "simulated"])
    assert args.data == "simulated"
    assert cli.build_parser().parse_args([name, *extra]).data is None


@pytest.mark.parametrize("name", ["paper-update", "daily-summary"])
def test_cron_commands_do_not_take_the_flag(name: str) -> None:
    with pytest.raises(cli.ArgumentError):
        cli.build_parser().parse_args([name, "--data", "live"])


def test_bad_flag_value_is_a_json_error(tmp_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_json(capsys, "data-status", "--data", "fake")
    assert code == 1
    assert out["ok"] is False
    assert "Bad command line" in out["errors"][0]


# ------------------------------------------------- one command, one mode


def test_simulated_flag_writes_only_to_the_simulated_folder(
    tmp_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_json(capsys, "validate", json.dumps(STRATEGY), "--data", "simulated")
    assert code == 0
    assert out["data_mode"] == "fake"
    assert out["banner"] == config.SIMULATED_LABEL
    assert (tmp_root / "outputs" / "simulated" / "bot.sqlite").exists()
    assert not (tmp_root / "outputs" / "live").exists()


def test_live_flag_overrides_a_fake_default(
    tmp_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EODHD_MODE", "fake")
    code, out = run_json(capsys, "validate", json.dumps(STRATEGY), "--data", "live")
    assert code == 0
    assert out["data_mode"] == "live"
    assert "banner" not in out
    assert (tmp_root / "outputs" / "live" / "bot.sqlite").exists()
    assert not (tmp_root / "outputs" / "simulated").exists()


def test_simulated_and_live_records_never_mix(
    tmp_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A strategy validated in simulated mode is unknown to live mode, and the reverse."""
    _, sim = run_json(capsys, "validate", json.dumps(STRATEGY), "--data", "simulated")
    sid = sim["strategy_id"]
    code, live = run_json(capsys, "report", sid, "--data", "live")
    assert code == 1
    assert "has not been validated" in live["errors"][0]
    assert live["data_mode"] == "live"


def test_simulated_approval_never_touches_the_live_record(
    tmp_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Approval state lives in the mode's own database, so live counts stay at zero."""
    config.apply_data_override("simulated")
    assert storage.db_path() == tmp_root / "outputs" / "simulated" / "bot.sqlite"
    storage.save_strategy("strat_0000000000000000", json.dumps(STRATEGY), None)
    storage.record_approval("strat_0000000000000000")
    assert storage.find_stored_approval("strat_0000000000000000") is not None
    config.apply_data_override("live")
    assert storage.db_path() == tmp_root / "outputs" / "live" / "bot.sqlite"
    assert storage.find_stored_approval("strat_0000000000000000") is None
    assert storage.count_used_approvals() == 0


def test_simulated_approval_note_only_in_fake_mode(tmp_root: Path) -> None:
    out: dict[str, Any] = {}
    commands._add_simulated_approval_note(out)
    assert out == {}
    config.apply_data_override("simulated")
    commands._add_simulated_approval_note(out)
    assert out["simulated_approval"] is True
    assert out["approval_note"] == reporting.SIMULATED_APPROVAL_NOTE


# ------------------------------------------------------- universe fields (D33)


def test_universe_fields_in_sample_and_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = reporting.universe_fields("sample")
    assert sample["universe"] == universe_label("sample")
    assert "200 of 741" in sample["universe"] and "seed 20261102" in sample["universe"]
    assert sample["limitations"] == [SAMPLE_LIMITATIONS]
    full = reporting.universe_fields("full")
    assert full["universe"] == universe_label("full")
    assert full["limitations"] == []
    assert reporting.benchmark_labels("sample")["equal_weight_sp500"] == (
        "Equal-weight, same sample"
    )
    assert "same sample" not in reporting.benchmark_labels("full")["equal_weight_sp500"]


def test_data_status_and_paper_status_carry_the_universe(
    tmp_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in ("data-status", "paper-status"):
        _, out = run_json(capsys, command, "--data", "simulated")
        assert out["universe"] == universe_label("sample")
        assert out["limitations"] == [SAMPLE_LIMITATIONS]


def test_caption_has_universe_and_too_few_flag_next_to_trades() -> None:
    window = {
        "label": "primary",
        "start": "2016-01-04",
        "end": "2023-12-29",
        "closed_trades": 12,
        "too_few_trades": True,
        "by_cost_bps": {"5": {"total_return_pct": 1.5, "max_drawdown_pct": -3.2}},
        "benchmarks": {
            "spy": {"total_return_pct": 100.0},
            "equal_weight_sp500": {"total_return_pct": 90.0},
        },
    }
    text = reporting.telegram_caption("Test", window, 2, "sample")
    assert "Closed trades 12. Too few trades to judge" in text
    assert "Equal-weight, same sample 90.0%" in text
    assert text.endswith(SAMPLE_LIMITATIONS)
    assert universe_label("sample") in text
    full = reporting.telegram_caption("Test", window, 2, "full")
    assert SAMPLE_LIMITATIONS not in full
    assert universe_label("full") in full


def test_daily_summary_text_states_the_universe() -> None:
    from datetime import date

    text = reporting.summary_text([{"lines": ["Strategy x"]}], date(2026, 11, 2))
    assert universe_label() in text.splitlines()[1]


def test_approval_message_states_the_universe() -> None:
    text = commands.approval_message("strat_0000000000000000", "Name", "Rules.", "123456")
    assert universe_label() in text
    long = commands.approval_message("strat_0000000000000000", "Name", "x" * 5000, "123456")
    assert len(long) + len(f"{config.SIMULATED_LABEL}: ") <= commands.TELEGRAM_TEXT_LIMIT
    assert long.endswith("It expires in 10 minutes.")
