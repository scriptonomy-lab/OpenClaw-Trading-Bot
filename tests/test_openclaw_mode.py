"""Tests for the OpenClaw eval mode. No network and no OpenClaw install needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import openclaw_mode as om
from evals import scoring

FIXTURE = Path(__file__).parent / "fixtures" / "openclaw_session_sample.json"
REPO = Path(__file__).parent.parent


def sample() -> dict:
    return json.loads(FIXTURE.read_text())


# ------------------------------------------------------------ transcript parsing


def test_parse_recorded_session() -> None:
    data = sample()
    parsed = om.parse_session(data["messages"], data["user_turns"])
    kinds = [e["type"] for e in parsed.events]
    assert kinds[0] == "user"
    calls = [e for e in parsed.events if e["type"] == "tool_call"]
    assert [c["command"] for c in calls] == data["expected_commands"]
    assert [c["allowed"] for c in calls] == data["expected_allowed"]
    blocked = [c for c in calls if not c["allowed"]]
    assert blocked, "the sample must hold a blocked exec attempt"
    final = scoring.final_reply({"events": parsed.events}, 0)
    assert final and "toolCall" not in final
    assert parsed.usage["total_tokens"] > 0


def test_blocked_attempt_is_seen_by_scorer() -> None:
    data = sample()
    parsed = om.parse_session(data["messages"], data["user_turns"])
    check = scoring.check_no_blocked_attempts({"events": parsed.events})
    assert not check.passed


def test_split_command() -> None:
    assert om.split_command("openclaw-bot blocks") == ("blocks", "blocks")
    assert om.split_command("/x/.venv/bin/openclaw-bot validate '{}'") == (
        "validate",
        "validate '{}'",
    )
    assert om.split_command("ls -la") == ("ls", "ls -la")
    assert om.split_command("openclaw-bot blocks && ls") == ("blocks", "blocks && ls")


def test_real_code_is_mapped_back_to_placeholder() -> None:
    nested = {
        "role": "custom",
        "customType": "openclaw.nested-tool.v1",
        "details": {
            "toolName": "exec",
            "input": {"command": "openclaw-bot approve strat_0123456789abcdef --code 111222"},
            "result": {"content": [{"type": "text", "text": '{"ok": true}'}], "details": {}},
        },
    }
    messages = [{"role": "user", "content": "x"}, nested]
    parsed = om.parse_session(messages, ["The code is 482913."], {0: "111222"})
    call = next(e for e in parsed.events if e["type"] == "tool_call")
    assert "--code 482913" in call["args"] and "111222" not in call["args"]
    assert any(e["type"] == "note" and "substituted" in e["text"] for e in parsed.events)


def test_approval_code_substitution(tmp_path: Path) -> None:
    code_file = tmp_path / "code.txt"
    assert om.approval_code_for_turn("The code is 482913.", code_file) == (
        "The code is 482913.",
        None,
    )
    code_file.write_text("135790\n")
    assert om.approval_code_for_turn("The code is 482913.", code_file) == (
        "The code is 135790.",
        "135790",
    )
    assert om.approval_code_for_turn("Yes, run it.", code_file) == ("Yes, run it.", None)


# --------------------------------------------------------------- hash verification


def test_heldout_hash_mismatch_refuses(tmp_path: Path) -> None:
    held = tmp_path / om.HELDOUT_NAME
    held.write_text('{"cases": []}')
    hash_file = tmp_path / "recorded.sha256"
    hash_file.write_text(om.sha256_file(held) + "  cases_heldout.json\n")
    assert om.verify_cases_file(held, hash_file) == om.sha256_file(held)
    held.write_text('{"cases": [{"id": "changed"}]}')
    with pytest.raises(om.HeldoutHashError):
        om.verify_cases_file(held, hash_file)


def test_heldout_without_recorded_hash_refuses(tmp_path: Path) -> None:
    held = tmp_path / om.HELDOUT_NAME
    held.write_text('{"cases": []}')
    with pytest.raises(om.HeldoutHashError):
        om.verify_cases_file(held, tmp_path / "missing.sha256")


def test_other_case_files_are_not_hash_checked(tmp_path: Path) -> None:
    other = tmp_path / "cases.json"
    other.write_text('{"cases": []}')
    assert om.verify_cases_file(other, tmp_path / "missing") == om.sha256_file(other)


def test_repo_heldout_file_matches_its_record() -> None:
    held = REPO / "evals" / om.HELDOUT_NAME
    if held.exists():
        assert om.verify_cases_file(held)


# ------------------------------------------------------------- gateway environment


def test_gateway_env_is_isolated_and_fake(tmp_path: Path) -> None:
    env = om.build_gateway_env(
        state_dir=tmp_path / "state",
        config_path=tmp_path / "state" / "openclaw.json",
        home_dir=tmp_path / "home",
        port=om.EVAL_PORT,
        code_file=tmp_path / "code.txt",
        openai_key="sk-test-not-real",
        token="t",
        model="openai/gpt-6.1-sol",
        node_bin_dir=tmp_path / "node",
    )
    assert env["EODHD_MODE"] == "fake"
    assert env["OPENCLAW_BOT_TEST_APPROVAL_CODE_FILE"] == str(tmp_path / "code.txt")
    assert env["OPENCLAW_STATE_DIR"] == str(tmp_path / "state")
    assert env["OPENCLAW_GATEWAY_PORT"] == str(om.EVAL_PORT) != "18789"
    assert not [k for k in env if "TELEGRAM" in k or "EODHD_API" in k]


def test_gateway_config_policy_copied_telegram_inert(tmp_path: Path) -> None:
    rendered = om.json5_loads((REPO / "openclaw" / "openclaw.json5").read_text())
    config = om.build_gateway_config(rendered, om.EVAL_PORT, "tok", tmp_path / "bin", "openai/m")
    assert config["tools"]["deny"] == rendered["tools"]["deny"]
    assert config["tools"]["exec"]["mode"] == "allowlist"
    assert config["tools"]["exec"]["pathPrepend"] == [str(tmp_path / "bin")]
    assert config["gateway"]["port"] == om.EVAL_PORT != 18789
    assert config["plugins"]["entries"]["telegram"] == {"enabled": False}
    assert "botToken" not in config["channels"]["telegram"]
    assert config["agents"]["defaults"]["model"] == {"primary": "openai/m"}


def test_approvals_pin_sandbox_binary_and_keep_regex(tmp_path: Path) -> None:
    rendered = json.loads((REPO / "openclaw" / "exec-approvals.json").read_text())
    binary = tmp_path / ".venv" / "bin" / "openclaw-bot"
    out = om.build_approvals(rendered, binary)
    old = rendered["agents"]["main"]["allowlist"][0]
    new = out["agents"]["main"]["allowlist"][0]
    assert new["pattern"] == str(binary)
    assert new["argPattern"] == old["argPattern"]
    assert out["defaults"] == rendered["defaults"]


def test_json5_loads_rendered_template() -> None:
    config = om.json5_loads((REPO / "openclaw" / "openclaw.json5").read_text())
    assert config["tools"]["profile"] == "minimal"
    assert config["tools"]["exec"]["mode"] == "allowlist"


def test_sandbox_has_no_env_file(tmp_path: Path) -> None:
    binary = om.make_sandbox(tmp_path / "sb")
    assert binary.exists()
    assert not (tmp_path / "sb" / ".env").exists()
    case = {"tool_overrides": {"backtest": "backtest_injected.json", "why": "other.json"}}
    om.reset_case_state(tmp_path / "sb", case)
    saved = json.loads((tmp_path / "sb" / "overrides.json").read_text())
    assert saved == {"backtest": {"file": "backtest_injected.json", "mode": "merge"}}


def test_override_modes() -> None:
    assert om.override_mode("why_injected.json") == "merge"
    assert om.override_mode("backtest_plain_few.json") == "replace"
    assert om.override_mode("backtest_news.json") is None


def test_sandbox_binary_runs_and_applies_overrides(tmp_path: Path) -> None:
    import subprocess

    root = tmp_path / "sb"
    binary = om.make_sandbox(root)
    env = {"PATH": "/usr/bin:/bin"}

    def run(*args: str) -> dict:
        done = subprocess.run([str(binary), *args], env=env, capture_output=True, text=True)
        return json.loads(done.stdout)

    plain = run("blocks")
    assert plain["ok"] and plain["data_mode"] == "fake" and "latest_headline_label" not in plain
    # merge mode keeps the real fields and adds the injected one
    (root / "overrides.json").write_text(
        json.dumps({"blocks": {"file": "backtest_injected.json", "mode": "merge"}})
    )
    merged = run("blocks")
    assert merged["schema"] == plain["schema"] and "latest_headline_label" in merged
    # replace mode swaps the output
    (root / "overrides.json").write_text(
        json.dumps({"blocks": {"file": "backtest_plain_few.json", "mode": "replace"}})
    )
    assert "schema" not in run("blocks")
