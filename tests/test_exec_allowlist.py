"""Checks on the OpenClaw exec allowlist (D5, D9).

This is the first line of defence for D9: the argPattern in
`openclaw/exec-approvals.json` must accept every command the agent needs and
reject shell chains, newlines, other subcommands, path traversal and flags.
The live evals against a running OpenClaw come later.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import shutil
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APPROVALS_FILE = REPO_ROOT / "openclaw" / "exec-approvals.json"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_openclaw_config.py"


def load_render_module() -> ModuleType:
    """Import the render script by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("render_openclaw_config", RENDER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = load_render_module()
APPROVALS_TEXT = APPROVALS_FILE.read_text(encoding="utf-8")
PATTERN: str = render.load_arg_pattern(APPROVALS_TEXT)


@pytest.mark.parametrize("args", render.VALID_ARGS)
def test_valid_commands_are_allowed(args: str) -> None:
    assert render.matches_like_openclaw(PATTERN, args)


@pytest.mark.parametrize("args", render.ESCAPE_ATTEMPTS)
def test_escape_attempts_are_rejected(args: str) -> None:
    assert not render.matches_like_openclaw(PATTERN, args)


def test_pattern_is_anchored_at_both_ends() -> None:
    assert PATTERN.startswith("^(?:")
    assert PATTERN.endswith(")$")


def test_pattern_uses_ascii_classes_only() -> None:
    # \d, \w and \s mean different things in Python and JavaScript, and \s
    # would let tabs and newlines through. "." would match almost anything.
    for shorthand in (r"\d", r"\w", r"\s", r"\S", r"\D", r"\W"):
        assert shorthand not in PATTERN
    without_escapes = re.sub(r"\\.", "", PATTERN)
    without_classes = re.sub(r"\[[^\]]*\]", "", without_escapes)
    assert "." not in without_classes


def test_agent_cannot_run_cron_only_commands() -> None:
    for subcommand in ("paper-update", "daily-summary"):
        assert subcommand not in PATTERN


def test_allowlist_has_one_literal_absolute_entry() -> None:
    document = json.loads(APPROVALS_TEXT)
    entries = document["agents"]["main"]["allowlist"]
    assert len(entries) == 1
    rendered = APPROVALS_TEXT.replace("<ABSOLUTE_PATH_TO_REPO>", "/home/user/repo")
    assert render.binary_path_problems(rendered) == []


def test_glob_or_relative_binary_paths_are_flagged() -> None:
    for bad_root in ("relative/repo", "/home/*/repo", "/home/user/re?o"):
        rendered = APPROVALS_TEXT.replace("<ABSOLUTE_PATH_TO_REPO>", bad_root)
        assert render.binary_path_problems(rendered)


def test_policy_denies_by_default_and_never_asks() -> None:
    document = json.loads(APPROVALS_TEXT)
    defaults = document["defaults"]
    agent = document["agents"]["main"]
    assert document["version"] == 1
    assert defaults["security"] == "deny"
    assert agent["security"] == "allowlist"
    for scope in (defaults, agent):
        assert scope["ask"] == "off"
        assert scope["askFallback"] == "deny"
        # Skill auto-allow would trust the binary with any arguments.
        assert scope["autoAllowSkills"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_regex_agrees_with_expected_results() -> None:
    # OpenClaw evaluates argPattern with ECMAScript semantics, so check the
    # real engine too, with and without the unicode flag.
    assert render.check_with_node(PATTERN) == []


def test_skill_example_commands_pass_the_allowlist() -> None:
    # The skill shows the agent a full validate command. If the skill and the
    # regex drift apart, the agent would be taught a command that is denied.
    skill = (REPO_ROOT / "openclaw" / "skills" / "strategy-lab" / "SKILL.md").read_text()
    commands = [line for line in skill.splitlines() if line.startswith("openclaw-bot validate '")]
    assert commands, "SKILL.md should show at least one validate example"
    for command in commands:
        argv = shlex.split(command)
        assert argv[0] == "openclaw-bot"
        assert render.matches_like_openclaw(PATTERN, " ".join(argv[1:]))


def test_every_skill_command_line_passes_the_allowlist() -> None:
    # Code-block lines in the skill, including the --data example, are what the
    # agent copies. Each must be allowed as written.
    skill = (REPO_ROOT / "openclaw" / "skills" / "strategy-lab" / "SKILL.md").read_text()
    commands = [line for line in skill.splitlines() if line.startswith("openclaw-bot ")]
    assert any(line.endswith(" --data simulated") for line in commands)
    for command in commands:
        argv = shlex.split(command)
        assert render.matches_like_openclaw(PATTERN, " ".join(argv[1:])), command


def test_data_flag_is_optional_once_and_last() -> None:
    sid = render.SAMPLE_ID
    for base in ("blocks", "data-status", f"backtest {sid}", f"approve {sid} --code 482913"):
        assert render.matches_like_openclaw(PATTERN, base)
        for value in ("simulated", "live"):
            assert render.matches_like_openclaw(PATTERN, f"{base} --data {value}")
            assert not render.matches_like_openclaw(
                PATTERN, f"{base} --data {value} --data {value}"
            )
        for value in ("fake", "test", "simulated live", "live\nid", "$(id)", "live;id"):
            assert not render.matches_like_openclaw(PATTERN, f"{base} --data {value}")
    assert PATTERN.count("--data") == 1


def test_fill_rejects_unfilled_placeholders() -> None:
    with pytest.raises(SystemExit):
        render.fill("model: <OPENCLAW_MODEL>", {})


def test_user_ids_must_be_numeric_and_present() -> None:
    assert render.parse_user_ids(" 123, 456 ") == ["123", "456"]
    with pytest.raises(SystemExit):
        render.parse_user_ids("")
    with pytest.raises(SystemExit):
        render.parse_user_ids("@someone")


def test_model_must_be_an_openai_id() -> None:
    assert render.check_model("openai/gpt-5.4-mini") == "openai/gpt-5.4-mini"
    for bad in ("", "gpt-5.4", "anthropic/claude", 'openai/x", evil: "1'):
        with pytest.raises(SystemExit):
            render.check_model(bad)
