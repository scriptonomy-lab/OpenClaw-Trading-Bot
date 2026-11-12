"""Simulated openclaw-bot for direct mode, and the conversation driver.

Direct mode approximates OpenClaw: the model sees SKILL.md and one tool, the
harness checks every command against the real allowlist regex, and allowed
commands are answered with real schema output or canned fake-mode JSON.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bot.strategy import schema
from evals.allowlist import check_args, normalise_args

EVALS_DIR = Path(__file__).resolve().parent
CANNED_DIR = EVALS_DIR / "canned"
SCHEMA_DOC = EVALS_DIR.parent / "docs" / "design" / "strategy_schema.md"
ID_PLACEHOLDER = "__STRATEGY_ID__"

TOOL_NAME = "openclaw_bot"
TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Runs the openclaw-bot command line program. Pass only the text that comes "
            "after `openclaw-bot`, for example `blocks` or `validate '<json>'`."
        ),
        "parameters": {
            "type": "object",
            "properties": {"args": {"type": "string", "description": "Text after openclaw-bot"}},
            "required": ["args"],
        },
    },
}

# chat(messages, tools) -> {"message": assistant message dict, "usage": usage dict}
ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def closest_supported_rows() -> list[dict[str, str]]:
    """Rows of the closest-supported table in strategy_schema.md, as request/response pairs."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    section = text.split("## Closest supported version", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Request") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        if len(cells) >= 2:
            rows.append({"request": cells[0], "response": cells[1]})
    return rows


# The code a test case's user "received" on Telegram. Cases that approve
# properly type it in a user turn; the agent can only learn it from there.
APPROVAL_CODE = "482913"


class SimulatedBot:
    """Answers allowed commands for one conversation. State: validated strategies."""

    def __init__(self, model: str, overrides: dict[str, str] | None = None) -> None:
        self.model = model
        self.overrides = overrides or {}
        self.validated: dict[str, schema.StrategyDefinition] = {}
        self.backtested: set[str] = set()
        self.attempts = 0

    def run(self, args: str) -> tuple[bool, str, str]:
        """Return (allowed, command name, result text) for the agent's argument string."""
        allowed, reason = check_args(args)
        parts = args.strip().split(None, 1)
        command = parts[0] if parts else ""
        if not allowed:
            return False, command, _dump({"ok": False, "errors": [f"Blocked: {reason}"]})
        joined = normalise_args(args) or ""
        return True, command, _dump(self._dispatch(joined))

    def _dispatch(self, joined: str) -> dict[str, Any]:
        command, _, rest = joined.partition(" ")
        if command == "blocks":
            return self._blocks()
        if command == "validate":
            return self._validate(rest)
        if command == "approve" and " --code " in rest:
            strategy_id, _, code = rest.partition(" --code ")
            return self._approve_with_code(strategy_id, code)
        if command in ("backtest", "report", "approve"):
            return self._by_id(command, rest)
        if command == "why":
            return self._canned(self.overrides.get("why", "why.json"), "")
        if command == "paper-status":
            return self._canned("paper_status.json", "")
        return self._canned("data_status.json", "")

    def _base(self, ok: bool) -> dict[str, Any]:
        return {"ok": ok, "data_mode": "fake", "model": self.model}

    def _blocks(self) -> dict[str, Any]:
        out = self._base(True)
        out["schema"] = schema.strategy_json_schema()
        out["closest_supported"] = closest_supported_rows()
        return out

    def _validate(self, json_text: str) -> dict[str, Any]:
        try:
            strategy = schema.strategy_from_json(json_text)
        except ValidationError as error:
            out = self._base(False)
            out["errors"] = schema.explain_validation_error(error)
            return out
        sid = schema.strategy_id(strategy)
        self.validated[sid] = strategy
        self.attempts += 1
        out = self._base(True)
        out["strategy_id"] = sid
        out["version"] = self.attempts
        out["rules_text"] = schema.describe_rules(strategy)
        out["test_windows"] = [
            {"label": w.label, "start": w.start.isoformat(), "end": w.end.isoformat(),
             "coverage_warning": w.coverage_warning}
            for w in schema.backtest_windows(strategy)
        ]  # fmt: skip
        return out

    def _by_id(self, command: str, strategy_id: str) -> dict[str, Any]:
        strategy = self.validated.get(strategy_id)
        if strategy is None:
            out = self._base(False)
            out["errors"] = ["That strategy id has not been validated in this conversation."]
            return out
        if command == "approve":
            if strategy_id not in self.backtested:
                out = self._base(False)
                out["errors"] = ["Run a backtest before approving."]
                return out
            # Like the real CLI (D20): the code goes to the user, never to the agent.
            out = self._base(True)
            out.update(
                {"status": "code_sent", "strategy_id": strategy_id, "expires_in_minutes": 10}
            )
            return out
        self.backtested.add(strategy_id)
        default = "backtest_news.json" if schema.uses_news(strategy) else "backtest_plain.json"
        return self._canned(self.overrides.get("backtest", default), strategy_id)

    def _approve_with_code(self, strategy_id: str, code: str) -> dict[str, Any]:
        """Approve only with the code a case says was sent to the user's Telegram."""
        if strategy_id not in self.backtested or code != APPROVAL_CODE:
            out = self._base(False)
            out["errors"] = ["That code is not valid for this strategy."]
            return out
        return self._canned("approve.json", strategy_id)

    def _canned(self, filename: str, strategy_id: str) -> dict[str, Any]:
        text = (
            (CANNED_DIR / filename).read_text(encoding="utf-8").replace(ID_PLACEHOLDER, strategy_id)
        )
        out: dict[str, Any] = json.loads(text)
        if "model" in out:
            out["model"] = self.model
        return out


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


def strip_frontmatter(skill_text: str) -> str:
    """Remove the YAML frontmatter block so the body is the system prompt."""
    return re.sub(r"\A---\n.*?\n---\n", "", skill_text, count=1, flags=re.DOTALL)


def _tool_args(call: dict[str, Any]) -> str:
    """Pull the `args` string out of one tool call, tolerating bad JSON."""
    try:
        parsed = json.loads(call["function"].get("arguments") or "{}")
        return str(parsed.get("args", "")) if isinstance(parsed, dict) else ""
    except (json.JSONDecodeError, KeyError):
        return ""


def run_case(
    case: dict[str, Any], chat: ChatFn, system_prompt: str, model: str, max_rounds: int = 8
) -> dict[str, Any]:
    """Drive one case through the model and return the full transcript."""
    bot = SimulatedBot(model, case.get("tool_overrides"))
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    events: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    for turn, user_text in enumerate(case["turns"]):
        messages.append({"role": "user", "content": user_text})
        events.append({"turn": turn, "type": "user", "text": user_text})
        _run_turn(turn, messages, events, usage, chat, bot, max_rounds)
    return {"case_id": case["id"], "category": case["category"], "model": model,
            "events": events, "usage": usage}  # fmt: skip


def _run_turn(
    turn: int,
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    usage: dict[str, int],
    chat: ChatFn,
    bot: SimulatedBot,
    max_rounds: int,
) -> None:
    """Let the model answer one user message, running tool calls until it replies in text."""
    for _ in range(max_rounds):
        response = chat(messages, [TOOL_SPEC])
        for key in usage:
            usage[key] += int(response.get("usage", {}).get(key, 0))
        message = response["message"]
        calls = message.get("tool_calls") or []
        messages.append(message)
        events.append({"turn": turn, "type": "assistant", "text": message.get("content") or "",
                       "has_tool_calls": bool(calls)})  # fmt: skip
        if not calls:
            return
        for call in calls:
            _run_call(turn, call, messages, events, bot)
    events.append({"turn": turn, "type": "note", "text": "round limit reached"})


def _run_call(
    turn: int,
    call: dict[str, Any],
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    bot: SimulatedBot,
) -> None:
    """Run one tool call through the allowlist and the simulated bot."""
    if call["function"].get("name") != TOOL_NAME:
        allowed, command, result = False, "", _dump({"ok": False, "errors": ["Unknown tool."]})
        args = f"<tool {call['function'].get('name')}>"
    else:
        args = _tool_args(call)
        allowed, command, result = bot.run(args)
    events.append({"turn": turn, "type": "tool_call", "args": args, "command": command,
                   "allowed": allowed})  # fmt: skip
    events.append({"turn": turn, "type": "tool_result", "command": command, "content": result})
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})


def shlex_quote_json(json_text: str) -> str:
    """Wrap JSON in single quotes as the skill shows it (helper for tests)."""
    return shlex.quote(json_text)
