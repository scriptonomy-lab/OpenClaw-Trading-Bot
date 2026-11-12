"""Run the agent eval set and write results.

    uv run python evals/run_evals.py --mode direct [--model openai/<id>] [--cases a,b]

Direct mode is an approximation of OpenClaw (see the summary header). Needs
OPENAI_API_KEY and a pinned model (OPENCLAW_MODEL or --model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from bot.config import secret

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import openclaw_mode, scoring  # noqa: E402
from evals.harness import ChatFn, run_case, strip_frontmatter  # noqa: E402
from evals.responses_api import (  # noqa: E402
    messages_to_items,
    output_to_message,
    tools_to_responses,
    usage_from_responses,
)

CASES_FILE = REPO_ROOT / "evals" / "cases.json"
SKILL_FILE = REPO_ROOT / "openclaw" / "skills" / "strategy-lab" / "SKILL.md"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_TOKEN_LIMIT = 3_000_000
OPTIONAL_PARAMS: dict[str, Any] = {"temperature": 0, "seed": 7}
APPROXIMATION_NOTE = (
    "Direct mode is an approximation of OpenClaw. The model gets SKILL.md as its only "
    "system prompt and one stand-in exec tool. There is no real OpenClaw exec policy, "
    "no OpenClaw system prompt or tool set, and no Telegram. The allowlist regex is the "
    "real one, tool outputs are real schema output or canned fake-mode JSON."
)


class TokenBudgetExceededError(RuntimeError):
    """Raised when a run would pass its total token limit."""


def load_cases(path: Path = CASES_FILE) -> list[dict[str, Any]]:
    """Read a case file. The held-out file is hash-checked first and refused if it changed."""
    openclaw_mode.verify_cases_file(path)
    cases: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return cases


def resolve_model(cli_model: str | None) -> tuple[str, str]:
    """Return (pinned id as given, API model id). Refuses to run with no model (D8)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    pinned = (cli_model or os.environ.get("OPENCLAW_MODEL", "")).strip()
    if not pinned:
        raise SystemExit(
            "No model is pinned. Set OPENCLAW_MODEL in .env (for example "
            "OPENCLAW_MODEL=openai/<exact-model-id>) or pass --model openai/<id>."
        )
    return pinned, pinned.removeprefix("openai/")


class OpenAIChat:
    """Chat Completions caller with retry, a token budget and optional-parameter fallback."""

    def __init__(self, api_model: str, token_limit: int, reasoning_effort: str) -> None:
        key = secret("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        self.api_model = api_model
        self.token_limit = token_limit
        self.headers = {"Authorization": f"Bearer {key}"}
        self.params = dict(OPTIONAL_PARAMS)
        # Chat Completions rejects function tools on these models unless this is "none".
        self.reasoning_effort = reasoning_effort
        self.dropped: list[str] = []
        self.total_tokens = 0

    def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.total_tokens >= self.token_limit:
            raise TokenBudgetExceededError(f"{self.total_tokens} tokens used")
        for _ in range(8):
            payload = {
                "model": self.api_model,
                "messages": messages,
                "tools": tools,
                "reasoning_effort": self.reasoning_effort,
                **self.params,
            }
            reply = requests.post(OPENAI_URL, headers=self.headers, json=payload, timeout=180)
            if reply.status_code == 200:
                body = reply.json()
                usage = body.get("usage", {})
                self.total_tokens += int(usage.get("total_tokens", 0))
                reasoning = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )
                return {
                    "message": body["choices"][0]["message"],
                    "usage": {**usage, "reasoning_tokens": reasoning},
                }
            if reply.status_code == 400 and self._drop_rejected_param(reply):
                continue
            if "insufficient_quota" in reply.text or "credit_balance_exhausted" in reply.text:
                raise TokenBudgetExceededError("OpenAI account has no credits remaining.")
            if reply.status_code in (429, 500, 502, 503, 504):
                print(f"retry after HTTP {reply.status_code}: {reply.text[:200]}", file=sys.stderr)
                time.sleep(10)
                continue
            if reply.status_code == 400:
                print(f"HTTP 400: {reply.text[:300]}", file=sys.stderr)
            raise RuntimeError(f"OpenAI error {reply.status_code}: {reply.text[:500]}")
        raise RuntimeError("OpenAI call failed after retries.")

    def _drop_rejected_param(self, reply: requests.Response) -> bool:
        """If the API rejected temperature or seed, drop it and report that we did."""
        try:
            error = reply.json().get("error", {})
        except ValueError:
            return False
        bad = error.get("param")
        if bad in self.params:
            self.dropped.append(f"{bad}: {error.get('message', '')[:120]}")
            del self.params[bad]
            return True
        return False


class OpenAIResponses:
    """Responses API caller. Same retry and token guard; no temperature or seed."""

    def __init__(self, api_model: str, token_limit: int, reasoning_effort: str) -> None:
        key = secret("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        self.api_model = api_model
        self.token_limit = token_limit
        self.headers = {"Authorization": f"Bearer {key}"}
        # If the model rejects the effort, step down this list; None means omit it.
        self.efforts: list[str | None] = [reasoning_effort, "minimal", None]
        self.params: dict[str, Any] = {}
        self.dropped: list[str] = []
        self.total_tokens = 0
        self.system_prompt = ""

    @property
    def reasoning_effort(self) -> str:
        """The effort actually in use ("omitted" if the model rejected every value)."""
        return self.efforts[0] or "omitted"

    def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.total_tokens >= self.token_limit:
            raise TokenBudgetExceededError(f"{self.total_tokens} tokens used")
        for _ in range(8):
            payload: dict[str, Any] = {
                "model": self.api_model,
                "instructions": messages[0]["content"],
                "input": messages_to_items(messages),
                "tools": tools_to_responses(tools),
            }
            if self.efforts[0] is not None:
                payload["reasoning"] = {"effort": self.efforts[0]}
            reply = requests.post(RESPONSES_URL, headers=self.headers, json=payload, timeout=300)
            if reply.status_code == 200:
                body = reply.json()
                usage = usage_from_responses(body.get("usage", {}))
                self.total_tokens += usage["total_tokens"]
                return {"message": output_to_message(body.get("output", [])), "usage": usage}
            if "insufficient_quota" in reply.text or "credit_balance_exhausted" in reply.text:
                raise TokenBudgetExceededError("OpenAI account has no credits remaining.")
            if reply.status_code == 400 and "reasoning" in reply.text and len(self.efforts) > 1:
                self.dropped.append(f"effort {self.efforts[0]}: {reply.text[:160]}")
                self.efforts.pop(0)
                continue
            if reply.status_code in (429, 500, 502, 503, 504):
                print(f"retry after HTTP {reply.status_code}: {reply.text[:200]}", file=sys.stderr)
                time.sleep(10)
                continue
            raise RuntimeError(f"OpenAI error {reply.status_code}: {reply.text[:500]}")
        raise RuntimeError("OpenAI call failed after retries.")


def render_transcript(t: dict[str, Any]) -> str:
    """Plain text view of one transcript for the summary."""
    lines = []
    for e in t["events"]:
        if e["type"] == "user":
            lines.append(f"[turn {e['turn']}] USER: {e['text']}")
        elif e["type"] == "assistant":
            suffix = " (tool call follows)" if e.get("has_tool_calls") else ""
            lines.append(f"[turn {e['turn']}] AGENT{suffix}: {e['text']}")
        elif e["type"] == "tool_call":
            state = "allowed" if e["allowed"] else "BLOCKED"
            lines.append(f"[turn {e['turn']}] TOOL CALL ({state}): {e['args']}")
        elif e["type"] == "tool_result":
            lines.append(f"[turn {e['turn']}] TOOL RESULT: {e['content'][:1500]}")
        else:
            lines.append(f"[turn {e['turn']}] NOTE: {e['text']}")
    return "\n".join(lines)


def build_summary(results_doc: dict[str, Any], transcripts: dict[str, dict[str, Any]]) -> str:
    """The summary.md text. Every number is read from results_doc."""
    h = results_doc["header"]
    out = [
        "# Agent eval summary", "",
        f"- Mode: {h['mode']}", f"- Model (exact, D8): {h['model']}",
        f"- Date: {h['date']}", f"- Data mode: {h['data_mode']}",
        f"- API: {h['api']}, reasoning effort: {h['reasoning_effort']}",
        f"- Sampling parameters sent: {h['sampling_params']}",
        f"- Tokens used: {h['usage']['total_tokens']} "
        f"(input {h['usage']['prompt_tokens']}, output {h['usage']['completion_tokens']}, "
        f"of which reasoning {h['usage']['reasoning_tokens']})",
    ]  # fmt: skip
    out += [f"- {line}" for line in h.get("extra_lines", [])]
    out += [
        "", f"**{h['approximation_note']}**", "",
        "## Accuracy", "", "| Category | Passed | Total | Accuracy % |", "|---|---|---|---|",
    ]  # fmt: skip
    for name, a in results_doc["accuracy"].items():
        out.append(f"| {name} | {a['passed']} | {a['total']} | {a['accuracy_pct']} |")
    if results_doc["not_run"]:
        out += ["", f"Not run (token limit hit): {', '.join(results_doc['not_run'])}"]
    out += ["", "## Failures", ""]
    failed = [r for r in results_doc["cases"] if not r["passed"]]
    if not failed:
        out.append("None.")
    for r in failed:
        bad = [f"{c['name']} ({c['detail']})" for c in r["checks"] if not c["passed"]]
        out += [f"### {r['case_id']} ({r['category']})", "", "Failed checks: " + "; ".join(bad),
                "", "```", render_transcript(transcripts[r["case_id"]]), "```", ""]  # fmt: skip
    return "\n".join(out) + "\n"


def run_direct(args: argparse.Namespace) -> int:
    """Run the cases against the Chat Completions API."""
    pinned, api_model = resolve_model(args.model)
    cases = load_cases(Path(args.cases_file))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    system_prompt = strip_frontmatter(SKILL_FILE.read_text(encoding="utf-8"))
    api_class = OpenAIResponses if args.api == "responses" else OpenAIChat
    chat = api_class(api_model, args.token_limit, args.reasoning_effort)
    results: list[dict[str, Any]] = []
    transcripts: dict[str, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    not_run: list[str] = []
    for case in cases:
        if not_run:
            not_run.append(case["id"])
            continue
        try:
            t = run_case(case, _as_chat(chat), system_prompt, pinned)
        except TokenBudgetExceededError:
            not_run.append(case["id"])
            continue
        transcripts[case["id"]] = t
        for key in usage:
            usage[key] += t["usage"][key]
        results.append(scoring.score_case(case, t))
        print(f"{case['id']}: {'PASS' if results[-1]['passed'] else 'FAIL'}", flush=True)
    now = datetime.now(UTC)
    sampling = ", ".join(f"{k}={v}" for k, v in chat.params.items()) or "none (API rejected them)"
    sampling += f", reasoning_effort={chat.reasoning_effort}"
    if chat.dropped:
        sampling += f" (fallbacks used: {len(chat.dropped)})"
    header = {
        "mode": "direct", "model": pinned, "date": now.strftime("%Y-%m-%d %H:%M UTC"),
        "data_mode": "fake (canned tool outputs)", "approximation_note": APPROXIMATION_NOTE,
        "api": args.api,
        "reasoning_effort": chat.reasoning_effort,
        "sampling_params": sampling, "dropped_params": chat.dropped, "usage": usage,
        "token_limit": args.token_limit,
    }  # fmt: skip
    doc = {"header": header, "accuracy": scoring.accuracy(results), "cases": results,
           "not_run": not_run}  # fmt: skip
    write_outputs(now, doc, transcripts, Path(args.out_root))
    return 2 if not_run else 0


def _as_chat(chat: OpenAIChat | OpenAIResponses) -> ChatFn:
    return chat


def write_outputs(
    now: datetime, doc: dict[str, Any], transcripts: dict[str, dict[str, Any]], out_root: Path
) -> Path:
    """Write results.json, transcripts.jsonl and summary.md under outputs/simulated/evals/."""
    folder = out_root / "simulated" / "evals" / now.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with (folder / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for t in transcripts.values():
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    (folder / "summary.md").write_text(build_summary(doc, transcripts), encoding="utf-8")
    print(f"Wrote {folder}")
    return folder


OPENCLAW_NOTE = (
    "OpenClaw mode runs the real OpenClaw gateway (isolated state dir, Telegram plugin off) "
    "with the rendered tool policy and exec allowlist. Tool outputs are REAL openclaw-bot "
    "CLI outputs in fake data mode, not canned. Two exceptions, only when the real command "
    "succeeded: injection cases (tool_overrides named *_injected.json) get the injected "
    "fields added to the real output, and backtest_plain_few.json replaces the output. "
    "Other overrides are ignored. The agent sees OpenClaw's own system prompt, tools and "
    "skill loading."
)


def run_openclaw_case(
    gateway: openclaw_mode.EvalGateway, case: dict[str, Any], model: str, thinking: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Send each user turn to a fresh session, then build the transcript from the session."""
    openclaw_mode.reset_case_state(gateway.sandbox, case)
    session_id = f"eval-{case['id'].replace('_', '-')}-{int(time.time())}"
    real_codes: dict[int, str] = {}
    failures: list[str] = []
    payloads: dict[int, str] = {}
    for turn, text in enumerate(case["turns"]):
        sent, real = openclaw_mode.approval_code_for_turn(text, gateway.code_file)
        if real:
            real_codes[turn] = real
        reply = gateway.send(session_id, sent, thinking)
        if reply.get("ok") is False or reply.get("status") not in (None, "ok"):
            failures.append(f"turn {turn}: {json.dumps(reply)[:300]}")
            break
        texts = [p.get("text") or "" for p in (reply.get("result") or {}).get("payloads", [])]
        payloads[turn] = "\n".join(t for t in texts if t)
    messages = gateway.session_messages(session_id)
    parsed = openclaw_mode.parse_session(messages, case["turns"], real_codes)
    events = parsed.events
    for turn, text in payloads.items():  # fall back to the CLI payload if the transcript lacks it
        has_final = any(
            e["turn"] == turn and e["type"] == "assistant" and not e.get("has_tool_calls")
            for e in events
        )
        if text and not has_final:
            events.append(
                {"turn": turn, "type": "assistant", "text": text, "has_tool_calls": False}
            )
    events += [{"turn": max(len(payloads), 1) - 1, "type": "note", "text": f} for f in failures]
    events.sort(key=lambda e: e["turn"])
    transcript = {
        "case_id": case["id"], "category": case["category"], "model": model,
        "session_id": session_id, "events": events, "usage": parsed.usage,
    }  # fmt: skip
    return transcript, messages


def run_openclaw(args: argparse.Namespace) -> int:
    """Run the cases through an isolated OpenClaw gateway."""
    pinned, _ = resolve_model(args.model)
    cases_path = Path(args.cases_file)
    cases_sha = openclaw_mode.verify_cases_file(cases_path)
    cases = load_cases(cases_path)
    if args.cases:
        cases = [c for c in cases if c["id"] in set(args.cases.split(","))]
    key = secret("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    results: list[dict[str, Any]] = []
    transcripts: dict[str, dict[str, Any]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "reasoning_tokens": 0, "cost_usd": 0.0}  # fmt: skip
    not_run: list[str] = []
    with openclaw_mode.EvalGateway(pinned, key) as gateway:
        version = gateway.version()
        for case in cases:
            if usage["total_tokens"] >= args.token_limit:
                not_run.append(case["id"])
                continue
            t, messages = run_openclaw_case(gateway, case, pinned, args.thinking)
            transcripts[case["id"]], raw[case["id"]] = t, messages
            for k in usage:
                usage[k] += t["usage"][k]
            results.append(scoring.score_case(case, t))
            print(f"{case['id']}: {'PASS' if results[-1]['passed'] else 'FAIL'}", flush=True)
    now = datetime.now(UTC)
    header = {
        "mode": "openclaw", "model": pinned, "date": now.strftime("%Y-%m-%d %H:%M UTC"),
        "data_mode": "fake (tool outputs are real CLI outputs, not canned)",
        "approximation_note": OPENCLAW_NOTE,
        "api": "OpenClaw gateway via `openclaw agent --json`",
        "reasoning_effort": args.thinking or "OpenClaw default",
        "sampling_params": "OpenClaw defaults (harness sets none)", "dropped_params": [],
        "usage": usage, "token_limit": args.token_limit, "openclaw_version": version,
        "cases_file": str(cases_path), "cases_sha256": cases_sha,
        "extra_lines": [
            f"OpenClaw version: {version}",
            f"Cases file: {cases_path} (sha256 {cases_sha})",
            f"Cost reported by OpenClaw: ${usage['cost_usd']:.4f}",
            "Approval code: the real code the bot wrote to its test code file replaces "
            f"{openclaw_mode.APPROVAL_CODE_PLACEHOLDER} in the message sent to the agent; "
            "the transcript keeps the placeholder and has a note.",
        ],
    }  # fmt: skip
    doc = {"header": header, "accuracy": scoring.accuracy(results), "cases": results,
           "not_run": not_run}  # fmt: skip
    folder = write_outputs(now, doc, transcripts, Path(args.out_root))
    raw_dir = folder / "sessions_raw"
    raw_dir.mkdir(exist_ok=True)
    for case_id, messages in raw.items():
        text = openclaw_mode.scrub(json.dumps(messages, indent=1, ensure_ascii=False))
        (raw_dir / f"{case_id}.json").write_text(text + "\n", encoding="utf-8")
    return 2 if not_run else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run the agent evals.")
    parser.add_argument("--mode", choices=["direct", "openclaw"], required=True)
    parser.add_argument("--model", help="override OPENCLAW_MODEL, for example openai/<id>")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument(
        "--cases-file", default=str(CASES_FILE), help="case file (default evals/cases.json)"
    )
    parser.add_argument("--thinking", help="openclaw mode: thinking level (default: OpenClaw's)")
    parser.add_argument("--api", choices=["chat", "responses"], default="responses")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="sent to the API; function tools need 'none' on some models",
    )
    parser.add_argument("--token-limit", type=int, default=DEFAULT_TOKEN_LIMIT)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs"))
    args = parser.parse_args(argv)
    return run_direct(args) if args.mode == "direct" else run_openclaw(args)


if __name__ == "__main__":
    raise SystemExit(main())
