"""Support code for `run_evals.py --mode openclaw`: an isolated OpenClaw gateway and transcripts.

How it works (OpenClaw 2026.10.9):

* A temp state dir and config, never the user's `~/.openclaw`. Paths come from the
  environment variables OPENCLAW_STATE_DIR, OPENCLAW_CONFIG_PATH and OPENCLAW_HOME
  (https://docs.openclaw.ai/help/environment) and the port from OPENCLAW_GATEWAY_PORT
  plus `openclaw gateway --port` (https://docs.openclaw.ai/cli/gateway).
* One message per call: `openclaw agent --agent main --session-id <id> --message <text>
  --json` (https://docs.openclaw.ai/cli/agent). The reply is in `result.payloads[].text`.
* Tool calls are not in that JSON. They are read from the session transcript with the
  gateway method `sessions.get` (`openclaw gateway call sessions.get`, see
  https://docs.openclaw.ai/cli/gateway). The transcript itself lives in
  `<state dir>/agents/main/agent/openclaw-agent.sqlite`, table `transcript_events`
  (some rows are zstd compressed, so we ask the gateway to decode them).
* Exec policy: the same `tools.exec` block and the same approvals document as the
  rendered files, applied with `openclaw approvals set --gateway`
  (https://docs.openclaw.ai/cli/approvals, https://docs.openclaw.ai/tools/exec-approvals).

The model runs in "code mode": its `exec` tool takes JavaScript, and the shell commands it
runs appear as nested records (custom message `openclaw.nested-tool.v1`) with the command
and the result. A command the allowlist refuses appears there with `isError` and the text
"exec denied: allowlist miss".

Sandbox project (why): the real `openclaw-bot` reads the repo `.env`, which holds the
Telegram token, and sends charts and approval codes to the user's Telegram even in fake
mode. So the eval gateway's allowlist pins a copy of `bot/` in a temp folder that has no
`.env`. The code is the same files; only the folder differs. The copy's entry script also
forces fake data mode and can replace the output of one command with a canned injection
file (cases that carry `tool_overrides`), but only when the real command succeeded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
RENDERED_CONFIG = REPO_ROOT / "openclaw" / "rendered" / "openclaw.json5"
RENDERED_APPROVALS = REPO_ROOT / "openclaw" / "rendered" / "exec-approvals.json"
HELDOUT_NAME = "cases_heldout.json"
HELDOUT_HASH_FILE = EVALS_DIR / "cases_heldout.sha256"
EVAL_PORT = 18889
EVAL_GATEWAY_TOKEN = "eval-gateway-local-token"
AGENT_ID = "main"
APPROVAL_CODE_PLACEHOLDER = "482913"
NVM_NODE_BIN = Path.home() / ".nvm" / "versions" / "node" / "v26.13.0" / "bin"
DENIED_RE = re.compile(r"\bdenied\b|allowlist miss|outside .*workspace", re.IGNORECASE)
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")


class HeldoutHashError(RuntimeError):
    """The held-out case file does not match its recorded sha256."""


# ------------------------------------------------------------------ case file hash


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cases_file(path: Path, hash_file: Path = HELDOUT_HASH_FILE) -> str:
    """Return the file's sha256. For the held-out file, refuse if it differs from the record."""
    actual = sha256_file(path)
    if path.name != HELDOUT_NAME:
        return actual
    if not hash_file.exists():
        raise HeldoutHashError(f"No recorded hash at {hash_file}; refusing to run.")
    recorded = hash_file.read_text(encoding="utf-8").split()[0].strip().lower()
    if recorded != actual:
        raise HeldoutHashError(
            f"{path.name} sha256 {actual} does not match the recorded {recorded}. Refusing to run."
        )
    return actual


# ------------------------------------------------------------------------- JSON5


def _blank_comments(text: str) -> str:
    """Drop // comments that sit outside strings."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def json5_loads(text: str) -> dict[str, Any]:
    """Parse the small JSON5 subset our rendered config uses: comments, bare keys, commas."""
    pieces = re.split(r'("(?:[^"\\]|\\.)*")', _blank_comments(text))
    fixed = []
    for index, piece in enumerate(pieces):
        if index % 2 == 0:  # outside strings
            piece = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', piece)
            piece = re.sub(r",(\s*[}\]])", r"\1", piece)
        fixed.append(piece)
    loaded: dict[str, Any] = json.loads("".join(fixed))
    return loaded


# --------------------------------------------------------------- config and env


def build_gateway_config(
    rendered: dict[str, Any], port: int, token: str, sandbox_bin_dir: Path, model: str
) -> dict[str, Any]:
    """Derive the eval gateway config from the rendered one. Tool policy is copied as is.

    Telegram: the strategy-lab skill only loads when `channels.telegram.enabled` is true
    (its `requires.config`). So that key stays true, but the Telegram plugin is disabled,
    the bot token is removed and the gateway env has no token, so nothing connects.
    """
    config = json.loads(json.dumps(rendered))
    config["agents"]["defaults"]["model"] = {"primary": model}
    config["agents"]["defaults"]["models"] = {model: {}}
    telegram = dict(config.get("channels", {}).get("telegram", {}))
    telegram.pop("botToken", None)
    telegram["enabled"] = True
    config["channels"] = {"telegram": telegram}
    config["plugins"] = {
        "entries": {
            "openai": {"enabled": True},
            "codex": {"enabled": False},
            "telegram": {"enabled": False},
        }
    }
    config["gateway"] = {
        "mode": "local",
        "port": port,
        "bind": "loopback",
        "auth": {"mode": "token", "token": token},
    }
    config["telemetry"] = {"enabled": False}
    config["tools"]["exec"]["pathPrepend"] = [str(sandbox_bin_dir)]
    return config


def build_approvals(rendered: dict[str, Any], sandbox_binary: Path) -> dict[str, Any]:
    """The rendered approvals with the pinned binary path swapped for the sandbox copy."""
    approvals = json.loads(json.dumps(rendered))
    for entry in approvals["agents"][AGENT_ID]["allowlist"]:
        entry["pattern"] = str(sandbox_binary)
    return approvals


def build_gateway_env(
    *,
    state_dir: Path,
    config_path: Path,
    home_dir: Path,
    port: int,
    code_file: Path,
    openai_key: str,
    token: str,
    model: str,
    node_bin_dir: Path,
    base_path: str = "/usr/bin:/bin:/usr/sbin:/sbin",
) -> dict[str, str]:
    """Environment for the eval gateway. Built from scratch, so no Telegram token leaks in."""
    return {
        "PATH": f"{node_bin_dir}:{base_path}",
        "HOME": str(Path.home()),
        "OPENCLAW_HOME": str(home_dir),
        "OPENCLAW_STATE_DIR": str(state_dir),
        "OPENCLAW_CONFIG_PATH": str(config_path),
        "OPENCLAW_GATEWAY_PORT": str(port),
        "OPENCLAW_GATEWAY_TOKEN": token,
        "OPENCLAW_MODEL": model,
        "EODHD_MODE": "fake",
        "OPENCLAW_BOT_TEST_APPROVAL_CODE_FILE": str(code_file),
        "OPENAI_API_KEY": openai_key,
    }


# ----------------------------------------------------------------------- sandbox

ENTRY_SCRIPT = '''"""Entry point for the eval copy of openclaw-bot (written by openclaw_mode.py)."""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["EODHD_MODE"] = "fake"
os.environ.setdefault("OPENCLAW_BOT_TEST_APPROVAL_CODE_FILE", str(HERE / "approval_code.txt"))

from bot.cli import main  # noqa: E402


def canned_for(argv: list[str]) -> tuple[str, dict] | None:
    """(mode, canned data) for this command if the current case asks for an override."""
    path = HERE / "overrides.json"
    if not path.exists() or not argv:
        return None
    entry = json.loads(path.read_text()).get(argv[0])
    if not entry:
        return None
    text = (HERE / "canned" / entry["file"]).read_text()
    strategy_id = argv[1] if len(argv) > 1 else ""
    return entry["mode"], json.loads(text.replace("__STRATEGY_ID__", strategy_id))


def apply_override(real_text: str, mode: str, canned: dict) -> str:
    """Merge adds the canned keys the real output lacks; replace swaps the whole output."""
    if mode == "replace":
        return json.dumps(canned, indent=2) + "\\n"
    real = json.loads(real_text)
    for key, value in canned.items():
        real.setdefault(key, value)
    return json.dumps(real, indent=2) + "\\n"


def run() -> int:
    """Run the real command. Change its output only if it succeeded and an override exists."""
    override = canned_for(sys.argv[1:])
    if override is None:
        return int(main())
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = int(main())
    try:
        real_ok = bool(json.loads(buffer.getvalue()).get("ok"))
    except ValueError:
        real_ok = False
    text = apply_override(buffer.getvalue(), *override) if real_ok else buffer.getvalue()
    sys.stdout.write(text)
    return code


sys.exit(run())
'''


def make_sandbox(root: Path) -> Path:
    """Copy bot/ (no .env) into `root`. Returns the path of the sandbox `openclaw-bot`."""
    shutil.copytree(
        REPO_ROOT / "bot", root / "bot", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (root / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "docs" / "design", root / "docs" / "design", dirs_exist_ok=True)
    shutil.copytree(EVALS_DIR / "canned", root / "canned", dirs_exist_ok=True)
    (root / "entry.py").write_text(ENTRY_SCRIPT, encoding="utf-8")
    binary = root / ".venv" / "bin" / "openclaw-bot"
    binary.parent.mkdir(parents=True, exist_ok=True)
    python = REPO_ROOT / ".venv" / "bin" / "python3"
    binary.write_text(f'#!/bin/sh\nexec "{python}" "{root / "entry.py"}" "$@"\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary


REPLACE_OVERRIDES = {"backtest_plain_few.json"}


def override_mode(filename: str) -> str | None:
    """How a canned file is applied in OpenClaw mode.

    `*_injected.json`: merge the injected fields into the real output, so the numbers stay
    consistent. `backtest_plain_few.json`: replace, because the fake data cannot be made to
    give few trades. Anything else: None, the real output is used.
    """
    if filename.endswith("_injected.json"):
        return "merge"
    return "replace" if filename in REPLACE_OVERRIDES else None


def reset_case_state(sandbox: Path, case: dict[str, Any]) -> None:
    """Fresh bot database, no leftover code, and this case's output overrides."""
    outputs = sandbox / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)
    (sandbox / "approval_code.txt").unlink(missing_ok=True)
    overrides = {}
    for command, filename in (case.get("tool_overrides") or {}).items():
        mode = override_mode(filename)
        if mode:
            overrides[command] = {"file": filename, "mode": mode}
    (sandbox / "overrides.json").write_text(json.dumps(overrides), encoding="utf-8")


def approval_code_for_turn(text: str, code_file: Path) -> tuple[str, str | None]:
    """Swap the case's fixed code for the real one the bot wrote. Returns (text, real code)."""
    if APPROVAL_CODE_PLACEHOLDER not in text or not code_file.exists():
        return text, None
    real = code_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]{6}", real):
        return text, None
    return text.replace(APPROVAL_CODE_PLACEHOLDER, real), real


# ---------------------------------------------------------------- transcript parse


def split_command(command: str) -> tuple[str, str]:
    """(command name, args) the way the scorer expects.

    `openclaw-bot validate '{...}'` gives ("validate", "validate '{...}'"). Anything that is
    not an openclaw-bot call keeps its whole text as args and its first word as the name.
    """
    stripped = command.strip()
    first, _, rest = stripped.partition(" ")
    if Path(first).name == "openclaw-bot":
        args = rest.strip()
        return (args.split(None, 1)[0] if args else ""), args
    return (first, stripped)


def _text_of(content: Any) -> str:
    """Join the text blocks of a message or tool result."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def _clean_output(text: str) -> str:
    """The JSON object inside a tool output, or the text unchanged if there is none."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            json.loads(text[start : end + 1])
            return text[start : end + 1]
        except ValueError:
            return text
    return text


OMITTED_MARKER = "[Nested tool output omitted"


def _aggregated_output_after(messages: list[dict[str, Any]], index: int) -> str:
    """Stdout of the exec the agent saw, from the next top-level result.

    Long nested outputs are stored as a placeholder; the full text is in the `aggregated`
    field of the wrapping tool result.
    """
    for later in messages[index + 1 :]:
        if later.get("role") != "toolResult":
            continue
        try:
            body = json.loads(_text_of(later.get("content")))
        except ValueError:
            continue
        value = body.get("value") if isinstance(body, dict) else None
        if isinstance(value, dict) and "aggregated" in value:
            return str(value["aggregated"])  # code did `return await exec(...)`
        try:
            inner = json.loads(_text_of(body.get("output")))
        except (ValueError, AttributeError, TypeError):
            continue  # code did `text(await exec(...))`
        return str(inner.get("aggregated", ""))
    return ""


def _nested_result_text(result: dict[str, Any]) -> tuple[bool, str]:
    """(is error, text) of a nested tool result."""
    text = _text_of(result.get("content"))
    details = result.get("details") or {}
    return details.get("status") == "error", details.get("error") or text


def _tool_events(
    turn: int, name: str, tool_input: dict[str, Any], is_error: bool, text: str, real_code: str
) -> list[dict[str, Any]]:
    """A tool_call event and, for exec, a tool_result event. `wait` is plumbing, not a call."""
    if name == "wait":
        return []
    if name == "exec":
        command, args = split_command(str(tool_input.get("command", "")))
    else:
        command, args = name, json.dumps(tool_input, sort_keys=True)
    denied = is_error and bool(DENIED_RE.search(text))
    if real_code:
        args = args.replace(real_code, APPROVAL_CODE_PLACEHOLDER)
        text = text.replace(real_code, APPROVAL_CODE_PLACEHOLDER)
    events: list[dict[str, Any]] = [
        {"turn": turn, "type": "tool_call", "args": args, "command": command, "allowed": not denied}
    ]
    if name == "exec":
        content = _clean_output(text)
        events.append(
            {
                "turn": turn,
                "type": "tool_result",
                "command": command,
                "content": content,
                "raw_content": text,
            }  # fmt: skip
        )
    return events


@dataclass
class ParsedSession:
    """Events and usage read from one session."""

    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)


def parse_session(
    messages: list[dict[str, Any]], user_turns: list[str], real_codes: dict[int, str] | None = None
) -> ParsedSession:
    """Turn `sessions.get` messages into scorer events.

    `user_turns` are the case's own user texts (with the fixed code). `real_codes` maps a
    turn to the real code that was sent in its place; it is mapped back in tool calls.
    """
    real_codes = real_codes or {}
    parsed = ParsedSession(usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                  "reasoning_tokens": 0, "cost_usd": 0.0})  # fmt: skip
    turn = -1
    pending: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            turn += 1
            text = user_turns[turn] if turn < len(user_turns) else _text_of(message["content"])
            parsed.events.append({"turn": turn, "type": "user", "text": text})
            if turn in real_codes:
                parsed.events.append({"turn": turn, "type": "note", "text": (
                    f"harness substituted the approval code {APPROVAL_CODE_PLACEHOLDER} with the "
                    "real code from the bot's test code file in the message sent to the agent; "
                    "tool calls below show the placeholder again"
                )})  # fmt: skip
        elif role == "assistant":
            _add_assistant(parsed, turn, message, pending)
        elif role == "toolResult":
            _add_top_level_result(parsed, turn, message, pending, real_codes.get(turn, ""))
        elif role == "custom" and message.get("customType") == "openclaw.nested-tool.v1":
            d = message["details"]
            is_error, text = _nested_result_text(d.get("result") or {})
            if text.startswith(OMITTED_MARKER):
                text = _aggregated_output_after(messages, index) or text
            parsed.events += _tool_events(
                turn, d.get("toolName", ""), d.get("input") or {}, is_error, text,
                real_codes.get(turn, ""),
            )  # fmt: skip
    return parsed


def _add_assistant(
    parsed: ParsedSession, turn: int, message: dict[str, Any], pending: dict[str, dict[str, Any]]
) -> None:
    """Add an assistant event, its token use, and remember direct (non-code-mode) tool calls."""
    blocks = message.get("content") or []
    has_calls = any(b.get("type") == "toolCall" for b in blocks)
    parsed.events.append(
        {"turn": turn, "type": "assistant", "text": _text_of(blocks), "has_tool_calls": has_calls}
    )
    for block in blocks:
        args = block.get("arguments") or {}
        if block.get("type") == "toolCall" and "code" not in args:
            pending[block["id"]] = {"name": block.get("name", ""), "input": args}
    usage = message.get("usage") or {}
    u = parsed.usage
    u["prompt_tokens"] += (
        usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0)
    )
    u["completion_tokens"] += usage.get("output", 0)
    u["total_tokens"] += usage.get("totalTokens", 0)
    u["reasoning_tokens"] += usage.get("reasoningTokens", 0)
    u["cost_usd"] += (usage.get("cost") or {}).get("total", 0.0)


def _add_top_level_result(
    parsed: ParsedSession,
    turn: int,
    message: dict[str, Any],
    pending: dict[str, dict[str, Any]],
    real_code: str,
) -> None:
    """A result for a direct tool call (the model called exec without code mode)."""
    call = pending.pop(message.get("toolCallId", ""), None)
    if call is None:
        return
    text = _text_of(message.get("content"))
    parsed.events += _tool_events(
        turn, call["name"], call["input"], bool(message.get("isError")), text, real_code
    )


# --------------------------------------------------------------------- gateway


def find_openclaw() -> Path:
    """The openclaw binary: PATH first, then the nvm Node 26.13.0 install."""
    found = shutil.which("openclaw")
    if found:
        return Path(found)
    fallback = NVM_NODE_BIN / "openclaw"
    if fallback.exists():
        return fallback
    raise SystemExit("openclaw not found. Run: source ~/.nvm/nvm.sh && nvm use 26.13.0")


def scrub(text: str) -> str:
    """Remove anything that looks like an API key."""
    return SECRET_RE.sub("sk-REDACTED", text)


def last_json(text: str) -> dict[str, Any]:
    """Parse the JSON object in CLI output that may have log lines before it."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON in output: {text[:200]}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON output is not an object")
    return value


class EvalGateway:
    """An isolated OpenClaw gateway in a temp directory. Use as a context manager."""

    def __init__(self, model: str, openai_key: str, port: int = EVAL_PORT) -> None:
        self.model = model
        self.port = port
        self.openclaw = find_openclaw()
        self.root = Path(tempfile.mkdtemp(prefix="openclaw-eval-"))
        self.sandbox = self.root / "sandbox"
        self.state_dir = self.root / "state"
        self.code_file = self.sandbox / "approval_code.txt"
        self.process: subprocess.Popen[bytes] | None = None
        binary = make_sandbox(self.sandbox)
        for folder in (self.state_dir, self.root / "home"):
            folder.mkdir(parents=True, exist_ok=True)
        self.config_path = self.state_dir / "openclaw.json"
        rendered = json5_loads(RENDERED_CONFIG.read_text(encoding="utf-8"))
        config = build_gateway_config(rendered, port, EVAL_GATEWAY_TOKEN, binary.parent, model)
        self.config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
        approvals = build_approvals(json.loads(RENDERED_APPROVALS.read_text()), binary)
        self.approvals_path = self.root / "exec-approvals.json"
        self.approvals_path.write_text(json.dumps(approvals), encoding="utf-8")
        self.env = build_gateway_env(
            state_dir=self.state_dir, config_path=self.config_path, home_dir=self.root / "home",
            port=port, code_file=self.code_file, openai_key=openai_key,
            token=EVAL_GATEWAY_TOKEN, model=model, node_bin_dir=self.openclaw.parent,
        )  # fmt: skip

    def cli(self, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        """Run an openclaw CLI command against the eval gateway."""
        return subprocess.run(
            [str(self.openclaw), *args], env=self.env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )  # fmt: skip

    def version(self) -> str:
        """The OpenClaw version line."""
        return self.cli("--version").stdout.strip().splitlines()[0]

    def __enter__(self) -> EvalGateway:
        if socket.socket().connect_ex(("127.0.0.1", self.port)) == 0:
            raise SystemExit(f"Port {self.port} is already in use; not starting.")
        log = (self.root / "gateway.log").open("wb")
        self.process = subprocess.Popen(
            [str(self.openclaw), "gateway", "--port", str(self.port)],
            env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
        self._wait_ready()
        done = self.cli("approvals", "set", "--file", str(self.approvals_path), "--gateway")
        if done.returncode != 0:
            raise RuntimeError(f"approvals set failed: {done.stderr[:300]}")
        return self

    def _wait_ready(self, seconds: int = 90) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    "eval gateway exited early; see gateway.log in " + str(self.root)
                )
            if self.cli("gateway", "call", "health", "--json", timeout=20).returncode == 0:
                return
            time.sleep(2)
        raise RuntimeError("eval gateway did not become ready")

    def __exit__(self, *_exc: object) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        shutil.rmtree(self.root, ignore_errors=True)

    def send(
        self, session_id: str, text: str, thinking: str | None = None, timeout: int = 300
    ) -> dict[str, Any]:
        """Send one user message. Returns the CLI JSON (`result.payloads`, `result.meta`)."""
        extra = ["--thinking", thinking] if thinking else []
        out = self.cli(
            "agent", "--agent", AGENT_ID, "--session-id", session_id, "--message", text,
            "--json", "--timeout", str(timeout), *extra, timeout=timeout + 60,
        )  # fmt: skip
        try:
            return last_json(out.stdout)
        except ValueError:
            return {"ok": False, "error": {"message": (out.stdout + out.stderr)[:500]}}

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """All decoded transcript messages of a session, from the gateway."""
        params = json.dumps({"key": f"agent:{AGENT_ID}:explicit:{session_id}", "limit": 2000})
        out = self.cli("gateway", "call", "sessions.get", "--params", params, "--json")
        messages: list[dict[str, Any]] = last_json(out.stdout).get("messages", [])
        return messages
