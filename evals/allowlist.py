"""The exec allowlist check, reused from the project's own render script (D5, D9)."""

from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
APPROVALS_FILE = REPO_ROOT / "openclaw" / "exec-approvals.json"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_openclaw_config.py"

# Real OpenClaw gets argv, not one string. A newline or tab inside a string would
# vanish when we split it, so these are rejected before any splitting.
CONTROL_CHARS = ("\n", "\r", "\t", "\x00")


def _load_render_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_openclaw_config", RENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RENDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RENDER = _load_render_module()
ARG_PATTERN: str = _RENDER.load_arg_pattern(APPROVALS_FILE.read_text(encoding="utf-8"))


def normalise_args(args: str) -> str | None:
    """Turn the agent's argument string into the argv-joined form, or None if unparsable."""
    if any(char in args for char in CONTROL_CHARS):
        return None
    try:
        argv = shlex.split(args)
    except ValueError:
        return None
    return " ".join(argv)


def check_args(args: str) -> tuple[bool, str]:
    """Return (allowed, reason) for the text the agent put after `openclaw-bot`."""
    joined = normalise_args(args)
    if joined is None:
        return False, "Could not parse the command safely."
    if not bool(_RENDER.matches_like_openclaw(ARG_PATTERN, joined)):
        return False, "Command does not match the exec allowlist."
    return True, "ok"


def is_rejected(command_string: str) -> bool:
    """True if the regex rejects this exact argv-joined string (used by the D9 test)."""
    return not bool(_RENDER.matches_like_openclaw(ARG_PATTERN, command_string))
