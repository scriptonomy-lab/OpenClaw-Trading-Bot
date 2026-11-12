"""The only commands OpenClaw is allowed to run (docs/design/cli.md).

Every command prints one JSON object with ``ok``, ``data_mode`` and ``model``,
except ``daily-summary``, which prints plain text for Telegram. The exit code
is 0 when ``ok`` is true and 1 otherwise. The fake-data warning goes to stderr
when the EODHD client is imported, so stdout stays clean.

The agent-facing commands take an optional ``--data simulated|live`` flag that
sets the data mode for that one command (D34). Without it, EODHD_MODE decides.
"""

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any, NoReturn

from bot import commands, reporting
from bot.config import DATA_FLAG_TO_MODE, apply_data_override, data_mode
from bot.data import eodhd_client
from bot.data.holdout import HoldoutAccessError

# Commands the agent may run, so each takes the --data flag. Cron jobs do not.
AGENT_COMMANDS: tuple[str, ...] = (
    "blocks",
    "validate",
    "backtest",
    "report",
    "approve",
    "paper-status",
    "why",
    "data-status",
)


class ArgumentError(Exception):
    """Bad command-line arguments. Turned into a JSON error, not a usage dump."""


class JsonArgumentParser(argparse.ArgumentParser):
    """An argument parser that raises instead of printing usage and exiting."""

    def error(self, message: str) -> NoReturn:
        """Raise ArgumentError with a short message."""
        raise ArgumentError(f"Bad command line: {message}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser with every subcommand in the CLI contract."""
    parser = JsonArgumentParser(prog="openclaw-bot")
    sub = parser.add_subparsers(dest="command")
    commands_by_name = {name: sub.add_parser(name) for name in AGENT_COMMANDS}
    for name in ("paper-update", "daily-summary"):
        sub.add_parser(name)
    commands_by_name["validate"].add_argument("strategy_json")
    for name in ("backtest", "report", "approve"):
        commands_by_name[name].add_argument("strategy_id")
    commands_by_name["approve"].add_argument("--code", dest="code")
    commands_by_name["why"].add_argument("ticker")
    commands_by_name["why"].add_argument("--date", dest="day")
    for command in commands_by_name.values():
        command.add_argument("--data", dest="data", choices=tuple(DATA_FLAG_TO_MODE), default=None)
    return parser


def _handlers(args: argparse.Namespace) -> dict[str, Callable[[], dict[str, Any]]]:
    """Map each JSON command to a zero-argument call."""
    return {
        "blocks": commands.blocks_command,
        "validate": lambda: commands.validate_command(args.strategy_json),
        "backtest": lambda: commands.backtest_command(args.strategy_id),
        "report": lambda: commands.report_command(args.strategy_id),
        "approve": lambda: commands.approve_command(args.strategy_id, args.code),
        "paper-status": commands.paper_status_command,
        "paper-update": commands.paper_update_command,
        "why": lambda: commands.why_command(args.ticker, args.day),
        "data-status": commands.data_status_command,
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run one JSON command and turn any problem into a plain-English failure."""
    try:
        return _handlers(args)[args.command]()
    except commands.CommandError as error:
        return reporting.failure([str(error)])
    except HoldoutAccessError as error:
        return reporting.failure([f"Holdout guard: {error}"])
    except Exception as error:
        return reporting.failure([f"The command failed: {type(error).__name__}: {error}"])


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ArgumentError as error:
        print(json.dumps(reporting.failure([str(error)])))
        return 1
    if args.command is None:
        parser.print_help()
        return 0
    if getattr(args, "data", None) is not None:
        before = data_mode()
        apply_data_override(args.data)
        if data_mode() != before:
            # The import-time warning used the old mode, so say the new one.
            eodhd_client.announce_mode()
    if args.command == "daily-summary":
        print(commands.daily_summary_text())
        return 0
    result = run_command(args)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
