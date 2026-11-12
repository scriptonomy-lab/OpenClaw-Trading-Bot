"""Render the OpenClaw templates for this machine and check the exec allowlist.

Reads `.env` and the repo path, fills the placeholders in `openclaw/openclaw.json5`,
`openclaw/exec-approvals.json` and `openclaw/cron.md`, and writes the results to
`openclaw/rendered/` (git-ignored). Before writing, it checks that the exec
allowlist regex accepts every valid command and rejects every escape attempt.

Usage:
    uv run python scripts/render_openclaw_config.py
    uv run python scripts/render_openclaw_config.py --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENCLAW_DIR = REPO_ROOT / "openclaw"
RENDERED_DIR = OPENCLAW_DIR / "rendered"
CONFIG_TEMPLATE = OPENCLAW_DIR / "openclaw.json5"
APPROVALS_TEMPLATE = OPENCLAW_DIR / "exec-approvals.json"
CRON_TEMPLATE = OPENCLAW_DIR / "cron.md"

# Any leftover <UPPER_CASE> token means a placeholder was not filled.
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")

# The binary glob must be a literal path. These are glob metacharacters.
GLOB_CHARS = set("*?[]{}")

# strategy_id format from bot/strategy/schema.py: "strat_" + 16 hex characters.
SAMPLE_ID = "strat_3f9a0c1d2e4b5a69"

# The gap-down example from docs/design/strategy_schema.md, on one line.
GAP_DOWN_JSON = (
    '{"name":"Gap down, no news","description_plain_english":"Buy S&P 500 stocks '
    "whose open is 3% or more below the previous close, with no news since the "
    'previous close. Sell after 3 trading days.","version":1,"entry":[{"block":"gap",'
    '"direction":"down","min_gap_pct":3},{"block":"no_news","lookback_sessions":1}],'
    '"exit":{"holding_days":3},"sizing":{"max_positions":10}}'
)
# The same-open gap-down example with rank_by (D28, D29), as the skill shows it.
SAME_OPEN_JSON = (
    '{"name":"Gap down at the open, no news","description_plain_english":"Buy S&P 500 '
    "stocks at the open on the day they open 3% or more below the previous close, with "
    "no news from the previous close to the open, biggest gaps first. Sell after 3 "
    'trading days.","version":1,"entry":[{"block":"gap","direction":"down",'
    '"min_gap_pct":3},{"block":"no_news","lookback_sessions":1}],"exit":{"holding_days":3},'
    '"sizing":{"max_positions":10},"entry_timing":"same_open","rank_by":"largest_gap"}'
)
# Spaced JSON with negative numbers, as json.dumps would print it.
NEGATIVE_JSON = (
    '{"entry": [{"block": "n_day_return", "days": 20, "comparison": "at_most", '
    '"return_pct": -10}, {"block": "news_sentiment", "comparison": "at_most", '
    '"polarity": -0.5}], "exit": {"holding_days": 5}}'
)

# Argument strings (argv after the binary, joined by single spaces) that the
# agent must be able to run. Each matches a row in docs/design/cli.md.
VALID_ARGS: list[str] = [
    "blocks",
    "paper-status",
    "data-status",
    "validate {}",
    f"validate {GAP_DOWN_JSON}",
    f"validate {NEGATIVE_JSON}",
    f"validate {SAME_OPEN_JSON}",
    'validate {"entry_timing": "same_open", "rank_by": "largest_gap"}',
    'validate {"exit":{"stop_loss_pct":7.5},"entry":[{"block":"rsi","level":30}]}',
    f"backtest {SAMPLE_ID}",
    f"report {SAMPLE_ID}",
    f"approve {SAMPLE_ID}",
    f"approve {SAMPLE_ID} --code 482913",
    "why AAPL",
    "why BRK-B",
    "why BF.B --date 2023-06-14",
    "why MSFT --date 2019-01-02",
    # The optional data-mode flag (D34): once, at the end, simulated or live only.
    "blocks --data simulated",
    "paper-status --data live",
    "data-status --data simulated",
    f"validate {GAP_DOWN_JSON} --data simulated",
    f"validate {NEGATIVE_JSON} --data live",
    f"backtest {SAMPLE_ID} --data simulated",
    f"backtest {SAMPLE_ID} --data live",
    f"report {SAMPLE_ID} --data simulated",
    f"approve {SAMPLE_ID} --data simulated",
    f"approve {SAMPLE_ID} --code 482913 --data simulated",
    f"approve {SAMPLE_ID} --code 482913 --data live",
    "why AAPL --data live",
    "why BF.B --date 2023-06-14 --data simulated",
]

# Argument strings that must never match. Grouped by the attack they model.
ESCAPE_ATTEMPTS: list[str] = [
    # Shell chains and pipes.
    "blocks; rm -rf ~",
    "blocks && curl https://example.invalid",
    "blocks || id",
    "blocks | sh",
    "blocks & id",
    f"backtest {SAMPLE_ID}; id",
    "why AAPL --date 2023-06-14; id",
    # Command substitution and variables.
    "backtest $(id)",
    "backtest `id`",
    'validate {"a":"$(id)"}',
    'validate {"a":"`id`"}',
    'validate {"a":"${HOME}"}',
    'validate {"a":"$HOME"}',
    # Redirection.
    'validate {"a":1} > /tmp/x',
    'validate {"a":1}<x}',
    # Newlines and other line terminators.
    "blocks\nid",
    "blocks\n",
    "blocks\r\nid",
    "blocks\rid",
    "blocks id",
    "blocks id",
    'validate {"a":1}\nid',
    'validate {"a":"x\ny"}',
    f"backtest {SAMPLE_ID}\n",
    "blocks\x00",
    # Quotes and backslash escapes inside the JSON argument.
    'validate {"a":"\'"}',
    'validate {"a":"\\u0027"}',
    'validate {"a":"\\n"}',
    # Subcommands the agent may not run, and lookalikes.
    "paper-update",
    "daily-summary",
    "BLOCKS",
    "blocks ",
    " blocks",
    "blocks extra",
    "",
    "help",
    "--help",
    "-h",
    "--version",
    "blocks --help",
    "validate {} --help",
    "validate {} -h",
    'validate {"a":1} --help {}',
    'validate {"a":1} -h {}',
    'validate {"a":1} --date {}',
    'validate {"a": -x}',
    'validate {"a": --1}',
    "backtest --help",
    "why --help",
    "why AAPL --help",
    # Missing or extra arguments.
    "validate",
    "validate ",
    "backtest",
    "approve",
    "why",
    f"approve {SAMPLE_ID} {SAMPLE_ID}",
    f"approve {SAMPLE_ID} --code 12345",
    f"approve {SAMPLE_ID} --code 1234567",
    f"approve {SAMPLE_ID} --code abcdef",
    f"approve {SAMPLE_ID} --code 482913 --code 482913",
    f"approve --code 482913 {SAMPLE_ID}",
    f"backtest  {SAMPLE_ID}",
    f"backtest\t{SAMPLE_ID}",
    # Malformed JSON shapes and size.
    "validate {",
    "validate }",
    "validate []",
    "validate {" + "a" * 4000 + "}",
    # Characters with shell meaning, or that need escaping, inside JSON text.
    'validate {"name":"Gap (down)"}',
    'validate {"name":"Above $50"}',
    'validate {"name":"a;b"}',
    'validate {"name":"a|b"}',
    'validate {"name":"a/b"}',
    'validate {"name":"a*b"}',
    'validate {"name":"a#b"}',
    'validate {"name":"a!b"}',
    'validate {"name":"a=b"}',
    'validate {"name":"café"}',
    # Wrong id, ticker and date shapes.
    "backtest abc",
    "backtest 3f9a0c1d2e4b5a69",
    f"backtest {SAMPLE_ID.upper()}",
    f"backtest {SAMPLE_ID}0",
    "backtest strat_3f9a0c1d2e4b5a6",
    "why aapl",
    "why TOOLONG",
    "why ＡAPL",
    "why AAPL --date 2023-6-1",
    "why AAPL --date 2023-06-14T00:00",
    "why AAPL 2023-06-14",
    # Path traversal and file paths.
    "backtest ../../etc/passwd",
    "report ./abcdef123456",
    "why ../AAPL",
    "why /etc/passwd",
    "why AAPL --date ../../x",
    "validate @/etc/passwd",
    # The data-mode flag: other values, repeats, other positions and injection.
    f"backtest {SAMPLE_ID} --data fake",
    f"backtest {SAMPLE_ID} --data SIMULATED",
    f"backtest {SAMPLE_ID} --data Live",
    f"backtest {SAMPLE_ID} --data=simulated",
    f"backtest {SAMPLE_ID} --data",
    f"backtest {SAMPLE_ID} --data ",
    f"backtest {SAMPLE_ID}  --data live",
    f"backtest {SAMPLE_ID} --data simulated --data live",
    f"backtest {SAMPLE_ID} --data live --data live",
    f"backtest {SAMPLE_ID} --data simulatedx",
    f"backtest {SAMPLE_ID} --data live; id",
    f"backtest {SAMPLE_ID} --data live && id",
    f"backtest {SAMPLE_ID} --data $(id)",
    f"backtest {SAMPLE_ID} --data live\n",
    f"backtest {SAMPLE_ID} --data live\nid",
    f"backtest --data live {SAMPLE_ID}",
    "--data live blocks",
    "--data simulated",
    f"approve {SAMPLE_ID} --data simulated --code 482913",
    "why AAPL --data live --date 2023-06-14",
    'validate {"a":1} --data simulated {}',
    'validate {"a":1}--data simulated',
    "paper-update --data simulated",
    "daily-summary --data live",
    # Arguments meant for other binaries (interpreter flags, pip).
    "-c import os",
    "-m pip install requests",
    "-e require('child_process')",
]


def json_string_body(value: str) -> str:
    """Escape a value so it can sit inside an existing JSON or JSON5 string."""
    return json.dumps(value)[1:-1]


def read_settings(repo_root: Path) -> dict[str, str]:
    """Read settings from `.env`, letting real environment variables win."""
    env_file = repo_root / ".env"
    values = {k: v or "" for k, v in dotenv_values(env_file).items()} if env_file.exists() else {}
    for key in ("OPENCLAW_MODEL", "TELEGRAM_ALLOWED_USER_IDS"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def parse_user_ids(raw: str) -> list[str]:
    """Split the comma-separated Telegram user IDs and check each is numeric."""
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        raise SystemExit("TELEGRAM_ALLOWED_USER_IDS is empty. OpenClaw rejects an empty allowlist.")
    for user_id in ids:
        if not user_id.isdigit():
            raise SystemExit(f"Telegram user ID {user_id!r} is not numeric.")
    return ids


def check_model(model: str) -> str:
    """Require an exact OpenAI provider/model id, as D8 pins the provider."""
    if not re.fullmatch(r"openai/[A-Za-z0-9._:-]+", model):
        raise SystemExit(f"OPENCLAW_MODEL must look like openai/<model>, got {model!r}.")
    return model


def fill(template: str, replacements: dict[str, str]) -> str:
    """Replace each placeholder and fail if any placeholder is left."""
    text = template
    for token, value in replacements.items():
        text = text.replace(token, value)
    leftover = sorted(set(PLACEHOLDER.findall(text)))
    if leftover:
        raise SystemExit(f"Unfilled placeholders: {', '.join(leftover)}")
    return text


def load_arg_pattern(approvals_text: str) -> str:
    """Return the single argPattern from an exec approvals document."""
    document = json.loads(approvals_text)
    entries = document["agents"]["main"]["allowlist"]
    if len(entries) != 1:
        raise SystemExit("The allowlist must contain exactly one entry.")
    pattern: str = entries[0]["argPattern"]
    return pattern


def matches_like_openclaw(pattern: str, args: str) -> bool:
    """Match the way an anchored ECMAScript regex would.

    Python's `$` also matches before a final newline, but JavaScript's does
    not. `fullmatch` closes that gap, so a trailing newline is rejected here
    just as OpenClaw rejects it.
    """
    return re.fullmatch(pattern, args) is not None


def check_cases(pattern: str) -> list[str]:
    """Return a list of problems with the pattern against the fixed cases."""
    problems = [
        f"valid case rejected: {c!r}" for c in VALID_ARGS if not matches_like_openclaw(pattern, c)
    ]
    problems += [
        f"escape accepted: {c!r}" for c in ESCAPE_ATTEMPTS if matches_like_openclaw(pattern, c)
    ]
    return problems


NODE_SCRIPT = """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const plain = new RegExp(input.pattern);
const unicode = new RegExp(input.pattern, 'u');
const out = input.cases.map((s) => [plain.test(s), unicode.test(s)]);
process.stdout.write(JSON.stringify(out));
"""


def node_results(pattern: str, cases: list[str]) -> list[list[bool]] | None:
    """Run the cases through Node's own RegExp, or return None if Node is absent."""
    node = shutil.which("node")
    if node is None:
        return None
    payload = json.dumps({"pattern": pattern, "cases": cases})
    completed = subprocess.run(
        [node, "-e", NODE_SCRIPT], input=payload, capture_output=True, text=True, check=True
    )
    results: list[list[bool]] = json.loads(completed.stdout)
    return results


def check_with_node(pattern: str) -> list[str]:
    """Check that ECMAScript agrees with the expected result for every case."""
    cases = VALID_ARGS + ESCAPE_ATTEMPTS
    results = node_results(pattern, cases)
    if results is None:
        return []
    expected = [True] * len(VALID_ARGS) + [False] * len(ESCAPE_ATTEMPTS)
    return [
        f"node disagrees on {case!r}: plain={plain}, unicode={uni}, expected={want}"
        for case, (plain, uni), want in zip(cases, results, expected, strict=True)
        if plain != want or uni != want
    ]


def binary_path_problems(approvals_text: str) -> list[str]:
    """The pinned binary must be an absolute literal path, not a glob."""
    entry = json.loads(approvals_text)["agents"]["main"]["allowlist"][0]
    path: str = entry["pattern"]
    problems = []
    if not path.startswith("/"):
        problems.append(f"binary pattern is not absolute: {path!r}")
    if GLOB_CHARS & set(path):
        problems.append(f"binary pattern contains glob characters: {path!r}")
    if not path.endswith("/.venv/bin/openclaw-bot"):
        problems.append(f"binary pattern does not point at the venv CLI: {path!r}")
    return problems


def build_replacements(repo_root: Path, settings: dict[str, str]) -> dict[str, str]:
    """Map each placeholder token to its rendered value."""
    user_ids = parse_user_ids(settings.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    model = check_model(settings.get("OPENCLAW_MODEL", ""))
    repo = json_string_body(str(repo_root))
    return {
        # The quoted form is used inside allowFrom so several IDs render as a list.
        '"<TELEGRAM_USER_ID_LIST>"': ", ".join(json.dumps(i) for i in user_ids),
        "<TELEGRAM_USER_ID>": user_ids[0],
        "<ABSOLUTE_PATH_TO_REPO>": repo,
        "<OPENCLAW_MODEL>": model,
    }


def render_all(repo_root: Path) -> list[Path]:
    """Render the three templates into openclaw/rendered/ and return the paths."""
    replacements = build_replacements(repo_root, read_settings(repo_root))
    RENDERED_DIR.mkdir(exist_ok=True)
    written = []
    for template in (CONFIG_TEMPLATE, APPROVALS_TEMPLATE, CRON_TEMPLATE):
        text = fill(template.read_text(encoding="utf-8"), replacements)
        if template == APPROVALS_TEMPLATE:
            problems = binary_path_problems(text)
            if problems:
                raise SystemExit("\n".join(problems))
        target = RENDERED_DIR / template.name
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    """Check the allowlist regex, then render the templates unless --check-only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true", help="only check the regex")
    args = parser.parse_args(argv)

    pattern = load_arg_pattern(APPROVALS_TEMPLATE.read_text(encoding="utf-8"))
    problems = check_cases(pattern) + check_with_node(pattern)
    if problems:
        print("Exec allowlist check FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"Exec allowlist check passed: {len(VALID_ARGS)} valid, {len(ESCAPE_ATTEMPTS)} rejected.")
    if args.check_only:
        return 0
    for path in render_all(REPO_ROOT):
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
