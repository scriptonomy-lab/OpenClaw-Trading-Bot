"""Go-live check. Run once a real EODHD key is in .env:

    uv run python -m bot.verify_live

The steps run in order and any failure stops the run with a non-zero exit code:

1. Check the key and plan (GET /user). A bad key or a free plan stops here.
2. Call every endpoint the project uses once and diff each real response
   against the fake shapes, key for key. Writes shapes.md.
3. Estimate the API calls with the real universe size, compare with the plan's
   daily limit, then wait for explicit confirmation before the full download.
4. Run the integration tests (pytest -m integration) against live data.
5. Run the gap-down example backtest and recompute three trades by hand from
   the raw cached rows. Writes trade_checks.md.

The command refuses to run on fake data. ``--dry-run-fake`` runs every step
against the fake transport so the command itself can be tested. A dry run
proves nothing about EODHD and writes to outputs/simulated/.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import requests

from bot.config import (
    BACKTEST_END,
    DEFAULT_COST_BPS_PER_SIDE,
    PROJECT_ROOT,
    SIMULATED_LABEL,
    data_mode,
    load_secrets,
)
from bot.data import eodhd_client as client
from bot.data.fake_eodhd import (
    FAKE_PAID_TOKEN,
    UNVERIFIED_SHAPES,
    FakeEodhdTransport,
    fake_shape_examples,
)
from bot.data.holdout import assert_before_holdout
from bot.universe import universe_label, universe_mode

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DRY_RUN_TICKER_LIMIT = 10
CONFIRM_WORD = "DOWNLOAD"
TOLERANCE = 1e-6
NEW_YORK = ZoneInfo("America/New_York")
FAKE_ONLY_KEYS = ("eod_free_plan_warning",)  # needs a free key, so it cannot run live

# Responses whose outer keys are data (row numbers, symbols). Only the first entry is diffed.
KEYED_COLLECTIONS = frozenset({"sentiments", "fundamentals_components", "fundamentals_history"})

# Which UNVERIFIED_SHAPES entry is settled by which endpoint key.
UNVERIFIED_TO_KEY: tuple[tuple[str, str], ...] = (
    ("GET /fundamentals/GSPC.INDX?filter=Components", "fundamentals_components"),
    ("HistoricalTickerComponents", "fundamentals_history"),
    ("GET /calendar/earnings", "calendar_earnings"),
    ("GET /symbol-change-history", "symbol_change_history"),
    ("GET /exchange-symbol-list", "exchange_symbol_list_delisted"),
    ("GET /user", "user"),
)


class VerifyError(Exception):
    """A step failed. The message says what to fix."""


@dataclass
class Run:
    """Settings and results shared by the steps of one run."""

    dry_run: bool
    assume_yes: bool
    token: str
    stamp: str
    out_dir: Path
    input_fn: Callable[[str], str] = input
    user: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_root(self) -> Path:
        """Where the raw cached responses of this run live."""
        return client.FAKE_CACHE_DIR if self.dry_run else client.DEFAULT_CACHE_DIR

    @property
    def label(self) -> str:
        """Heading label for reports."""
        return SIMULATED_LABEL if self.dry_run else "LIVE DATA"


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py by path, since scripts is not a package."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"verify_live_{name}", path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ HTTP


def api_get(run: Run, path: str, params: dict[str, Any]) -> tuple[int, Any]:
    """One GET to EODHD (or the fake transport in a dry run). Returns (status, body).

    Dates in the holdout are refused here too. The token is never printed.
    """
    for name in ("from", "to"):
        if name in params:
            assert_before_holdout(date.fromisoformat(str(params[name])))
    query = {"api_token": run.token, "fmt": "json", **params}
    url = client.BASE_URL + path
    try:
        if run.dry_run:
            response: Any = FakeEodhdTransport().get(url, params=query, timeout=0)
        else:
            response = requests.get(url, params=query, timeout=client.REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise VerifyError(f"Network error on {path}: {type(exc).__name__}.") from None
    try:
        body = response.json()
    except ValueError:
        body = str(response.text)[:200]
    return int(response.status_code), body


# ------------------------------------------------------------------ step 1


def step_plan(run: Run) -> None:
    """Check the key and the plan with GET /user. Stop on 401 or on a free plan."""
    status, body = api_get(run, "/user", {})
    if status == 401:
        raise VerifyError("The EODHD key was rejected (HTTP 401). Check EODHD_API_KEY in .env.")
    if status != 200 or not isinstance(body, dict):
        raise VerifyError(f"GET /user returned HTTP {status}: {str(body)[:120]}")
    run.user = body
    plan = str(body.get("subscriptionType", "unknown"))
    print(f"  Plan: {plan}")
    print(f"  Daily limit: {body.get('dailyRateLimit')} (extra limit {body.get('extraLimit')})")
    print(f"  Calls used today: {body.get('apiRequests')} (on {body.get('apiRequestsDate')})")
    if plan.strip().lower() == "free":
        raise VerifyError(
            "WARNING: this is a FREE plan. It cannot run this project (no membership "
            "history, no earnings calendar, prices limited to one year). Upgrade the plan."
        )


# ------------------------------------------------------------------ step 2: shapes


@dataclass(frozen=True)
class ShapeDiff:
    """One difference between the fake shape and the real response."""

    path: str
    # missing_key, extra_key, type_change, empty_real, http_error, null_difference,
    # empty_reference
    kind: str
    detail: str

    @property
    def fatal(self) -> bool:
        """Null versus a value, or an empty fake reference, is a warning, not a mismatch.

        Both leave the endpoint unverified, because ``verified_lines`` needs no diffs at all.
        """
        return self.kind not in ("null_difference", "empty_reference")

    def line(self) -> str:
        """One readable line."""
        return f"{self.path}: {self.kind}: {self.detail}"


def type_name(value: Any) -> str:
    """Coarse JSON type: ints and floats are both 'number'."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    return type(value).__name__


def diff_shape(expected: Any, actual: Any, path: str = "$") -> list[ShapeDiff]:
    """Compare keys and value types, recursing into dicts and the first list record."""
    want, got = type_name(expected), type_name(actual)
    if want != got:
        kind = "null_difference" if "null" in (want, got) else "type_change"
        return [ShapeDiff(path, kind, f"expected {want}, got {got}")]
    if isinstance(expected, dict):
        return _diff_dicts(expected, actual, path)
    if isinstance(expected, list):
        if not expected:
            return [_empty_reference(path)]
        if not actual:
            return [ShapeDiff(path, "empty_real", "real list is empty, shape cannot be checked")]
        return diff_shape(expected[0], actual[0], f"{path}[0]")
    return []


def _empty_reference(path: str) -> ShapeDiff:
    """A fake example with nothing in it proves nothing, so the shape stays unverified (L7)."""
    return ShapeDiff(path, "empty_reference", "fake reference is empty, shape cannot be checked")


def _diff_dicts(expected: dict[str, Any], actual: dict[str, Any], path: str) -> list[ShapeDiff]:
    """Missing keys, extra keys, then the shared keys one by one."""
    diffs = [
        ShapeDiff(f"{path}.{k}", "missing_key", "in fake, not in real")
        for k in expected
        if k not in actual
    ]
    diffs += [
        ShapeDiff(f"{path}.{k}", "extra_key", "in real, not in fake")
        for k in actual
        if k not in expected
    ]
    for key in expected:
        if key in actual:
            diffs += diff_shape(expected[key], actual[key], f"{path}.{key}")
    return diffs


def unwrap_first(payload: Any) -> tuple[str, Any]:
    """('dict' or 'list', first entry) for a collection, or (type name, None) if empty."""
    if isinstance(payload, dict) and payload:
        return "dict", next(iter(payload.values()))
    if isinstance(payload, list) and payload:
        return "list", payload[0]
    return type_name(payload), None


def compare_endpoint(key: str, expected: Any, status: int, body: Any) -> list[ShapeDiff]:
    """All differences for one endpoint, including a non-200 status."""
    if status != 200:
        return [ShapeDiff("$", "http_error", f"HTTP {status}: {str(body)[:100]}")]
    if key not in KEYED_COLLECTIONS:
        return diff_shape(expected, body)
    want_kind, want_first = unwrap_first(expected)
    got_kind, got_first = unwrap_first(body)
    if want_first is None:
        return [_empty_reference("$")]
    if want_kind != got_kind:
        return [ShapeDiff("$", "type_change", f"expected {want_kind} of rows, got {got_kind}")]
    if got_first is None:
        return [ShapeDiff("$", "empty_real", "real collection is empty, shape cannot be checked")]
    return diff_shape(want_first, got_first, "$[first]")


@dataclass(frozen=True)
class EndpointSpec:
    """One real call to make: the fake example key, the path and the query."""

    key: str
    path: str
    params: dict[str, Any]


@dataclass(frozen=True)
class EndpointResult:
    """Outcome of one endpoint check."""

    spec: EndpointSpec
    status: int
    diffs: tuple[ShapeDiff, ...]

    @property
    def fatal(self) -> bool:
        """True when any difference is a real mismatch."""
        return any(d.fatal for d in self.diffs)


def endpoint_specs(fake: bool) -> list[EndpointSpec]:
    """One call per endpoint family. Dates stay before the holdout."""

    def sym(live: str, fake_code: str) -> str:
        return f"{fake_code if fake else live}.US"

    index = "/fundamentals/GSPC.INDX"
    return [
        EndpointSpec(
            "eod", f"/eod/{sym('AAPL', 'SIM001')}", {"from": "2020-08-25", "to": "2020-09-02"}
        ),
        EndpointSpec(
            "splits", f"/splits/{sym('AAPL', 'SIM003')}", {"from": "2015-01-01", "to": "2023-12-31"}
        ),
        EndpointSpec(
            "div", f"/div/{sym('AAPL', 'SIM002')}", {"from": "2022-01-01", "to": "2022-12-31"}
        ),
        EndpointSpec(
            "news",
            "/news",
            {"s": sym("AAPL", "SIM001"), "from": "2023-01-03", "to": "2023-01-10", "limit": 5},
        ),
        EndpointSpec(
            "sentiments",
            "/sentiments",
            {"s": sym("AAPL", "SIM001"), "from": "2023-01-03", "to": "2023-01-10"},
        ),
        EndpointSpec(
            "calendar_earnings",
            "/calendar/earnings",
            {"symbols": sym("AAPL", "SIM004"), "from": "2021-01-01", "to": "2021-12-31"},
        ),
        EndpointSpec("fundamentals_components", index, {"filter": "Components"}),
        EndpointSpec("fundamentals_history", index, {"filter": "HistoricalTickerComponents"}),
        EndpointSpec("exchange_symbol_list_delisted", "/exchange-symbol-list/US", {"delisted": 1}),
        EndpointSpec(
            "symbol_change_history",
            "/symbol-change-history",
            {"from": "2015-01-01", "to": "2023-12-31", "ex": "US"},
        ),
        EndpointSpec("user", "/user", {}),
    ]


def reference_examples() -> dict[str, Any]:
    """The fake shapes. The rename example is empty in its default window, so widen it."""
    examples = fake_shape_examples()
    if not examples.get("symbol_change_history"):
        wide = {"api_token": FAKE_PAID_TOKEN, "from": "2015-01-01", "to": "2023-12-31", "ex": "US"}
        url = client.BASE_URL + "/symbol-change-history"
        examples["symbol_change_history"] = FakeEodhdTransport().get(url, params=wide).json()
    return examples


def check_endpoints(run: Run) -> list[EndpointResult]:
    """Make one call per endpoint and diff it against the fake shape."""
    reference = reference_examples()
    results = []
    for spec in endpoint_specs(fake=run.dry_run):
        status, body = api_get(run, spec.path, spec.params)
        diffs = tuple(compare_endpoint(spec.key, reference[spec.key], status, body))
        results.append(EndpointResult(spec, status, diffs))
        verdict = "match" if not diffs else f"{len(diffs)} difference(s)"
        print(f"  {spec.key:<32} HTTP {status}  {verdict}")
    return results


def verified_lines(results: list[EndpointResult], today: date) -> list[str]:
    """One line per UNVERIFIED shape: verified live today, or still open."""
    clean = {r.spec.key for r in results if r.status == 200 and not r.diffs}
    lines = []
    for text in UNVERIFIED_SHAPES:
        key = next((k for needle, k in UNVERIFIED_TO_KEY if needle in text), None)
        if key in clean:
            lines.append(f"- verified live on {today.isoformat()}: {text}")
        else:
            lines.append(f"- STILL UNVERIFIED (manual check): {text}")
    return lines


def render_shapes_report(run: Run, results: list[EndpointResult]) -> str:
    """The shapes.md text."""
    today = datetime.now().date()
    lines = [
        f"# Endpoint shapes: {run.label}",
        "",
        f"Run {run.stamp}. Fake shape versus real response.",
        "",
    ]
    if run.dry_run:
        lines += ["DRY RUN: fake compared with fake. Nothing here is verified against EODHD.", ""]
    lines += ["| Endpoint | Path | HTTP | Result |", "|---|---|---|---|"]
    for r in results:
        verdict = "match" if not r.diffs else "MISMATCH" if r.fatal else "match with warnings"
        lines.append(f"| {r.spec.key} | `{r.spec.path}` | {r.status} | {verdict} |")
    for r in (r for r in results if r.diffs):
        lines += ["", f"## Differences: {r.spec.key}", ""]
        lines += [f"- {d.line()}" for d in r.diffs]
    lines += ["", "## UNVERIFIED shapes", ""]
    if run.dry_run:
        lines.append("Dry run, so none of these count as verified.")
    else:
        lines += verified_lines(results, today)
    lines += [
        "",
        "Not covered by this check: " + ", ".join(FAKE_ONLY_KEYS) + " (needs a free key),",
        "401 and 403 bodies, empty-result shapes, and meaning of fields (for example whether",
        "EndDate is inclusive). Null versus value differences are warnings, not failures.",
    ]
    return "\n".join(lines) + "\n"


def step_shapes(run: Run) -> None:
    """Diff every real response against the fakes and write shapes.md. Fail on any mismatch."""
    results = check_endpoints(run)
    run.out_dir.mkdir(parents=True, exist_ok=True)
    path = run.out_dir / "shapes.md"
    path.write_text(render_shapes_report(run, results))
    print(f"  Report: {path}")
    bad = [r for r in results if r.fatal]
    for r in bad:
        print(f"  MISMATCH {r.spec.key}:")
        for diff in r.diffs:
            print(f"    {diff.line()}")
    if bad:
        raise VerifyError(f"{len(bad)} endpoint shape(s) differ from the fakes. See {path}.")


# ------------------------------------------------------------------ step 3: estimate and download


def step_estimate(run: Run) -> None:
    """Print the call estimate for the chosen universe and compare it with the daily limit."""
    download = load_script("download_data")
    estimate = load_script("estimate_api_calls")
    mode = universe_mode()
    universe = download.load_universe(mode=mode)
    print(f"  {universe_label(mode)}")
    lines = estimate.realistic_lines(len(universe))
    for line in lines:
        print(f"  {line.name:<22}{line.high:>9,}  {line.note}")
    total = sum(x.high for x in lines)
    cap = int(download.CALL_CAP)
    limit = int(run.user.get("dailyRateLimit") or 0)
    print(f"\n  Plan daily limit: {limit:,}. Estimate: {total:,}. Run cap: {cap:,}.")
    if total > cap:
        raise VerifyError(f"The estimate is above the {cap:,} call cap. Nothing was downloaded.")
    if limit <= 0:
        print("  WARNING: the plan reports no daily limit, cannot compare.")
    elif total <= limit:
        print("  The estimate fits inside one day.")
    else:
        days = math.ceil(total / limit)
        print(f"  WARNING: the estimate needs about {days} day(s) at this limit.")
        print("  Cached responses cost nothing on a rerun, so the download can resume.")


def confirm_download(run: Run) -> None:
    """Stop unless the user types DOWNLOAD (or --yes was passed). Dry runs skip the gate."""
    if run.dry_run:
        print("  Dry run: confirmation skipped.")
        return
    if run.assume_yes:
        print("  --yes given: starting the download without a prompt.")
        return
    try:
        answer = run.input_fn(f"Type {CONFIRM_WORD} to start the full download: ")
    except EOFError:
        raise VerifyError("No terminal to confirm on. Pass --yes to confirm explicitly.") from None
    if answer.strip() != CONFIRM_WORD:
        raise VerifyError("Download not confirmed. Nothing was downloaded.")


def run_full_download(limit: int | None) -> int:
    """Run the logic of scripts/download_data.py. Returns the number of failed tickers."""
    module = load_script("download_data")
    try:
        return int(module.run_download(limit))
    except (module.EstimateTooHighError, module.BudgetExceededError) as error:
        raise VerifyError(f"Download stopped: {error}") from None


def step_download(run: Run) -> None:
    """Ask for confirmation, then download. A dry run downloads only 10 tickers."""
    confirm_download(run)
    failed = run_full_download(DRY_RUN_TICKER_LIMIT if run.dry_run else None)
    if failed:
        raise VerifyError(f"{failed} ticker(s) failed to download. See the messages above.")


# ------------------------------------------------------------------ step 4: integration tests


def parse_pytest_counts(output: str) -> dict[str, int]:
    """Counts such as {"passed": 1, "skipped": 12} from pytest's last summary line."""
    for line in reversed(output.strip().splitlines()):
        found = re.findall(r"(\d+) (passed|failed|skipped|deselected|error|errors)", line)
        if found:
            return {word: int(number) for number, word in found}
    return {}


def check_integration_counts(output: str) -> None:
    """Live mode needs at least one pass and no skipped test (L6)."""
    counts = parse_pytest_counts(output)
    if counts.get("passed", 0) < 1:
        raise VerifyError("No integration test passed. They may all have been skipped.")
    if counts.get("skipped", 0) > 0:
        raise VerifyError(
            f"{counts['skipped']} integration tests were skipped. Live mode needs none skipped."
        )


def step_integration(run: Run) -> None:
    """Run pytest -m integration in a child process and keep its output."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "integration",
        "-q",
        "-o",
        "addopts=",
        "tests/integration",
    ]
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    run.out_dir.mkdir(parents=True, exist_ok=True)
    (run.out_dir / "integration.txt").write_text(proc.stdout + proc.stderr)
    tail = (proc.stdout.strip().splitlines() or ["(no output)"])[-12:]
    print("\n".join(f"  {line}" for line in tail))
    if proc.returncode != 0:
        raise VerifyError(f"Integration tests failed (pytest exit {proc.returncode}).")
    if not run.dry_run:
        check_integration_counts(proc.stdout)


# ------------------------------------------------------------------ step 5: hand-checked trades


def read_json(path: Path) -> Any:
    """Load one cached raw response."""
    if not path.exists():
        raise VerifyError(f"Raw cache file missing: {path}")
    return json.loads(path.read_text())


def raw_eod(root: Path, symbol: str) -> dict[str, dict[str, Any]]:
    """Raw daily rows by date, read straight from the cached JSON."""
    rows = read_json(root / "eod" / symbol / "from=2014-01-01_to=2023-12-31.json")
    return {row["date"]: row for row in rows}


def raw_splits(root: Path, symbol: str) -> list[tuple[str, float]]:
    """(date, ratio) pairs from the cached splits file."""
    rows = read_json(root / "splits" / symbol / "to=2023-12-31.json")
    pairs = []
    for row in rows:
        new, old = str(row["split"]).split("/")
        pairs.append((row["date"], float(new) / float(old)))
    return sorted(pairs)


def raw_dividends(root: Path, symbol: str) -> list[tuple[str, float]]:
    """(ex-date, unadjustedValue) pairs from the cached dividends file."""
    rows = read_json(root / "div" / symbol / "from=2014-01-01_to=2023-12-31.json")
    return sorted((row["date"], float(row["unadjustedValue"])) for row in rows)


def raw_news(root: Path, symbol: str) -> list[dict[str, Any]]:
    """Every cached news row for the symbol (windows may be split across files)."""
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "news").glob(f"*s={symbol}_*.json")):
        rows += read_json(path)
    return rows


def count_news(rows: list[dict[str, Any]], symbol: str, start: datetime, end: datetime) -> int:
    """Articles tagged with the exact symbol in [start, end), each link once."""
    links = set()
    for row in rows:
        if symbol not in row.get("symbols", []):
            continue
        moment = datetime.fromisoformat(row["date"]).astimezone(NEW_YORK)
        if start <= moment < end:
            links.add(row["link"])
    return len(links)


def close_time(day: str) -> datetime:
    """16:00 New York on a date."""
    return datetime.fromisoformat(day).replace(hour=16, tzinfo=NEW_YORK)


def split_ratio_between(splits: list[tuple[str, float]], after: str, up_to: str) -> float:
    """Product of split ratios with after < date <= up_to."""
    ratio = 1.0
    for day, value in splits:
        if after < day <= up_to:
            ratio *= value
    return ratio


def hand_exit(
    bars: dict[str, dict[str, Any]],
    days: list[str],
    start: int,
    entry_price: float,
    rules: dict[str, Any],
    splits: list[tuple[str, float]],
) -> tuple[str, float]:
    """Exit date and raw price. Holding period: raw open ``holding_days`` sessions after the fill.

    Stop or target: the first bar from the fill day on whose low or high crosses the
    level (stop wins), filled at the level, or at the open if it gapped through the stop.
    """
    fill_day = days[start]
    if rules.get("holding_days") is not None and rules["reason"] == "holding_period":
        end = start + int(rules["holding_days"])
        # The ledger says nothing was touched. Check the bars before the exit open (L8).
        early = first_touch(bars, days[start:end], fill_day, entry_price, rules, splits)
        if early is not None:
            raise VerifyError(
                f"Ledger exit is the holding period, but the raw bars touched a stop or "
                f"target on {early[0]}."
            )
        return days[end], float(bars[days[end]]["open"])
    hit = first_touch(bars, days[start:], fill_day, entry_price, rules, splits)
    if hit is None:
        raise VerifyError("Hand check found no stop or target hit in the raw rows.")
    return hit


def first_touch(
    bars: dict[str, dict[str, Any]],
    scan_days: list[str],
    fill_day: str,
    entry_price: float,
    rules: dict[str, Any],
    splits: list[tuple[str, float]],
) -> tuple[str, float] | None:
    """The first day whose low or high crosses the stop or target, with its fill price."""
    for day in scan_days:
        bar = bars[day]
        adjust = split_ratio_between(splits, fill_day, day)
        stop = (
            entry_price * (1 - rules["stop_pct"] / 100) / adjust if rules.get("stop_pct") else None
        )
        target = (
            entry_price * (1 + rules["target_pct"] / 100) / adjust
            if rules.get("target_pct")
            else None
        )
        if stop is not None and bar["low"] <= stop:
            return day, min(float(bar["open"]), stop)
        if target is not None and bar["high"] >= target:
            return day, target
    return None


@dataclass(frozen=True)
class RawView:
    """Raw cached rows for one symbol plus the trading calendar (from SPY)."""

    symbol: str
    bars: dict[str, dict[str, Any]]
    days: list[str]
    splits: list[tuple[str, float]]
    dividends: list[tuple[str, float]]


def load_raw_view(root: Path, symbol: str) -> RawView:
    """Read the raw rows the hand check needs."""
    return RawView(
        symbol,
        raw_eod(root, symbol),
        sorted(raw_eod(root, "SPY.US")),
        raw_splits(root, symbol),
        raw_dividends(root, symbol),
    )


def hand_signal(view: RawView, root: Path, signal: str, lookback: int) -> tuple[float, int]:
    """Gap percent on the signal day and the article count in the news window."""
    i = view.days.index(signal)
    prev_day = view.days[i - 1]
    ratio = split_ratio_between(view.splits, prev_day, signal)
    dividend = sum(v for d, v in view.dividends if d == signal)
    previous_close = view.bars[prev_day]["close"] / ratio
    gap = (view.bars[signal]["open"] + dividend) / previous_close * 100 - 100
    window_start = close_time(view.days[i - lookback])
    articles = raw_news(root, view.symbol)
    return gap, count_news(articles, view.symbol, window_start, close_time(signal))


def hand_trade(
    root: Path, trade: dict[str, Any], equity: float, rules: dict[str, Any]
) -> dict[str, Any]:
    """Recompute one ledger row from raw rows with plain arithmetic. No engine code."""
    view = load_raw_view(root, trade["symbol"])
    signal = str(trade["signal_date"])
    lookback = int(rules["news_lookback"])
    gap, news = hand_signal(view, root, signal, lookback)
    start = view.days.index(signal) + 1
    fill_day = view.days[start]
    entry_price = float(view.bars[fill_day]["open"])
    notional = equity / rules["max_positions"]
    shares = notional / entry_price
    entry_cost = notional * rules["bps"] / 10_000
    exit_rules = {**rules, "reason": trade["exit_reason"]}
    exit_day, exit_price = hand_exit(
        view.bars, view.days, start, entry_price, exit_rules, view.splits
    )
    exit_shares = shares * split_ratio_between(view.splits, fill_day, exit_day)
    paid = sum(
        v * shares * split_ratio_between(view.splits, fill_day, d)
        for d, v in view.dividends
        if fill_day < d <= exit_day
    )
    exit_cost = exit_shares * exit_price * rules["bps"] / 10_000
    spent = entry_price * shares + entry_cost
    pnl = exit_price * exit_shares - exit_cost + paid - spent
    return {
        "entry_date": fill_day,
        "entry_price": entry_price,
        "entry_shares": shares,
        "entry_cost": entry_cost,
        "exit_date": exit_day,
        "exit_price": exit_price,
        "exit_shares": exit_shares,
        "exit_cost": exit_cost,
        "dividends": paid,
        "pnl": pnl,
        "return_pct": pnl / spent * 100,
        "gap_pct": gap,
        f"news_count_{lookback}": float(news),
    }


def compare_fields(
    ledger: dict[str, Any], hand: dict[str, Any]
) -> list[tuple[str, Any, Any, bool]]:
    """(field, ledger value, hand value, match) for every hand-computed field."""
    rows = []
    for name, mine in hand.items():
        theirs = ledger.get(name) if name in ledger else ledger["signal_values"].get(name)
        if isinstance(mine, str):
            same = str(theirs) == mine
        else:
            same = theirs is not None and math.isclose(
                float(theirs), mine, rel_tol=1e-9, abs_tol=TOLERANCE
            )
        rows.append((name, theirs, mine, same))
    return rows


def pick_trades(trades: list[Any], is_full_size: Callable[[Any], bool]) -> list[tuple[str, Any]]:
    """Three trades by docs/trade_checks.md rules: ordinary exit, stop or target, dividend or split.

    The middle trade of each group is used, so the choice is deterministic and not hand-picked.
    A missing group is filled with another ordinary trade on a different symbol.
    """
    ordered = sorted(trades, key=lambda t: (t.signal_date, t.symbol))
    usable = [
        t for t in ordered if t.exit_reason in ("holding_period", "stop_loss", "profit_target")
    ]
    usable = [t for t in usable if is_full_size(t)] or usable

    def middle(group: list[Any]) -> Any | None:
        return group[len(group) // 2] if group else None

    stop = [t for t in usable if t.exit_reason != "holding_period"]
    event = [t for t in usable if t.dividends > 0 or "split_while_held" in t.flags]
    plain = [t for t in usable if t.exit_reason == "holding_period" and t not in event]
    chosen = [
        ("ordinary exit", middle(plain)),
        ("stop or target", middle(stop)),
        ("dividend or split", middle(event)),
    ]
    picked = [(label, t) for label, t in chosen if t is not None]
    used = {t.symbol for _, t in picked}
    spare = [t for t in plain if t.symbol not in used]
    while len(picked) < 3 and spare:
        extra = spare.pop(len(spare) // 2)
        picked.append(("ordinary exit (filler: no trade in the missing group)", extra))
        spare = [t for t in spare if t.symbol != extra.symbol]
    if len(picked) < 3:
        raise VerifyError("The ledger has fewer than three usable trades to check.")
    return picked


def run_example_ledger(run: Run) -> tuple[Any, Any]:
    """Run the gap-down example backtest on cached data. Returns (strategy, result)."""
    from bot.backtest.costs import make_cost_model
    from bot.backtest.engine import run_backtest
    from bot.data.market import PRICE_START, load_market_data
    from bot.strategy.schema import backtest_windows, strategy_from_json

    strategy = strategy_from_json(load_script("run_example_backtest").EXAMPLE_JSON)
    market = load_market_data(PRICE_START, BACKTEST_END)
    window = backtest_windows(strategy)[0]
    result = run_backtest(strategy, market, window, make_cost_model(DEFAULT_COST_BPS_PER_SIDE))
    print(f"  Backtest {window.start} to {window.end}: {len(result.trades)} trades.")
    return strategy, result


def hand_rules(strategy: Any) -> dict[str, Any]:
    """The few strategy numbers the hand check needs, read from the definition."""
    lookback = next((c.lookback_sessions for c in strategy.entry if c.block == "no_news"), 1)
    return {
        "max_positions": strategy.sizing.max_positions,
        "holding_days": strategy.exit.holding_days,
        "stop_pct": strategy.exit.stop_loss_pct,
        "target_pct": strategy.exit.profit_target_pct,
        "bps": DEFAULT_COST_BPS_PER_SIDE,
        "news_lookback": lookback,
    }


def render_trade_report(run: Run, strategy_id: str, checks: list[dict[str, Any]]) -> str:
    """The trade_checks.md text."""
    lines = [f"# Trade checks: {run.label}", "", f"Run {run.stamp}. Strategy {strategy_id}.", ""]
    lines += [
        "Each value was recomputed from the raw cached EODHD rows with plain arithmetic.",
        f"Tolerance {TOLERANCE}. The only engine value used is the portfolio equity at the signal",
        "close, which sets the position size. Rules: docs/trade_checks.md.",
        "",
    ]
    for number, check in enumerate(checks, start=1):
        trade = check["trade"]
        lines += [
            f"## Check {number}: {trade['symbol']}, signal {trade['signal_date']}"
            f" ({check['label']})",
            "",
        ]
        lines += [f"Equity at the signal close (engine): {check['equity']:.6f}.", ""]
        lines += ["| Field | Ledger | Hand | |", "|---|---|---|---|"]
        for name, ledger_value, hand_value, same in check["rows"]:
            lines.append(
                f"| {name} | {ledger_value} | {hand_value} | {'match' if same else 'MISMATCH'} |"
            )
        lines.append("")
    lines.append(
        "Result: " + ("all fields match." if all_match(checks) else "MISMATCH, see tables.")
    )
    return "\n".join(lines) + "\n"


def all_match(checks: list[dict[str, Any]]) -> bool:
    """True when every field of every check matched."""
    return all(row[3] for check in checks for row in check["rows"])


def step_trades(run: Run) -> None:
    """Run the example backtest, hand-check three trades, write trade_checks.md."""
    strategy, result = run_example_ledger(run)
    rules = hand_rules(strategy)
    curve = result.equity_curve

    def equity_at(trade: Any) -> float:
        return float(curve.loc[curve.index == str(trade.signal_date), "equity"].iloc[0])

    def full_size(trade: Any) -> bool:
        target = equity_at(trade) / rules["max_positions"]
        return math.isclose(trade.entry_price * trade.entry_shares, target, rel_tol=1e-9)

    checks = []
    for label, trade in pick_trades(result.trades, full_size):
        row = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(trade).items()}
        hand = hand_trade(run.cache_root, row, equity_at(trade), rules)
        checks.append(
            {
                "label": label,
                "trade": row,
                "equity": equity_at(trade),
                "rows": compare_fields(row, hand),
            }
        )
    run.out_dir.mkdir(parents=True, exist_ok=True)
    path = run.out_dir / "trade_checks.md"
    path.write_text(render_trade_report(run, result.strategy_id, checks))
    print(f"  Report: {path}")
    if not all_match(checks):
        raise VerifyError(f"Hand check does not match the ledger. See {path}.")


# ------------------------------------------------------------------ command line


def steps() -> list[tuple[str, Callable[[Run], None]]]:
    """The steps in order. Looked up at call time so tests can swap one."""
    return [
        ("Key and plan", step_plan),
        ("Endpoint shapes", step_shapes),
        ("Call estimate", step_estimate),
        ("Full download", step_download),
        ("Integration tests", step_integration),
        ("Hand-checked trades", step_trades),
    ]


def build_parser() -> argparse.ArgumentParser:
    """Command line options."""
    parser = argparse.ArgumentParser(prog="python -m bot.verify_live", description="Go-live check.")
    parser.add_argument(
        "--yes", action="store_true", help="confirm the full download without a prompt"
    )
    parser.add_argument("--dry-run-fake", action="store_true", help="run every step on fake data")
    return parser


def make_run(
    args: argparse.Namespace, output_root: Path | None, input_fn: Callable[[str], str]
) -> Run:
    """Build the Run: token, output folder and timestamp."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = "simulated" if args.dry_run_fake else "live"
    root = output_root or PROJECT_ROOT / "outputs"
    token = FAKE_PAID_TOKEN if args.dry_run_fake else load_secrets().eodhd_api_key
    return Run(
        args.dry_run_fake, args.yes, token, stamp, root / folder / "verify" / stamp, input_fn
    )


def main(
    argv: list[str] | None = None,
    *,
    output_root: Path | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Run every step in order. Returns 0 on success, 1 on a failed step, 2 if refused."""
    args = build_parser().parse_args(argv)
    if args.dry_run_fake:
        os.environ["EODHD_MODE"] = "fake"
        print(
            f"*** {SIMULATED_LABEL}: dry run on the fake transport. Proves nothing about EODHD. ***"
        )
    elif data_mode() == "fake":
        print(
            "Refusing to run: the data mode is fake (EODHD_MODE=fake or no EODHD_API_KEY).",
            file=sys.stderr,
        )
        print(
            "Put a real key in .env, or use --dry-run-fake to test this command.", file=sys.stderr
        )
        return 2
    run = make_run(args, output_root, input_fn)
    all_steps = steps()
    for number, (name, step) in enumerate(all_steps, start=1):
        print(f"\nStep {number}/{len(all_steps)}: {name}")
        try:
            step(run)
        except Exception as exc:
            print(
                f"\nFAILED at step {number} ({name}): {type(exc).__name__}: {exc}", file=sys.stderr
            )
            return 1
    print(f"\nAll {len(all_steps)} steps passed. Reports: {run.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
