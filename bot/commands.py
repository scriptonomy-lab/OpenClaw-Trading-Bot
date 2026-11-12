"""The logic behind each CLI command. Each function returns the JSON dict to print.

bot/cli.py only parses arguments and prints. Trading logic is not here: this
file runs the backtest engine, stores the results and shapes them for display.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import traceback
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bot import reporting, storage, telegram_send
from bot.backtest.charts import plot_cost_sensitivity, plot_drawdown, plot_equity_vs_benchmarks
from bot.backtest.costs import make_cost_model
from bot.backtest.engine import (
    INITIAL_CAPITAL,
    BacktestResult,
    PortfolioState,
    run_backtest,
    run_cost_sensitivity,
    state_from_json,
)
from bot.backtest.metrics import benchmark_curves, compute_metrics
from bot.config import (
    BACKTEST_END,
    DEFAULT_COST_BPS_PER_SIDE,
    HOLDOUT_START,
    PROJECT_ROOT,
    SIMULATED_LABEL,
    data_mode,
    output_dir,
    secret,
)
from bot.data import eodhd_client as client
from bot.data.holdout import HoldoutAccessError, holdout_session
from bot.data.market import PRICE_START, LoadReport, load_market_data, load_market_report
from bot.paper.trader import paper_today, paper_update, start_paper, trade_row
from bot.strategy.blocks import MarketData
from bot.strategy.schema import (
    BacktestWindow,
    StrategyDefinition,
    backtest_windows,
    describe_rules,
    explain_validation_error,
    strategy_from_json,
    strategy_id,
    strategy_json_schema,
    strategy_to_json,
)

STRATEGY_ID_PATTERN = re.compile(r"^strat_[0-9a-f]{16}$")
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
SCHEMA_DOC = PROJECT_ROOT / "docs" / "design" / "strategy_schema.md"
NO_REPLY = "NO_REPLY"


class CommandError(Exception):
    """A problem the user can act on. The message is plain English."""


def chat_id() -> str | None:
    """The conversation id from OPENCLAW_CHAT_ID, or None. The agent cannot set it."""
    return os.getenv("OPENCLAW_CHAT_ID", "").strip() or None


def load_strategy(strategy_id_: str) -> tuple[str, StrategyDefinition]:
    """The stored JSON and strategy for an id. Raises CommandError if it was never validated."""
    if not STRATEGY_ID_PATTERN.match(strategy_id_):
        raise CommandError("That is not a strategy id. It looks like strat_ plus 16 hex digits.")
    stored = storage.get_strategy_json(strategy_id_)
    if stored is None:
        raise CommandError("That strategy id has not been validated. Run validate first.")
    return stored, strategy_from_json(stored)


# --------------------------------------------------------------------- blocks


def blocks_command() -> dict[str, Any]:
    """Supported blocks, parameters and bounds, plus the closest-supported answers."""
    schema = strategy_json_schema()
    summaries = [_block_summary(spec) for spec in schema["$defs"].values()]
    out = reporting.base(True)
    out["blocks"] = [item for item in summaries if item is not None]
    out["schema"] = schema
    out["closest_supported"] = _closest_supported()
    return out


def _block_summary(spec: dict[str, Any]) -> dict[str, Any] | None:
    """One entry block as name, description and parameters. None for other definitions."""
    properties = spec.get("properties", {})
    if "block" not in properties:
        return None
    return {
        "block": properties["block"].get("const"),
        "description": spec.get("description", ""),
        "parameters": {key: value for key, value in properties.items() if key != "block"},
        "required": [key for key in spec.get("required", []) if key != "block"],
    }


def _closest_supported() -> list[dict[str, str]]:
    """Rows of the closest-supported table in strategy_schema.md as request and response."""
    if not SCHEMA_DOC.exists():
        return []
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    section = text.split("## Closest supported version", 1)[1].split("\n## ", 1)[0]
    rows: list[dict[str, str]] = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith(("| Request", "|---")):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        if len(cells) >= 2:
            rows.append({"request": cells[0], "response": cells[1]})
    return rows


# ------------------------------------------------------------------- validate


def validate_command(text: str) -> dict[str, Any]:
    """Validate a strategy, store it as a draft and repeat the rules back."""
    try:
        strategy = strategy_from_json(text)
    except ValidationError as error:
        return reporting.failure(explain_validation_error(error))
    sid = strategy_id(strategy)
    attempt = storage.save_strategy(sid, strategy_to_json(strategy), chat_id())
    out = reporting.base(True)
    out["strategy_id"] = sid
    out["version"] = attempt
    out["rules_text"] = describe_rules(strategy)
    out["test_windows"] = [
        {
            "label": w.label,
            "start": w.start.isoformat(),
            "end": w.end.isoformat(),
            "coverage_warning": w.coverage_warning,
        }
        for w in backtest_windows(strategy)
    ]
    return out


# ------------------------------------------------------------------- backtest


def backtest_command(strategy_id_: str) -> dict[str, Any]:
    """Run every window at three cost levels, save everything and send the chart."""
    stored, strategy = load_strategy(strategy_id_)
    market, load_report = load_market_report(PRICE_START, BACKTEST_END)
    save_load_report(load_report)
    tried = storage.versions_tried(chat_id(), including=strategy_id_)
    runs: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for window in backtest_windows(strategy):
        window_runs, window_trades = _run_window(strategy_id_, strategy, market, window, tried)
        runs.extend(window_runs)
        trades.extend(window_trades)
    storage.replace_backtest(strategy_id_, "backtest", runs, trades, chat_id())
    out = reporting.report_payload(stored, storage.get_runs(strategy_id_))
    out["strategy_id"] = strategy_id_
    out["data_quality"] = load_report.to_dict()
    out["telegram"] = _send_chart(strategy.name, out, runs[0]["chart_paths"])
    return out


def save_load_report(report: LoadReport) -> None:
    """Keep the latest load report so data-status can show the skipped symbols."""
    path = output_dir() / "load_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict()))


def _run_window(
    sid: str, strategy: StrategyDefinition, market: MarketData, window: BacktestWindow, tried: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backtest one window at 0, 5 and 20 bps. Returns run rows and the 5 bps ledger."""
    results = run_cost_sensitivity(strategy, market, window)
    default = results[DEFAULT_COST_BPS_PER_SIDE]
    charts = _save_charts(sid, default, results, market, tried)
    runs = [
        {
            "window_label": window.label,
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "cost_bps": bps,
            "metrics": reporting.run_metrics(compute_metrics(result, market, tried)),
            "chart_paths": charts,
        }
        for bps, result in sorted(results.items())
    ]
    ledger = [{**trade_row(t), "window_label": window.label} for t in default.trades]
    return runs, ledger


def _save_charts(
    sid: str,
    default: BacktestResult,
    results: dict[float, BacktestResult],
    market: MarketData,
    tried: int,
) -> list[str]:
    """Save the equity, drawdown and cost charts. Returns their absolute paths."""
    metrics = compute_metrics(default, market, tried)
    curves = benchmark_curves(default, market)
    folder = output_dir() / "charts"
    stem = f"{sid}_{default.window.label}"
    paths = [
        plot_equity_vs_benchmarks(curves, metrics, folder / f"{stem}_equity.png"),
        plot_drawdown(curves, metrics, folder / f"{stem}_drawdown.png"),
        plot_cost_sensitivity(results, folder / f"{stem}_cost_sensitivity.png"),
    ]
    return [str(path) for path in paths]


def _send_chart(name: str, report: dict[str, Any], charts: list[str]) -> dict[str, object]:
    """Send the primary equity chart to the allowed Telegram users (D21)."""
    caption = reporting.telegram_caption(
        name, report["windows"][0], report["versions_tried"], report["universe_mode"]
    )
    return telegram_send.send_photo(Path(charts[0]), caption).to_dict()


def report_command(strategy_id_: str) -> dict[str, Any]:
    """The saved result for a strategy, without rerunning it."""
    stored, _ = load_strategy(strategy_id_)
    runs = storage.get_runs(strategy_id_)
    if not runs:
        raise CommandError("No backtest is saved for that strategy. Run backtest first.")
    out = reporting.report_payload(stored, runs)
    out["strategy_id"] = strategy_id_
    holdout = storage.get_runs(strategy_id_, "holdout")
    if holdout:
        out["holdout"] = {
            **holdout[0]["metrics"],
            "holdout_runs_so_far": storage.count_used_approvals(),
        }
    out["holdout_runs_so_far"] = storage.count_used_approvals()
    return out


# -------------------------------------------------------------------- approve


CODE_PATTERN = re.compile(r"^[0-9]{6}$")
CODE_MINUTES = 10
MAX_WRONG_CODES = 3
TEST_CODE_FILE_ENV = "OPENCLAW_BOT_TEST_APPROVAL_CODE_FILE"
CODE_SECRET_NAME = "APPROVAL_CODE_SECRET"
TELEGRAM_TEXT_LIMIT = 4096
HOLDOUT_LOG = "holdout_errors.log"
SUMMARY_META_PREFIX = "last_summary:"


def now() -> datetime:
    """The current UTC time. Tests replace this to check expiry."""
    return datetime.now(UTC)


def approval_secret() -> str:
    """The HMAC key for code hashes (L1): APPROVAL_CODE_SECRET from .env.

    Live mode refuses to work without it. Fake mode makes a random key on first
    use and keeps it in the fake database, so tests need no setup.
    """
    value = secret(CODE_SECRET_NAME)
    if value:
        return value
    if data_mode() == "live":
        raise CommandError(
            f"{CODE_SECRET_NAME} is not set in .env, so no approval code can be made in live mode."
        )
    return storage.set_meta_if_absent("approval_code_secret", secrets.token_hex(32))


def hash_code(strategy_id_: str, code: str) -> str:
    """HMAC-SHA256 of the strategy id and code, keyed by the secret. Only this hash is stored."""
    key = approval_secret().encode()
    return hmac.new(key, f"{strategy_id_}|{code}".encode(), hashlib.sha256).hexdigest()


def approve_command(strategy_id_: str, code: str | None = None) -> dict[str, Any]:
    """Without a code: send a one-time code to the user. With the code: run the approval (D20).

    An approval row that was recorded but never started its holdout (a crash
    in between) is resumed: the user asks for a new code and approves again.
    The holdout still runs once, because the run is claimed with one
    ``UPDATE ... WHERE used = 0``.
    """
    _, strategy = load_strategy(strategy_id_)
    if not storage.get_runs(strategy_id_):
        raise CommandError("Run a backtest before approving.")
    stored = storage.find_stored_approval(strategy_id_)
    if stored is not None and stored.used:
        raise CommandError("This strategy was already approved. The holdout runs only once.")
    if code is None:
        return _send_code(strategy_id_, strategy)
    _check_code(strategy_id_, code)
    return _approve_with_valid_code(strategy_id_, strategy)


def approval_message(strategy_id_: str, name: str, rules: str, code: str) -> str:
    """The Telegram text: strategy id first, the full rules, then the code (M5).

    Only the rules are cut, and only if the whole text would pass Telegram's
    4096 characters (the fake-mode label is counted too).
    """
    head = f"{strategy_id_}\n{name}\n{reporting.universe_text()}\n\n"
    tail = (
        f"\n\nTo approve this strategy and run the one-time holdout test, reply with this "
        f"code: {code}. It expires in {CODE_MINUTES} minutes."
    )
    room = TELEGRAM_TEXT_LIMIT - len(f"{SIMULATED_LABEL}: ") - len(head) - len(tail)
    shown = rules if len(rules) <= room else rules[: room - 3] + "..."
    return head + shown + tail


def _send_code(strategy_id_: str, strategy: StrategyDefinition) -> dict[str, Any]:
    """Create a one-time code, store its hash and send it straight to the user. Never print it."""
    test_file = os.getenv(TEST_CODE_FILE_ENV, "").strip()
    if test_file and data_mode() == "live":
        raise CommandError("The test approval code file is refused in live mode.")
    row = storage.get_approval_code(strategy_id_)
    if row is not None and row["failed_attempts"] >= MAX_WRONG_CODES:
        raise CommandError("Approval is locked after 3 wrong codes for this strategy.")
    if _code_still_active(row):
        raise CommandError(
            f"A code was already sent. Use it, or ask again after it expires ({CODE_MINUTES} "
            "minutes after it was sent)."
        )
    code = f"{secrets.randbelow(10**6):06d}"
    code_hash = hash_code(strategy_id_, code)
    expires = now() + timedelta(minutes=CODE_MINUTES)
    storage.save_approval_code(strategy_id_, code_hash, expires.isoformat())
    text = approval_message(strategy_id_, strategy.name, describe_rules(strategy), code)
    sent = telegram_send.send_text(text)
    if sent.sent == 0 and not (test_file and data_mode() == "fake"):
        # Keep the row so earlier wrong attempts still count (M2). Only the hash goes.
        storage.void_approval_code(strategy_id_)
        raise CommandError("Approval needs Telegram to send a one-time code, and it is not set up.")
    if test_file and data_mode() == "fake":
        Path(test_file).write_text(code)
    out = reporting.base(True)
    out.update(status="code_sent", strategy_id=strategy_id_, expires_in_minutes=CODE_MINUTES)
    out["next_step"] = "Ask the user for the code they received on Telegram."
    _add_simulated_approval_note(out)
    return out


def _add_simulated_approval_note(out: dict[str, Any]) -> None:
    """In fake mode, say that this approval is simulated and kept apart from live (D34)."""
    if data_mode() == "fake":
        out["simulated_approval"] = True
        out["approval_note"] = reporting.SIMULATED_APPROVAL_NOTE


def _code_still_active(row: dict[str, Any] | None) -> bool:
    """True if an unused, unexpired code was sent (L2: one code per 10 minutes)."""
    if row is None or row["used"] or not row["code_hash"]:
        return False
    return now() <= datetime.fromisoformat(row["expires_at"])


def _check_code(strategy_id_: str, code: str) -> None:
    """Check format, lockout, expiry, single use and the hash in one transaction (M1)."""
    given = hash_code(strategy_id_, code) if CODE_PATTERN.match(code) else ""
    result = storage.check_approval_code(strategy_id_, given, now(), MAX_WRONG_CODES)
    if result.status == "ok":
        return
    if result.status == "locked":
        raise CommandError("Approval is locked after 3 wrong codes for this strategy.")
    if result.status == "none":
        raise CommandError("No unused approval code. Run approve without a code to get one.")
    if result.status == "expired":
        raise CommandError("That code has expired. Run approve without a code to get a new one.")
    if result.tries_left <= 0:
        raise CommandError("Wrong code. Approval is now locked for this strategy.")
    raise CommandError(f"Wrong code. {result.tries_left} tries left.")


def _log_holdout_failure(strategy_id_: str, error: BaseException) -> None:
    """Write the full error to a log file, never into the output (L3)."""
    path = output_dir() / HOLDOUT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    detail = "".join(traceback.format_exception(error))
    with path.open("a", encoding="utf-8") as log:
        log.write(f"{now().isoformat()} {strategy_id_}\n{detail}\n")


def _approve_with_valid_code(strategy_id_: str, strategy: StrategyDefinition) -> dict[str, Any]:
    """Write the approval, run the holdout once (D12) and start paper trading.

    If an approval row exists that was never started, this resumes it. If the
    holdout was already claimed, it stops. The claim itself is atomic.
    """
    today = paper_today()
    if today < HOLDOUT_START:
        raise CommandError("There is no data after 2023 yet, so the holdout cannot run.")
    if not storage.record_approval(strategy_id_):
        existing = storage.find_stored_approval(strategy_id_)
        if existing is None or existing.used:
            raise CommandError("This strategy was already approved. The holdout runs only once.")
    try:
        holdout = _run_holdout(strategy_id_, strategy, today)
    except HoldoutAccessError as error:
        _log_holdout_failure(strategy_id_, error)
        raise CommandError(
            "The holdout guard refused this run. It runs only once per strategy."
        ) from error
    except Exception as error:
        _log_holdout_failure(strategy_id_, error)
        raise CommandError(
            "The holdout run started and failed. It cannot be repeated for this strategy. "
            f"Details are in {reporting.relative_path(str(output_dir() / HOLDOUT_LOG))}."
        ) from error
    start_paper(strategy_id_, today)
    out = reporting.base(True)
    out.update(strategy_id=strategy_id_, holdout=holdout, paper_trading_started=True)
    out["paper_start_day"] = today.isoformat()
    out.update(reporting.universe_fields())
    _add_simulated_approval_note(out)
    return out


def _run_holdout(sid: str, strategy: StrategyDefinition, today: date) -> dict[str, Any]:
    """The one holdout run, from fresh cash on 2024-01-01 to today, inside the session."""
    window = BacktestWindow("holdout", HOLDOUT_START, today, None)
    costs = make_cost_model(DEFAULT_COST_BPS_PER_SIDE)
    tried = storage.versions_tried(chat_id(), including=sid)
    with holdout_session(sid):
        market = load_market_data(PRICE_START, today, approved=True)
        result = run_backtest(strategy, market, window, costs, holdout_approved=True)
        metrics = compute_metrics(result, market, tried)
    run = {
        "window_label": "holdout",
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "cost_bps": costs.bps_per_side,
        "metrics": reporting.run_metrics(metrics),
        "chart_paths": [],
    }
    ledger = [{**trade_row(t), "window_label": "holdout"} for t in result.trades]
    storage.replace_backtest(sid, "holdout", [run], ledger, chat_id())
    return reporting.holdout_payload(metrics, storage.count_used_approvals())


# ---------------------------------------------------------------------- paper


def paper_update_command() -> dict[str, Any]:
    """Step every paper portfolio forward to today (cron only)."""
    updates: list[dict[str, Any]] = []
    errors: list[str] = []
    today = paper_today()
    for run in storage.list_paper_runs():
        try:
            result = paper_update(run["strategy_id"], today)
        except Exception as error:
            errors.append(f"{run['strategy_id']}: {type(error).__name__}: {error}")
            continue
        updates.append(
            {
                "strategy_id": result.strategy_id,
                "days_stepped": [day.isoformat() for day in result.stepped_days],
                "last_date": None if result.last_date is None else result.last_date.isoformat(),
                "equity": None if result.equity is None else reporting.money(result.equity),
                "closed_trades": result.closed_trades,
            }
        )
    out = reporting.base(not errors)
    out["updates"] = updates
    if errors:
        out["errors"] = errors
    return out


def _paper_portfolio(sid: str) -> dict[str, Any] | None:
    """The current paper portfolio for one run, or None before the first step."""
    latest = storage.latest_paper_state(sid)
    ledger = storage.paper_ledger(sid)
    if latest is None or not ledger:
        return None
    state = state_from_json(latest[1])
    equity = float(ledger[-1]["equity"])
    previous = float(ledger[-2]["equity"]) if len(ledger) > 1 else INITIAL_CAPITAL
    return {
        "strategy_id": sid,
        "positions": _positions(state),
        "pending_orders": [
            {
                "ticker": o.symbol.removesuffix(".US"),
                "side": o.side,
                "decided_on": o.decided_on.isoformat(),
            }
            for o in state.pending
        ],
        "cash": reporting.money(state.cash),
        "holdings_value": reporting.money(float(ledger[-1]["holdings_value"])),
        "equity": reporting.money(equity),
        "day_change_pct": reporting.percent_change(equity, previous),
        "total_return_pct": reporting.percent_change(equity, INITIAL_CAPITAL),
        "last_update": latest[0],
    }


def _positions(state: PortfolioState) -> list[dict[str, Any]]:
    """Open positions as display-ready rows."""
    return [
        {
            "ticker": p.symbol.removesuffix(".US"),
            "shares": round(p.shares, 4),
            "entry_price": reporting.money(p.entry_price),
            "entry_date": p.entry_date.isoformat(),
            "last_close": reporting.money(p.last_close),
            "market_value": reporting.money(p.shares * p.last_close),
            "dividends_received": reporting.money(p.dividends),
            # Includes dividends paid while held. Entry and exit costs are not included.
            "return_since_entry_pct": reporting.percent_change(
                p.shares * p.last_close + p.dividends, p.entry_shares * p.entry_price
            ),
            "days_held": p.days_held,
        }
        for p in state.positions.values()
    ]


def paper_status_command() -> dict[str, Any]:
    """Current paper positions, cash, equity and the latest update date."""
    out = reporting.base(True)
    out.update(reporting.universe_fields())
    runs = storage.list_paper_runs()
    portfolios = [p for p in (_paper_portfolio(r["strategy_id"]) for r in runs) if p]
    if not portfolios:
        out.update(paper_trading_started=bool(runs), positions=[], message="No paper update yet.")
        return out
    out.update(portfolios[0])
    out["paper_trading_started"] = True
    out["other_paper_strategies"] = [p["strategy_id"] for p in portfolios[1:]]
    out["portfolios"] = portfolios
    return out


def daily_summary_text() -> str:
    """Plain text for Telegram. NO_REPLY when no paper portfolio was updated today.

    Each portfolio lists every trade since its last summarised day, not just
    today's, so a catch-up run reports the missed days (M6). The last summary
    is stored per strategy.
    """
    today = paper_today()
    blocks: list[dict[str, Any]] = []
    for run in storage.list_paper_runs():
        portfolio = _paper_portfolio(run["strategy_id"])
        if portfolio is not None and portfolio["last_update"] == today.isoformat():
            since = _summary_baseline(run["strategy_id"], portfolio["last_update"])
            blocks.append({"lines": _summary_lines(portfolio, since)})
            _remember_summary(run["strategy_id"], portfolio, since)
    return reporting.summary_text(blocks, today) if blocks else NO_REPLY


def _summary_baseline(sid: str, day: str) -> dict[str, Any]:
    """The day and equity to compare with: the last summary before ``day``, else the start."""
    saved = storage.get_meta(SUMMARY_META_PREFIX + sid)
    record = json.loads(saved) if saved else None
    if record is not None and record["day"] == day:
        # A repeat run on the same day gives the same summary again.
        return {"day": record["prev_day"], "equity": record["prev_equity"]}
    if record is not None:
        return {"day": record["day"], "equity": record["equity"]}
    return {"day": None, "equity": INITIAL_CAPITAL}


def _remember_summary(sid: str, portfolio: dict[str, Any], since: dict[str, Any]) -> None:
    """Store this summary's day and equity, plus the baseline it used."""
    record = {
        "day": portfolio["last_update"],
        "equity": portfolio["equity"],
        "prev_day": since["day"],
        "prev_equity": since["equity"],
    }
    storage.set_meta(SUMMARY_META_PREFIX + sid, json.dumps(record))


def _summary_lines(portfolio: dict[str, Any], since: dict[str, Any]) -> list[str]:
    """Plain lines for one portfolio, from its rounded numbers."""
    sid = portfolio["strategy_id"]
    since_day = since["day"]
    sold = storage.list_trades(sid, "paper", exited_after=since_day)
    change = reporting.percent_change(portfolio["equity"], since["equity"])
    label = f"since {since_day}" if since_day else "since start"
    lines = [
        f"Strategy {sid}",
        f"Equity {portfolio['equity']} (day {portfolio['day_change_pct']}%, "
        f"{label} {change}%, since start {portfolio['total_return_pct']}%)",
        f"Cash {portfolio['cash']}. Open positions: {len(portfolio['positions'])}.",
    ]
    for trade in sold:
        ticker = str(trade["symbol"]).removesuffix(".US")
        lines.append(
            f"Sold {ticker} on {trade['exit_date']} ({trade['exit_reason']}), "
            f"{reporting.r2(trade['return_pct'])}%"
        )
    bought = [p for p in portfolio["positions"] if since_day is None or p["entry_date"] > since_day]
    lines.extend(f"Bought {p['ticker']} on {p['entry_date']} at {p['entry_price']}" for p in bought)
    for order in portfolio["pending_orders"]:
        lines.append(f"Next open: {order['side']} {order['ticker']}")
    return lines


# ------------------------------------------------------------------------ why


def why_command(ticker: str, day: str | None) -> dict[str, Any]:
    """Explain a stored entry from its saved signal values only."""
    if not TICKER_PATTERN.match(ticker):
        raise CommandError("That does not look like a ticker. Use capitals, such as AAPL.")
    wanted = _parse_day(day)
    symbol = f"{ticker}.US"
    # Holdout trades are never offered here, whatever strategy they belong to (review M4).
    candidates = _paper_candidates(symbol) + storage.find_trades(symbol, ("backtest", "paper"))
    if wanted is not None:
        key = wanted.isoformat()
        candidates = [c for c in candidates if key in (c["signal_date"], c["entry_date"])]
    if not candidates:
        raise CommandError(f"No stored entry for {ticker}" + (f" on {day}." if day else "."))
    candidates.sort(key=lambda item: item["signal_date"], reverse=True)
    pick = candidates[0]
    out = reporting.base(True)
    out.update(
        ticker=ticker,
        date=pick["signal_date"],
        entry_date=pick["entry_date"],
        source=pick["source"],
        status=pick.get("status", "closed trade"),
        strategy_id=pick["strategy_id"],
        blocks_fired=reporting.blocks_fired(pick["signal_values"]),
        note="Values were measured at the decision close and saved then.",
    )
    return out


def _parse_day(day: str | None) -> date | None:
    """Parse a YYYY-MM-DD option, or raise a plain error."""
    if day is None:
        return None
    try:
        return date.fromisoformat(day)
    except ValueError as error:
        raise CommandError("The date must look like 2024-03-15.") from error


def _paper_candidates(symbol: str) -> list[dict[str, Any]]:
    """Open paper positions and pending buys for a symbol, from the saved state."""
    found: list[dict[str, Any]] = []
    for run in storage.list_paper_runs():
        latest = storage.latest_paper_state(run["strategy_id"])
        if latest is None:
            continue
        state = state_from_json(latest[1])
        head = {"source": "paper", "strategy_id": run["strategy_id"]}
        position = state.positions.get(symbol)
        if position is not None:
            found.append(
                {
                    **head,
                    "status": "open position",
                    "signal_date": position.signal_date.isoformat(),
                    "entry_date": position.entry_date.isoformat(),
                    "signal_values": position.signal_values,
                }
            )
        for order in state.pending:
            if order.symbol == symbol and order.side == "buy":
                decided = order.decided_on.isoformat()
                found.append(
                    {
                        **head,
                        "status": "buy pending at next open",
                        "signal_date": decided,
                        "entry_date": decided,
                        "signal_values": order.signal_values,
                    }
                )
    return found


# ---------------------------------------------------------------- data status


def data_status_command() -> dict[str, Any]:
    """Data mode, cache coverage and where the data-quality report is."""
    cache = client.FAKE_CACHE_DIR if data_mode() == "fake" else client.DEFAULT_CACHE_DIR
    report = output_dir() / "data_quality.md"
    out = reporting.base(True)
    out.update(reporting.universe_fields())
    out["cache_coverage"] = _cache_coverage(cache)
    out["holdout"] = (
        f"Data from {HOLDOUT_START.isoformat()} is sealed until a strategy is approved."
    )
    out["holdout_runs_so_far"] = storage.count_used_approvals()
    out["quality_report"] = reporting.relative_path(str(report))
    out["quality_report_exists"] = report.exists()
    saved = output_dir() / "load_report.json"
    out["last_load"] = json.loads(saved.read_text()) if saved.exists() else None
    return out


def _cache_coverage(cache: Path) -> dict[str, Any]:
    """Cached symbol folders per endpoint, and the price files cached for SPY."""
    if not cache.exists():
        return {"cached": False}
    counts = {
        folder.name: sum(1 for item in folder.iterdir() if item.is_dir())
        for folder in sorted(cache.iterdir())
        if folder.is_dir() and folder.name in ("eod", "div", "splits", "news", "sentiments")
    }
    spy = cache / "eod" / "SPY.US"
    ranges = sorted(item.stem for item in spy.glob("*.json")) if spy.exists() else []
    return {"cached": True, "symbols_per_endpoint": counts, "spy_price_files": ranges}
