"""Display-ready numbers and text for the CLI (D21).

Every figure is computed in Python and rounded once here, with percentages
already in percent. The agent quotes these values as they are. Nothing here
returns raw news text (D6).
"""

from datetime import date
from typing import Any

from bot.backtest.metrics import Metrics, ReturnStats
from bot.config import (
    DEFAULT_COST_BPS_PER_SIDE,
    MIN_TRADES_TO_JUDGE,
    PROJECT_ROOT,
    SIMULATED_LABEL,
    data_mode,
    secret,
)
from bot.strategy.schema import backtest_windows, strategy_from_json
from bot.universe import SAMPLE_LIMITATIONS, UniverseMode, universe_label, universe_mode

DECIMALS = 2
SHARPE_NOTE = "Sharpe uses daily returns and a risk-free rate of 0."
TOO_FEW_NOTE = f"Too few trades to judge this strategy (under {MIN_TRADES_TO_JUDGE})."
SIMULATED_APPROVAL_NOTE = (
    "This approval and holdout run used simulated data. They are stored with the simulated "
    "results only. The live holdout record is untouched."
)


def universe_fields(mode: UniverseMode | None = None) -> dict[str, Any]:
    """The universe sentence and, for the sample, the limitations note (D33)."""
    chosen = mode or universe_mode()
    return {
        "universe_mode": chosen,
        "universe": universe_label(chosen),
        "limitations": [SAMPLE_LIMITATIONS] if chosen == "sample" else [],
    }


def benchmark_labels(mode: UniverseMode | None = None) -> dict[str, str]:
    """Plain names for the benchmark keys. Equal-weight uses the same tickers as the run."""
    chosen = mode or universe_mode()
    equal = "Equal-weight, same sample" if chosen == "sample" else "Equal-weight, full universe"
    return {"spy": "SPY", "equal_weight_sp500": equal}


def stored_universe(runs: list[dict[str, Any]]) -> UniverseMode:
    """The universe a stored run used. Rows saved before D33 count as the current one."""
    mode = runs[0]["metrics"].get("universe_mode") if runs else None
    if mode == "full":
        return "full"
    if mode == "sample":
        return "sample"
    return universe_mode()


def universe_text(mode: UniverseMode | None = None) -> str:
    """The universe sentence plus, for the sample, the limitations note, as one line."""
    fields = universe_fields(mode)
    return " ".join([fields["universe"], *fields["limitations"]])


def r2(value: float | None) -> float | None:
    """Round once to two decimals. None stays None."""
    return None if value is None else round(float(value), DECIMALS)


def cost_key(bps: float) -> str:
    """A cost level as a JSON key, so 5.0 becomes "5"."""
    return f"{bps:g}"


def base(ok: bool) -> dict[str, Any]:
    """The fields every command output starts with."""
    out: dict[str, Any] = {"ok": ok, "data_mode": data_mode(), "model": secret("OPENCLAW_MODEL")}
    if data_mode() == "fake":
        out["banner"] = SIMULATED_LABEL
    return out


def failure(errors: list[str]) -> dict[str, Any]:
    """A failed command output with plain-English errors."""
    out = base(False)
    out["errors"] = errors
    return out


def relative_path(path: str) -> str:
    """A path relative to the project root when it is inside it."""
    try:
        return str(PROJECT_ROOT.joinpath(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return path


def stats_dict(stats: ReturnStats) -> dict[str, float | None]:
    """The four return statistics, rounded."""
    return {
        "total_return_pct": r2(stats.total_return_pct),
        "max_drawdown_pct": r2(stats.max_drawdown_pct),
        "sharpe": r2(stats.sharpe),
    }


def run_metrics(metrics: Metrics) -> dict[str, Any]:
    """Everything stored and shown for one window at one cost level."""
    own = metrics.strategy_returns
    return {
        "total_return_pct": r2(own.total_return_pct),
        "cagr_pct": r2(own.cagr_pct),
        "max_drawdown_pct": r2(own.max_drawdown_pct),
        "sharpe": r2(own.sharpe),
        "win_rate_pct": r2(metrics.win_rate_pct),
        "average_trade_return_pct": r2(metrics.average_trade_return_pct),
        "closed_trades": metrics.closed_trades,
        "open_at_end": metrics.open_at_end,
        "data_ended_trades": metrics.data_ended_trades,
        "too_few_trades": metrics.too_few_trades,
        "versions_tried": metrics.versions_tried,
        "universe_mode": universe_mode(),
        "benchmarks": {
            "spy": stats_dict(metrics.spy),
            "equal_weight_sp500": stats_dict(metrics.equal_weight),
        },
    }


def window_payload(label: str, runs: list[dict[str, Any]], warning: str | None) -> dict[str, Any]:
    """One window of a report: facts at the default cost, plus results at each cost level."""
    default = _default_run(runs)["metrics"]
    by_cost = {
        cost_key(run["cost_bps"]): {
            key: run["metrics"][key]
            for key in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe")
            + ("win_rate_pct", "average_trade_return_pct", "closed_trades", "too_few_trades")
        }
        for run in sorted(runs, key=lambda item: item["cost_bps"])
    }
    first = runs[0]
    window: dict[str, Any] = {
        "label": label,
        "start": first["window_start"],
        "end": first["window_end"],
        "coverage_warning": warning,
        "closed_trades": default["closed_trades"],
        "open_at_end": default["open_at_end"],
        "data_ended_trades": default["data_ended_trades"],
        "too_few_trades": default["too_few_trades"],
        "by_cost_bps": by_cost,
        "benchmarks": default["benchmarks"],
    }
    if default["too_few_trades"]:
        window["note"] = TOO_FEW_NOTE
    return window


def _default_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The run at the default cost level."""
    for run in runs:
        if run["cost_bps"] == DEFAULT_COST_BPS_PER_SIDE:
            return run
    return runs[0]


def report_payload(strategy_json: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The backtest report built from saved runs. Used by both ``backtest`` and ``report``."""
    warnings = {
        w.label: w.coverage_warning for w in backtest_windows(strategy_from_json(strategy_json))
    }
    labels = list(dict.fromkeys(run["window_label"] for run in runs))
    windows = [
        window_payload(label, [r for r in runs if r["window_label"] == label], warnings.get(label))
        for label in labels
    ]
    first = runs[0]
    charts = first["chart_paths"]
    mode = stored_universe(runs)
    out = base(True)
    out.update(universe_fields(mode))
    out["benchmark_labels"] = benchmark_labels(mode)
    out["windows"] = windows
    out["versions_tried"] = _default_run(runs)["metrics"]["versions_tried"]
    out["chart_path"] = relative_path(charts[0]) if charts else None
    out["notes"] = [
        SHARPE_NOTE,
        "Costs are per side. The default is 5 bps; 0 and 20 bps are shown too.",
    ]
    return out


def telegram_caption(
    strategy_name: str,
    window: dict[str, Any],
    versions_tried: int,
    mode: UniverseMode | None = None,
) -> str:
    """A short caption for the equity chart, from the window's rounded numbers.

    The too-few-trades note sits next to the trade count, and the universe
    sentence comes last, so a cut at Telegram's limit drops the least.
    """
    own = window["by_cost_bps"][cost_key(DEFAULT_COST_BPS_PER_SIDE)]
    spy = window["benchmarks"]["spy"]
    equal = window["benchmarks"]["equal_weight_sp500"]
    equal_name = benchmark_labels(mode)["equal_weight_sp500"]
    too_few = f" {TOO_FEW_NOTE}" if window["too_few_trades"] else ""
    return (
        f"{strategy_name}, {window['label']} window {window['start']} to {window['end']}. "
        f"Strategy {own['total_return_pct']}% at 5 bps per side, SPY {spy['total_return_pct']}%, "
        f"{equal_name} {equal['total_return_pct']}%. "
        f"Max drawdown {own['max_drawdown_pct']}%. Closed trades {window['closed_trades']}."
        f"{too_few} Versions tried {versions_tried}. {universe_text(mode)}"
    )


def holdout_payload(metrics: Metrics, runs_so_far: int) -> dict[str, Any]:
    """The holdout result block of ``approve``, with the count of holdout runs so far."""
    run = run_metrics(metrics)
    return {
        "start": metrics.start.isoformat(),
        "end": metrics.end.isoformat(),
        "cost_bps_per_side": metrics.cost_bps_per_side,
        "total_return_pct": run["total_return_pct"],
        "max_drawdown_pct": run["max_drawdown_pct"],
        "sharpe": run["sharpe"],
        "win_rate_pct": run["win_rate_pct"],
        "closed_trades": run["closed_trades"],
        "open_at_end": run["open_at_end"],
        "too_few_trades": run["too_few_trades"],
        "benchmarks": run["benchmarks"],
        "benchmark_labels": benchmark_labels(),
        "holdout_runs_so_far": runs_so_far,
        **universe_fields(),
        **({"note": TOO_FEW_NOTE} if run["too_few_trades"] else {}),
    }


# ------------------------------------------------------------- why and paper

MEASURE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("gap_pct", "gap"),
    ("return_", "n_day_return"),
    ("close_vs_ma", "price_vs_moving_average"),
    ("rsi", "rsi"),
    ("relative_volume_", "relative_volume"),
    ("news_count_", "news"),
    ("news_polarity_", "news_sentiment"),
    ("days_to_next_earnings", "days_to_earnings"),
    ("days_since_earnings", "days_since_earnings"),
    ("spy_vs_ma", "spy_filter"),
)


def blocks_fired(values: dict[str, Any]) -> list[dict[str, Any]]:
    """Stored signal values as a list of block, measure and value (rounded)."""
    fired: list[dict[str, Any]] = []
    for measure, value in values.items():
        block = next(
            (name for prefix, name in MEASURE_BLOCKS if measure.startswith(prefix)), measure
        )
        shown = r2(value) if isinstance(value, float) else value
        fired.append({"block": block, "measure": measure, "value": shown})
    return fired


def money(value: float) -> float:
    """A money amount rounded to cents."""
    return round(float(value), DECIMALS)


def percent_change(new: float, old: float) -> float:
    """(new / old - 1) as a rounded percent."""
    return round((new / old - 1.0) * 100.0, DECIMALS)


def summary_text(blocks: list[dict[str, Any]], day: date) -> str:
    """Plain text for Telegram, one block of lines per portfolio. Fake mode adds the label."""
    header = f"Paper portfolio summary for {day.isoformat()}"
    title = f"{SIMULATED_LABEL}: {header}" if data_mode() == "fake" else header
    lines = [title, universe_text()]
    for block in blocks:
        lines.extend(["", *block["lines"]])
    return "\n".join(lines)
