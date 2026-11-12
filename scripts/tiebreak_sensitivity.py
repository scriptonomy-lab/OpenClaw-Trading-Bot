"""How much do results depend on the slot tie-break? Rerun the three worked examples with 5 salts.

When more stocks qualify than there are free slots, the engine picks them by a
fixed hash shuffle. A salt gives a different shuffle that is just as fair. This
script runs each worked example from docs/design/strategy_schema.md on its
primary window at the default cost, once with the normal shuffle and once per
salt, and reports the min, median and max of total return, Sharpe and max
drawdown over the 5 salts. In fake mode every number is SIMULATED DATA:

    EODHD_MODE=fake uv run python scripts/tiebreak_sensitivity.py

Writes ``tiebreak_sensitivity.md`` and ``tiebreak_sensitivity.json`` under the
output folder. The salt is a research option only. Strategies cannot set it.
"""

import json
import statistics
import sys
from dataclasses import dataclass
from typing import Any

from bot.backtest.costs import make_cost_model
from bot.backtest.engine import BacktestResult, run_backtest
from bot.backtest.metrics import compute_metrics
from bot.config import (
    BACKTEST_END,
    DEFAULT_COST_BPS_PER_SIDE,
    SIMULATED_LABEL,
    data_mode,
    output_dir,
)
from bot.data.market import PRICE_START, load_market_report
from bot.strategy.blocks import MarketData
from bot.strategy.schema import (
    StrategyDefinition,
    backtest_windows,
    strategy_from_json,
    strategy_id,
)

SALTS = ("salt-1", "salt-2", "salt-3", "salt-4", "salt-5")
METRIC_NAMES = ("total_return_pct", "sharpe", "max_drawdown_pct")

# The three worked examples from docs/design/strategy_schema.md, unchanged.
EXAMPLES: dict[str, dict[str, Any]] = {
    "Gap down, no news": {
        "name": "Gap down, no news",
        "description_plain_english": "Buy S&P 500 stocks whose open is 3% or more below "
        "the previous close, with no news since the previous close. Sell after 3 trading days.",
        "entry": [
            {"block": "gap", "direction": "down", "min_gap_pct": 3},
            {"block": "no_news", "lookback_sessions": 1},
        ],
        "exit": {"holding_days": 3},
        "sizing": {"max_positions": 10},
    },
    "Trend momentum": {
        "name": "Trend momentum",
        "description_plain_english": "Trend momentum worked example from the schema note.",
        "entry": [
            {"block": "spy_filter", "ma_days": 200, "position": "above"},
            {"block": "n_day_return", "days": 126, "comparison": "at_least", "return_pct": 20},
            {"block": "price_vs_moving_average", "ma_days": 50, "position": "above"},
            {"block": "rsi", "period": 14, "comparison": "at_most", "level": 70},
        ],
        "exit": {"holding_days": 20, "stop_loss_pct": 10},
        "sizing": {"max_positions": 10},
    },
    "Positive news, heavy volume": {
        "name": "Positive news, heavy volume",
        "description_plain_english": "News sentiment worked example from the schema note.",
        "entry": [
            {
                "block": "news_sentiment",
                "lookback_sessions": 1,
                "comparison": "at_least",
                "polarity": 0.5,
                "min_articles": 3,
            },
            {"block": "relative_volume", "lookback_days": 20, "comparison": "at_least", "ratio": 2},
            {"block": "days_to_earnings", "relation": "not_within", "days": 5},
        ],
        "exit": {"holding_days": 5, "stop_loss_pct": 5, "profit_target_pct": 8},
        "sizing": {"max_positions": 5},
    },
}


@dataclass(frozen=True)
class RunNumbers:
    """The three metrics of one run, plus how many signals missed a slot."""

    total_return_pct: float
    sharpe: float | None
    max_drawdown_pct: float
    missed_signals: int


def missed_signals(result: BacktestResult) -> int:
    """Signals that did not get a slot, summed from the engine's notes."""
    total = 0
    for note in result.notes:
        if "entry signals did not get a slot" in note:
            total += int(note.split(": ", 1)[1].split()[0])
    return total


def run_once(strategy: StrategyDefinition, market: MarketData, salt: str) -> RunNumbers:
    """One backtest on the primary window at the default cost."""
    window = backtest_windows(strategy)[0]
    costs = make_cost_model(DEFAULT_COST_BPS_PER_SIDE)
    result = run_backtest(strategy, market, window, costs, tiebreak_salt=salt)
    stats = compute_metrics(result, market, versions_tried=1).strategy_returns
    return RunNumbers(
        stats.total_return_pct, stats.sharpe, stats.max_drawdown_pct, missed_signals(result)
    )


def spread(values: list[float | None]) -> dict[str, float | None]:
    """Min, median and max, ignoring runs where the value is undefined (Sharpe None)."""
    known = [value for value in values if value is not None]
    if not known:
        return {"min": None, "median": None, "max": None}
    return {"min": min(known), "median": statistics.median(known), "max": max(known)}


def study(name: str, market: MarketData) -> dict[str, Any]:
    """The default run and the 5 salted runs of one example, summarised."""
    strategy = strategy_from_json(json.dumps(EXAMPLES[name]))
    window = backtest_windows(strategy)[0]
    default = run_once(strategy, market, "")
    salted = [run_once(strategy, market, salt) for salt in SALTS]
    return {
        "strategy": name,
        "strategy_id": strategy_id(strategy),
        "window": f"{window.start} to {window.end}",
        "missed_signals_default": default.missed_signals,
        "default": {metric: getattr(default, metric) for metric in METRIC_NAMES},
        "salts": {metric: spread([getattr(r, metric) for r in salted]) for metric in METRIC_NAMES},
    }


def number(value: float | None) -> str:
    """A value for the table, or "n/a"."""
    return "n/a" if value is None else f"{value:.2f}"


def markdown(label: str, rows: list[dict[str, Any]]) -> str:
    """The results as a Markdown table, headed by the data label."""
    lines = [
        f"# Slot tie-break sensitivity ({label})",
        "",
        f"Primary window, {DEFAULT_COST_BPS_PER_SIDE:g} bps per side. Salts: {', '.join(SALTS)}.",
        "",
        "| Strategy | Metric | Default | Min | Median | Max | Missed signals (default) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        for metric in METRIC_NAMES:
            cells = row["salts"][metric]
            lines.append(
                f"| {row['strategy']} | {metric} | {number(row['default'][metric])} | "
                f"{number(cells['min'])} | {number(cells['median'])} | {number(cells['max'])} | "
                f"{row['missed_signals_default']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run all three examples, print the table, and save it under the output folder."""
    label = SIMULATED_LABEL if data_mode() == "fake" else "LIVE DATA"
    market, _ = load_market_report(PRICE_START, BACKTEST_END)
    rows = [study(name, market) for name in EXAMPLES]
    text = markdown(label, rows)
    folder = output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tiebreak_sensitivity.md").write_text(text)
    payload = {"label": label, "salts": list(SALTS), "results": rows}
    (folder / "tiebreak_sensitivity.json").write_text(json.dumps(payload, indent=2))
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
