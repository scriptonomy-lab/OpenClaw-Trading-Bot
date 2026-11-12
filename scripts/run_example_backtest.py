"""Run the brief's gap-down example end to end and print the results as JSON.

Reads cached data through the client wrappers, so it works offline once the
data is downloaded. In fake mode every number and chart is SIMULATED DATA:

    EODHD_MODE=fake uv run python scripts/run_example_backtest.py

It runs the primary window, the cost sensitivity, the metrics and the charts,
then checks the two planted gap-down cases (with and without news).
"""

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from bot.backtest.charts import plot_cost_sensitivity, plot_drawdown, plot_equity_vs_benchmarks
from bot.backtest.costs import make_cost_model
from bot.backtest.engine import BacktestResult, Trade, run_backtest, run_cost_sensitivity
from bot.backtest.metrics import benchmark_curves, compute_metrics
from bot.config import (
    BACKTEST_END,
    DEFAULT_COST_BPS_PER_SIDE,
    SIMULATED_LABEL,
    data_mode,
    output_dir,
)
from bot.data.market import PRICE_START, load_market_report
from bot.strategy.blocks import (
    MarketData,
    gap_block,
    no_news_block,
    slice_to_decision_date,
)
from bot.strategy.schema import (
    GapCondition,
    NoNewsCondition,
    StrategyDefinition,
    backtest_windows,
    strategy_from_json,
)

EXAMPLE_DESCRIPTION = (
    "Buy S&P 500 stocks whose open is 3% or more below the previous close, "
    "with no news since the previous close. Sell after 3 trading days."
)
EXAMPLE_JSON = json.dumps(
    {
        "name": "Gap down, no news",
        "description_plain_english": EXAMPLE_DESCRIPTION,
        "version": 1,
        "entry": [
            {"block": "gap", "direction": "down", "min_gap_pct": 3},
            {"block": "no_news", "lookback_sessions": 1},
        ],
        "exit": {"holding_days": 3},
        "sizing": {"max_positions": 10},
    }
)
NO_NEWS_CASE = ("SIM001.US", date(2022, 2, 10))
WITH_NEWS_CASE = ("SIM002.US", date(2022, 6, 14))


def trade_row(trade: Trade) -> dict[str, Any]:
    """A ledger row as plain JSON values."""
    return json.loads(json.dumps(asdict(trade), default=str))


def planted_case(
    result: BacktestResult, market: MarketData, strategy: StrategyDefinition, case: tuple[str, date]
) -> dict[str, Any]:
    """What the strategy did on one planted gap-down day."""
    symbol, day = case
    data = slice_to_decision_date(market, day)
    gap = GapCondition(block="gap", direction="down", min_gap_pct=3)
    quiet = NoNewsCondition(block="no_news", lookback_sessions=1)
    trades = [t for t in result.trades if t.symbol == symbol and t.signal_date == day]
    return {
        "symbol": symbol,
        "signal_date": day.isoformat(),
        "is_member": symbol in data.members,
        "gap_block": bool(gap_block(data, gap).get(symbol, False)),
        "no_news_block": bool(no_news_block(data, quiet).get(symbol, False)),
        "trades": [trade_row(t) for t in trades],
    }


def save_charts(
    result: BacktestResult, sensitivity: dict[float, BacktestResult], market: MarketData
) -> list[str]:
    """Write the three charts under the output folder and return their paths."""
    metrics = compute_metrics(result, market, versions_tried=1)
    curves = benchmark_curves(result, market)
    folder = output_dir()
    paths: list[Path] = [
        plot_equity_vs_benchmarks(curves, metrics, folder / "example_equity.png"),
        plot_drawdown(curves, metrics, folder / "example_drawdown.png"),
        plot_cost_sensitivity(sensitivity, folder / "example_cost_sensitivity.png"),
    ]
    return [str(path) for path in paths]


def main() -> None:
    """Run the example and print one JSON report."""
    strategy = strategy_from_json(EXAMPLE_JSON)
    window = backtest_windows(strategy)[0]
    market, load_report = load_market_report(PRICE_START, BACKTEST_END)
    costs = make_cost_model(DEFAULT_COST_BPS_PER_SIDE)
    result = run_backtest(strategy, market, window, costs)
    sensitivity = run_cost_sensitivity(strategy, market, window)
    metrics = compute_metrics(result, market, versions_tried=1)
    report = {
        "label": SIMULATED_LABEL if data_mode() == "fake" else "LIVE DATA",
        "window": {"start": str(window.start), "end": str(window.end), "label": window.label},
        "metrics": json.loads(json.dumps(asdict(metrics), default=str)),
        "metrics_by_cost_bps": {
            str(bps): json.loads(
                json.dumps(asdict(compute_metrics(r, market, 1).strategy_returns), default=str)
            )
            for bps, r in sensitivity.items()
        },
        "data_quality": load_report.to_dict(),
        "trade_count": len(result.trades),
        "planted_no_news": planted_case(result, market, strategy, NO_NEWS_CASE),
        "planted_with_news": planted_case(result, market, strategy, WITH_NEWS_CASE),
        "charts": save_charts(result, sensitivity, market),
    }
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
