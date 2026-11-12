"""Charts for reports, drawn with matplotlib and saved as PNG files.

Interface stubs for Phase 3. See docs/design/backtest.md.

Charts only draw numbers computed in bot/backtest/metrics.py. Every chart
title or footnote states the cost level, the window, the number of versions
tried and the universe (D33), so a chart shared on its own is not misleading.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bot.backtest.engine import BacktestResult  # noqa: E402
from bot.backtest.metrics import Metrics  # noqa: E402
from bot.config import MIN_TRADES_TO_JUDGE, output_dir, report_title  # noqa: E402
from bot.reporting import benchmark_labels, universe_fields  # noqa: E402

FIGURE_SIZE = (10, 5.5)
# Colour-blind safe: dark blue, orange, grey.
COLORS = {"strategy": "#1f4e79", "spy": "#e69f00", "equal_weight": "#7f7f7f"}


def plot_equity_vs_benchmarks(curves: pd.DataFrame, metrics: Metrics, path: Path) -> Path:
    """Strategy against SPY and the equal-weight S&P 500, all starting at 100.

    ``curves`` is the output of ``benchmark_curves``. Uses a log scale only if
    a curve more than triples, and says so in the axis label. Returns ``path``.
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    scaled = curves * 100.0
    log_scale = bool((scaled.max() > 300.0).any())
    labels = _curve_labels(metrics)
    for column in ("strategy", "spy", "equal_weight"):
        axis.plot(scaled.index, scaled[column], label=labels[column], color=COLORS[column])
    if log_scale:
        axis.set_yscale("log")
    axis.set_ylabel("Value, start = 100" + (" (log scale)" if log_scale else ""))
    axis.set_title(
        _titled(f"Strategy against benchmarks, {metrics.start} to {metrics.end}"), fontsize=11
    )
    axis.legend(loc="upper left")
    return _save(figure, axis, _footnote(metrics), path)


def plot_drawdown(curves: pd.DataFrame, metrics: Metrics, path: Path) -> Path:
    """Drawdown from the previous peak over time, for the strategy and both benchmarks."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    labels = _curve_labels(metrics)
    for column in ("strategy", "spy", "equal_weight"):
        drawdown = (curves[column] / curves[column].cummax() - 1.0) * 100.0
        axis.plot(drawdown.index, drawdown, label=labels[column], color=COLORS[column])
    axis.set_ylabel("Fall from previous peak (%)")
    axis.set_title(_titled(f"Drawdown, {metrics.start} to {metrics.end}"), fontsize=11)
    axis.legend(loc="lower left")
    return _save(figure, axis, _footnote(metrics), path)


def plot_cost_sensitivity(
    results: dict[float, BacktestResult], path: Path, *, versions_tried: int | None = None
) -> Path:
    """Strategy equity at each cost level (0, 5 and 20 bps per side) on one chart.

    ``versions_tried`` comes from storage, like the other charts. If the caller
    does not pass it, the footnote says it was not recorded.
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    first = next(iter(results.values()))
    for bps in sorted(results):
        equity = results[bps].equity_curve["equity"]
        axis.plot(equity.index, equity, label=f"{bps:g} bps per side")
    axis.set_ylabel("Portfolio value")
    window = first.window
    axis.set_title(_titled(f"Cost sensitivity, {window.start} to {window.end}"), fontsize=11)
    axis.legend(loc="upper left")
    tried = "not recorded" if versions_tried is None else str(versions_tried)
    note = (
        f"Same rules at each cost level. Versions tried: {tried}. "
        "Every point values open positions at that day's close, net of one sell cost."
    )
    return _save(figure, axis, note + "\n" + _universe_note(), path)


def _titled(title: str) -> str:
    """The chart title with the simulated label when needed, then the universe sentence."""
    universe: str = universe_fields()["universe"]
    return report_title(title) + "\n" + universe


def _universe_note() -> str:
    """The universe sentence and, for the sample, the limitations note (D33)."""
    fields = universe_fields()
    return " ".join([fields["universe"], *fields["limitations"]])


def _curve_labels(metrics: Metrics) -> dict[str, str]:
    """Legend text with each curve's total return, as computed in metrics.py."""
    return {
        "strategy": f"Strategy ({metrics.strategy_returns.total_return_pct:+.1f}%)",
        "spy": f"SPY ({metrics.spy.total_return_pct:+.1f}%)",
        "equal_weight": (
            f"{benchmark_labels()['equal_weight_sp500']} "
            f"({metrics.equal_weight.total_return_pct:+.1f}%)"
        ),
    }


def _footnote(metrics: Metrics) -> str:
    """The line printed under every chart so it is not misleading when shared alone."""
    too_few = (
        f" Closed trades: {metrics.closed_trades}, too few to judge (under {MIN_TRADES_TO_JUDGE})."
        if metrics.too_few_trades
        else f" Closed trades: {metrics.closed_trades}."
    )
    return (
        f"Costs: {metrics.cost_bps_per_side:g} bps per side. "
        f"Window: {metrics.start} to {metrics.end}. "
        f"Versions tried: {metrics.versions_tried}.{too_few}\n"
        "Strategy is net of costs, including the cost of selling open positions. "
        "Benchmarks include dividends, no costs.\n" + _universe_note()
    )


def _save(figure: Figure, axis: Axes, footnote: str, path: Path) -> Path:
    """Decorate, save under the output folder, close the figure and return the path."""
    target = _checked_path(path)
    axis.grid(alpha=0.3)
    figure.text(0.01, 0.01, footnote, fontsize=8, color="#444444", ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return path


def _checked_path(path: Path) -> Path:
    """The resolved path, if it lies under the output folder. Else raises ValueError."""
    target = path.resolve()
    if not target.is_relative_to(output_dir().resolve()):
        raise ValueError(f"Charts may only be saved under {output_dir()}, not {path}.")
    return target
