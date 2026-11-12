# ruff: noqa: E501  (the worked-example JSON is copied verbatim from the doc)
"""Build the Phase 6 results summary: one Markdown file plus charts, all numbers from code.

    EODHD_MODE=fake uv run python scripts/build_results_summary.py

Writes ``output_dir()/results_summary.md`` and charts under ``output_dir()/results/``.
Nothing here computes a figure in prose. Every number is formatted from
``compute_metrics`` or read from a stored file. The holdout is never run.
"""

import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtest.charts import plot_cost_sensitivity, plot_equity_vs_benchmarks
from bot.backtest.engine import BacktestResult, run_cost_sensitivity
from bot.backtest.metrics import (
    FULL_UNIVERSE_SKIPPED,
    Metrics,
    ReturnStats,
    benchmark_curves,
    compute_metrics,
    full_universe_equal_weight,
    return_stats,
)
from bot.config import (
    BACKTEST_END,
    BACKTEST_START,
    COST_SENSITIVITY_BPS,
    DEFAULT_COST_BPS_PER_SIDE,
    HOLDOUT_START,
    MIN_TRADES_TO_JUDGE,
    SIMULATED_LABEL,
    data_mode,
    output_dir,
    report_title,
    secret,
)
from bot.data.market import PRICE_START, load_market_report
from bot.data.news import ROBUSTNESS_NEWS_RULES
from bot.strategy.blocks import MarketData
from bot.strategy.schema import (
    BacktestWindow,
    StrategyDefinition,
    backtest_windows,
    strategy_from_json,
)
from bot.universe import SAMPLE_LIMITATIONS, load_sample, universe_label, universe_mode

# These JSON texts are copied from "Worked examples" in docs/design/strategy_schema.md.
# tests/test_results_summary.py checks they still match the doc, so they cannot drift.
GAP_DOWN_JSON = """{
  "name": "Gap down, no news",
  "description_plain_english": "Buy S&P 500 stocks whose open is 3% or more below the previous close, with no news since the previous close. Sell after 3 trading days.",
  "version": 1,
  "entry": [
    {"block": "gap", "direction": "down", "min_gap_pct": 3},
    {"block": "no_news", "lookback_sessions": 1}
  ],
  "exit": {"holding_days": 3},
  "sizing": {"max_positions": 10}
}"""
MOMENTUM_JSON = """{
  "name": "Trend momentum",
  "description_plain_english": "When SPY is above its 200-day average, buy S&P 500 stocks up at least 20% over the last 126 trading days that close above their 50-day average and have an RSI of 70 or less. Sell after 20 trading days, or earlier on a 10% stop loss.",
  "version": 1,
  "entry": [
    {"block": "spy_filter", "ma_days": 200, "position": "above"},
    {"block": "n_day_return", "days": 126, "comparison": "at_least", "return_pct": 20},
    {"block": "price_vs_moving_average", "ma_days": 50, "position": "above"},
    {"block": "rsi", "period": 14, "comparison": "at_most", "level": 70}
  ],
  "exit": {"holding_days": 20, "stop_loss_pct": 10},
  "sizing": {"max_positions": 10}
}"""
NEWS_SENTIMENT_JSON = """{
  "name": "Positive news, heavy volume",
  "description_plain_english": "Buy S&P 500 stocks with at least 3 articles since the previous close whose average polarity is 0.5 or higher, on at least twice their normal volume, with no earnings reaction in the next 5 trading days. Sell after 5 trading days, at an 8% profit, or at a 5% loss.",
  "version": 1,
  "entry": [
    {"block": "news_sentiment", "lookback_sessions": 1, "comparison": "at_least", "polarity": 0.5, "min_articles": 3},
    {"block": "relative_volume", "lookback_days": 20, "comparison": "at_least", "ratio": 2},
    {"block": "days_to_earnings", "relation": "not_within", "days": 5}
  ],
  "exit": {"holding_days": 5, "stop_loss_pct": 5, "profit_target_pct": 8},
  "sizing": {"max_positions": 5}
}"""
WORKED_EXAMPLES = (GAP_DOWN_JSON, MOMENTUM_JSON, NEWS_SENTIMENT_JSON)

VERSIONS_TRIED = 1  # Scripted runs: each strategy is run once, never tuned.
EVAL_RUNS_SHOWN = 5

SEALED_TEXT = "No strategy has been approved yet; the holdout is sealed."
DIRECT_MODE_NOTE = (
    "Direct mode approximates OpenClaw. The model gets SKILL.md as its only system prompt "
    "and one stand-in exec tool. There is no real OpenClaw exec policy, system prompt or "
    "Telegram, so a real OpenClaw run can score differently."
)


@dataclass(frozen=True)
class StrategyRun:
    """One strategy in one window, run at each cost level."""

    strategy: StrategyDefinition
    window: BacktestWindow
    results: dict[float, BacktestResult]
    metrics: dict[float, Metrics]
    charts: list[Path]
    full_equal_weight: ReturnStats | str | None = None


def slug(text: str) -> str:
    """A file-name-safe version of a name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def summary_strategies() -> list[tuple[StrategyDefinition, str]]:
    """The three worked examples plus the gap-down rules without the news filter (D4)."""
    examples = [strategy_from_json(text) for text in WORKED_EXAMPLES]
    gap_down = examples[0]
    no_filter = gap_down.model_copy(
        update={
            "name": "Gap down, no news filter",
            "description_plain_english": (
                "Buy S&P 500 stocks whose open is 3% or more below the previous close. "
                "Sell after 3 trading days. This is the gap-down example without the "
                "no-news filter, run on 2016-2023 to show what the filter adds (D4)."
            ),
            "entry": [condition for condition in gap_down.entry if condition.block != "no_news"],
        }
    )
    no_filter = StrategyDefinition.model_validate(no_filter.model_dump())
    return [(example, "worked example") for example in examples] + [(no_filter, "D4 comparison")]


def run_one_window(
    strategy: StrategyDefinition, market: MarketData, window: BacktestWindow, folder: Path
) -> StrategyRun:
    """Run one strategy in one window at 0, 5 and 20 bps, and draw its two charts."""
    results = run_cost_sensitivity(strategy, market, window)
    metrics = {bps: compute_metrics(r, market, VERSIONS_TRIED) for bps, r in results.items()}
    base = results[DEFAULT_COST_BPS_PER_SIDE]
    base_metrics = metrics[DEFAULT_COST_BPS_PER_SIDE]
    stem = f"{slug(strategy.name)}_{window.label}_{window.start.year}_{window.end.year}"
    charts = [
        plot_equity_vs_benchmarks(
            benchmark_curves(base, market), base_metrics, folder / f"{stem}_equity.png"
        ),
        plot_cost_sensitivity(results, folder / f"{stem}_cost.png", versions_tried=VERSIONS_TRIED),
    ]
    full = full_equal_weight_stats(market, pd.DatetimeIndex(base.equity_curve.index))
    return StrategyRun(strategy, window, results, metrics, charts, full)


_FULL_EQUAL_WEIGHT: dict[tuple[str, str], ReturnStats | str] = {}


def full_equal_weight_stats(market: MarketData, days: pd.DatetimeIndex) -> ReturnStats | str | None:
    """Equal weight over every historical member, or the reason it was skipped.

    Only in sample mode: in full mode the normal equal-weight row already covers
    every member. Cached per window, since it reads hundreds of price files.
    """
    if universe_mode() == "full":
        return None
    key = (days[0].isoformat(), days[-1].isoformat())
    if key not in _FULL_EQUAL_WEIGHT:
        curve = full_universe_equal_weight(market.trading_days, days, PRICE_START)
        _FULL_EQUAL_WEIGHT[key] = curve if isinstance(curve, str) else return_stats(curve)
    return _FULL_EQUAL_WEIGHT[key]


def clip_window(window: BacktestWindow, clip: tuple[date, date] | None) -> BacktestWindow:
    """Replace the dates with a shorter span (tests only). The label stays."""
    if clip is None:
        return window
    return replace(window, start=clip[0], end=clip[1])


def run_all(market: MarketData, folder: Path, clip: tuple[date, date] | None) -> list[StrategyRun]:
    """Run every strategy in every window it is meant for."""
    runs: list[StrategyRun] = []
    for strategy, _ in summary_strategies():
        for window in backtest_windows(strategy):
            runs.append(run_one_window(strategy, market, clip_window(window, clip), folder))
    return runs


# ----------------------------------------------------------------- formatting


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    """A number as text, or n/a when there is none."""
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def stats_row(label: str, stats: ReturnStats) -> str:
    """A benchmark row of the metrics table."""
    cells = [
        label,
        "",
        fmt(stats.total_return_pct, 1, "%"),
        fmt(stats.cagr_pct, 1, "%"),
        fmt(stats.max_drawdown_pct, 1, "%"),
        fmt(stats.sharpe),
        "",
        "",
    ]
    return "| " + " | ".join(cells) + " |"


def metrics_row(bps: float, metrics: Metrics) -> str:
    """A strategy row of the metrics table for one cost level."""
    own = metrics.strategy_returns
    flag = f"YES, under {MIN_TRADES_TO_JUDGE}" if metrics.too_few_trades else "no"
    cells = [
        f"Strategy at {bps:g} bps",
        str(metrics.closed_trades),
        fmt(own.total_return_pct, 1, "%"),
        fmt(own.cagr_pct, 1, "%"),
        fmt(own.max_drawdown_pct, 1, "%"),
        fmt(own.sharpe),
        fmt(metrics.win_rate_pct, 1, "%"),
        flag,
    ]
    return "| " + " | ".join(cells) + " |"


def full_universe_name() -> str:
    """The label for equal weight over every historical member, with the count."""
    return f"Equal-weight, all {load_sample().population} (no costs)"


def equal_weight_rows(base: Metrics, full: ReturnStats | str | None) -> list[str]:
    """Equal weight on the run's own tickers, then on every member when it was computed."""
    if universe_mode() == "full":
        return [stats_row("Equal-weight, full universe (no costs)", base.equal_weight)]
    rows = [stats_row("Equal-weight, same sample (no costs)", base.equal_weight)]
    if isinstance(full, ReturnStats):
        rows.append(stats_row(full_universe_name(), full))
    else:
        reason = full if isinstance(full, str) else FULL_UNIVERSE_SKIPPED
        rows.append("| " + " | ".join([full_universe_name(), "", reason] + [""] * 5) + " |")
    return rows


def run_section(run: StrategyRun, folder: Path) -> list[str]:
    """Markdown for one strategy in one window."""
    window = run.window
    base = run.metrics[DEFAULT_COST_BPS_PER_SIDE]
    flag = f" (too few trades: under {MIN_TRADES_TO_JUDGE})" if base.too_few_trades else ""
    lines = [
        f"#### {run.strategy.name}: {window.label} window {window.start} to {window.end}{flag}",
        "",
        f"Versions tried: {VERSIONS_TRIED} (a scripted run, never tuned).",
        f"Universe: {universe_label()}",
    ]
    if window.coverage_warning:
        lines += ["", f"Footnote: {window.coverage_warning}"]
    lines += [
        "",
        "| Row | Closed trades | Total return | CAGR | Max drawdown | Sharpe | Win rate "
        "| Too few trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [metrics_row(bps, run.metrics[bps]) for bps in COST_SENSITIVITY_BPS]
    lines += [stats_row("SPY (no costs)", base.spy)]
    lines += equal_weight_rows(base, run.full_equal_weight)
    if base.too_few_trades:
        lines += ["", f"Too few trades to judge: fewer than {MIN_TRADES_TO_JUDGE} closed trades."]
    lines.append("")
    lines += [
        f"![{path.stem}]({path.relative_to(folder.parent).as_posix()})" for path in run.charts
    ]
    return lines + [""]


# ------------------------------------------------------------------- sections


def git_commit() -> str:
    """The short commit hash (read-only), or a note when git is not available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown (git not available)"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def header_lines() -> list[str]:
    """Title, banner and run settings."""
    mode = data_mode()
    lines = [f"# {report_title('Results summary')}", ""]
    if mode == "fake":
        lines += [
            f"> **{SIMULATED_LABEL}.** Every number below describes synthetic data made by the "
            "fake EODHD generator. These numbers are not findings about any real market.",
            "",
        ]
    costs = ", ".join(f"{bps:g}" for bps in COST_SENSITIVITY_BPS)
    lines += [f"**Universe: {universe_label()}**", ""]
    if universe_mode() == "sample":
        lines += [f"Limitations: {SAMPLE_LIMITATIONS}", ""]
    lines += [
        f"- Data mode: {mode}",
        f"- Date generated: {date.today().isoformat()}",
        f"- Test window: {BACKTEST_START} to {BACKTEST_END}; news strategies run "
        "2021-2023 only, with a footnote about 2019-2020 (D4)",
        f"- Sealed holdout: {HOLDOUT_START} onward, not used here",
        f"- Costs: {costs} bps per side; {DEFAULT_COST_BPS_PER_SIDE:g} bps is the default",
        "- Sharpe uses a risk-free rate of 0. Benchmarks include dividends and no costs.",
        "- Company news (D31): an article counts for a stock only if it is tagged on at most "
        "2 US symbols. This rule was chosen on coverage and article quality before any "
        "returns were seen.",
        f"- OPENCLAW_MODEL: {secret('OPENCLAW_MODEL') or 'not set'}",
        f"- Git commit: {git_commit()}",
        "",
    ]
    return lines


def strategies_lines(runs: list[StrategyRun], folder: Path) -> list[str]:
    """The strategies section: rules in plain English, then every window."""
    lines = ["## Strategies", ""]
    kinds = dict((s.name, kind) for s, kind in summary_strategies())
    seen: list[str] = []
    for run in runs:
        name = run.strategy.name
        if name not in seen:
            seen.append(name)
            lines += [
                f"### {name} ({kinds[name]})",
                "",
                run.strategy.description_plain_english,
                "",
            ]
        lines += run_section(run, folder)
    return lines


def holdout_rows() -> list[dict[str, Any]]:
    """Stored holdout runs, read with a plain SELECT. Empty when none exist."""
    path = output_dir() / "bot.sqlite"
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT strategy_id, window_start, window_end, cost_bps, metrics_json "
                "FROM backtest_runs WHERE kind = 'holdout' ORDER BY id"
            ).fetchall()
    except sqlite3.Error:
        return []
    return [{**dict(row), "metrics": json.loads(row["metrics_json"])} for row in rows]


def holdout_lines() -> list[str]:
    """The holdout section: stored results only, never a new run."""
    rows = holdout_rows()
    lines = ["## Holdout (2024 onward)", ""]
    if not rows:
        return lines + [SEALED_TEXT, ""]
    lines += [
        "| Strategy | Window | Cost (bps) | Total return | Max drawdown | Sharpe | Closed trades |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['strategy_id']} | {row['window_start']} to {row['window_end']} "
            f"| {row['cost_bps']:g} | {fmt(m.get('total_return_pct'), 1, '%')} "
            f"| {fmt(m.get('max_drawdown_pct'), 1, '%')} | {fmt(m.get('sharpe'))} "
            f"| {m.get('closed_trades', 'n/a')} |"
        )
    return lines + ["", "These are the stored results of the single allowed holdout run.", ""]


def load_eval_runs(model: str) -> list[dict[str, Any]]:
    """Eval results for the pinned model, oldest first, last few only."""
    runs: list[dict[str, Any]] = []
    for path in sorted((output_dir() / "evals").glob("*/results.json")):
        try:
            data = json.loads(path.read_text())
            header, accuracy = data["header"], data["accuracy"]
        except (OSError, ValueError, KeyError):
            continue
        if header.get("model") == model:
            runs.append({"folder": path.parent.name, "header": header, "accuracy": accuracy})
    return runs[-EVAL_RUNS_SHOWN:]


def eval_lines() -> list[str]:
    """Agent accuracy per category for each eval run of the pinned model."""
    model = secret("OPENCLAW_MODEL")
    lines = ["## Agent accuracy", "", f"Pinned model: {model or 'not set'}.", ""]
    runs = load_eval_runs(model) if model else []
    if not runs:
        return lines + ["No eval runs found for the pinned model.", ""]
    lines += [f"{DIRECT_MODE_NOTE}", ""]
    for run in runs:
        h = run["header"]
        lines.append(
            f"- Run {run['folder']}: model {h.get('model')}, API {h.get('api', 'not recorded')}, "
            f"effort {h.get('reasoning_effort', 'not recorded')}, mode {h.get('mode')}"
        )
    lines += ["", "Accuracy in percent (passed/total):", ""]
    lines += eval_table(runs)
    return lines + ["", "Fewer runs than the spread needs make the min and max less reliable.", ""]


def eval_table(runs: list[dict[str, Any]]) -> list[str]:
    """Category by run table with the min to max spread."""
    names = [run["folder"] for run in runs]
    lines = [
        "| Category | " + " | ".join(names) + " | Spread (min to max) |",
        "|---|" + "---|" * (len(names) + 1),
    ]
    categories = sorted({c for run in runs for c in run["accuracy"] if c != "TOTAL"}) + ["TOTAL"]
    for category in categories:
        cells: list[str] = []
        values: list[float] = []
        for run in runs:
            entry = run["accuracy"].get(category)
            if entry is None:
                cells.append("n/a")
                continue
            values.append(float(entry["accuracy_pct"]))
            cells.append(f"{entry['accuracy_pct']:.1f} ({entry['passed']}/{entry['total']})")
        spread = f"{min(values):.1f} to {max(values):.1f}" if values else "n/a"
        lines.append(f"| {category} | " + " | ".join(cells) + f" | {spread} |")
    return lines


def news_coverage_table(quality_text: str) -> list[str]:
    """The news coverage table (year, median, p10) copied from the data quality report."""
    rows = re.findall(r"^\| (\d{4}) \| (\d+) \| ([\d.]+) \| ([\d.]+) \|", quality_text, re.M)
    if not rows:
        return []
    lines = ["News articles per ticker per year (member tickers):", ""]
    lines += ["| Year | Member tickers | Median | p10 |", "|---|---|---|---|"]
    lines += [f"| {y} | {n} | {median} | {p10} |" for y, n, median, p10 in rows]
    return lines + [""]


def quality_lines() -> list[str]:
    """Link the data quality report and show the news coverage table when it exists."""
    report = output_dir() / "data_quality.md"
    lines = ["## Data quality", ""]
    if not report.exists():
        return lines + ["No data quality report found. Run the data quality script first.", ""]
    lines += [f"Full report: [data_quality.md]({report.name})", ""]
    return lines + news_coverage_table(report.read_text())


MEANING_LINES = [
    "## What these numbers do and do not show",
    "",
    "- A strategy that loses money is a valid result. Some rows above are negative and "
    "none of them were tuned to look better. Each strategy was run once.",
    "- Costs matter. The 0, 5 and 20 bps rows use the same rules, and returns usually "
    "fall as costs rise. Short holding periods pay costs more often.",
    f"- A result with fewer than {MIN_TRADES_TO_JUDGE} closed trades is flagged. Such a "
    "result is too thin to judge, whatever its return.",
    "- News coverage is too thin before 2021, so news strategies are tested on 2021-2023 "
    "only. 2019-2020 is a coverage footnote, not a result (D4).",
    "- Benchmarks include dividends and no costs, so they are slightly flattering.",
    "- In sample mode the fair benchmark is equal weight on the same sample. Equal weight "
    "on every historical member is shown only when all their prices are in the cache.",
    "- Past results do not predict future ones. The holdout stays sealed until a strategy "
    "is approved, so nothing here has been tested on it.",
    "- In fake mode the data is synthetic. The numbers show that the machinery runs, not "
    "how any strategy behaves in real markets. Live numbers need a live data run.",
    "",
]


ROBUSTNESS_TITLES = {
    "us_symbols_at_most_1": "Company news = tagged on at most 1 US symbol",
    "us_symbols_at_most_3": "Company news = tagged on at most 3 US symbols",
}


def robustness_metrics(
    markets: dict[str, MarketData], clip: tuple[date, date] | None
) -> dict[str, Metrics]:
    """The gap-down, no-news strategy at the default cost under each other news rule."""
    gap_down = strategy_from_json(GAP_DOWN_JSON)
    window = clip_window(backtest_windows(gap_down)[0], clip)
    out: dict[str, Metrics] = {}
    for name, market in markets.items():
        results = run_cost_sensitivity(gap_down, market, window)
        out[name] = compute_metrics(results[DEFAULT_COST_BPS_PER_SIDE], market, VERSIONS_TRIED)
    return out


def robustness_lines(metrics: dict[str, Metrics] | None) -> list[str]:
    """The robustness, secondary section: the same gap-down rules under the other news rules."""
    lines = [
        "## Robustness, secondary: other company-news rules",
        "",
        "The main results use the locked rule (at most 2 US symbols, D31). These rows run the "
        "gap-down, no-news strategy unchanged under the 1-symbol and 3-symbol rules. They are "
        "a check on how much the definition matters, not alternatives to pick from. The rule "
        "was chosen on coverage and article quality before any returns were seen.",
        "",
    ]
    if not metrics:
        return lines + ["Not run in this build.", ""]
    lines += [
        "| Row | Closed trades | Total return | CAGR | Max drawdown | Sharpe | Win rate "
        "| Too few trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, result in metrics.items():
        row = metrics_row(DEFAULT_COST_BPS_PER_SIDE, result)
        title = ROBUSTNESS_TITLES.get(name, name)
        lines.append(row.replace("| Strategy at ", f"| {title}, robustness, at ", 1))
    window = next(iter(metrics.values()))
    lines += ["", f"Window: {window.start} to {window.end}. Universe: {universe_label()}", ""]
    return lines


def build_summary(
    market: MarketData,
    clip: tuple[date, date] | None = None,
    robustness_markets: dict[str, MarketData] | None = None,
) -> Path:
    """Run everything, write results_summary.md and return its path.

    ``robustness_markets`` maps each name in ROBUSTNESS_NEWS_RULES to a market
    loaded with that news rule. Without it the robustness section says it was not run.
    """
    folder = output_dir() / "results"
    folder.mkdir(parents=True, exist_ok=True)
    runs = run_all(market, folder, clip)
    robust = robustness_metrics(robustness_markets, clip) if robustness_markets else None
    lines = header_lines() + strategies_lines(runs, folder) + robustness_lines(robust)
    lines += holdout_lines() + eval_lines() + quality_lines() + MEANING_LINES
    target = output_dir() / "results_summary.md"
    target.write_text("\n".join(lines) + "\n")
    return target


def main() -> None:
    """Load the market once, build the summary and print where it went."""
    market, _ = load_market_report(PRICE_START, BACKTEST_END)
    robustness = {
        name: load_market_report(PRICE_START, BACKTEST_END, news_rule=rule)[0]
        for name, rule in ROBUSTNESS_NEWS_RULES.items()
    }
    print(build_summary(market, robustness_markets=robustness))


if __name__ == "__main__":
    main()
