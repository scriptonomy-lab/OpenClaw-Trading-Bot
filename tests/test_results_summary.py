"""Tests for scripts/build_results_summary.py (fake mode, temporary output folder)."""

import importlib.util
import json
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

from bot import config
from bot.backtest.costs import make_cost_model
from bot.backtest.engine import run_backtest
from bot.backtest.metrics import compute_metrics
from bot.data.market import PRICE_START, load_market_report
from bot.strategy.blocks import MarketData
from bot.strategy.schema import BacktestWindow, backtest_windows, strategy_from_json

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "design" / "strategy_schema.md"
CLIP = (date(2022, 1, 3), date(2022, 6, 30))


def load_script() -> ModuleType:
    """Import the script by path."""
    spec = importlib.util.spec_from_file_location(
        "build_results_summary", ROOT / "scripts" / "build_results_summary.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_results_summary"] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


@pytest.fixture(autouse=True)
def fake_mode_tmp_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Fake mode, with outputs going to a temporary folder."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(script, "secret", lambda name: "openai/test-model")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path / "outputs" / "simulated"


@pytest.fixture(scope="module")
def market() -> MarketData:
    """One short market load shared by the tests that need it."""
    import os

    os.environ["EODHD_MODE"] = "fake"
    try:
        return load_market_report(PRICE_START, CLIP[1])[0]
    finally:
        os.environ.pop("EODHD_MODE", None)


def doc_json_blocks() -> list[str]:
    """The JSON blocks under 'Worked examples' in the schema doc."""
    text = DOC.read_text().split("## Worked examples", 1)[1]
    return re.findall(r"```json\n(.*?)```", text, re.S)


def test_worked_examples_match_the_doc() -> None:
    """The three strategies in the script equal the doc's JSON."""
    blocks = doc_json_blocks()
    assert len(blocks) == 3
    for in_script, in_doc in zip(script.WORKED_EXAMPLES, blocks, strict=True):
        assert json.loads(in_script) == json.loads(in_doc)
        strategy_from_json(in_script)


def test_summary_has_banner_and_matches_compute_metrics(market: MarketData, tmp_path: Path) -> None:
    """The banner is present and the gap-down 5 bps return equals compute_metrics."""
    target = script.build_summary(market, CLIP)
    text = target.read_text()
    assert "SIMULATED DATA" in text.splitlines()[0]
    assert "not findings" in text
    assert "No strategy has been approved yet; the holdout is sealed." in text
    assert "No eval runs found" in text
    gap_down = strategy_from_json(script.GAP_DOWN_JSON)
    window = backtest_windows(gap_down)[0]
    window = BacktestWindow(window.label, CLIP[0], CLIP[1], window.coverage_warning)
    result = run_backtest(gap_down, market, window, make_cost_model(5.0))
    expected = f"{compute_metrics(result, market, 1).strategy_returns.total_return_pct:.1f}%"
    section = text.split("#### Gap down, no news: primary", 1)[1].split("####", 1)[0]
    row = next(line for line in section.splitlines() if line.startswith("| Strategy at 5 bps"))
    assert row.split("|")[3].strip() == expected
    assert (tmp_path / "outputs" / "simulated" / "results").is_dir()


def test_summary_states_universe_limits_and_benchmarks(
    market: MarketData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D33: the sample sentence, the limitations note and both equal-weight rows.

    The labels follow EODHD_UNIVERSE, so sample mode is set here. The all-741
    benchmark is forced to the skipped case, whatever the local cache holds.
    """
    from bot.universe import SAMPLE_LIMITATIONS, universe_label

    monkeypatch.setenv("EODHD_UNIVERSE", "sample")
    monkeypatch.setattr(
        script, "full_universe_equal_weight", lambda *_: script.FULL_UNIVERSE_SKIPPED
    )
    script._FULL_EQUAL_WEIGHT.clear()
    text = script.build_summary(market, CLIP).read_text()
    label = universe_label("sample")
    assert label in text
    assert SAMPLE_LIMITATIONS in text
    assert text.count(f"Universe: {label}") >= 4
    assert "Equal-weight, same sample (no costs)" in text
    assert "Equal-weight, all 741 (no costs)" in text
    assert "skipped: full-universe prices not downloaded" in text
    assert "before any returns were seen" in text
    for heading in (line for line in text.splitlines() if line.startswith("#### ")):
        section = text.split(heading, 1)[1].split("####", 1)[0]
        if "| YES, under 30 |" in section.split("| Strategy at 5 bps", 1)[1].split("\n", 1)[0]:
            assert "too few trades: under 30" in heading


def test_full_mode_has_no_limitations_or_all_741_row(market: MarketData) -> None:
    from bot.universe import SAMPLE_LIMITATIONS, universe_label

    text = script.build_summary(market, CLIP).read_text()
    assert universe_label("full") in text
    assert SAMPLE_LIMITATIONS not in text
    assert "Equal-weight, full universe (no costs)" in text
    assert "all 741" not in text


def test_robustness_rows_are_labelled_secondary(market: MarketData) -> None:
    """The gap-down strategy under the other news rules is a labelled secondary section."""
    markets = {name: market for name in script.ROBUSTNESS_NEWS_RULES}
    text = script.build_summary(market, CLIP, robustness_markets=markets).read_text()
    section = text.split("## Robustness, secondary", 1)[1].split("\n## ", 1)[0]
    assert "at most 1 US symbol, robustness, at 5 bps" in section
    assert "at most 3 US symbols, robustness, at 5 bps" in section
    assert "before any returns were seen" in section
    plain = script.build_summary(market, CLIP).read_text()
    assert "Not run in this build." in plain


def test_four_strategies_and_windows_are_present(market: MarketData) -> None:
    """The no-news-filter comparison runs, and news strategies get one window."""
    text = script.build_summary(market, CLIP).read_text()
    assert "Gap down, no news filter" in text
    assert text.count("#### Gap down, no news:") == 1
    assert text.count("#### Positive news, heavy volume:") == 1
    assert text.count("#### Trend momentum:") == 1
    assert text.count("#### Gap down, no news filter:") == 1


def test_holdout_section_shows_stored_result(
    fake_mode_tmp_output: Path,
) -> None:
    """A stored holdout run is shown. With no database the text says sealed."""
    assert script.SEALED_TEXT in "\n".join(script.holdout_lines())
    fake_mode_tmp_output.mkdir(parents=True)
    with sqlite3.connect(fake_mode_tmp_output / "bot.sqlite") as db:
        db.executescript(
            "CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_id TEXT, kind TEXT,"
            " window_label TEXT, window_start TEXT, window_end TEXT, cost_bps REAL,"
            " metrics_json TEXT, chart_paths_json TEXT, data_mode TEXT, chat_id TEXT,"
            " created_at TEXT);"
        )
        db.execute(
            "INSERT INTO backtest_runs VALUES (1,'abc','holdout','holdout','2024-01-01',"
            "'2024-06-30',5.0,'{\"total_return_pct\": 1.5, \"closed_trades\": 4}','[]','fake',"
            "NULL,'now')"
        )
    text = "\n".join(script.holdout_lines())
    assert "abc" in text and "1.5%" in text
    assert script.SEALED_TEXT not in text


def test_eval_section_reads_runs_for_pinned_model(fake_mode_tmp_output: Path) -> None:
    """Runs for the pinned model appear with spread. Other models are left out."""
    for name, model, passed in (
        ("r1", "openai/test-model", 8),
        ("r2", "openai/test-model", 10),
        ("r3", "openai/other", 1),
    ):
        folder = fake_mode_tmp_output / "evals" / name
        folder.mkdir(parents=True)
        accuracy = {
            "clear": {"passed": passed, "total": 10, "accuracy_pct": passed * 10.0},
            "TOTAL": {"passed": passed, "total": 10, "accuracy_pct": passed * 10.0},
        }
        header = {"model": model, "mode": "direct"}
        (folder / "results.json").write_text(json.dumps({"header": header, "accuracy": accuracy}))
    text = "\n".join(script.eval_lines())
    assert "80.0 (8/10)" in text and "100.0 (10/10)" in text
    assert "80.0 to 100.0" in text
    assert "r3" not in text
    assert "approximates OpenClaw" in text
