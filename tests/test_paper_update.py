"""``paper_update`` itself, in the fake world with a temporary database and cache (Phase 5 review).

Covers the start day, catch-up against daily stepping, same-day repeats, parity
with ``run_backtest`` through the real update path (H1), and split handling (H3).
"""

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot import storage
from bot.backtest.costs import make_cost_model
from bot.backtest.engine import (
    OpenPosition,
    PortfolioState,
    run_backtest,
    state_from_json,
    state_to_json,
)
from bot.config import DEFAULT_COST_BPS_PER_SIDE
from bot.data import eodhd_client as client
from bot.data import fake_eodhd
from bot.data.holdout import paper_session
from bot.data.market import load_market_data
from bot.paper.trader import (
    PAPER_LOOKBACK_DAYS,
    UnexplainedPriceJumpError,
    paper_update,
)
from bot.strategy.schema import BacktestWindow, StrategyDefinition, strategy_from_json, strategy_id

START = date(2025, 7, 7)
END = date(2025, 7, 18)


def _strategy(entry: list[dict[str, Any]], holding_days: int) -> StrategyDefinition:
    return strategy_from_json(
        json.dumps(
            {
                "name": "paper test",
                "description_plain_english": "A strategy for the paper update tests.",
                "entry": entry,
                "exit": {"holding_days": holding_days},
                "sizing": {"max_positions": 10},
            }
        )
    )


GAP_DOWN = _strategy([{"block": "gap", "direction": "down", "min_gap_pct": 1.0}], 3)
EARNINGS = _strategy([{"block": "days_to_earnings", "relation": "within", "days": 3}], 4)


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One cache folder for the module, so the fake world is cached once."""
    return tmp_path_factory.mktemp("paper_cache")


@pytest.fixture(autouse=True)
def fake_world(tmp_path: Path, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fake mode, a shared temporary cache folder, and a database per test."""
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", cache_root / "fake")
    use_db(monkeypatch, tmp_path / "bot.sqlite")
    yield


def use_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point the storage module at one database file."""
    monkeypatch.setattr(storage, "db_path", lambda: path)


def register(strategy: StrategyDefinition, start_day: date) -> str:
    """Store the strategy, mark it approved and used, and record the paper start day."""
    sid = strategy_id(strategy)
    storage.save_strategy(sid, strategy.model_dump_json(), None)
    storage.record_approval(sid)
    storage.set_approval_used(sid)
    storage.start_paper_run(sid, start_day.isoformat())
    return sid


def trading_days_between(first: date, last: date) -> list[date]:
    """Weekdays in the fake calendar between two dates."""
    days = fake_eodhd.trading_days()
    return [d for d in days if first <= d <= last]


def ledger(sid: str) -> list[tuple[str, float, float]]:
    return [(r["day"], r["cash"], r["equity"]) for r in storage.paper_ledger(sid)]


def transport_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every request the fake transport answers."""
    seen: list[str] = []
    original = fake_eodhd.FakeEodhdTransport.get

    def counting(self: Any, url: str, params: dict[str, Any] | None = None, timeout: float = 0):  # type: ignore[no-untyped-def]
        seen.append(url)
        return original(self, url, params, timeout)

    monkeypatch.setattr(fake_eodhd.FakeEodhdTransport, "get", counting)
    return seen


def test_start_day_is_the_first_stepped_day() -> None:
    sid = register(GAP_DOWN, START)
    result = paper_update(sid, START)
    assert result.stepped_days == [START]
    assert [r["day"] for r in storage.paper_ledger(sid)] == [START.isoformat()]
    # Data before the start day is never stepped.
    assert result.last_date == START


def test_catch_up_equals_daily_stepping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for strategy in (GAP_DOWN, EARNINGS):
        sid = strategy_id(strategy)
        use_db(monkeypatch, tmp_path / f"daily_{sid}.sqlite")
        register(strategy, START)
        for day in trading_days_between(START, END):
            paper_update(sid, day)
        daily = (ledger(sid), storage.list_trades(sid, "paper"))
        use_db(monkeypatch, tmp_path / f"catchup_{sid}.sqlite")
        register(strategy, START)
        jumped = paper_update(sid, END)
        assert len(jumped.stepped_days) == len(trading_days_between(START, END))
        assert (ledger(sid), storage.list_trades(sid, "paper")) == daily
        # A cron that skips every other day also lands on the same ledger.
        use_db(monkeypatch, tmp_path / f"sparse_{sid}.sqlite")
        register(strategy, START)
        for day in trading_days_between(START, END)[::2] + [END]:
            paper_update(sid, day)
        assert (ledger(sid), storage.list_trades(sid, "paper")) == daily


def test_same_day_repeat_changes_nothing_and_calls_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = register(EARNINGS, START)
    paper_update(sid, END)
    before = (ledger(sid), storage.list_trades(sid, "paper"), storage.latest_paper_state(sid))
    seen = transport_calls(monkeypatch)
    again = paper_update(sid, END)
    assert again.stepped_days == []
    assert seen == []
    assert (
        ledger(sid),
        storage.list_trades(sid, "paper"),
        storage.latest_paper_state(sid),
    ) == before


def test_earnings_visibility_does_not_depend_on_cron_frequency() -> None:
    sid = register(EARNINGS, START)
    days = trading_days_between(START, END)
    qualifying_daily = 0
    qualifying_wide = 0
    with paper_session(sid):
        wide = load_market_data(
            START - timedelta(days=PAPER_LOOKBACK_DAYS), END + timedelta(days=60), approved=True
        )
        for day in days:
            daily = load_market_data(day - timedelta(days=PAPER_LOOKBACK_DAYS), day, approved=True)
            a = daily.days_to_next_earnings.loc[pd.Timestamp(day)]
            b = wide.days_to_next_earnings.loc[pd.Timestamp(day)].reindex(a.index)
            pd.testing.assert_series_equal(a, b, check_names=False)
            qualifying_daily += int((a <= 3).sum())
            qualifying_wide += int((b <= 3).sum())
    assert qualifying_daily == qualifying_wide > 0


@pytest.mark.parametrize(
    ("strategy", "last_day"),
    [(GAP_DOWN, END), (EARNINGS, date(2025, 7, 31))],
    ids=["gap_down", "earnings"],
)
def test_paper_replay_matches_backtest(strategy: StrategyDefinition, last_day: date) -> None:
    end = last_day
    sid = register(strategy, START)
    for day in trading_days_between(START, end):
        paper_update(sid, day)
    with paper_session(sid):
        market = load_market_data(
            START - timedelta(days=PAPER_LOOKBACK_DAYS), end + timedelta(days=60), approved=True
        )
        window = BacktestWindow("parity", START, end, None)
        result = run_backtest(
            strategy,
            market,
            window,
            make_cost_model(DEFAULT_COST_BPS_PER_SIDE),
            holdout_approved=True,
        )
    paper = storage.paper_ledger(sid)
    curve = result.equity_curve
    # The backtest values the last day net of closing costs and blocks entries on it.
    assert len(paper) == len(curve)
    for row, (_, expected) in zip(paper[:-1], curve.iloc[:-1].iterrows(), strict=True):
        assert row["equity"] == pytest.approx(float(expected["equity"]))
        assert row["open_positions"] == int(expected["open_positions"])
    closed = [t for t in result.trades if t.exit_reason != "open_at_end"]
    stored = storage.list_trades(sid, "paper")
    assert [(t["symbol"], t["entry_date"], t["exit_date"]) for t in stored] == [
        (t.symbol, t.entry_date.isoformat(), t.exit_date.isoformat()) for t in closed
    ]
    assert len(stored) > 0


# ------------------------------------------------------------------------ H3

SPLIT_STOCK = "SIM050.US"
SPLIT_DAY = date(2025, 2, 10)
BEFORE = date(2025, 2, 7)
NEVER = _strategy([{"block": "gap", "direction": "up", "min_gap_pct": 50}], 5)


def seed_held_position(sid: str) -> float:
    """Save a paper state that holds SIM050 the day before its 2-for-1 split. Returns the price."""
    with paper_session(sid):
        market = load_market_data(BEFORE - timedelta(days=30), BEFORE, approved=True)
    price = float(market.raw_close[SPLIT_STOCK].loc[pd.Timestamp(BEFORE)])
    position = OpenPosition(
        symbol=SPLIT_STOCK, signal_date=date(2025, 2, 5), entry_date=date(2025, 2, 6),
        entry_price=price, entry_shares=100.0, entry_cost=0.0, shares=100.0,
        stop_price=price * 0.9, target_price=None, days_held=1, dividends=0.0,
        last_close=price, last_close_date=BEFORE, signal_values={}, flags=(),
    )  # fmt: skip
    state = PortfolioState(
        cash=50_000.0, positions={SPLIT_STOCK: position}, pending=(), last_date=BEFORE
    )
    equity = {
        "cash": 50_000.0,
        "holdings_value": price * 100,
        "equity": 50_000.0 + price * 100,
        "open_positions": 1,
    }
    storage.save_paper_day(sid, BEFORE.isoformat(), state_to_json(state), equity, [])
    return price


def test_held_stock_split_keeps_value_and_fires_no_stop() -> None:
    sid = register(NEVER, date(2025, 2, 3))
    price = seed_held_position(sid)
    result = paper_update(sid, SPLIT_DAY)
    assert result.stepped_days == [SPLIT_DAY]
    assert result.closed_trades == 0
    latest = storage.latest_paper_state(sid)
    assert latest is not None
    held = state_from_json(latest[1]).positions[SPLIT_STOCK]
    assert held.shares == pytest.approx(200.0)
    assert result.equity is not None
    assert result.equity == pytest.approx(50_000.0 + price * 100, rel=0.08)


def test_missing_split_makes_the_update_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = register(NEVER, date(2025, 2, 3))
    seed_held_position(sid)
    real_splits = client.get_splits
    real_eod = client.get_eod_prices
    hide = {"on": True}

    def splits(ticker: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [] if hide["on"] and ticker == "SIM050" else real_splits(ticker, **kwargs)

    def eod(ticker: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = real_eod(ticker, *args, **kwargs)
        if hide["on"] and ticker == "SIM050":
            # The adjusted close does not show the split, so inference cannot find it.
            return [{**r, "adjusted_close": r["close"]} for r in rows]
        return rows

    monkeypatch.setattr(client, "get_splits", splits)
    monkeypatch.setattr(client, "get_eod_prices", eod)
    with pytest.raises(UnexplainedPriceJumpError):
        paper_update(sid, SPLIT_DAY)
    latest = storage.latest_paper_state(sid)
    assert latest is not None and latest[0] == BEFORE.isoformat()
    assert storage.list_trades(sid, "paper") == []
    # Once the split is listed, the next update steps the day normally.
    hide["on"] = False
    assert paper_update(sid, SPLIT_DAY).stepped_days == [SPLIT_DAY]
