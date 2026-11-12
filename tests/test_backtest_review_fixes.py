"""Tests for the phase 2 and 3 review fixes (docs/reviews/phase_2_3.md): M1, M2, M3, L1, L2, L6, L7.

Toy markets are typed in by hand so every expected value can be worked out on
paper. The parity test at the end uses the planted fake world.
"""

import json
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bot.backtest import charts
from bot.backtest.costs import CostModel
from bot.backtest.engine import (
    DATA_ENDED_AFTER_MISSING_SESSIONS,
    BacktestResult,
    SkippedTradingDayError,
    close_out_at_end,
    initial_state,
    run_backtest,
    run_cost_sensitivity,
    state_from_json,
    state_to_json,
    step,
)
from bot.backtest.metrics import compute_metrics, equal_weight_benchmark
from bot.data.prices import RawEodRow, build_price_frame, invalid_price_rows
from bot.strategy.blocks import MarketData, slice_to_decision_date
from bot.strategy.schema import BacktestWindow, StrategyDefinition, strategy_from_json, strategy_id
from tests.backtest_helpers import (
    Bar,
    assemble_market,
    build_world_market,
    day_of,
    engine_market,
    fake_universe_codes,
    flat_bar,
    gap_up_strategy,
    load_raw_world,
    raw_frame,
    run_toy,
    toy_days,
)

NAN = float("nan")
NO_COST = CostModel(bps_per_side=0.0)

# Day 0 closes at 100, day 1 gaps up 5% (the signal), day 2 opens at 100 (the fill).
SIGNAL = {0: flat_bar(100.0), 1: Bar(105, 106, 104, 105), 2: Bar(100, 101, 99, 100)}


def bars_with(extra: dict[int, Bar]) -> dict[int, Bar]:
    return {**SIGNAL, **extra}


def flat_days(first: int, last: int, price: float = 100.0) -> dict[int, Bar]:
    return {n: flat_bar(price) for n in range(first, last + 1)}


def one_stock(bars: dict[int, Bar], count: int = 10) -> MarketData:
    return engine_market(count, {"Z.US": bars})


# ------------------------------------------------------------------------ M1


def eod_row(day: str, o: float, h: float, low: float, c: float) -> RawEodRow:
    return RawEodRow(date=day, open=o, high=h, low=low, close=c, adjusted_close=c, volume=1000.0)


def test_invalid_price_rows_lists_zero_negative_nan_and_inverted_rows() -> None:
    rows = [
        eod_row("2023-03-01", 100, 101, 99, 100),
        eod_row("2023-03-02", 0.0, 101, 99, 100),
        eod_row("2023-03-03", 100, 101, 0.0, 100),
        eod_row("2023-03-06", -1.0, 101, 99, 100),
        eod_row("2023-03-07", NAN, 101, 99, 100),
        eod_row("2023-03-08", 100, 98, 99, 100),
        eod_row("2023-03-09", 100, 101, 99, 0.0),
        eod_row("2023-03-10", 100, 100, 100, 100),
    ]
    assert invalid_price_rows(rows) == [
        date(2023, 3, 2),
        date(2023, 3, 3),
        date(2023, 3, 6),
        date(2023, 3, 7),
        date(2023, 3, 8),
        date(2023, 3, 9),
    ]
    frame = build_price_frame(rows, [], [])
    assert [stamp.date() for stamp in frame.index] == [date(2023, 3, 1), date(2023, 3, 10)]


def test_a_frame_of_only_bad_rows_is_empty() -> None:
    assert build_price_frame([eod_row("2023-03-01", 0.0, 1, 1, 1)], [], []).empty


def test_zero_open_on_the_sell_day_does_not_sell_at_zero() -> None:
    bars = bars_with({**flat_days(3, 7), 5: Bar(0.0, 101, 99, 100)})
    result = run_toy(gap_up_strategy(holding_days=3), one_stock(bars, 8))
    (trade,) = [t for t in result.trades if t.exit_reason == "holding_period"]
    # The bad row is a missing day. The sell waits and fills at day 6's open.
    assert trade.exit_price == 100.0
    assert trade.exit_date == day_of(one_stock(bars, 8), 6)
    assert trade.pnl == pytest.approx(0.0)


def test_zero_low_does_not_fire_a_stop() -> None:
    bars = bars_with({**flat_days(3, 7), 3: Bar(100, 101, 0.0, 100)})
    result = run_toy(gap_up_strategy(stop_loss_pct=5, holding_days=3), one_stock(bars, 8))
    assert "stop_loss" not in {t.exit_reason for t in result.trades}
    assert all(t.pnl >= 0 for t in result.trades)


def test_zero_open_on_the_signal_day_is_not_a_gap_down() -> None:
    bars = {0: flat_bar(100.0), 1: Bar(0.0, 106, 99, 100), **flat_days(2, 5)}
    strategy = strategy_from_json(
        json.dumps(
            {
                "name": "gd test",
                "description_plain_english": "gap down test strategy",
                "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 3}],
                "exit": {"holding_days": 2},
                "sizing": {"max_positions": 10},
            }
        )
    )
    assert run_toy(strategy, one_stock(bars, 6)).trades == []


# ------------------------------------------------------------------------ M2


def resumes_lower(price: float = 80.0, missing: tuple[int, ...] = (3,)) -> MarketData:
    bars = bars_with({n: flat_bar(100.0) for n in range(3, 9) if n not in missing})
    for n in range(max(missing) + 1, 9):
        bars[n] = flat_bar(price)
    return one_stock(bars, 9)


def test_one_missing_day_while_held_does_not_close_the_position() -> None:
    market = resumes_lower()
    result = run_toy(gap_up_strategy(holding_days=3), market)
    (trade,) = result.trades
    # Held from day 2. Day 3 is missing. Day 4 is the third held day, so the sell
    # is decided at its close and fills at day 5's open of 80: a real loss.
    assert trade.exit_reason == "holding_period"
    assert trade.exit_price == 80.0
    assert trade.exit_date == day_of(market, 5)
    assert trade.pnl == pytest.approx(-2000.0)
    assert not any("data_ended" in t.flags for t in result.trades)
    held = result.equity_curve["open_positions"]
    assert held.iloc[2] == held.iloc[3] == 1


def test_a_holding_period_sell_due_in_the_gap_fills_at_the_first_open_after_it() -> None:
    market = resumes_lower()
    result = run_toy(gap_up_strategy(holding_days=2), market)
    (trade,) = result.trades
    # Day 3 (missing) is the second held day, so the sell is decided there and
    # keeps that date until day 4's open of 80.
    assert trade.exit_signal_date == day_of(market, 3)
    assert trade.exit_date == day_of(market, 4)
    assert trade.exit_price == 80.0
    assert trade.pnl == pytest.approx(-2000.0)


def test_a_sell_waiting_through_a_longer_gap_leaves_a_note() -> None:
    market = resumes_lower(missing=(3, 4))
    result = run_toy(gap_up_strategy(holding_days=2), market)
    (trade,) = result.trades
    assert trade.exit_signal_date == day_of(market, 3)
    assert trade.exit_date == day_of(market, 5)
    assert any(f"{day_of(market, 4)}: sell of Z.US decided on" in n for n in result.notes)


def test_a_stop_due_in_the_gap_fills_at_the_gap_open() -> None:
    market = resumes_lower()
    result = run_toy(gap_up_strategy(stop_loss_pct=5), market)
    (trade,) = result.trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_date == day_of(market, 4)
    assert trade.exit_price == 80.0  # min(open 80, stop 95)
    assert "stop_gap_through" in trade.flags


def test_a_target_the_open_already_reached_fills_at_the_open_after_a_gap() -> None:
    market = resumes_lower(price=120.0)
    result = run_toy(gap_up_strategy(profit_target_pct=10), market)
    (trade,) = result.trades
    assert trade.exit_reason == "profit_target"
    assert trade.exit_date == day_of(market, 4)
    assert trade.exit_price == 120.0  # open 120 >= target 110, not the target itself
    assert "gap_fill" in trade.flags


def test_a_target_is_still_the_target_price_without_a_gap() -> None:
    bars = bars_with({3: Bar(101, 111, 100, 101), **flat_days(4, 6, 101.0)})
    (trade,) = run_toy(gap_up_strategy(profit_target_pct=10), one_stock(bars, 7)).trades
    assert trade.exit_price == pytest.approx(110.0)
    assert "gap_fill" not in trade.flags


def test_data_ended_only_after_the_limit_and_dated_the_first_missing_session() -> None:
    assert DATA_ENDED_AFTER_MISSING_SESSIONS == 5
    bars = bars_with({})  # rows on days 0 to 2 only
    market = one_stock(bars, 12)
    result = run_toy(gap_up_strategy(holding_days=20), market)
    (trade,) = result.trades
    assert trade.exit_reason == "data_ended"
    assert "data_ended" in trade.flags
    assert trade.exit_date == day_of(market, 3)  # first missing session
    assert trade.exit_price == 100.0  # last available close
    open_positions = list(result.equity_curve["open_positions"])
    # Missing on days 3 to 6 (4 sessions): still held. Day 7 is the fifth: closed.
    assert open_positions[2:7] == [1, 1, 1, 1, 1]
    assert open_positions[7] == 0
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(100_000.0)


def test_four_missing_sessions_then_trading_again_keeps_the_position() -> None:
    bars = bars_with({n: flat_bar(90.0) for n in range(7, 10)})
    result = run_toy(gap_up_strategy(holding_days=20), one_stock(bars, 10))
    (trade,) = result.trades
    assert trade.exit_reason == "open_at_end"
    assert trade.exit_price == 90.0


def test_open_position_in_a_gap_at_the_end_is_valued_at_the_last_close() -> None:
    result = run_toy(gap_up_strategy(holding_days=20), one_stock(bars_with({}), 5))
    (trade,) = result.trades
    assert trade.exit_reason == "open_at_end"
    assert trade.exit_price == 100.0


PLANTED_GAPS = [
    ("SIM020", date(2017, 3, 14)),
    ("SIM025", date(2019, 11, 6)),
    ("SIM033", date(2021, 8, 18)),
]


@pytest.mark.parametrize(("code", "gap_day"), PLANTED_GAPS)
def test_planted_price_gaps_do_not_end_a_held_position(code: str, gap_day: date) -> None:
    world = load_raw_world((code,))
    market = build_world_market(world)
    symbol = f"{code}.US"
    assert pd.isna(market.close.at[pd.Timestamp(gap_day), symbol])
    always = strategy_from_json(
        json.dumps(
            {
                "name": "Always buy",
                "description_plain_english": "Buy on any day with a return of at least -50%.",
                "entry": [
                    {
                        "block": "n_day_return",
                        "days": 1,
                        "comparison": "at_least",
                        "return_pct": -50,
                    }
                ],
                "exit": {"holding_days": 20},
                "sizing": {"max_positions": 1},
            }
        )
    )
    start = (pd.Timestamp(gap_day) - pd.Timedelta(days=45)).date()
    end = (pd.Timestamp(gap_day) + pd.Timedelta(days=45)).date()
    window = BacktestWindow("primary", start, end, None)
    result = run_backtest(always, market, window, NO_COST)
    assert not [t for t in result.trades if t.exit_reason == "data_ended"]
    across = [t for t in result.trades if t.entry_date < gap_day < t.exit_date]
    assert across, "expected a position held across the planted gap"
    assert across[0].exit_reason == "holding_period"


# ------------------------------------------------------------------------ M3


def split_market() -> MarketData:
    bars = {**SIGNAL, 3: flat_bar(100), 4: flat_bar(25), 5: flat_bar(25), 6: flat_bar(25)}
    return engine_market(8, {"SPL.US": bars}, splits={"SPL.US": {4: 4.0}})


def test_step_refuses_to_skip_a_trading_day() -> None:
    market = split_market()
    strategy = gap_up_strategy(holding_days=5)
    state = initial_state()
    for number in (0, 1, 2, 3):
        state = step(
            state,
            slice_to_decision_date(market, day_of(market, number)),
            strategy,
            NO_COST,
            allow_new_entries=True,
        ).state
    with pytest.raises(SkippedTradingDayError, match="not processed"):
        step(
            state,
            slice_to_decision_date(market, day_of(market, 5)),
            strategy,
            NO_COST,
            allow_new_entries=True,
        )
    assert isinstance(SkippedTradingDayError("x"), ValueError)


def test_replaying_both_days_keeps_equity_through_the_split() -> None:
    market = split_market()
    strategy = gap_up_strategy(holding_days=5)
    state = initial_state()
    equity = []
    for number in range(6):
        result = step(
            state,
            slice_to_decision_date(market, day_of(market, number)),
            strategy,
            NO_COST,
            allow_new_entries=True,
        )
        state = result.state
        equity.append(result.equity.equity)
    assert equity[4] == pytest.approx(100_000.0)
    assert equity[5] == pytest.approx(100_000.0)
    assert state.positions["SPL.US"].shares == pytest.approx(400.0)


def test_the_first_step_may_start_on_any_trading_day() -> None:
    market = split_market()
    data = slice_to_decision_date(market, day_of(market, 3))
    step(initial_state(), data, gap_up_strategy(holding_days=5), NO_COST, allow_new_entries=True)


# ------------------------------------------------------------------------ L1


def with_open(market: MarketData, symbol: str, number: int, value: float) -> MarketData:
    opens = market.open.copy()
    opens.iloc[number, opens.columns.get_loc(symbol)] = value
    return replace(market, open=opens)


def test_a_sell_with_a_nan_open_keeps_its_decision_date_and_fills_later() -> None:
    bars = {**SIGNAL, **flat_days(3, 8)}
    market = with_open(one_stock(bars, 9), "Z.US", 5, NAN)
    result = run_toy(gap_up_strategy(holding_days=3), market)
    (trade,) = result.trades
    # Bought day 2, third held day is 4, so the sell is decided at day 4's close.
    assert trade.exit_signal_date == day_of(market, 4)
    assert trade.exit_date == day_of(market, 6)
    assert trade.exit_reason == "holding_period"
    assert any(f"{day_of(market, 5)}: sell of Z.US" in note for note in result.notes)


def test_a_carried_sell_does_not_free_a_slot() -> None:
    nan_day = {**SIGNAL, **flat_days(3, 9)}
    other = {**flat_days(0, 4), 5: Bar(105, 106, 104, 105), **flat_days(6, 9)}
    market = engine_market(10, {"A.US": nan_day, "B.US": other})
    market = with_open(market, "A.US", 5, NAN)
    strategy = gap_up_strategy(holding_days=3, max_positions=1)
    result = run_toy(strategy, market)
    # B gaps up on day 5 while A's sell (decided day 4) could not fill. No slot is free.
    assert {t.symbol for t in result.trades} == {"A.US"}
    assert any("did not get a slot" in note for note in result.notes)


# ------------------------------------------------------------------------ L2


def test_equal_weight_keeps_the_move_across_a_missing_day() -> None:
    bars = {0: flat_bar(100.0), 1: flat_bar(100.0), 3: flat_bar(110.0), 4: flat_bar(110.0)}
    market = one_stock(bars, 5)
    curve = equal_weight_benchmark(market, market.trading_days)
    # The stock's own total_return on day 3 is 110 / 100 - 1, covering the missing day 2.
    assert curve.iloc[2] == pytest.approx(1.0)
    assert curve.iloc[3] == pytest.approx(1.10)
    assert curve.iloc[4] == pytest.approx(1.10)


def test_equal_weight_includes_dividends_across_a_gap() -> None:
    days = toy_days(5)
    frame = raw_frame(
        days,
        {0: flat_bar(100.0), 1: flat_bar(100.0), 3: flat_bar(100.0), 4: flat_bar(100.0)},
        dividends={3: 2.0},
    )
    market = assemble_market(days, {"D.US": frame})
    curve = equal_weight_benchmark(market, days)
    assert curve.iloc[3] == pytest.approx(1.02)


# ------------------------------------------------------------------------ L6, L7


def test_cost_sensitivity_footnote_lists_versions_tried_and_the_sell_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    market = one_stock(bars_with(flat_days(3, 7)), 8)
    days = market.trading_days
    window = BacktestWindow("primary", days[0].date(), days[7].date(), None)
    results = run_cost_sensitivity(gap_up_strategy(holding_days=3), market, window)
    seen: list[str] = []

    def record(figure: object, axis: object, footnote: str, path: Path) -> Path:
        seen.append(footnote)
        return path

    monkeypatch.setattr(charts, "_save", record)
    charts.plot_cost_sensitivity(results, tmp_path / "c.png", versions_tried=3)
    charts.plot_cost_sensitivity(results, tmp_path / "c.png")
    assert "Versions tried: 3." in seen[0]
    assert "net of one sell cost" in seen[0]
    assert "Versions tried: not recorded." in seen[1]


def test_metrics_start_is_the_first_trading_day_used() -> None:
    market = one_stock(bars_with(flat_days(3, 7)), 8)
    first = market.trading_days[0].date()
    window = BacktestWindow("primary", date(2023, 2, 20), market.trading_days[7].date(), None)
    result = run_backtest(gap_up_strategy(holding_days=3), market, window, NO_COST)
    metrics = compute_metrics(result, market, versions_tried=1)
    assert metrics.start == first
    assert window.start < metrics.start


# ------------------------------------------------------ parity on the planted world

MOMENTUM = strategy_from_json(
    json.dumps(
        {
            "name": "Momentum parity",
            "description_plain_english": "Momentum with a stop, used for the parity test.",
            "entry": [
                {"block": "n_day_return", "days": 20, "comparison": "at_least", "return_pct": 5},
                {"block": "rsi", "period": 14, "comparison": "at_most", "level": 70},
            ],
            "exit": {"holding_days": 20, "stop_loss_pct": 8},
            "sizing": {"max_positions": 20},
        }
    )
)
PARITY_START = date(2020, 3, 2)
PARITY_END = date(2020, 6, 30)
# The fake world has no stock that is held when its data ends for good, so the
# test cuts SIM057's rows from this date on. SIM057 is bought on 2020-03-23.
CUT_SYMBOL = "SIM057.US"
CUT_FROM = date(2020, 3, 30)


def cut_rows_from(market: MarketData, symbol: str, first_day: date) -> MarketData:
    """A copy of ``market`` where ``symbol`` has no rows from ``first_day`` on (a delisting)."""
    stamp = pd.Timestamp(first_day)
    tables = {}
    for name in (
        "open",
        "high",
        "low",
        "close",
        "raw_close",
        "dividend",
        "split_factor",
        "total_return_index",
    ):
        table = getattr(market, name).copy()
        table.loc[table.index >= stamp, symbol] = NAN
        tables[name] = table
    return replace(market, **tables)


def replay_every_day(
    market: MarketData, strategy: StrategyDefinition, window: BacktestWindow, costs: CostModel
) -> tuple[list, list[float]]:
    """Run ``step`` day by day, saving and loading the state as JSON each day."""
    days = market.trading_days[
        (market.trading_days >= pd.Timestamp(window.start))
        & (market.trading_days <= pd.Timestamp(window.end))
    ]
    state = initial_state()
    trades: list = []
    equity: list[float] = []
    data = slice_to_decision_date(market, days[0].date())
    for position, stamp in enumerate(days):
        data = slice_to_decision_date(market, stamp.date())
        last = position == len(days) - 1
        result = step(state, data, strategy, costs, allow_new_entries=not last)
        state = state_from_json(state_to_json(result.state))
        assert state == result.state
        trades.extend(result.closed_trades)
        equity.append(result.equity.equity)
    trades.extend(close_out_at_end(state, data, costs, strategy_id(strategy)))
    return trades, equity


def test_json_replay_equals_run_backtest_on_the_planted_world() -> None:
    world = load_raw_world(fake_universe_codes())
    market = cut_rows_from(build_world_market(world), CUT_SYMBOL, CUT_FROM)
    window = BacktestWindow("primary", PARITY_START, PARITY_END, None)
    costs = CostModel(bps_per_side=5.0)
    expected = run_backtest(MOMENTUM, market, window, costs)
    reasons = Counter(t.exit_reason for t in expected.trades)
    assert reasons["stop_loss"] > 0
    assert reasons["data_ended"] == 1
    assert any("split_while_held" in t.flags for t in expected.trades)
    trades, equity = replay_every_day(market, MOMENTUM, window, costs)
    assert trades == expected.trades
    curve = list(expected.equity_curve["equity"])
    # Every point is equal, the last one included: both paths report net
    # liquidation value every day, so there is no end-of-test adjustment.
    assert equity == curve
    assert curve[-1] == pytest.approx(100_000.0 + sum(t.pnl for t in expected.trades))


_ = (BacktestResult,)
