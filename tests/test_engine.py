"""Engine rules on tiny hand-built markets.

Most scenarios share one shape. A stock closes at 100 on day 0, opens 5% higher
on day 1 (a gap up, so the signal fires at the close of day 1) and the fill is
at the open of day 2, at a raw price of 100. With 10 positions and 100,000 in
cash each buy is 10,000, which is exactly 100 shares. Tests change the bars
after the fill to trigger one rule at a time. Costs are 0 bps unless a test
says otherwise, so the arithmetic stays visible.
"""

import copy
import hashlib
from datetime import date

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import (
    INITIAL_CAPITAL,
    BacktestResult,
    PortfolioState,
    Trade,
    close_out_at_end,
    entry_priority,
    initial_state,
    raw_bars_today,
    run_backtest,
    run_cost_sensitivity,
    state_from_json,
    state_to_json,
    step,
)
from bot.data.holdout import HoldoutAccessError
from bot.data.membership import MembershipSpell
from bot.strategy.blocks import MarketData, slice_to_decision_date
from bot.strategy.schema import BacktestWindow, strategy_id
from tests.backtest_helpers import (
    Bar,
    day_of,
    engine_market,
    flat_bar,
    gap_up_strategy,
    round_trip_market,
    run_toy,
    spell_membership,
    toy_days,
)

HOLD = Bar(100.0, 101.0, 99.0, 100.0)


def template(overrides: dict[int, Bar] | None = None, count: int = 10) -> dict[int, Bar]:
    """Signal on day 1, fill on day 2 at 100, then quiet days at 100.

    Days listed in ``overrides`` replace the template. A day after the first
    override that is not listed stays flat at the previous close, so a test
    never creates a second gap by accident.
    """
    overrides = overrides or {}
    bars = {0: flat_bar(100.0), 1: Bar(105.0, 106.0, 104.0, 105.0)}
    for number in range(2, count):
        if number in overrides:
            bars[number] = overrides[number]
        elif overrides and number > min(overrides):
            bars[number] = flat_bar(bars[number - 1].close)
        else:
            bars[number] = HOLD
    return bars


def one_stock(
    overrides: dict[int, Bar] | None = None,
    *,
    count: int = 10,
    dividends: dict[int, float] | None = None,
) -> MarketData:
    return engine_market(
        count,
        {"AAA.US": template(overrides, count)},
        dividends={"AAA.US": dividends} if dividends else None,
    )


def check_pnl_identity(trade: Trade) -> None:
    """pnl and return_pct follow the formulas in the Trade docstring."""
    cost_basis = trade.entry_price * trade.entry_shares + trade.entry_cost
    pnl = trade.exit_price * trade.exit_shares - trade.exit_cost + trade.dividends - cost_basis
    assert trade.pnl == pytest.approx(pnl, abs=1e-6)
    assert trade.return_pct == pytest.approx(pnl / cost_basis * 100, abs=1e-6)


# --------------------------------------------------------------- fills


def test_signal_at_the_close_fills_at_the_next_open() -> None:
    market = one_stock()
    result = run_toy(gap_up_strategy(holding_days=3), market)
    (trade,) = result.trades
    assert trade.symbol == "AAA.US"
    assert trade.signal_date == day_of(market, 1)
    assert trade.entry_date == day_of(market, 2)  # the next trading day, not the signal day
    assert trade.entry_price == 100.0  # day 2's raw open, not day 1's open (105) or close (105)
    assert trade.entry_shares == pytest.approx(100.0)  # 10,000 / 100
    assert trade.signal_values["gap_pct"] == pytest.approx(5.0)
    check_pnl_identity(trade)


def test_no_same_day_fill_when_the_next_open_is_very_different() -> None:
    # Day 2 opens at 80. The fill must use 80 and size the shares from 10,000 / 80 = 125.
    market = one_stock({2: Bar(80.0, 81.0, 79.0, 80.0), 3: flat_bar(80.0)})
    (trade,) = run_toy(gap_up_strategy(holding_days=3), market).trades
    assert (trade.entry_price, trade.entry_shares) == (80.0, pytest.approx(125.0))


def test_a_stock_is_not_bought_again_while_held() -> None:
    # Day 3 gaps up 9% again while the stock is held.
    market = one_stock({3: Bar(109.0, 110.0, 108.0, 109.0), 4: flat_bar(109.0)})
    result = run_toy(gap_up_strategy(holding_days=5), market)
    assert len(result.trades) == 1


def test_orders_are_not_placed_before_the_window_starts() -> None:
    market = one_stock()
    # The signal is the close of day 1. A window starting on day 2 has no decision before it.
    assert run_toy(gap_up_strategy(holding_days=3), market, first_day=2).trades == []
    assert len(run_toy(gap_up_strategy(holding_days=3), market, first_day=1).trades) == 1


# ------------------------------------------------------- holding periods


@pytest.mark.parametrize("holding_days", [1, 2, 3, 5])
def test_holding_period_sells_at_the_open_of_fill_day_plus_n(holding_days: int) -> None:
    market = one_stock({n: Bar(100.0 + n, 120.0, 90.0, 100.0 + n) for n in range(3, 10)})
    (trade,) = run_toy(gap_up_strategy(holding_days=holding_days), market).trades
    fill_day = 2
    sell_day = fill_day + holding_days  # bought at the open of F, sold at the open of F + N
    assert trade.exit_reason == "holding_period"
    assert trade.entry_date == day_of(market, fill_day)
    assert trade.exit_date == day_of(market, sell_day)
    assert trade.exit_signal_date == day_of(market, sell_day - 1)
    assert trade.exit_price == 100.0 + sell_day  # the raw open of the sell day
    assert trade.trading_days_held == holding_days
    check_pnl_identity(trade)


# ----------------------------------------------------------- stops and targets


def test_stop_fills_at_the_stop_price_intraday() -> None:
    # Stop 10% below 100 is 90. Day 3 opens at 99 and trades down to 89.
    market = one_stock({3: Bar(99.0, 100.0, 89.0, 95.0), 4: flat_bar(95.0)})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10), market).trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_date == day_of(market, 3)
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.exit_signal_date is None  # it happened during the day
    assert trade.trading_days_held == 2  # held on days 2 and 3
    assert trade.pnl == pytest.approx(-1_000.0)
    assert trade.return_pct == pytest.approx(-10.0)
    check_pnl_identity(trade)


def test_stop_is_hit_when_the_low_equals_the_stop() -> None:
    touched = one_stock({3: Bar(99.0, 100.0, 90.0, 95.0), 4: flat_bar(95.0)})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10), touched).trades
    assert trade.exit_reason == "stop_loss"
    missed = one_stock({3: Bar(99.0, 100.0, 90.01, 95.0), 4: flat_bar(95.0)})
    (still_open,) = run_toy(gap_up_strategy(stop_loss_pct=10), missed).trades
    assert still_open.exit_reason == "open_at_end"


def test_stop_gap_through_fills_at_the_worse_open() -> None:
    market = one_stock({3: Bar(85.0, 88.0, 84.0, 86.0), 4: flat_bar(86.0)})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10), market).trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == 85.0  # the open, below the stop at 90
    assert "stop_gap_through" in trade.flags
    assert trade.pnl == pytest.approx(-1_500.0)


def test_target_fills_at_the_target_price() -> None:
    # Target 20% above 100 is 120. Day 3 trades up to 121.
    market = one_stock({3: Bar(101.0, 121.0, 100.0, 118.0), 4: flat_bar(118.0)})
    (trade,) = run_toy(gap_up_strategy(profit_target_pct=20), market).trades
    assert trade.exit_reason == "profit_target"
    assert trade.exit_date == day_of(market, 3)
    assert trade.exit_price == pytest.approx(120.0)
    assert trade.pnl == pytest.approx(2_000.0)


def test_target_never_fills_above_the_target_even_after_a_gap_up() -> None:
    # A 30% gap on day 1 makes the signal. Day 3 opens at 125, above the target of 120.
    market = engine_market(
        8,
        {
            "AAA.US": {
                0: flat_bar(100.0),
                1: Bar(130.0, 131.0, 129.0, 130.0),
                2: Bar(100.0, 101.0, 99.0, 110.0),
                3: Bar(125.0, 130.0, 124.0, 128.0),
                **{n: flat_bar(128.0) for n in range(4, 8)},
            }
        },
    )
    strategy = gap_up_strategy(profit_target_pct=20, min_gap_pct=20)
    (trade,) = run_toy(strategy, market).trades
    assert trade.exit_reason == "profit_target"
    assert trade.exit_price == pytest.approx(120.0)  # not the better open of 125


def test_stop_wins_when_both_are_touched_on_the_same_day() -> None:
    market = one_stock({3: Bar(100.0, 125.0, 85.0, 100.0), 4: HOLD})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10, profit_target_pct=20), market).trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(90.0)


def test_stop_and_target_are_checked_on_the_fill_day() -> None:
    stopped = one_stock({2: Bar(100.0, 101.0, 88.0, 95.0), 3: flat_bar(95.0)})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10), stopped).trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_date == trade.entry_date == day_of(stopped, 2)
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.trading_days_held == 1  # the fill day counts as day 1
    hit = one_stock({2: Bar(100.0, 125.0, 99.0, 110.0), 3: flat_bar(110.0)})
    (target,) = run_toy(gap_up_strategy(profit_target_pct=20), hit).trades
    assert (target.exit_reason, target.exit_date) == ("profit_target", day_of(hit, 2))
    assert target.exit_price == pytest.approx(120.0)


def test_first_rule_hit_wins_between_holding_period_and_stop() -> None:
    # holding_days 3 would sell at the open of day 5, but the stop is hit on day 3.
    market = one_stock({3: Bar(99.0, 100.0, 89.0, 95.0), 4: flat_bar(95.0)})
    (trade,) = run_toy(gap_up_strategy(holding_days=3, stop_loss_pct=10), market).trades
    assert trade.exit_reason == "stop_loss"


# --------------------------------------------------------------- data ended


def test_data_ended_closes_at_the_last_close() -> None:
    # The stock has rows up to day 4 (close 97) and none after.
    bars = {n: HOLD for n in range(2, 5)} | {4: Bar(98.0, 99.0, 96.0, 97.0)}
    stock = {0: flat_bar(100.0), 1: Bar(105.0, 106.0, 104.0, 105.0)} | bars
    market = engine_market(10, {"AAA.US": stock, "BBB.US": {n: flat_bar(50.0) for n in range(10)}})
    result = run_toy(gap_up_strategy(holding_days=10), market)
    (trade,) = result.trades
    assert trade.exit_reason == "data_ended"
    assert "data_ended" in trade.flags
    assert trade.exit_price == 97.0
    assert trade.exit_shares == pytest.approx(100.0)
    assert trade.exit_date == day_of(market, 5)  # the first day with no row
    assert trade.pnl == pytest.approx(97 * 100 - 10_000)
    check_pnl_identity(trade)
    # Nothing is left open, and the cash is back in the portfolio.
    assert result.equity_curve["open_positions"].iloc[-1] == 0
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(INITIAL_CAPITAL - 300.0)


# ---------------------------------------------------------- end of the test


def test_no_entry_on_the_last_day_of_the_window() -> None:
    # The signal is the close of day 1. If day 1 is the window's last day, the fill would be
    # outside the window, so nothing is bought even though the market has more days.
    market = one_stock()
    result = run_toy(gap_up_strategy(holding_days=3), market, last_day=1)
    assert result.trades == []
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(INITIAL_CAPITAL)


def test_open_at_end_is_valued_net_of_one_sell_cost() -> None:
    # Window ends on day 4. Bought day 2 at 100 (100 shares, entry cost 5.0 at 5 bps).
    # Day 4 closes at 107: value 10,700, sell cost 10,700 * 5 / 10,000 = 5.35.
    # pnl = 10,700 - 5.35 - (10,000 + 5.0) = 689.65
    market = round_trip_market()
    result = run_toy(gap_up_strategy(holding_days=3), market, last_day=4, bps=5.0)
    (trade,) = result.trades
    assert trade.exit_reason == "open_at_end"  # the sell that would fill on day 5 is not filled
    assert trade.exit_date == day_of(market, 4)
    assert trade.exit_price == 107.0
    assert trade.exit_cost == pytest.approx(5.35)
    assert trade.entry_cost == pytest.approx(5.0)
    assert trade.pnl == pytest.approx(689.65)
    assert trade.return_pct == pytest.approx(689.65 / 10_005.0 * 100)
    check_pnl_identity(trade)


def test_a_buy_filling_on_the_last_day_is_valued_open_at_end() -> None:
    market = one_stock()
    (trade,) = run_toy(gap_up_strategy(holding_days=3), market, last_day=2, bps=5.0).trades
    # Bought at 100 on day 2 and valued at that day's close of 100: only the two costs are lost.
    assert trade.exit_reason == "open_at_end"
    assert trade.pnl == pytest.approx(-10.0)  # 10,000 - 5 - (10,000 + 5)


def test_close_out_at_end_drops_pending_orders() -> None:
    market = round_trip_market()
    strategy = gap_up_strategy(holding_days=3)
    costs = CostModel(0.0)
    state = initial_state()
    for number in range(5):
        data = slice_to_decision_date(market, day_of(market, number))
        state = step(state, data, strategy, costs, allow_new_entries=True).state
    assert any(order.side == "sell" for order in state.pending)  # decided at the close of day 4
    trades = close_out_at_end(state, data, costs, strategy_id(strategy))
    assert [t.exit_reason for t in trades] == ["open_at_end"]


# ---------------------------------------------------------------- dividends


def test_dividend_with_ex_date_while_held_is_credited() -> None:
    # Ex-date on day 4: shares held at the close of day 3 get 0.5 * 100 = 50.
    market = one_stock(dividends={4: 0.5})
    result = run_toy(gap_up_strategy(holding_days=3), market)
    (trade,) = result.trades
    assert trade.dividends == pytest.approx(50.0)
    assert trade.pnl == pytest.approx(50.0)  # exit and entry both at 100
    check_pnl_identity(trade)
    curve = result.equity_curve["equity"]
    assert curve.iloc[3] == pytest.approx(100_000.0)
    assert curve.iloc[4] == pytest.approx(100_050.0)  # the cash arrived on day 4


def test_dividend_on_the_buy_day_is_not_received() -> None:
    market = one_stock(dividends={2: 0.5})  # bought at the open of the ex-date
    (trade,) = run_toy(gap_up_strategy(holding_days=3), market).trades
    assert trade.dividends == 0.0


def test_dividend_on_the_sell_day_is_still_received() -> None:
    market = one_stock(dividends={5: 0.5})  # sold at the open of the ex-date
    (trade,) = run_toy(gap_up_strategy(holding_days=3), market).trades
    assert trade.exit_date == day_of(market, 5)
    assert trade.dividends == pytest.approx(50.0)


def test_dividend_before_the_buy_is_not_received() -> None:
    market = one_stock(dividends={1: 0.5})
    (trade,) = run_toy(gap_up_strategy(holding_days=3), market).trades
    assert trade.dividends == 0.0


# -------------------------------------------------------------------- splits


def split_market(extra: dict[int, Bar] | None = None) -> MarketData:
    """A 4-for-1 split with ex-date day 4. Raw prices fall from about 100 to about 25."""
    bars = {
        0: flat_bar(100.0),
        1: Bar(105.0, 106.0, 104.0, 105.0),
        2: HOLD,
        3: HOLD,
        4: Bar(25.0, 25.5, 24.5, 25.0),
        5: Bar(25.5, 26.5, 25.0, 26.0),
    }
    bars |= extra or {}
    for number in range(6, 10):
        bars.setdefault(number, flat_bar(bars[number - 1].close))
    return engine_market(10, {"SPL.US": bars}, splits={"SPL.US": {4: 4.0}})


def test_split_during_a_hold_changes_shares_not_value() -> None:
    market = split_market()
    result = run_toy(gap_up_strategy(holding_days=3), market)
    (trade,) = result.trades
    assert trade.entry_price == 100.0  # raw, as traded, not the adjusted 25
    assert trade.entry_shares == pytest.approx(100.0)
    assert trade.exit_shares == pytest.approx(400.0)
    assert trade.exit_price == 25.5  # raw open of day 5
    assert "split_while_held" in trade.flags
    assert trade.pnl == pytest.approx(25.5 * 400 - 10_000)
    assert trade.return_pct == pytest.approx(2.0)
    check_pnl_identity(trade)
    equity = result.equity_curve["equity"]
    assert equity.iloc[3] == pytest.approx(100_000.0)  # 90,000 cash + 100 shares at 100
    assert equity.iloc[4] == pytest.approx(100_000.0)  # 90,000 cash + 400 shares at 25


def test_stop_price_is_divided_by_the_split_factor() -> None:
    # Stop 10% below 100 is 90 raw, which is 22.5 after the split. Day 4 trades down to 24.5,
    # which is above 22.5 but far below 90, so an unadjusted stop would fire on day 4.
    market = split_market({5: Bar(24.0, 24.5, 22.0, 23.0)})
    (trade,) = run_toy(gap_up_strategy(stop_loss_pct=10), market).trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_date == day_of(market, 5)
    assert trade.exit_price == pytest.approx(22.5)
    assert trade.exit_shares == pytest.approx(400.0)
    assert trade.pnl == pytest.approx(22.5 * 400 - 10_000)


def test_raw_bars_today_recovers_raw_prices() -> None:
    market = split_market()
    before = raw_bars_today(slice_to_decision_date(market, day_of(market, 3)), ["SPL.US"])
    # The adjusted close on day 3 is 25, but the raw close that day was 100.
    assert before.loc["SPL.US", "open"] == pytest.approx(100.0)
    assert before.loc["SPL.US", "close"] == pytest.approx(100.0)
    on_split = raw_bars_today(slice_to_decision_date(market, day_of(market, 4)), ["SPL.US"])
    assert on_split.loc["SPL.US", "open"] == pytest.approx(25.0)
    assert on_split.loc["SPL.US", "split_factor"] == pytest.approx(4.0)
    missing = raw_bars_today(slice_to_decision_date(market, day_of(market, 4)), ["NONE.US"])
    assert missing.loc["NONE.US"].isna().all()


# --------------------------------------------------------------- membership


def test_leaving_the_index_does_not_force_an_exit_but_blocks_new_entries() -> None:
    days = toy_days(12)
    bars = template({6: Bar(110.0, 111.0, 109.0, 110.0), 7: flat_bar(110.0)}, 12)
    for n in range(8, 12):
        bars[n] = flat_bar(110.0)
    spells = [MembershipSpell("AAA.US", date(2023, 1, 1), days[3].date())]  # a member to day 3
    market = engine_market(12, {"AAA.US": bars}, membership=spell_membership(days, spells))
    result = run_toy(gap_up_strategy(holding_days=5), market)
    # Bought on day 2 while a member. Day 6 gaps up 10% but the stock is no longer a member, so
    # it is not bought again, and the original hold runs its full 5 days to the open of day 7.
    (trade,) = result.trades
    assert trade.exit_reason == "holding_period"
    assert trade.exit_date == day_of(market, 7)


# ------------------------------------------------------------------- sizing


def two_stocks_signalling_together() -> MarketData:
    return engine_market(
        10,
        {
            "AAA.US": template(),
            "BBB.US": {
                0: flat_bar(50.0),
                1: Bar(52.5, 53.0, 52.0, 52.5),
                **{n: Bar(50.0, 50.5, 49.5, 50.0) for n in range(2, 10)},
            },
        },
    )


def test_each_buy_is_equity_over_max_positions() -> None:
    market = two_stocks_signalling_together()
    result = run_toy(gap_up_strategy(holding_days=8, max_positions=4), market)
    by_symbol = {t.symbol: t for t in result.trades}
    # 100,000 / 4 = 25,000 each: 250 shares at 100 and 500 shares at 50.
    assert by_symbol["AAA.US"].entry_shares == pytest.approx(250.0)
    assert by_symbol["BBB.US"].entry_shares == pytest.approx(500.0)


def test_a_later_buy_uses_the_equity_at_its_decision_close() -> None:
    # AAA is bought on day 2 (250 shares at 100) and closes day 3 at 104. BBB signals at the
    # close of day 3: equity = 75,000 cash + 250 * 104 = 101,000, so the target is 25,250.
    # BBB fills at 50 on day 4: 505 shares.
    aaa = template(
        {3: Bar(100.0, 105.0, 99.0, 104.0), **{n: flat_bar(104.0) for n in range(4, 10)}}
    )
    bbb = {
        **{n: flat_bar(50.0) for n in range(0, 3)},
        3: Bar(52.5, 53.0, 52.0, 52.5),
        **{n: Bar(50.0, 50.5, 49.5, 50.0) for n in range(4, 10)},
    }
    market = engine_market(10, {"AAA.US": aaa, "BBB.US": bbb})
    result = run_toy(gap_up_strategy(holding_days=8, max_positions=4), market)
    by_symbol = {t.symbol: t for t in result.trades}
    assert by_symbol["BBB.US"].entry_date == day_of(market, 4)
    assert by_symbol["BBB.US"].entry_shares == pytest.approx(505.0)


def test_a_buy_is_limited_by_cash_and_leaves_room_for_the_cost() -> None:
    # One position: the target is all 100,000. At 5 bps the notional is 100,000 / 1.0005.
    result = run_toy(gap_up_strategy(holding_days=5, max_positions=1), one_stock(), bps=5.0)
    (trade,) = result.trades
    notional = 100_000.0 / 1.0005
    assert trade.entry_shares * trade.entry_price == pytest.approx(notional)
    assert trade.entry_cost == pytest.approx(notional * 0.0005)
    cash_after_fill = result.equity_curve["cash"].iloc[2]
    assert -1e-6 <= cash_after_fill < 1e-6  # never negative: no leverage


def sell_funds_buy_market() -> MarketData:
    """AAA is held when BBB signals. AAA's sale at the open of day 5 pays for BBB."""
    aaa = {
        0: flat_bar(100.0),
        1: Bar(105.0, 106.0, 104.0, 105.0),
        2: HOLD,
        3: Bar(100.0, 115.0, 100.0, 112.0),
        4: Bar(113.0, 121.0, 112.0, 120.0),
        **{n: Bar(110.0, 111.0, 109.0, 110.0) for n in range(5, 10)},
    }
    bbb = {
        **{n: flat_bar(50.0) for n in range(4)},
        4: Bar(52.5, 53.0, 52.0, 52.5),
        **{n: Bar(50.0, 51.0, 49.0, 50.0) for n in range(5, 10)},
    }
    return engine_market(10, {"AAA.US": aaa, "BBB.US": bbb})


def test_sells_fill_before_buys_and_cash_limits_the_buy() -> None:
    market = sell_funds_buy_market()
    result = run_toy(gap_up_strategy(holding_days=3, max_positions=1), market, last_day=6)
    by_symbol = {t.symbol: t for t in result.trades}
    # AAA: 1,000 shares bought at 100, sold at the day 5 open of 110 for 110,000.
    assert by_symbol["AAA.US"].entry_shares == pytest.approx(1_000.0)
    assert by_symbol["AAA.US"].exit_date == day_of(market, 5)
    assert by_symbol["AAA.US"].exit_price == 110.0
    # BBB targets the day 4 close equity of 120,000 but only 110,000 of cash is there.
    # 110,000 / 50 = 2,200 shares, bought at the same open as the sale.
    bbb = by_symbol["BBB.US"]
    assert bbb.entry_date == day_of(market, 5)
    assert bbb.entry_shares == pytest.approx(2_200.0)
    assert bbb.entry_price == 50.0
    assert result.equity_curve["cash"].iloc[5] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- tie-break


def reference_order(sid: str, day: date, symbols: list[str]) -> list[str]:
    """The tie-break order from the spec: sha256 of "strategy_id|date|symbol", ascending."""
    return sorted(symbols, key=lambda s: hashlib.sha256(f"{sid}|{day}|{s}".encode()).hexdigest())


SYMBOLS = [f"S{n:02d}.US" for n in range(10)]


def test_entry_priority_is_the_documented_hash_order() -> None:
    day = date(2023, 3, 2)
    assert entry_priority("strat_x", day, SYMBOLS) == reference_order("strat_x", day, SYMBOLS)


def test_entry_priority_is_repeatable_and_ignores_input_order() -> None:
    day = date(2023, 3, 2)
    first = entry_priority("strat_x", day, SYMBOLS)
    assert entry_priority("strat_x", day, list(reversed(SYMBOLS))) == first
    assert entry_priority("strat_x", day, SYMBOLS) == first
    assert sorted(first) == SYMBOLS  # a permutation: nothing lost or added


def test_entry_priority_is_not_alphabetical_and_changes_with_the_day_and_strategy() -> None:
    day = date(2023, 3, 2)
    order = entry_priority("strat_x", day, SYMBOLS)
    assert order != SYMBOLS  # sorting by ticker would favour early letters
    assert entry_priority("strat_x", date(2023, 3, 3), SYMBOLS) != order
    assert entry_priority("strat_y", day, SYMBOLS) != order


def test_when_slots_run_out_the_hash_order_decides() -> None:
    names = ["XXX.US", "YYY.US", "ZZZ.US"]
    market = engine_market(8, {name: template(count=8) for name in names})
    strategy = gap_up_strategy(holding_days=5, max_positions=1)
    result = run_toy(strategy, market)
    expected = reference_order(strategy_id(strategy), day_of(market, 1), names)
    assert [t.symbol for t in result.trades] == [expected[0]]
    two = run_toy(gap_up_strategy(holding_days=5, max_positions=2), market)
    assert {t.symbol for t in two.trades} == set(expected[:2])


# ------------------------------------------------------------ the day step


def play(market: MarketData, strategy_obj, costs: CostModel, days: int) -> list[PortfolioState]:  # type: ignore[no-untyped-def]
    """The state after each of the first ``days`` days, stepping by hand."""
    states = []
    state = initial_state()
    for number in range(days):
        data = slice_to_decision_date(market, day_of(market, number))
        state = step(state, data, strategy_obj, costs, allow_new_entries=True).state
        states.append(state)
    return states


def test_step_does_not_change_the_state_it_was_given() -> None:
    market = round_trip_market()
    strategy = gap_up_strategy(holding_days=3)
    costs = CostModel(5.0)
    states = play(market, strategy, costs, 4)
    before = copy.deepcopy(states[3])
    data = slice_to_decision_date(market, day_of(market, 4))
    step(states[3], data, strategy, costs, allow_new_entries=True)
    assert states[3] == before


def test_step_requires_a_later_day_than_the_state() -> None:
    market = round_trip_market()
    strategy = gap_up_strategy(holding_days=3)
    states = play(market, strategy, CostModel(0.0), 3)
    data = slice_to_decision_date(market, day_of(market, 2))
    with pytest.raises(ValueError):
        step(states[2], data, strategy, CostModel(0.0), allow_new_entries=True)


def test_step_places_no_buy_when_entries_are_not_allowed() -> None:
    market = round_trip_market()
    strategy = gap_up_strategy(holding_days=3)
    state = initial_state()
    for number in range(2):
        data = slice_to_decision_date(market, day_of(market, number))
        allowed = number == 0
        state = step(state, data, strategy, CostModel(0.0), allow_new_entries=allowed).state
    assert state.pending == ()


def test_initial_state_is_all_cash() -> None:
    state = initial_state()
    assert state.cash == INITIAL_CAPITAL == 100_000.0
    assert state.positions == {} and state.pending == () and state.last_date is None
    assert initial_state(5_000.0).cash == 5_000.0


def test_equity_curve_day_by_day() -> None:
    # round_trip_market at 0 bps: day 2 buys 100 shares at 100 (cash 90,000); closes are
    # 100, 102 and 107 on days 2 to 4; day 5 sells at 110 for 11,000.
    result = run_toy(gap_up_strategy(holding_days=3), round_trip_market())
    curve = result.equity_curve
    assert list(curve.columns[:4]) == ["cash", "holdings_value", "equity", "open_positions"]
    assert list(curve["equity"].iloc[:8]) == pytest.approx(
        [100_000, 100_000, 100_000, 100_200, 100_700, 101_000, 101_000, 101_000]
    )
    assert list(curve["open_positions"].iloc[:8]) == [0, 0, 1, 1, 1, 0, 0, 0]
    assert list(curve["holdings_value"].iloc[:6]) == pytest.approx(
        [0, 0, 10_000, 10_200, 10_700, 0]
    )
    # At 0 bps there is nothing to pay to sell, so net and gross agree.
    assert (curve["liquidation_cost"] == 0.0).all()
    assert (curve["equity"] == curve["cash"] + curve["holdings_value"]).all()


def test_every_equity_point_is_net_of_the_cost_of_selling_open_positions() -> None:
    # The same round trip at 5 bps. Day 2 holds about 99.95 shares (10,000 less the buy
    # cost, at 100) worth 9,995.0 at the close; selling them would cost 5 bps of that.
    result = run_toy(gap_up_strategy(holding_days=3), round_trip_market(), bps=5.0)
    curve = result.equity_curve
    expected_cost = curve["holdings_value"] * 5.0 / 10_000.0
    assert list(curve["liquidation_cost"]) == pytest.approx(list(expected_cost))
    assert curve["liquidation_cost"].iloc[2] > 0.0
    net = curve["cash"] + curve["holdings_value"] - curve["liquidation_cost"]
    assert list(curve["equity"]) == pytest.approx(list(net))
    # The last point needs no special treatment: it equals capital plus every trade's pnl.
    assert curve["equity"].iloc[-1] == pytest.approx(
        INITIAL_CAPITAL + sum(t.pnl for t in result.trades)
    )


def test_result_fields() -> None:
    market = round_trip_market()
    strategy = gap_up_strategy(holding_days=3)
    result = run_toy(strategy, market, bps=5.0)
    assert result.strategy_id == strategy_id(strategy)
    assert result.initial_capital == INITIAL_CAPITAL
    assert result.costs == CostModel(5.0)
    assert list(result.equity_curve.index) == list(market.trading_days)
    assert result.equity_curve["equity"].iloc[0] == INITIAL_CAPITAL


# ------------------------------------------------------ state, parity, costs


def test_state_json_round_trips_at_every_stage() -> None:
    market = sell_funds_buy_market()
    strategy = gap_up_strategy(holding_days=3, max_positions=1)
    kinds = set()
    for state in play(market, strategy, CostModel(5.0), 8):
        assert state_from_json(state_to_json(state)) == state
        kinds.add((bool(state.positions), bool(state.pending)))
    # The run passes through "nothing", "order only", "position only" and "position and order".
    assert {(False, False), (False, True), (True, False), (True, True)} <= kinds


def replay_through_json(
    market: MarketData,
    strategy_obj,
    window: BacktestWindow,
    costs: CostModel,  # type: ignore[no-untyped-def]
) -> tuple[list[Trade], list[float]]:
    """What the paper trader does: one step a day, saving and reloading the state each time."""
    days = [d for d in market.trading_days if window.start <= d.date() <= window.end]
    sid = strategy_id(strategy_obj)
    state = initial_state()
    trades: list[Trade] = []
    equity: list[float] = []
    for number, day in enumerate(days):
        data = slice_to_decision_date(market, day.date())
        result = step(state, data, strategy_obj, costs, allow_new_entries=number != len(days) - 1)
        state = state_from_json(state_to_json(result.state))
        trades += result.closed_trades
        equity.append(result.equity.equity)
    trades += close_out_at_end(state, data, costs, sid)
    return trades, equity


def test_replay_through_json_equals_run_backtest() -> None:
    market = sell_funds_buy_market()
    strategy = gap_up_strategy(holding_days=3, max_positions=1)
    costs = CostModel(5.0)
    days = market.trading_days
    window = BacktestWindow("primary", days[0].date(), days[7].date(), None)
    expected = run_backtest(strategy, market, window, costs)
    trades, equity = replay_through_json(market, strategy, window, costs)
    assert sorted(trades, key=lambda t: t.symbol) == sorted(expected.trades, key=lambda t: t.symbol)
    curve = list(expected.equity_curve["equity"])
    # Every point matches, the last one included: equity is net liquidation value every day.
    assert any(t.exit_reason == "open_at_end" for t in expected.trades)
    assert equity == curve
    assert curve[-1] == pytest.approx(INITIAL_CAPITAL + sum(t.pnl for t in expected.trades))


def test_cost_sensitivity_runs_three_levels_with_the_same_signals() -> None:
    market = sell_funds_buy_market()
    strategy = gap_up_strategy(holding_days=3, max_positions=1)
    days = market.trading_days
    window = BacktestWindow("primary", days[0].date(), days[7].date(), None)
    results = run_cost_sensitivity(strategy, market, window)
    assert sorted(results) == [0.0, 5.0, 20.0]
    # Signals are the same at every level. Fills can differ on real data, because a buy
    # can be cancelled for lack of cash at one cost level and not at another (review L9).
    signals = {
        bps: sorted((t.symbol, t.signal_date) for t in result.trades)
        for bps, result in results.items()
    }
    assert signals[0.0] == signals[5.0] == signals[20.0]
    ends = [results[bps].equity_curve["equity"].iloc[-1] for bps in (0.0, 5.0, 20.0)]
    assert ends[0] > ends[1] > ends[2]
    assert all(
        isinstance(r, BacktestResult) and r.costs.bps_per_side == b for b, r in results.items()
    )


# ------------------------------------------------------------------ holdout


def test_a_window_reaching_into_the_holdout_raises() -> None:
    market = round_trip_market()
    window = BacktestWindow("primary", date(2023, 3, 1), date(2024, 1, 31), None)
    with pytest.raises(HoldoutAccessError):
        run_backtest(gap_up_strategy(holding_days=3), market, window, CostModel(5.0))


def test_approved_flag_alone_does_not_open_the_holdout() -> None:
    market = round_trip_market()
    window = BacktestWindow("holdout", date(2024, 1, 1), date(2024, 1, 31), None)
    with pytest.raises(HoldoutAccessError):
        run_backtest(
            gap_up_strategy(holding_days=3), market, window, CostModel(5.0), holdout_approved=True
        )


def test_the_last_pre_holdout_day_is_allowed() -> None:
    market = round_trip_market()
    window = BacktestWindow("primary", date(2023, 3, 1), date(2023, 12, 31), None)
    assert run_backtest(gap_up_strategy(holding_days=3), market, window, CostModel(5.0)).trades


_ = pd  # pandas is used through the helpers
