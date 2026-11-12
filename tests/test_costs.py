"""Cost math: hand-computed numbers for 0, 5 and 20 bps per side.

The round trip used throughout: buy 100 shares at a raw open of 100 (notional
10,000), sell them at a raw open of 110 (notional 11,000).

    0 bps:  entry cost  0.0, exit cost  0.0, pnl 11000 - 10000          = 1000.0, return 10%
    5 bps:  entry cost  5.0, exit cost  5.5, pnl 11000 - 5.5 - 10005    =  989.5
            return 989.5 / 10005 * 100 = 9.8900549...
    20 bps: entry cost 20.0, exit cost 22.0, pnl 11000 - 22 - 10020     =  958.0
            return 958 / 10020 * 100   = 9.5608782...
"""

import pytest

from bot.backtest.costs import (
    MAX_COST_BPS,
    CostModel,
    affordable_notional,
    fill_cost,
    make_cost_model,
    sensitivity_cost_models,
)
from tests.backtest_helpers import gap_up_strategy, round_trip_market, run_toy


def test_default_is_five_bps() -> None:
    assert CostModel().bps_per_side == 5.0


@pytest.mark.parametrize(
    ("notional", "bps", "expected"),
    [
        (10_000.0, 0.0, 0.0),
        (10_000.0, 5.0, 5.0),
        (10_000.0, 20.0, 20.0),
        (11_000.0, 5.0, 5.5),
        (11_000.0, 20.0, 22.0),
        (-11_000.0, 5.0, 5.5),  # abs(): a negative notional costs the same
    ],
)
def test_fill_cost(notional: float, bps: float, expected: float) -> None:
    assert fill_cost(notional, CostModel(bps_per_side=bps)) == pytest.approx(expected, abs=1e-12)


def test_make_cost_model_checks_bounds() -> None:
    assert make_cost_model(0.0).bps_per_side == 0.0
    assert make_cost_model(MAX_COST_BPS).bps_per_side == MAX_COST_BPS
    with pytest.raises(ValueError):
        make_cost_model(-0.1)
    with pytest.raises(ValueError):
        make_cost_model(MAX_COST_BPS + 0.1)


def test_sensitivity_levels_are_0_5_and_20() -> None:
    assert [model.bps_per_side for model in sensitivity_cost_models()] == [0.0, 5.0, 20.0]


def test_affordable_notional_is_the_target_when_cash_is_plentiful() -> None:
    assert affordable_notional(10_000.0, 100_000.0, CostModel(5.0)) == 10_000.0


def test_affordable_notional_leaves_room_for_the_cost() -> None:
    # 100,000 cash at 5 bps: notional = 100000 / 1.0005, and notional + cost = 100000.
    notional = affordable_notional(200_000.0, 100_000.0, CostModel(5.0))
    assert notional == pytest.approx(100_000.0 / 1.0005, rel=1e-12)
    assert notional + fill_cost(notional, CostModel(5.0)) == pytest.approx(100_000.0, rel=1e-12)
    # At 20 bps: 100000 / 1.002.
    assert affordable_notional(200_000.0, 100_000.0, CostModel(20.0)) == pytest.approx(
        100_000.0 / 1.002, rel=1e-12
    )


def test_affordable_notional_is_zero_without_cash() -> None:
    assert affordable_notional(10_000.0, 0.0, CostModel(5.0)) == 0.0
    assert affordable_notional(10_000.0, -50.0, CostModel(5.0)) == 0.0


@pytest.mark.parametrize(
    ("bps", "entry_cost", "exit_cost", "pnl", "return_pct"),
    [
        (0.0, 0.0, 0.0, 1000.0, 10.0),
        (5.0, 5.0, 5.5, 989.5, 989.5 / 10_005.0 * 100),
        (20.0, 20.0, 22.0, 958.0, 958.0 / 10_020.0 * 100),
    ],
)
def test_round_trip_through_the_engine(
    bps: float, entry_cost: float, exit_cost: float, pnl: float, return_pct: float
) -> None:
    market = round_trip_market()
    result = run_toy(gap_up_strategy(holding_days=3), market, bps=bps)
    (trade,) = result.trades
    assert trade.exit_reason == "holding_period"
    assert (trade.entry_price, trade.entry_shares) == (100.0, pytest.approx(100.0))
    assert trade.exit_price == 110.0
    assert trade.entry_cost == pytest.approx(entry_cost)
    assert trade.exit_cost == pytest.approx(exit_cost)
    assert trade.pnl == pytest.approx(pnl)
    assert trade.return_pct == pytest.approx(return_pct)


def test_round_trip_return_values_by_hand() -> None:
    assert pytest.approx(9.8900549, abs=1e-6) == 989.5 / 10_005.0 * 100
    assert pytest.approx(9.5608782, abs=1e-6) == 958.0 / 10_020.0 * 100


def test_costs_lower_the_ending_equity_in_order() -> None:
    market = round_trip_market()
    ends = [
        run_toy(gap_up_strategy(holding_days=3), market, bps=bps).equity_curve["equity"].iloc[-1]
        for bps in (0.0, 5.0, 20.0)
    ]
    # Cash at the end is 100000 + pnl, since the trade is closed by day 5.
    assert ends == [pytest.approx(101_000.0), pytest.approx(100_989.5), pytest.approx(100_958.0)]
