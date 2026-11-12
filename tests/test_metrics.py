"""Metric formulas on small equity curves and trade lists, with hand-computed answers."""

import math
from datetime import date

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, ExitReason, Trade
from bot.backtest.metrics import (
    benchmark_curves,
    cagr_pct,
    compute_metrics,
    daily_returns,
    equal_weight_benchmark,
    max_drawdown_pct,
    return_stats,
    sharpe_ratio,
    spy_benchmark,
    total_return_pct,
    win_rate_pct,
)
from bot.config import MIN_TRADES_TO_JUDGE
from bot.strategy.blocks import MarketData
from bot.strategy.schema import BacktestWindow
from tests.backtest_helpers import (
    assemble_market,
    flat_bar,
    gap_up_strategy,
    raw_frame,
    toy_days,
)


def series(values: list[float], start: str = "2023-03-01") -> pd.Series:
    """An equity curve on consecutive weekdays."""
    return pd.Series(values, index=toy_days(len(values), start), dtype="float64")


def make_trade(pnl: float, reason: ExitReason = "holding_period", return_pct: float = 0.0) -> Trade:
    """A minimal Trade. Only pnl, return_pct and exit_reason matter for the metrics."""
    day = date(2023, 3, 1)
    return Trade(
        strategy_id="strat_test",
        symbol="AAA.US",
        signal_date=day,
        entry_date=day,
        entry_price=100.0,
        entry_shares=1.0,
        entry_cost=0.0,
        exit_signal_date=None,
        exit_date=day,
        exit_price=100.0,
        exit_shares=1.0,
        exit_cost=0.0,
        exit_reason=reason,
        dividends=0.0,
        pnl=pnl,
        return_pct=return_pct,
        trading_days_held=1,
        signal_values={},
        flags=(),
    )


# ----------------------------------------------------------- return statistics


def test_daily_returns_drop_the_first_day() -> None:
    returns = daily_returns(series([100.0, 110.0, 99.0]))
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_total_return() -> None:
    # 121 / 100 - 1 = 21%
    assert total_return_pct(series([100.0, 110.0, 99.0, 121.0])) == pytest.approx(21.0)
    assert total_return_pct(series([100.0, 90.0])) == pytest.approx(-10.0)


def test_cagr_uses_calendar_days_over_365_25() -> None:
    # 2016-01-01 to 2020-01-01 is 1461 calendar days = 4 * 365.25, so the exponent is exactly
    # 0.25. 1600 / 100 = 16 and 16 ** 0.25 = 2, so CAGR is 100%. The middle points are ignored.
    equity = pd.Series(
        [100.0, 300.0, 200.0, 1600.0],
        index=pd.to_datetime(["2016-01-01", "2017-07-01", "2018-07-01", "2020-01-01"]),
    )
    assert cagr_pct(equity) == pytest.approx(100.0)


def test_cagr_for_a_loss() -> None:
    # 100 -> 6.25 over 1461 days: 0.0625 ** 0.25 = 0.5, so -50%.
    equity = pd.Series([100.0, 6.25], index=pd.to_datetime(["2016-01-01", "2020-01-01"]))
    assert cagr_pct(equity) == pytest.approx(-50.0)


def test_max_drawdown_is_the_largest_peak_to_trough_fall() -> None:
    # Peak 120 -> trough 90 is -25%. Peak 150 -> 120 is only -20%.
    assert max_drawdown_pct(series([100.0, 120.0, 90.0, 150.0, 120.0, 135.0])) == pytest.approx(
        -25.0
    )


def test_max_drawdown_of_a_rising_curve_is_zero() -> None:
    assert max_drawdown_pct(series([100.0, 101.0, 105.0])) == 0.0


def test_sharpe_hand_value() -> None:
    # Returns: 0.02, 0, 0.02, 0. Mean 0.01. Sample std (ddof=1): sqrt(4 * 0.01**2 / 3).
    # Sharpe = 0.01 / 0.0115470054 * sqrt(252) = sqrt(756) / 2 = 13.7477270848.
    equity = series([100.0, 102.0, 102.0, 104.04, 104.04])
    assert sharpe_ratio(equity) == pytest.approx(math.sqrt(756) / 2, rel=1e-9)
    assert sharpe_ratio(equity) == pytest.approx(13.7477270848, rel=1e-9)


def test_sharpe_is_negative_for_a_negative_mean() -> None:
    # Returns -0.02, 0, -0.02, 0: the mirror image, so -13.7477...
    equity = series([100.0, 98.0, 98.0, 96.04, 96.04])
    assert sharpe_ratio(equity) == pytest.approx(-math.sqrt(756) / 2, rel=1e-9)


def test_sharpe_is_none_without_spread() -> None:
    assert sharpe_ratio(series([100.0, 100.0, 100.0, 100.0])) is None


def test_return_stats_bundle() -> None:
    equity = series([100.0, 102.0, 102.0, 104.04, 104.04])
    stats = return_stats(equity)
    assert stats.total_return_pct == pytest.approx(4.04)
    assert stats.max_drawdown_pct == 0.0
    assert stats.sharpe == pytest.approx(13.7477270848, rel=1e-9)
    assert stats.cagr_pct == pytest.approx(cagr_pct(equity))


# ---------------------------------------------------------------- trade stats


def test_win_rate_counts_only_positive_pnl_and_skips_open_at_end() -> None:
    trades = [
        make_trade(100.0),
        make_trade(-50.0),
        make_trade(0.0),  # flat is not a win
        make_trade(30.0, "stop_loss"),
        make_trade(500.0, "open_at_end"),  # left out of the rate
    ]
    # 2 wins out of 4 closed trades.
    assert win_rate_pct(trades) == pytest.approx(50.0)


def test_win_rate_is_none_without_closed_trades() -> None:
    assert win_rate_pct([]) is None
    assert win_rate_pct([make_trade(10.0, "open_at_end")]) is None


# ----------------------------------------------------------------- benchmarks


def benchmark_market() -> MarketData:
    """Five days, SPY with a dividend, and three stocks with changing membership."""
    days = toy_days(5)
    spy = raw_frame(
        days,
        {
            0: flat_bar(100.0),
            1: flat_bar(102.0),
            2: flat_bar(101.0),
            3: flat_bar(103.02),
            4: flat_bar(103.02),
        },
        dividends={2: 1.0},
    )
    stocks = {
        "A.US": raw_frame(
            days,
            {
                0: flat_bar(100.0),
                1: flat_bar(110.0),
                2: flat_bar(110.0),
                3: flat_bar(112.2),
                4: flat_bar(123.42),
            },
        ),
        "B.US": raw_frame(
            days, {0: flat_bar(100.0), 1: flat_bar(120.0), 2: flat_bar(108.0), 3: flat_bar(108.0)}
        ),
        "C.US": raw_frame(
            days, {n: flat_bar(p) for n, p in enumerate([100.0, 100.0, 100.0, 150.0, 150.0])}
        ),
    }
    membership = pd.DataFrame(True, index=days, columns=["A.US", "B.US", "C.US"], dtype=bool)
    membership.loc[days[:2], "C.US"] = False  # C joins the index on day 2
    return assemble_market(days, stocks, spy_frame=spy, membership=membership)


def test_spy_benchmark_includes_the_dividend() -> None:
    market = benchmark_market()
    curve = spy_benchmark(market, market.trading_days)
    # Day 1: 102/100. Day 2: (101 + 1) / 102 - 1 = 0. Day 3: 103.02 / 101 - 1 = 2%. Day 4: 0.
    assert list(curve) == pytest.approx(
        [1.0, 1.02, 1.02, 1.02 * 103.02 / 101.0, 1.02 * 103.02 / 101.0]
    )


def test_spy_benchmark_starts_at_one_on_the_first_day_given() -> None:
    market = benchmark_market()
    curve = spy_benchmark(market, market.trading_days[1:])
    assert curve.iloc[0] == 1.0
    assert curve.iloc[1] == pytest.approx(1.0)  # day 2 return is 0
    assert curve.iloc[2] == pytest.approx(103.02 / 101.0)


def test_equal_weight_uses_previous_close_membership_and_drops_ended_data() -> None:
    market = benchmark_market()
    curve = equal_weight_benchmark(market, market.trading_days)
    # Day 1: members on day 0 are A, B: mean(+10%, +20%) = +15%.
    # Day 2: members on day 1 are A, B (C joins on day 2, so it is not counted yet):
    #        mean(0%, -10%) = -5%.
    # Day 3: members on day 2 are A, B, C: mean(+2%, 0%, +50%) = +17.33%.
    # Day 4: B has no row, so it drops out: mean(A +10%, C 0%) = +5%.
    expected = 1.15 * 0.95 * (1 + (0.02 + 0.0 + 0.5) / 3) * 1.05
    assert curve.iloc[0] == 1.0
    assert curve.iloc[1] == pytest.approx(1.15)
    assert curve.iloc[2] == pytest.approx(1.15 * 0.95)
    assert curve.iloc[4] == pytest.approx(expected)


def make_result(market: MarketData, equity: list[float], trades: list[Trade]) -> BacktestResult:
    """A BacktestResult built by hand around a toy market."""
    days = market.trading_days
    curve = pd.DataFrame(
        {
            "cash": equity,
            "holdings_value": 0.0,
            "equity": equity,
            "open_positions": 0,
        },
        index=days,
    )
    return BacktestResult(
        strategy_id="strat_test",
        strategy=gap_up_strategy(holding_days=3),
        window=BacktestWindow("primary", days[0].date(), days[-1].date(), None),
        costs=CostModel(bps_per_side=5.0),
        initial_capital=equity[0],
        equity_curve=curve,
        trades=trades,
        notes=[],
    )


def test_benchmark_curves_columns_and_start() -> None:
    market = benchmark_market()
    result = make_result(market, [100_000.0, 101_000.0, 99_000.0, 102_000.0, 102_000.0], [])
    curves = benchmark_curves(result, market)
    assert list(curves.columns) == ["strategy", "spy", "equal_weight"]
    assert list(curves.index) == list(market.trading_days)
    assert list(curves.iloc[0]) == [1.0, 1.0, 1.0]
    assert curves["strategy"].iloc[-1] == pytest.approx(1.02)


# -------------------------------------------------------------- compute_metrics


def test_compute_metrics_fields() -> None:
    market = benchmark_market()
    trades = [
        make_trade(100.0, return_pct=2.0),
        make_trade(-40.0, "stop_loss", return_pct=-1.0),
        make_trade(10.0, "data_ended", return_pct=0.5),
        make_trade(500.0, "open_at_end", return_pct=9.0),
    ]
    result = make_result(market, [100_000.0, 101_000.0, 99_000.0, 102_000.0, 102_000.0], trades)
    metrics = compute_metrics(result, market, versions_tried=4)
    assert metrics.strategy_id == "strat_test"
    assert metrics.start == market.trading_days[0].date()
    assert metrics.end == market.trading_days[-1].date()
    assert metrics.cost_bps_per_side == 5.0
    assert metrics.versions_tried == 4
    assert metrics.strategy_returns.total_return_pct == pytest.approx(2.0)
    assert metrics.strategy_returns.max_drawdown_pct == pytest.approx((99_000 / 101_000 - 1) * 100)
    curves = benchmark_curves(result, market)
    assert metrics.spy.total_return_pct == pytest.approx((curves["spy"].iloc[-1] - 1) * 100)
    assert metrics.equal_weight.total_return_pct == pytest.approx(
        (curves["equal_weight"].iloc[-1] - 1) * 100
    )
    # Closed trades are everything except open_at_end, and data_ended is one of them.
    assert metrics.closed_trades == 3
    assert metrics.open_at_end == 1
    assert metrics.data_ended_trades == 1
    assert metrics.win_rate_pct == pytest.approx(200.0 / 3)  # 2 of 3 closed trades won
    assert metrics.average_trade_return_pct == pytest.approx((2.0 - 1.0 + 0.5) / 3)
    assert metrics.too_few_trades is True


@pytest.mark.parametrize(
    ("closed", "too_few"),
    [(MIN_TRADES_TO_JUDGE - 1, True), (MIN_TRADES_TO_JUDGE, False)],
)
def test_too_few_trades_below_thirty(closed: int, too_few: bool) -> None:
    assert MIN_TRADES_TO_JUDGE == 30
    market = benchmark_market()
    trades = [make_trade(1.0) for _ in range(closed)] + [make_trade(1.0, "open_at_end")]
    result = make_result(market, [100_000.0] * 5, trades)
    assert compute_metrics(result, market, versions_tried=1).too_few_trades is too_few


# ------------------------------------------- equal weight on every member (D33)


def _eod_rows(closes: list[float], days: pd.DatetimeIndex) -> list[dict[str, object]]:
    """Raw daily bars with no split, so adjusted close equals close."""
    return [
        {
            "date": day.date().isoformat(),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adjusted_close": c,
            "volume": 1000.0,
        }
        for day, c in zip(days, closes, strict=True)
    ]


def test_full_universe_equal_weight_is_skipped_when_prices_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bot.backtest import metrics as metrics_module
    from bot.data import eodhd_client as client

    def missing(**_: object) -> list[object]:
        raise client.OfflineCacheMissError("not cached")

    monkeypatch.setattr(client, "get_index_history", missing)
    days = pd.bdate_range("2020-01-02", periods=4)
    result = metrics_module.full_universe_equal_weight(days, days, date(2020, 1, 1))
    assert (
        result
        == metrics_module.FULL_UNIVERSE_SKIPPED
        == ("skipped: full-universe prices not downloaded")
    )


def test_full_universe_equal_weight_averages_cached_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bot.backtest import metrics as metrics_module
    from bot.data import eodhd_client as client

    days = pd.bdate_range("2020-01-02", periods=4)
    closes = {"AAA": [10.0, 11.0, 12.1, 12.1], "BBB": [20.0, 18.0, 18.0, 19.8]}
    rows = [
        {"Code": code, "Name": code, "StartDate": "2019-01-01", "EndDate": None,
         "IsActiveNow": 1, "IsDelisted": 0}
        for code in closes
    ]  # fmt: skip
    monkeypatch.setattr(client, "get_index_history", lambda **_: rows)
    monkeypatch.setattr(client, "get_eod_prices", lambda t, s, e, **_: _eod_rows(closes[t], days))
    monkeypatch.setattr(client, "get_splits", lambda t, **_: [])
    monkeypatch.setattr(client, "get_dividends", lambda t, s, e, **_: [])
    curve = metrics_module.full_universe_equal_weight(days, days, date(2020, 1, 1))
    assert not isinstance(curve, str)
    # Day 2: (+10% and -10%) / 2 = 0. Day 3: (+10% and 0%) / 2 = +5%. Day 4: (0% and +10%) / 2.
    expected = [1.0, 1.0, 1.05, 1.05 * 1.05]
    assert curve.tolist() == pytest.approx(expected)
