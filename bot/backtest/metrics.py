"""Performance metrics and benchmark curves. Every reported number is computed here.

Interface stubs for Phase 3. See docs/design/backtest.md.

Conventions:

- The strategy curve is the engine's ``equity`` column: net liquidation value
  (cash plus open positions at the close, minus the cost of selling them) on
  every day, not only the last. Total return, CAGR, drawdown and Sharpe all
  use it.
- Daily returns are ``equity / previous equity - 1`` from the equity curve.
- Sharpe ratio: mean daily return / standard deviation of daily returns
  (ddof=1) * sqrt(252). The risk-free rate is taken as 0, and every report
  says so.
- CAGR: (final / initial) ** (365.25 / calendar days) - 1.
- Max drawdown: the largest fall from a previous peak, as a negative percent.
- Win rate: the share of closed trades with pnl > 0. Trades still open at
  the end ("open_at_end") are counted separately and not in the win rate.
- Under config.MIN_TRADES_TO_JUDGE closed trades, ``too_few_trades`` is True and the
  report says the result is too thin to judge.

Benchmarks, both before costs and including dividends:

- SPY: buy and hold, from SPY's ``total_return``.
- Equal-weight S&P 500, rebalanced daily: day D's return is the plain average
  of ``total_return`` on D over the stocks that were members at the close of
  D-1 and have a row on D. A stock whose data ends drops out that day; it is
  not forward filled. After a missing day, the stock's own ``total_return``
  covers the whole move since its previous row. Daily rebalancing is the simplest version to explain
  ("the average member's daily return") and needs no holdings bookkeeping.
  It ignores rebalancing costs, so it is slightly flattering.

Both start at 1.0 at the close of the window's first day, like the strategy.

In sample mode (D33) the equal-weight benchmark uses the same sample as the
strategy, which is the fair comparison. ``full_universe_equal_weight`` adds
the same benchmark over every historical member, but only from prices that
are already in the cache. It never downloads. If any member's prices are
missing it returns a "skipped" message instead of a curve.
"""

import math
from dataclasses import dataclass
from datetime import date
from typing import cast

import pandas as pd

from bot.backtest.engine import BacktestResult, Trade
from bot.config import BACKTEST_END, MIN_TRADES_TO_JUDGE
from bot.data import eodhd_client as client
from bot.data.membership import RawMembershipRow, build_membership_history, membership_matrix
from bot.data.prices import (
    RawDividendRow,
    RawEodRow,
    RawSplitRow,
    build_price_frame,
    infer_splits_from_adjusted_ratio,
    merge_split_sources,
    parse_split_rows,
    total_return_index,
)
from bot.strategy.blocks import MarketData

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0
FULL_UNIVERSE_SKIPPED = "skipped: full-universe prices not downloaded"
# Daily returns with a spread below this are treated as constant.
ZERO_SPREAD = 1e-12


@dataclass(frozen=True)
class ReturnStats:
    """Return statistics for one equity curve. Percent values are in percent units."""

    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float | None


@dataclass(frozen=True)
class Metrics:
    """Everything a report shows for one backtest.

    ``versions_tried`` comes from storage (the number of strategy versions the
    user ran in this conversation), never from the agent. ``sharpe`` is None
    when daily returns have zero spread.
    """

    strategy_id: str
    start: date
    end: date
    cost_bps_per_side: float
    strategy_returns: ReturnStats
    spy: ReturnStats
    equal_weight: ReturnStats
    closed_trades: int
    open_at_end: int
    win_rate_pct: float | None
    average_trade_return_pct: float | None
    data_ended_trades: int
    too_few_trades: bool
    versions_tried: int


def daily_returns(equity: pd.Series) -> pd.Series:
    """Day-over-day returns of an equity curve. The first day is dropped."""
    return (equity / equity.shift(1) - 1.0).iloc[1:]


def total_return_pct(equity: pd.Series) -> float:
    """(last / first - 1) * 100."""
    return float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)


def cagr_pct(equity: pd.Series) -> float:
    """Compound annual growth in percent, using calendar days / 365.25."""
    calendar_days = (equity.index[-1] - equity.index[0]).days
    growth = float(equity.iloc[-1] / equity.iloc[0])
    if calendar_days <= 0:
        return 0.0
    if growth <= 0.0:
        return -100.0
    return float(growth ** (365.25 / calendar_days) - 1.0) * 100.0


def max_drawdown_pct(equity: pd.Series) -> float:
    """The largest peak-to-trough fall in percent, as a negative number (0.0 if none)."""
    drawdown = equity / equity.cummax() - 1.0
    return min(float(drawdown.min()) * 100.0, 0.0)


def sharpe_ratio(equity: pd.Series) -> float | None:
    """Annualised Sharpe with a risk-free rate of 0. None when the spread is zero."""
    returns = daily_returns(equity)
    if len(returns) < 2:
        return None
    spread = float(returns.std(ddof=1))
    if spread <= ZERO_SPREAD:
        return None
    excess = float(returns.mean()) - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    return excess / spread * math.sqrt(TRADING_DAYS_PER_YEAR)


def return_stats(equity: pd.Series) -> ReturnStats:
    """All four return statistics for one curve."""
    return ReturnStats(
        total_return_pct=total_return_pct(equity),
        cagr_pct=cagr_pct(equity),
        max_drawdown_pct=max_drawdown_pct(equity),
        sharpe=sharpe_ratio(equity),
    )


def win_rate_pct(trades: list[Trade]) -> float | None:
    """Percent of closed trades (not "open_at_end") with pnl > 0. None if there are none."""
    closed = _closed_trades(trades)
    if not closed:
        return None
    wins = sum(1 for trade in closed if trade.pnl > 0)
    return wins / len(closed) * 100.0


def spy_benchmark(market: MarketData, days: pd.DatetimeIndex) -> pd.Series:
    """SPY buy and hold with dividends, 1.0 at the first of ``days``."""
    returns = market.spy["total_return"].reindex(days).fillna(0.0)
    return _curve_from_returns(returns)


def equal_weight_benchmark(market: MarketData, days: pd.DatetimeIndex) -> pd.Series:
    """The daily rebalanced equal-weight point-in-time S&P 500, 1.0 at the first of ``days``."""
    index = market.total_return_index.reindex(columns=market.close.columns)
    return equal_weight_curve(index, market.membership, days)


def equal_weight_curve(
    total_return_table: pd.DataFrame, membership: pd.DataFrame, days: pd.DatetimeIndex
) -> pd.Series:
    """Daily rebalanced equal weight over the table's stocks, 1.0 at the first of ``days``.

    ``total_return_table`` is growth of 1.0 per stock on the trading calendar.
    ``membership`` is the date x symbol member table on the same calendar.
    """
    # Each stock's own total return: the first row after a missing day carries
    # the whole move across the gap (review L2).
    returns = total_return_table / total_return_table.ffill().shift(1) - 1.0
    was_member = (
        membership.reindex(
            index=total_return_table.index, columns=total_return_table.columns, fill_value=False
        )
        .shift(1, fill_value=False)
        .astype(bool)
    )
    daily = returns.where(was_member).mean(axis=1).reindex(days).fillna(0.0)
    return _curve_from_returns(daily)


def full_universe_equal_weight(
    calendar: pd.DatetimeIndex, days: pd.DatetimeIndex, price_start: date
) -> pd.Series | str:
    """Equal weight over every historical member in ``days``, from cached prices only.

    ``calendar`` is the trading calendar (``MarketData.trading_days``) and
    ``price_start`` the start of the downloaded price range. Returns
    FULL_UNIVERSE_SKIPPED when the membership or any member's prices, splits
    or dividends are not in the cache. Nothing is downloaded.
    """
    try:
        rows = cast(list[RawMembershipRow], client.get_index_history(offline=True))
        history = build_membership_history(rows)
        members = history.universe(days[0].date(), days[-1].date())
        columns = {
            symbol: total_return_index(_cached_price_frame(symbol, calendar, price_start))
            for symbol in sorted(members)
        }
    except client.OfflineCacheMissError:
        return FULL_UNIVERSE_SKIPPED
    table = pd.DataFrame(columns, index=calendar).astype("float64")
    return equal_weight_curve(table, membership_matrix(history, calendar), days)


def _cached_price_frame(symbol: str, calendar: pd.DatetimeIndex, start: date) -> pd.DataFrame:
    """One price frame built only from cached rows. Raises OfflineCacheMissError on a miss."""
    ticker = symbol.removesuffix(".US")
    end = BACKTEST_END
    eod = cast(list[RawEodRow], client.get_eod_prices(ticker, start, end, offline=True))
    splits = cast(list[RawSplitRow], client.get_splits(ticker, offline=True))
    dividends = client.get_dividends(ticker, start, end, offline=True)
    merged = merge_split_sources(
        parse_split_rows(splits), infer_splits_from_adjusted_ratio(eod), calendar
    )
    return build_price_frame(eod, merged, cast(list[RawDividendRow], dividends))


def benchmark_curves(result: BacktestResult, market: MarketData) -> pd.DataFrame:
    """Strategy, SPY and equal-weight curves on the result's dates, all 1.0 on day one.

    Columns: ``strategy``, ``spy``, ``equal_weight``. Used by the charts and
    by ``compute_metrics``.
    """
    days = pd.DatetimeIndex(result.equity_curve.index)
    strategy = net_equity(result)
    curves = pd.DataFrame(
        {
            "strategy": strategy / strategy.iloc[0],
            "spy": spy_benchmark(market, days),
            "equal_weight": equal_weight_benchmark(market, days),
        },
        index=days,
    )
    curves.index.name = "date"
    return curves


def compute_metrics(result: BacktestResult, market: MarketData, versions_tried: int) -> Metrics:
    """All metrics for one backtest, including both benchmarks."""
    curves = benchmark_curves(result, market)
    closed = _closed_trades(result.trades)
    return Metrics(
        strategy_id=result.strategy_id,
        start=curves.index[0].date(),
        end=result.window.end,
        cost_bps_per_side=result.costs.bps_per_side,
        strategy_returns=return_stats(curves["strategy"]),
        spy=return_stats(curves["spy"]),
        equal_weight=return_stats(curves["equal_weight"]),
        closed_trades=len(closed),
        open_at_end=sum(1 for t in result.trades if t.exit_reason == "open_at_end"),
        win_rate_pct=win_rate_pct(result.trades),
        average_trade_return_pct=(
            sum(t.return_pct for t in closed) / len(closed) if closed else None
        ),
        data_ended_trades=sum(1 for t in result.trades if t.exit_reason == "data_ended"),
        too_few_trades=len(closed) < MIN_TRADES_TO_JUDGE,
        versions_tried=versions_tried,
    )


def net_equity(result: BacktestResult) -> pd.Series:
    """The equity curve: net liquidation value on every day, so every point is comparable."""
    return result.equity_curve["equity"].astype("float64")


def _closed_trades(trades: list[Trade]) -> list[Trade]:
    """Trades that really ended, without those still open at the end of the test."""
    return [trade for trade in trades if trade.exit_reason != "open_at_end"]


def _curve_from_returns(returns: pd.Series) -> pd.Series:
    """A curve of 1.0 on the first day, grown by the daily returns after it."""
    steps = returns.astype("float64").copy()
    steps.iloc[0] = 0.0
    return (1.0 + steps).cumprod()
