"""Paper trading. It only calls ``engine.step``: there is no trading logic here.

Each run loads the saved state, steps the engine once per new trading day, and
saves the state, the daily equity row and the closed trades. Fills come from
daily data after each close (D7). For a same-open strategy, the update after
D's close records the buy at D's open price, decided by ``engine.step`` from
what was known at 09:30 on D only, exactly as in the backtest. So it is a
simulated open order, not a live one. Paper data is forward data (2024 onward in
live mode), so it is only read inside ``paper_session``, which exists only
for a strategy that went through the approval flow.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from bot import storage
from bot.backtest.costs import CostModel, make_cost_model
from bot.backtest.engine import (
    INITIAL_CAPITAL,
    BacktestResult,
    EquityPoint,
    PortfolioState,
    Trade,
    close_out_at_end,
    equity_frame,
    initial_state,
    state_from_json,
    state_to_json,
    step,
)
from bot.config import DEFAULT_COST_BPS_PER_SIDE, data_mode
from bot.data.eodhd_client import forward_as_of
from bot.data.fake_eodhd import LAST_PRICE_DAY
from bot.data.holdout import NEW_YORK, assert_before_holdout, paper_session
from bot.data.market import load_market_data
from bot.data.prices import COMMON_SPLIT_FACTORS
from bot.strategy.blocks import DecisionData, MarketData, slice_to_decision_date
from bot.strategy.schema import (
    BacktestWindow,
    StrategyDefinition,
    strategy_from_json,
    strategy_id,
)

# Enough history for the longest look-back (250 days of RSI or moving average
# plus the 252-day return), with room for holidays.
PAPER_LOOKBACK_DAYS = 500
TODAY_OVERRIDE_ENV = "OPENCLAW_BOT_TODAY"
# Same tolerance as split inference in bot/data/prices.py.
JUMP_TOLERANCE = 0.05


class UnexplainedPriceJumpError(RuntimeError):
    """A held or candidate stock's raw close jumped like a split that the data does not list.

    Stepping would book a fake loss or gain (H3). Nothing is saved. A later
    update retries, after the split list has refreshed.
    """


@dataclass(frozen=True)
class PaperUpdate:
    """What one ``paper_update`` call did."""

    strategy_id: str
    stepped_days: list[date]
    last_date: date | None
    equity: float | None
    closed_trades: int
    notes: list[str]


def paper_today() -> date:
    """The day paper trading treats as today.

    Live mode: the current date in New York. Fake mode: the last day of the
    simulated data, unless ``OPENCLAW_BOT_TODAY`` (YYYY-MM-DD) says otherwise.
    The override is read only in fake mode and is not reachable by the agent.
    """
    if data_mode() == "live":
        return datetime.now(NEW_YORK).date()
    override = os.getenv(TODAY_OVERRIDE_ENV, "").strip()
    return date.fromisoformat(override) if override else LAST_PRICE_DAY


def start_paper(strategy_id_: str, today: date) -> PaperUpdate:
    """Record the paper start day and take the first step. Call after the holdout run."""
    storage.start_paper_run(strategy_id_, today.isoformat())
    return paper_update(strategy_id_, today)


def paper_update(strategy_id_: str, today: date) -> PaperUpdate:
    """Step the engine once per trading day after the last saved day, up to ``today``.

    Catches up when days were missed. Does nothing when no new bar exists yet.
    """
    run = storage.get_paper_run(strategy_id_)
    stored = storage.get_strategy_json(strategy_id_)
    if run is None or stored is None:
        raise ValueError(f"Paper trading was not started for {strategy_id_}.")
    strategy = strategy_from_json(stored)
    latest = storage.latest_paper_state(strategy_id_)
    state = state_from_json(latest[1]) if latest else initial_state()
    start_day = date.fromisoformat(run["start_day"])
    with paper_session(strategy_id_), forward_as_of(today):
        market = load_market_data(today - timedelta(days=PAPER_LOOKBACK_DAYS), today, approved=True)
        days = _new_days(market, state.last_date, start_day, today)
        costs = make_cost_model(DEFAULT_COST_BPS_PER_SIDE)
        return _run_days(strategy_id_, strategy, market, state, days, costs)


def _new_days(
    market: MarketData, last_date: date | None, start_day: date, today: date
) -> list[pd.Timestamp]:
    """Trading days to process: after the last saved day, from the start day, up to today."""
    floor = pd.Timestamp(start_day if last_date is None else last_date + timedelta(days=1))
    days = market.trading_days
    return list(days[(days >= floor) & (days <= pd.Timestamp(today))])


def _run_days(
    strategy_id_: str,
    strategy: StrategyDefinition,
    market: MarketData,
    state: PortfolioState,
    days: list[pd.Timestamp],
    costs: CostModel,
) -> PaperUpdate:
    """Step each day with the engine and save it. Returns a summary."""
    notes: list[str] = []
    closed = 0
    last: EquityPoint | None = None
    for stamp in days:
        data = slice_to_decision_date(market, stamp.date())
        check_unexplained_jumps(data, set(state.positions))
        result = step(state, data, strategy, costs, allow_new_entries=True)
        saved = state_to_json(result.state)
        state = state_from_json(saved)
        trades = [trade_row(trade) for trade in result.closed_trades]
        storage.save_paper_day(
            strategy_id_, stamp.date().isoformat(), saved, asdict(result.equity), trades
        )
        notes.extend(result.notes)
        closed += len(trades)
        last = result.equity
    return PaperUpdate(
        strategy_id=strategy_id_,
        stepped_days=[stamp.date() for stamp in days],
        last_date=state.last_date,
        equity=None if last is None else last.equity,
        closed_trades=closed,
        notes=notes,
    )


def check_unexplained_jumps(data: DecisionData, held: set[str]) -> None:
    """Refuse the day if a held or member stock moved like an unlisted split (H3).

    Compares the raw close with the previous raw close, after the split factor
    on the day. A remaining ratio that matches a common split factor within
    ``JUMP_TOLERANCE`` has no listed split behind it.
    """
    raw = data.market.raw_close
    if len(raw) < 2:
        return
    symbols = [s for s in sorted(held | set(data.members)) if s in raw.columns]
    ratio = raw.iloc[-1][symbols] / raw.iloc[-2][symbols]
    ratio = ratio * data.market.split_factor.iloc[-1].reindex(symbols).fillna(1.0)
    for symbol, value in ratio.items():
        if pd.isna(value) or value <= 0:
            continue
        for factor in COMMON_SPLIT_FACTORS:
            if abs((1.0 / float(value)) / factor - 1.0) <= JUMP_TOLERANCE:
                raise UnexplainedPriceJumpError(
                    f"{symbol} raw close moved by a factor of {float(value):.3f} on "
                    f"{data.decision_date} with no split listed. Nothing was saved. "
                    "Run the update again after the split data refreshes."
                )


def trade_row(trade: Trade) -> dict[str, Any]:
    """A ledger row as plain JSON values."""
    row: dict[str, Any] = json.loads(json.dumps(asdict(trade), default=str))
    return row


def replay_paper(
    strategy: StrategyDefinition,
    market: MarketData,
    window: BacktestWindow,
    costs: CostModel | None = None,
) -> BacktestResult:
    """Replay paper trading over a window, saving the state through JSON every day.

    Used by the parity test: the result, every equity point included, must
    equal ``run_backtest`` on the same inputs. Like the backtest, it blocks
    next-open entries on the window's last day and lists open positions at the
    end (the paper trader itself never does either, because its next open is
    real). Equity is net liquidation value every day in both, so the last
    point needs no special treatment. Same-open strategies need nothing extra
    here: ``step`` decides and fills them at each day's open.
    """
    assert_before_holdout(window.end)
    model = costs or make_cost_model(DEFAULT_COST_BPS_PER_SIDE)
    in_window = (market.trading_days >= pd.Timestamp(window.start)) & (
        market.trading_days <= pd.Timestamp(window.end)
    )
    days = list(market.trading_days[in_window])
    state = initial_state()
    trades: list[Trade] = []
    notes: list[str] = []
    points: list[EquityPoint] = []
    data = slice_to_decision_date(market, days[0].date())
    for position, stamp in enumerate(days):
        data = slice_to_decision_date(market, stamp.date())
        result = step(state, data, strategy, model, allow_new_entries=position < len(days) - 1)
        state = state_from_json(state_to_json(result.state))
        trades.extend(result.closed_trades)
        notes.extend(result.notes)
        points.append(result.equity)
    sid = strategy_id(strategy)
    trades.extend(close_out_at_end(state, data, model, sid))
    return BacktestResult(
        strategy_id=sid,
        strategy=strategy,
        window=window,
        costs=model,
        initial_capital=INITIAL_CAPITAL,
        equity_curve=equity_frame(points),
        trades=trades,
        notes=notes,
    )
