"""The backtest engine: one day-step function shared by the backtester and paper trading.

Interface stubs for Phase 3. See docs/design/backtest.md for the reasoning.

``step`` processes one trading day D, given data cut at the close of D. The
backtester calls it once per day in a loop. The paper trader calls it once
after each real close, with the state it saved the day before. Because both
call the same function on the same inputs, replaying the paper trader over
history must give the same ledger as ``run_backtest`` (the parity test).

Order of work inside ``step`` for day D:

1. Splits and dividends for positions held overnight from D-1. A split
   multiplies shares and divides the stop and target prices by its factor.
   A dividend with ex-date D is paid in cash to shares held at the close of
   D-1, matching ``total_return`` in bot/data/prices.py. A position bought at
   the open of D gets nothing. A position sold at the open of D still gets it.
2. Ended data: a held stock with no price row on D is NOT closed at once. A
   halt, a late bar or a bad tick looks the same as a delisting on the day
   (review M2). The position is kept and valued at its last close. Only after
   ``DATA_ENDED_AFTER_MISSING_SESSIONS`` missing sessions in a row is it
   closed at that last close, dated the first missing session, with exit_reason
   "data_ended" and a flag. Its pending orders are then cancelled.
3. Pending sells fill at the open of D, then pending buys fill at the open of
   D, so cash from sells can fund buys. A sell with no price on D stays pending
   with its original decision date and fills at the first open after the gap
   (review L1, M2). A buy for a stock with no row on D is cancelled.
3b. Same-open strategies only (``entry_timing`` "same_open"): entries are
   decided at 09:30 on D from ``slice_to_open`` (D's open, the overnight news
   and everything else as of D-1's close) and bought at D's open, right after
   the sells. Free slots are max_positions minus positions still held after
   those sells. A stock sold at this open is not bought back at it. Each buy
   targets the equity at D-1's close divided by max_positions, limited by the
   cash left after this open's sells. ``allow_new_entries`` does not apply:
   these buys fill on D, inside the window.
4. Stops and targets are checked against D's low and high, including for
   stocks bought at today's open:
   - Stop: hit when low <= stop. Fill at the lower of the open and the stop,
     so a gap down through the stop fills at the (worse) open.
   - Target: hit when high >= target. Fill at the target price, even if the
     open was above it. This gives up any gap-up bonus on purpose.
   - First bar after a gap in the data (the stock had no row on the previous
     trading day): a target that the open already reached fills at the open,
     since a real holder could not have sold earlier. Otherwise the rules
     above apply to that bar.
   - Both hit on the same day: daily bars do not say which came first, so we
     assume the stop did (the worse outcome).
5. Mark to market at D's close and add one to each position's days held. The
   fill day counts as day 1. Equity is net liquidation value: cash plus each
   position at the close minus the cost of selling it (D16 applied every day).
6. Decisions at D's close, filled at the next open:
   - Holding period: a position with days held >= holding_days gets a sell
     order. With holding_days 3, a stock bought at the open of day F is sold
     at the open of F+3.
   - Entries (next-open strategies only), when ``allow_new_entries`` is
     True: entry_signals over the members on D, minus stocks already held.
     Free slots are max_positions minus open positions plus positions with a
     pending sell. If more stocks qualify than there are slots,
     ``entry_priority`` picks them (by ``rank_by`` first, if set).
     Each buy targets equity at D's close divided by max_positions.

Prices in the ledger are raw, as-traded prices (EODHD's unadjusted values),
recovered as adjusted price * raw_close / close on the same row. Raw prices
do not move when a later split is added to the data, so a backtest and a
paper trader that rebuilt its frames on a later day see the same numbers, and
trades can be hand-checked against raw EODHD rows. Shares are fractional.
There is no leverage and no short selling: cash never goes below zero.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from bot.backtest.costs import CostModel, affordable_notional, fill_cost, sensitivity_cost_models
from bot.data.aliases import company_key
from bot.data.holdout import assert_before_holdout
from bot.strategy.blocks import (
    DecisionData,
    MarketData,
    OpenDecisionData,
    SignalValues,
    entry_signals,
    entry_signals_at_open,
    rank_scores,
    signal_values,
    signal_values_at_open,
    slice_to_decision_date,
    slice_to_open,
)
from bot.strategy.schema import BacktestWindow, ExitRules, StrategyDefinition, strategy_id

INITIAL_CAPITAL = 100_000.0

# A held stock with no row for this many trading sessions in a row is treated
# as gone for good. Fewer than this is a halt or a late bar: the position is
# kept at its last close. Five sessions is one trading week.
DATA_ENDED_AFTER_MISSING_SESSIONS = 5

# Raw prices are adjusted values times raw_close / close. Rounding removes the
# float noise from that round trip, so ledger prices match EODHD's raw rows.
RAW_PRICE_DECIMALS = 6

ExitReason = Literal["holding_period", "stop_loss", "profit_target", "data_ended", "open_at_end"]
OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class Trade:
    """One finished round trip: a row of the trade ledger. Prices are raw, as traded.

    ``pnl`` = exit_price * exit_shares - exit_cost + dividends
    - (entry_price * entry_shares + entry_cost).
    ``return_pct`` = pnl / (entry_price * entry_shares + entry_cost) * 100.
    ``exit_shares`` differs from ``entry_shares`` only if a split happened while held.
    ``exit_signal_date`` is the close that triggered a holding-period exit, and
    None for exits that happen during the day or at the end of the test.
    ``flags`` are short labels such as "split_while_held", "stop_gap_through",
    "data_ended" or "holdout_cutoff" that the report explains.
    """

    strategy_id: str
    symbol: str
    signal_date: date
    entry_date: date
    entry_price: float
    entry_shares: float
    entry_cost: float
    exit_signal_date: date | None
    exit_date: date
    exit_price: float
    exit_shares: float
    exit_cost: float
    exit_reason: ExitReason
    dividends: float
    pnl: float
    return_pct: float
    trading_days_held: int
    signal_values: SignalValues
    flags: tuple[str, ...]


@dataclass(frozen=True)
class OpenPosition:
    """A position held at the end of a day. Prices are raw; ``shares`` includes later splits."""

    symbol: str
    signal_date: date
    entry_date: date
    entry_price: float
    entry_shares: float
    entry_cost: float
    shares: float
    stop_price: float | None
    target_price: float | None
    days_held: int
    dividends: float
    last_close: float
    last_close_date: date
    signal_values: SignalValues
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PendingOrder:
    """An order decided at a close and filled at the next open.

    Buys carry ``target_notional`` (equity / max_positions at the decision
    close). Sells carry ``exit_reason``.
    """

    symbol: str
    side: OrderSide
    decided_on: date
    target_notional: float | None
    exit_reason: ExitReason | None
    signal_values: SignalValues


@dataclass(frozen=True)
class PortfolioState:
    """Everything the engine carries from one day to the next. Saved by the paper trader."""

    cash: float
    positions: dict[str, OpenPosition]
    pending: tuple[PendingOrder, ...]
    last_date: date | None


@dataclass(frozen=True)
class EquityPoint:
    """The portfolio at one day's close.

    ``holdings_value`` is open positions at the close (market value).
    ``liquidation_cost`` is what selling all of them would cost.
    ``equity`` = cash + holdings_value - liquidation_cost, the net liquidation
    value. Every point of the curve uses this one definition, and sizing uses it.
    """

    day: date
    cash: float
    holdings_value: float
    equity: float
    open_positions: int
    liquidation_cost: float


@dataclass(frozen=True)
class StepResult:
    """What one call of ``step`` returns. ``notes`` lists cancelled orders and other events."""

    state: PortfolioState
    closed_trades: list[Trade]
    equity: EquityPoint
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    """One backtest of one strategy on one window at one cost level.

    ``equity_curve`` is indexed by date with columns cash, holdings_value,
    equity, open_positions and liquidation_cost. ``equity`` is net liquidation
    value on every day. ``trades`` includes positions still open at the end,
    with exit_reason "open_at_end".
    """

    strategy_id: str
    strategy: StrategyDefinition
    window: BacktestWindow
    costs: CostModel
    initial_capital: float
    equity_curve: pd.DataFrame
    trades: list[Trade]
    notes: list[str]


class SkippedTradingDayError(ValueError):
    """``step`` was asked to process a day while an earlier trading day was never processed.

    A skipped day loses its splits, dividends and fills for good. Callers must
    replay the missed days one by one, oldest first (review M3).
    """


# ------------------------------------------------------------------ the step


def initial_state(initial_capital: float = INITIAL_CAPITAL) -> PortfolioState:
    """All cash, no positions, no orders."""
    return PortfolioState(cash=initial_capital, positions={}, pending=(), last_date=None)


def step(
    state: PortfolioState,
    data: DecisionData,
    strategy: StrategyDefinition,
    costs: CostModel,
    *,
    allow_new_entries: bool,
    tiebreak_salt: str = "",
) -> StepResult:
    """Process one trading day (see the module docstring for the order of work).

    Pure: ``state`` is not changed; the new state is in the result. Raises
    ValueError if ``data.decision_date`` is not after ``state.last_date``, and
    SkippedTradingDayError (a ValueError) if a trading day lies strictly
    between them.
    ``allow_new_entries`` is False on the last day of a backtest window,
    because the next open is outside the window (D16 at the end of 2023). It
    only blocks entries decided at today's close; same-open entries fill
    today and are not affected. The paper trader always passes True.
    ``tiebreak_salt`` changes the hash shuffle, for research runs only (see
    ``entry_priority``). Strategies and the paper trader never set it.
    """
    today = data.decision_date
    if state.last_date is not None and today <= state.last_date:
        raise ValueError(f"{today} is not after the last processed day {state.last_date}.")
    _check_no_skipped_day(state, data)
    sid = strategy_id(strategy)
    previous_equity = net_liquidation_value(state, costs)
    working = apply_corporate_actions(state, data)
    working, ended, notes = _close_ended(working, data, costs, sid)
    working, sold, fill_notes = _fill_orders(working, data, costs, sid, strategy.exit)
    notes = notes + fill_notes
    if strategy.entry_timing == "same_open":
        sold_today = {trade.symbol for trade in sold}
        working, open_notes = _enter_at_open(
            working, data, strategy, costs, previous_equity, sold_today, tiebreak_salt
        )
        notes = notes + open_notes
    working, stopped = check_stops_and_targets(working, data, costs, sid)
    working, equity = mark_to_market(working, data, costs)
    working = decide_exits(working, strategy)
    if allow_new_entries and strategy.entry_timing == "next_open":
        working, entry_notes = _plan_entries(working, data, strategy, equity.equity, tiebreak_salt)
        notes = notes + entry_notes
    return StepResult(
        state=working, closed_trades=ended + sold + stopped, equity=equity, notes=notes
    )


def raw_bars_today(data: DecisionData, symbols: list[str]) -> pd.DataFrame:
    """Raw open, high, low, close, dividend and split_factor on the decision date.

    Indexed by symbol. Raw values are adjusted values times raw_close / close
    on the same row. Missing stocks are NaN.
    """
    market = data.market

    def last_row(frame: pd.DataFrame) -> pd.Series:
        return frame.iloc[-1].reindex(symbols).astype("float64")

    ratio = last_row(market.raw_close) / last_row(market.close)
    bars = pd.DataFrame(
        {
            "open": _clean_price(last_row(market.open) * ratio),
            "high": _clean_price(last_row(market.high) * ratio),
            "low": _clean_price(last_row(market.low) * ratio),
            "close": last_row(market.raw_close),
            "dividend": _clean_price(last_row(market.dividend) * ratio),
            "split_factor": last_row(market.split_factor),
        },
        index=pd.Index(symbols, name="symbol"),
    )
    return bars


def apply_corporate_actions(state: PortfolioState, data: DecisionData) -> PortfolioState:
    """Step 1: splits change shares and stop/target prices; dividends add cash.

    A held stock with no row today (NaN split factor and dividend) is skipped
    here. A split or dividend dated inside a gap shows up on the stock's first
    row after the gap, because the price frame moves it to the next row.
    """
    bars = raw_bars_today(data, list(state.positions))
    cash = state.cash
    positions: dict[str, OpenPosition] = {}
    for symbol, position in state.positions.items():
        updated, dividend_cash = _apply_actions(position, _bar(bars, symbol))
        positions[symbol] = updated
        cash += dividend_cash
    return replace(state, cash=cash, positions=positions)


def close_ended_positions(
    state: PortfolioState, data: DecisionData, costs: CostModel, strategy_id: str
) -> tuple[PortfolioState, list[Trade]]:
    """Step 2: close stocks missing for too many sessions at their last close, flagged "data_ended".

    A backtest and a paper trader cannot know on the day whether a missing row
    is a delisting, so both wait ``DATA_ENDED_AFTER_MISSING_SESSIONS`` sessions.
    The real exit price may differ (a takeover premium, or a bankruptcy loss),
    so the report lists these trades.
    """
    updated, trades, _ = _close_ended(state, data, costs, strategy_id)
    return updated, trades


def fill_pending_orders(
    state: PortfolioState,
    data: DecisionData,
    costs: CostModel,
    strategy_id: str,
    exit_rules: ExitRules | None = None,
) -> tuple[PortfolioState, list[Trade]]:
    """Step 3: sells, then buys, at today's raw open, each charged ``fill_cost``.

    A buy uses ``affordable_notional`` and sets its stop and target from the
    fill price and ``exit_rules``. The stub had no way to pass the stop and
    target percentages, so ``exit_rules`` is an added optional argument.
    Buys are all consumed. Sells for a stock with no price today stay pending.
    """
    updated, trades, _ = _fill_orders(state, data, costs, strategy_id, exit_rules)
    return updated, trades


def check_stops_and_targets(
    state: PortfolioState, data: DecisionData, costs: CostModel, strategy_id: str
) -> tuple[PortfolioState, list[Trade]]:
    """Step 4: exit on today's low and high, with the fill rules in the module docstring.

    Runs before ``mark_to_market`` adds today to days held, so a trade closed
    here records ``trading_days_held`` as days held so far plus one. A stock
    stopped out on its fill day therefore shows 1, not 0.
    """
    bars = raw_bars_today(data, list(state.positions))
    gap_symbols = {s: _is_first_bar_after_gap(p, data) for s, p in state.positions.items()}
    cash = state.cash
    kept: dict[str, OpenPosition] = {}
    trades: list[Trade] = []
    for symbol, position in state.positions.items():
        hit = _stop_or_target_hit(position, _bar(bars, symbol), after_gap=gap_symbols[symbol])
        if hit is None:
            kept[symbol] = position
            continue
        price, reason, flags = hit
        cost = fill_cost(position.shares * price, costs)
        cash += position.shares * price - cost
        trades.append(
            _make_trade(
                position,
                strategy_id=strategy_id,
                exit_date=data.decision_date,
                exit_price=price,
                exit_cost=cost,
                exit_reason=reason,
                exit_signal_date=None,
                trading_days_held=position.days_held + 1,
                extra_flags=flags,
            )
        )
    return replace(state, cash=cash, positions=kept), trades


def mark_to_market(
    state: PortfolioState, data: DecisionData, costs: CostModel
) -> tuple[PortfolioState, EquityPoint]:
    """Step 5: value positions at today's raw close, net of selling them, and add one to days held.

    The stub had no ``costs`` argument; it is needed because equity is net
    liquidation value on every day (one sell cost per open position).
    """
    bars = raw_bars_today(data, list(state.positions))
    positions: dict[str, OpenPosition] = {}
    holdings = 0.0
    liquidation = 0.0
    for symbol, position in state.positions.items():
        close = _bar_value(bars, symbol, "close")
        if np.isnan(close):
            # Reachable since M2: a stock missing for fewer than
            # DATA_ENDED_AFTER_MISSING_SESSIONS sessions is held at its last close.
            close = position.last_close
            close_date = position.last_close_date
        else:
            close_date = data.decision_date
        positions[symbol] = replace(
            position, days_held=position.days_held + 1, last_close=close, last_close_date=close_date
        )
        holdings += position.shares * close
        liquidation += fill_cost(position.shares * close, costs)
    point = EquityPoint(
        day=data.decision_date,
        cash=state.cash,
        holdings_value=holdings,
        equity=state.cash + holdings - liquidation,
        open_positions=len(positions),
        liquidation_cost=liquidation,
    )
    return replace(state, positions=positions, last_date=data.decision_date), point


def decide_exits(state: PortfolioState, strategy: StrategyDefinition) -> PortfolioState:
    """Step 6a: add a sell order for each position that reached its holding period."""
    holding_days = strategy.exit.holding_days
    if holding_days is None:
        return state
    already_selling = {order.symbol for order in state.pending if order.side == "sell"}
    orders = []
    for symbol, position in state.positions.items():
        if position.days_held >= holding_days and symbol not in already_selling:
            decided_on = state.last_date or position.last_close_date
            orders.append(PendingOrder(symbol, "sell", decided_on, None, "holding_period", {}))
    return replace(state, pending=state.pending + tuple(orders))


def decide_entries(
    state: PortfolioState, data: DecisionData, strategy: StrategyDefinition, equity: float
) -> PortfolioState:
    """Step 6b: add buy orders for the top candidates by ``entry_priority``, up to free slots."""
    updated, _ = _plan_entries(state, data, strategy, equity, "")
    return updated


def entry_priority(
    strategy_id: str,
    decision_date: date,
    symbols: list[str],
    *,
    scores: pd.Series | None = None,
    salt: str = "",
) -> list[str]:
    """Order candidates: highest ``scores`` first if given, then by the hash shuffle.

    The shuffle sorts by sha256 of "strategy_id|decision_date|symbol",
    ascending. It is fixed and repeatable, uses nothing from the future, does
    not favour early letters of the alphabet, and gives the paper trader the
    same order as the backtest. ``scores`` come from ``rank_by``. A non-empty
    ``salt`` is appended as "|salt" to give a different but equally fair
    shuffle; it exists only to measure how much results depend on the shuffle.
    """

    def shuffle_key(symbol: str) -> str:
        text = f"{strategy_id}|{decision_date.isoformat()}|{symbol}"
        if salt:
            text = f"{text}|{salt}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    if scores is None:
        return sorted(symbols, key=shuffle_key)

    def ranked_key(symbol: str) -> tuple[float, str]:
        return (-float(scores[symbol]), shuffle_key(symbol))

    return sorted(symbols, key=ranked_key)


def close_out_at_end(
    state: PortfolioState, data: DecisionData, costs: CostModel, strategy_id: str
) -> list[Trade]:
    """Value positions still open on the last day at that day's raw close, net of one sell cost.

    Each becomes a Trade with exit_reason "open_at_end". Pending orders are
    dropped. On 2023-12-29 this is D16: nothing is sold in the holdout.
    """
    bars = raw_bars_today(data, list(state.positions))
    trades: list[Trade] = []
    for symbol, position in state.positions.items():
        close = _bar_value(bars, symbol, "close")
        price = position.last_close if np.isnan(close) else close
        trades.append(
            _make_trade(
                position,
                strategy_id=strategy_id,
                exit_date=data.decision_date,
                exit_price=price,
                exit_cost=fill_cost(position.shares * price, costs),
                exit_reason="open_at_end",
                exit_signal_date=None,
                trading_days_held=position.days_held,
                extra_flags=(),
            )
        )
    return trades


# ------------------------------------------------------------- whole tests


def run_backtest(
    strategy: StrategyDefinition,
    market: MarketData,
    window: BacktestWindow,
    costs: CostModel,
    *,
    holdout_approved: bool = False,
    tiebreak_salt: str = "",
) -> BacktestResult:
    """Run ``step`` over every trading day in ``window``, then ``close_out_at_end``.

    Calls ``assert_before_holdout(window.end, approved=holdout_approved)``
    first. Starts from ``initial_state()``. Data before ``window.start`` may be
    used for lookbacks but no order is placed before it. Entries that would
    fill after the window (next-open entries decided on its last day) are
    blocked. ``tiebreak_salt`` is for research runs only (see ``entry_priority``).
    """
    assert_before_holdout(window.end, approved=holdout_approved)
    sid = strategy_id(strategy)
    in_window = (market.trading_days >= pd.Timestamp(window.start)) & (
        market.trading_days <= pd.Timestamp(window.end)
    )
    days = market.trading_days[in_window]
    if len(days) == 0:
        raise ValueError(f"No trading days between {window.start} and {window.end}.")
    state = initial_state()
    trades: list[Trade] = []
    notes: list[str] = []
    points: list[EquityPoint] = []
    data = slice_to_decision_date(market, days[0].date())
    for position, stamp in enumerate(days):
        data = slice_to_decision_date(market, stamp.date())
        last_day = position == len(days) - 1
        result = step(
            state,
            data,
            strategy,
            costs,
            allow_new_entries=not last_day,
            tiebreak_salt=tiebreak_salt,
        )
        state = result.state
        trades.extend(result.closed_trades)
        notes.extend(result.notes)
        points.append(result.equity)
    # The last point is already net of selling what is still open, so it
    # equals the starting capital plus the pnl of every trade below.
    trades.extend(close_out_at_end(state, data, costs, sid))
    return BacktestResult(
        strategy_id=sid,
        strategy=strategy,
        window=window,
        costs=costs,
        initial_capital=INITIAL_CAPITAL,
        equity_curve=equity_frame(points),
        trades=trades,
        notes=notes,
    )


def run_cost_sensitivity(
    strategy: StrategyDefinition, market: MarketData, window: BacktestWindow
) -> dict[float, BacktestResult]:
    """One backtest per level in ``sensitivity_cost_models()``, keyed by bps per side.

    Signals are the same at every level. Position sizes can differ a little,
    because costs change cash and equity.
    """
    results: dict[float, BacktestResult] = {}
    for model in sensitivity_cost_models():
        results[model.bps_per_side] = run_backtest(strategy, market, window, model)
    return results


# ------------------------------------------------------- saving paper state


def state_to_json(state: PortfolioState) -> str:
    """Serialise the state for the paper trader's database. Dates as ISO strings."""
    payload = {
        "cash": state.cash,
        "last_date": _iso_or_none(state.last_date),
        "positions": [_position_to_dict(position) for position in state.positions.values()],
        "pending": [_pending_to_dict(order) for order in state.pending],
    }
    return json.dumps(payload, sort_keys=True)


def state_from_json(text: str) -> PortfolioState:
    """The inverse of ``state_to_json``. The parity test round-trips through both every day."""
    payload = json.loads(text)
    positions = [_position_from_dict(raw) for raw in payload["positions"]]
    return PortfolioState(
        cash=float(payload["cash"]),
        positions={position.symbol: position for position in positions},
        pending=tuple(_pending_from_dict(raw) for raw in payload["pending"]),
        last_date=_date_or_none(payload["last_date"]),
    )


# ------------------------------------------------------------------ helpers


def net_liquidation_value(state: PortfolioState, costs: CostModel) -> float:
    """Cash plus each position at its last close, minus the cost of selling it.

    For a saved end-of-day state this equals that day's ``EquityPoint.equity``,
    so same-open sizing gets the previous close's equity from the state alone.
    """
    value = state.cash
    for position in state.positions.values():
        market_value = position.shares * position.last_close
        value += market_value - fill_cost(market_value, costs)
    return value


def _bar(bars: pd.DataFrame, symbol: str) -> pd.Series:
    """One symbol's row of ``raw_bars_today``."""
    return cast(pd.Series, bars.loc[symbol])


def _bar_value(bars: pd.DataFrame, symbol: str, column: str) -> float:
    """One number from ``raw_bars_today``."""
    return float(cast(float, bars.at[symbol, column]))


def _clean_price(values: pd.Series) -> pd.Series:
    """Round away float noise left by adjusting and un-adjusting a price."""
    return values.round(RAW_PRICE_DECIMALS)


def _iso_or_none(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _date_or_none(text: str | None) -> date | None:
    return None if text is None else date.fromisoformat(text)


def equity_frame(points: list[EquityPoint]) -> pd.DataFrame:
    """The equity curve as a DataFrame indexed by date."""
    frame = pd.DataFrame(
        {
            "cash": [p.cash for p in points],
            "holdings_value": [p.holdings_value for p in points],
            "equity": [p.equity for p in points],
            "open_positions": [p.open_positions for p in points],
            "liquidation_cost": [p.liquidation_cost for p in points],
        },
        index=pd.DatetimeIndex([pd.Timestamp(p.day) for p in points], name="date"),
    )
    return frame


def _make_trade(
    position: OpenPosition,
    *,
    strategy_id: str,
    exit_date: date,
    exit_price: float,
    exit_cost: float,
    exit_reason: ExitReason,
    exit_signal_date: date | None,
    trading_days_held: int,
    extra_flags: tuple[str, ...],
) -> Trade:
    """Turn a position and its exit into a ledger row."""
    spent = position.entry_price * position.entry_shares + position.entry_cost
    proceeds = exit_price * position.shares - exit_cost
    pnl = proceeds + position.dividends - spent
    return Trade(
        strategy_id=strategy_id,
        symbol=position.symbol,
        signal_date=position.signal_date,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        entry_shares=position.entry_shares,
        entry_cost=position.entry_cost,
        exit_signal_date=exit_signal_date,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_shares=position.shares,
        exit_cost=exit_cost,
        exit_reason=exit_reason,
        dividends=position.dividends,
        pnl=pnl,
        return_pct=pnl / spent * 100.0,
        trading_days_held=trading_days_held,
        signal_values=dict(position.signal_values),
        flags=position.flags + extra_flags,
    )


def _apply_actions(position: OpenPosition, bar: pd.Series) -> tuple[OpenPosition, float]:
    """Apply today's split and dividend to one position. Returns it and the dividend cash."""
    factor = float(bar["split_factor"])
    if np.isnan(factor):
        return position, 0.0
    if factor != 1.0:
        flags = position.flags
        if "split_while_held" not in flags:
            flags = flags + ("split_while_held",)
        position = replace(
            position,
            shares=position.shares * factor,
            stop_price=None if position.stop_price is None else position.stop_price / factor,
            target_price=None if position.target_price is None else position.target_price / factor,
            flags=flags,
        )
    dividend = float(bar["dividend"])
    if np.isnan(dividend) or dividend <= 0.0:
        return position, 0.0
    cash = dividend * position.shares
    return replace(position, dividends=position.dividends + cash), cash


def _close_ended(
    state: PortfolioState, data: DecisionData, costs: CostModel, strategy_id: str
) -> tuple[PortfolioState, list[Trade], list[str]]:
    """Step 2 with notes: close held stocks missing for too many sessions in a row.

    The exit is dated the first missing session and priced at the last close,
    the same value the stock was carried at. A shorter gap keeps the position.
    """
    bars = raw_bars_today(data, list(state.positions))
    cash = state.cash
    kept: dict[str, OpenPosition] = {}
    trades: list[Trade] = []
    notes: list[str] = []
    for symbol, position in state.positions.items():
        if not np.isnan(_bar_value(bars, symbol, "close")):
            kept[symbol] = position
            continue
        missing = _missing_sessions(position, data)
        if len(missing) < DATA_ENDED_AFTER_MISSING_SESSIONS:
            kept[symbol] = position
            notes.append(
                f"{data.decision_date}: {symbol} has no data for {len(missing)} "
                f"session(s), kept at last close {position.last_close}."
            )
            continue
        price = position.last_close
        cost = fill_cost(position.shares * price, costs)
        cash += position.shares * price - cost
        trades.append(
            _make_trade(
                position,
                strategy_id=strategy_id,
                exit_date=missing[0],
                exit_price=price,
                exit_cost=cost,
                exit_reason="data_ended",
                exit_signal_date=None,
                trading_days_held=position.days_held,
                extra_flags=("data_ended",),
            )
        )
        notes.append(
            f"{data.decision_date}: {symbol} has no data since {missing[0]}, "
            f"closed at last close {price}."
        )
    pending = tuple(o for o in state.pending if o.side == "buy" or o.symbol in kept)
    return replace(state, cash=cash, positions=kept, pending=pending), trades, notes


def _missing_sessions(position: OpenPosition, data: DecisionData) -> list[date]:
    """Trading days after the position's last close, up to and including today."""
    days = data.market.trading_days
    later = days[days > pd.Timestamp(position.last_close_date)]
    return [stamp.date() for stamp in later]


def _check_no_skipped_day(state: PortfolioState, data: DecisionData) -> None:
    """Raise SkippedTradingDayError if a trading day lies between the last day and today."""
    if state.last_date is None:
        return
    days = data.market.trading_days
    between = days[
        (days > pd.Timestamp(state.last_date)) & (days < pd.Timestamp(data.decision_date))
    ]
    if len(between) > 0:
        raise SkippedTradingDayError(
            f"Cannot step to {data.decision_date}: trading day {between[0].date()} after "
            f"{state.last_date} was not processed. Replay the missed days one by one."
        )


def _sells_that_will_fill(state: PortfolioState, data: DecisionData) -> int:
    """Pending sells that free a slot tomorrow: decided today, on a stock that has a price today.

    A sell carried over because its stock had no price may not fill tomorrow
    either, so it does not free a slot (review L1).
    """
    sold = list(dict.fromkeys(o.symbol for o in state.pending if o.side == "sell"))
    bars = raw_bars_today(data, sold)
    count = 0
    for order in state.pending:
        has_price = not np.isnan(_bar_value(bars, order.symbol, "close"))
        if order.side == "sell" and order.decided_on == data.decision_date and has_price:
            count += 1
    return count


def _fill_orders(
    state: PortfolioState,
    data: DecisionData,
    costs: CostModel,
    strategy_id: str,
    exit_rules: ExitRules | None,
) -> tuple[PortfolioState, list[Trade], list[str]]:
    """Step 3 with notes: fill sells, then buys, at today's raw open."""
    symbols = list(dict.fromkeys(order.symbol for order in state.pending))
    bars = raw_bars_today(data, symbols)
    cash = state.cash
    positions = dict(state.positions)
    trades: list[Trade] = []
    notes: list[str] = []
    carried: list[PendingOrder] = []
    for order in state.pending:
        position = positions.get(order.symbol)
        if order.side != "sell" or position is None or order.exit_reason is None:
            continue
        price = _bar_value(bars, order.symbol, "open")
        if not price > 0.0:
            # No usable price today (a gap in the data). Keep the order with its
            # original decision date until the stock trades again (review L1, M2).
            carried.append(order)
            notes.append(
                f"{data.decision_date}: sell of {order.symbol} decided on {order.decided_on} "
                "has no price today, still pending."
            )
            continue
        cost = fill_cost(position.shares * price, costs)
        cash += position.shares * price - cost
        del positions[order.symbol]
        trades.append(
            _make_trade(
                position,
                strategy_id=strategy_id,
                exit_date=data.decision_date,
                exit_price=price,
                exit_cost=cost,
                exit_reason=order.exit_reason,
                exit_signal_date=order.decided_on,
                trading_days_held=position.days_held,
                extra_flags=(),
            )
        )
    for order in state.pending:
        if order.side == "buy":
            cash = _fill_buy(order, bars, cash, positions, data, costs, exit_rules, notes)
    return replace(state, cash=cash, positions=positions, pending=tuple(carried)), trades, notes


def _fill_buy(
    order: PendingOrder,
    bars: pd.DataFrame,
    cash: float,
    positions: dict[str, OpenPosition],
    data: DecisionData,
    costs: CostModel,
    exit_rules: ExitRules | None,
    notes: list[str],
) -> float:
    """Fill one buy at today's open and add the position. Returns the cash left."""
    today = data.decision_date
    price = _bar_value(bars, order.symbol, "open")
    if np.isnan(price) or price <= 0.0:
        notes.append(f"{today}: buy of {order.symbol} cancelled, no price today.")
        return cash
    notional = affordable_notional(order.target_notional or 0.0, cash, costs)
    if notional <= 0.0:
        notes.append(f"{today}: buy of {order.symbol} cancelled, no cash.")
        return cash
    cost = fill_cost(notional, costs)
    shares = notional / price
    stop, target = _exit_levels(price, exit_rules)
    positions[order.symbol] = OpenPosition(
        symbol=order.symbol,
        signal_date=order.decided_on,
        entry_date=today,
        entry_price=price,
        entry_shares=shares,
        entry_cost=cost,
        shares=shares,
        stop_price=stop,
        target_price=target,
        days_held=0,
        dividends=0.0,
        last_close=price,
        last_close_date=today,
        signal_values=dict(order.signal_values),
        flags=(),
    )
    # Tiny negative cash can only come from float rounding.
    return max(cash - notional - cost, 0.0)


def _exit_levels(price: float, rules: ExitRules | None) -> tuple[float | None, float | None]:
    """The stop and target prices for a fill at ``price``."""
    if rules is None:
        return None, None
    stop = None
    target = None
    if rules.stop_loss_pct is not None:
        stop = price * (1.0 - rules.stop_loss_pct / 100.0)
    if rules.profit_target_pct is not None:
        target = price * (1.0 + rules.profit_target_pct / 100.0)
    return stop, target


def _is_first_bar_after_gap(position: OpenPosition, data: DecisionData) -> bool:
    """True when the position's last close is older than the previous trading day."""
    days = data.market.trading_days
    if len(days) < 2:
        return False
    return position.last_close_date < days[-2].date()


def _stop_or_target_hit(
    position: OpenPosition, bar: pd.Series, *, after_gap: bool = False
) -> tuple[float, ExitReason, tuple[str, ...]] | None:
    """Fill price, reason and flags if today's bar hits the stop or target. The stop wins.

    On the first bar after a gap in the data, a target the open already
    reached fills at the open (flag "gap_fill"), since the holder could not
    have sold earlier. A stop already fills at the lower of open and stop.
    """
    stop, target = position.stop_price, position.target_price
    if stop is not None and bar["low"] <= stop:
        open_price = float(bar["open"])
        if np.isnan(open_price):
            return stop, "stop_loss", ()
        if open_price < stop:
            return open_price, "stop_loss", ("stop_gap_through",)
        return stop, "stop_loss", ()
    if target is not None and bar["high"] >= target:
        open_price = float(bar["open"])
        if after_gap and open_price > target:
            return open_price, "profit_target", ("gap_fill",)
        return target, "profit_target", ()
    return None


def unheld_candidates(signalled: list[str], held_symbols: Iterable[str]) -> list[str]:
    """Signalled symbols whose company is not already held.

    A renamed stock has two symbols but one price series, so a held FB blocks
    META for as long as the position is open (D36). Both entry paths use this.
    """
    held = {company_key(symbol) for symbol in held_symbols}
    return [symbol for symbol in signalled if company_key(symbol) not in held]


def _plan_entries(
    state: PortfolioState,
    data: DecisionData,
    strategy: StrategyDefinition,
    equity: float,
    salt: str,
) -> tuple[PortfolioState, list[str]]:
    """Step 6b with notes: choose which signals get a buy order."""
    signals = entry_signals(data, strategy)
    candidates = unheld_candidates(list(signals[signals].index), state.positions)
    selling = _sells_that_will_fill(state, data)
    max_positions = strategy.sizing.max_positions
    free_slots = max(max_positions - len(state.positions) + selling, 0)
    ordered = _order_candidates(data, strategy, candidates, salt)
    chosen = ordered[:free_slots]
    target_notional = max(equity, 0.0) / max_positions
    orders = tuple(
        PendingOrder(
            symbol=symbol,
            side="buy",
            decided_on=data.decision_date,
            target_notional=target_notional,
            exit_reason=None,
            signal_values=signal_values(data, strategy, symbol),
        )
        for symbol in chosen
    )
    notes = []
    if len(ordered) > len(chosen):
        missed = len(ordered) - len(chosen)
        notes.append(f"{data.decision_date}: {missed} entry signals did not get a slot.")
    return replace(state, pending=state.pending + orders), notes


def _order_candidates(
    view: DecisionData | OpenDecisionData,
    strategy: StrategyDefinition,
    candidates: list[str],
    salt: str,
) -> list[str]:
    """Candidates in buying order: by ``rank_by`` if set, then by the hash shuffle."""
    scores = None
    if strategy.rank_by is not None and candidates:
        scores = rank_scores(view, strategy, candidates)
    return entry_priority(
        strategy_id(strategy), view.decision_date, candidates, scores=scores, salt=salt
    )


def _enter_at_open(
    state: PortfolioState,
    data: DecisionData,
    strategy: StrategyDefinition,
    costs: CostModel,
    previous_equity: float,
    sold_today: set[str],
    salt: str,
) -> tuple[PortfolioState, list[str]]:
    """Step 3b: decide same-open entries at 09:30 on D and buy them at D's open.

    Runs after this open's sells, so slots and cash are what is left after
    them. The first trading day of the data has no previous close, so nothing
    is bought on it.
    """
    if len(data.market.trading_days) < 2:
        return state, []
    view = slice_to_open(data.market, data.decision_date)
    signals = entry_signals_at_open(view, strategy)
    held = set(state.positions) | sold_today
    candidates = unheld_candidates(list(signals[signals].index), held)
    max_positions = strategy.sizing.max_positions
    free_slots = max(max_positions - len(state.positions), 0)
    chosen = _order_candidates(view, strategy, candidates, salt)[:free_slots]
    target_notional = max(previous_equity, 0.0) / max_positions
    bars = raw_bars_today(data, chosen)
    cash = state.cash
    positions = dict(state.positions)
    notes: list[str] = []
    for symbol in chosen:
        values = signal_values_at_open(view, strategy, symbol)
        order = PendingOrder(symbol, "buy", data.decision_date, target_notional, None, values)
        cash = _fill_buy(order, bars, cash, positions, data, costs, strategy.exit, notes)
    if len(candidates) > len(chosen):
        missed = len(candidates) - len(chosen)
        notes.append(f"{data.decision_date}: {missed} entry signals did not get a slot.")
    return replace(state, cash=cash, positions=positions), notes


def _position_to_dict(position: OpenPosition) -> dict[str, Any]:
    """One position as JSON-safe values."""
    raw = asdict(position)
    for key in ("signal_date", "entry_date", "last_close_date"):
        raw[key] = raw[key].isoformat()
    raw["flags"] = list(position.flags)
    return raw


def _position_from_dict(raw: dict[str, Any]) -> OpenPosition:
    """The inverse of ``_position_to_dict``."""
    fields = dict(raw)
    for key in ("signal_date", "entry_date", "last_close_date"):
        fields[key] = date.fromisoformat(fields[key])
    fields["flags"] = tuple(fields["flags"])
    return OpenPosition(**fields)


def _pending_to_dict(order: PendingOrder) -> dict[str, Any]:
    """One pending order as JSON-safe values."""
    raw = asdict(order)
    raw["decided_on"] = order.decided_on.isoformat()
    return raw


def _pending_from_dict(raw: dict[str, Any]) -> PendingOrder:
    """The inverse of ``_pending_to_dict``."""
    fields = dict(raw)
    fields["decided_on"] = date.fromisoformat(fields["decided_on"])
    return PendingOrder(**fields)
