"""Planted cases from ``bot.data.fake_eodhd.PLANTED`` through the real backtest in fake mode.

The raw rows come from the fake EODHD client (``EODHD_MODE=fake``). The market
is built the way the real pipeline builds it. Every expected date and price
comes from the planted facts or from the raw rows, not from the engine.

Symbols in the ledger carry the exchange suffix, for example ``SIM001.US``.
"""

import json
from datetime import date

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, Trade, run_backtest
from bot.data import fake_eodhd as fake
from bot.strategy.blocks import (
    MarketData,
    entry_signals,
    measure_days_since_last_earnings,
    measure_days_to_next_earnings,
    slice_to_decision_date,
)
from bot.strategy.schema import BacktestWindow, StrategyDefinition, strategy_from_json
from tests.backtest_helpers import (
    RawWorld,
    build_world_market,
    fake_universe_codes,
    load_raw_world,
    planted_market,
    raw_close_on,
    raw_open_on,
    symbol_of,
)

COSTS = CostModel(bps_per_side=5.0)


def d(text: str) -> date:
    return date.fromisoformat(text)


def make_strategy(
    entry: list[dict[str, object]], exit_rules: dict[str, object], max_positions: int = 50
) -> StrategyDefinition:
    return strategy_from_json(
        json.dumps(
            {
                "name": "Planted case",
                "description_plain_english": "A strategy used to check a planted case.",
                "entry": entry,
                "exit": exit_rules,
                "sizing": {"max_positions": max_positions},
            }
        )
    )


def run(strategy: StrategyDefinition, market: MarketData, start: str, end: str) -> BacktestResult:
    window = BacktestWindow("primary", d(start), d(end), None)
    return run_backtest(strategy, market, window, COSTS)


def trades_for(result: BacktestResult, code: str) -> list[Trade]:
    return [t for t in result.trades if t.symbol == symbol_of(code)]


GAP_DOWN = {"block": "gap", "direction": "down", "min_gap_pct": 3}
NO_NEWS = {"block": "no_news", "lookback_sessions": 1}
ALWAYS = {"block": "n_day_return", "days": 1, "comparison": "at_least", "return_pct": -90}


@pytest.fixture(scope="module")
def planted() -> tuple[RawWorld, MarketData]:
    return planted_market()


# ------------------------------------------------------ gap down, with and without news


def test_planted_dates_match_the_fake_world() -> None:
    # The cases below hard-code the days the issue describes. This ties them to PLANTED.
    assert fake.PLANTED["gap_down_no_news"]["ticker"] == "SIM001"
    assert fake.PLANTED["gap_down_no_news"]["date"] == "2022-02-10"
    assert fake.PLANTED["gap_down_with_news"]["ticker"] == "SIM002"
    assert fake.PLANTED["gap_down_with_news"]["date"] == "2022-06-14"


def test_gap_down_with_no_news_filter_buys_the_quiet_name_only(
    planted: tuple[RawWorld, MarketData],
) -> None:
    world, market = planted
    strategy = make_strategy([GAP_DOWN, NO_NEWS], {"holding_days": 3})
    result = run(strategy, market, "2022-01-03", "2022-07-29")
    quiet = [t for t in trades_for(result, "SIM001") if t.entry_date == d("2022-02-11")]
    assert len(quiet) == 1
    assert quiet[0].signal_date == d("2022-02-10")
    assert quiet[0].entry_price == raw_open_on(world, "SIM001", "2022-02-11")
    assert quiet[0].signal_values["gap_pct"] == pytest.approx(-5.0, abs=1.0)
    # SIM002 gaps down 6% on 2022-06-14 but has at least 3 articles in the window.
    noisy = [t for t in trades_for(result, "SIM002") if t.entry_date == d("2022-06-15")]
    assert noisy == []


def test_gap_down_without_the_news_filter_buys_both(
    planted: tuple[RawWorld, MarketData],
) -> None:
    world, market = planted
    strategy = make_strategy([GAP_DOWN], {"holding_days": 3})
    result = run(strategy, market, "2022-01-03", "2022-07-29")
    entries = {(t.symbol, t.entry_date): t for t in result.trades}
    assert (symbol_of("SIM001"), d("2022-02-11")) in entries
    both = entries[(symbol_of("SIM002"), d("2022-06-15"))]
    assert both.entry_price == raw_open_on(world, "SIM002", "2022-06-15")
    assert both.exit_reason == "holding_period"
    # Held on 15, 16 and 17 June. Monday 20 June 2022 was a market holiday, so the sale
    # is on the 21st.
    assert both.exit_date == d("2022-06-21")


def test_every_ledger_price_is_a_raw_eodhd_price(planted: tuple[RawWorld, MarketData]) -> None:
    world, market = planted
    strategy = make_strategy([GAP_DOWN], {"holding_days": 3})
    result = run(strategy, market, "2022-01-03", "2022-07-29")
    assert result.trades
    for trade in result.trades:
        code = trade.symbol.removesuffix(".US")
        assert trade.entry_price == raw_open_on(world, code, trade.entry_date.isoformat())
        if trade.exit_reason == "holding_period":
            assert trade.exit_price == raw_open_on(world, code, trade.exit_date.isoformat())


# ------------------------------------------------------------- removed from the index


def test_removed_name_cannot_be_entered_after_its_last_member_day(
    planted: tuple[RawWorld, MarketData],
) -> None:
    _, market = planted
    spec = fake.PLANTED["removed_still_trading"]
    last_member = d(spec["last_member_day"])
    assert d(spec["first_non_member_day"]) > last_member
    strategy = make_strategy([ALWAYS], {"holding_days": 1})
    result = run(strategy, market, "2018-05-01", "2018-08-31")
    removed = trades_for(result, "SIM007")
    assert removed, "the always-true strategy should trade the name while it is a member"
    assert all(t.signal_date <= last_member for t in removed)
    # The strategy keeps trading a name that stays in the index, so the cut-off is the index.
    assert any(
        t.signal_date > d(spec["first_non_member_day"]) for t in trades_for(result, "SIM001")
    )


def test_signals_include_the_last_member_day_and_exclude_the_next(
    planted: tuple[RawWorld, MarketData],
) -> None:
    _, market = planted
    spec = fake.PLANTED["removed_still_trading"]
    strategy = make_strategy([ALWAYS], {"holding_days": 1})
    symbol = symbol_of(spec["ticker"])
    last = entry_signals(slice_to_decision_date(market, d(spec["last_member_day"])), strategy)
    after = entry_signals(slice_to_decision_date(market, d(spec["first_non_member_day"])), strategy)
    assert symbol in last.index and bool(last[symbol])
    assert symbol not in after.index


# ------------------------------------------------------------------- delisting


def test_delisted_name_while_held_exits_as_data_ended(
    planted: tuple[RawWorld, MarketData],
) -> None:
    world, market = planted
    spec = fake.PLANTED["delisted"]
    strategy = make_strategy([ALWAYS], {"stop_loss_pct": 50})
    result = run(strategy, market, "2020-03-02", "2020-06-30")
    trades = trades_for(result, "SIM008")
    assert trades
    last = trades[-1]
    # Removed from the index on 2020-03-20, but held names are kept until an exit rule fires.
    assert last.entry_date < d(spec["removed_on"])
    assert last.exit_reason == "data_ended"
    assert "data_ended" in last.flags
    assert last.exit_price == raw_close_on(world, "SIM008", spec["last_trading_day"])
    assert last.exit_date == d("2020-04-20")  # the first session with no row
    assert last.exit_date > d(spec["last_trading_day"])
    assert last.exit_cost == pytest.approx(last.exit_price * last.exit_shares * 0.0005)
    assert last.pnl == pytest.approx(
        last.exit_price * last.exit_shares
        - last.exit_cost
        + last.dividends
        - (last.entry_price * last.entry_shares + last.entry_cost)
    )
    # Nothing re-enters it afterwards: there are no more rows.
    assert result.equity_curve["open_positions"].iloc[-1] >= 0


# ------------------------------------------------------------------------ split


def test_split_during_a_hold_keeps_value_continuous(
    planted: tuple[RawWorld, MarketData],
) -> None:
    world, market = planted
    spec = fake.PLANTED["split_before_holdout"]
    assert spec["factor"] == 4.0
    strategy = make_strategy([ALWAYS], {"stop_loss_pct": 50})
    result = run(strategy, market, "2017-05-15", "2017-07-14")
    trade = trades_for(result, "SIM003")[0]
    split_day = d(spec["date"])
    assert trade.entry_date < split_day
    assert trade.exit_date > split_day
    assert trade.exit_reason == "open_at_end"
    assert "split_while_held" in trade.flags
    assert trade.exit_shares == pytest.approx(4.0 * trade.entry_shares)
    # Raw prices, not split-adjusted ones.
    assert trade.entry_price == raw_open_on(world, "SIM003", trade.entry_date.isoformat())
    assert trade.exit_price == raw_close_on(world, "SIM003", "2017-07-14")
    # Value is continuous: a mistake in the share count would show as a change of 4 times.
    start_value = trade.entry_price * trade.entry_shares
    end_value = trade.exit_price * trade.exit_shares
    assert 0.7 < end_value / start_value < 1.4
    curve = result.equity_curve["equity"]
    before = curve.loc[pd.Timestamp(split_day) - pd.tseries.offsets.BDay(1)]
    on_day = curve.loc[pd.Timestamp(split_day)]
    assert abs(on_day / before - 1) < 0.05


# --------------------------------------------------------------------- earnings


def earnings_symbol() -> str:
    return symbol_of(fake.PLANTED["earnings_after_market"]["ticker"])


def test_earnings_after_market_counts_from_the_next_session(
    planted: tuple[RawWorld, MarketData],
) -> None:
    _, market = planted
    spec = fake.PLANTED["earnings_after_market"]
    assert spec["report_date"] == "2021-04-27"
    assert spec["reaction_session"] == "2021-04-28"
    symbol = earnings_symbol()

    def to_next(day: str) -> float:
        return measure_days_to_next_earnings(slice_to_decision_date(market, d(day)))[symbol]

    def since(day: str) -> float:
        return measure_days_since_last_earnings(slice_to_decision_date(market, d(day)))[symbol]

    # The report is after the close of 27 April, so the market first reacts on the 28th.
    assert to_next("2021-04-27") == 1
    assert to_next("2021-04-26") == 2
    assert to_next("2021-04-22") == 4  # 23, 26, 27, 28 April
    assert to_next("2021-04-14") == 10  # the edge of what is visible
    assert pd.isna(to_next("2021-04-13"))  # 11 sessions ahead: not visible
    assert pd.isna(to_next("2021-04-28"))  # the reaction has happened; the next report is far off
    assert since("2021-04-27") > 30  # the previous reaction was 2021-02-02
    assert since("2021-04-28") == 0
    assert since("2021-04-29") == 1
    assert since("2021-05-12") == 10


def test_earnings_blocks_in_the_backtest(planted: tuple[RawWorld, MarketData]) -> None:
    _, market = planted
    symbol = earnings_symbol()
    near = make_strategy(
        [{"block": "days_to_earnings", "relation": "within", "days": 1}], {"holding_days": 2}
    )
    away = make_strategy(
        [{"block": "days_to_earnings", "relation": "not_within", "days": 5}], {"holding_days": 2}
    )
    since = make_strategy(
        [{"block": "days_since_earnings", "relation": "within", "days": 0}], {"holding_days": 2}
    )
    on_27 = slice_to_decision_date(market, d("2021-04-27"))
    on_22 = slice_to_decision_date(market, d("2021-04-22"))
    on_28 = slice_to_decision_date(market, d("2021-04-28"))
    assert bool(entry_signals(on_27, near)[symbol])
    assert not bool(entry_signals(on_22, near)[symbol])
    assert not bool(entry_signals(on_22, away)[symbol])  # 4 sessions ahead is within 5
    assert bool(entry_signals(on_27, away)[symbol]) is False
    # Not within 5 sessions: false at 5 sessions out, true at 6, 10, and when nothing is visible.
    for day, expected in (
        ("2021-04-21", False),
        ("2021-04-20", True),
        ("2021-04-14", True),
        ("2021-04-13", True),
    ):
        signal = entry_signals(slice_to_decision_date(market, d(day)), away)[symbol]
        assert bool(signal) is expected, day
    assert bool(entry_signals(on_28, since)[symbol])
    result = run(since, market, "2021-04-01", "2021-05-31")
    reaction = [t for t in trades_for(result, "SIM004") if t.signal_date == d("2021-04-28")]
    assert len(reaction) == 1
    assert reaction[0].entry_date == d("2021-04-29")


# -------------------------------------------------- the whole fake universe, a short run


def test_whole_universe_run_has_raw_prices_and_sane_equity() -> None:
    codes = fake_universe_codes()
    world = load_raw_world(codes)
    market = build_world_market(world)
    strategy = make_strategy([GAP_DOWN, NO_NEWS], {"holding_days": 3, "stop_loss_pct": 8}, 10)
    result = run(strategy, market, "2021-01-04", "2022-12-30")
    curve = result.equity_curve["equity"]
    assert len(curve) > 400
    assert (curve > 0).all()
    assert result.equity_curve["cash"].min() >= -1e-6  # no leverage
    assert (result.equity_curve["open_positions"] <= 10).all()
    for trade in result.trades:
        code = trade.symbol.removesuffix(".US")
        assert trade.entry_price == raw_open_on(world, code, trade.entry_date.isoformat())
        assert trade.entry_date > trade.signal_date
        assert trade.exit_date >= trade.entry_date
