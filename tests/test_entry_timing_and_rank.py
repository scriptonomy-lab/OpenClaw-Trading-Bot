"""Same-open entries, rank_by, the tie-break salt and the open news window.

Toy markets are typed in by hand so each number can be worked out on paper.
The planted-world tests use the fake EODHD data (SIMULATED DATA).
"""

import copy
import hashlib
import json
from datetime import date
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from bot.backtest.costs import CostModel
from bot.backtest.engine import (
    INITIAL_CAPITAL,
    entry_priority,
    initial_state,
    run_backtest,
    step,
)
from bot.data.news import (
    build_news_frame,
    open_news_window,
    parse_news_rows,
)
from bot.paper.trader import replay_paper
from bot.strategy.blocks import rank_scores, slice_to_decision_date
from bot.strategy.schema import (
    BacktestWindow,
    StrategyDefinition,
    describe_rules,
    explain_validation_error,
    strategy_from_json,
    strategy_id,
    strategy_to_json,
)
from tests.backtest_helpers import (
    Bar,
    day_of,
    engine_market,
    flat_bar,
    planted_market,
    raw_open_on,
    round_trip_market,
    run_toy,
    symbol_of,
)

NO_COST = CostModel(0.0)

GAP_DOWN_EXAMPLE: dict[str, Any] = {
    "name": "Gap down, no news",
    "description_plain_english": (
        "Buy S&P 500 stocks whose open is 3% or more below the previous close, with no news "
        "since the previous close. Sell after 3 trading days."
    ),
    "version": 1,
    "entry": [
        {"block": "gap", "direction": "down", "min_gap_pct": 3},
        {"block": "no_news", "lookback_sessions": 1},
    ],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}


def build(spec: dict[str, Any]) -> StrategyDefinition:
    return strategy_from_json(json.dumps(spec))


def gap_strategy(
    *,
    timing: str = "same_open",
    direction: str = "up",
    holding_days: int | None = 3,
    stop_loss_pct: float | None = None,
    max_positions: int = 10,
    rank_by: str | None = None,
) -> StrategyDefinition:
    """A one-block gap strategy, so a signal is set by typing an open price."""
    exit_rules: dict[str, Any] = {}
    if holding_days is not None:
        exit_rules["holding_days"] = holding_days
    if stop_loss_pct is not None:
        exit_rules["stop_loss_pct"] = stop_loss_pct
    spec: dict[str, Any] = {
        "name": "Toy gap",
        "description_plain_english": "Buy a gap of a few percent for the timing tests.",
        "entry": [{"block": "gap", "direction": direction, "min_gap_pct": 4}],
        "exit": exit_rules,
        "sizing": {"max_positions": max_positions},
        "entry_timing": timing,
    }
    if rank_by is not None:
        spec["rank_by"] = rank_by
    return build(spec)


# ------------------------------------------------------------------ schema


def test_existing_strategy_ids_do_not_change() -> None:
    # Pinned before entry_timing and rank_by existed. Defaults must not move an id (D12).
    assert strategy_id(build(GAP_DOWN_EXAMPLE)) == "strat_cc4f3f996f50717d"
    spelled_out = dict(GAP_DOWN_EXAMPLE, entry_timing="next_open", rank_by=None)
    assert strategy_id(build(spelled_out)) == "strat_cc4f3f996f50717d"


def test_timing_and_rank_by_change_the_id() -> None:
    base = strategy_id(build(GAP_DOWN_EXAMPLE))
    same_open = strategy_id(build(dict(GAP_DOWN_EXAMPLE, entry_timing="same_open")))
    ranked = strategy_id(build(dict(GAP_DOWN_EXAMPLE, rank_by="largest_gap")))
    assert len({base, same_open, ranked}) == 3


def test_new_fields_round_trip_through_json() -> None:
    strategy = build(dict(GAP_DOWN_EXAMPLE, entry_timing="same_open", rank_by="largest_gap"))
    assert strategy_from_json(strategy_to_json(strategy)) == strategy


def errors_for(spec: dict[str, Any]) -> list[str]:
    with pytest.raises(ValidationError) as caught:
        build(spec)
    return explain_validation_error(caught.value)


def test_same_open_without_a_gap_is_rejected_in_plain_english() -> None:
    spec = copy.deepcopy(GAP_DOWN_EXAMPLE)
    spec["entry"] = [{"block": "no_news", "lookback_sessions": 1}]
    spec["entry_timing"] = "same_open"
    lines = errors_for(spec)
    assert len(lines) == 1
    assert lines[0].startswith("entry_timing 'same_open' needs a gap condition.")


def test_unknown_timing_is_rejected() -> None:
    lines = errors_for(dict(GAP_DOWN_EXAMPLE, entry_timing="at_noon"))
    assert lines == ["entry_timing must be one of 'next_open' or 'same_open'. You gave 'at_noon'."]


def test_rank_by_needs_its_block() -> None:
    lines = errors_for(dict(GAP_DOWN_EXAMPLE, rank_by="lowest_rsi"))
    assert lines == [
        "rank_by 'lowest_rsi' needs an entry condition with block 'rsi'. Add one or remove rank_by."
    ]


def test_rank_by_with_two_different_look_backs_is_rejected() -> None:
    spec = copy.deepcopy(GAP_DOWN_EXAMPLE)
    spec["entry"] += [
        {"block": "rsi", "period": 5, "comparison": "at_most", "level": 40},
        {"block": "rsi", "period": 14, "comparison": "at_most", "level": 50},
    ]
    spec["rank_by"] = "lowest_rsi"
    assert errors_for(spec)[0].startswith("rank_by 'lowest_rsi' is unclear")
    # The same look-back twice (a range) is fine: both conditions read one number.
    spec["entry"][-1]["period"] = 5
    assert build(spec).rank_by == "lowest_rsi"


def test_describe_rules_says_what_is_read_when() -> None:
    spec = copy.deepcopy(GAP_DOWN_EXAMPLE)
    spec["entry"].append({"block": "rsi", "period": 14, "comparison": "at_most", "level": 40})
    next_text = describe_rules(build(spec))
    assert "Buy at the next open when ALL of these are true at the close:" in next_text
    spec["entry_timing"] = "same_open"
    spec["rank_by"] = "largest_gap"
    text = describe_rules(build(spec))
    assert "Buy at the open (09:30 New York) of the same day" in text
    assert "Today's open is at least 3% below yesterday's close." in text
    assert "from the previous close (16:00 New York) to today's open (09:30 New York)" in text
    assert "The 14-day RSI is at most 40, as of the previous close." in text
    assert "buy the largest gaps (in either direction) first" in text


# --------------------------------------------------------- the open news window


def test_open_window_ends_at_0930_new_york_and_is_half_open() -> None:
    days = pd.DatetimeIndex(["2022-02-09", "2022-02-10"], name="date")
    rows = [
        {"date": "2022-02-09T21:00:00+00:00", "title": "a", "link": "a", "symbols": ["X.US"]},
        {"date": "2022-02-10T14:29:59+00:00", "title": "b", "link": "b", "symbols": ["X.US"]},
        {"date": "2022-02-10T14:30:00+00:00", "title": "c", "link": "c", "symbols": ["X.US"]},
        {"date": "2022-02-10T20:59:00+00:00", "title": "d", "link": "d", "symbols": ["X.US"]},
    ]
    articles = parse_news_rows(rows, "X.US")  # type: ignore[arg-type]
    frame = build_news_frame(articles, pd.Series(dtype="float64"), days)
    # 16:00 EST on the 9th is 21:00 UTC (in); 09:30 EST on the 10th is 14:30 UTC (out).
    assert frame["open_article_count"].iloc[1] == 2
    assert frame["article_count"].iloc[1] == 4
    assert pd.isna(frame["open_article_count"].iloc[0])


def test_open_window_follows_daylight_saving_time() -> None:
    # DST started on Sunday 2021-03-14. Friday 16:00 is EST (21:00 UTC), Monday 09:30 is
    # EDT (13:30 UTC). A fixed EST offset would put the end at 14:30 UTC.
    days = pd.DatetimeIndex(["2021-03-12", "2021-03-15"], name="date")
    start, end = open_news_window(date(2021, 3, 15), days)
    assert start.isoformat() == "2021-03-12T16:00:00-05:00"
    assert end.isoformat() == "2021-03-15T09:30:00-04:00"
    rows = [
        {"date": "2021-03-15T13:29:00+00:00", "title": "in", "link": "1", "symbols": ["X.US"]},
        {"date": "2021-03-15T13:31:00+00:00", "title": "out", "link": "2", "symbols": ["X.US"]},
    ]
    articles = parse_news_rows(rows, "X.US")  # type: ignore[arg-type]
    frame = build_news_frame(articles, pd.Series(dtype="float64"), days)
    assert frame["open_article_count"].iloc[1] == 1


# ------------------------------------------------------------- the engine


def test_same_open_buys_at_the_open_of_the_gap_day() -> None:
    # round_trip_market: day 1 opens 105 after a close of 100 (a 5% gap up).
    market = round_trip_market()
    same = run_toy(gap_strategy(), market)
    trade = same.trades[0]
    assert trade.signal_date == trade.entry_date == day_of(market, 1)
    assert trade.entry_price == 105.0
    assert trade.entry_shares == pytest.approx(10_000.0 / 105.0)
    # Held on days 1, 2 and 3, sold at the open of day 4.
    assert trade.exit_date == day_of(market, 4) and trade.exit_price == 103.0
    assert trade.trading_days_held == 3
    assert trade.signal_values["gap_pct"] == pytest.approx(5.0)
    following = run_toy(gap_strategy(timing="next_open"), market)
    assert following.trades[0].entry_date == day_of(market, 2)


def sell_then_signal_market(a_open_day_3: float) -> Any:
    """A bought on day 1 (gap up), sold at the open of day 3. B gaps up 5% on day 3."""
    return engine_market(
        5,
        {
            "A.US": {
                0: flat_bar(100.0),
                1: Bar(105.0, 111.0, 104.0, 110.0),
                2: flat_bar(110.0),
                3: Bar(a_open_day_3, a_open_day_3, a_open_day_3, a_open_day_3),
                4: flat_bar(a_open_day_3),
            },
            "B.US": {
                0: flat_bar(50.0),
                1: flat_bar(50.0),
                2: flat_bar(50.0),
                3: flat_bar(52.5),
                4: flat_bar(52.5),
            },
        },
    )


def test_same_open_sizes_from_the_previous_close_and_skips_the_stock_just_sold() -> None:
    # A opens 5% up on day 3 too, but it is sold at that open, so it is not bought back.
    market = sell_then_signal_market(115.5)
    result = run_toy(gap_strategy(holding_days=2, max_positions=1), market)
    entries = [(t.symbol, t.entry_date) for t in result.trades]
    assert entries == [("A.US", day_of(market, 1)), ("B.US", day_of(market, 3))]
    b_trade = result.trades[1]
    # Target: equity at the close of day 2 (A's shares at 110) over 1 position. Cash after
    # selling A at 115.5 is larger, so the target binds.
    assert b_trade.entry_price * b_trade.entry_shares == pytest.approx(100_000.0 / 105.0 * 110.0)


def test_same_open_buy_is_limited_by_cash_after_the_opening_sells() -> None:
    market = sell_then_signal_market(104.5)
    result = run_toy(gap_strategy(holding_days=2, max_positions=1), market)
    b_trade = next(t for t in result.trades if t.symbol == "B.US")
    assert b_trade.entry_date == day_of(market, 3)
    assert b_trade.entry_price * b_trade.entry_shares == pytest.approx(100_000.0 / 105.0 * 104.5)


def test_stop_is_checked_on_the_same_open_fill_day() -> None:
    market = engine_market(
        4,
        {"S.US": {0: flat_bar(100.0), 1: Bar(105.0, 106.0, 100.0, 101.0), 2: flat_bar(101.0)}},
    )
    result = run_toy(gap_strategy(holding_days=None, stop_loss_pct=2), market, last_day=2)
    trade = result.trades[0]
    assert trade.entry_date == trade.exit_date == day_of(market, 1)
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(105.0 * 0.98)
    assert trade.trading_days_held == 1


def test_same_open_entries_on_the_last_day_fill_inside_the_window() -> None:
    market = round_trip_market()
    same = run_toy(gap_strategy(), market, last_day=1)
    assert [(t.entry_date, t.exit_reason) for t in same.trades] == [
        (day_of(market, 1), "open_at_end")
    ]
    following = run_toy(gap_strategy(timing="next_open"), market, last_day=1)
    assert following.trades == []


def test_first_day_of_the_data_buys_nothing_at_the_open() -> None:
    market = round_trip_market()
    data = slice_to_decision_date(market, day_of(market, 0))
    result = step(initial_state(), data, gap_strategy(), NO_COST, allow_new_entries=True)
    assert result.state.positions == {}
    assert result.equity.equity == INITIAL_CAPITAL


def three_gaps_market() -> Any:
    """Three stocks gap down on day 2: 4%, 6% and 5%."""
    stocks = {}
    for symbol, gap in (("G4.US", 4.0), ("G6.US", 6.0), ("G5.US", 5.0)):
        opened = 100.0 * (1 - gap / 100)
        stocks[symbol] = {
            0: flat_bar(100.0),
            1: flat_bar(100.0),
            2: Bar(opened, opened, opened, opened),
            3: flat_bar(opened),
            4: flat_bar(opened),
        }
    return engine_market(5, stocks)


@pytest.mark.parametrize("timing", ["next_open", "same_open"])
def test_rank_by_largest_gap_takes_the_biggest_gap_first(timing: str) -> None:
    market = three_gaps_market()
    strategy = gap_strategy(timing=timing, direction="down", max_positions=1, rank_by="largest_gap")
    result = run_toy(strategy, market)
    assert [t.symbol for t in result.trades] == ["G6.US"]
    assert any("2 entry signals did not get a slot" in note for note in result.notes)


def test_rank_scores_turn_lowest_rsi_into_highest_score() -> None:
    days = 20
    rising = {n: flat_bar(100.0 + n) for n in range(days)}
    falling = {n: flat_bar(200.0 - n) for n in range(days)}
    market = engine_market(days, {"UP.US": rising, "DOWN.US": falling})
    spec = {
        "name": "Rsi rank",
        "description_plain_english": "Rank by the lowest RSI for a unit test.",
        "entry": [{"block": "rsi", "period": 5, "comparison": "at_least", "level": 1}],
        "exit": {"holding_days": 1},
        "sizing": {"max_positions": 1},
        "rank_by": "lowest_rsi",
    }
    data = slice_to_decision_date(market, day_of(market, days - 1))
    scores = rank_scores(data, build(spec), ["UP.US", "DOWN.US"])
    assert scores["DOWN.US"] > scores["UP.US"]


# ------------------------------------------------------------- the salt


SYMBOLS = [f"S{n:02d}.US" for n in range(12)]


def test_no_salt_is_the_documented_hash_order() -> None:
    day = date(2023, 3, 2)

    def key(symbol: str) -> str:
        return hashlib.sha256(f"strat_x|{day.isoformat()}|{symbol}".encode()).hexdigest()

    assert entry_priority("strat_x", day, SYMBOLS) == sorted(SYMBOLS, key=key)
    assert entry_priority("strat_x", day, SYMBOLS, salt="") == sorted(SYMBOLS, key=key)


def test_a_salt_gives_another_repeatable_order() -> None:
    day = date(2023, 3, 2)
    plain = entry_priority("strat_x", day, SYMBOLS)
    salted = entry_priority("strat_x", day, SYMBOLS, salt="salt-1")
    assert salted != plain and sorted(salted) == sorted(plain)
    assert entry_priority("strat_x", day, list(reversed(SYMBOLS)), salt="salt-1") == salted


def test_scores_come_first_and_ties_use_the_hash() -> None:
    day = date(2023, 3, 2)
    scores = pd.Series({"A.US": 1.0, "B.US": 3.0, "C.US": 1.0})
    order = entry_priority("strat_x", day, ["A.US", "B.US", "C.US"], scores=scores)
    assert order[0] == "B.US"
    assert order[1:] == entry_priority("strat_x", day, ["A.US", "C.US"])


def test_salt_reaches_run_backtest() -> None:
    market = three_gaps_market()
    strategy = gap_strategy(timing="next_open", direction="down", max_positions=1)
    days = market.trading_days
    window = BacktestWindow("primary", days[0].date(), days[-1].date(), None)
    picks = {
        run_backtest(strategy, market, window, NO_COST, tiebreak_salt=salt).trades[0].symbol
        for salt in ("", "a", "b", "c", "d", "e", "f")
    }
    assert len(picks) > 1


# ------------------------------------------------ planted world (SIMULATED DATA)


SAME_OPEN_GAP_DOWN = dict(GAP_DOWN_EXAMPLE, entry_timing="same_open")


def test_planted_quiet_gap_down_is_bought_at_its_own_open() -> None:
    world, market = planted_market()
    window = BacktestWindow("primary", date(2022, 1, 3), date(2022, 7, 29), None)
    result = run_backtest(build(SAME_OPEN_GAP_DOWN), market, window, CostModel(5.0))
    quiet = [t for t in result.trades if t.symbol == symbol_of("SIM001")]
    planted = [t for t in quiet if t.entry_date == date(2022, 2, 10)]
    assert len(planted) == 1
    assert planted[0].signal_date == date(2022, 2, 10)
    assert planted[0].entry_price == raw_open_on(world, "SIM001", "2022-02-10")
    # SIM002 has articles before 09:30 on 2022-06-14, so it is skipped at that open too.
    assert not [
        t
        for t in result.trades
        if t.symbol == symbol_of("SIM002") and t.entry_date == date(2022, 6, 14)
    ]
    assert all(t.signal_date == t.entry_date for t in result.trades)


def test_same_open_paper_replay_equals_the_backtest_every_day() -> None:
    _, market = planted_market()
    window = BacktestWindow("primary", date(2022, 1, 3), date(2022, 7, 29), None)
    strategy = build(SAME_OPEN_GAP_DOWN)
    expected = run_backtest(strategy, market, window, CostModel(5.0))
    replayed = replay_paper(strategy, market, window, CostModel(5.0))
    assert expected.trades
    assert replayed.trades == expected.trades
    pd.testing.assert_frame_equal(replayed.equity_curve, expected.equity_curve)
    final = expected.equity_curve["equity"].iloc[-1]
    assert final == pytest.approx(INITIAL_CAPITAL + sum(t.pnl for t in expected.trades))
