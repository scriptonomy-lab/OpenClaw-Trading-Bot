"""The lookahead test: signals from data known at D equal signals from full history at D.

This is the strict variant from docs/design/backtest.md. For each sampled
decision date D, with the fake EODHD world (``EODHD_MODE=fake``):

1. Full history: build MarketData from all pre-holdout raw rows, slice at D.
2. Rebuilt at D: cut the raw rows to what was known at D, rebuild the frames
   from scratch, build MarketData, slice at D.

How each source is cut for the rebuild (see ``build_world_market``):

- Prices, splits, dividends: rows dated on or before D.
- News: articles published before 16:00 New York on D.
- Daily sentiment: rows dated before D.
- Membership: the table up to D.
- Earnings: the designed exception (D15). ``days_to_next`` may see a reaction
  session up to 10 trading days after D, and an after-market report needs the
  full calendar to be placed. So events whose earliest reaction session is at
  most 10 trading days after D are kept, aligned on the full calendar. The
  tests below check that nothing further out is visible.

Rebuilding matters because split anchoring moves adjusted price levels when a
split falls between D and 2023. Only scale-free blocks pass.

Same-open entries (``entry_timing`` "same_open") get the same test with the
cut moved to 09:30 New York on D: the rebuild keeps D's row but overwrites its
high, low and close with the open and multiplies its volume, for every stock
and SPY, and keeps only news published before 09:30 on D. Signals, measures,
rank numbers and stored signal values from ``slice_to_open`` must equal those
from full history. Teeth tests check that the overwrite really changes what a
close-time read of D would see.

Runtime is a few seconds: 24 tickers, about 24 dates, 20 conditions.
"""

import random
from collections import Counter
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot.data.earnings import EARNINGS_VISIBILITY_TRADING_DAYS
from bot.strategy.blocks import (
    DecisionData,
    OpenDecisionData,
    evaluate_condition,
    evaluate_condition_at_open,
    measure_close_vs_ma_pct,
    measure_days_since_last_earnings,
    measure_days_to_next_earnings,
    measure_gap_pct,
    measure_gap_pct_at_open,
    measure_n_day_return_pct,
    measure_news_count,
    measure_news_count_at_open,
    measure_news_polarity,
    measure_news_polarity_at_open,
    measure_relative_volume,
    measure_rsi,
    measure_spy_vs_ma_pct,
    rank_scores,
    signal_values_at_open,
    slice_to_decision_date,
    slice_to_open,
)
from bot.strategy.schema import (
    DaysSinceEarningsCondition,
    DaysToEarningsCondition,
    EntryCondition,
    GapCondition,
    NDayReturnCondition,
    NewsSentimentCondition,
    NoNewsCondition,
    PriceVsMovingAverageCondition,
    RelativeVolumeCondition,
    RsiCondition,
    SpyFilterCondition,
    StrategyDefinition,
    strategy_from_json,
)
from tests.backtest_helpers import (
    RawWorld,
    build_world_market,
    fake_universe_codes,
    load_raw_world,
    symbol_of,
)

PLANTED_CODES = ("SIM001", "SIM002", "SIM003", "SIM004", "SIM007", "SIM008")
SEED = 20260501
RANDOM_DATE_COUNT = 18
# Dates chosen because something happens near them: a split ex-date (and the day before),
# the planted gap-downs, the planted earnings report, an index removal, a delisting,
# and the last pre-holdout trading day.
SPECIAL_DATES = (
    date(2017, 6, 9),
    date(2017, 6, 12),
    date(2021, 4, 14),
    date(2021, 4, 27),
    date(2022, 2, 10),
    date(2018, 6, 18),
    date(2020, 4, 17),
    date(2023, 12, 29),
)


def chosen_codes() -> tuple[str, ...]:
    """The planted tickers plus every fourth other ticker, 24 in all."""
    others = [c for c in fake_universe_codes() if c not in PLANTED_CODES]
    return tuple(sorted({*PLANTED_CODES, *others[::4][: 24 - len(PLANTED_CODES)]}))


@pytest.fixture(scope="module")
def world() -> RawWorld:
    return load_raw_world(chosen_codes())


@pytest.fixture(scope="module")
def full_market(world: RawWorld):  # type: ignore[no-untyped-def]
    return build_world_market(world)


@pytest.fixture(scope="module")
def sample_dates(full_market) -> list[date]:  # type: ignore[no-untyped-def]
    """Seeded random trading days from 2016-03 to 2023-12, plus the special dates."""
    days = [
        d.date()
        for d in full_market.trading_days
        if date(2016, 3, 1) <= d.date() <= date(2023, 12, 29)
    ]
    picked = random.Random(SEED).sample(days, RANDOM_DATE_COUNT)
    return sorted({*picked, *SPECIAL_DATES})


@pytest.fixture(scope="module")
def rebuilt(world: RawWorld, sample_dates: list[date]) -> dict[date, DecisionData]:
    """For each sampled D: MarketData rebuilt from raw rows cut at D, sliced at D."""
    return {d: slice_to_decision_date(build_world_market(world, cut=d), d) for d in sample_dates}


@pytest.fixture(scope="module")
def full(full_market, sample_dates: list[date]) -> dict[date, DecisionData]:  # type: ignore[no-untyped-def]
    return {d: slice_to_decision_date(full_market, d) for d in sample_dates}


CONDITIONS: dict[str, EntryCondition] = {
    "gap_down": GapCondition(block="gap", direction="down", min_gap_pct=1),
    "gap_up": GapCondition(block="gap", direction="up", min_gap_pct=1),
    "n_day_return_5": NDayReturnCondition(
        block="n_day_return", days=5, comparison="at_least", return_pct=0
    ),
    "n_day_return_60": NDayReturnCondition(
        block="n_day_return", days=60, comparison="at_most", return_pct=5
    ),
    "above_ma_20": PriceVsMovingAverageCondition(
        block="price_vs_moving_average", ma_days=20, position="above"
    ),
    "below_ma_50": PriceVsMovingAverageCondition(
        block="price_vs_moving_average", ma_days=50, position="below"
    ),
    "rsi_14_low": RsiCondition(block="rsi", period=14, comparison="at_most", level=45),
    "rsi_5_high": RsiCondition(block="rsi", period=5, comparison="at_least", level=60),
    "rel_volume_high": RelativeVolumeCondition(
        block="relative_volume", lookback_days=20, comparison="at_least", ratio=1.1
    ),
    "rel_volume_low": RelativeVolumeCondition(
        block="relative_volume", lookback_days=5, comparison="at_most", ratio=1
    ),
    "no_news_1": NoNewsCondition(block="no_news", lookback_sessions=1),
    "no_news_3": NoNewsCondition(block="no_news", lookback_sessions=3),
    "sentiment_up": NewsSentimentCondition(
        block="news_sentiment",
        lookback_sessions=1,
        comparison="at_least",
        polarity=0,
        min_articles=1,
    ),
    "sentiment_down": NewsSentimentCondition(
        block="news_sentiment",
        lookback_sessions=3,
        comparison="at_most",
        polarity=0,
        min_articles=2,
    ),
    "earnings_within_5": DaysToEarningsCondition(
        block="days_to_earnings", relation="within", days=5
    ),
    "earnings_not_within_3": DaysToEarningsCondition(
        block="days_to_earnings", relation="not_within", days=3
    ),
    "since_earnings_within_3": DaysSinceEarningsCondition(
        block="days_since_earnings", relation="within", days=3
    ),
    "since_earnings_not_within_10": DaysSinceEarningsCondition(
        block="days_since_earnings", relation="not_within", days=10
    ),
    "spy_above_50": SpyFilterCondition(block="spy_filter", ma_days=50, position="above"),
    "spy_below_200": SpyFilterCondition(block="spy_filter", ma_days=200, position="below"),
}

MEASURES: dict[str, Callable[[DecisionData], pd.Series | float]] = {
    "gap_pct": measure_gap_pct,
    "n_day_return_5": lambda data: measure_n_day_return_pct(data, 5),
    "n_day_return_60": lambda data: measure_n_day_return_pct(data, 60),
    "close_vs_ma_20": lambda data: measure_close_vs_ma_pct(data, 20),
    "close_vs_ma_200": lambda data: measure_close_vs_ma_pct(data, 200),
    "rsi_14": lambda data: measure_rsi(data, 14),
    "rsi_5": lambda data: measure_rsi(data, 5),
    "relative_volume_20": lambda data: measure_relative_volume(data, 20),
    "news_count_1": lambda data: measure_news_count(data, 1),
    "news_count_3": lambda data: measure_news_count(data, 3),
    "news_polarity_3": lambda data: measure_news_polarity(data, 3),
    "days_to_next": measure_days_to_next_earnings,
    "days_since_last": measure_days_since_last_earnings,
    "spy_vs_ma_50": lambda data: measure_spy_vs_ma_pct(data, 50),
    "spy_vs_ma_200": lambda data: measure_spy_vs_ma_pct(data, 200),
}


# ------------------------------------------------------------ the sample


def test_sample_is_seeded_and_covers_the_special_dates(sample_dates: list[date]) -> None:
    assert len(sample_dates) >= RANDOM_DATE_COUNT
    assert set(SPECIAL_DATES) <= set(sample_dates)
    assert all(date(2016, 3, 1) <= d <= date(2023, 12, 29) for d in sample_dates)
    assert all(d < date(2024, 1, 1) for d in sample_dates)  # nothing near the holdout


def test_universe_includes_planted_names_and_enough_others(world: RawWorld) -> None:
    assert set(PLANTED_CODES) <= set(world.codes)
    assert len(world.codes) >= 20


# ----------------------------------------------------- one test per block


@pytest.mark.parametrize("name", list(CONDITIONS))
def test_block_signals_equal_with_data_cut_at_the_decision_date(
    name: str,
    full: dict[date, DecisionData],
    rebuilt: dict[date, DecisionData],
) -> None:
    condition = CONDITIONS[name]
    true_count = 0
    checked = 0
    for day, full_data in full.items():
        cut_data = rebuilt[day]
        assert list(cut_data.members) == list(full_data.members), day
        expected = evaluate_condition(full_data, condition)
        actual = evaluate_condition(cut_data, condition)
        pd.testing.assert_series_equal(actual, expected, check_names=False, obj=f"{name} at {day}")
        true_count += int(expected.sum())
        checked += len(expected)
    # The test is only worth something if the blocks actually fire and actually refuse.
    assert checked > 0
    assert true_count > 0, f"{name} never fired on the sampled days, so the test proved nothing"
    assert true_count < checked, f"{name} fired for every stock on every day"


@pytest.mark.parametrize("name", list(MEASURES))
def test_measurements_equal_with_data_cut_at_the_decision_date(
    name: str,
    full: dict[date, DecisionData],
    rebuilt: dict[date, DecisionData],
) -> None:
    measure = MEASURES[name]
    finite = 0
    for day, full_data in full.items():
        expected = np.atleast_1d(np.asarray(measure(full_data), dtype="float64"))
        actual = np.atleast_1d(np.asarray(measure(rebuilt[day]), dtype="float64"))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
        finite += int(np.isfinite(expected).sum())
    assert finite > 0, f"{name} was NaN everywhere"


# ---------------------------------------------- the harness has teeth


def test_rebuilding_changes_adjusted_price_levels_across_a_split(
    world: RawWorld,
    full_market,
    rebuilt: dict[date, DecisionData],  # type: ignore[no-untyped-def]
) -> None:
    # SIM003 splits 4-for-1 on 2017-06-12. At the close of 2017-06-09 the full-history frame is
    # divided by 4 (anchored on the last split before 2024). The rebuilt frame is not. So an
    # absolute price level would differ, which is exactly what the strict test is there to catch.
    day = date(2017, 6, 9)
    symbol = symbol_of("SIM003")
    full_close = full_market.close.loc[pd.Timestamp(day), symbol]
    cut_close = rebuilt[day].market.close.loc[pd.Timestamp(day), symbol]
    assert cut_close / full_close == pytest.approx(4.0)
    # The raw close is the same either way.
    full_raw = full_market.raw_close.loc[pd.Timestamp(day), symbol]
    assert rebuilt[day].market.raw_close.loc[pd.Timestamp(day), symbol] == full_raw


def test_nothing_is_dated_after_the_decision_date(rebuilt: dict[date, DecisionData]) -> None:
    for day, data in rebuilt.items():
        stamp = pd.Timestamp(day)
        assert data.market.trading_days.max() == stamp
        for frame in (
            data.market.open,
            data.market.close,
            data.market.total_return_index,
            data.market.article_count,
            data.market.days_to_next_earnings,
            data.market.membership,
        ):
            assert frame.index.max() == stamp


# ------------------------------------------------- earnings: the designed exception


def test_earnings_values_are_never_further_ahead_than_ten_sessions(full_market) -> None:  # type: ignore[no-untyped-def]
    table = full_market.days_to_next_earnings
    assert table.max().max() <= EARNINGS_VISIBILITY_TRADING_DAYS
    assert table.min().min() >= 1  # today's reaction session has already happened by the close


def test_next_earnings_is_the_one_place_a_block_sees_past_the_decision_date(
    world: RawWorld, full: dict[date, DecisionData]
) -> None:
    # SIM004 reports after the market on 2021-04-27, so the reaction session is 2021-04-28.
    day = date(2021, 4, 27)
    symbol = symbol_of("SIM004")
    assert measure_days_to_next_earnings(full[day])[symbol] == 1
    # A rebuild that keeps no event after D (visibility 0) cannot know that date. The designed
    # rebuild (visibility 10) does. This is why the test keeps events up to 10 sessions ahead.
    strict = slice_to_decision_date(build_world_market(world, cut=day, earnings_visibility=0), day)
    assert np.isnan(measure_days_to_next_earnings(strict)[symbol])


def test_events_beyond_ten_sessions_are_invisible_either_way(
    world: RawWorld, full: dict[date, DecisionData], rebuilt: dict[date, DecisionData]
) -> None:
    # Keeping events up to 20 sessions ahead must change nothing, because values past 10 are
    # NaN by design. So the 10-session rebuild already shows everything a block could see.
    for day in (date(2021, 4, 14), date(2021, 4, 27), date(2022, 2, 10)):
        wide = slice_to_decision_date(
            build_world_market(world, cut=day, earnings_visibility=20), day
        )
        np.testing.assert_array_equal(
            measure_days_to_next_earnings(wide).to_numpy(),
            measure_days_to_next_earnings(rebuilt[day]).to_numpy(),
        )
        np.testing.assert_array_equal(
            measure_days_to_next_earnings(wide).to_numpy(),
            measure_days_to_next_earnings(full[day]).to_numpy(),
        )


# ------------------------------------------------ same-open: the cut at 09:30 on D


@pytest.fixture(scope="module")
def rebuilt_open(world: RawWorld, sample_dates: list[date]) -> dict[date, OpenDecisionData]:
    """For each sampled D: MarketData rebuilt from what was known at 09:30 on D."""
    return {
        d: slice_to_open(build_world_market(world, cut=d, at_open=True), d) for d in sample_dates
    }


@pytest.fixture(scope="module")
def full_open(full_market, sample_dates: list[date]) -> dict[date, OpenDecisionData]:  # type: ignore[no-untyped-def]
    return {d: slice_to_open(full_market, d) for d in sample_dates}


@pytest.fixture(scope="module")
def rebuilt_open_at_close(world: RawWorld, sample_dates: list[date]) -> dict[date, DecisionData]:
    """The same 09:30 rebuild, read the wrong way (as if at D's close), for the teeth tests."""
    return {
        d: slice_to_decision_date(build_world_market(world, cut=d, at_open=True), d)
        for d in sample_dates
    }


OPEN_MEASURES: dict[str, Callable[[OpenDecisionData], pd.Series]] = {
    "gap_pct": measure_gap_pct_at_open,
    "news_count_1": lambda data: measure_news_count_at_open(data, 1),
    "news_count_3": lambda data: measure_news_count_at_open(data, 3),
    "news_polarity_1": lambda data: measure_news_polarity_at_open(data, 1),
    "news_polarity_3": lambda data: measure_news_polarity_at_open(data, 3),
}


def ranked_strategy(entry: list[dict[str, object]], rank_by: str) -> StrategyDefinition:
    """A same-open strategy with a gap block, the given extra block and ``rank_by``."""
    import json

    return strategy_from_json(
        json.dumps(
            {
                "name": "Lookahead rank probe",
                "description_plain_english": "Only used to compute rank numbers in a test.",
                "entry": [{"block": "gap", "direction": "down", "min_gap_pct": 1}, *entry],
                "exit": {"holding_days": 3},
                "sizing": {"max_positions": 5},
                "entry_timing": "same_open",
                "rank_by": rank_by,
            }
        )
    )


RANK_STRATEGIES: dict[str, StrategyDefinition] = {
    "largest_gap": ranked_strategy([], "largest_gap"),
    "highest_sentiment": ranked_strategy(
        [
            {
                "block": "news_sentiment",
                "lookback_sessions": 2,
                "comparison": "at_least",
                "polarity": -1,
            }
        ],
        "highest_sentiment",
    ),
    "strongest_n_day_return": ranked_strategy(
        [{"block": "n_day_return", "days": 20, "comparison": "at_least", "return_pct": -90}],
        "strongest_n_day_return",
    ),
    "highest_relative_volume": ranked_strategy(
        [{"block": "relative_volume", "comparison": "at_least", "ratio": 0.1}],
        "highest_relative_volume",
    ),
    "lowest_rsi": ranked_strategy(
        [{"block": "rsi", "comparison": "at_least", "level": 1}], "lowest_rsi"
    ),
}


@pytest.mark.parametrize("name", list(CONDITIONS))
def test_same_open_signals_equal_with_data_cut_at_the_open(
    name: str,
    full_open: dict[date, OpenDecisionData],
    rebuilt_open: dict[date, OpenDecisionData],
) -> None:
    condition = CONDITIONS[name]
    true_count = 0
    checked = 0
    for day, full_data in full_open.items():
        cut_data = rebuilt_open[day]
        assert list(cut_data.members) == list(full_data.members), day
        expected = evaluate_condition_at_open(full_data, condition)
        actual = evaluate_condition_at_open(cut_data, condition)
        pd.testing.assert_series_equal(actual, expected, check_names=False, obj=f"{name} at {day}")
        true_count += int(expected.sum())
        checked += len(expected)
    assert checked > 0
    assert true_count > 0, f"{name} never fired at the open on the sampled days"
    assert true_count < checked, f"{name} fired for every stock on every day"


@pytest.mark.parametrize("name", list(OPEN_MEASURES))
def test_same_open_measurements_equal_with_data_cut_at_the_open(
    name: str,
    full_open: dict[date, OpenDecisionData],
    rebuilt_open: dict[date, OpenDecisionData],
) -> None:
    measure = OPEN_MEASURES[name]
    finite = 0
    for day, full_data in full_open.items():
        expected = measure(full_data).to_numpy(dtype="float64")
        actual = measure(rebuilt_open[day]).to_numpy(dtype="float64")
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
        finite += int(np.isfinite(expected).sum())
    assert finite > 0, f"{name} was NaN everywhere"


@pytest.mark.parametrize("name", list(RANK_STRATEGIES))
def test_same_open_rank_numbers_equal_with_data_cut_at_the_open(
    name: str,
    full_open: dict[date, OpenDecisionData],
    rebuilt_open: dict[date, OpenDecisionData],
) -> None:
    strategy = RANK_STRATEGIES[name]
    finite = 0
    for day, full_data in full_open.items():
        symbols = list(full_data.members)
        expected = rank_scores(full_data, strategy, symbols).to_numpy(dtype="float64")
        actual = rank_scores(rebuilt_open[day], strategy, symbols).to_numpy(dtype="float64")
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
        finite += int(np.isfinite(expected).sum())
    assert finite > 0


def test_same_open_stored_signal_values_equal_with_data_cut_at_the_open(
    full_open: dict[date, OpenDecisionData],
    rebuilt_open: dict[date, OpenDecisionData],
) -> None:
    strategy = RANK_STRATEGIES["highest_sentiment"]
    for day, full_data in full_open.items():
        for symbol in list(full_data.members)[:4]:
            expected = signal_values_at_open(full_data, strategy, symbol)
            actual = signal_values_at_open(rebuilt_open[day], strategy, symbol)
            assert actual.keys() == expected.keys()
            for key, value in expected.items():
                if value is None:
                    assert actual[key] is None, (day, symbol, key)
                else:
                    assert actual[key] == pytest.approx(value, rel=1e-9), (day, symbol, key)


def test_the_open_cut_hides_the_close_volume_and_late_news(
    full: dict[date, DecisionData], rebuilt_open_at_close: dict[date, DecisionData]
) -> None:
    # The harness has teeth: reading the 09:30 rebuild as if it were D's close gives
    # different close-time answers. So a same-open block that secretly read D's close,
    # volume or afternoon news would fail the tests above.
    changed: Counter[str] = Counter()
    for day, full_data in full.items():
        cut = rebuilt_open_at_close[day]
        for name in ("above_ma_20", "rsi_5_high", "rel_volume_high", "spy_above_50"):
            condition = CONDITIONS[name]
            if not evaluate_condition(cut, condition).equals(
                evaluate_condition(full_data, condition)
            ):
                changed[name] += 1
        late = measure_news_count(cut, 1).fillna(-1) != measure_news_count(full_data, 1).fillna(-1)
        changed["news_after_0930"] += int(late.any())
    assert changed["above_ma_20"] > 0
    assert changed["rsi_5_high"] > 0
    assert changed["rel_volume_high"] > 0
    assert changed["news_after_0930"] > 0


def test_open_time_gap_equals_the_close_time_gap(
    full: dict[date, DecisionData], full_open: dict[date, OpenDecisionData]
) -> None:
    # The gap is fully known at the open, so both ways of reading it must agree exactly.
    for day, full_data in full.items():
        expected = measure_gap_pct(full_data).to_numpy(dtype="float64")
        actual = measure_gap_pct_at_open(full_open[day]).to_numpy(dtype="float64")
        np.testing.assert_array_equal(actual, expected)
