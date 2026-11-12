"""Each building block on a tiny hand-built market, with hand-computed values.

The market has 30 trading days (day 0 to day 29). The decision day D is day 29
unless a test says otherwise. Each stock has one job, named in its symbol.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot.data.membership import MembershipHistory, MembershipSpell
from bot.strategy.blocks import (
    DecisionData,
    MarketData,
    build_market_data,
    days_since_earnings_block,
    days_to_earnings_block,
    entry_signals,
    evaluate_condition,
    gap_block,
    measure_close_vs_ma_pct,
    measure_days_since_last_earnings,
    measure_days_to_next_earnings,
    measure_gap_pct,
    measure_n_day_return_pct,
    measure_news_count,
    measure_news_polarity,
    measure_relative_volume,
    measure_rsi,
    measure_spy_vs_ma_pct,
    n_day_return_block,
    news_sentiment_block,
    no_news_block,
    price_vs_moving_average_block,
    relative_volume_block,
    rsi_block,
    signal_values,
    slice_to_decision_date,
    spy_filter_block,
)
from bot.strategy.schema import (
    DaysSinceEarningsCondition,
    DaysToEarningsCondition,
    GapCondition,
    NDayReturnCondition,
    NewsSentimentCondition,
    NoNewsCondition,
    PriceVsMovingAverageCondition,
    RelativeVolumeCondition,
    RsiCondition,
    SpyFilterCondition,
    strategy_from_json,
)
from tests.backtest_helpers import Bar, assemble_market, flat_bar, raw_frame, toy_days

DAYS = toy_days(30)
LAST = 29
D = DAYS[LAST].date()


def flats(closes: dict[int, float], volumes: dict[int, float] | None = None) -> dict[int, Bar]:
    """Flat bars (open = high = low = close) for the given day numbers."""
    return {n: flat_bar(price, (volumes or {}).get(n, 1_000.0)) for n, price in closes.items()}


def constant(price: float, last: int = LAST) -> dict[int, float]:
    return {n: price for n in range(last + 1)}


def build() -> MarketData:
    """The block test market. See each stock's comment for its job."""
    frames: dict[str, pd.DataFrame] = {}

    def add(symbol: str, bars: dict[int, Bar], **extra: dict[int, float]) -> None:
        frames[symbol] = raw_frame(DAYS, bars, **extra)  # type: ignore[arg-type]

    # Gap down of exactly 25% (0.75 is exact in binary, so there is no rounding to argue about).
    add("GAPD.US", {**flats(constant(100.0, 28)), LAST: Bar(75.0, 80.0, 74.0, 80.0)})
    # Close 128 yesterday, open 123 today with a 1.0 dividend: (123 + 1) / 128 - 1 = -3.125%.
    # Without the dividend the gap would be 123 / 128 - 1 = -3.90625%.
    add(
        "GAPX.US",
        {**flats(constant(128.0, 28)), LAST: Bar(123.0, 125.0, 122.0, 124.0)},
        dividends={LAST: 1.0},
    )
    add("GAPU.US", {**flats(constant(100.0, 28)), LAST: Bar(110.0, 112.0, 109.0, 111.0)})
    add("RISE.US", flats({n: 100.0 + n for n in range(30)}))
    add("FALL.US", flats({n: 200.0 - n for n in range(30)}))
    add("FLAT.US", flats(constant(50.0)))
    # Five closes only. RSI(2) by hand is in test_rsi_hand_value.
    add("LATE.US", flats({25: 44.0, 26: 46.0, 27: 45.0, 28: 47.0, 29: 46.0}))
    # The same closes with a missing day (a halt) on day 27: the day is skipped.
    add("HALT.US", flats({24: 44.0, 25: 46.0, 26: 45.0, 28: 47.0, 29: 46.0}))
    # Total return: closes 100 to day 24, then 102, 104, 106, 108, 110, with a 2.0 dividend on 27.
    nday = {n: 100.0 for n in range(25)} | {25: 102.0, 26: 104.0, 27: 106.0, 28: 108.0, 29: 110.0}
    add("NDAY.US", flats(nday), dividends={27: 2.0})
    # Volume: 1000 to day 18, 2000 on days 19 to 23, then 800 900 1000 1100 1200, then 2500 today.
    volume = {n: 1_000.0 for n in range(19)}
    volume |= {n: 2_000.0 for n in range(19, 24)}
    volume |= dict(zip(range(24, 29), (800.0, 900.0, 1_000.0, 1_100.0, 1_200.0), strict=True))
    volume[LAST] = 2_500.0
    add("VOL.US", flats(constant(100.0), volume))
    # Same volumes, but no row at all on day 26, which is inside every baseline.
    add("VOLGAP.US", flats({n: 100.0 for n in range(30) if n != 26}, volume))
    # Data ends on day 28: no row today.
    add("GONE.US", flats(constant(100.0, 28)))
    # Has prices but is not in the index.
    add("NOTMEM.US", flats(constant(100.0)))

    spy = raw_frame(DAYS, flats({n: 400.0 + n for n in range(30)}))
    symbols = sorted(
        [
            *frames,
            "N0.US",
            "N1.US",
            "N2.US",
            "N3.US",
            "NODATA.US",
            "E1.US",
            "E2.US",
            "E3.US",
            "E4.US",
            "E5.US",
        ]
    )
    membership = pd.DataFrame(True, index=DAYS, columns=symbols, dtype=bool)
    membership["NOTMEM.US"] = False
    return assemble_market(
        DAYS,
        frames,
        spy_frame=spy,
        membership=membership,
        article_count=news_counts(),
        mean_polarity=news_polarity(),
        days_to_next=days_to_next(),
        days_since_last=days_since_last(),
    )


def table(columns: dict[str, dict[int, float]], fill: float = float("nan")) -> pd.DataFrame:
    """A date x symbol table from {symbol: {day number: value}}, NaN elsewhere."""
    frame = pd.DataFrame(fill, index=DAYS, columns=list(columns), dtype="float64")
    for symbol, values in columns.items():
        for number, value in values.items():
            frame.iloc[number, frame.columns.get_loc(symbol)] = value  # type: ignore[index]
    return frame


def news_counts() -> pd.DataFrame:
    """N0: none. N1: 0, 1, 3 articles on days 27 to 29. N2: 2 yesterday. N3: unknown today."""
    zeros = {n: 0.0 for n in range(30)}
    return table(
        {
            "N0.US": zeros,
            "N1.US": {**zeros, 28: 1.0, 29: 3.0},
            "N2.US": {**zeros, 28: 2.0},
            "N3.US": {**zeros, 29: float("nan")},
        }
    )


def news_polarity() -> pd.DataFrame:
    """N1: 0.8 on day 28 (one article) and 0.2 on day 29 (three). Days without articles are NaN."""
    return table({"N1.US": {28: 0.8, 29: 0.2}, "N2.US": {28: -0.5}})


def days_to_next() -> pd.DataFrame:
    """E1 reports in 3 trading days, E5 in 10, E2 and E4 have nothing visible. E3 has no column."""
    return table(
        {
            "E1.US": {29: 3.0},
            "E2.US": {},
            "E4.US": {},
            "E5.US": {29: 10.0},
        }
    )


def days_since_last() -> pd.DataFrame:
    """E1 reacted 7 sessions ago, E4 today. E2 and E5 have no past report."""
    return table(
        {
            "E1.US": {29: 7.0},
            "E2.US": {},
            "E4.US": {29: 0.0},
            "E5.US": {},
        }
    )


@pytest.fixture(scope="module")
def market() -> MarketData:
    return build()


@pytest.fixture(scope="module")
def data(market: MarketData) -> DecisionData:
    return slice_to_decision_date(market, D)


def passes(result: pd.Series, *symbols: str) -> bool:
    """True when every given symbol passes (a single symbol is the common case)."""
    return all(bool(result[s]) for s in symbols)


def only(result: pd.Series) -> set[str]:
    """The set of symbols that pass."""
    return {s for s, ok in result.items() if ok}


# ------------------------------------------------------------ slice and build


def test_slice_cuts_every_table_at_the_decision_date(market: MarketData) -> None:
    cut = slice_to_decision_date(market, DAYS[20].date())
    assert cut.decision_date == DAYS[20].date()
    last = pd.Timestamp(DAYS[20])
    tables = [
        cut.market.open,
        cut.market.close,
        cut.market.raw_close,
        cut.market.dividend,
        cut.market.split_factor,
        cut.market.total_return_index,
        cut.market.article_count,
        cut.market.mean_polarity,
        cut.market.days_to_next_earnings,
        cut.market.days_since_last_earnings,
        cut.market.membership,
        cut.market.spy,
    ]
    for frame in tables:
        assert frame.index.max() == last
        assert len(frame) == 21
    assert cut.market.trading_days.max() == last
    assert all(series.index.max() == last for series in cut.market.volume_by_symbol.values())


def test_slice_members_are_sorted_and_point_in_time(data: DecisionData) -> None:
    assert list(data.members) == sorted(data.members)
    assert "NOTMEM.US" not in data.members
    assert "GAPD.US" in data.members


def test_slice_rejects_a_non_trading_day(market: MarketData) -> None:
    with pytest.raises(ValueError):
        slice_to_decision_date(market, date(2023, 3, 4))  # a Saturday


def test_build_market_data_lays_out_tables_by_symbol() -> None:
    days = toy_days(5)
    spy = raw_frame(days, flats({n: 400.0 for n in range(5)}))
    a = raw_frame(days, flats({n: 10.0 + n for n in range(5)}))
    b = raw_frame(days, flats({2: 20.0, 3: 21.0, 4: 22.0}))
    news = pd.DataFrame(
        {
            "article_count": [np.nan, 0, 1, 0, 2],
            "mean_polarity": [np.nan, np.nan, 0.5, np.nan, 0.1],
            "daily_sentiment": np.nan,
            "open_article_count": [np.nan, 0, 0, 0, 1],
            "open_mean_polarity": [np.nan, np.nan, np.nan, np.nan, 0.3],
        },
        index=days,
    )
    earnings = pd.DataFrame(
        {"days_to_next": [np.nan] * 4 + [2.0], "days_since_last": 1.0}, index=days
    )
    history = MembershipHistory(
        spells=(
            MembershipSpell("A.US", date(2023, 1, 1), None),
            MembershipSpell("B.US", date(2023, 3, 6), None),
        )
    )
    market = build_market_data(
        {"A.US": a, "B.US": b}, {"A.US": news}, {"A.US": earnings}, spy, history
    )
    assert list(market.trading_days) == list(days)
    assert market.close.loc[days[3], "A.US"] == 13.0
    assert np.isnan(market.close.loc[days[1], "B.US"])  # no forward fill
    assert list(market.article_count.columns) == ["A.US"]  # B has no news frame: no column
    assert market.article_count.loc[days[4], "A.US"] == 2
    assert market.open_article_count.loc[days[4], "A.US"] == 1
    assert market.days_to_next_earnings.loc[days[4], "A.US"] == 2.0
    assert market.total_return_index.loc[days[0], "A.US"] == 1.0
    assert bool(market.membership.loc[days[2], "B.US"]) is False
    assert bool(market.membership.loc[days[3], "B.US"]) is True
    assert market.volume_by_symbol["B.US"].reindex(days).isna().sum() == 2


# --------------------------------------------------------------------- shape


def test_every_block_returns_a_bool_series_on_the_members(data: DecisionData) -> None:
    conditions = [
        GapCondition(block="gap", direction="down", min_gap_pct=3),
        NDayReturnCondition(block="n_day_return", days=5, comparison="at_least", return_pct=1),
        PriceVsMovingAverageCondition(block="price_vs_moving_average", ma_days=5, position="above"),
        RsiCondition(block="rsi", comparison="at_least", level=50),
        RelativeVolumeCondition(block="relative_volume", comparison="at_least", ratio=1),
        NoNewsCondition(block="no_news"),
        NewsSentimentCondition(block="news_sentiment", comparison="at_least", polarity=0),
        DaysToEarningsCondition(block="days_to_earnings", relation="within", days=5),
        DaysSinceEarningsCondition(block="days_since_earnings", relation="within", days=5),
        SpyFilterCondition(block="spy_filter", ma_days=5, position="above"),
    ]
    for condition in conditions:
        result = evaluate_condition(data, condition)
        assert result.dtype == bool, condition.block
        assert result.index.equals(data.members), condition.block


# ----------------------------------------------------------------------- gap


def test_gap_measure_hand_values(data: DecisionData) -> None:
    gap = measure_gap_pct(data)
    assert gap["GAPD.US"] == pytest.approx(-25.0)  # 75 / 100 - 1
    assert gap["GAPX.US"] == pytest.approx(-3.125)  # (123 + 1) / 128 - 1: dividend added back
    assert gap["GAPU.US"] == pytest.approx(10.0)
    assert gap["FLAT.US"] == pytest.approx(0.0)
    assert np.isnan(gap["GONE.US"])  # no row today


def test_gap_down_block(data: DecisionData) -> None:
    result = gap_block(data, GapCondition(block="gap", direction="down", min_gap_pct=3))
    assert only(result) == {"GAPD.US", "GAPX.US"}


def test_gap_adds_the_dividend_back(data: DecisionData) -> None:
    # Raw gap is -3.906%, which would pass 3.5. With the dividend it is -3.125%, which does not.
    result = gap_block(data, GapCondition(block="gap", direction="down", min_gap_pct=3.5))
    assert only(result) == {"GAPD.US"}


def test_gap_threshold_is_inclusive(data: DecisionData) -> None:
    # 3.125 is exact in binary: the stock sits exactly on the threshold and passes.
    result = gap_block(data, GapCondition(block="gap", direction="down", min_gap_pct=3.125))
    assert passes(result, "GAPX.US")
    assert not passes(
        gap_block(data, GapCondition(block="gap", direction="down", min_gap_pct=3.13)), "GAPX.US"
    )
    assert passes(
        gap_block(data, GapCondition(block="gap", direction="up", min_gap_pct=10)), "GAPU.US"
    )


def test_gap_up_block(data: DecisionData) -> None:
    result = gap_block(data, GapCondition(block="gap", direction="up", min_gap_pct=5))
    assert only(result) == {"GAPU.US"}


# ------------------------------------------------------------------- n-day return


def test_n_day_return_hand_values(data: DecisionData) -> None:
    returns = measure_n_day_return_pct(data, 5)
    # Day 29 close 110 against day 24 close 100, with the 2.0 dividend paid on day 27:
    # (102/100) * (104/102) * ((106 + 2)/104) * (108/106) * (110/108) - 1
    # = 110 * 108 / (106 * 100) - 1, because most terms cancel.
    assert returns["NDAY.US"] == pytest.approx((110 * 108 / (106 * 100) - 1) * 100)
    assert returns["NDAY.US"] == pytest.approx(12.0754717, abs=1e-6)  # a price-only return is 10%
    assert returns["RISE.US"] == pytest.approx((129 / 124 - 1) * 100)  # 4.0322581
    assert returns["FLAT.US"] == pytest.approx(0.0)
    assert np.isnan(measure_n_day_return_pct(data, 10)["LATE.US"])  # only 5 closes
    assert np.isnan(returns["GONE.US"])


def test_n_day_return_block_counts_dividends(data: DecisionData) -> None:
    at_least = NDayReturnCondition(
        block="n_day_return", days=5, comparison="at_least", return_pct=11
    )
    assert passes(n_day_return_block(data, at_least), "NDAY.US")
    at_most = NDayReturnCondition(block="n_day_return", days=5, comparison="at_most", return_pct=11)
    assert not passes(n_day_return_block(data, at_most), "NDAY.US")
    assert passes(n_day_return_block(data, at_most), "RISE.US", "FLAT.US", "FALL.US")


def test_n_day_return_missing_data_never_passes(data: DecisionData) -> None:
    for comparison, level in (("at_least", -90.0), ("at_most", 500.0)):
        result = n_day_return_block(
            data,
            NDayReturnCondition(
                block="n_day_return", days=10, comparison=comparison, return_pct=level
            ),  # type: ignore[arg-type]
        )
        assert not result["LATE.US"]
        assert not result["GONE.US"]


# ----------------------------------------------------------- moving average


def test_close_vs_ma_hand_values(data: DecisionData) -> None:
    vs_ma = measure_close_vs_ma_pct(data, 5)
    # RISE: closes 125 to 129, average 127, so 129 / 127 - 1 = 1.5748%. Today is in the average.
    assert vs_ma["RISE.US"] == pytest.approx((129 / 127 - 1) * 100)
    # FALL: closes 175 down to 171, average 173.
    assert vs_ma["FALL.US"] == pytest.approx((171 / 173 - 1) * 100)
    assert vs_ma["FLAT.US"] == pytest.approx(0.0)
    assert np.isnan(measure_close_vs_ma_pct(data, 10)["LATE.US"])  # needs all 10 closes


def test_price_vs_ma_blocks(data: DecisionData) -> None:
    above = price_vs_moving_average_block(
        data,
        PriceVsMovingAverageCondition(block="price_vs_moving_average", ma_days=5, position="above"),
    )
    below = price_vs_moving_average_block(
        data,
        PriceVsMovingAverageCondition(block="price_vs_moving_average", ma_days=5, position="below"),
    )
    assert passes(above, "RISE.US") and not passes(below, "RISE.US")
    assert passes(below, "FALL.US") and not passes(above, "FALL.US")
    # Equal passes neither side.
    assert not passes(above, "FLAT.US") and not passes(below, "FLAT.US")


def test_price_vs_ma_missing_history_is_false(data: DecisionData) -> None:
    for position in ("above", "below"):
        result = price_vs_moving_average_block(
            data,
            PriceVsMovingAverageCondition(
                block="price_vs_moving_average",
                ma_days=10,
                position=position,  # type: ignore[arg-type]
            ),
        )
        assert not result["LATE.US"]
        assert not result["GONE.US"]


# ----------------------------------------------------------------------- RSI


def test_rsi_hand_value(data: DecisionData) -> None:
    # LATE closes 44 46 45 47 46, period 2. Changes +2 -1 +2 -1.
    # gains 2 0 2 0 and losses 0 1 0 1.
    # Seed: mean of the first 2 values: avg_gain 1.0, avg_loss 0.5.
    # Next gain 2:  avg_gain = (1.0 * 1 + 2) / 2 = 1.5, avg_loss = (0.5 * 1 + 0) / 2 = 0.25.
    # Next loss 1:  avg_gain = (1.5 * 1 + 0) / 2 = 0.75, avg_loss = (0.25 * 1 + 1) / 2 = 0.625.
    # RS = 0.75 / 0.625 = 1.2, RSI = 100 - 100 / 2.2 = 54.5454...
    rsi = measure_rsi(data, 2)
    assert rsi["LATE.US"] == pytest.approx(100 - 100 / 2.2)
    assert rsi["LATE.US"] == pytest.approx(54.5454545, abs=1e-6)
    assert rsi["HALT.US"] == pytest.approx(100 - 100 / 2.2)  # the missing day is skipped


def test_rsi_with_three_closes(market: MarketData) -> None:
    # At day 27 LATE has closes 44 46 45: changes +2 -1. Seed only: avg_gain 1, avg_loss 0.5.
    # RS = 2, RSI = 100 - 100 / 3 = 66.6667.
    early = slice_to_decision_date(market, DAYS[27].date())
    assert measure_rsi(early, 2)["LATE.US"] == pytest.approx(200 / 3)
    # With two closes (day 26) there is only one change, fewer than the period: NaN.
    assert np.isnan(measure_rsi(slice_to_decision_date(market, DAYS[26].date()), 2)["LATE.US"])


def test_rsi_edge_values(data: DecisionData) -> None:
    rsi = measure_rsi(data, 14)
    assert rsi["RISE.US"] == 100.0  # no losses
    assert rsi["FALL.US"] == pytest.approx(0.0)  # no gains
    assert rsi["FLAT.US"] == 50.0  # no moves
    assert np.isnan(measure_rsi(data, 5)["LATE.US"])  # needs 6 closes, has 5
    assert np.isnan(rsi["GONE.US"])  # no close today


def test_rsi_block(data: DecisionData) -> None:
    low = RsiCondition(block="rsi", period=2, comparison="at_most", level=54.5)
    high = RsiCondition(block="rsi", period=2, comparison="at_least", level=54.5)
    assert not passes(rsi_block(data, low), "LATE.US")
    assert passes(rsi_block(data, high), "LATE.US")
    over = RsiCondition(block="rsi", period=2, comparison="at_least", level=55)
    assert not passes(rsi_block(data, over), "LATE.US")
    both_ends = RsiCondition(block="rsi", comparison="at_least", level=50)
    assert passes(rsi_block(data, both_ends), "FLAT.US", "RISE.US")
    assert not passes(rsi_block(data, both_ends), "FALL.US")
    for comparison in ("at_least", "at_most"):
        short = RsiCondition(block="rsi", period=5, comparison=comparison, level=50)  # type: ignore[arg-type]
        assert not rsi_block(data, short)["LATE.US"]
        assert not rsi_block(data, short)["GONE.US"]


# ----------------------------------------------------------- relative volume


def test_relative_volume_excludes_today(data: DecisionData) -> None:
    ratio5 = measure_relative_volume(data, 5)
    # Baseline days 24 to 28: (800 + 900 + 1000 + 1100 + 1200) / 5 = 1000. Today 2500.
    assert ratio5["VOL.US"] == pytest.approx(2.5)
    # Baseline days 19 to 28: (5 * 2000 + 5000) / 10 = 1500, so 2500 / 1500.
    assert measure_relative_volume(data, 10)["VOL.US"] == pytest.approx(2500 / 1500)
    assert ratio5["RISE.US"] == pytest.approx(1.0)


def test_relative_volume_missing_baseline_day_is_nan(data: DecisionData) -> None:
    assert np.isnan(measure_relative_volume(data, 5)["VOLGAP.US"])
    assert np.isnan(measure_relative_volume(data, 5)["GONE.US"])


def test_relative_volume_block(data: DecisionData) -> None:
    at_least = RelativeVolumeCondition(
        block="relative_volume", lookback_days=5, comparison="at_least", ratio=2
    )
    at_most = RelativeVolumeCondition(
        block="relative_volume", lookback_days=5, comparison="at_most", ratio=2
    )
    assert passes(relative_volume_block(data, at_least), "VOL.US")
    assert not passes(relative_volume_block(data, at_most), "VOL.US")
    ten = RelativeVolumeCondition(
        block="relative_volume", lookback_days=10, comparison="at_least", ratio=2
    )
    assert not passes(relative_volume_block(data, ten), "VOL.US")  # 1.667 is under 2
    for block in (at_least, at_most):
        assert not relative_volume_block(data, block)["VOLGAP.US"]


# --------------------------------------------------------------------- news


def test_news_count_and_polarity_hand_values(data: DecisionData) -> None:
    assert measure_news_count(data, 1)["N1.US"] == 3
    assert measure_news_count(data, 2)["N1.US"] == 4  # day 28 and day 29
    assert measure_news_count(data, 2)["N2.US"] == 2
    assert np.isnan(measure_news_count(data, 1)["NODATA.US"])  # no news column
    # Day 28: 1 article at 0.8. Day 29: 3 articles at 0.2. Weighted: (0.8 + 0.6) / 4 = 0.35.
    assert measure_news_polarity(data, 2)["N1.US"] == pytest.approx(0.35)
    assert measure_news_polarity(data, 1)["N1.US"] == pytest.approx(0.2)
    assert measure_news_polarity(data, 3)["N1.US"] == pytest.approx(
        0.35
    )  # an empty day adds nothing
    assert np.isnan(measure_news_polarity(data, 2)["N0.US"])  # no articles


def test_no_news_block(data: DecisionData) -> None:
    one = no_news_block(data, NoNewsCondition(block="no_news", lookback_sessions=1))
    assert passes(one, "N0.US", "N2.US")  # N2's article was yesterday, outside one window
    assert not passes(one, "N1.US")
    two = no_news_block(data, NoNewsCondition(block="no_news", lookback_sessions=2))
    assert passes(two, "N0.US")
    assert not passes(
        two,
        "N2.US",
    ) and not passes(two, "N1.US")
    twenty = no_news_block(data, NoNewsCondition(block="no_news", lookback_sessions=20))
    assert passes(twenty, "N0.US")


def test_no_news_needs_data(data: DecisionData) -> None:
    result = no_news_block(data, NoNewsCondition(block="no_news"))
    assert not result["NODATA.US"]  # no data is not the same as no news
    assert not result["N3.US"]  # an unknown count today is not zero
    assert not result["RISE.US"]  # a stock with prices but no news column


def test_news_sentiment_block(data: DecisionData) -> None:
    def block(comparison: str, polarity: float, min_articles: int, sessions: int) -> pd.Series:
        return news_sentiment_block(
            data,
            NewsSentimentCondition(
                block="news_sentiment",
                lookback_sessions=sessions,
                comparison=comparison,  # type: ignore[arg-type]
                polarity=polarity,
                min_articles=min_articles,
            ),
        )

    assert passes(block("at_least", 0.3, 4, 2), "N1.US")  # 4 articles, average 0.35
    assert not passes(block("at_least", 0.3, 5, 2), "N1.US")  # not enough articles
    assert not passes(block("at_least", 0.4, 4, 2), "N1.US")  # average too low
    assert passes(block("at_most", 0.4, 4, 2), "N1.US")
    assert not passes(block("at_most", 0.3, 4, 2), "N1.US")
    assert passes(block("at_least", 0.1, 3, 1), "N1.US")  # one session: 3 articles at 0.2
    assert not passes(block("at_least", 0.1, 4, 1), "N1.US")
    result = block("at_most", 1.0, 1, 1)
    assert not result["N0.US"]  # no articles at all
    assert not result["NODATA.US"]
    assert not result["RISE.US"]


# ------------------------------------------------------------------ earnings


def test_days_to_earnings(data: DecisionData) -> None:
    assert measure_days_to_next_earnings(data)["E1.US"] == 3
    assert np.isnan(measure_days_to_next_earnings(data)["E2.US"])

    def block(relation: str, days: int) -> pd.Series:
        return days_to_earnings_block(
            data,
            DaysToEarningsCondition(block="days_to_earnings", relation=relation, days=days),  # type: ignore[arg-type]
        )

    assert passes(block("within", 3), "E1.US")  # at most
    assert not passes(block("within", 2), "E1.US")
    assert passes(block("not_within", 2), "E1.US")
    assert not passes(block("not_within", 3), "E1.US")
    assert passes(block("within", 10), "E5.US")
    # No visible report: not within is true, within is false.
    assert passes(block("not_within", 5), "E2.US", "E4.US")
    assert not passes(block("within", 10), "E2.US") and not passes(block("within", 10), "E4.US")
    # No earnings data at all fails both.
    assert not block("within", 5)["E3.US"] and not block("not_within", 5)["E3.US"]


def test_days_since_earnings(data: DecisionData) -> None:
    assert measure_days_since_last_earnings(data)["E4.US"] == 0
    assert measure_days_since_last_earnings(data)["E1.US"] == 7
    assert np.isnan(measure_days_since_last_earnings(data)["E2.US"])

    def block(relation: str, days: int) -> pd.Series:
        return days_since_earnings_block(
            data,
            DaysSinceEarningsCondition(block="days_since_earnings", relation=relation, days=days),  # type: ignore[arg-type]
        )

    assert passes(block("within", 7), "E1.US")
    assert not passes(block("within", 6), "E1.US")
    assert passes(block("not_within", 6), "E1.US")
    assert not passes(block("not_within", 7), "E1.US")
    assert passes(block("within", 0), "E4.US")  # 0 means today
    assert not passes(block("not_within", 0), "E4.US")
    assert passes(block("not_within", 60), "E2.US", "E5.US")  # none yet
    assert not passes(block("within", 60), "E2.US")
    assert not block("within", 60)["E3.US"] and not block("not_within", 60)["E3.US"]


# ----------------------------------------------------------------- SPY filter


def test_spy_filter(data: DecisionData) -> None:
    # SPY closes 400 + day. Day 29 is 429. Five-day average 427, so 429 / 427 - 1 = 0.4684%.
    assert measure_spy_vs_ma_pct(data, 5) == pytest.approx((429 / 427 - 1) * 100)
    above = spy_filter_block(
        data, SpyFilterCondition(block="spy_filter", ma_days=5, position="above")
    )
    below = spy_filter_block(
        data, SpyFilterCondition(block="spy_filter", ma_days=5, position="below")
    )
    assert above.all() and not below.any()  # the same answer for every member
    assert list(above.index) == list(data.members)


def test_spy_filter_without_enough_history_is_false_for_everyone(data: DecisionData) -> None:
    for position in ("above", "below"):
        result = spy_filter_block(
            data,
            SpyFilterCondition(block="spy_filter", ma_days=50, position=position),  # type: ignore[arg-type]
        )
        assert not result.any()
    assert np.isnan(measure_spy_vs_ma_pct(data, 50))


# --------------------------------------------------------------- combination


def strategy_with(*conditions: dict[str, object]):  # type: ignore[no-untyped-def]
    return strategy_from_json(
        json.dumps(
            {
                "name": "Block test",
                "description_plain_english": "A strategy used to test combining blocks.",
                "entry": list(conditions),
                "exit": {"holding_days": 3},
                "sizing": {"max_positions": 5},
            }
        )
    )


def test_entry_signals_are_an_and_of_the_conditions(data: DecisionData) -> None:
    gap = {"block": "gap", "direction": "down", "min_gap_pct": 3}
    volume = {"block": "relative_volume", "lookback_days": 5, "comparison": "at_most", "ratio": 5}
    assert only(entry_signals(data, strategy_with(gap))) == {"GAPD.US", "GAPX.US"}
    # Only GAPD and GAPX have a big gap. Both have an ordinary volume ratio, so both pass.
    assert only(entry_signals(data, strategy_with(gap, volume))) == {"GAPD.US", "GAPX.US"}
    tight = {"block": "gap", "direction": "down", "min_gap_pct": 20}
    assert only(entry_signals(data, strategy_with(gap, tight))) == {"GAPD.US"}
    assert entry_signals(data, strategy_with(tight, gap)).equals(
        entry_signals(data, strategy_with(gap, tight))
    )


def test_signal_values_report_the_measured_gap(data: DecisionData) -> None:
    strategy = strategy_with({"block": "gap", "direction": "down", "min_gap_pct": 3})
    assert signal_values(data, strategy, "GAPD.US")["gap_pct"] == pytest.approx(-25.0)
