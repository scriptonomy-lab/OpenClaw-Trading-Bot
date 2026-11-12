"""Schema tests: the worked examples, bad inputs and their plain English lines, ids, windows.

Expected texts come from docs/design/strategy_schema.md ("Errors the agent can
relay") and from the field rules in the same note.
"""

import copy
import json
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from bot.strategy.schema import (
    NEWS_COVERAGE_FOOTNOTE,
    StrategyDefinition,
    backtest_windows,
    describe_rules,
    explain_validation_error,
    strategy_from_json,
    strategy_id,
    strategy_json_schema,
    strategy_to_json,
)

EXAMPLE_1: dict[str, Any] = {
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

EXAMPLE_2: dict[str, Any] = {
    "name": "Trend momentum",
    "description_plain_english": (
        "When SPY is above its 200-day average, buy S&P 500 stocks up at least 20% over the last "
        "126 trading days that close above their 50-day average and have an RSI of 70 or less. "
        "Sell after 20 trading days, or earlier on a 10% stop loss."
    ),
    "version": 1,
    "entry": [
        {"block": "spy_filter", "ma_days": 200, "position": "above"},
        {"block": "n_day_return", "days": 126, "comparison": "at_least", "return_pct": 20},
        {"block": "price_vs_moving_average", "ma_days": 50, "position": "above"},
        {"block": "rsi", "period": 14, "comparison": "at_most", "level": 70},
    ],
    "exit": {"holding_days": 20, "stop_loss_pct": 10},
    "sizing": {"max_positions": 10},
}

EXAMPLE_3: dict[str, Any] = {
    "name": "Positive news, heavy volume",
    "description_plain_english": (
        "Buy S&P 500 stocks with at least 3 articles since the previous close whose average "
        "polarity is 0.5 or higher, on at least twice their normal volume, with no earnings "
        "reaction in the next 5 trading days. Sell after 5 trading days, at an 8% profit, or at "
        "a 5% loss."
    ),
    "version": 1,
    "entry": [
        {
            "block": "news_sentiment",
            "lookback_sessions": 1,
            "comparison": "at_least",
            "polarity": 0.5,
            "min_articles": 3,
        },
        {"block": "relative_volume", "lookback_days": 20, "comparison": "at_least", "ratio": 2},
        {"block": "days_to_earnings", "relation": "not_within", "days": 5},
    ],
    "exit": {"holding_days": 5, "stop_loss_pct": 5, "profit_target_pct": 8},
    "sizing": {"max_positions": 5},
}


def parse(data: dict[str, Any]) -> StrategyDefinition:
    return strategy_from_json(json.dumps(data))


def lines_for(data: dict[str, Any] | str) -> list[str]:
    """The relayed error lines for an invalid strategy."""
    text = data if isinstance(data, str) else json.dumps(data)
    with pytest.raises(ValidationError) as caught:
        strategy_from_json(text)
    return explain_validation_error(caught.value)


def changed(**updates: Any) -> dict[str, Any]:
    """Example 1 with top-level fields replaced."""
    data = copy.deepcopy(EXAMPLE_1)
    data.update(updates)
    return data


# ------------------------------------------------------------ worked examples


@pytest.mark.parametrize(
    "example", [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3], ids=["gap", "momentum", "news"]
)
def test_worked_examples_validate_and_round_trip(example: dict[str, Any]) -> None:
    strategy = parse(example)
    assert strategy.universe == "sp500_point_in_time"
    assert strategy.side == "long"
    assert strategy.schema_version == 1
    assert strategy_from_json(strategy_to_json(strategy)) == strategy


def test_defaults_are_filled_in() -> None:
    rsi = parse(changed(entry=[{"block": "rsi", "comparison": "at_most", "level": 30}]))
    assert rsi.entry[0].period == 14  # type: ignore[union-attr]
    assert parse(EXAMPLE_1).sizing.method == "equal_weight"


def test_describe_rules_reads_back_what_will_run() -> None:
    text = describe_rules(parse(EXAMPLE_3))
    assert "At least 3 article(s) since the previous close" in text
    assert "after 5 trading days (sold at the next open)" in text
    assert "up to 5 positions" in text


def test_json_schema_shows_bounds() -> None:
    text = json.dumps(strategy_json_schema())
    assert '"minimum": 0.1' in text
    assert '"maximum": 50' in text


# ------------------------------------------------------------------ bad input


def test_gap_out_of_bounds() -> None:
    data = copy.deepcopy(EXAMPLE_1)
    data["entry"][0]["min_gap_pct"] = 80
    assert lines_for(data) == [
        "Entry condition 1 (gap): min_gap_pct must be between 0.1 and 50. You gave 80."
    ]


def test_no_exit() -> None:
    assert lines_for(changed(exit={})) == [
        "Exit rules: Give at least one exit: holding_days, stop_loss_pct or profit_target_pct."
    ]


def test_short_side_refused() -> None:
    assert lines_for(changed(side="short")) == ["side must be one of 'long'. You gave 'short'."]


def test_unknown_block() -> None:
    lines = lines_for(changed(entry=[{"block": "reddit_mentions"}]))
    assert len(lines) == 1
    assert lines[0].startswith(
        "Entry condition 1: 'reddit_mentions' is not a supported building block. "
        "Supported blocks: 'gap', 'n_day_return',"
    )


def test_missing_block_name() -> None:
    assert lines_for(changed(entry=[{"direction": "down"}])) == [
        "Entry condition 1: Each entry condition needs a 'block' field naming its building block."
    ]


@pytest.mark.parametrize("field", ["leverage", "short", "margin"])
def test_extra_top_level_fields_rejected(field: str) -> None:
    assert lines_for(changed(**{field: 2})) == [f"'{field}' is not a supported field. Remove it."]


def test_extra_field_inside_a_condition_and_in_exit() -> None:
    data = copy.deepcopy(EXAMPLE_1)
    data["entry"][0]["short"] = True
    assert lines_for(data) == [
        "Entry condition 1 (gap): 'short' is not a supported field. Remove it."
    ]
    assert lines_for(changed(exit={"holding_days": 3, "trailing": 5})) == [
        "Exit rules: 'trailing' is not a supported field. Remove it."
    ]


def test_bad_comparison_word() -> None:
    entry = [{"block": "rsi", "comparison": "below", "level": 30}]
    assert lines_for(changed(entry=entry)) == [
        "Entry condition 1 (rsi): comparison must be one of 'at_least' or 'at_most'. "
        "You gave 'below'."
    ]


def test_wrong_types_are_not_coerced() -> None:
    assert lines_for(changed(exit={"holding_days": 3.5})) == [
        "Exit rules: holding_days must be a whole number. You gave 3.5."
    ]
    assert lines_for(changed(exit={"holding_days": 3.0})) == [
        "Exit rules: holding_days must be a whole number. You gave 3.0."
    ]
    assert lines_for(changed(exit={"holding_days": True})) == [
        "Exit rules: holding_days must be a whole number. You gave True."
    ]
    data = copy.deepcopy(EXAMPLE_1)
    data["entry"][0]["min_gap_pct"] = "5"
    assert lines_for(data) == [
        "Entry condition 1 (gap): min_gap_pct must be a number. You gave '5'."
    ]


def test_earnings_days_capped_at_ten() -> None:
    entry = [{"block": "days_to_earnings", "relation": "within", "days": 30}]
    assert lines_for(changed(entry=entry)) == [
        "Entry condition 1 (days_to_earnings): days must be a whole number between 1 and 10. "
        "You gave 30."
    ]


def test_duplicate_conditions() -> None:
    gap = EXAMPLE_1["entry"][0]
    assert lines_for(changed(entry=[gap, gap])) == [
        "Entry condition 2 is the same as condition 1. Remove one."
    ]


def test_same_block_twice_with_different_parameters_is_allowed() -> None:
    entry = [
        {"block": "n_day_return", "days": 5, "comparison": "at_least", "return_pct": -10},
        {"block": "n_day_return", "days": 5, "comparison": "at_most", "return_pct": 10},
    ]
    assert len(parse(changed(entry=entry)).entry) == 2


def test_empty_entry_list() -> None:
    assert lines_for(changed(entry=[])) == [
        "entry must have between 1 and 8 conditions. You gave 0."
    ]


def test_more_than_eight_conditions() -> None:
    entry = [
        {"block": "rsi", "period": period, "comparison": "at_least", "level": 30}
        for period in range(2, 11)
    ]
    assert lines_for(changed(entry=entry)) == [
        "entry must have between 1 and 8 conditions. You gave 9."
    ]
    assert len(parse(changed(entry=entry[:8])).entry) == 8


def test_several_problems_give_one_line_each() -> None:
    data = changed(side="short")
    data["entry"][0]["min_gap_pct"] = 80
    assert lines_for(data) == [
        "side must be one of 'long'. You gave 'short'.",
        "Entry condition 1 (gap): min_gap_pct must be between 0.1 and 50. You gave 80.",
    ]


def test_text_fields_and_not_json() -> None:
    assert lines_for(changed(name="ab")) == ["name must be 3 to 60 characters long. You gave 2."]
    assert lines_for(changed(name="a\nbc")) == [
        "name contains control characters. Use plain text on one line."
    ]
    assert lines_for("not json")[0].startswith("The strategy is not valid JSON")


def test_sizing_and_version_bounds() -> None:
    assert lines_for(changed(sizing={"max_positions": 51})) == [
        "Sizing: max_positions must be a whole number between 1 and 50. You gave 51."
    ]
    assert lines_for(changed(version=0)) == [
        "version must be a whole number between 1 and 500. You gave 0."
    ]


@pytest.mark.parametrize(
    ("condition", "ok"),
    [
        ({"block": "gap", "direction": "up", "min_gap_pct": 0.1}, True),
        ({"block": "gap", "direction": "up", "min_gap_pct": 0.09}, False),
        ({"block": "gap", "direction": "up", "min_gap_pct": 50}, True),
        ({"block": "gap", "direction": "up", "min_gap_pct": 50.1}, False),
        ({"block": "price_vs_moving_average", "ma_days": 5, "position": "above"}, True),
        ({"block": "price_vs_moving_average", "ma_days": 4, "position": "above"}, False),
        ({"block": "price_vs_moving_average", "ma_days": 250, "position": "below"}, True),
        ({"block": "price_vs_moving_average", "ma_days": 251, "position": "below"}, False),
        ({"block": "days_since_earnings", "relation": "within", "days": 0}, True),
        ({"block": "days_since_earnings", "relation": "within", "days": 60}, True),
        ({"block": "days_since_earnings", "relation": "within", "days": 61}, False),
        ({"block": "rsi", "comparison": "at_most", "level": 99}, True),
        ({"block": "rsi", "comparison": "at_most", "level": 100}, False),
        ({"block": "n_day_return", "days": 252, "comparison": "at_least", "return_pct": -90}, True),
        (
            {"block": "n_day_return", "days": 253, "comparison": "at_least", "return_pct": -90},
            False,
        ),
        ({"block": "relative_volume", "comparison": "at_least", "ratio": 20}, True),
        ({"block": "relative_volume", "comparison": "at_least", "ratio": 20.5}, False),
        ({"block": "no_news", "lookback_sessions": 20}, True),
        ({"block": "no_news", "lookback_sessions": 21}, False),
        ({"block": "news_sentiment", "comparison": "at_most", "polarity": -1}, True),
        ({"block": "news_sentiment", "comparison": "at_most", "polarity": 1.01}, False),
        (
            {"block": "news_sentiment", "comparison": "at_most", "polarity": 0, "min_articles": 51},
            False,
        ),
    ],
)
def test_bounds_are_inclusive(condition: dict[str, Any], ok: bool) -> None:
    data = changed(entry=[condition])
    if ok:
        parse(data)
    else:
        assert lines_for(data)


@pytest.mark.parametrize(
    ("exit_rules", "ok"),
    [
        ({"holding_days": 1}, True),
        ({"holding_days": 252}, True),
        ({"holding_days": 253}, False),
        ({"stop_loss_pct": 0.5}, True),
        ({"stop_loss_pct": 0.4}, False),
        ({"profit_target_pct": 200}, True),
        ({"profit_target_pct": 201}, False),
    ],
)
def test_exit_bounds_are_inclusive(exit_rules: dict[str, Any], ok: bool) -> None:
    data = changed(exit=exit_rules)
    if ok:
        parse(data)
    else:
        assert lines_for(data)


# ------------------------------------------------------------------------ id


def test_id_format_and_stability() -> None:
    sid = strategy_id(parse(EXAMPLE_1))
    assert sid.startswith("strat_")
    assert len(sid) == len("strat_") + 16
    assert strategy_id(parse(EXAMPLE_1)) == sid


def test_id_ignores_name_description_and_version() -> None:
    base = strategy_id(parse(EXAMPLE_1))
    renamed = changed(
        name="A totally new name", version=7, description_plain_english="Something else."
    )
    assert strategy_id(parse(renamed)) == base


def test_id_ignores_condition_order() -> None:
    reordered = changed(entry=list(reversed(EXAMPLE_1["entry"])))
    assert strategy_id(parse(reordered)) == strategy_id(parse(EXAMPLE_1))
    momentum = copy.deepcopy(EXAMPLE_2)
    momentum["entry"] = list(reversed(momentum["entry"]))
    assert strategy_id(parse(momentum)) == strategy_id(parse(EXAMPLE_2))


def test_id_ignores_defaults_written_out() -> None:
    explicit = changed(universe="sp500_point_in_time", side="long", schema_version=1)
    explicit["sizing"] = {"method": "equal_weight", "max_positions": 10}
    assert strategy_id(parse(explicit)) == strategy_id(parse(EXAMPLE_1))


def test_id_changes_when_a_rule_changes() -> None:
    base = strategy_id(parse(EXAMPLE_1))
    variants: list[dict[str, Any]] = []
    for mutate in (
        lambda d: d["entry"][0].update(min_gap_pct=4),
        lambda d: d["entry"][0].update(direction="up"),
        lambda d: d["entry"][1].update(lookback_sessions=2),
        lambda d: d.update(exit={"holding_days": 4}),
        lambda d: d.update(exit={"holding_days": 3, "stop_loss_pct": 5}),
        lambda d: d["sizing"].update(max_positions=9),
        lambda d: d.update(entry=d["entry"][:1]),
    ):
        data = copy.deepcopy(EXAMPLE_1)
        mutate(data)
        variants.append(data)
    ids = {strategy_id(parse(v)) for v in variants}
    assert base not in ids
    assert len(ids) == len(variants)


def test_id_treats_equal_numbers_alike() -> None:
    data = copy.deepcopy(EXAMPLE_1)
    data["entry"][0]["min_gap_pct"] = 3.0
    assert strategy_id(parse(data)) == strategy_id(parse(EXAMPLE_1))


# ------------------------------------------------------------------- windows


def test_news_strategy_gets_one_window_2021_to_2023() -> None:
    (window,) = backtest_windows(parse(EXAMPLE_1))
    assert (window.label, window.start, window.end) == (
        "primary",
        date(2021, 1, 1),
        date(2023, 12, 31),
    )
    assert window.coverage_warning == NEWS_COVERAGE_FOOTNOTE
    assert "2019 and 2020" in NEWS_COVERAGE_FOOTNOTE
    assert "no data" in NEWS_COVERAGE_FOOTNOTE


def test_sentiment_strategy_is_a_news_strategy() -> None:
    (window,) = backtest_windows(parse(EXAMPLE_3))
    assert window.start == date(2021, 1, 1)


def test_strategy_without_news_gets_one_window() -> None:
    (window,) = backtest_windows(parse(EXAMPLE_2))
    assert (window.label, window.start, window.end) == (
        "primary",
        date(2016, 1, 1),
        date(2023, 12, 31),
    )
    assert window.coverage_warning is None


def test_the_same_rules_without_no_news_is_a_different_strategy() -> None:
    without = changed(entry=EXAMPLE_1["entry"][:1])
    assert strategy_id(parse(without)) != strategy_id(parse(EXAMPLE_1))
    assert len(backtest_windows(parse(without))) == 1
