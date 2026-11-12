"""The strategy definition the agent fills in. The agent writes this, never code.

A strategy is a JSON object with fixed fields. Python validates it before
anything runs. Every number has bounds, unknown fields are rejected, and every
error message is a plain English sentence the agent can pass on to the user.

Entry conditions are all joined with AND. There is no OR on purpose: one list
of conditions that must all be true is easy to repeat back, easy to check, and
gives the agent fewer ways to get the logic wrong. See
docs/design/strategy_schema.md for the meaning and timing of every block.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic_core import ErrorDetails, PydanticCustomError

from bot.config import BACKTEST_END, BACKTEST_START

SCHEMA_VERSION = 1

# News coverage is too thin before 2021 (median articles per ticker-year jumps
# about 35x between 2020 and 2021 in the live sample), so news strategies get ONE
# results window, 2021-2023. 2019-2020 appears only as a footnote (D4, option b).
NEWS_PRIMARY_START = date(2021, 1, 1)
NEWS_COVERAGE_FOOTNOTE = (
    "News strategies are tested on 2021 to 2023 only. News coverage for 2019 and "
    "2020 is too thin to give results, because 'no news' there often means "
    "'no data'. Those years are not tested."
)

MAX_ENTRY_CONDITIONS = 8

# strict: no silent type changes, so "3" is not read as 3 and true is not 1.
# extra="forbid": the agent cannot add fields we do not understand.
# frozen: a validated strategy cannot be changed after the fact.
STRICT_CONFIG = ConfigDict(strict=True, extra="forbid", frozen=True)


def _number_text(value: float) -> str:
    """Show 50.0 as "50" and 0.5 as "0.5" in error messages."""
    return f"{value:g}"


def _float_between(low: float, high: float) -> Callable[[float, ValidationInfo], float]:
    """A validator that rejects a decimal number outside [low, high] with a plain message."""

    def check(value: float, info: ValidationInfo) -> float:
        if not low <= value <= high:
            raise PydanticCustomError(
                "out_of_range",
                "{field} must be between {low} and {high}. You gave {value}.",
                {
                    "field": info.field_name,
                    "low": _number_text(low),
                    "high": _number_text(high),
                    "value": _number_text(value),
                },
            )
        # Adding 0.0 turns -0.0 into 0.0, so equal rules always hash the same.
        return float(value) + 0.0

    return check


def _int_between(low: int, high: int) -> Callable[[int, ValidationInfo], int]:
    """A validator that rejects a whole number outside [low, high] with a plain message."""

    def check(value: int, info: ValidationInfo) -> int:
        if not low <= value <= high:
            raise PydanticCustomError(
                "out_of_range",
                "{field} must be a whole number between {low} and {high}. You gave {value}.",
                {"field": info.field_name, "low": low, "high": high, "value": value},
            )
        return value

    return check


def _text_between(min_chars: int, max_chars: int, *, single_line: bool) -> Callable[..., str]:
    """A validator for user-facing text: length limits and no control characters."""

    def check(value: str, info: ValidationInfo) -> str:
        cleaned = value.strip()
        if not min_chars <= len(cleaned) <= max_chars:
            raise PydanticCustomError(
                "bad_text",
                "{field} must be {low} to {high} characters long. You gave {count}.",
                {
                    "field": info.field_name,
                    "low": min_chars,
                    "high": max_chars,
                    "count": len(cleaned),
                },
            )
        allowed = {" "} if single_line else {" ", "\n"}
        if any(not char.isprintable() and char not in allowed for char in cleaned):
            raise PydanticCustomError(
                "bad_text",
                "{field} contains control characters. Use plain text"
                + (" on one line." if single_line else "."),
                {"field": info.field_name},
            )
        return cleaned

    return check


def _bounds(low: float, high: float) -> dict[str, Any]:
    """Bounds for the exported JSON schema, so the agent can see them too."""
    return {"minimum": low, "maximum": high}


# Bounded number types. Each alias states its unit and limits once.
GapPct = Annotated[
    float, Field(json_schema_extra=_bounds(0.1, 50)), AfterValidator(_float_between(0.1, 50))
]
ReturnPct = Annotated[
    float, Field(json_schema_extra=_bounds(-90, 500)), AfterValidator(_float_between(-90, 500))
]
LookbackDays = Annotated[
    int, Field(json_schema_extra=_bounds(1, 252)), AfterValidator(_int_between(1, 252))
]
MovingAverageDays = Annotated[
    int, Field(json_schema_extra=_bounds(5, 250)), AfterValidator(_int_between(5, 250))
]
RsiPeriod = Annotated[
    int, Field(json_schema_extra=_bounds(2, 50)), AfterValidator(_int_between(2, 50))
]
RsiLevel = Annotated[
    float, Field(json_schema_extra=_bounds(1, 99)), AfterValidator(_float_between(1, 99))
]
VolumeLookback = Annotated[
    int, Field(json_schema_extra=_bounds(5, 100)), AfterValidator(_int_between(5, 100))
]
VolumeRatio = Annotated[
    float, Field(json_schema_extra=_bounds(0.1, 20)), AfterValidator(_float_between(0.1, 20))
]
NewsSessions = Annotated[
    int, Field(json_schema_extra=_bounds(1, 20)), AfterValidator(_int_between(1, 20))
]
Polarity = Annotated[
    float, Field(json_schema_extra=_bounds(-1, 1)), AfterValidator(_float_between(-1, 1))
]
MinArticles = Annotated[
    int, Field(json_schema_extra=_bounds(1, 50)), AfterValidator(_int_between(1, 50))
]
# Capped at 10 because a report date is only treated as known 10 trading days
# ahead (EARNINGS_VISIBILITY_TRADING_DAYS in bot/data/earnings.py).
DaysToEarnings = Annotated[
    int, Field(json_schema_extra=_bounds(1, 10)), AfterValidator(_int_between(1, 10))
]
DaysSinceEarnings = Annotated[
    int, Field(json_schema_extra=_bounds(0, 60)), AfterValidator(_int_between(0, 60))
]
HoldingDays = Annotated[
    int, Field(json_schema_extra=_bounds(1, 252)), AfterValidator(_int_between(1, 252))
]
StopLossPct = Annotated[
    float, Field(json_schema_extra=_bounds(0.5, 50)), AfterValidator(_float_between(0.5, 50))
]
ProfitTargetPct = Annotated[
    float, Field(json_schema_extra=_bounds(0.5, 200)), AfterValidator(_float_between(0.5, 200))
]
MaxPositions = Annotated[
    int, Field(json_schema_extra=_bounds(1, 50)), AfterValidator(_int_between(1, 50))
]
VersionNumber = Annotated[
    int, Field(json_schema_extra=_bounds(1, 500)), AfterValidator(_int_between(1, 500))
]
StrategyName = Annotated[str, AfterValidator(_text_between(3, 60, single_line=True))]
PlainEnglish = Annotated[str, AfterValidator(_text_between(10, 800, single_line=False))]

Comparison = Literal["at_least", "at_most"]
Position = Literal["above", "below"]
Relation = Literal["within", "not_within"]

# When an entry is decided and filled. "next_open": decided at the close of D,
# bought at the open of D+1. "same_open": decided and bought at the open of D,
# using D's open, overnight news and everything else as of the close of D-1.
EntryTiming = Literal["next_open", "same_open"]

# How to choose when more stocks qualify than there are free slots. Each option
# reads the number behind one entry block, so that block must be present. Ties
# are then broken by the fixed hash shuffle.
RankBy = Literal[
    "largest_gap",
    "highest_sentiment",
    "strongest_n_day_return",
    "highest_relative_volume",
    "lowest_rsi",
]
RANK_BY_BLOCK: dict[str, str] = {
    "largest_gap": "gap",
    "highest_sentiment": "news_sentiment",
    "strongest_n_day_return": "n_day_return",
    "highest_relative_volume": "relative_volume",
    "lowest_rsi": "rsi",
}
RANK_BY_TEXT: dict[str, str] = {
    "largest_gap": "the largest gaps (in either direction) first",
    "highest_sentiment": "the highest average news polarity first",
    "strongest_n_day_return": "the highest n-day return first",
    "highest_relative_volume": "the highest relative volume first",
    "lowest_rsi": "the lowest RSI first",
}


# ---------------------------------------------------------------- entry blocks


class GapCondition(BaseModel):
    """Today's open versus yesterday's close, in percent. Dividend drops are added back."""

    model_config = STRICT_CONFIG
    block: Literal["gap"]
    direction: Literal["down", "up"]
    min_gap_pct: GapPct


class NDayReturnCondition(BaseModel):
    """Total return (with dividends) over the last ``days`` trading days, in percent."""

    model_config = STRICT_CONFIG
    block: Literal["n_day_return"]
    days: LookbackDays
    comparison: Comparison
    return_pct: ReturnPct


class PriceVsMovingAverageCondition(BaseModel):
    """Today's close above or below its simple moving average over ``ma_days`` closes."""

    model_config = STRICT_CONFIG
    block: Literal["price_vs_moving_average"]
    ma_days: MovingAverageDays
    position: Position


class RsiCondition(BaseModel):
    """Wilder's RSI of the close over ``period`` days, compared with ``level``."""

    model_config = STRICT_CONFIG
    block: Literal["rsi"]
    period: RsiPeriod = 14
    comparison: Comparison
    level: RsiLevel


class RelativeVolumeCondition(BaseModel):
    """Today's volume divided by the average of the previous ``lookback_days`` days."""

    model_config = STRICT_CONFIG
    block: Literal["relative_volume"]
    lookback_days: VolumeLookback = 20
    comparison: Comparison
    ratio: VolumeRatio


class NoNewsCondition(BaseModel):
    """No articles in the last ``lookback_sessions`` news windows. 1 means since the last close."""

    model_config = STRICT_CONFIG
    block: Literal["no_news"]
    lookback_sessions: NewsSessions = 1


class NewsSentimentCondition(BaseModel):
    """Average article polarity (-1 to 1) in the last ``lookback_sessions`` news windows."""

    model_config = STRICT_CONFIG
    block: Literal["news_sentiment"]
    lookback_sessions: NewsSessions = 1
    comparison: Comparison
    polarity: Polarity
    min_articles: MinArticles = 1


class DaysToEarningsCondition(BaseModel):
    """Whether the next earnings reaction is within ``days`` trading days."""

    model_config = STRICT_CONFIG
    block: Literal["days_to_earnings"]
    relation: Relation
    days: DaysToEarnings


class DaysSinceEarningsCondition(BaseModel):
    """Whether the last earnings reaction was within ``days`` trading days (0 means today)."""

    model_config = STRICT_CONFIG
    block: Literal["days_since_earnings"]
    relation: Relation
    days: DaysSinceEarnings


class SpyFilterCondition(BaseModel):
    """SPY's close above or below its own moving average. Same answer for every stock."""

    model_config = STRICT_CONFIG
    block: Literal["spy_filter"]
    ma_days: MovingAverageDays = 200
    position: Position


EntryCondition = Annotated[
    GapCondition
    | NDayReturnCondition
    | PriceVsMovingAverageCondition
    | RsiCondition
    | RelativeVolumeCondition
    | NoNewsCondition
    | NewsSentimentCondition
    | DaysToEarningsCondition
    | DaysSinceEarningsCondition
    | SpyFilterCondition,
    Field(discriminator="block"),
]

NEWS_BLOCKS: frozenset[str] = frozenset({"no_news", "news_sentiment"})


def _check_entry_count(value: list[Any], info: ValidationInfo) -> list[Any]:
    """Require 1 to MAX_ENTRY_CONDITIONS entry conditions."""
    if not 1 <= len(value) <= MAX_ENTRY_CONDITIONS:
        raise PydanticCustomError(
            "out_of_range",
            "{field} must have between 1 and {high} conditions. You gave {count}.",
            {"field": info.field_name, "high": MAX_ENTRY_CONDITIONS, "count": len(value)},
        )
    return value


# ------------------------------------------------------------- exits, sizing


class ExitRules(BaseModel):
    """When to sell. Any combination, but at least one. The first rule hit wins."""

    model_config = STRICT_CONFIG
    holding_days: HoldingDays | None = None
    stop_loss_pct: StopLossPct | None = None
    profit_target_pct: ProfitTargetPct | None = None

    @model_validator(mode="after")
    def _needs_one_exit(self) -> "ExitRules":
        """Without any exit a position could be held forever."""
        if (
            self.holding_days is None
            and self.stop_loss_pct is None
            and self.profit_target_pct is None
        ):
            raise PydanticCustomError(
                "strategy_rule",
                "Give at least one exit: holding_days, stop_loss_pct or profit_target_pct.",
            )
        return self


class Sizing(BaseModel):
    """Equal weight: each new position gets equity divided by ``max_positions``."""

    model_config = STRICT_CONFIG
    method: Literal["equal_weight"] = "equal_weight"
    max_positions: MaxPositions


# ------------------------------------------------------------------- strategy


class StrategyDefinition(BaseModel):
    """One complete strategy. Long only, point-in-time S&P 500, entries joined with AND."""

    model_config = STRICT_CONFIG
    schema_version: Literal[1] = 1
    name: StrategyName
    description_plain_english: PlainEnglish
    version: VersionNumber = 1
    universe: Literal["sp500_point_in_time"] = "sp500_point_in_time"
    side: Literal["long"] = "long"
    entry: Annotated[list[EntryCondition], AfterValidator(_check_entry_count)]
    exit: ExitRules
    sizing: Sizing
    entry_timing: EntryTiming = "next_open"
    rank_by: RankBy | None = None

    @model_validator(mode="after")
    def _same_open_needs_a_gap(self) -> "StrategyDefinition":
        """Without a gap block, same_open would only re-read the previous close."""
        if self.entry_timing == "same_open" and not any(c.block == "gap" for c in self.entry):
            raise PydanticCustomError(
                "strategy_rule",
                "entry_timing 'same_open' needs a gap condition. Without one, every condition "
                "is read at the previous close, which is the same as buying at the next open. "
                "Add a gap condition or use entry_timing 'next_open'.",
            )
        return self

    @model_validator(mode="after")
    def _rank_by_has_its_block(self) -> "StrategyDefinition":
        """rank_by reads one block's number, so that block must be there, with one setting."""
        if self.rank_by is None:
            return self
        block = RANK_BY_BLOCK[self.rank_by]
        settings = {rank_setting(c) for c in self.entry if c.block == block}
        if not settings:
            raise PydanticCustomError(
                "strategy_rule",
                "rank_by '{rank_by}' needs an entry condition with block '{block}'. "
                "Add one or remove rank_by.",
                {"rank_by": self.rank_by, "block": block},
            )
        if len(settings) > 1:
            raise PydanticCustomError(
                "strategy_rule",
                "rank_by '{rank_by}' is unclear: there is more than one {block} condition with "
                "a different look-back. Keep one look-back or remove rank_by.",
                {"rank_by": self.rank_by, "block": block},
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_conditions(self) -> "StrategyDefinition":
        """A repeated condition adds nothing, and it would change the strategy id."""
        seen: dict[str, int] = {}
        for number, condition in enumerate(self.entry, start=1):
            key = condition.model_dump_json()
            if key in seen:
                raise PydanticCustomError(
                    "strategy_rule",
                    "Entry condition {number} is the same as condition {first}. Remove one.",
                    {"number": number, "first": seen[key]},
                )
            seen[key] = number
        return self


def rank_setting(condition: EntryCondition) -> int | None:
    """The look-back that sets the number rank_by reads from this block (None for gap)."""
    match condition:
        case NDayReturnCondition():
            return condition.days
        case RsiCondition():
            return condition.period
        case RelativeVolumeCondition():
            return condition.lookback_days
        case NewsSentimentCondition():
            return condition.lookback_sessions
        case _:
            return None


def rank_condition(strategy: StrategyDefinition) -> EntryCondition | None:
    """The entry condition whose number ``rank_by`` sorts on, or None without rank_by."""
    if strategy.rank_by is None:
        return None
    block = RANK_BY_BLOCK[strategy.rank_by]
    return next(c for c in strategy.entry if c.block == block)


# ------------------------------------------------------------ JSON and the id


def strategy_from_json(text: str) -> StrategyDefinition:
    """Parse and validate a strategy from JSON text. Raises ValidationError."""
    return StrategyDefinition.model_validate_json(text)


def strategy_to_json(strategy: StrategyDefinition) -> str:
    """The strategy as indented JSON. strategy_from_json gives back an equal object."""
    return strategy.model_dump_json(indent=2)


def strategy_json_schema() -> dict[str, Any]:
    """The JSON schema of StrategyDefinition, including every bound."""
    return StrategyDefinition.model_json_schema()


def rules_fingerprint(strategy: StrategyDefinition) -> str:
    """Canonical JSON of the trading rules only, used to build the strategy id.

    ``name``, ``description_plain_english`` and ``version`` are left out, so
    renaming a strategy cannot buy a second holdout run (D12). Entry
    conditions are sorted because AND does not depend on order.
    ``entry_timing`` and ``rank_by`` are left out when they hold their
    defaults, so strategies written before these fields existed keep their id.
    """
    rules = strategy.model_dump(
        mode="json", exclude={"name", "description_plain_english", "version"}
    )
    if rules["entry_timing"] == "next_open":
        del rules["entry_timing"]
    if rules["rank_by"] is None:
        del rules["rank_by"]
    rules["entry"] = sorted(rules["entry"], key=lambda item: json.dumps(item, sort_keys=True))
    return json.dumps(rules, sort_keys=True, separators=(",", ":"))


def strategy_id(strategy: StrategyDefinition) -> str:
    """A stable id for the trading rules, such as "strat_3f9a0c1d2e4b5a69"."""
    digest = hashlib.sha256(rules_fingerprint(strategy).encode("utf-8")).hexdigest()
    return f"strat_{digest[:16]}"


# --------------------------------------------------------------- test windows


@dataclass(frozen=True)
class BacktestWindow:
    """One period to backtest, with a footnote to print next to its results if needed
    (`coverage_warning` holds the footnote text; the name is kept for the CLI).

    "holdout" is only used by the approval flow, for the single run from 2024-01-01.
    """

    label: Literal["primary", "holdout"]
    start: date
    end: date
    coverage_warning: str | None


def uses_news(strategy: StrategyDefinition) -> bool:
    """True if any entry condition reads news or sentiment."""
    return any(condition.block in NEWS_BLOCKS for condition in strategy.entry)


def backtest_windows(strategy: StrategyDefinition) -> tuple[BacktestWindow, ...]:
    """The one results window to test (D4, option b).

    News strategies: 2021-2023 only. The window carries the coverage footnote in
    `coverage_warning` (show it as a footnote next to results, not as a result).
    All others: 2016-2023, no footnote.
    """
    if not uses_news(strategy):
        return (BacktestWindow("primary", BACKTEST_START, BACKTEST_END, None),)
    return (BacktestWindow("primary", NEWS_PRIMARY_START, BACKTEST_END, NEWS_COVERAGE_FOOTNOTE),)


# ------------------------------------------------- rules read back to the user


def describe_condition(condition: EntryCondition, timing: EntryTiming = "next_open") -> str:
    """One condition as a short plain English sentence, built from the validated fields.

    With ``timing`` "same_open", news is read up to today's open and every
    other block except gap is read at the previous close, and the text says so.
    """
    if timing == "same_open":
        return _describe_at_open(condition)
    match condition:
        case GapCondition():
            word = "below" if condition.direction == "down" else "above"
            return (
                f"Today's open is at least {_number_text(condition.min_gap_pct)}% "
                f"{word} yesterday's close."
            )
        case NDayReturnCondition():
            return (
                f"The {condition.days}-day return is {_comparison_text(condition.comparison)} "
                f"{_number_text(condition.return_pct)}%."
            )
        case PriceVsMovingAverageCondition():
            return f"The close is {condition.position} its {condition.ma_days}-day moving average."
        case RsiCondition():
            return (
                f"The {condition.period}-day RSI is {_comparison_text(condition.comparison)} "
                f"{_number_text(condition.level)}."
            )
        case RelativeVolumeCondition():
            return (
                f"Today's volume is {_comparison_text(condition.comparison)} "
                f"{_number_text(condition.ratio)} times its {condition.lookback_days}-day average."
            )
        case _:
            return _describe_event_condition(condition)


def _describe_event_condition(condition: EntryCondition) -> str:
    """Sentences for the news, earnings and market filter blocks."""
    match condition:
        case NoNewsCondition():
            return f"No news articles {_sessions_text(condition.lookback_sessions)}."
        case NewsSentimentCondition():
            return (
                f"At least {condition.min_articles} article(s) "
                f"{_sessions_text(condition.lookback_sessions)}, with average polarity "
                f"{_comparison_text(condition.comparison)} {_number_text(condition.polarity)}."
            )
        case DaysToEarningsCondition():
            word = "is" if condition.relation == "within" else "is not"
            return f"The next earnings reaction {word} within {condition.days} trading days."
        case DaysSinceEarningsCondition():
            word = "was" if condition.relation == "within" else "was not"
            return f"The last earnings reaction {word} within {condition.days} trading days."
        case SpyFilterCondition():
            return f"SPY closes {condition.position} its {condition.ma_days}-day moving average."
        case _:
            raise ValueError(f"No description for block {condition.block!r}.")


def describe_rules(strategy: StrategyDefinition) -> str:
    """The rules Python will actually run, in plain English, for the user to confirm."""
    timing = strategy.entry_timing
    lines = [
        f"{strategy.name} (version {strategy.version})",
        "Universe: stocks in the S&P 500 on the day of the signal. Long only.",
        _timing_text(timing),
    ]
    lines += [
        f"  {n}. {describe_condition(c, timing)}" for n, c in enumerate(strategy.entry, start=1)
    ]
    if strategy.rank_by is not None:
        lines.append(
            "If more stocks qualify than there are free slots, buy "
            f"{RANK_BY_TEXT[strategy.rank_by]}. Ties are broken by a fixed shuffle."
        )
    lines.append("Sell at the first of:")
    lines += [f"  - {text}" for text in _exit_texts(strategy.exit)]
    lines.append(
        f"Sizing: up to {strategy.sizing.max_positions} positions, each 1/"
        f"{strategy.sizing.max_positions} of the portfolio when bought."
    )
    return "\n".join(lines)


def _timing_text(timing: EntryTiming) -> str:
    """The line that says when entries are decided and bought."""
    if timing == "same_open":
        return (
            "Buy at the open (09:30 New York) of the same day when ALL of these are true "
            "at that open:"
        )
    return "Buy at the next open when ALL of these are true at the close:"


def _describe_at_open(condition: EntryCondition) -> str:
    """A condition read at today's open: gap and news as of the open, the rest at the last close."""
    match condition:
        case GapCondition():
            return describe_condition(condition)
        case NoNewsCondition():
            return f"No news articles {_sessions_text_at_open(condition.lookback_sessions)}."
        case NewsSentimentCondition():
            return (
                f"At least {condition.min_articles} article(s) "
                f"{_sessions_text_at_open(condition.lookback_sessions)}, with average polarity "
                f"{_comparison_text(condition.comparison)} {_number_text(condition.polarity)}."
            )
        case _:
            sentence = describe_condition(condition).removesuffix(".")
            return f"{sentence}, as of the previous close."


def _sessions_text_at_open(sessions: int) -> str:
    """How a news lookback ending at today's open reads in a sentence."""
    if sessions == 1:
        return "from the previous close (16:00 New York) to today's open (09:30 New York)"
    return (
        f"from the close {sessions} trading days ago (16:00 New York) "
        "to today's open (09:30 New York)"
    )


def _exit_texts(rules: ExitRules) -> list[str]:
    """Each exit rule as a short phrase."""
    texts: list[str] = []
    if rules.holding_days is not None:
        texts.append(f"after {rules.holding_days} trading days (sold at the next open)")
    if rules.stop_loss_pct is not None:
        texts.append(
            f"the price falling {_number_text(rules.stop_loss_pct)}% below the buy price "
            "(stop loss)"
        )
    if rules.profit_target_pct is not None:
        texts.append(
            f"the price rising {_number_text(rules.profit_target_pct)}% above the buy price "
            "(profit target)"
        )
    return texts


def _comparison_text(comparison: Comparison) -> str:
    """ "at_least" reads as "at least"."""
    return "at least" if comparison == "at_least" else "at most"


def _sessions_text(sessions: int) -> str:
    """How a news lookback reads in a sentence."""
    if sessions == 1:
        return "since the previous close"
    return f"in the last {sessions} trading days (close to close)"


# ------------------------------------------------- errors the agent can relay


def explain_validation_error(error: ValidationError) -> list[str]:
    """Turn a ValidationError into short plain English lines, one per problem."""
    return [_explain_one(item) for item in error.errors()]


def _explain_one(item: ErrorDetails) -> str:
    """One pydantic error dict as one sentence, prefixed with where it happened."""
    kind: str = item["type"]
    loc: tuple[str | int, ...] = tuple(item["loc"])
    context: dict[str, Any] = dict(item.get("ctx") or {})
    if kind == "strategy_rule":
        return _where(loc) + item["msg"]
    if kind in ("union_tag_invalid", "union_tag_not_found"):
        return _where(loc) + _explain_block_tag(kind, context)
    field = str(loc[-1]) if loc else "strategy"
    prefix = _where(loc[:-1])
    if kind in ("out_of_range", "bad_text"):
        return prefix + item["msg"]
    return prefix + _explain_builtin(kind, field, item.get("input"), context, item["msg"])


def _explain_block_tag(kind: str, context: dict[str, Any]) -> str:
    """Messages for a missing or unknown ``block`` name."""
    if kind == "union_tag_not_found":
        return "Each entry condition needs a 'block' field naming its building block."
    return (
        f"'{context.get('tag')}' is not a supported building block. "
        f"Supported blocks: {context.get('expected_tags')}."
    )


def _explain_builtin(kind: str, field: str, given: Any, context: dict[str, Any], raw: str) -> str:
    """Plain wording for pydantic's own error types."""
    if kind == "missing":
        return f"{field} is required."
    if kind == "extra_forbidden":
        return f"'{field}' is not a supported field. Remove it."
    if kind == "literal_error":
        return f"{field} must be one of {context.get('expected')}. You gave {given!r}."
    if kind in ("int_type", "int_from_float", "int_parsing"):
        return f"{field} must be a whole number. You gave {given!r}."
    if kind in ("float_type", "float_parsing"):
        return f"{field} must be a number. You gave {given!r}."
    if kind == "string_type":
        return f"{field} must be text. You gave {given!r}."
    if kind in ("model_type", "model_attributes_type", "dict_type"):
        return f"{field} must be an object with named fields."
    if kind == "list_type":
        return f"{field} must be a list."
    if kind == "json_invalid":
        return f"The strategy is not valid JSON ({context.get('error', raw)})."
    return f"{field}: {raw}"


def _where(loc: tuple[str | int, ...]) -> str:
    """A readable prefix such as "Entry condition 2 (rsi): " for an error location."""
    if not loc:
        return ""
    if loc[0] == "entry" and len(loc) >= 2 and isinstance(loc[1], int):
        block = f" ({loc[2]})" if len(loc) >= 3 else ""
        return f"Entry condition {loc[1] + 1}{block}: "
    if loc[0] == "exit":
        return "Exit rules: "
    if loc[0] == "sizing":
        return "Sizing: "
    return ""
