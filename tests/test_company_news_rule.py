"""The optional company-news rule (D31): default changes nothing, options filter as specified."""

import inspect
from datetime import date
from typing import Any

import pandas as pd
import pytest

from bot.data import eodhd_client as client
from bot.data import market
from bot.data.news import (
    COMPANY_NEWS_RULE,
    CURRENT_NEWS_RULE,
    CompanyNewsRule,
    build_news_frame,
    company_name_phrases,
    parse_news_rows,
    row_matches_rule,
    title_names_company,
    us_symbol_count,
)
from bot.strategy.blocks import news_sentiment_block, no_news_block, slice_to_decision_date
from bot.strategy.schema import NewsSentimentCondition, NoNewsCondition
from tests.backtest_helpers import assemble_market, flat_bar, raw_frame

SYMBOL = "AAPL.US"
NAMES = ("Apple Inc.",)
DAYS = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
STRICT = CompanyNewsRule(max_us_symbols=1)


def row(
    when: str, link: str, symbols: list[str], title: str = "Markets today", polarity: float = 0.0
) -> Any:
    return {
        "date": when,
        "title": title,
        "content": "text",
        "link": link,
        "symbols": symbols,
        "tags": [],
        "sentiment": {"polarity": polarity, "neg": 0.0, "neu": 1.0, "pos": 0.0},
    }


COMPANY = row("2023-01-04T15:00:00+00:00", "a", [SYMBOL, "AAPL.MX"], "Apple raises prices", 0.8)
ROUNDUP = row("2023-01-04T16:00:00+00:00", "b", [SYMBOL, "MSFT.US", "AMZN.US"], "Dow wrap", -0.9)
LATE_ROUNDUP = row("2023-01-05T15:00:00+00:00", "c", [SYMBOL, "MSFT.US", "KO.US"], "Movers", -0.5)
ROWS = [COMPANY, ROUNDUP, LATE_ROUNDUP]


def mixed_rows() -> list[Any]:
    """Rows with duplicates, foreign tags, other symbols and a big roundup."""
    many = [SYMBOL, *(f"T{n}.US" for n in range(40))]
    return [
        *ROWS,
        row("2023-01-04T15:05:00+00:00", "a", [SYMBOL]),
        row("2023-01-04T17:00:00+00:00", "d", ["MSFT.US"]),
        row("2023-01-05T14:00:00+00:00", "e", many, "Fund buys 40 stocks"),
        row("2023-01-05T14:30:00+00:00", "f", ["AAPL.MX", "AAPL34.SA"]),
        row("2023-01-05T15:00:00+00:00", "", [SYMBOL], "Apple news"),
    ]


# ---------------------------------------------------------------- default rule


def test_default_rule_keeps_every_row() -> None:
    assert CompanyNewsRule() == CURRENT_NEWS_RULE
    assert CURRENT_NEWS_RULE.is_default()
    for raw_row in mixed_rows():
        assert row_matches_rule(raw_row, SYMBOL, CURRENT_NEWS_RULE, NAMES)


def test_default_rule_output_equals_call_without_rule() -> None:
    rows = mixed_rows()
    assert parse_news_rows(rows, SYMBOL) == parse_news_rows(rows, SYMBOL, CURRENT_NEWS_RULE, NAMES)


def test_loader_and_parser_default_to_current_rule() -> None:
    for function in (market.load_market_data, market.load_market_report):
        assert inspect.signature(function).parameters["news_rule"].default == COMPANY_NEWS_RULE
    assert inspect.signature(parse_news_rows).parameters["rule"].default == CURRENT_NEWS_RULE


# ------------------------------------------------------------------- options


def test_us_symbol_count_ignores_foreign_listings_and_repeats() -> None:
    raw_row = row("2023-01-04T15:00:00+00:00", "x", [SYMBOL, SYMBOL, "AAPL.MX", "MSFT.US"])
    assert us_symbol_count(raw_row) == 2


@pytest.mark.parametrize(
    ("limit", "kept_links"), [(1, ["a"]), (2, ["a"]), (3, ["a", "b", "c"]), (None, ["a", "b", "c"])]
)
def test_max_us_symbols(limit: int | None, kept_links: list[str]) -> None:
    rule = CompanyNewsRule(max_us_symbols=limit)
    assert [a.link for a in parse_news_rows(ROWS, SYMBOL, rule, NAMES)] == kept_links


def test_name_in_title_rule() -> None:
    rule = CompanyNewsRule(require_name_in_title=True)
    assert [a.link for a in parse_news_rows(ROWS, SYMBOL, rule, NAMES)] == ["a"]
    by_ticker = row("2023-01-04T15:00:00+00:00", "t", [SYMBOL, "MSFT.US"], "Is AAPL a buy?")
    assert row_matches_rule(by_ticker, SYMBOL, rule, ())


def test_combine_any_and_all() -> None:
    title_only = row("2023-01-04T15:00:00+00:00", "t", [SYMBOL, "MSFT.US", "KO.US"], "Apple up")
    few_only = row("2023-01-04T15:00:00+00:00", "f", [SYMBOL, "MSFT.US"], "Big tech")
    neither = row("2023-01-04T15:00:00+00:00", "n", [SYMBOL, "MSFT.US", "KO.US"], "Dow wrap")
    either = CompanyNewsRule(max_us_symbols=2, require_name_in_title=True, combine="any")
    both = CompanyNewsRule(max_us_symbols=2, require_name_in_title=True, combine="all")
    assert [
        row_matches_rule(r, SYMBOL, either, NAMES) for r in (title_only, few_only, neither)
    ] == [
        True,
        True,
        False,
    ]
    assert [row_matches_rule(r, SYMBOL, both, NAMES) for r in (title_only, few_only, neither)] == [
        False,
        False,
        False,
    ]


def test_rule_applies_before_dedup() -> None:
    """A wide copy of a story is dropped first, so its narrow copy survives."""
    wide = row("2023-01-04T15:00:00+00:00", "same", [SYMBOL, "MSFT.US"], "Story")
    narrow = row("2023-01-04T15:10:00+00:00", "same", [SYMBOL], "Story")
    kept = parse_news_rows([wide, narrow], SYMBOL, STRICT)
    assert len(kept) == 1
    assert kept[0].published_at == pd.Timestamp("2023-01-04T15:10:00+00:00")
    assert len(parse_news_rows([wide, narrow], SYMBOL)) == 1


def test_company_name_phrases() -> None:
    assert company_name_phrases("The Coca-Cola Company") == ("coca cola", "cocacola")
    assert company_name_phrases("Amazon.com Inc") == ("amazon",)
    assert company_name_phrases("Etsy, Inc.") == ("etsy",)
    assert company_name_phrases("Example Holdings Class A") == (
        "example holdings",
        "exampleholdings",
    )
    assert company_name_phrases("Inc") == ()


def test_title_match_rules() -> None:
    assert title_names_company("COCA-COLA beats", "KO.US", ("The Coca-Cola Company",))
    assert title_names_company("ExxonMobil profit", "XOM.US", ("Exxon Mobil Corp",))
    assert title_names_company("Buffett adds to BRK.B", "BRK-B.US")
    assert title_names_company("Don&#039;t sell KO now", "KO.US")
    assert not title_names_company("kombucha is ko", "KO.US")
    assert not title_names_company("TKO wins", "KO.US")
    assert not title_names_company("Pineapple prices", SYMBOL, NAMES)


# ------------------------------------------- both blocks see one article set


def frame_for(rule: CompanyNewsRule) -> pd.DataFrame:
    articles = parse_news_rows(ROWS, SYMBOL, rule, NAMES)
    return build_news_frame(articles, pd.Series(float("nan"), index=DAYS), DAYS)


def blocks_on(rule: CompanyNewsRule, day: date) -> tuple[bool, bool]:
    """(no_news, positive sentiment) for AAPL on ``day`` with news built under ``rule``."""
    frame = frame_for(rule)
    prices = {SYMBOL: raw_frame(DAYS, {n: flat_bar(100.0) for n in range(len(DAYS))})}
    tables = {
        column: frame[[column]].rename(columns={column: SYMBOL})
        for column in ("article_count", "mean_polarity")
    }
    stock = assemble_market(
        DAYS, prices, article_count=tables["article_count"], mean_polarity=tables["mean_polarity"]
    )
    data = slice_to_decision_date(stock, day)
    quiet = NoNewsCondition(block="no_news", lookback_sessions=1)
    happy = NewsSentimentCondition(
        block="news_sentiment", lookback_sessions=1, comparison="at_least", polarity=0.5
    )
    return bool(no_news_block(data, quiet)[SYMBOL]), bool(news_sentiment_block(data, happy)[SYMBOL])


def test_frame_counts_and_polarity_use_the_kept_articles() -> None:
    strict, current = frame_for(STRICT), frame_for(CURRENT_NEWS_RULE)
    day = pd.Timestamp("2023-01-04")
    assert strict.loc[day, "article_count"] == 1
    assert strict.loc[day, "mean_polarity"] == pytest.approx(0.8)
    assert current.loc[day, "article_count"] == 2
    assert current.loc[day, "mean_polarity"] == pytest.approx(-0.05)


def test_no_news_and_sentiment_blocks_agree_on_the_filtered_set() -> None:
    assert blocks_on(CURRENT_NEWS_RULE, date(2023, 1, 4)) == (False, False)
    assert blocks_on(STRICT, date(2023, 1, 4)) == (False, True)
    assert blocks_on(CURRENT_NEWS_RULE, date(2023, 1, 5)) == (False, False)
    assert blocks_on(STRICT, date(2023, 1, 5)) == (True, False)


# ------------------------------------------------- unknown stays unknown


def test_first_day_stays_nan_under_every_rule() -> None:
    for rule in (CURRENT_NEWS_RULE, STRICT, CompanyNewsRule(require_name_in_title=True)):
        frame = frame_for(rule)
        assert frame.iloc[0][["article_count", "open_article_count"]].isna().all()


def test_unsupported_symbol_stays_unknown_with_a_strict_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise client.NewsSymbolUnsupportedError(422, "no news for this code")

    monkeypatch.setattr(client, "get_news", refuse)
    frame = market._news_frame(
        "OLD_old", "OLD_old.US", DAYS, date(2023, 1, 1), date(2023, 1, 31), False, rule=STRICT
    )
    assert frame.isna().all().all()
    assert not no_news_block_fires(frame)


def no_news_block_fires(frame: pd.DataFrame) -> bool:
    """True if any day reads as 'known zero articles'."""
    return bool((frame["article_count"] == 0).any())


def test_loader_news_frame_applies_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "get_news", lambda *a, **k: ROWS)
    monkeypatch.setattr(client, "get_daily_sentiment", lambda *a, **k: [])
    args = ("AAPL", SYMBOL, DAYS, date(2023, 1, 1), date(2023, 1, 31), False)
    strict = market._news_frame(*args, rule=STRICT, names=NAMES)
    current = market._news_frame(*args, rule=CURRENT_NEWS_RULE)
    assert strict["article_count"].iloc[1:].tolist() == [1.0, 0.0, 0.0]
    assert current["article_count"].iloc[1:].tolist() == [2.0, 1.0, 0.0]


def test_default_loader_rule_drops_articles_tagged_on_three_us_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The locked 2-symbol rule is the default, and both news blocks see the same set."""
    monkeypatch.setattr(client, "get_news", lambda *a, **k: ROWS)
    monkeypatch.setattr(client, "get_daily_sentiment", lambda *a, **k: [])
    args = ("AAPL", SYMBOL, DAYS, date(2023, 1, 1), date(2023, 1, 31), False)
    default = market._news_frame(*args)
    explicit = market._news_frame(*args, rule=COMPANY_NEWS_RULE)
    everything = market._news_frame(*args, rule=CURRENT_NEWS_RULE)
    pd.testing.assert_frame_equal(default, explicit)
    # ROUNDUP and LATE_ROUNDUP carry 3 .US symbols, so only COMPANY counts.
    assert default["article_count"].iloc[1:].tolist() == [1.0, 0.0, 0.0]
    assert everything["article_count"].iloc[1:].tolist() == [2.0, 1.0, 0.0]
    # The sentiment block reads the same filtered articles: no -0.9 or -0.5 polarity.
    assert default["mean_polarity"].iloc[2] == pytest.approx(0.8) or pd.isna(
        default["mean_polarity"].iloc[2]
    )
    assert (default["mean_polarity"].dropna() >= 0).all()


def test_company_names_by_code_keeps_every_spelling() -> None:
    rows: Any = [
        {"Code": "TER", "Name": "Teradyne Inc"},
        {"Code": "TER", "Name": "Teradyne, Inc."},
        {"Code": "TER", "Name": "Teradyne Inc"},
        {"Code": "X", "Name": ""},
    ]
    assert market.company_names_by_code(rows) == {
        "TER": ("Teradyne Inc", "Teradyne, Inc."),
        "X": (),
    }
