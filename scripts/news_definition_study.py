"""Study what "company news" should mean before news strategies run (D31).

Offline only. In live mode every read comes from data/cache/eodhd and any
network request raises, so the study makes zero EODHD calls. In fake mode
(EODHD_MODE=fake) it runs on the fake transport and every number is
SIMULATED DATA.

    uv run python scripts/news_definition_study.py
    uv run python scripts/news_definition_study.py --dump-sample sample.json

It answers the user's three questions on the live sample, 2021-2023:

1. Per ticker and year, the share of decision windows with zero articles.
2. How many symbols each article is tagged with, and 100 hand-labelled
   articles (20 each for 5 tickers). The labels are a model's judgement,
   not a human's.
3. Candidate stricter definitions (``CompanyNewsRule``), each with its
   zero-news share, its effect on the labelled articles and the gap-down
   example's trade count. The study does not pick one.

Writes ``news_definition_study.md`` and ``.json`` under the output folder.
"""

import argparse
import hashlib
import html
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bot.backtest.costs import make_cost_model
from bot.backtest.engine import run_backtest
from bot.config import BACKTEST_END, MEMBERSHIP_START, SIMULATED_LABEL, data_mode, output_dir
from bot.data import eodhd_client as client
from bot.data import market as market_loader
from bot.data.market import PRICE_START, company_names_by_code
from bot.data.membership import MembershipHistory, RawMembershipRow, build_membership_history
from bot.data.news import (
    CURRENT_NEWS_RULE,
    CompanyNewsRule,
    RawNewsRow,
    align_daily_sentiment,
    build_news_frame,
    parse_news_rows,
    row_matches_rule,
    title_names_company,
    us_symbol_count,
)
from bot.strategy.blocks import MarketData, build_market_data
from bot.strategy.schema import BacktestWindow, StrategyDefinition, strategy_from_json

SAMPLE_TICKERS = [
    *("AAPL", "TSLA", "NVDA", "BRK-B", "BF-B", "APC_old", "CSRA_old", "AET", "TWTR", "ATVI"),
    *("CELG", "MON", "AGN", "ETSY", "TER", "CTLT", "AMD", "MSFT", "AMZN", "XOM"),
    *("JPM", "JNJ", "PG", "KO"),
]
STUDY_START = date(2021, 1, 1)
YEARS = (2021, 2022, 2023)
LABEL_TICKERS = ("AAPL", "TSLA", "JNJ", "TER", "ETSY")
ARTICLES_PER_TICKER = 20
SEED = "news-definition-study-2026-11-02"
MIN_SPLIT_DAYS = 30
COST_BPS = 5.0
TITLE_CHARS = 90
BACKTEST_LABEL = "live sample, 23 tickers, not the S&P 500, pipeline check only"
FAKE_BACKTEST_LABEL = "SIMULATED DATA, 23 fake names, pipeline check only"
MODEL_LABEL_NOTE = (
    "Labels are a language model's judgement from the title and the first part of the "
    "text, not a human's. Treat them as a rough check."
)
LABELS = ("company-specific", "roundup/list", "passing mention", "wrong company")

OPTIONS: dict[str, CompanyNewsRule] = {
    "current (no filter)": CURRENT_NEWS_RULE,
    ".US symbols <= 1": CompanyNewsRule(max_us_symbols=1),
    ".US symbols <= 2": CompanyNewsRule(max_us_symbols=2),
    ".US symbols <= 3": CompanyNewsRule(max_us_symbols=3),
    ".US symbols <= 5": CompanyNewsRule(max_us_symbols=5),
    "name or ticker in title": CompanyNewsRule(require_name_in_title=True),
    "title OR .US <= 2": CompanyNewsRule(
        max_us_symbols=2, require_name_in_title=True, combine="any"
    ),
    "title AND .US <= 5": CompanyNewsRule(max_us_symbols=5, require_name_in_title=True),
}

GAP_DOWN = {
    "name": "Gap down, no news",
    "description_plain_english": (
        "Buy S&P 500 stocks whose open is 3% or more below the previous close, "
        "with no news since the previous close. Sell after 3 trading days."
    ),
    "version": 1,
    "entry": [
        {"block": "gap", "direction": "down", "min_gap_pct": 3},
        {"block": "no_news", "lookback_sessions": 1},
    ],
    "exit": {"holding_days": 3},
    "sizing": {"max_positions": 10},
}

# Hand labels for the 100 sampled articles, keyed by the first 12 hex digits
# of sha1(link). Set after reading each title and the start of its text
# (see --dump-sample). A language model's judgement, not a human's.
# "wrong company": tagged TER.US but about TerrAscend (CSE: TER), a cannabis firm.
HAND_LABELS: dict[str, str] = {
    "afe4f36fa372": "roundup/list",  # AAPL #0
    "dd810e1e9092": "company-specific",  # AAPL #1
    "c40e17dfc0a3": "company-specific",  # AAPL #2
    "f6ec3ca1c523": "company-specific",  # AAPL #3
    "2943ab7efff1": "passing mention",  # AAPL #4
    "119aeb4f1834": "roundup/list",  # AAPL #5
    "3c6b599e760a": "roundup/list",  # AAPL #6
    "df983226db97": "roundup/list",  # AAPL #7
    "edb6481e4408": "roundup/list",  # AAPL #8
    "2f51823a7142": "passing mention",  # AAPL #9
    "9011ac05b446": "company-specific",  # AAPL #10
    "62435637cbff": "company-specific",  # AAPL #11
    "e9ecc19b1669": "passing mention",  # AAPL #12
    "8b3fe0fdf69e": "roundup/list",  # AAPL #13
    "0a620e3871ff": "roundup/list",  # AAPL #14
    "d4a29756e0a1": "roundup/list",  # AAPL #15
    "b391c25a97a7": "roundup/list",  # AAPL #16
    "3927a7fe83f6": "company-specific",  # AAPL #17
    "e89c1eaeb196": "passing mention",  # AAPL #18
    "68fd68550e56": "roundup/list",  # AAPL #19
    "6cb971bd500d": "company-specific",  # TSLA #20
    "146360af30bf": "roundup/list",  # TSLA #21
    "92140a242d75": "company-specific",  # TSLA #22
    "0785bc2f890d": "roundup/list",  # TSLA #23
    "414b4ef90229": "passing mention",  # TSLA #24
    "ed52f4643376": "passing mention",  # TSLA #25
    "ff7b3c9e1075": "company-specific",  # TSLA #26
    "2513615cb97d": "roundup/list",  # TSLA #27
    "d2fe41e65538": "roundup/list",  # TSLA #28
    "dbc193a4433e": "company-specific",  # TSLA #29
    "19fee89a4070": "roundup/list",  # TSLA #30
    "59457bd3703a": "passing mention",  # TSLA #31
    "de46169e542d": "passing mention",  # TSLA #32
    "97fd566295cf": "company-specific",  # TSLA #33
    "1196cc3c5310": "company-specific",  # TSLA #34
    "1589383e0ad4": "company-specific",  # TSLA #35
    "80091ea7fdf9": "company-specific",  # TSLA #36
    "e72d52864b24": "company-specific",  # TSLA #37
    "689a141e69db": "roundup/list",  # TSLA #38
    "6b19bb01c358": "passing mention",  # TSLA #39
    "f12720f43524": "company-specific",  # JNJ #40
    "c412575e1f90": "company-specific",  # JNJ #41
    "523f91a2f5b2": "roundup/list",  # JNJ #42
    "870fd129fd8b": "company-specific",  # JNJ #43
    "f566da6a523d": "company-specific",  # JNJ #44
    "a3726ada5538": "roundup/list",  # JNJ #45
    "02501c97a1eb": "passing mention",  # JNJ #46
    "595c8575115b": "passing mention",  # JNJ #47
    "f754cb54a9bc": "roundup/list",  # JNJ #48
    "931c04cffff0": "company-specific",  # JNJ #49
    "ba0cb4cdab79": "roundup/list",  # JNJ #50
    "f3df297b8cab": "company-specific",  # JNJ #51
    "109dfcfb5164": "passing mention",  # JNJ #52
    "b6886ef5daca": "roundup/list",  # JNJ #53
    "f44c4617ee42": "roundup/list",  # JNJ #54
    "f673e7060c8c": "company-specific",  # JNJ #55
    "59a580538fa4": "company-specific",  # JNJ #56
    "7a83e3234915": "passing mention",  # JNJ #57
    "075ff424c7ab": "roundup/list",  # JNJ #58
    "e0f7256fcc57": "company-specific",  # JNJ #59
    "cd95729f36be": "company-specific",  # TER #60
    "807b71a4d803": "passing mention",  # TER #61
    "7b7770a3af20": "wrong company",  # TER #62
    "810e42e53048": "wrong company",  # TER #63
    "ae1461770ea3": "company-specific",  # TER #64
    "546eacf6a3a1": "passing mention",  # TER #65
    "98643178c66e": "passing mention",  # TER #66
    "07e08b2780de": "company-specific",  # TER #67
    "576eaed8423b": "company-specific",  # TER #68
    "cfcb0a9238bb": "roundup/list",  # TER #69
    "baefa4ca2118": "passing mention",  # TER #70
    "37a95fa1bc2b": "passing mention",  # TER #71
    "0664b455f787": "roundup/list",  # TER #72
    "4b143e4967da": "roundup/list",  # TER #73
    "4bcadae3a5a3": "roundup/list",  # TER #74
    "93b4ac777cf0": "roundup/list",  # TER #75
    "26ad417ad8a7": "company-specific",  # TER #76
    "9eb182d8a477": "roundup/list",  # TER #77
    "05928ae2c0ff": "passing mention",  # TER #78
    "f8fb17cfb7e4": "roundup/list",  # TER #79
    "b287baa4ba33": "company-specific",  # ETSY #80
    "45e33f4d9103": "roundup/list",  # ETSY #81
    "98868aa13bba": "roundup/list",  # ETSY #82
    "410b0fcb4f46": "company-specific",  # ETSY #83
    "735bf31fb0c1": "roundup/list",  # ETSY #84
    "86cd3b0d54e2": "roundup/list",  # ETSY #85
    "0454a1f7bd40": "roundup/list",  # ETSY #86
    "1c886f4aa2df": "company-specific",  # ETSY #87
    "09b70a92fb5e": "company-specific",  # ETSY #88
    "bbf137200f4c": "company-specific",  # ETSY #89
    "af10d3b13967": "company-specific",  # ETSY #90
    "13cae116baf4": "company-specific",  # ETSY #91
    "5b7889b9f927": "roundup/list",  # ETSY #92
    "9a62780cf06f": "company-specific",  # ETSY #93
    "5e520e8b52da": "company-specific",  # ETSY #94
    "305f1da0e351": "roundup/list",  # ETSY #95
    "b2d425a78e7d": "company-specific",  # ETSY #96
    "7e3c2e1ea487": "company-specific",  # ETSY #97
    "31a7f292445b": "roundup/list",  # ETSY #98
    "1f35855af3b9": "passing mention",  # ETSY #99
}


@dataclass(frozen=True)
class SampleData:
    """Everything the study reads: tickers, names, calendar, membership and raw rows."""

    tickers: list[str]
    names: dict[str, tuple[str, ...]]
    days: pd.DatetimeIndex
    history: MembershipHistory
    news_rows: dict[str, list[RawNewsRow]]
    offline: bool


# ------------------------------------------------------------------ loading


def block_network() -> None:
    """In live mode, make any EODHD request raise, so a cache miss cannot spend calls."""

    def refuse(url: str, query: dict[str, Any], token: str) -> Any:
        raise RuntimeError(f"News study is offline only, refused a request to {url}.")

    client._http_get = refuse


def cached_news_rows(ticker: str, start: date, end: date, offline: bool) -> list[RawNewsRow]:
    """Cached articles for a window, joining the halves the download split for busy tickers.

    Same halving as scripts/download_data.py: a missing file means "try the halves".
    """
    try:
        return cast(list[RawNewsRow], client.get_news(ticker, start, end, offline=offline))
    except client.OfflineCacheMissError:
        if (end - start).days < MIN_SPLIT_DAYS:
            raise
    middle = date.fromordinal((start.toordinal() + end.toordinal()) // 2)
    first = cached_news_rows(ticker, start, middle, offline)
    second = cached_news_rows(ticker, date.fromordinal(middle.toordinal() + 1), end, offline)
    return first + second


def pick_tickers(history: MembershipHistory, offline: bool) -> list[str]:
    """The live sample with news, or in fake mode 23 fake members of 2021-2023."""
    if offline:
        return [t for t in SAMPLE_TICKERS if not client.is_news_unavailable(t)]
    members = sorted(s.removesuffix(".US") for s in history.universe(STUDY_START, BACKTEST_END))
    return [t for t in members if not t.endswith("_old")][:23]


def load_sample() -> SampleData:
    """Membership, names, SPY calendar and raw news rows for the sample tickers."""
    offline = data_mode() == "live"
    raw = cast(list[RawMembershipRow], client.get_index_history(offline=offline))
    history = build_membership_history(raw)
    spy = client.get_eod_prices("SPY", PRICE_START, BACKTEST_END, offline=offline)
    days = pd.DatetimeIndex(sorted(pd.Timestamp(row["date"]) for row in spy))
    tickers = pick_tickers(history, offline)
    rows = {t: cached_news_rows(t, MEMBERSHIP_START, BACKTEST_END, offline) for t in tickers}
    return SampleData(tickers, company_names_by_code(raw), days, history, rows, offline)


# ------------------------------------------------------------ item 1: zero-news share


def article_counts(data: SampleData, ticker: str, rule: CompanyNewsRule) -> pd.Series:
    """Daily ``article_count`` exactly as the no_news block sees it, under ``rule``."""
    symbol = f"{ticker}.US"
    articles = parse_news_rows(data.news_rows[ticker], symbol, rule, data.names.get(ticker, ()))
    no_sentiment = pd.Series(float("nan"), index=data.days)
    return build_news_frame(articles, no_sentiment, data.days)["article_count"]


def member_price_days(data: SampleData, ticker: str, closes: pd.Series | None) -> pd.DatetimeIndex:
    """Trading days in the study window when the ticker was a member and had a price row."""
    symbol = f"{ticker}.US"
    window = data.days[
        (data.days >= pd.Timestamp(STUDY_START)) & (data.days <= pd.Timestamp(BACKTEST_END))
    ]
    keep = [d for d in window if symbol in data.history.members_on(d.date())]
    if closes is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex([d for d in keep if pd.notna(closes.get(d))])


def zero_share_table(
    data: SampleData, rule: CompanyNewsRule, eligible: dict[str, pd.DatetimeIndex]
) -> dict[str, dict[int, dict[str, float | int | None]]]:
    """Per ticker and year: eligible days and the share of them with zero articles."""
    table: dict[str, dict[int, dict[str, float | int | None]]] = {}
    for ticker in data.tickers:
        counts = article_counts(data, ticker, rule)
        table[ticker] = {}
        for year in YEARS:
            days = eligible[ticker][eligible[ticker].year == year]
            values = counts.reindex(days)
            share = float((values == 0).mean()) if len(days) else None
            table[ticker][year] = {"days": len(days), "zero_share": share}
    return table


def year_summary(
    table: dict[str, dict[int, dict[str, float | int | None]]],
) -> dict[int, dict[str, float | int | None]]:
    """Median and 10th percentile of the zero-news share across tickers, per year."""
    summary: dict[int, dict[str, float | int | None]] = {}
    for year in YEARS:
        shares = [
            float(cast(float, row[year]["zero_share"]))
            for row in table.values()
            if row[year]["zero_share"] is not None
        ]
        series = pd.Series(shares, dtype="float64")
        summary[year] = {
            "tickers": len(shares),
            "median": float(series.median()) if shares else None,
            "p10": float(series.quantile(0.1)) if shares else None,
        }
    return summary


# --------------------------------------------------------- item 2: symbols per article


def base_code(symbol: str) -> str:
    """Ticker without its exchange suffix. Brazilian ``.SA`` receipts drop their digits too."""
    code, _, exchange = symbol.rpartition(".")
    code = code or symbol
    return code.rstrip("0123456789") if exchange == "SA" else code


def symbol_counts(row: RawNewsRow) -> dict[str, int]:
    """Raw, ``.US`` and distinct-base symbol counts for one article."""
    symbols = list(dict.fromkeys(row.get("symbols", [])))
    return {
        "raw": len(symbols),
        "us": us_symbol_count(row),
        "base": len({base_code(s) for s in symbols}),
    }


def deduped_rows(data: SampleData, ticker: str) -> list[RawNewsRow]:
    """Raw rows behind the current-rule articles for ``ticker`` in 2021-2023, by (date, link)."""
    symbol = f"{ticker}.US"
    articles = parse_news_rows(data.news_rows[ticker], symbol)
    tagged = sorted(
        (r for r in data.news_rows[ticker] if symbol in r.get("symbols", [])),
        key=lambda r: r["date"],
    )
    by_key: dict[tuple[str, str], RawNewsRow] = {}
    for row in tagged:
        by_key.setdefault((row.get("link", ""), row.get("title", "")), row)
    kept = [
        by_key[(a.link, a.title)]
        for a in articles
        if a.published_at.year in YEARS and (a.link, a.title) in by_key
    ]
    return sorted(kept, key=lambda r: (r["date"], r.get("link", "")))


def bucket(count: int) -> str:
    """Histogram bucket for a symbol count."""
    for upper, name in ((1, "1"), (2, "2"), (3, "3"), (5, "4-5"), (10, "6-10"), (20, "11-20")):
        if count <= upper:
            return name
    return ">20"


def symbol_distribution(data: SampleData) -> dict[str, Any]:
    """Symbol-count histograms over unique articles (by link) across the sample, 2021-2023."""
    unique: dict[str, RawNewsRow] = {}
    for ticker in data.tickers:
        for row in deduped_rows(data, ticker):
            unique.setdefault(row.get("link") or row.get("title", ""), row)
    counts = [symbol_counts(row) for row in unique.values()]
    order = ["1", "2", "3", "4-5", "6-10", "11-20", ">20"]
    result: dict[str, Any] = {"articles": len(counts)}
    for kind in ("raw", "us", "base"):
        values = [c[kind] for c in counts]
        histogram = Counter(bucket(v) for v in values)
        result[kind] = {
            "median": statistics.median(values) if values else None,
            "histogram": {b: histogram.get(b, 0) for b in order},
        }
    return result


# ------------------------------------------------------------- item 2: hand labels


def sample_rows(data: SampleData) -> dict[str, list[RawNewsRow]]:
    """20 articles per label ticker, drawn with a fixed seed from the sorted, de-duplicated list."""
    picked: dict[str, list[RawNewsRow]] = {}
    for ticker in LABEL_TICKERS:
        if ticker not in data.tickers:
            continue
        rows = deduped_rows(data, ticker)
        rng = random.Random(f"{SEED}-{ticker}")
        picked[ticker] = rng.sample(rows, min(ARTICLES_PER_TICKER, len(rows)))
    return picked


def dump_sample(sample: dict[str, list[RawNewsRow]], path: Path) -> None:
    """Write the sampled articles with text to ``path`` for labelling. Not an output file."""
    out = [
        {
            "ticker": ticker,
            "link": row.get("link", ""),
            "date": row["date"],
            "title": row.get("title", ""),
            "symbols": row.get("symbols", []),
            "content": row.get("content", "")[:700],
        }
        for ticker, rows in sample.items()
        for row in rows
    ]
    path.write_text(json.dumps(out, indent=1))


def link_key(row: RawNewsRow) -> str:
    """Short stable key for an article: the first 12 hex digits of sha1(link)."""
    return hashlib.sha1(row.get("link", "").encode()).hexdigest()[:12]


def labelled_records(data: SampleData, sample: dict[str, list[RawNewsRow]]) -> list[dict[str, Any]]:
    """One record per sampled article: truncated title, counts, title match and label."""
    records = []
    for ticker, rows in sample.items():
        names = data.names.get(ticker, ())
        for row in rows:
            title = " ".join(html.unescape(row.get("title", "")).split())
            records.append(
                {
                    "ticker": ticker,
                    "date": row["date"][:10],
                    "title": title[:TITLE_CHARS] + ("..." if len(title) > TITLE_CHARS else ""),
                    **symbol_counts(row),
                    "title_match": title_names_company(row.get("title", ""), f"{ticker}.US", names),
                    "label": HAND_LABELS.get(link_key(row), "unlabelled"),
                    "row": row,
                    "names": names,
                }
            )
    return records


def label_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by label, by .US symbol bucket and by title match."""
    by_label = Counter(r["label"] for r in records)
    by_us = {
        label: dict(Counter(bucket(r["us"]) for r in records if r["label"] == label))
        for label in by_label
    }
    by_title = {
        label: sum(1 for r in records if r["label"] == label and r["title_match"])
        for label in by_label
    }
    return {
        "by_label": dict(by_label),
        "us_bucket_by_label": by_us,
        "title_match_by_label": by_title,
    }


def kept_by_label(records: list[dict[str, Any]], rule: CompanyNewsRule) -> dict[str, list[int]]:
    """Per label: [kept, total] of the hand-labelled articles under ``rule``."""
    kept: dict[str, list[int]] = {label: [0, 0] for label in LABELS}
    for record in records:
        symbol = f"{record['ticker']}.US"
        names = record.get("names", ())
        cell = kept.setdefault(record["label"], [0, 0])
        cell[1] += 1
        if row_matches_rule(record["row"], symbol, rule, names):
            cell[0] += 1
    return kept


# -------------------------------------------------------------- item 3: backtests


def base_market(data: SampleData) -> MarketData:
    """Prices, earnings and current-rule news for the sample, membership cut to the sample."""
    spy = market_loader._price_frame("SPY", None, PRICE_START, BACKTEST_END, False)
    prices: dict[str, pd.DataFrame] = {}
    earnings: dict[str, pd.DataFrame] = {}
    for ticker in SAMPLE_TICKERS if data.offline else data.tickers:
        symbol = f"{ticker}.US"
        prices[symbol] = market_loader._price_frame(
            ticker, data.days, PRICE_START, BACKTEST_END, False
        )
        earnings[symbol] = market_loader._earnings_frame(
            ticker, data.days, PRICE_START, BACKTEST_END, False
        )
    news = news_frames(data, CURRENT_NEWS_RULE, prices)
    market = build_market_data(prices, news, earnings, spy, data.history)
    return replace(market, membership=market.membership.reindex(columns=sorted(prices)))


def news_frames(
    data: SampleData, rule: CompanyNewsRule, prices: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Daily news frames under ``rule``. Symbols without news coverage get the unknown frame."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in prices:
        ticker = symbol.removesuffix(".US")
        if ticker not in data.tickers:
            frames[symbol] = market_loader.unknown_news_frame(data.days)
            continue
        daily = client.get_daily_sentiment(
            ticker, MEMBERSHIP_START, BACKTEST_END, offline=data.offline
        )
        articles = parse_news_rows(data.news_rows[ticker], symbol, rule, data.names.get(ticker, ()))
        sentiment = align_daily_sentiment(cast(Any, daily), data.days)
        frames[symbol] = build_news_frame(articles, sentiment, data.days)
    return frames


def with_news_rule(
    data: SampleData, market: MarketData, rule: CompanyNewsRule, symbols: list[str]
) -> MarketData:
    """``market`` with its four article tables rebuilt under ``rule``."""
    frames = news_frames(data, rule, {s: pd.DataFrame() for s in symbols})

    def panel(column: str) -> pd.DataFrame:
        table = pd.DataFrame({s: frames[s][column] for s in sorted(frames)}, index=data.days)
        table.index.name = "date"
        return table.astype("float64")

    return replace(
        market,
        article_count=panel("article_count"),
        mean_polarity=panel("mean_polarity"),
        open_article_count=panel("open_article_count"),
        open_mean_polarity=panel("open_mean_polarity"),
    )


def gap_down_strategies() -> tuple[StrategyDefinition, StrategyDefinition]:
    """The gap-down example, and the same rules without the no_news block."""
    without = dict(GAP_DOWN, name="Gap down", entry=GAP_DOWN["entry"][:1])  # type: ignore[index]
    return strategy_from_json(json.dumps(GAP_DOWN)), strategy_from_json(json.dumps(without))


def trade_count(strategy: StrategyDefinition, market: MarketData) -> int:
    """Trades of ``strategy`` on 2021-2023 at 5 bps per side."""
    window = BacktestWindow("primary", STUDY_START, BACKTEST_END, None)
    return len(run_backtest(strategy, market, window, make_cost_model(COST_BPS)).trades)


def options_table(
    data: SampleData,
    market: MarketData,
    eligible: dict[str, pd.DatetimeIndex],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per candidate rule: zero-news summary, labelled articles kept, gap-down trades."""
    with_filter, _ = gap_down_strategies()
    rows = []
    for name, rule in OPTIONS.items():
        table = zero_share_table(data, rule, eligible)
        ruled = with_news_rule(data, market, rule, list(market.close.columns))
        rows.append(
            {
                "option": name,
                "rule": {
                    "max_us_symbols": rule.max_us_symbols,
                    "require_name_in_title": rule.require_name_in_title,
                    "combine": rule.combine,
                },
                "zero_share_by_year": year_summary(table),
                "labelled_kept": kept_by_label(records, rule),
                "gap_down_trades": trade_count(with_filter, ruled),
            }
        )
    return rows


def run_study() -> dict[str, Any]:
    """Run all three parts and return one JSON-ready report."""
    data = load_sample()
    market = base_market(data)
    closes = {t: market.close.get(f"{t}.US") for t in data.tickers}
    eligible = {t: member_price_days(data, t, closes[t]) for t in data.tickers}
    current = zero_share_table(data, CURRENT_NEWS_RULE, eligible)
    records = labelled_records(data, sample_rows(data)) if data.offline else []
    _, without_filter = gap_down_strategies()
    return {
        "label": SIMULATED_LABEL if data_mode() == "fake" else "LIVE DATA",
        "backtest_label": BACKTEST_LABEL if data.offline else FAKE_BACKTEST_LABEL,
        "members_with_prices_2021_2023": sorted(t for t, days in eligible.items() if len(days)),
        "tickers": data.tickers,
        "excluded_no_news": [t for t in SAMPLE_TICKERS if t not in data.tickers]
        if data.offline
        else [],
        "zero_share_current": current,
        "zero_share_current_summary": year_summary(current),
        "symbol_distribution": symbol_distribution(data),
        "label_note": MODEL_LABEL_NOTE,
        "labelled": [{k: v for k, v in r.items() if k not in ("row", "names")} for r in records],
        "label_summary": label_summary(records),
        "options": options_table(data, market, eligible, records),
        "gap_down_trades_without_no_news": trade_count(without_filter, market),
    }


# ---------------------------------------------------------------- markdown


RULE_LIMITS = [
    "Title test: the membership `Name` with The, Inc, Corp, Co, Company, Ltd, plc, "
    "Class A/B and .com removed, matched as a phrase ignoring case and punctuation, "
    'plus the same words joined ("exxonmobil"), plus the bare ticker as a case '
    "sensitive word (a dash also matches a dot, so BRK.B counts).",
    'It misses short or informal names: "Exxon" for Exxon Mobil, "JPMorgan" for '
    'JPMorgan Chase, "Activision" for Activision Blizzard, "J&J" for Johnson & '
    'Johnson, "P&G", and subsidiaries (LitePoint for Teradyne).',
    'It can match the wrong thing: "PG" in "PG&E", a common word that equals a '
    "short ticker in an all-caps title, and roundups whose title names the company.",
    "MON's membership name is the SPAC that reused the ticker (D27); it has no "
    "2021-2023 membership, so it does not affect these numbers.",
    "Symbol counts use EODHD's tags. A foreign listing with its own code (TEY.F for "
    "Teradyne) cannot be tied to the company from the tag alone, so the base-code "
    "count can still over-count.",
    "The rule filters only per-article data (counts, polarity, open-window columns). "
    "The daily /sentiments value is EODHD's aggregate and is not filtered; no block reads it.",
]


def pct(value: float | int | None) -> str:
    """A share as a percentage with one decimal, or n/a."""
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def zero_share_lines(report: dict[str, Any]) -> list[str]:
    """Item 1: per ticker per year zero-news share, then median and p10."""
    lines = [
        "| Ticker | " + " | ".join(str(y) for y in YEARS) + " |",
        "|---|" + "---|" * len(YEARS),
    ]
    for ticker, years in report["zero_share_current"].items():
        cells = [
            f"{pct(years[y]['zero_share'])} ({years[y]['days']} d)"
            if years[y]["days"]
            else "not a member with prices"
            for y in YEARS
        ]
        lines.append(f"| {ticker} | " + " | ".join(cells) + " |")
    summary = report["zero_share_current_summary"]
    for stat in ("median", "p10"):
        cells = [pct(summary[y][stat]) for y in YEARS]
        lines.append(f"| **{stat}** | " + " | ".join(cells) + " |")
    return lines


def distribution_lines(report: dict[str, Any]) -> list[str]:
    """Item 2: symbol-count histograms over unique articles."""
    dist = report["symbol_distribution"]
    order = list(dist["raw"]["histogram"])
    lines = [
        f"Unique articles (by link) tagged with a sample ticker, 2021-2023: {dist['articles']}.",
        "",
        "| Count | " + " | ".join(order) + " | median |",
        "|---|" + "---|" * (len(order) + 1),
    ]
    names = {"raw": "all symbols", "us": ".US symbols", "base": "distinct base codes"}
    for kind, name in names.items():
        cells = [str(dist[kind]["histogram"][b]) for b in order]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {dist[kind]['median']} |")
    return lines


def label_lines(report: dict[str, Any]) -> list[str]:
    """Item 2: label counts, then all 100 labelled articles."""
    summary = report["label_summary"]
    lines = [f"_{report['label_note']}_", ""]
    for label, count in sorted(summary["by_label"].items()):
        match = summary["title_match_by_label"][label]
        buckets = summary["us_bucket_by_label"][label]
        lines.append(
            f"- {label}: {count} (title names company: {match}; .US symbol buckets: {buckets})"
        )
    lines += [
        "",
        "| Ticker | Date | Title | All | .US | Base | Title match | Label |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["labelled"]:
        title = r["title"].replace("|", "/")
        lines.append(
            f"| {r['ticker']} | {r['date']} | {title} | {r['raw']} | {r['us']} | "
            f"{r['base']} | {'yes' if r['title_match'] else 'no'} | {r['label']} |"
        )
    return lines


def option_lines(report: dict[str, Any]) -> list[str]:
    """Item 3: one row per candidate rule."""
    years = " | ".join(f"{y} median / p10" for y in YEARS)
    kept = " | ".join(f"{label} kept" for label in LABELS)
    lines = [
        f"| Option | {years} | {kept} | Gap-down trades |",
        "|---|" + "---|" * (len(YEARS) + len(LABELS) + 1),
    ]
    for row in report["options"]:
        zero = row["zero_share_by_year"]
        shares = [f"{pct(zero[y]['median'])} / {pct(zero[y]['p10'])}" for y in YEARS]
        labelled = [
            f"{row['labelled_kept'][lab][0]}/{row['labelled_kept'][lab][1]}" for lab in LABELS
        ]
        cells = [row["option"], *shares, *labelled, str(row["gap_down_trades"])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


SOURCE_NOTE = {
    False: "Offline: read from the cache with the network blocked, so no EODHD calls.",
    True: "Fake transport, no network. Hand labels do not apply.",
}


def render_markdown(report: dict[str, Any]) -> str:
    """The whole report as markdown."""
    title = "News definition study (D31)"
    if report["label"] == SIMULATED_LABEL:
        title = f"{SIMULATED_LABEL}: {title}"
    parts = [
        f"# {title}",
        "",
        f"Data: {report['label']}. {SOURCE_NOTE[report['label'] == SIMULATED_LABEL]}",
        f"Tickers with news ({len(report['tickers'])}): {', '.join(report['tickers'])}.",
        f"Excluded, news unknown (D26): {', '.join(report['excluded_no_news']) or 'none'}.",
        "Window: 16:00 New York on the previous trading day to 16:00 on the decision day, "
        "exact .US symbol, de-duplicated as parse_news_rows does. Days counted: trading "
        "days when the ticker was an S&P 500 member and had a price row.",
        "",
        "## 1. Share of decision windows with zero articles (current definition)",
        "",
        *zero_share_lines(report),
        "",
        "## 2. Symbols per article",
        "",
        *distribution_lines(report),
        "",
        "### 100 hand-labelled articles (AAPL, TSLA, JNJ, TER, ETSY; 20 each, fixed seed)",
        "",
        *(label_lines(report) if report["labelled"] else ["Not applicable in fake mode."]),
        "",
        "## 3. Candidate company-news definitions",
        "",
        f"Backtest numbers: {report['backtest_label']}. Of the sample, "
        f"{len(report['members_with_prices_2021_2023'])} names were S&P 500 members with "
        "prices in 2021-2023, so only they can trade. Gap down >= 3%, no news since the "
        "previous close, hold 3, max 10, next_open, 5 bps, 2021-01-01 to 2023-12-31.",
        "",
        *option_lines(report),
        "",
        f"Gap-down trades without the no_news block (reference, same window): "
        f"{report['gap_down_trades_without_no_news']}.",
        "",
        "Zero-news share: median and 10th percentile across tickers of each ticker's "
        "share of zero-article windows. Labelled columns: kept / labelled.",
        "",
        "## Rule limits",
        "",
        *(f"- {line}" for line in RULE_LIMITS),
        "",
        "No option is recommended here. The user decides (D31).",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    """Run the study offline and write the markdown and JSON reports."""
    parser = argparse.ArgumentParser(description="Company-news definition study (D31).")
    parser.add_argument("--dump-sample", type=Path, help="write the 100 sampled articles here")
    args = parser.parse_args()
    if data_mode() == "live":
        block_network()
    if args.dump_sample:
        dump_sample(sample_rows(load_sample()), args.dump_sample)
        return
    report = run_study()
    folder = output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "news_definition_study.json").write_text(json.dumps(report, indent=1, default=str))
    (folder / "news_definition_study.md").write_text(render_markdown(report))
    print(f"Wrote {folder / 'news_definition_study.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
