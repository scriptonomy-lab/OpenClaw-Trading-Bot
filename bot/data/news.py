"""News counts and sentiment per symbol, aligned to decision times.

Interface stubs for Phase 1. See docs/design/data_layer.md for the reasoning.

Decision time for day D is 16:00 America/New_York on D. The news window for
a decision on D is half-open:

    [16:00 New York on the previous trading day, 16:00 New York on D)

Article timestamps are converted to New York time with zoneinfo, so daylight
saving time is handled by the time zone rules, not by a fixed offset. Trading
days come from the SPY price index. A Monday window starts at Friday 16:00.

Only an article whose ``symbols`` list contains the exact ``.US`` symbol
counts for that symbol. Articles are de-duplicated per symbol by ``link`` and
by (normalised title, timestamp), so the same story syndicated on two
domains counts once. The copy kept is the one with the earliest timestamp,
then the one seen first.

An optional ``CompanyNewsRule`` (D31, pending the user's choice) can keep only
"company news": articles tagged on few ``.US`` symbols, or whose title names
the company. It is applied in ``parse_news_rows`` before de-duplication, so
every column built from the articles (counts, polarity, open-window columns)
uses one article set, and the no-news and sentiment blocks agree. The default
rule keeps every article, which is the behaviour before D31. The daily
``/sentiments`` value is EODHD's own aggregate, so no rule can filter it.

Raw text never leaves this module's storage path (D6). What the agent and the
building blocks see is NewsWindowSummary or the daily news frame: counts,
scores and dates only.

Daily news frame, one per symbol:

- Index: ``pd.DatetimeIndex`` named ``date`` (decision dates, tz-naive).
- ``article_count`` (float64): articles in the window ending at that
  decision. NaN on the first day: it has no earlier close, so the count is
  unknown, not zero. "No news" logic must test ``article_count == 0``, which is
  False for NaN, or use ``no_news_days``. Never fill NaN with 0.
- ``mean_polarity`` (float64): mean per-article ``sentiment.polarity``, NaN
  when there are no articles.
- ``daily_sentiment`` (float64): the ``/sentiments`` ``normalized`` value,
  count-weighted over the days it covers and lagged (see
  ``align_daily_sentiment``). NaN when missing.
- ``open_article_count`` and ``open_mean_polarity`` (float64): the same two
  numbers for the shorter open window of D, used by strategies that buy at
  D's open (``entry_timing`` "same_open"). That window is half-open:

      [16:00 New York on the previous trading day, 09:30 New York on D)

  It is the first part of D's close window. An article at exactly 09:30 is
  not in it. NaN on the first day, like the close window.
"""

import html
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bot.config import HOLDOUT_START
from bot.data.membership import MembershipHistory

NEW_YORK = ZoneInfo("America/New_York")
DECISION_TIME = time(16, 0)
# The regular session opens at 09:30 New York time. Same-open entries decide here.
OPEN_TIME = time(9, 30)

NEWS_FRAME_COLUMNS: tuple[str, ...] = (
    "article_count",
    "mean_polarity",
    "daily_sentiment",
    "open_article_count",
    "open_mean_polarity",
)


class RawSentiment(TypedDict):
    """The per-article ``sentiment`` object."""

    polarity: float
    neg: float
    neu: float
    pos: float


class RawNewsRow(TypedDict, total=False):
    """One row of ``GET /api/news``. Title and content are untrusted text."""

    date: str
    title: str
    content: str
    link: str
    symbols: list[str]
    tags: list[str]
    sentiment: RawSentiment


class RawDailySentimentRow(TypedDict):
    """One row of ``GET /api/sentiments``. ``date`` has no time and no zone."""

    date: str
    count: int
    normalized: float


@dataclass(frozen=True)
class NewsArticle:
    """One stored article. Only this module and storage see ``title`` and ``link``."""

    symbol: str
    published_at: datetime
    polarity: float | None
    title: str
    link: str


@dataclass(frozen=True)
class NewsWindowSummary:
    """What the agent may see about news for one symbol and one decision."""

    symbol: str
    decision_date: date
    window_start: datetime
    window_end: datetime
    article_count: int
    mean_polarity: float | None


# Words dropped from a company name before matching it in a title.
NAME_SUFFIX_WORDS = frozenset(
    {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "com"}
)


@dataclass(frozen=True)
class CompanyNewsRule:
    """Which tagged articles count as news about the company (D31, pending).

    ``max_us_symbols``: keep an article only if it is tagged with at most this
    many distinct ``.US`` symbols. None means no limit.
    ``require_name_in_title``: keep an article only if its title contains the
    company name or the bare ticker (see ``title_names_company``).
    ``combine``: with both tests on, "all" needs both to pass and "any" needs
    one. The default rule has no test on, so it keeps every article.
    """

    max_us_symbols: int | None = None
    require_name_in_title: bool = False
    combine: Literal["all", "any"] = "all"

    def is_default(self) -> bool:
        """True when the rule keeps every article (the behaviour before D31)."""
        return self.max_us_symbols is None and not self.require_name_in_title


CURRENT_NEWS_RULE = CompanyNewsRule()

# The company-news definition the user locked (D31) before any returns were
# seen: an article counts for a stock only if it is tagged on at most two US
# symbols. Both news blocks use it. The 1- and 3-symbol rules are robustness
# rows only.
COMPANY_NEWS_RULE = CompanyNewsRule(max_us_symbols=2)
ROBUSTNESS_NEWS_RULES = {
    "us_symbols_at_most_1": CompanyNewsRule(max_us_symbols=1),
    "us_symbols_at_most_3": CompanyNewsRule(max_us_symbols=3),
}


def us_symbol_count(row: RawNewsRow) -> int:
    """Distinct ``.US`` symbols an article is tagged with."""
    return len({s for s in row.get("symbols", []) if s.endswith(".US")})


def company_name_phrases(name: str) -> tuple[str, ...]:
    """Normalised phrases for a company name, for title matching.

    Drops a leading "The", "Class X", and suffixes such as Inc, Corp, Co,
    Company, Ltd, plc and ".com". Punctuation becomes a space, so
    "The Coca-Cola Company" gives "coca cola". The joined form without
    spaces ("cocacola", "exxonmobil") is added for names of two or more words.
    """
    words = _normalise_title(name).split()
    if words[:1] == ["the"]:
        words = words[1:]
    kept: list[str] = []
    skip_next = False
    for word in words:
        if skip_next:
            skip_next = False
        elif word == "class":
            skip_next = True
        elif word not in NAME_SUFFIX_WORDS:
            kept.append(word)
    if not kept:
        return ()
    phrases = [" ".join(kept)]
    if len(kept) > 1:
        phrases.append("".join(kept))
    return tuple(phrases)


def title_names_company(title: str, symbol: str, names: Sequence[str] = ()) -> bool:
    """True if the title contains a company name phrase or the bare ticker as a word.

    The name test ignores case and punctuation. The ticker test is case
    sensitive on the raw title (HTML entities decoded), so "KO" does not match
    "ko". A dash in the ticker also matches a dot, slash or space ("BRK.B").
    """
    text = html.unescape(title)
    padded = f" {_normalise_title(text)} "
    for name in names:
        if any(f" {phrase} " in padded for phrase in company_name_phrases(name)):
            return True
    ticker = symbol.removesuffix(".US")
    pattern = r"[.\-/ ]".join(re.escape(part) for part in ticker.split("-"))
    return re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text) is not None


def row_matches_rule(
    row: RawNewsRow, symbol: str, rule: CompanyNewsRule, names: Sequence[str] = ()
) -> bool:
    """True if ``rule`` keeps this row for ``symbol``. Does not check the symbol tag itself."""
    tests: list[bool] = []
    if rule.max_us_symbols is not None:
        tests.append(us_symbol_count(row) <= rule.max_us_symbols)
    if rule.require_name_in_title:
        tests.append(title_names_company(row.get("title", ""), symbol, names))
    if not tests:
        return True
    return all(tests) if rule.combine == "all" else any(tests)


def decision_datetime(day: date) -> datetime:
    """16:00 America/New_York on ``day``, as a time zone aware datetime."""
    return datetime.combine(day, DECISION_TIME, tzinfo=NEW_YORK)


def open_datetime(day: date) -> datetime:
    """09:30 America/New_York on ``day``, as a time zone aware datetime. DST comes from zoneinfo."""
    return datetime.combine(day, OPEN_TIME, tzinfo=NEW_YORK)


def open_news_window(
    decision_date: date, trading_days: pd.DatetimeIndex
) -> tuple[datetime, datetime]:
    """The half-open window [previous close, this open) for a same-open decision.

    Raises ValueError like ``news_window``.
    """
    start, _ = news_window(decision_date, trading_days)
    return start, open_datetime(decision_date)


def news_window(decision_date: date, trading_days: pd.DatetimeIndex) -> tuple[datetime, datetime]:
    """The half-open window [previous close, this close) for a decision on ``decision_date``.

    Raises ValueError if ``decision_date`` is not in ``trading_days`` or is the first one.
    """
    stamp = pd.Timestamp(decision_date)
    position = int(trading_days.searchsorted(stamp, side="left"))
    if position >= len(trading_days) or trading_days[position] != stamp:
        raise ValueError(f"{decision_date} is not a trading day.")
    if position == 0:
        raise ValueError(f"{decision_date} is the first trading day, so it has no earlier close.")
    previous_day = trading_days[position - 1].date()
    return decision_datetime(previous_day), decision_datetime(decision_date)


def parse_news_rows(
    rows: list[RawNewsRow],
    symbol: str,
    rule: CompanyNewsRule = CURRENT_NEWS_RULE,
    names: Sequence[str] = (),
) -> list[NewsArticle]:
    """Keep rows tagged with exactly ``symbol`` (a ``.US`` symbol), de-duplicated.

    ``rule`` (default: keep all) drops rows that are not company news before
    de-duplication, so one copy of a story that passes is enough. ``names``
    are the company's names for the title test.

    A row is a duplicate if its ``link`` was already kept, or if its
    (normalised title, timestamp) was. Of duplicates the earliest timestamp
    wins, then the row seen first. The result is sorted by timestamp. Rows
    without a link or without a title cannot be matched on that field and are
    only compared on the other one.

    Raises ValueError for a timestamp without a time zone, because its New York
    time would be a guess.
    """
    candidates = [
        (_article_from_row(row, symbol, row.get("link", "")), order)
        for order, row in enumerate(rows)
        if symbol in row.get("symbols", []) and row_matches_rule(row, symbol, rule, names)
    ]
    candidates.sort(key=lambda pair: (pair[0].published_at, pair[1]))
    articles: list[NewsArticle] = []
    seen_links: set[str] = set()
    seen_titles: set[tuple[str, datetime]] = set()
    for article, _ in candidates:
        title_key = (_normalise_title(article.title), article.published_at)
        if article.link and article.link in seen_links:
            continue
        if title_key[0] and title_key in seen_titles:
            continue
        seen_links.add(article.link)
        seen_titles.add(title_key)
        articles.append(article)
    return articles


def _normalise_title(title: str) -> str:
    """Lower case, keep letters and digits, single spaces. Empty if nothing is left."""
    return " ".join("".join(c if c.isalnum() else " " for c in title.casefold()).split())


def summarize_window(
    articles: list[NewsArticle], decision_date: date, trading_days: pd.DatetimeIndex
) -> NewsWindowSummary:
    """Count articles and average polarity in the window for ``decision_date``."""
    start, end = news_window(decision_date, trading_days)
    inside = [article for article in articles if start <= article.published_at < end]
    polarities = [a.polarity for a in inside if a.polarity is not None]
    mean_polarity = sum(polarities) / len(polarities) if polarities else None
    # The signature has no symbol, so it comes from the articles (one symbol per call).
    symbol = articles[0].symbol if articles else ""
    return NewsWindowSummary(
        symbol=symbol,
        decision_date=decision_date,
        window_start=start,
        window_end=end,
        article_count=len(inside),
        mean_polarity=mean_polarity,
    )


def align_daily_sentiment(
    rows: list[RawDailySentimentRow], trading_days: pd.DatetimeIndex
) -> pd.Series:
    """Map ``/sentiments`` values to the decisions allowed to use them.

    The endpoint's date has no time, so the value may include articles up to
    the end of that day. A value dated X is first used at the close of the
    first trading day strictly after X. When several values map to one
    decision day (a Monday takes Friday, Saturday and Sunday), the result is
    their mean weighted by each row's ``count``, so a well-sampled Friday is
    not dropped for a thin Sunday. If all counts are 0 the plain mean is used.
    Returns a float64 Series on ``trading_days``, NaN where there is no value.
    """
    groups: dict[int, list[RawDailySentimentRow]] = {}
    for row in rows:
        position = int(trading_days.searchsorted(pd.Timestamp(row["date"]), side="right"))
        if position < len(trading_days):
            groups.setdefault(position, []).append(row)
    aligned = pd.Series(float("nan"), index=trading_days, dtype="float64")
    for position, members in groups.items():
        aligned.iloc[position] = _weighted_sentiment(members)
    return aligned


def _weighted_sentiment(members: list[RawDailySentimentRow]) -> float:
    """Count-weighted mean of ``normalized``. Plain mean when the counts sum to 0."""
    total = sum(float(m["count"]) for m in members)
    if total <= 0:
        return sum(float(m["normalized"]) for m in members) / len(members)
    return sum(float(m["count"]) * float(m["normalized"]) for m in members) / total


def build_news_frame(
    articles: list[NewsArticle],
    daily_sentiment: pd.Series,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the daily news frame (see module docstring) for one symbol."""
    counts, means = _window_stats(articles, trading_days, DECISION_TIME)
    open_counts, open_means = _window_stats(articles, trading_days, OPEN_TIME)
    frame = pd.DataFrame(
        {
            "article_count": counts,
            "mean_polarity": means,
            "daily_sentiment": daily_sentiment.reindex(trading_days).astype("float64").to_numpy(),
            "open_article_count": open_counts,
            "open_mean_polarity": open_means,
        },
        index=trading_days,
    )
    frame.index.name = "date"
    return frame


def no_news_days(frame: pd.DataFrame) -> pd.Series:
    """True on days known to have no articles. NaN (unknown) counts as False."""
    return frame["article_count"] == 0


def coverage_by_year(
    article_counts: dict[str, pd.Series],
    history: MembershipHistory,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Articles per ticker-year, summarised per year (D4).

    A ticker-year counts only if the ticker was a member at some point that
    year (from ``history``), so years before a name joined do not pull the
    median down. A member with no entry in ``article_counts`` counts as 0
    articles. ``article_counts`` maps a ``.US`` symbol to a daily count series.
    ``years`` defaults to the years from the first spell to the holdout year.

    Columns: ``member_tickers``, ``median``, ``p10``, ``p90``, ``share_zero``,
    all across member ticker-years only. Index is ``year``.
    """
    if years is None:
        first = min(spell.start.year for spell in history.spells)
        years = range(first, HOLDOUT_START.year)
    yearly = {
        symbol: counts.groupby(pd.DatetimeIndex(counts.index).year).sum()
        for symbol, counts in article_counts.items()
    }
    rows = []
    for year in years:
        members = sorted(history.universe(date(year, 1, 1), date(year, 12, 31)))
        values = pd.Series(
            [float(yearly[s].get(year, 0)) if s in yearly else 0.0 for s in members],
            dtype="float64",
        )
        rows.append(
            {
                "year": year,
                "member_tickers": len(values),
                "median": float(values.median()) if len(values) else float("nan"),
                "p10": float(values.quantile(0.1)) if len(values) else float("nan"),
                "p90": float(values.quantile(0.9)) if len(values) else float("nan"),
                "share_zero": float((values == 0).mean()) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _article_from_row(row: RawNewsRow, symbol: str, link: str) -> NewsArticle:
    """Build one NewsArticle, converting the timestamp to New York time."""
    moment = datetime.fromisoformat(row["date"])
    if moment.tzinfo is None:
        raise ValueError(f"News timestamp has no time zone: {row['date']!r}.")
    sentiment = row.get("sentiment")
    polarity = float(sentiment["polarity"]) if sentiment else None
    return NewsArticle(
        symbol=symbol,
        published_at=moment.astimezone(NEW_YORK),
        polarity=polarity,
        title=row.get("title", ""),
        link=link,
    )


def _window_stats(
    articles: list[NewsArticle], trading_days: pd.DatetimeIndex, end_time: time
) -> tuple[np.ndarray, np.ndarray]:
    """Article count and mean polarity for each decision day.

    Day D's window runs from 16:00 New York on the previous trading day up to,
    but not including, ``end_time`` New York on D: 16:00 for the normal window,
    09:30 for the open window. The first day has no earlier close, so both are
    NaN (unknown, not zero).
    """
    ordered = sorted(articles, key=lambda article: article.published_at)
    stamps = np.array([a.published_at.timestamp() for a in ordered], dtype="float64")
    polarity = np.array([0.0 if a.polarity is None else a.polarity for a in ordered])
    has_polarity = np.array([0 if a.polarity is None else 1 for a in ordered])
    sum_before = np.concatenate([[0.0], np.cumsum(polarity)])
    scored_before = np.concatenate([[0], np.cumsum(has_polarity)])
    closes = np.array([decision_datetime(d.date()).timestamp() for d in trading_days])
    window_ends = np.array(
        [datetime.combine(d.date(), end_time, tzinfo=NEW_YORK).timestamp() for d in trading_days]
    )
    # Half-open windows: an article stamped exactly at a cut belongs to the next window.
    starts = np.searchsorted(stamps, closes, side="left")
    ends = np.searchsorted(stamps, window_ends, side="left")
    counts = np.full(len(trading_days), np.nan)
    means = np.full(len(trading_days), np.nan)
    for i in range(1, len(trading_days)):
        counts[i] = ends[i] - starts[i - 1]
        scored = scored_before[ends[i]] - scored_before[starts[i - 1]]
        if scored > 0:
            means[i] = (sum_before[ends[i]] - sum_before[starts[i - 1]]) / scored
    return counts, means
