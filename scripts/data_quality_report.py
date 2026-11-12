"""Write the data-quality report from the cached download (no API calls).

Covers universe symbols with an empty price frame, price gaps inside
membership spans, large unexplained price moves, dividends dropped for falling
outside the price frame, and news coverage per member ticker-year (D4). The
report goes to outputs/simulated/ or outputs/live/ depending on the data mode.
Run scripts/download_data.py first.
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bot.config import BACKTEST_END, MEMBERSHIP_START, data_mode, output_dir, report_title
from bot.data import eodhd_client as client
from bot.data.membership import (
    MembershipHistory,
    MembershipSpell,
    RawMembershipRow,
    build_membership_history,
)
from bot.data.news import coverage_by_year, parse_news_rows
from bot.data.prices import (
    RawDividendRow,
    RawEodRow,
    RawSplitRow,
    build_price_frame,
    dividends_outside_frame,
    infer_splits_from_adjusted_ratio,
    large_unexplained_moves,
    merge_split_sources,
    parse_split_rows,
)
from bot.universe import load_sample, universe_label, universe_mode

PRICE_START = date(2014, 1, 1)
OFFLINE = {"offline": True}
EXAMPLE_LIMIT = 5
MIN_SPLIT_DAYS = 30


def load_history() -> MembershipHistory:
    """Membership history from the cached index file."""
    rows = cast(list[RawMembershipRow], client.get_index_history(offline=True))
    return build_membership_history(rows)


def trading_calendar() -> pd.DatetimeIndex:
    """Trading days are the SPY dates, as the data layer note says."""
    rows = client.get_eod_prices("SPY", PRICE_START, BACKTEST_END, offline=True)
    return pd.DatetimeIndex([pd.Timestamp(row["date"]) for row in rows])


def member_days(
    history: MembershipHistory, symbol: str, calendar: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Trading days inside any membership spell of the symbol, 2015 to 2023."""
    inside = pd.Series(False, index=calendar)
    for spell in history.spells:
        if spell.symbol != symbol:
            continue
        end = spell.end or BACKTEST_END
        inside |= (calendar >= pd.Timestamp(spell.start)) & (calendar <= pd.Timestamp(end))
    inside &= calendar >= pd.Timestamp(MEMBERSHIP_START)
    return calendar[inside.to_numpy()]


def price_checks(
    tickers: list[str], history: MembershipHistory, calendar: pd.DatetimeIndex
) -> tuple[list[str], list[tuple[str, int, list[str]]], list[tuple[str, int, list[str]]], int, int]:
    """Return (empty frames, price gaps, large moves, dividends dropped, dividends seen)."""
    missing: list[str] = []
    gaps: list[tuple[str, int, list[str]]] = []
    flagged: list[tuple[str, int, list[str]]] = []
    dropped = seen = 0
    for ticker in tickers:
        try:
            eod = client.get_eod_prices(ticker, PRICE_START, BACKTEST_END, **OFFLINE)
        except client.OfflineCacheMissError:
            # An empty /eod answer is never cached, so "not cached" after a full download
            # means EODHD had no prices under this code.
            eod = []
        if not eod:
            missing.append(ticker)
            continue
        have = pd.DatetimeIndex([pd.Timestamp(row["date"]) for row in eod])
        absent = member_days(history, f"{ticker}.US", calendar).difference(have)
        if len(absent):
            gaps.append(
                (ticker, len(absent), [d.date().isoformat() for d in absent[:EXAMPLE_LIMIT]])
            )
        dividends = cast(
            list[RawDividendRow], client.get_dividends(ticker, PRICE_START, BACKTEST_END, **OFFLINE)
        )
        dropped += dividends_outside_frame(dividends, have)
        seen += len(dividends)
        bad = split_flags(ticker, eod, dividends)
        if bad:
            flagged.append((ticker, len(bad), [d.isoformat() for d in bad[:EXAMPLE_LIMIT]]))
    return missing, gaps, flagged, dropped, seen


def split_flags(
    ticker: str, eod: list[dict[str, Any]], dividends: list[RawDividendRow]
) -> list[date]:
    """Large one-day moves left after applying reported plus inferred splits."""
    raw_eod = cast(list[RawEodRow], eod)
    reported = parse_split_rows(cast(list[RawSplitRow], client.get_splits(ticker, **OFFLINE)))
    days = pd.DatetimeIndex([pd.Timestamp(row["date"]) for row in raw_eod]).sort_values()
    merged = merge_split_sources(reported, infer_splits_from_adjusted_ratio(raw_eod), days)
    return large_unexplained_moves(build_price_frame(raw_eod, merged, dividends))


def cached_news(ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    """Cached articles for a window, joining the halves the download split (busy tickers).

    The download halves a window the API cannot page in one query, so a busy
    ticker is cached as several files. A missing file means "try the halves".
    """
    try:
        return client.get_news(ticker, start, end, **OFFLINE)
    except client.OfflineCacheMissError:
        if (end - start).days < MIN_SPLIT_DAYS:
            raise
    middle = date.fromordinal((start.toordinal() + end.toordinal()) // 2)
    first = cached_news(ticker, start, middle)
    return first + cached_news(ticker, date.fromordinal(middle.toordinal() + 1), end)


def news_counts(tickers: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    """Daily US-tagged article counts per ticker, plus the tickers with news unavailable.

    A ticker EODHD rejects for news (reused codes such as APC_old) is unknown,
    not zero, so it gets no series and is listed instead (D26).
    """
    counts: dict[str, pd.Series] = {}
    unavailable: list[str] = []
    for ticker in tickers:
        symbol = f"{ticker}.US"
        try:
            rows = cached_news(ticker, MEMBERSHIP_START, BACKTEST_END)
        except client.NewsSymbolUnsupportedError:
            unavailable.append(ticker)
            continue
        articles = parse_news_rows(rows, symbol)  # type: ignore[arg-type]
        days = pd.DatetimeIndex(
            [pd.Timestamp(a.published_at.astimezone(client_ny()).date()) for a in articles]
        )
        counts[symbol] = pd.Series(1, index=days, dtype="int64").groupby(level=0).sum()
    return counts, unavailable


def client_ny() -> Any:
    """New York time zone, imported lazily to keep the import block short."""
    from bot.data.holdout import NEW_YORK

    return NEW_YORK


def coverage_table(counts: dict[str, pd.Series], history: MembershipHistory) -> pd.DataFrame:
    """Per year: member ticker-years, median, p10, p90, share with zero articles."""
    years = range(MEMBERSHIP_START.year, BACKTEST_END.year + 1)
    return coverage_by_year(counts, history, years).reset_index()


def coverage_history(
    history: MembershipHistory, tickers: list[str], unavailable: list[str]
) -> MembershipHistory:
    """Spells of the reported tickers that have news data, for the coverage statistics."""
    keep = {f"{t}.US" for t in tickers} - {f"{t}.US" for t in unavailable}
    spells: tuple[MembershipSpell, ...] = tuple(s for s in history.spells if s.symbol in keep)
    return MembershipHistory(spells)


def coverage_break_lines(coverage: pd.DataFrame) -> list[str]:
    """The 2020 to 2021 coverage break as a finding (D4): table, jump ratio, plain English."""
    by_year = coverage.set_index("year")
    wanted = [y for y in range(2019, 2024) if y in by_year.index]
    lines = [
        "## Coverage break 2020 to 2021",
        "",
        "Articles per member ticker-year, by year:",
        "",
        "| Year | Median | p10 |",
        "|---|---|---|",
    ]
    for year in wanted:
        median, p10 = by_year.loc[year, "median"], by_year.loc[year, "p10"]
        lines.append(f"| {year} | {median:.1f} | {p10:.1f} |")
    before = float(by_year.loc[2020, "median"]) if 2020 in by_year.index else float("nan")
    after = float(by_year.loc[2021, "median"]) if 2021 in by_year.index else float("nan")
    if before == before and after == after:  # neither is NaN
        ratio = f"{after / before:.1f} times" if before > 0 else "not defined (2020 median is 0)"
        jump = f"**Jump in the median, 2021 vs 2020: {ratio}** ({before:.1f} to {after:.1f})."
        lines += ["", jump, ""]
    else:
        lines += ["", "**Jump in the median, 2021 vs 2020: not available (years missing).**", ""]
    if data_mode() == "fake":
        lines += [
            "**SIMULATED DATA.** The fake coverage curve is made up and differs from the "
            "live one. Only the live numbers describe EODHD.",
            "",
        ]
    lines += [
        "Why it matters: the news data does not thin out gradually, it switches on. Before "
        "2021 a typical stock has only a handful of articles a year, so a rule such as "
        "'no news' passes almost every day just because nothing was recorded. That is "
        "missing data, not a quiet stock. So news strategies are tested on 2021 to 2023 "
        "only. 2019 and 2020 are mentioned as a coverage footnote next to the results and "
        "are not shown as results (D4).",
        "",
    ]
    return lines


def build_report(
    tickers: list[str],
    history: MembershipHistory,
    *,
    sample: bool = False,
    sample_label: str = "",
) -> str:
    """Assemble the markdown text. ``sample`` says the tickers are a sample, not the universe."""
    calendar = trading_calendar()
    missing, gaps, flagged, dropped, seen = price_checks(tickers, history, calendar)
    # News is only downloaded for names with prices, so the others are left out here.
    priced = [t for t in tickers if t not in missing]
    counts, unavailable = news_counts(priced)
    coverage = coverage_table(counts, coverage_history(history, priced, unavailable))
    lines = [f"# {report_title('Data quality report')}", ""]
    lines += source_lines()
    if sample:
        lines += [
            f"**SAMPLE REPORT: covers only the {len(tickers)} tickers listed, not the full "
            "universe. Do not read these numbers as S&P 500 coverage.**",
            "",
        ]
        if sample_label:
            lines += [f"**{sample_label}**", ""]
    lines += [
        f"Universe: {len(tickers)} tickers ever in the index {MEMBERSHIP_START} to {BACKTEST_END}, "
        f"plus SPY. Trading days in SPY: {len(calendar)}.",
        "",
        f"## Universe symbols with an empty price frame ({len(missing)})",
        "",
        "These names are silently out of the backtest. Share-class tickers such as BRK.B may "
        "need a different symbol on EODHD (M8).",
        "",
        ", ".join(missing) if missing else "None.",
        "",
        "## Price gaps inside membership spans",
        "",
        "Trading days (SPY dates) inside a member's spell with no price row.",
        "",
    ]
    lines += table_lines(("Ticker", "Missing days", "First examples"), gaps)
    lines += ["", "## Large unexplained moves (over 40% in one day)", ""]
    lines += table_lines(("Ticker", "Flagged moves", "Dates"), flagged)
    lines += [
        "",
        "## Dividends outside the price frame (L6)",
        "",
        f"{dropped} of {seen} dividends fall before the first or after the last price row "
        "and are dropped.",
    ]
    lines += [
        "",
        f"## News unavailable for these symbols ({len(unavailable)})",
        "",
        "EODHD rejects these symbols on the news endpoints (HTTP 422, reused ticker codes). "
        "Their news is unknown, not zero, and no news-based block can fire for them (D26).",
        "",
        ", ".join(unavailable) if unavailable else "None.",
    ]
    lines += ["", "## News coverage per ticker per year (D4)", ""]
    lines += [frame_to_markdown(coverage), ""]
    lines += [
        "Every column counts only tickers that were members at some point in that year. "
        "A member with no articles counts as 0. Symbols with news unavailable are excluded "
        f"from the median, p10, p90 and share-zero ({len(unavailable)} excluded).",
        "",
    ]
    lines += coverage_break_lines(coverage)
    lines += known_limitation_lines(missing_member_days(missing, history, calendar))
    return "\n".join(lines).rstrip("\n") + "\n"


def missing_member_days(
    missing: list[str], history: MembershipHistory, calendar: pd.DatetimeIndex
) -> dict[str, int]:
    """Member trading days (2015 to 2023) for each ticker that has no price frame."""
    return {t: len(member_days(history, f"{t}.US", calendar)) for t in missing}


def known_limitation_lines(missing_days: dict[str, int] | None = None) -> list[str]:
    """Known EODHD data limitations found by hand (D27, D31, missing renamed codes)."""
    lines = [
        "",
        "## Known EODHD limitations",
        "",
        "- **Mis-tagged news (D31).** Some articles about TerrAscend, a cannabis company "
        "listed in Canada under the ticker TER, are tagged `TER.US`, which is Teradyne. "
        "They count as Teradyne news. Limiting news to articles tagged on at most two US "
        "symbols (the company-news rule) does not remove this mis-tag.",
        "- **Reused tickers (D27).** A ticker reused by a later company can point at the "
        "wrong prices. Monsanto (`MON`) has no prices under its code; `MON.US` is a 2021 "
        "SPAC. Members whose prices start late or have long gaps are listed above.",
    ]
    if missing_days:
        lines += missing_price_lines(missing_days)
    return lines


def missing_price_lines(missing_days: dict[str, int]) -> list[str]:
    """Plain-English note on members with no prices under the code membership uses."""
    detail = ", ".join(f"{t} {n}" for t, n in sorted(missing_days.items()))
    return [
        "- **Members with no prices (survivorship gap).** Membership lists a company under "
        "the code it had at the time, and EODHD serves prices only under the current code. "
        "The loader maps the renamed codes ACE, FB, LB, MHFI, PCLN, Q and RE to their "
        "current codes for prices, splits and dividends (D36). Membership and news keep "
        "the old code. Names bought out for cash or stock, such as COV and IILGV, have "
        "none under any code. VSXY has none either. UA-C is class C stock and is not "
        "mapped to UA, which is class A. These stocks are out of the backtest while they "
        "were members.",
        f"  Member trading days with no prices (ticker, days): {detail}. "
        f"Total: {sum(missing_days.values())}.",
        "- **Rows with no dates are dropped (D36).** Millrose Properties (MRP) and Ralliant "
        "(RAL) are 2025 spin-offs. Membership gives them no start date, no end date and "
        "IsActiveNow 0, so they are not members in 2015 to 2023.",
    ]


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """A markdown table without the optional tabulate package."""
    columns = list(frame.columns)
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def source_lines() -> list[str]:
    """The bold data-source header."""
    if data_mode() == "fake":
        return [
            "**Data source: SIMULATED (fake EODHD).** These numbers describe synthetic data "
            "only. They say nothing about real markets or real EODHD coverage.",
            "",
        ]
    return ["**Data source: EODHD (live).**", ""]


def table_lines(header: tuple[str, ...], rows: list[tuple[str, int, list[str]]]) -> list[str]:
    """A small markdown table, or 'None.' when empty."""
    if not rows:
        return ["None."]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for ticker, count, examples in rows:
        out.append(f"| {ticker} | {count} | {', '.join(examples)} |")
    return out


def pick_tickers(universe: list[str], wanted: list[str]) -> list[str]:
    """Keep the wanted codes that were members, case-sensitive (APC_old is not APC_OLD)."""
    missing = sorted(set(wanted) - set(universe))
    if missing:
        print(f"Not members in 2015-2023, skipped: {missing}", file=sys.stderr)
    return [t for t in wanted if t in universe]


def main() -> None:
    """Write the report and print the key numbers."""
    parser = argparse.ArgumentParser(description="Write the data-quality report from the cache.")
    parser.add_argument(
        "--tickers", default="", help="comma-separated sample, case-sensitive, e.g. AAPL,APC_old"
    )
    args = parser.parse_args()
    wanted = [t.strip() for t in args.tickers.split(",") if t.strip()]
    history = load_history()
    universe = history.universe(MEMBERSHIP_START, BACKTEST_END)
    tickers = sorted(symbol.removesuffix(".US") for symbol in universe)
    if wanted:
        tickers = pick_tickers(tickers, wanted)
    label = ""
    if not wanted and universe_mode() == "sample" and data_mode() == "live":
        frozen = set(load_sample().tickers)
        tickers = [t for t in tickers if t in frozen]
        wanted = tickers
        label = universe_label("sample")
    text = build_report(tickers, history, sample=bool(wanted), sample_label=label)
    folder = output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path: Path = folder / "data_quality.md"
    path.write_text(text)
    print(text)
    print(f"Wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
