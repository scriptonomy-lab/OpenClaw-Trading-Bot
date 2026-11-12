"""Historical S&P 500 membership, built from GSPC.INDX HistoricalTickerComponents.

Interface stubs for Phase 1. See docs/design/data_layer.md for the reasoning.

Holdout rules applied when the raw rows are parsed:

- A row with StartDate on or after HOLDOUT_START is dropped. A name added in
  2024 or later is not part of the backtest universe.
- An EndDate on or after HOLDOUT_START becomes None. Seen from 2023, the name
  is still a member, so a removal in 2025 is not visible in 2023.
- IsActiveNow and IsDelisted describe today, not the past. They are kept
  apart in DownloadHint, which may decide what to download but is never
  passed to a building block.

Date convention (provisional, see D2): StartDate and EndDate are both
inclusive. This is checked against the ``historical=1`` snapshots on known
index change dates once the real key works.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TypedDict

import numpy as np
import pandas as pd

from bot.config import HOLDOUT_START
from bot.data.aliases import ALIAS_LAST_DAY, PRICE_ALIASES
from bot.data.holdout import _is_unsealed, assert_before_holdout

logger = logging.getLogger(__name__)

EXCHANGE_SUFFIX = ".US"

# Stand-in start for rows that joined before EODHD's change history (D25).
HISTORY_BEGINS = date(1900, 1, 1)


class RawMembershipRow(TypedDict):
    """One row of HistoricalTickerComponents as EODHD documents it."""

    Code: str
    Name: str
    StartDate: str | None
    EndDate: str | None
    IsActiveNow: int
    IsDelisted: int


@dataclass(frozen=True)
class MembershipSpell:
    """One continuous period in the index. ``end`` None means still a member at the cutoff.

    A name that left and rejoined has two spells.
    """

    symbol: str
    start: date
    end: date | None


@dataclass(frozen=True)
class DownloadHint:
    """Today's state of a ticker, for choosing what to download only."""

    symbol: str
    is_active_now: bool
    is_delisted: bool


def _parse_optional_date(value: str | None) -> date | None:
    """Parse ``YYYY-MM-DD``. None and empty strings give None."""
    if value is None or value == "":
        return None
    return date.fromisoformat(value)


def to_symbol(code: str) -> str:
    """Map a bare index Code such as "AAPL" to the EOD symbol "AAPL.US"."""
    return f"{code}{EXCHANGE_SUFFIX}"


# Active members whose StartDate is empty but who joined the index after the
# data begins (D36). D25 would call them members since 1900, which lets the
# backtest hold them before they were in the index. Dates are the effective
# dates in S&P Dow Jones Indices press releases.
MANUAL_START_DATES: dict[str, date] = {
    "IR": date(
        2020, 3, 3
    ),  # Gardner Denver Holdings joined, replacing Cimarex. Renamed Ingersoll Rand.
    "WAB": date(2019, 2, 27),  # Wabtec joined, replacing Goodyear.
}


def _fallback_start(row: RawMembershipRow) -> date:
    """Start for a row with no StartDate: a manual join date, else the D25 default.

    The manual date applies only to an active row with no dates at all, so an
    older row for the same code cannot be mis-dated if the file is refreshed.
    """
    no_dates = not row["EndDate"] and bool(row["IsActiveNow"])
    return MANUAL_START_DATES.get(row["Code"], HISTORY_BEGINS) if no_dates else HISTORY_BEGINS


def _has_no_dates_and_inactive(row: RawMembershipRow) -> bool:
    """True for a row with no StartDate, no EndDate and IsActiveNow 0 (D36).

    A genuine early member (D25) has no StartDate but is still in the index,
    so it is active now. A row with both dates empty that is not
    active now has nothing to date it. These are the 2025 spin-offs MRP and
    RAL, which D25 would make members on every day from 2015 to 2023. The
    flags are used only for this check and for the manual dates below.
    """
    return not row["StartDate"] and not row["EndDate"] and not row["IsActiveNow"]


def parse_membership_rows(
    rows: list[RawMembershipRow], *, approved: bool = False
) -> list[MembershipSpell]:
    """Turn raw rows into spells, applying the holdout rules in the module docstring.

    With ``approved=True`` and an open holdout or paper session, the holdout
    rules are skipped and the full history is kept (D23).

    A row with no StartDate joined before EODHD's change history begins, so
    it counts as a member from ``HISTORY_BEGINS`` (D25). Rows with no dates at
    all are dropped by ``_has_no_dates_and_inactive``. Spells of renamed codes
    lose the days their price code already covers (``drop_alias_overlaps``).
    Raises ValueError for a row whose EndDate is before its StartDate.
    """
    unsealed = _is_unsealed(approved)
    spells: list[MembershipSpell] = []
    for row in rows:
        if _has_no_dates_and_inactive(row):
            continue
        start = _parse_optional_date(row["StartDate"]) or _fallback_start(row)
        if start >= HOLDOUT_START and not unsealed:
            continue
        end = _parse_optional_date(row["EndDate"])
        if end is not None and end < start:
            raise ValueError(f"Membership row for {row['Code']} ends before it starts.")
        # A removal in the holdout must stay invisible, so the name is still a member.
        if end is not None and end >= HOLDOUT_START and not unsealed:
            end = None
        spells.append(MembershipSpell(symbol=to_symbol(row["Code"]), start=start, end=end))
    return drop_alias_overlaps(spells)


def drop_alias_overlaps(spells: list[MembershipSpell]) -> list[MembershipSpell]:
    """Remove days from an old code's spells that its price code already covers.

    FB and META share one price series. If both are members on the same day
    (the swap day, or all of ACE's years because CB also has an old Chubb
    spell), the company would be held twice. The alias code loses those days
    and the price code keeps them. A spell that is fully covered disappears.
    A spell of an old code that starts after its rename date is a reused code,
    a different company. It is dropped with a warning, because the price code
    would give it the wrong company's prices (``_is_reused_code``).
    """
    spells = [spell for spell in spells if not _is_reused_code(spell)]
    result = [s for s in spells if s.symbol.removesuffix(".US") not in PRICE_ALIASES]
    for spell in spells:
        code = spell.symbol.removesuffix(".US")
        if code in PRICE_ALIASES:
            target = to_symbol(PRICE_ALIASES[code])
            covering = [s for s in spells if s.symbol == target]
            result.extend(_subtract(spell, covering))
    return sorted(result, key=lambda spell: (spell.symbol, spell.start))


def _is_reused_code(spell: MembershipSpell) -> bool:
    """True when an alias code starts a spell after its rename, so it is another company."""
    code = spell.symbol.removesuffix(EXCHANGE_SUFFIX)
    last_day = ALIAS_LAST_DAY.get(code)
    if last_day is None or spell.start <= last_day:
        return False
    logger.warning(
        "%s starts %s, after its rename on %s. Dropped: the code looks reused by "
        "another company and would get the wrong prices (D36).",
        spell.symbol,
        spell.start,
        last_day,
    )
    return True


def _subtract(spell: MembershipSpell, covering: list[MembershipSpell]) -> list[MembershipSpell]:
    """The pieces of ``spell`` left after removing every covering spell (inclusive dates)."""
    pieces = [spell]
    for cover in covering:
        next_pieces: list[MembershipSpell] = []
        for piece in pieces:
            next_pieces.extend(_cut(piece, cover))
        pieces = next_pieces
    return pieces


def _cut(piece: MembershipSpell, cover: MembershipSpell) -> list[MembershipSpell]:
    """Split one spell around one covering spell."""
    cover_end = cover.end
    if cover_end is not None and cover_end < piece.start:
        return [piece]
    if piece.end is not None and cover.start > piece.end:
        return [piece]
    out: list[MembershipSpell] = []
    if cover.start > piece.start:
        out.append(MembershipSpell(piece.symbol, piece.start, cover.start - timedelta(days=1)))
    if cover_end is not None and (piece.end is None or cover_end < piece.end):
        out.append(MembershipSpell(piece.symbol, cover_end + timedelta(days=1), piece.end))
    return out


def download_hints(rows: list[RawMembershipRow]) -> list[DownloadHint]:
    """Today's active and delisted flags per symbol, for the downloader only."""
    return [
        DownloadHint(
            symbol=to_symbol(row["Code"]),
            is_active_now=bool(row["IsActiveNow"]),
            is_delisted=bool(row["IsDelisted"]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class MembershipHistory:
    """All membership spells up to the holdout cutoff."""

    spells: tuple[MembershipSpell, ...]

    def members_on(self, day: date, *, approved: bool = False) -> frozenset[str]:
        """Symbols in the index on ``day``. Raises HoldoutAccessError for a holdout day.

        ``approved`` is passed to the guard (D12).
        """
        assert_before_holdout(day, approved=approved)
        return frozenset(
            spell.symbol
            for spell in self.spells
            if spell.start <= day and (spell.end is None or day <= spell.end)
        )

    def universe(self, start: date, end: date, *, approved: bool = False) -> frozenset[str]:
        """Every symbol that was a member on at least one day from ``start`` to ``end``.

        With start MEMBERSHIP_START and end BACKTEST_END this is the full list
        of tickers whose prices, news and earnings we download.
        """
        assert_before_holdout(end, approved=approved)
        return frozenset(
            spell.symbol
            for spell in self.spells
            if spell.start <= end and (spell.end is None or spell.end >= start)
        )


def build_membership_history(
    rows: list[RawMembershipRow], *, approved: bool = False
) -> MembershipHistory:
    """Parse raw rows into a MembershipHistory. See ``parse_membership_rows`` for ``approved``."""
    return MembershipHistory(spells=tuple(parse_membership_rows(rows, approved=approved)))


def membership_matrix(
    history: MembershipHistory, days: pd.DatetimeIndex, *, approved: bool = False
) -> pd.DataFrame:
    """A date x symbol table of bool: True when the symbol is a member that day.

    ``days`` is the trading calendar (the SPY price index). Used by the
    backtester to limit candidates to members on the decision day.
    """
    if len(days) > 0:
        assert_before_holdout(days[-1].date(), approved=approved)
    symbols = sorted({spell.symbol for spell in history.spells})
    columns = {symbol: np.zeros(len(days), dtype=bool) for symbol in symbols}
    for spell in history.spells:
        inside = days >= pd.Timestamp(spell.start)
        if spell.end is not None:
            inside = inside & (days <= pd.Timestamp(spell.end))
        columns[spell.symbol] |= np.asarray(inside)
    return pd.DataFrame(columns, index=days, dtype=bool)
