"""Point-in-time membership tests on synthetic rows."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from bot.data.holdout import HoldoutAccessError
from bot.data.membership import (
    MembershipSpell,
    build_membership_history,
    download_hints,
    membership_matrix,
    parse_membership_rows,
    to_symbol,
)


def row(
    code: str,
    start: str | None,
    end: str | None,
    active: int = 0,
    delisted: int = 0,
) -> Any:
    return {
        "Code": code,
        "Name": code,
        "StartDate": start,
        "EndDate": end,
        "IsActiveNow": active,
        "IsDelisted": delisted,
    }


ROWS = [
    row("OLD", "2010-01-01", "2018-06-15", active=0, delisted=1),
    row("KEEP", "2012-03-01", None, active=1),
    row("LATE", "2017-05-01", "2025-03-01", active=0, delisted=1),
    row("NEW24", "2024-02-01", None, active=1),
]


def test_to_symbol() -> None:
    assert to_symbol("AAPL") == "AAPL.US"


def test_removed_stock_is_member_before_end_and_not_after() -> None:
    history = build_membership_history(ROWS)
    assert "OLD.US" in history.members_on(date(2016, 1, 4))
    assert "OLD.US" not in history.members_on(date(2019, 1, 2))


def test_start_and_end_dates_are_inclusive() -> None:
    history = build_membership_history(ROWS)
    assert "OLD.US" not in history.members_on(date(2009, 12, 31))
    assert "OLD.US" in history.members_on(date(2010, 1, 1))
    assert "OLD.US" in history.members_on(date(2018, 6, 15))
    assert "OLD.US" not in history.members_on(date(2018, 6, 16))


def test_end_date_in_holdout_means_still_member() -> None:
    spells = parse_membership_rows(ROWS)
    late = [s for s in spells if s.symbol == "LATE.US"]
    assert late == [MembershipSpell("LATE.US", date(2017, 5, 1), None)]
    history = build_membership_history(ROWS)
    assert "LATE.US" in history.members_on(date(2023, 12, 29))
    assert "LATE.US" not in history.members_on(date(2017, 4, 28))


def test_start_date_in_holdout_is_dropped() -> None:
    spells = parse_membership_rows(ROWS)
    assert all(s.symbol != "NEW24.US" for s in spells)
    history = build_membership_history(ROWS)
    assert "NEW24.US" not in history.universe(date(2015, 1, 1), date(2023, 12, 31))


def test_active_and_delisted_flags_do_not_change_membership() -> None:
    flipped = [
        row("A", "2016-01-01", None, active=0, delisted=1),
        row("B", "2016-01-01", None, active=1, delisted=0),
    ]
    history = build_membership_history(flipped)
    assert history.members_on(date(2020, 1, 2)) == frozenset({"A.US", "B.US"})
    spells = parse_membership_rows(flipped)
    assert not any(hasattr(s, "is_delisted") or hasattr(s, "is_active_now") for s in spells)


def test_download_hints_carry_todays_flags() -> None:
    hints = {h.symbol: h for h in download_hints(ROWS)}
    assert hints["OLD.US"].is_delisted is True
    assert hints["OLD.US"].is_active_now is False
    assert hints["KEEP.US"].is_active_now is True
    assert hints["KEEP.US"].is_delisted is False


def test_members_on_holdout_day_raises() -> None:
    history = build_membership_history(ROWS)
    with pytest.raises(HoldoutAccessError):
        history.members_on(date(2024, 1, 1))


def test_rejoining_name_has_two_spells() -> None:
    rows = [row("RE", "2012-01-01", "2014-12-31"), row("RE", "2018-01-01", None)]
    history = build_membership_history(rows)
    assert len([s for s in history.spells if s.symbol == "RE.US"]) == 2
    assert "RE.US" in history.members_on(date(2013, 6, 3))
    assert "RE.US" not in history.members_on(date(2016, 6, 3))
    assert "RE.US" in history.members_on(date(2019, 6, 3))


def test_bad_rows_raise() -> None:
    # A missing StartDate is not bad data: live EODHD uses it for names that
    # joined before its history begins (D25). See the early-member tests below.
    # Active now, so it is a genuine early member. With both dates empty and
    # not active, the row is dropped instead (D36, tests/test_price_aliases.py).
    assert parse_membership_rows([row("X", None, None, active=1)])[0].symbol == "X.US"
    with pytest.raises(ValueError):
        parse_membership_rows([row("X", "2018-01-02", "2018-01-01")])


def test_universe_covers_members_across_range() -> None:
    history = build_membership_history(ROWS)
    # OLD left 2018-06-15, so it is in a 2018 range and not in a 2019 range.
    assert history.universe(date(2018, 1, 1), date(2018, 12, 31)) == frozenset(
        {"OLD.US", "KEEP.US", "LATE.US"}
    )
    assert history.universe(date(2019, 1, 1), date(2019, 12, 31)) == frozenset(
        {"KEEP.US", "LATE.US"}
    )
    # LATE started 2017-05-01, so it is not in 2016.
    assert history.universe(date(2016, 1, 1), date(2016, 12, 31)) == frozenset(
        {"OLD.US", "KEEP.US"}
    )
    # Range endpoints are inclusive.
    assert "OLD.US" in history.universe(date(2018, 6, 15), date(2018, 6, 20))
    assert "OLD.US" not in history.universe(date(2018, 6, 16), date(2018, 6, 20))
    assert "LATE.US" in history.universe(date(2016, 12, 1), date(2017, 5, 1))
    assert "LATE.US" not in history.universe(date(2016, 12, 1), date(2017, 4, 30))


def test_membership_matrix() -> None:
    history = build_membership_history(ROWS)
    days = pd.DatetimeIndex(["2018-06-14", "2018-06-15", "2018-06-18"])
    matrix = membership_matrix(history, days)
    assert matrix.loc[pd.Timestamp("2018-06-15"), "OLD.US"]
    assert not matrix.loc[pd.Timestamp("2018-06-18"), "OLD.US"]
    assert matrix.loc[pd.Timestamp("2018-06-18"), "KEEP.US"]
    assert "NEW24.US" not in matrix.columns
    assert all(str(t) == "bool" for t in matrix.dtypes)


def test_missing_start_date_means_early_member() -> None:
    """Live EODHD leaves StartDate empty for names that joined before its history (D25)."""
    from bot.data.membership import build_membership_history

    rows = [
        {"Code": "AET", "Name": "Aetna", "StartDate": None, "EndDate": "2018-12-03",
         "IsActiveNow": 0, "IsDelisted": 1},
        {"Code": "OLD", "Name": "Old member", "StartDate": None, "EndDate": None,
         "IsActiveNow": 1, "IsDelisted": 0},
    ]  # fmt: skip
    history = build_membership_history(rows)  # type: ignore[arg-type]
    assert "AET.US" in history.members_on(date(2016, 1, 4))
    assert "AET.US" not in history.members_on(date(2018, 12, 4))
    assert "OLD.US" in history.members_on(date(2023, 6, 1))


def test_client_keeps_rows_with_missing_start_date() -> None:
    """The client's holdout clip keeps early members and still drops 2024 starts."""
    from bot.data.eodhd_client import _clip_rows

    rows = [
        {"Code": "AET", "StartDate": None, "EndDate": "2018-12-03"},
        {"Code": "NEW", "StartDate": "2024-03-19", "EndDate": None},
        {"Code": "GONE", "StartDate": "2010-01-04", "EndDate": "2025-02-01"},
    ]
    kept = {row["Code"]: row for row in _clip_rows(rows)}
    assert set(kept) == {"AET", "GONE"}
    assert kept["GONE"]["EndDate"] is None
