"""Membership checks against the real S&P 500 history (pytest -m integration).

Costs a few API calls (the first run fetches the history, later runs read the cache).
"""

from datetime import date, timedelta
from typing import cast

import pytest

from bot.data import eodhd_client as client
from bot.data.membership import MembershipHistory, RawMembershipRow, build_membership_history

pytestmark = pytest.mark.integration

# Known index additions, effective before the open on the date given.
# Source for each: S&P Dow Jones Indices press releases, cross-read with the
# Wikipedia page "List of S&P 500 component changes".
# confirmed=False means the date comes from memory and needs confirmation
# against the press release. Those cases are expected to fail until checked
# (xfail, non-strict) so a wrong date never breaks the run or hides a real problem.
KNOWN_ADDITIONS = [
    ("TSLA", date(2020, 12, 21), True),
    ("ETSY", date(2020, 9, 21), False),
    ("TER", date(2020, 9, 21), False),
    ("CTLT", date(2020, 9, 21), False),
]
START_TOLERANCE_DAYS = 3  # the feed may use the announcement or the effective date (D2, D15)


@pytest.fixture(scope="module")
def history() -> MembershipHistory:
    """Membership built from the real history rows."""
    rows = cast(list[RawMembershipRow], client.get_index_history())
    return build_membership_history(rows)


def _cases() -> list[object]:
    cases: list[object] = []
    for code, added, confirmed in KNOWN_ADDITIONS:
        marks = (
            [] if confirmed else [pytest.mark.xfail(reason="date needs confirmation", strict=False)]
        )
        cases.append(pytest.param(code, added, id=code, marks=marks))
    return cases


@pytest.mark.parametrize(("code", "added"), _cases())
def test_known_addition_dates(history: MembershipHistory, code: str, added: date) -> None:
    """Not a member a few days before, a member a few days after, start near the known date."""
    symbol = f"{code}.US"
    assert symbol not in history.members_on(added - timedelta(days=4))
    assert symbol in history.members_on(added + timedelta(days=4))
    starts = [s.start for s in history.spells if s.symbol == symbol]
    assert any(abs((start - added).days) <= START_TOLERANCE_DAYS for start in starts)


def test_delisted_former_member_has_prices_to_its_last_day(history: MembershipHistory) -> None:
    """A removed, delisted name is a member before removal and has prices up to that day."""
    rows = cast(list[RawMembershipRow], client.get_index_history())
    removed = [
        row
        for row in rows
        if row["IsDelisted"] and row["EndDate"] and "2016-01-01" <= row["EndDate"] <= "2023-06-30"
    ]
    assert removed, "No delisted former member found in 2016-2023; the history looks incomplete."
    row = sorted(removed, key=lambda r: (r["EndDate"] or "", r["Code"]))[len(removed) // 2]
    last = date.fromisoformat(str(row["EndDate"]))
    symbol = f"{row['Code']}.US"
    assert symbol in history.members_on(last - timedelta(days=30))
    prices = client.get_eod_prices(
        row["Code"], last - timedelta(days=90), last + timedelta(days=30)
    )
    assert prices, f"{symbol} has no prices: a survivorship leak."
    last_price_day = date.fromisoformat(prices[-1]["date"])
    assert abs((last - last_price_day).days) <= 10
