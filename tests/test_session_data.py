"""Session data (D23): unclipped membership in sessions and the paper forward cache.

All tests run on the fake world with a temporary cache folder.
"""

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from bot.data import eodhd_client as client
from bot.data import fake_eodhd, holdout
from bot.data.holdout import HoldoutAccessError, HoldoutApproval, holdout_session, paper_session
from bot.data.membership import RawMembershipRow, build_membership_history

TOKEN = fake_eodhd.default_fake_token()
LATE = re.compile(r'"(?:StartDate|EndDate)": "20(?:2[4-9]|3\d)-')


@pytest.fixture(autouse=True)
def sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let holdout and paper sessions open for any strategy, without a database."""
    monkeypatch.setattr(
        holdout,
        "find_approval",
        lambda sid: HoldoutApproval(sid, datetime(2025, 1, 1, tzinfo=UTC), sid == "paper"),
    )
    monkeypatch.setattr(holdout, "mark_approval_used", lambda sid: None)
    monkeypatch.setattr(holdout.storage, "get_paper_run", lambda sid: {"start_day": "2026-01-02"})


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Every request the fake transport answers, as (path, query)."""
    seen: list[tuple[str, dict[str, Any]]] = []
    original = fake_eodhd.FakeEodhdTransport.get

    def counting(self: Any, url: str, params: dict[str, Any] | None = None, timeout: float = 0):  # type: ignore[no-untyped-def]
        seen.append((url.split("/api", 1)[1], dict(params or {})))
        return original(self, url, params, timeout)

    monkeypatch.setattr(fake_eodhd.FakeEodhdTransport, "get", counting)
    return seen


def _rows(base: Path, *, approved: bool = False) -> list[RawMembershipRow]:
    rows = client.get_index_history(approved=approved, cache_dir=base, api_token=TOKEN)
    return rows  # type: ignore[return-value]


def _late_changes() -> tuple[str, date, str, date]:
    """A ticker removed in 2024 and one added in 2024, from the fake world."""
    raw = fake_eodhd.FakeEodhdTransport().get(
        client.BASE_URL + f"/fundamentals/{client.INDEX_SYMBOL}",
        params={"filter": "HistoricalTickerComponents", "api_token": TOKEN},
    )
    rows = list(raw.json().values())
    removed = next(r for r in rows if (r["EndDate"] or "").startswith("2024-03-18"))
    added = next(r for r in rows if (r["StartDate"] or "") >= "2024-03-19")
    return (
        removed["Code"],
        date.fromisoformat(removed["EndDate"]),
        added["Code"],
        date.fromisoformat(added["StartDate"]),
    )


def _check_membership(rows: list[RawMembershipRow]) -> None:
    removed, removed_on, added, added_on = _late_changes()
    history = build_membership_history(rows, approved=True)
    before = history.members_on(date(2024, 3, 15), approved=True)
    after = history.members_on(date(2024, 4, 15), approved=True)
    assert f"{removed}.US" in before
    assert f"{removed}.US" not in after
    assert f"{added}.US" not in before
    assert f"{added}.US" in after
    assert removed_on < added_on


def test_outside_session_membership_is_clipped_and_cached_clipped(tmp_path: Path) -> None:
    base = tmp_path / "eodhd"
    rows = _rows(base)
    assert not any(LATE.search(f'"StartDate": "{r["StartDate"]}"') for r in rows)
    assert all(not (r["EndDate"] or "").startswith(("2024", "2025", "2026")) for r in rows)
    cached = list(base.rglob("*.json"))
    assert cached
    assert not any(LATE.search(path.read_text()) for path in cached)
    with pytest.raises(HoldoutAccessError):
        client.get_eod_prices("SIM001", date(2024, 1, 2), date(2024, 2, 1), cache_dir=base)


def test_holdout_session_sees_unclipped_membership(tmp_path: Path) -> None:
    base = tmp_path / "eodhd"
    _rows(base)  # the clipped normal cache exists first
    with holdout_session("s1"):
        _check_membership(_rows(base, approved=True))
        # Without approved=True the caller still gets clipped rows.
        assert not any((r["StartDate"] or "") >= "2024-01-01" for r in _rows(base))
    assert not (tmp_path / "eodhd-holdout").exists()
    assert not any(LATE.search(path.read_text()) for path in base.rglob("*.json"))


def test_paper_session_sees_unclipped_membership(tmp_path: Path) -> None:
    base = tmp_path / "eodhd"
    _rows(base)
    with paper_session("paper"):
        _check_membership(_rows(base, approved=True))
    assert not any(LATE.search(path.read_text()) for path in base.rglob("*.json"))


def _paper_update(base: Path, today: date) -> None:
    """Read what one paper update reads for a few symbols."""
    with paper_session("paper"):
        start = date(2025, 6, 2)
        _rows(base, approved=True)
        for ticker in ("SIM001", "SIM002"):
            client.get_eod_prices(
                ticker, start, today, approved=True, cache_dir=base, api_token=TOKEN
            )
            client.get_dividends(
                ticker, start, today, approved=True, cache_dir=base, api_token=TOKEN
            )
            client.get_splits(ticker, approved=True, cache_dir=base, api_token=TOKEN)
            client.get_news(ticker, start, today, approved=True, cache_dir=base, api_token=TOKEN)
            client.get_daily_sentiment(
                ticker, start, today, approved=True, cache_dir=base, api_token=TOKEN
            )
            client.get_earnings(
                ticker, start, today, approved=True, cache_dir=base, api_token=TOKEN
            )


def test_forward_cache_survives_paper_updates(
    tmp_path: Path, calls: list[tuple[str, dict[str, Any]]]
) -> None:
    base = tmp_path / "eodhd"
    today = date(2026, 10, 30)
    _paper_update(base, today)
    first = len(calls)
    assert first > 0
    assert list((tmp_path / "eodhd-forward").rglob("*.json"))
    assert not (tmp_path / "eodhd-holdout").exists()
    _paper_update(base, today)
    assert len(calls) == first  # nothing downloaded again


def test_forward_cache_fetches_only_new_days(
    tmp_path: Path, calls: list[tuple[str, dict[str, Any]]]
) -> None:
    base = tmp_path / "eodhd"
    _paper_update(base, date(2026, 10, 28))
    calls.clear()
    with paper_session("paper"):
        rows = client.get_eod_prices(
            "SIM001", date(2025, 6, 2), date(2026, 10, 30), approved=True, cache_dir=base,
            api_token=TOKEN,
        )  # fmt: skip
    # Five trading days before the last covered row (H2), then everything new.
    assert [q["from"] for _, q in calls] == ["2026-10-21"]
    assert rows[-1]["date"] == "2026-10-30"
    stamps = [r["date"] for r in rows]
    assert stamps == sorted(set(stamps))
    # A later window start is served from the same entry.
    calls.clear()
    with paper_session("paper"):
        later = client.get_eod_prices(
            "SIM001", date(2025, 9, 1), date(2026, 10, 30), approved=True, cache_dir=base,
            api_token=TOKEN,
        )  # fmt: skip
    assert not calls
    assert later[0]["date"] >= "2025-09-01"


def test_non_session_read_cannot_see_forward_cache(tmp_path: Path) -> None:
    base = tmp_path / "eodhd"
    _paper_update(base, date(2026, 10, 30))
    forward = tmp_path / "eodhd-forward"
    assert list(forward.rglob("*.json"))
    # The same request outside a session is refused by the guard.
    with pytest.raises(HoldoutAccessError):
        client.get_eod_prices("SIM001", date(2025, 6, 2), date(2026, 10, 30), cache_dir=base)
    # A pre-holdout request outside a session ignores the forward folder.
    with pytest.raises(client.OfflineCacheMissError):
        client.get_eod_prices(
            "SIM001", date(2022, 6, 2), date(2022, 10, 30), cache_dir=base, offline=True
        )
    with pytest.raises(client.OfflineCacheMissError):
        client.get_index_history(cache_dir=base, offline=True, api_token=TOKEN)
    # Inside a holdout session the forward cache is not read either.
    with holdout_session("s1"), pytest.raises(client.OfflineCacheMissError):
        client.get_eod_prices(
            "SIM001", date(2025, 6, 2), date(2026, 10, 30), approved=True, cache_dir=base,
            offline=True,
        )  # fmt: skip


def test_holdout_session_cache_still_purged(tmp_path: Path) -> None:
    base = tmp_path / "eodhd"
    _paper_update(base, date(2026, 10, 30))
    with holdout_session("s1"):
        client.get_eod_prices(
            "SIM001", date(2025, 6, 2), date(2026, 10, 30), approved=True, cache_dir=base,
            api_token=TOKEN,
        )  # fmt: skip
        assert (tmp_path / "eodhd-holdout").exists()
    assert not (tmp_path / "eodhd-holdout").exists()
    assert list((tmp_path / "eodhd-forward").rglob("*.json"))


# ------------------------------------------------------------------ H2 and H3

ALL_DAYS = ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05"]


class LateBars:
    """A stub fetcher that behaves like a live API: it only returns published days."""

    def __init__(self) -> None:
        self.published = "2025-06-02"
        self.calls: list[dict[str, Any]] = []

    def __call__(self, token: str, params: dict[str, Any]) -> Any:
        self.calls.append(dict(params))
        rows = [
            {"date": d, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
             "adjusted_close": 10.5, "volume": 1}
            for d in ALL_DAYS
            if params["from"] <= d <= params["to"] and d <= self.published
        ]  # fmt: skip
        if not rows:
            raise client.EodhdEmptyResponseError("empty")
        return rows


def _load(base: Path, stub: LateBars, to: str, as_of: date) -> list[str]:
    with client.forward_as_of(as_of):
        rows = client._forward_load(
            "/eod/SIM001.US", {"from": "2025-06-01", "to": to}, base=base,
            offline=False, api_token="fake-x", fetch=stub,
        )  # fmt: skip
    return [r["date"] for r in rows]


def test_late_bar_is_picked_up_on_the_next_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = LateBars()
    base = tmp_path / "fake"
    # The clock says 15:00 New York on 06-03, before the bar exists.
    monkeypatch.setattr(client, "_utcnow", lambda: datetime(2025, 6, 3, 19, 0, tzinfo=UTC))
    assert _load(base, stub, "2025-06-03", date(2025, 6, 3)) == ["2025-06-02"]
    # Before 18:00 the same day is not settled, so a repeat asks again.
    n = len(stub.calls)
    _load(base, stub, "2025-06-03", date(2025, 6, 3))
    assert len(stub.calls) == n + 1
    # Next day, everything is published.
    stub.published = "2025-06-05"
    monkeypatch.setattr(client, "_utcnow", lambda: datetime(2025, 6, 4, 23, 0, tzinfo=UTC))
    assert _load(base, stub, "2025-06-04", date(2025, 6, 4)) == ALL_DAYS[:3]
    assert stub.calls[-1]["from"] < "2025-06-03"
    assert _load(base, stub, "2025-06-05", date(2025, 6, 5)) == ALL_DAYS


def test_settled_repeat_makes_no_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = LateBars()
    stub.published = "2025-06-05"
    base = tmp_path / "fake"
    monkeypatch.setattr(client, "_utcnow", lambda: datetime(2025, 6, 4, 23, 0, tzinfo=UTC))
    _load(base, stub, "2025-06-04", date(2025, 6, 4))
    n = len(stub.calls)
    _load(base, stub, "2025-06-04", date(2025, 6, 4))
    assert len(stub.calls) == n


def test_bar_published_after_the_fetch_is_replaced_not_duplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = LateBars()
    base = tmp_path / "fake"
    stub.published = "2025-06-03"
    monkeypatch.setattr(client, "_utcnow", lambda: datetime(2025, 6, 3, 23, 30, tzinfo=UTC))
    _load(base, stub, "2025-06-03", date(2025, 6, 3))
    # 06-03 settled at 19:30 but the vendor adds 06-04 and the next update asks for it.
    stub.published = "2025-06-04"
    monkeypatch.setattr(client, "_utcnow", lambda: datetime(2025, 6, 4, 23, 30, tzinfo=UTC))
    days = _load(base, stub, "2025-06-04", date(2025, 6, 4))
    assert days == ["2025-06-02", "2025-06-03", "2025-06-04"]


def test_splits_refresh_after_one_day(tmp_path: Path) -> None:
    import os
    import time

    base = tmp_path / "fake"
    seen: list[dict[str, Any]] = []
    listed: list[dict[str, str]] = [{"date": "2025-01-10", "split": "2/1"}]

    def fetch(token: str, params: dict[str, Any]) -> Any:
        seen.append(params)
        return list(listed)

    path = "/splits/SIM001.US"
    client._forward_plain(path, {}, base, False, "fake-x", fetch)
    listed.append({"date": "2025-06-04", "split": "4/1"})
    assert len(client._forward_plain(path, {}, base, False, "fake-x", fetch)) == 1
    file = client.cache_path(path, {}, client._forward_dir(base))
    old = time.time() - 26 * 3600
    os.utime(file, (old, old))
    assert len(client._forward_plain(path, {}, base, False, "fake-x", fetch)) == 2
