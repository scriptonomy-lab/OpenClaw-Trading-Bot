"""Holdout guard tests. No network."""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from bot.data import eodhd_client as client
from bot.data import holdout
from bot.data.holdout import (
    HoldoutAccessError,
    HoldoutApproval,
    assert_before_holdout,
    drop_holdout_rows,
    holdout_session,
)

APPROVAL_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def _approve(monkeypatch: pytest.MonkeyPatch, used: bool = False) -> list[str]:
    """Make find_approval return a record; return the list of used marks."""
    marks: list[str] = []
    monkeypatch.setattr(
        holdout, "find_approval", lambda sid: HoldoutApproval(sid, APPROVAL_TIME, used)
    )
    monkeypatch.setattr(holdout, "mark_approval_used", marks.append)
    return marks


def test_last_day_before_holdout_passes() -> None:
    assert_before_holdout(date(2023, 12, 31))


def test_first_holdout_day_raises() -> None:
    with pytest.raises(HoldoutAccessError):
        assert_before_holdout(date(2024, 1, 1))


def test_approved_flag_alone_still_raises() -> None:
    with pytest.raises(HoldoutAccessError):
        assert_before_holdout(date(2024, 1, 1), approved=True)


def test_datetime_input_uses_its_date() -> None:
    assert_before_holdout(datetime(2023, 12, 31, 23, 0))  # type: ignore[arg-type]
    with pytest.raises(HoldoutAccessError):
        assert_before_holdout(datetime(2024, 1, 1, 0, 0))  # type: ignore[arg-type]


def test_approved_path_works_only_inside_session(monkeypatch: pytest.MonkeyPatch) -> None:
    marks = _approve(monkeypatch)
    with holdout_session("strat-1"):
        assert_before_holdout(date(2024, 6, 3), approved=True)
        # Without the approved flag the session alone does not unseal.
        with pytest.raises(HoldoutAccessError):
            assert_before_holdout(date(2024, 6, 3))
    assert marks == ["strat-1"]
    with pytest.raises(HoldoutAccessError):
        assert_before_holdout(date(2024, 6, 3), approved=True)


def test_session_without_approval_record_refused() -> None:
    with pytest.raises(HoldoutAccessError), holdout_session("nobody"):
        pass


def test_used_approval_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    marks = _approve(monkeypatch, used=True)
    with pytest.raises(HoldoutAccessError), holdout_session("strat-1"):
        pass
    assert marks == []


def test_second_session_while_open_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _approve(monkeypatch)
    with holdout_session("strat-1"), pytest.raises(HoldoutAccessError), holdout_session("strat-2"):
        pass


def test_session_closes_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _approve(monkeypatch)
    with pytest.raises(RuntimeError), holdout_session("strat-1"):
        raise RuntimeError("boom")
    assert holdout.active_session_strategy() is None
    with pytest.raises(HoldoutAccessError):
        assert_before_holdout(date(2024, 1, 2), approved=True)


def test_drop_holdout_rows_by_plain_date() -> None:
    rows = [{"d": "2023-12-31"}, {"d": "2024-01-01"}, {"d": "2025-05-05"}]
    assert drop_holdout_rows(rows, "d") == [{"d": "2023-12-31"}]


def test_drop_holdout_rows_uses_new_york_date() -> None:
    # 03:00 UTC is 22:00 on Dec 31 in New York (kept). 05:00 UTC is 00:00 Jan 1 (dropped).
    rows = [
        {"d": "2024-01-01T03:00:00+00:00", "id": 1},
        {"d": "2024-01-01T05:00:00+00:00", "id": 2},
    ]
    kept = drop_holdout_rows(rows, "d")
    assert [r["id"] for r in kept] == [1]


def test_drop_holdout_rows_missing_or_none_date_raises() -> None:
    with pytest.raises(KeyError):
        drop_holdout_rows([{"x": "2023-01-01"}], "d")
    with pytest.raises(ValueError):
        drop_holdout_rows([{"d": None}], "d")
    with pytest.raises(ValueError):
        drop_holdout_rows([{"d": "not a date"}], "d")


def test_drop_holdout_rows_approved_session_keeps_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _approve(monkeypatch)
    rows = [{"d": "2023-12-31"}, {"d": "2024-03-01"}]
    with holdout_session("strat-1"):
        assert drop_holdout_rows(rows, "d", approved=True) == rows
    assert drop_holdout_rows(rows, "d", approved=True) == [{"d": "2023-12-31"}]


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network was used")

    monkeypatch.setattr(requests, "get", fail)


@pytest.mark.parametrize("end", [date(2024, 1, 1), date(2024, 6, 30)])
@pytest.mark.parametrize("approved", [False, True])
def test_client_wrappers_with_end_date_refuse_holdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, end: date, approved: bool
) -> None:
    _no_network(monkeypatch)
    start = date(2023, 1, 1)
    kw: dict[str, Any] = {"cache_dir": tmp_path, "api_token": "x", "approved": approved}
    calls = [
        lambda: client.get_eod_prices("AAPL", start, end, **kw),
        lambda: client.get_dividends("AAPL", start, end, **kw),
        lambda: client.get_news("AAPL", start, end, **kw),
        lambda: client.get_daily_sentiment("AAPL", start, end, **kw),
        lambda: client.get_earnings("AAPL", start, end, **kw),
        lambda: client.get_symbol_changes(start, end, **kw),
    ]
    for call in calls:
        with pytest.raises(HoldoutAccessError):
            call()
