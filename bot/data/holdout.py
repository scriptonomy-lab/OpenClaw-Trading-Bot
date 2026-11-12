"""The holdout guard. Data from config.HOLDOUT_START on is sealed.

Two keys are needed to read holdout data:

1. The caller passes ``approved=True``.
2. An approval session is open. Only ``holdout_session(strategy_id)`` opens
   one, and it checks a stored approval record for that strategy first.

So ``approved=True`` on its own still raises. This makes it hard to unseal
the holdout by accident, for example from a test or a copied function call.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from bot import config, storage

NEW_YORK = ZoneInfo("America/New_York")

# The strategy id of the open approval session, or None when no session is open.
_active_session: ContextVar[str | None] = ContextVar("holdout_session", default=None)
# True only while a paper session (not a holdout session) is open.
_paper_flag: ContextVar[bool] = ContextVar("paper_session", default=False)


# Functions called when a session closes, for example to delete session caches.
_close_hooks: list[Callable[[], None]] = []


def register_session_close_hook(hook: Callable[[], None]) -> None:
    """Run ``hook`` every time a holdout session closes. Registering twice is a no-op."""
    if hook not in _close_hooks:
        _close_hooks.append(hook)


class HoldoutAccessError(RuntimeError):
    """Raised when code tries to read holdout data without an approved session."""


@dataclass(frozen=True)
class HoldoutApproval:
    """A stored record that the user approved one strategy for its holdout test.

    ``used`` is True once the single holdout run has happened. The brief allows
    exactly one holdout run per approved strategy.
    """

    strategy_id: str
    approved_at: datetime
    used: bool


def find_approval(strategy_id: str) -> HoldoutApproval | None:
    """Look up the stored approval record for a strategy. Reading never creates the database."""
    stored = storage.find_stored_approval(strategy_id)
    if stored is None:
        return None
    return HoldoutApproval(stored.strategy_id, stored.approved_at, stored.used)


def mark_approval_used(strategy_id: str) -> bool:
    """Claim the strategy's single holdout run. False if someone else already claimed it."""
    return storage.set_approval_used(strategy_id)


@contextmanager
def holdout_session(strategy_id: str) -> Iterator[None]:
    """Open the only door to holdout data, for one approved strategy.

    Raises HoldoutAccessError if there is no approval record, if the record
    was already used, or if another session is already open.
    """
    if _active_session.get() is not None:
        raise HoldoutAccessError("A holdout session is already open.")
    approval = find_approval(strategy_id)
    if approval is None:
        raise HoldoutAccessError(f"No holdout approval on record for {strategy_id!r}.")
    if approval.used:
        raise HoldoutAccessError(f"The holdout run for {strategy_id!r} was already used.")
    # The claim is one UPDATE ... WHERE used = 0, so two callers cannot both win (M3).
    if mark_approval_used(strategy_id) is False:
        raise HoldoutAccessError(f"The holdout run for {strategy_id!r} was already used.")
    token = _active_session.set(strategy_id)
    try:
        yield
    finally:
        _active_session.reset(token)
        for hook in _close_hooks:
            hook()


@contextmanager
def paper_session(strategy_id: str) -> Iterator[None]:
    """Open the data door for paper trading one approved strategy. Never marks the approval.

    Paper trading reads data from 2024 on by design, but only for a strategy
    whose approval was already used by its holdout run and whose paper run was
    started by the approval flow. It opens the same session flag as
    ``holdout_session``, so the client wrappers behave the same way, and it
    runs the close hooks too. The hooks delete the session cache but not the
    forward cache, which only a paper session can read (D23). Raises HoldoutAccessError otherwise.
    """
    if _active_session.get() is not None:
        raise HoldoutAccessError("A holdout session is already open.")
    approval = find_approval(strategy_id)
    if approval is None or not approval.used:
        raise HoldoutAccessError(f"No finished holdout approval on record for {strategy_id!r}.")
    if storage.get_paper_run(strategy_id) is None:
        raise HoldoutAccessError(f"Paper trading was never started for {strategy_id!r}.")
    token = _active_session.set(strategy_id)
    paper_token = _paper_flag.set(True)
    try:
        yield
    finally:
        _paper_flag.reset(paper_token)
        _active_session.reset(token)
        for hook in _close_hooks:
            hook()


def is_paper_session() -> bool:
    """True while a paper session is open. A holdout session returns False."""
    return _active_session.get() is not None and _paper_flag.get()


def active_session_strategy() -> str | None:
    """Return the strategy id of the open approval session, or None."""
    return _active_session.get()


def _is_unsealed(approved: bool) -> bool:
    """True only when the caller says approved and a session is really open."""
    return approved and _active_session.get() is not None


def assert_before_holdout(end_date: date, *, approved: bool = False) -> None:
    """Raise HoldoutAccessError if end_date is on or after HOLDOUT_START.

    The check passes for holdout dates only when ``approved`` is True and a
    ``holdout_session`` is open. ``approved=True`` alone is not enough.
    """
    day = end_date.date() if isinstance(end_date, datetime) else end_date
    if day < config.HOLDOUT_START:
        return
    if _is_unsealed(approved):
        return
    hint = " approved=True also needs an open holdout_session." if approved else ""
    raise HoldoutAccessError(
        f"{day.isoformat()} is in the holdout (from {config.HOLDOUT_START.isoformat()}). "
        f"Only the approval flow may read it.{hint}"
    )


def row_calendar_date(value: object) -> date:
    """Turn a row's date value into a New York calendar date.

    Accepts ``YYYY-MM-DD`` or an ISO datetime string. A datetime with a time
    zone is converted to America/New_York first. A naive datetime keeps its own
    date. Anything else raises ValueError, so bad rows are never kept silently.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Row date must be a non-empty string, got {value!r}.")
    if len(value) == 10:
        return date.fromisoformat(value)
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is not None:
        moment = moment.astimezone(NEW_YORK)
    return moment.date()


def drop_holdout_rows(
    rows: list[dict[str, Any]], date_field: str, *, approved: bool = False
) -> list[dict[str, Any]]:
    """Return the rows whose date_field is before HOLDOUT_START.

    With ``approved=True`` and an open ``holdout_session``, all rows are kept.
    A missing or unparseable date raises (KeyError or ValueError).
    """
    if _is_unsealed(approved):
        return list(rows)
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row_calendar_date(row[date_field]) < config.HOLDOUT_START:
            kept.append(row)
    return kept
