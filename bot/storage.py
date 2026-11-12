"""SQLite storage for strategies, backtests, trades, approvals and paper trading.

The database file lives in ``config.output_dir()``, so fake and live runs
never share one. Plain functions open a short connection each, so there is no
global state. Lookups that only read (``find_approval``) never create the file.
"""

import hmac
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bot.config import data_mode, output_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY, json TEXT NOT NULL, attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL, chat_id TEXT);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL, kind TEXT NOT NULL,
    window_label TEXT NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL,
    cost_bps REAL NOT NULL, metrics_json TEXT NOT NULL, chart_paths_json TEXT NOT NULL,
    data_mode TEXT NOT NULL, chat_id TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL, source TEXT NOT NULL,
    window_label TEXT, symbol TEXT NOT NULL, signal_date TEXT NOT NULL, entry_date TEXT NOT NULL,
    exit_date TEXT NOT NULL, exit_reason TEXT NOT NULL, signal_values_json TEXT NOT NULL,
    row_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (
    strategy_id TEXT PRIMARY KEY, approved_at TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS approval_codes (
    strategy_id TEXT PRIMARY KEY, code_hash TEXT NOT NULL, expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0, failed_attempts INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_runs (
    strategy_id TEXT PRIMARY KEY, start_day TEXT NOT NULL, started_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_state (
    strategy_id TEXT NOT NULL, day TEXT NOT NULL, state_json TEXT NOT NULL,
    PRIMARY KEY (strategy_id, day));
CREATE TABLE IF NOT EXISTS paper_ledger (
    strategy_id TEXT NOT NULL, day TEXT NOT NULL, cash REAL NOT NULL,
    holdings_value REAL NOT NULL, equity REAL NOT NULL, open_positions INTEGER NOT NULL,
    PRIMARY KEY (strategy_id, day));
"""


@dataclass(frozen=True)
class StoredApproval:
    """One approval row. ``used`` is True once the single holdout run has started."""

    strategy_id: str
    approved_at: datetime
    used: bool


def db_path() -> Path:
    """The database file for the current data mode."""
    return output_dir() / "bot.sqlite"


def _now() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open the database (creating tables), commit on success, always close."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


@contextmanager
def _immediate() -> Iterator[sqlite3.Connection]:
    """Open a connection inside ``BEGIN IMMEDIATE``, so read-then-write steps are atomic.

    Other writers wait (up to 30 seconds) until this transaction ends.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
    finally:
        connection.close()


# ---------------------------------------------------------------- strategies


def save_strategy(strategy_id: str, strategy_json: str, chat_id: str | None) -> int:
    """Store a validated strategy. Returns its attempt number in the chat.

    A strategy that is already stored keeps its first attempt number.
    """
    with _connect() as db:
        row = db.execute(
            "SELECT attempt FROM strategies WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is not None:
            return int(row["attempt"])
        count = db.execute(
            "SELECT COUNT(*) FROM strategies WHERE chat_id IS ?", (chat_id,)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO strategies VALUES (?, ?, ?, ?, ?)",
            (strategy_id, strategy_json, count + 1, _now(), chat_id),
        )
        return int(count) + 1


def get_strategy_json(strategy_id: str) -> str | None:
    """The stored strategy JSON, or None if this id was never validated."""
    with _connect() as db:
        row = db.execute(
            "SELECT json FROM strategies WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
    return None if row is None else str(row["json"])


# ------------------------------------------------------------------ backtests


def versions_tried(chat_id: str | None, including: str | None = None) -> int:
    """Distinct strategy ids backtested in a chat (D18 item 2).

    ``including`` adds one id that is about to run. ``chat_id=None`` counts
    every chat, because the allowlisted CLI never receives a chat id.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT DISTINCT strategy_id FROM backtest_runs "
            "WHERE kind = 'backtest' AND (? IS NULL OR chat_id = ?)",
            (chat_id, chat_id),
        ).fetchall()
    ids = {str(row["strategy_id"]) for row in rows}
    if including is not None:
        ids.add(including)
    return len(ids)


def replace_backtest(
    strategy_id: str,
    kind: str,
    runs: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    chat_id: str | None,
) -> None:
    """Replace the saved runs and ledger of one strategy for ``kind`` (backtest or holdout).

    Each run dict has window_label, window_start, window_end, cost_bps,
    metrics and chart_paths. Each trade dict is a ledger row plus window_label.
    """
    with _connect() as db:
        db.execute(
            "DELETE FROM backtest_runs WHERE strategy_id = ? AND kind = ?", (strategy_id, kind)
        )
        db.execute("DELETE FROM trades WHERE strategy_id = ? AND source = ?", (strategy_id, kind))
        for run in runs:
            db.execute(
                "INSERT INTO backtest_runs (strategy_id, kind, window_label, window_start, "
                "window_end, cost_bps, metrics_json, chart_paths_json, data_mode, chat_id, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    strategy_id, kind, run["window_label"], run["window_start"],
                    run["window_end"], run["cost_bps"], json.dumps(run["metrics"]),
                    json.dumps(run["chart_paths"]), data_mode(), chat_id, _now(),
                ),
            )  # fmt: skip
        for trade in trades:
            _insert_trade(db, strategy_id, kind, trade)


def _insert_trade(
    db: sqlite3.Connection, strategy_id: str, source: str, trade: dict[str, Any]
) -> None:
    """Insert one ledger row. ``trade`` is the JSON form of an engine Trade."""
    db.execute(
        "INSERT INTO trades (strategy_id, source, window_label, symbol, signal_date, entry_date, "
        "exit_date, exit_reason, signal_values_json, row_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            strategy_id, source, trade.get("window_label"), trade["symbol"],
            trade["signal_date"], trade["entry_date"], trade["exit_date"],
            trade["exit_reason"], json.dumps(trade["signal_values"]), json.dumps(trade),
        ),
    )  # fmt: skip


def get_runs(strategy_id: str, kind: str = "backtest") -> list[dict[str, Any]]:
    """Saved runs for a strategy, oldest first, with metrics and chart paths decoded."""
    with _connect() as db:
        rows = db.execute(
            "SELECT * FROM backtest_runs WHERE strategy_id = ? AND kind = ? ORDER BY id",
            (strategy_id, kind),
        ).fetchall()
    return [
        {
            "window_label": row["window_label"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "cost_bps": float(row["cost_bps"]),
            "metrics": json.loads(row["metrics_json"]),
            "chart_paths": json.loads(row["chart_paths_json"]),
            "data_mode": row["data_mode"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def find_trades(
    symbol: str, sources: tuple[str, ...] = ("backtest", "paper")
) -> list[dict[str, Any]]:
    """Stored ledger rows for a symbol, newest signal first.

    Holdout rows are left out unless "holdout" is in ``sources`` (review M4).
    """
    marks = ",".join("?" for _ in sources)
    with _connect() as db:
        rows = db.execute(
            f"SELECT strategy_id, source, row_json FROM trades WHERE symbol = ? "
            f"AND source IN ({marks}) ORDER BY signal_date DESC, id DESC",
            (symbol, *sources),
        ).fetchall()
    return [
        {**json.loads(row["row_json"]), "source": row["source"], "strategy_id": row["strategy_id"]}
        for row in rows
    ]


# ------------------------------------------------------------------ approvals


def find_stored_approval(strategy_id: str) -> StoredApproval | None:
    """The approval row for a strategy, or None. Never creates the database."""
    if not db_path().exists():
        return None
    with _connect() as db:
        row = db.execute("SELECT * FROM approvals WHERE strategy_id = ?", (strategy_id,)).fetchone()
    if row is None:
        return None
    return StoredApproval(
        strategy_id=str(row["strategy_id"]),
        approved_at=datetime.fromisoformat(row["approved_at"]),
        used=bool(row["used"]),
    )


def record_approval(strategy_id: str) -> bool:
    """Write an approval row. Returns False if the strategy already has one."""
    with _connect() as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO approvals (strategy_id, approved_at, used) VALUES (?, ?, 0)",
            (strategy_id, _now()),
        )
        return cursor.rowcount == 1


def set_approval_used(strategy_id: str) -> bool:
    """Claim the single holdout run. True for exactly one caller, False if already claimed."""
    with _connect() as db:
        cursor = db.execute(
            "UPDATE approvals SET used = 1 WHERE strategy_id = ? AND used = 0", (strategy_id,)
        )
        return cursor.rowcount == 1


def count_used_approvals() -> int:
    """How many holdout runs this database has done (approvals marked used)."""
    if not db_path().exists():
        return 0
    with _connect() as db:
        return int(db.execute("SELECT COUNT(*) FROM approvals WHERE used = 1").fetchone()[0])


# --------------------------------------------------------------- paper trading


def start_paper_run(strategy_id: str, start_day: str) -> bool:
    """Record that paper trading starts on ``start_day``. False if it already started."""
    with _connect() as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO paper_runs VALUES (?, ?, ?)", (strategy_id, start_day, _now())
        )
        return cursor.rowcount == 1


def get_paper_run(strategy_id: str) -> dict[str, str] | None:
    """The paper run row (strategy_id, start_day, started_at), or None."""
    if not db_path().exists():
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT * FROM paper_runs WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
    return None if row is None else {key: str(value) for key, value in dict(row).items()}


def list_paper_runs() -> list[dict[str, str]]:
    """Every paper run, most recently started first."""
    with _connect() as db:
        rows = db.execute("SELECT * FROM paper_runs ORDER BY started_at DESC").fetchall()
    return [{key: str(value) for key, value in dict(row).items()} for row in rows]


def latest_paper_state(strategy_id: str) -> tuple[str, str] | None:
    """(day, state_json) of the newest saved paper state, or None."""
    with _connect() as db:
        row = db.execute(
            "SELECT day, state_json FROM paper_state WHERE strategy_id = ? "
            "ORDER BY day DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
    return None if row is None else (str(row["day"]), str(row["state_json"]))


def save_paper_day(
    strategy_id: str,
    day: str,
    state_json: str,
    ledger_row: dict[str, Any],
    trades: list[dict[str, Any]],
) -> None:
    """Save one paper day in one transaction: state, ledger row and closed trades."""
    with _connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO paper_state VALUES (?, ?, ?)", (strategy_id, day, state_json)
        )
        db.execute(
            "INSERT OR REPLACE INTO paper_ledger VALUES (?, ?, ?, ?, ?, ?)",
            (
                strategy_id, day, ledger_row["cash"], ledger_row["holdings_value"],
                ledger_row["equity"], ledger_row["open_positions"],
            ),
        )  # fmt: skip
        for trade in trades:
            _insert_trade(db, strategy_id, "paper", trade)


def paper_ledger(strategy_id: str) -> list[dict[str, Any]]:
    """Daily equity rows for one paper run, oldest first."""
    with _connect() as db:
        rows = db.execute(
            "SELECT * FROM paper_ledger WHERE strategy_id = ? ORDER BY day", (strategy_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_trades(
    strategy_id: str,
    source: str,
    exit_date: str | None = None,
    exited_after: str | None = None,
) -> list[dict[str, Any]]:
    """Ledger rows of one strategy and source.

    ``exit_date`` keeps only trades that exited on that day. ``exited_after``
    keeps trades that exited on a later day.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT row_json FROM trades WHERE strategy_id = ? AND source = ? "
            "AND (? IS NULL OR exit_date = ?) AND (? IS NULL OR exit_date > ?) ORDER BY id",
            (strategy_id, source, exit_date, exit_date, exited_after, exited_after),
        ).fetchall()
    return [json.loads(row["row_json"]) for row in rows]


# ------------------------------------------------------------ approval codes


@dataclass(frozen=True)
class CodeCheck:
    """Result of one atomic code check. ``status`` is ok, locked, none, expired or wrong."""

    status: str
    tries_left: int = 0


def save_approval_code(strategy_id: str, code_hash: str, expires_at: str) -> None:
    """Store a new one-time code hash. Earlier wrong attempts are kept, so a lockout stays."""
    with _connect() as db:
        db.execute(
            "INSERT INTO approval_codes (strategy_id, code_hash, expires_at) VALUES (?, ?, ?) "
            "ON CONFLICT(strategy_id) DO UPDATE SET code_hash = excluded.code_hash, "
            "expires_at = excluded.expires_at, used = 0",
            (strategy_id, code_hash, expires_at),
        )


def get_approval_code(strategy_id: str) -> dict[str, Any] | None:
    """The code row (code_hash, expires_at, used, failed_attempts), or None."""
    with _connect() as db:
        row = db.execute(
            "SELECT * FROM approval_codes WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
    return None if row is None else dict(row)


def check_approval_code(
    strategy_id: str, given_hash: str, now: datetime, max_wrong: int
) -> CodeCheck:
    """Read, compare and update the code row in one ``BEGIN IMMEDIATE`` transaction (M1).

    A wrong guess adds one failed attempt. Once ``max_wrong`` is reached no
    guess is compared at all. A right guess marks the code used, and only one
    caller can ever get "ok" for a code.
    """
    with _immediate() as db:
        row = db.execute(
            "SELECT * FROM approval_codes WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is not None and row["failed_attempts"] >= max_wrong:
            return CodeCheck("locked")
        if row is None or row["used"]:
            return CodeCheck("none")
        if now > datetime.fromisoformat(row["expires_at"]):
            return CodeCheck("expired")
        if not hmac.compare_digest(given_hash, row["code_hash"]):
            failed = int(row["failed_attempts"]) + 1
            db.execute(
                "UPDATE approval_codes SET failed_attempts = ? WHERE strategy_id = ?",
                (failed, strategy_id),
            )
            return CodeCheck("wrong", max_wrong - failed)
        cursor = db.execute(
            "UPDATE approval_codes SET used = 1 WHERE strategy_id = ? AND used = 0", (strategy_id,)
        )
        return CodeCheck("ok" if cursor.rowcount == 1 else "none")


def void_approval_code(strategy_id: str) -> None:
    """Make an unsent code useless but keep the row, so failed attempts are never reset (M2)."""
    with _connect() as db:
        db.execute(
            "UPDATE approval_codes SET code_hash = '', used = 1 WHERE strategy_id = ? AND used = 0",
            (strategy_id,),
        )


# ----------------------------------------------------------------------- meta


def get_meta(key: str) -> str | None:
    """A small stored value (a secret, a last-summary record), or None."""
    with _connect() as db:
        row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_meta(key: str, value: str) -> None:
    """Store a small value, replacing any earlier one."""
    with _connect() as db:
        db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def set_meta_if_absent(key: str, value: str) -> str:
    """Store ``value`` only if the key is new. Returns the value that is stored."""
    with _immediate() as db:
        db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
        return str(db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()["value"])
