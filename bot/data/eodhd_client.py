"""Thin EODHD client with a deterministic on-disk cache and holdout guards.

Every wrapper calls the holdout guard before it touches the cache or the
network, and filters returned rows afterwards. The API token is sent as a
query parameter but is never written to disk, logs or error messages.
"""

import json
import logging
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Any

import requests

from bot.config import CACHE_DIR, HOLDOUT_START, data_mode, load_secrets
from bot.data.aliases import price_code
from bot.data.earnings import nyse_holidays
from bot.data.fake_eodhd import FakeEodhdTransport, default_fake_token
from bot.data.holdout import (
    NEW_YORK,
    HoldoutAccessError,
    active_session_strategy,
    assert_before_holdout,
    drop_holdout_rows,
    is_paper_session,
    register_session_close_hook,
    row_calendar_date,
)

# urllib3 DEBUG logging prints the full query string, which holds the token.
logging.getLogger("urllib3").setLevel(logging.WARNING)

BASE_URL = "https://eodhd.com/api"
DEFAULT_CACHE_DIR = CACHE_DIR / "eodhd"
# Fake responses never share a folder with real ones.
FAKE_CACHE_DIR = CACHE_DIR / "fake"
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2.0
# The news endpoint documents offset+limit <= 10000 per query window.
NEWS_MAX_WINDOW = 10_000
INDEX_SYMBOL = "GSPC.INDX"
# News is cached without article bodies (D33). Only these fields are kept.
NEWS_KEPT_FIELDS = ("date", "title", "symbols", "sentiment", "link")


class EodhdError(Exception):
    """Base class for client errors. Messages never contain the token."""


class OfflineCacheMissError(EodhdError):
    """Raised when offline mode is on and the response is not cached."""


class EodhdHttpError(EodhdError):
    """Raised for a 4xx response or a 429/5xx that survived all retries."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class NewsSymbolUnsupportedError(EodhdHttpError):
    """Raised when /news or /sentiments rejects a symbol as invalid (HTTP 422).

    EODHD does this for reused-ticker codes such as ``APC_old``. News for the
    symbol is unknown, which is not the same as zero articles (D26).
    """


class EodhdPlanLimitError(EodhdError):
    """Raised when EODHD answers 200 but sends a plan warning instead of data."""


class NewsWindowTooLargeError(EodhdError):
    """Raised when a news window has more articles than one query can page."""


class NewsTruncatedError(EodhdError):
    """Raised when max_pages stopped a news window on a full page."""


class EodhdBadResponseError(EodhdError):
    """Raised for an error body or a wrong response shape. Never cached."""


class EodhdEmptyResponseError(EodhdError):
    """Raised when /eod returns no rows for the requested range. Never cached."""


# Parameters that carry a date. Outside a holdout session none may reach the holdout.
DATE_PARAMS = ("from", "to", "effective")
ERROR_KEYS = ("errors", "error", "message")


def cache_path(path: str, params: dict[str, Any], cache_dir: Path) -> Path:
    """Return the cache file for an endpoint and params. The token is never part of it."""
    clean = {k: v for k, v in params.items() if k not in ("api_token", "fmt")}
    key = "_".join(f"{k}={clean[k]}" for k in sorted(clean)) or "all"
    safe_key = re.sub(r"[^A-Za-z0-9._=,-]", "-", key)
    endpoint = path.strip("/")
    return cache_dir / endpoint / f"{safe_key}.json"


def announce_mode() -> None:
    """Print a loud warning to stderr when the data is fake."""
    if data_mode() != "fake":
        return
    lines = [
        "SIMULATED DATA: EODHD_MODE=fake (or no EODHD_API_KEY is set).",
        "Every EODHD response is synthetic. Results are NOT real.",
    ]
    width = max(len(line) for line in lines) + 4
    print(
        "\n".join(["*" * width, *(f"* {x:<{width - 4}} *" for x in lines), "*" * width]),
        file=sys.stderr,
    )


def _effective_cache_dir(cache_dir: Path, api_token: str | None) -> Path:
    """Swap the default real cache for the fake one in fake mode."""
    fake = bool(api_token and api_token.startswith("fake-")) or data_mode(api_token) == "fake"
    if cache_dir == DEFAULT_CACHE_DIR and fake:
        return FAKE_CACHE_DIR
    return cache_dir


def _http_get(url: str, query: dict[str, Any], token: str) -> Any:
    """The only place the fake transport replaces the network."""
    # A fake-* token always means the fake transport, so explicit fake plans work.
    if token.startswith("fake-") or data_mode(token) == "fake":
        return FakeEodhdTransport().get(url, params=query, timeout=REQUEST_TIMEOUT_SECONDS)
    return requests.get(url, params=query, timeout=REQUEST_TIMEOUT_SECONDS)


def _resolve_token(api_token: str | None) -> str:
    if not api_token and data_mode() == "fake":
        return default_fake_token()
    token = api_token or load_secrets().eodhd_api_key
    if not token:
        raise EodhdError("No EODHD token. Set EODHD_API_KEY or pass api_token='demo' explicitly.")
    return token


def _http_error(status: int, path: str) -> EodhdHttpError:
    if status == 403:
        text = f"HTTP 403 on {path}: the plan or token does not cover this endpoint."
    else:
        text = f"HTTP {status} on {path}."
    return EodhdHttpError(status, text)


def _network_error(path: str, name: str) -> EodhdError:
    """Build the error outside the except block so no cause or context is attached."""
    error = EodhdError(f"Network error on {path}: {name}")
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error


def _get_with_retry(path: str, params: dict[str, Any], token: str) -> Any:
    query = {**params, "api_token": token, "fmt": "json"}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        failure: str | None = None
        response = None
        try:
            response = _http_get(BASE_URL + path, query, token)
        except requests.RequestException as exc:
            # The exception text can include the full URL and so the token.
            failure = type(exc).__name__
        if failure is not None:
            if attempt == MAX_ATTEMPTS:
                raise _network_error(path, failure)
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
            continue
        assert response is not None
        status = response.status_code
        retryable = status == 429 or status >= 500
        if retryable and attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
            continue
        if status == 422 and _is_invalid_symbol_body(response, path):
            raise NewsSymbolUnsupportedError(status, f"HTTP 422 on {path}: invalid ticker symbol.")
        if status >= 400:
            raise _http_error(status, path)
        return _parse_json(response, path)
    raise EodhdError(f"Unreachable retry state on {path}")


def _is_invalid_symbol_body(response: Any, path: str) -> bool:
    """True for the 422 body that says the symbol is not a valid ticker, on news endpoints."""
    if not path.startswith(("/news", "/sentiments")):
        return False
    try:
        text = json.dumps(response.json())
    except ValueError:
        text = str(getattr(response, "text", ""))
    return "valid ticker symbol" in text


def _parse_json(response: Any, path: str) -> Any:
    try:
        return response.json()
    except ValueError:
        pass
    raise EodhdError(f"Non-JSON response on {path}")


def _raise_on_plan_warning(data: Any, path: str) -> None:
    """Fail on a plan warning disguised as data.

    The free plan answers HTTP 200 with [{"warning": "..."}] for dates it
    does not cover. Caching that would look like a real, empty history.
    """
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if isinstance(row, dict) and "warning" in row:
            raise EodhdPlanLimitError(f"EODHD plan warning on {path}: {row['warning']}")


def _expect(data: Any, kind: type, path: str) -> None:
    if not isinstance(data, kind):
        raise EodhdBadResponseError(
            f"Unexpected response on {path}: wanted {kind.__name__}, got {type(data).__name__}."
        )


def _validate_shape(path: str, params: dict[str, Any], data: Any) -> None:
    """Reject error bodies and wrong shapes before anything is cached."""
    if isinstance(data, dict):
        for key in ERROR_KEYS:
            if key in data:
                raise EodhdBadResponseError(f"EODHD error body on {path}: {key}={data[key]!r}")
    if path.startswith("/sentiments"):
        _expect(data, dict, path)
        for symbol in str(params.get("s", "")).split(","):
            if symbol not in data:
                raise EodhdBadResponseError(f"No {symbol} key in the response on {path}.")
    elif path.startswith("/calendar/earnings"):
        _expect(data, dict, path)
        _expect(data.get("earnings"), list, path)
    elif path.startswith("/fundamentals"):
        _expect(data, dict, path)
    elif path.split("/")[1] in LIST_ENDPOINTS:
        _expect(data, list, path)
    if path.startswith("/eod/") and data == []:
        raise EodhdEmptyResponseError(
            f"No price rows on {path} for {params.get('from')}..{params.get('to')}."
        )


LIST_ENDPOINTS = ("eod", "splits", "div", "news", "exchange-symbol-list", "symbol-change-history")


def _download(path: str, params: dict[str, Any], token: str) -> Any:
    """One network request, checked for plan warnings, error bodies and shape."""
    data = _get_with_retry(path, params, token)
    _raise_on_plan_warning(data, path)
    _validate_shape(path, params, data)
    return data


def _guard_params(params: dict[str, Any]) -> None:
    """Refuse any date parameter in the holdout unless a holdout session is open."""
    if active_session_strategy() is not None:
        return
    for key in DATE_PARAMS:
        value = params.get(key)
        if value and row_calendar_date(str(value)[:10]) >= HOLDOUT_START:
            raise HoldoutAccessError(
                f"Parameter {key}={value} is in the holdout. Only the approval flow may fetch it."
            )


# Holdout cache folders used by the open session. Deleted when it closes.
_holdout_dirs: set[Path] = set()


def _holdout_dir(base: Path) -> Path:
    """The holdout cache folder that sits beside a normal cache folder."""
    folder = base.with_name(base.name + "-holdout")
    _holdout_dirs.add(folder)
    return folder


def purge_holdout_caches() -> None:
    """Delete every holdout cache folder. Runs when a holdout session closes."""
    while _holdout_dirs:
        shutil.rmtree(_holdout_dirs.pop(), ignore_errors=True)


register_session_close_hook(purge_holdout_caches)


# Forward-data cache (D23). Paper sessions read and write it. Nothing else does.
# It sits beside the normal cache and is not deleted when a session closes.
FORWARD_SUFFIX = "-forward"
# Entries without a date range are fetched again after this many seconds.
# Membership changes, so it is refreshed daily. Splits are rare.
# Splits and dividends are refreshed daily too (H3): a split on a held stock
# must reach the engine before the day is stepped.
PLAIN_MAX_AGE_SECONDS = {"/fundamentals": 24 * 3600, "/splits/": 24 * 3600}
PLAIN_DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 3600
# Each update fetches this many trading days before the last covered row again.
REFETCH_TRADING_DAYS = 5
# Data for a day counts as complete from this New York time (H2).
SETTLE_TIME = clock_time(18, 0)
# (row date field) for endpoints whose rows are a date range that can be extended.
RANGE_DATE_FIELDS = {
    "/eod/": "date",
    "/div/": "date",
    "/news": "date",
    "/sentiments": "date",
    "/calendar/earnings": "report_date",
}

Fetcher = Callable[[str, dict[str, Any]], Any]


def _forward_dir(base: Path) -> Path:
    """The forward cache folder that sits beside a normal cache folder."""
    return base.with_name(base.name + FORWARD_SUFFIX)


def _range_field(path: str) -> str | None:
    for prefix, field in RANGE_DATE_FIELDS.items():
        if path.startswith(prefix):
            return field
    return None


def _split_rows(path: str, params: dict[str, Any], data: Any) -> tuple[list[Any], Any]:
    """Return the rows of a range response and a template to rebuild it."""
    if path.startswith("/sentiments"):
        return list(data.get(str(params["s"]), [])), data
    if path.startswith("/calendar/earnings"):
        return list(data.get("earnings", [])), data
    return list(data), None


def _join_rows(path: str, params: dict[str, Any], template: Any, rows: list[Any]) -> Any:
    if path.startswith("/sentiments"):
        return {**template, str(params["s"]): rows}
    if path.startswith("/calendar/earnings"):
        return {**template, "earnings": rows}
    return rows


def _within(rows: list[Any], field: str, first: date, last: date) -> list[Any]:
    return [row for row in rows if first <= row_calendar_date(row[field]) <= last]


def _forward_load(
    path: str,
    params: dict[str, Any],
    *,
    base: Path,
    offline: bool,
    api_token: str | None,
    fetch: Fetcher,
) -> Any:
    """Paper-session read. Serve from the forward cache and fetch only what is missing.

    A range entry remembers the last row date it actually received, never the
    requested ``to`` (H2). An update fetches again from five trading days
    before that date and replaces the rows in that window, so a bar or an
    article published late is picked up. Today counts as settled only after
    18:00 New York time. An update that asks for what an earlier settled
    fetch already covered makes no request. An entry without a date range is
    fetched again when it is older than its maximum age.
    """
    field = _range_field(path)
    if field is None or "from" not in params or "to" not in params:
        return _forward_plain(path, params, base, offline, api_token, fetch)
    series = {k: v for k, v in params.items() if k not in ("from", "to")}
    file_path = cache_path(path, series, _forward_dir(base))
    want_from, want_to = date.fromisoformat(params["from"]), date.fromisoformat(params["to"])
    as_of = _FORWARD_AS_OF.get() or want_to
    entry = json.loads(file_path.read_text()) if file_path.exists() else None
    if entry is not None and "settled" not in entry:
        entry = None  # written before H2: its "to" was the requested day, so it is not trusted
    if entry is not None and date.fromisoformat(entry["from"]) <= want_from:
        covered = date.fromisoformat(entry["requested_to"]) >= want_to
        if not (covered and date.fromisoformat(entry["settled"]) >= as_of):
            if offline:
                raise OfflineCacheMissError(f"Offline and not cached: {file_path}")
            entry = _refresh_entry(path, params, field, entry, as_of, api_token, fetch)
            _write_json(file_path, entry)
        rows, template = _split_rows(path, params, entry["data"])
        return _join_rows(path, params, template, _within(rows, field, want_from, want_to))
    if offline:
        raise OfflineCacheMissError(f"Offline and not cached: {file_path}")
    data = fetch(_resolve_token(api_token), params)
    rows, _ = _split_rows(path, params, data)
    _write_json(file_path, _new_entry(field, params["from"], params["to"], as_of, rows, data))
    return data


def _new_entry(
    field: str, first: str, requested_to: str, as_of: date, rows: list[Any], data: Any
) -> dict[str, Any]:
    """A forward cache entry. ``to`` is the last row date received, or None for no rows."""
    last = max((row_calendar_date(row[field]) for row in rows), default=None)
    return {
        "from": first,
        "to": None if last is None else last.isoformat(),
        "requested_to": requested_to,
        "settled": _settled_day(as_of, _utcnow()).isoformat(),
        "data": data,
    }


def _refresh_entry(
    path: str,
    params: dict[str, Any],
    field: str,
    entry: dict[str, Any],
    as_of: date,
    api_token: str | None,
    fetch: Fetcher,
) -> dict[str, Any]:
    """Fetch again from five trading days before the last covered row and merge the result."""
    rows, template = _split_rows(path, params, entry["data"])
    first = date.fromisoformat(entry["from"])
    fetch_to = max(date.fromisoformat(params["to"]), date.fromisoformat(entry["requested_to"]))
    anchor = first if entry["to"] is None else date.fromisoformat(entry["to"])
    anchor = min(anchor, date.fromisoformat(entry["settled"]))
    refetch_from = min(max(first, trading_days_back(anchor, REFETCH_TRADING_DAYS)), fetch_to)
    kept = [row for row in rows if row_calendar_date(row[field]) < refetch_from]
    tail_params = {**params, "from": refetch_from.isoformat(), "to": fetch_to.isoformat()}
    try:
        tail = fetch(_resolve_token(api_token), tail_params)
    except EodhdEmptyResponseError:
        tail = None
    if tail is not None:
        tail_rows, template = _split_rows(path, params, tail)
        kept = kept + tail_rows
    data = _join_rows(path, params, template, kept)
    return _new_entry(field, entry["from"], fetch_to.isoformat(), as_of, kept, data)


def trading_days_back(day: date, count: int) -> date:
    """The day ``count`` NYSE sessions before ``day`` (weekends and standard holidays skipped)."""
    holidays: set[date] = set()
    for year in range(day.year - 1, day.year + 1):
        holidays |= nyse_holidays(year)
    current = day
    remaining = count
    while remaining > 0:
        current -= timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            remaining -= 1
    return current


def _utcnow() -> datetime:
    """The clock. Tests replace it."""
    return datetime.now(UTC)


def _settled_day(as_of: date, now: datetime) -> date:
    """``as_of`` if the day's data is settled at ``now`` (18:00 New York), else the day before."""
    closes = datetime.combine(as_of, SETTLE_TIME, tzinfo=NEW_YORK)
    return as_of if now >= closes else as_of - timedelta(days=1)


_FORWARD_AS_OF: ContextVar[date | None] = ContextVar("forward_as_of", default=None)


@contextmanager
def forward_as_of(day: date) -> Iterator[None]:
    """Tell the forward cache which day the paper update treats as today."""
    token = _FORWARD_AS_OF.set(day)
    try:
        yield
    finally:
        _FORWARD_AS_OF.reset(token)


def _forward_plain(
    path: str,
    params: dict[str, Any],
    base: Path,
    offline: bool,
    api_token: str | None,
    fetch: Fetcher,
) -> Any:
    file_path = cache_path(path, params, _forward_dir(base))
    limit = next(
        (age for prefix, age in PLAIN_MAX_AGE_SECONDS.items() if path.startswith(prefix)),
        PLAIN_DEFAULT_MAX_AGE_SECONDS,
    )
    cached = file_path.exists()
    if cached and (offline or time.time() - file_path.stat().st_mtime <= limit):
        return json.loads(file_path.read_text())
    if offline:
        raise OfflineCacheMissError(f"Offline and not cached: {file_path}")
    data = fetch(_resolve_token(api_token), params)
    _write_json(file_path, data)
    return data


def _write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data))


def _cached_load(
    path: str,
    params: dict[str, Any],
    *,
    cache_dir: Path,
    offline: bool,
    api_token: str | None,
    fetch: Fetcher,
    sanitize: Callable[[Any], Any] | None = None,
    skip_normal_cache_in_session: bool = False,
) -> Any:
    """Guard, then read the cache, else fetch and cache the data.

    Outside a session only the normal cache is used. In a holdout session
    misses go to a holdout folder that is deleted when the session closes.
    In a paper session they go to the forward cache, which survives (D23).
    ``skip_normal_cache_in_session`` is for data that the normal cache holds
    in a clipped form, such as index membership: a session never reads it.
    """
    _guard_params(params)
    base = _effective_cache_dir(cache_dir, api_token)
    in_session = active_session_strategy() is not None
    file_path = cache_path(path, params, base)
    if not (in_session and skip_normal_cache_in_session) and file_path.exists():
        return json.loads(file_path.read_text())
    if in_session and is_paper_session():
        return _forward_load(
            path, params, base=base, offline=offline, api_token=api_token, fetch=fetch
        )
    if in_session:
        file_path = cache_path(path, params, _holdout_dir(base))
        if file_path.exists():
            return json.loads(file_path.read_text())
    if offline:
        raise OfflineCacheMissError(f"Offline and not cached: {file_path}")
    data = fetch(_resolve_token(api_token), params)
    if sanitize is not None and not in_session:
        data = sanitize(data)
    _write_json(file_path, data)
    return data


def _fetch_json(
    path: str,
    params: dict[str, Any],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
    sanitize: Callable[[Any], Any] | None = None,
    skip_normal_cache_in_session: bool = False,
) -> Any:
    """Private: return the JSON for an endpoint. Use the wrappers, not this.

    Refuses holdout date params outside a session. Sanitize runs before the
    data is cached (outside a session), for endpoints that cannot be date-limited.
    """
    return _cached_load(
        path,
        params,
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
        fetch=lambda token, query: _download(path, query, token),
        sanitize=sanitize,
        skip_normal_cache_in_session=skip_normal_cache_in_session,
    )


UNAVAILABLE_DIR = "news-unavailable"


def _unavailable_marker(ticker: str, cache_dir: Path, api_token: str | None) -> Path:
    base = _effective_cache_dir(cache_dir, api_token)
    return base / UNAVAILABLE_DIR / (re.sub(r"[^A-Za-z0-9._=,-]", "-", ticker) + ".json")


def is_news_unavailable(
    ticker: str, *, cache_dir: Path = DEFAULT_CACHE_DIR, api_token: str | None = None
) -> bool:
    """True when a marker says EODHD rejects this symbol for news and sentiment."""
    return _unavailable_marker(ticker, cache_dir, api_token).exists()


def _mark_news_unavailable(ticker: str, cache_dir: Path, api_token: str | None) -> None:
    """Cache a marker, never an empty news list, so reruns cost no calls."""
    _write_json(
        _unavailable_marker(ticker, cache_dir, api_token),
        {"news_unavailable": True, "symbol": _symbol(ticker)},
    )


def _raise_if_marked(ticker: str, cache_dir: Path, api_token: str | None) -> None:
    if _unavailable_marker(ticker, cache_dir, api_token).exists():
        raise NewsSymbolUnsupportedError(
            422, f"News for {_symbol(ticker)} is unavailable (cached marker)."
        )


def _symbol(ticker: str) -> str:
    """Add the US exchange suffix. The only place this happens."""
    return f"{ticker}.US"


def _price_symbol(ticker: str) -> str:
    """The symbol for prices, splits and dividends. Renamed stocks use their new code (D36)."""
    return _symbol(price_code(ticker))


def get_eod_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Daily bars: raw OHLC, split-adjusted volume, adjusted_close. 1 API call."""
    assert_before_holdout(end, approved=approved)
    params = {"from": start.isoformat(), "to": end.isoformat()}
    rows = _fetch_json(
        f"/eod/{_price_symbol(ticker)}",
        params,
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return drop_holdout_rows(rows, "date", approved=approved)


def get_splits(
    ticker: str,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """All splits up to the day before the holdout. 1 API call.

    The request carries ``to`` so holdout splits never reach the cache. Only
    an approved holdout session asks for the full history.
    """
    unsealed = approved and active_session_strategy() is not None
    last_day = HOLDOUT_START - timedelta(days=1)
    rows = _fetch_json(
        f"/splits/{_price_symbol(ticker)}",
        {} if unsealed else {"to": last_day.isoformat()},
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return drop_holdout_rows(rows, "date", approved=approved)


def get_dividends(
    ticker: str,
    start: date,
    end: date,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Dividends with ex-date in the range. 1 API call."""
    assert_before_holdout(end, approved=approved)
    params = {"from": start.isoformat(), "to": end.isoformat()}
    rows = _fetch_json(
        f"/div/{_price_symbol(ticker)}",
        params,
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return drop_holdout_rows(rows, "date", approved=approved)


def get_news(
    ticker: str,
    start: date,
    end: date,
    *,
    page_size: int = 1000,
    max_pages: int | None = None,
    allow_truncated: bool = False,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Articles tagged with the ticker, paged by offset until a short page.

    Each request costs 5 API calls plus 5 per ticker. The API allows
    offset + limit <= 10000. If a window fills that, this raises instead of
    returning a silently truncated list: call again with smaller windows
    (for example month by month). The same applies when ``max_pages`` stops
    on a full page, unless ``allow_truncated`` is True.

    All pages are fetched first and cached as one unit, so a later re-fetch
    cannot mix pages from different times. The cache key has no offset.
    """
    assert_before_holdout(end, approved=approved)
    params: dict[str, Any] = {
        "s": _symbol(ticker),
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    if max_pages is not None:
        # A capped window is a different object from a full one.
        params.update(limit=page_size, max_pages=max_pages)
    _raise_if_marked(ticker, cache_dir, api_token)
    try:
        articles: list[dict[str, Any]] = _cached_load(
            "/news",
            params,
            cache_dir=cache_dir,
            offline=offline,
            api_token=api_token,
            fetch=lambda token, query: _fetch_news_pages(
                query, ticker, page_size=page_size, max_pages=max_pages, token=token
            ),
        )
    except NewsSymbolUnsupportedError:
        _mark_news_unavailable(ticker, cache_dir, api_token)
        raise
    if max_pages is not None and len(articles) == max_pages * page_size and not allow_truncated:
        raise NewsTruncatedError(
            f"{ticker} {start}..{end} filled max_pages={max_pages} pages of {page_size}. "
            "Raise max_pages, use smaller windows, or pass allow_truncated=True."
        )
    return drop_holdout_rows(articles, "date", approved=approved)


def slim_news_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only date, title, symbols, sentiment and link. Drops ``content`` and ``tags``.

    The article body was 88% of the news cache and no code reads it (D33).
    Rows that are not dicts are passed through so the shape check still decides.
    """
    return [
        {k: row[k] for k in NEWS_KEPT_FIELDS if k in row} if isinstance(row, dict) else row
        for row in rows
    ]


def api_requests_used(*, api_token: str | None = None) -> int:
    """API calls used today, from ``GET /api/user``. That endpoint costs no calls."""
    body = _download("/user", {}, _resolve_token(api_token))
    if not isinstance(body, dict) or "apiRequests" not in body:
        raise EodhdBadResponseError("GET /user did not return apiRequests.")
    return int(body["apiRequests"])


def _fetch_news_pages(
    window: dict[str, Any], ticker: str, *, page_size: int, max_pages: int | None, token: str
) -> list[dict[str, Any]]:
    """Fetch every page of one news window, without caching single pages."""
    articles: list[dict[str, Any]] = []
    offset = 0
    pages = 0
    while max_pages is None or pages < max_pages:
        if offset + page_size > NEWS_MAX_WINDOW:
            raise NewsWindowTooLargeError(
                f"{ticker} {window['from']}..{window['to']} has more than "
                f"{NEWS_MAX_WINDOW} articles. Split the window into smaller date ranges."
            )
        query = {
            "s": window["s"],
            "from": window["from"],
            "to": window["to"],
            "limit": page_size,
            "offset": offset,
        }
        page = _download("/news", query, token)
        articles.extend(slim_news_rows(page))
        pages += 1
        if len(page) < page_size:
            break
        offset += page_size
    return articles


def get_daily_sentiment(
    ticker: str,
    start: date,
    end: date,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Daily sentiment rows {date, count, normalized}, newest first."""
    assert_before_holdout(end, approved=approved)
    symbol = _symbol(ticker)
    params = {"s": symbol, "from": start.isoformat(), "to": end.isoformat()}
    _raise_if_marked(ticker, cache_dir, api_token)
    try:
        data = _fetch_json(
            "/sentiments", params, cache_dir=cache_dir, offline=offline, api_token=api_token
        )
    except NewsSymbolUnsupportedError:
        _mark_news_unavailable(ticker, cache_dir, api_token)
        raise
    return drop_holdout_rows(data.get(symbol, []), "date", approved=approved)


def get_earnings(
    ticker: str,
    start: date,
    end: date,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Earnings rows. `report_date` is the announcement, `date` is the fiscal period end."""
    assert_before_holdout(end, approved=approved)
    params = {"symbols": _symbol(ticker), "from": start.isoformat(), "to": end.isoformat()}
    data = _fetch_json(
        "/calendar/earnings", params, cache_dir=cache_dir, offline=offline, api_token=api_token
    )
    return drop_holdout_rows(data.get("earnings", []), "report_date", approved=approved)


def _starts_before_holdout(row: dict[str, Any]) -> bool:
    """True if a membership row starts before the holdout.

    Live EODHD leaves StartDate empty for names that joined before its change
    history begins (161 of 911 rows on 2026-11-02, D25). Those are early
    members, so they are kept rather than treated as bad data.
    """
    start = row.get("StartDate")
    return not start or row_calendar_date(start) < HOLDOUT_START


def _clip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows that start in the holdout and null EndDate values in it."""
    kept = [row for row in rows if _starts_before_holdout(row)]
    clipped = []
    for row in kept:
        end = row.get("EndDate")
        if end and row_calendar_date(end) >= HOLDOUT_START:
            row = {**row, "EndDate": None}
        clipped.append(row)
    return clipped


def _clip_index_history(data: Any) -> Any:
    """Apply the holdout clip to a cached fundamentals dict, keeping its keys."""
    if not isinstance(data, dict):
        return data
    kept = {key: row for key, row in data.items() if _starts_before_holdout(row)}
    clipped = _clip_rows(list(kept.values()))
    return dict(zip(kept, clipped, strict=True))


def get_index_history(
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Raw S&P 500 membership rows (Code, StartDate, EndDate, IsDelisted, ...).

    This call cannot be date-limited. Outside a session, rows that start in
    the holdout are dropped and an EndDate in the holdout becomes None, before
    caching and again on return. Inside a session the normal (clipped) cache
    is never read: the full history is fetched into the session cache, so
    holdout and paper runs see membership changes from 2024 on (D23).
    Fundamentals may cost more than 1 call.
    """
    data = _fetch_json(
        f"/fundamentals/{INDEX_SYMBOL}",
        {"filter": "HistoricalTickerComponents"},
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
        sanitize=_clip_index_history,
        skip_normal_cache_in_session=True,
    )
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if approved and active_session_strategy() is not None:
        return rows
    return _clip_rows(rows)


def get_current_components(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Today's S&P 500 members. Only for cross-checks, never for historical universes."""
    data = _fetch_json(
        f"/fundamentals/{INDEX_SYMBOL}",
        {"filter": "Components"},
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return list(data.values()) if isinstance(data, dict) else list(data)


def get_delisted_symbols(
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """US symbols that are no longer listed. The rows carry no date, so none are dropped."""
    rows: list[dict[str, Any]] = _fetch_json(
        "/exchange-symbol-list/US",
        {"delisted": 1},
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return rows


def get_symbol_changes(
    start: date,
    end: date,
    *,
    approved: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Ticker renames with an effective date in the range."""
    assert_before_holdout(end, approved=approved)
    params = {"from": start.isoformat(), "to": end.isoformat(), "ex": "US"}
    rows = _fetch_json(
        "/symbol-change-history",
        params,
        cache_dir=cache_dir,
        offline=offline,
        api_token=api_token,
    )
    return drop_holdout_rows(rows, "effective", approved=approved)


announce_mode()
