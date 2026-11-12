"""Make one real call to each EODHD endpoint the project uses and print its shape.

Uses EODHD_API_KEY if set, otherwise EODHD's public "demo" token, which only
works for a handful of tickers such as AAPL.US. Run with:
    uv run python scripts/check_endpoints.py
"""

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("EODHD_API_KEY") or "demo"
BASE = "https://eodhd.com/api"

CHECKS: list[tuple[str, str, dict[str, Any]]] = [
    ("EOD prices", "/eod/AAPL.US", {"from": "2020-08-25", "to": "2020-09-02"}),
    ("EOD prices (SPY)", "/eod/SPY.US", {"from": "2020-08-25", "to": "2020-08-27"}),
    ("Splits", "/splits/AAPL.US", {}),
    ("Dividends", "/div/AAPL.US", {"from": "2023-01-01", "to": "2023-12-31"}),
    ("News", "/news", {"s": "AAPL.US", "from": "2016-01-04", "to": "2016-01-06", "limit": 3}),
    ("Sentiment", "/sentiments", {"s": "AAPL.US", "from": "2023-01-03", "to": "2023-01-05"}),
    (
        "Earnings calendar",
        "/calendar/earnings",
        {"symbols": "AAPL.US", "from": "2023-01-01", "to": "2023-12-31"},
    ),
    ("S&P 500 current components", "/fundamentals/GSPC.INDX", {"filter": "Components"}),
    (
        "S&P 500 membership history",
        "/fundamentals/GSPC.INDX",
        {"filter": "HistoricalTickerComponents"},
    ),
    (
        "S&P 500 snapshots",
        "/fundamentals/GSPC.INDX",
        {"historical": 1, "from": "2016-01-01", "to": "2016-01-31"},
    ),
    ("Delisted symbol list", "/exchange-symbol-list/US", {"delisted": 1}),
    (
        "Symbol change history",
        "/symbol-change-history",
        {"from": "2022-01-01", "to": "2022-03-01", "ex": "US"},
    ),
]


def describe(payload: Any) -> str:
    """Return a short description of a JSON payload's shape and first record."""
    if isinstance(payload, list):
        first = json.dumps(payload[0])[:300] if payload else "(empty)"
        return f"list of {len(payload)}; first = {first}"
    if isinstance(payload, dict):
        keys = list(payload)[:8]
        sample_key = keys[0] if keys else None
        sample = json.dumps(payload[sample_key])[:300] if sample_key else "(empty)"
        return f"dict with {len(payload)} keys {keys}; [{sample_key!r}] = {sample}"
    return f"{type(payload).__name__}: {str(payload)[:300]}"


def main() -> None:
    """Run every check and print status plus response shape."""
    print(f"Token: {'real key' if TOKEN != 'demo' else 'demo'}\n")
    for label, path, params in CHECKS:
        query = {"api_token": TOKEN, "fmt": "json", **params}
        response = requests.get(BASE + path, params=query, timeout=60)
        print(f"## {label}  GET {path}  {params}")
        print(f"HTTP {response.status_code}")
        try:
            print(describe(response.json()))
        except ValueError:
            print(f"non-JSON body: {response.text[:200]!r}")
        print()


if __name__ == "__main__":
    main()
