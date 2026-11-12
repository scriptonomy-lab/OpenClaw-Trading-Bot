"""Draw and freeze the random ticker sample (D33). Run once; the file is committed.

Population: every ticker that was an S&P 500 member at any time from
2016-01-01 to 2023-12-31, including removed and delisted names. The draw is
stratified so the shares of still-active, removed and delisted names match
the population (largest-remainder rounding). Run with:
    uv run python scripts/draw_sample.py --seed 20261102 --size 200
"""

import argparse
import json
import random
from datetime import UTC, datetime
from typing import Any

from bot.data import eodhd_client as client
from bot.universe import SAMPLE_PATH, tickers_hash

WINDOW_START = "2016-01-01"
WINDOW_END = "2023-12-31"


def population_by_status(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group every 2016-2023 member code as active, removed or delisted."""
    spells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        start = row.get("StartDate") or "1900-01-01"
        end = row.get("EndDate") or "9999-12-31"
        if start <= WINDOW_END and end >= WINDOW_START:
            spells.setdefault(row["Code"], []).append(row)
    groups: dict[str, list[str]] = {"active": [], "removed": [], "delisted": []}
    for code, code_rows in spells.items():
        # Status uses today's flags, which is fine for choosing a sample: it
        # never reaches a signal (membership dates do that).
        if any(r.get("IsDelisted") for r in code_rows):
            groups["delisted"].append(code)
        elif any(r.get("IsActiveNow") for r in code_rows):
            groups["active"].append(code)
        else:
            groups["removed"].append(code)
    return {name: sorted(codes) for name, codes in groups.items()}


def stratum_sizes(groups: dict[str, list[str]], size: int) -> dict[str, int]:
    """Split the sample size across groups in proportion, by largest remainder."""
    total = sum(len(codes) for codes in groups.values())
    exact = {name: size * len(codes) / total for name, codes in groups.items()}
    sizes = {name: int(value) for name, value in exact.items()}
    by_remainder = sorted(exact, key=lambda name: exact[name] - sizes[name], reverse=True)
    for name in by_remainder[: size - sum(sizes.values())]:
        sizes[name] += 1
    return sizes


def main() -> None:
    """Draw the sample with the given seed and write the frozen file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    if SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} already exists. The sample is frozen; do not redraw.")
    groups = population_by_status(list(client.get_index_history(offline=True)))
    sizes = stratum_sizes(groups, args.size)
    rng = random.Random(args.seed)
    tickers = sorted(code for name in groups for code in rng.sample(groups[name], sizes[name]))
    doc = {
        "seed": args.seed,
        "population": sum(len(codes) for codes in groups.values()),
        "population_window": [WINDOW_START, WINDOW_END],
        "strata": sizes,
        "population_strata": {name: len(codes) for name, codes in groups.items()},
        "drawn_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tickers": tickers,
        "sha256": tickers_hash(tickers),
    }
    SAMPLE_PATH.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"Wrote {SAMPLE_PATH}: {len(tickers)} tickers, strata {sizes}, seed {args.seed}")


if __name__ == "__main__":
    main()
