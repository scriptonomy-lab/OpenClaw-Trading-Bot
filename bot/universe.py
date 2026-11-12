"""Which tickers a run uses: a frozen random sample, or the full historical index.

The sample is drawn once (`scripts/draw_sample.py`), frozen in
`data/universe_sample.json` and committed, so every run uses the same names.
Point-in-time membership still applies: on each date the tradable universe is
the sample intersected with that day's real S&P 500 members (D33).
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bot.config import PROJECT_ROOT

SAMPLE_PATH = PROJECT_ROOT / "data" / "universe_sample.json"
UniverseMode = Literal["sample", "full"]


@dataclass(frozen=True)
class FrozenSample:
    """The frozen sample file: tickers, how they were drawn, and a hash."""

    tickers: tuple[str, ...]
    seed: int
    population: int
    strata: dict[str, int]
    sha256: str


def tickers_hash(tickers: list[str] | tuple[str, ...]) -> str:
    """Hash of the sorted ticker list, so a changed file is detected."""
    return hashlib.sha256(json.dumps(sorted(tickers)).encode()).hexdigest()


def load_sample(path: Path = SAMPLE_PATH) -> FrozenSample:
    """Read the frozen sample and refuse it if the tickers do not match the hash."""
    doc = json.loads(path.read_text())
    tickers = tuple(doc["tickers"])
    if tickers_hash(tickers) != doc["sha256"]:
        raise ValueError(f"{path} was changed after it was frozen (hash mismatch).")
    return FrozenSample(
        tickers=tickers,
        seed=int(doc["seed"]),
        population=int(doc["population"]),
        strata=dict(doc["strata"]),
        sha256=doc["sha256"],
    )


def universe_mode() -> UniverseMode:
    """`sample` unless EODHD_UNIVERSE=full is set. The full index is one flag away.

    In fake mode the default is `full`: the simulated market has its own
    made-up tickers, none of which are in the real frozen sample, so a sample
    there would be empty. An explicit EODHD_UNIVERSE always wins.
    """
    from bot.config import data_mode

    raw = os.getenv("EODHD_UNIVERSE", "").strip().lower()
    if not raw:
        return "full" if data_mode() == "fake" else "sample"
    value = raw
    if value not in ("sample", "full"):
        raise ValueError(f"EODHD_UNIVERSE must be 'sample' or 'full', got {value!r}.")
    return "full" if value == "full" else "sample"


def universe_label(mode: UniverseMode | None = None) -> str:
    """The sentence every results table, chart and message must carry."""
    if (mode or universe_mode()) == "full":
        return "Full universe of historical S&P 500 members."
    sample = load_sample()
    return (
        f"Random sample of {len(sample.tickers)} of {sample.population} historical "
        f"S&P 500 members (seed {sample.seed})."
    )


SAMPLE_LIMITATIONS = (
    "Smaller sample means fewer trades and wider uncertainty. "
    "Results describe the sample, not the full index."
)
