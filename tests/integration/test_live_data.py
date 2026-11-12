"""Live price and shape checks (pytest -m integration). Each test costs a few API calls."""

from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

from bot.config import load_secrets
from bot.data import eodhd_client as client
from bot.data.prices import RawEodRow, RawSplitRow, build_price_frame, parse_split_rows
from bot.verify_live import Run, check_endpoints

pytestmark = pytest.mark.integration


def test_aapl_four_for_one_split_is_adjusted() -> None:
    """AAPL split 4:1 on 2020-08-31: raw close drops about 4x, our adjusted close does not."""
    rows = client.get_eod_prices("AAPL", date(2020, 8, 25), date(2020, 9, 2))
    splits = parse_split_rows(cast(list[RawSplitRow], client.get_splits("AAPL")))
    assert any(s.ex_date == date(2020, 8, 31) and s.factor == 4.0 for s in splits)
    raw = {row["date"]: row for row in rows}
    assert raw["2020-08-28"]["close"] / raw["2020-08-31"]["close"] > 3.0
    frame = build_price_frame(cast(list[RawEodRow], rows), splits, [])
    jump = frame["close"].loc["2020-08-31"] / frame["close"].loc["2020-08-28"] - 1
    assert abs(jump) < 0.15


def test_real_shapes_match_the_fakes(tmp_path: Path) -> None:
    """Every real response has the same keys and value types as the fake example."""
    run = Run(
        dry_run=False,
        assume_yes=False,
        token=load_secrets().eodhd_api_key,
        stamp="integration",
        out_dir=tmp_path,
    )
    results = check_endpoints(run)
    bad: dict[str, Any] = {r.spec.key: [d.line() for d in r.diffs] for r in results if r.fatal}
    assert not bad, bad
