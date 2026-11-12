"""Renamed stocks: price aliases, overlap removal and the empty-dates rule (D36)."""

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from bot.backtest.engine import unheld_candidates
from bot.config import PROJECT_ROOT
from bot.data import eodhd_client as client
from bot.data.aliases import ALIAS_LAST_DAY, PRICE_ALIASES, company_key, price_code
from bot.data.membership import (
    MembershipSpell,
    drop_alias_overlaps,
    parse_membership_rows,
)
from tests.test_membership import row


def test_price_code_maps_old_codes_and_leaves_others() -> None:
    assert price_code("FB") == "META"
    assert price_code("AAPL") == "AAPL"
    assert "UA-C" not in PRICE_ALIASES


def test_prices_splits_and_dividends_use_the_alias_but_news_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[str] = []

    def fake_download(path: str, params: dict[str, Any], token: str) -> Any:
        paths.append(path)
        return [{"date": "2018-01-02", "title": "t", "symbols": []}]

    monkeypatch.setattr(client, "_download", fake_download)
    start, end = date(2018, 1, 1), date(2018, 1, 31)
    kwargs: dict[str, Any] = {"cache_dir": tmp_path, "api_token": "x"}
    client.get_eod_prices("FB", start, end, **kwargs)
    client.get_splits("FB", **kwargs)
    client.get_dividends("FB", start, end, **kwargs)
    client.get_news("FB", start, end, **kwargs)
    assert paths[:3] == ["/eod/META.US", "/splits/META.US", "/div/META.US"]
    assert paths[3] == "/news"
    assert any(p.name.startswith("from=2018-01-01_s=FB.US") for p in (tmp_path / "news").iterdir())


def spells(rows: list[Any]) -> list[tuple[str, date, date | None]]:
    return [(s.symbol, s.start, s.end) for s in parse_membership_rows(rows)]


def test_swap_day_is_held_once() -> None:
    result = spells(
        [row("FB", "2013-12-23", "2022-06-09"), row("META", "2022-06-09", None, active=1)]
    )
    assert ("FB.US", date(2013, 12, 23), date(2022, 6, 8)) in result
    assert ("META.US", date(2022, 6, 9), None) in result


def test_fully_covered_alias_spell_disappears() -> None:
    result = spells([row("ACE", "2010-07-14", "2016-01-15"), row("CB", "2010-07-14", "2016-01-15")])
    assert [s[0] for s in result] == ["CB.US"]


def test_alias_spell_with_no_overlap_is_unchanged() -> None:
    result = spells([row("FB", "2013-12-23", "2020-01-01"), row("META", "2022-06-09", None)])
    assert ("FB.US", date(2013, 12, 23), date(2020, 1, 1)) in result


def test_cut_in_the_middle_leaves_two_pieces() -> None:
    old = MembershipSpell("FB.US", date(2015, 1, 1), date(2020, 1, 1))
    target = MembershipSpell("META.US", date(2017, 1, 1), date(2017, 12, 31))
    pieces = drop_alias_overlaps([old, target])
    assert [(p.start, p.end) for p in pieces if p.symbol == "FB.US"] == [
        (date(2015, 1, 1), date(2016, 12, 31)),
        (date(2018, 1, 1), date(2020, 1, 1)),
    ]


def test_row_with_no_dates_and_not_active_is_dropped() -> None:
    result = spells([row("MRP", None, None, active=0), row("OLDIE", "2012-01-01", None, active=1)])
    assert [s[0] for s in result] == ["OLDIE.US"]


def test_early_member_with_no_dates_that_is_active_keeps_the_d25_rule() -> None:
    result = spells([row("OLDCO", None, None, active=1)])
    assert result == [("OLDCO.US", date(1900, 1, 1), None)]


def test_empty_start_with_an_end_is_still_an_early_member() -> None:
    result = spells([row("GONE", None, "2012-05-01", active=0)])
    assert result == [("GONE.US", date(1900, 1, 1), date(2012, 5, 1))]


def test_reused_code_after_the_rename_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # LB became BBWI in 2021. A new LB spell in 2025 is another company.
    with caplog.at_level("WARNING"):
        result = drop_alias_overlaps([MembershipSpell("LB.US", date(2025, 3, 1), None)])
    assert result == []
    assert "reused" in caplog.text


def test_spell_starting_on_the_last_day_is_not_a_reused_code() -> None:
    result = drop_alias_overlaps([MembershipSpell("FB.US", date(2022, 6, 8), date(2022, 6, 8))])
    assert len(result) == 1


def test_company_key_is_the_same_before_and_after_a_rename() -> None:
    assert company_key("FB.US") == company_key("META.US")
    assert company_key("AAPL.US") != company_key("META.US")
    assert company_key("BRK.B.US") == "BRK.B.US"


def test_held_old_code_blocks_the_new_code() -> None:
    assert unheld_candidates(["META.US", "AAPL.US"], {"FB.US"}) == ["AAPL.US"]
    assert unheld_candidates(["META.US"], set()) == ["META.US"]
    # Sold at this open (same_open) counts as held, so no round trip on one company.
    assert unheld_candidates(["META.US"], {"FB.US"} | set()) == []


def test_ir_and_wab_start_on_their_index_join_dates() -> None:
    result = spells([row("IR", None, None, active=1), row("WAB", None, None, active=1)])
    assert ("IR.US", date(2020, 3, 3), None) in result
    assert ("WAB.US", date(2019, 2, 27), None) in result


def test_cached_alias_spells_all_end_on_or_before_their_last_day() -> None:
    path = (
        PROJECT_ROOT
        / "data/cache/eodhd/fundamentals/GSPC.INDX/filter=HistoricalTickerComponents.json"
    )
    if not path.exists():
        pytest.skip("no cached membership file")
    import json

    spells = parse_membership_rows(list(json.loads(path.read_text()).values()))
    for spell in spells:
        last_day = ALIAS_LAST_DAY.get(spell.symbol.removesuffix(".US"))
        if last_day is not None:
            assert spell.end is not None and spell.end <= last_day, spell


def test_cached_empty_dates_rows_dropped_are_exactly_mrp_and_ral() -> None:
    path = (
        PROJECT_ROOT
        / "data/cache/eodhd/fundamentals/GSPC.INDX/filter=HistoricalTickerComponents.json"
    )
    if not path.exists():
        pytest.skip("no cached membership file")
    import json

    rows = list(json.loads(path.read_text()).values())
    kept = {s.symbol for s in parse_membership_rows(rows)}
    no_dates = {r["Code"]: r for r in rows if not r["StartDate"] and not r["EndDate"]}
    assert {c for c in no_dates if f"{c}.US" not in kept} == {"MRP", "RAL"}
    assert {"IR.US", "WAB.US"} <= kept
    starts = {
        s.symbol: s.start for s in parse_membership_rows(rows) if s.symbol in ("IR.US", "WAB.US")
    }
    assert starts == {"IR.US": date(2020, 3, 3), "WAB.US": date(2019, 2, 27)}


@pytest.mark.parametrize("timing", ["same_open", "next_open"])
def test_engine_never_holds_one_company_twice_across_a_rename(timing: str) -> None:
    """FB and META share one price series. A held FB must block META on the swap day."""
    import pandas as pd

    from tests.backtest_helpers import Bar, engine_market, flat_bar, run_toy, toy_days
    from tests.test_entry_timing_and_rank import gap_strategy

    days = toy_days(8)
    series = {
        0: flat_bar(100.0),
        1: Bar(105, 106, 104, 105),
        2: flat_bar(105.0),
        3: Bar(110.25, 111, 110, 110.25),
        4: flat_bar(110.25),
        5: flat_bar(110.25),
        6: flat_bar(110.25),
        7: flat_bar(110.25),
    }
    members = pd.DataFrame(False, index=days, columns=["FB.US", "META.US"])
    members.loc[days[:3], "FB.US"] = True
    members.loc[days[3:], "META.US"] = True
    market = engine_market(8, {"FB.US": series, "META.US": series}, membership=members)
    result = run_toy(gap_strategy(timing=timing, holding_days=5, max_positions=10), market)
    spans = [(t.symbol, t.entry_date, t.exit_date) for t in result.trades]
    for i, first in enumerate(spans):
        for second in spans[i + 1 :]:
            assert first[2] <= second[1] or second[2] <= first[1], (first, second)


def test_manual_start_date_ignores_an_older_row_for_the_same_code() -> None:
    # A refreshed file could add an old IR row that ended before 2020. It must keep D25.
    result = spells([row("IR", None, "2012-05-01", active=0), row("IR", None, None, active=1)])
    assert ("IR.US", date(1900, 1, 1), date(2012, 5, 1)) in result
    assert ("IR.US", date(2020, 3, 3), None) in result
