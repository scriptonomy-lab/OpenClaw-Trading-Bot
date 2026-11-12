"""News is cached without article bodies (D33), and the slim script is safe to repeat."""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from bot.data import eodhd_client as client

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "slim_news_cache.py"
FULL_ROW = {
    "date": "2023-01-04T15:00:00+00:00",
    "title": "Apple news",
    "content": "long body " * 100,
    "link": "https://x/1",
    "symbols": ["AAPL.US"],
    "tags": ["EARNINGS"],
    "sentiment": {"polarity": 0.1, "neg": 0.0, "neu": 0.9, "pos": 0.1},
}


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EODHD_MODE", "fake")
    monkeypatch.setattr(client, "FAKE_CACHE_DIR", tmp_path / "cache")


def test_slim_rows_keep_only_five_fields() -> None:
    (row,) = client.slim_news_rows([FULL_ROW])
    assert set(row) == {"date", "title", "symbols", "sentiment", "link"}


def test_cached_news_has_no_content(tmp_path: Path) -> None:
    articles = client.get_news("SIM005", date(2022, 1, 1), date(2022, 3, 31))
    assert articles
    assert all("content" not in a and "tags" not in a for a in articles)
    (cached,) = (tmp_path / "cache" / "news").glob("*.json")
    assert "content" not in cached.read_text()


def load_slimmer() -> Any:
    spec = importlib.util.spec_from_file_location("slim_news_cache_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_slim_script_shrinks_and_is_idempotent(tmp_path: Path) -> None:
    news = tmp_path / "news"
    news.mkdir()
    file = news / "from=2015-01-01_s=AAPL.US_to=2023-12-31.json"
    file.write_text(json.dumps([FULL_ROW, FULL_ROW]))
    slimmer = load_slimmer()
    changed, before, after = slimmer.slim_cache(tmp_path)
    assert changed == 1 and after < before / 2
    first = file.read_text()
    changed, before_again, after_again = slimmer.slim_cache(tmp_path)
    assert changed == 0 and before_again == after_again
    assert file.read_text() == first
    assert "content" not in first
