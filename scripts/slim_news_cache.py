"""Rewrite cached news files to the slim form: date, title, symbols, sentiment, link.

Article bodies (`content`) and `tags` were most of the news cache (637 MB for
24 tickers) and no code reads them (D33). Safe to run again: a file that is
already slim is left alone. Prints the size before and after.

    uv run python scripts/slim_news_cache.py [--cache-dir data/cache/eodhd]
"""

import argparse
import json
from pathlib import Path

from bot.data.eodhd_client import DEFAULT_CACHE_DIR, slim_news_rows


def slim_file(path: Path) -> bool:
    """Slim one cached news file in place. Returns True if the file changed."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return False
    slim = slim_news_rows(data)
    if slim == data:
        return False
    path.write_text(json.dumps(slim))
    return True


def slim_cache(cache_dir: Path) -> tuple[int, int, int]:
    """Slim every file under ``<cache_dir>/news``. Returns (changed, bytes before, after)."""
    changed = before = after = 0
    for path in sorted((cache_dir / "news").glob("*.json")):
        size = path.stat().st_size
        before += size
        if slim_file(path):
            changed += 1
        after += path.stat().st_size
    return changed, before, after


def main() -> None:
    """Parse the flag, slim the cache and report sizes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    changed, before, after = slim_cache(args.cache_dir)
    print(f"Files rewritten: {changed}.")
    print(f"News cache size: {before / 1e6:,.1f} MB before, {after / 1e6:,.1f} MB after.")


if __name__ == "__main__":
    main()
