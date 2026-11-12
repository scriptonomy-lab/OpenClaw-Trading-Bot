"""Rescore saved eval transcripts with the current scorer, without calling any model.

Use this after a scorer fix. It prints the original and the new accuracy side
by side and never overwrites the original results. Run with:
    uv run python evals/rescore.py outputs/simulated/evals/<run folder> [...]
"""

import json
import sys
from pathlib import Path
from typing import Any

from evals.run_evals import load_cases
from evals.scoring import accuracy, score_case


def rescore_folder(folder: Path) -> dict[str, Any]:
    """Score every transcript in a run folder and return old and new accuracy."""
    cases = {case["id"]: case for case in load_cases()}
    transcripts = [
        json.loads(line) for line in (folder / "transcripts.jsonl").read_text().splitlines()
    ]
    new_results = [score_case(cases[t["case_id"]], t) for t in transcripts if t["case_id"] in cases]
    old = json.loads((folder / "results.json").read_text())
    return {"folder": folder.name, "old": old.get("accuracy"), "new": accuracy(new_results),
            "new_failures": [r["case_id"] for r in new_results if not r["passed"]]}  # fmt: skip


def main() -> int:
    """Rescore each folder given on the command line and write rescore.json next to it."""
    for name in sys.argv[1:]:
        folder = Path(name)
        report = rescore_folder(folder)
        (folder / "rescore.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
