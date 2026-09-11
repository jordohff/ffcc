"""Patch ONLY the weekly-projections section of the published artifact's
board_data.json, leaving the main board (half_ppr/ppr player rows,
replacement constants, generated_at) untouched.

Why this exists as a separate script from build_board_artifact_data.py: the
main board's rows are MERGED with composite/consensus data (Dataroma, Scott
Barrett, etc. - see consensus.py) sourced from data/paid/, which is
deliberately gitignored (licensed third-party data, not ours to redistribute
even into a private clone - see .gitignore's own comment). An automated
environment that only has this project's git history (e.g. a scheduled cloud
run) will NEVER have data/paid/ populated, so it cannot regenerate the main
board's consensus-merged rows at all - but it CAN regenerate the weekly
projections section, which is built entirely from data/weekly_locks/*.csv
(committed, no paid-data dependency) via build_weekly() in
build_board_artifact_data.py, reused here rather than reimplemented.

Usage:
    uv run scripts/build_weekly_artifact_patch.py --base-json path/to/current_board_data.json
    (writes the patched result to output/draft_rankings/board_data.json,
    same output path build_board_artifact_data.py uses, so the existing
    splice-into-the-artifact-HTML procedure is unchanged)
"""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_board_artifact_data import BOARD_DIR, build_weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-json", type=str, required=True,
        help="Path to a local copy of the CURRENTLY PUBLISHED board_data.json (e.g. saved after "
             "reading the live artifact) - its 'generated_at'/'half_ppr'/'ppr'/'replacement' "
             "sections are carried through unchanged; only 'weekly'/'weekly_meta' are replaced.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_json)
    data = json.loads(base_path.read_text(encoding="utf-8"))

    for required in ["generated_at", "half_ppr", "ppr", "replacement"]:
        if required not in data:
            raise SystemExit(f"--base-json is missing '{required}' - is this really a board_data.json?")

    weekly_half, weekly_half_meta = build_weekly("half_ppr")
    weekly_ppr, weekly_ppr_meta = build_weekly("ppr")
    data["weekly"] = {"half_ppr": weekly_half, "ppr": weekly_ppr}
    data["weekly_meta"] = {"half_ppr": weekly_half_meta, "ppr": weekly_ppr_meta}

    out_path = BOARD_DIR / "board_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB) - "
          f"weekly section refreshed ({len(weekly_half)} half_ppr / {len(weekly_ppr)} ppr rows), "
          f"main board sections carried over unchanged from {base_path.name}")


if __name__ == "__main__":
    main()
