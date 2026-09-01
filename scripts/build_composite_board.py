"""Build a composite (consensus) draft board blending this project's own
model with 4 trusted external ranking sources the user supplied: Dataroma,
Scott Barrett (FantasyPoints), John Hansen (FantasyPoints), and The
Coachspeak Index (Greg Brainos). Weighted per consensus.SOURCE_WEIGHTS, per
the user's explicit choice (2026-08-31) - see src/ffmodel/consensus.py for
the full blending methodology and per-source caveats (scoring format,
positional-only CSI).

Run build_draft_rankings.py first (for the scoring format you want) - this
script reads its already-written output CSV rather than rebuilding the
model.

Usage:
    uv run scripts/build_composite_board.py
    uv run scripts/build_composite_board.py --scoring half_ppr
"""

import argparse
from pathlib import Path

import pandas as pd

from ffmodel.consensus import build_composite_board, load_barrett, load_csi, load_dataroma, load_hansen

BOARD_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"
SOURCE_DIR = Path(__file__).resolve().parents[1] / "data" / "paid" / "ConsensusRankings"

# Dataroma is the one source with a real, scoring-specific export available
# (2026-08-31) - picked by --scoring, falling back to the PPR file if a
# half-PPR-specific one isn't present (fails soft, matching this project's
# established pattern elsewhere - e.g. the offensive-coordinator CSV loader
# - rather than erroring out over an optional refinement).
DATAROMA_FILES = {
    "ppr": "ff-dataroma-redraft-rankings (1).csv",
    "half_ppr": "ff-dataroma-redraft-rankings-half-ppr.csv",
}
BARRETT_FILE = "rankings.redraft.barrett.csv"
HANSEN_FILE = "rankings.redraft.top-200.csv"
CSI_FILE = "CSI 2026 Redraft Rankings (half-PPR).pdf"

DISPLAY_COLS = [
    "player_display_name", "position", "team",
    "consensus_overall_rank", "our_overall_rank",
    "dataroma_overall_rank", "barrett_overall_rank", "hansen_overall_rank",
    "csi_position_rank", "csi_tier",
    "consensus_position_pct", "n_sources",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scoring", choices=["half_ppr", "ppr"], default="ppr",
        help="Dataroma is loaded per-scoring (a real half-PPR export, if present - see DATAROMA_FILES). "
             "Hansen is confirmed PPR, Barrett is assumed PPR (not stated in its export) regardless of this "
             "flag - no half-PPR export exists for those yet. CSI is half-PPR regardless of this flag "
             "(positional-only, no overall rank affected by scoring format the way point totals are).",
    )
    parser.add_argument("--draft-season", type=int, default=2026)
    args = parser.parse_args()

    board_path = BOARD_DIR / f"draft_rankings_{args.draft_season}_{args.scoring}.csv"
    if not board_path.exists():
        raise SystemExit(
            f"Missing {board_path} - run "
            f"`uv run scripts/build_draft_rankings.py --scoring {args.scoring} --draft-season {args.draft_season}` first."
        )
    board = pd.read_csv(board_path)

    dataroma_file = DATAROMA_FILES[args.scoring]
    if not (SOURCE_DIR / dataroma_file).exists():
        print(f"  No {args.scoring}-specific Dataroma file ({dataroma_file}) - falling back to the PPR export")
        dataroma_file = DATAROMA_FILES["ppr"]

    for f in [dataroma_file, BARRETT_FILE, HANSEN_FILE, CSI_FILE]:
        if not (SOURCE_DIR / f).exists():
            raise SystemExit(f"Missing {SOURCE_DIR / f} - check data/paid/ConsensusRankings/")

    print("Loading external ranking sources...")
    dataroma = load_dataroma(str(SOURCE_DIR / dataroma_file))
    barrett = load_barrett(str(SOURCE_DIR / BARRETT_FILE))
    hansen = load_hansen(str(SOURCE_DIR / HANSEN_FILE))
    csi = load_csi(str(SOURCE_DIR / CSI_FILE))
    print(f"  Dataroma: {len(dataroma)} players, Barrett: {len(barrett)}, Hansen: {len(hansen)}, CSI: {len(csi)}")

    print("\nMatching sources to our board...")
    composite = build_composite_board(board, dataroma, barrett, hansen, csi)

    out_path = BOARD_DIR / f"composite_board_{args.draft_season}_{args.scoring}.csv"
    composite.to_csv(out_path, index=False)
    print(f"\nWrote {len(composite)} rows to {out_path}")

    print("\nTop 40 by consensus overall rank:")
    print(composite[DISPLAY_COLS].head(40).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
