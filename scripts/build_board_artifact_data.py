"""Build the JSON data blob embedded in the published draft-board Artifact
(output/draft_rankings/board_data.json) - merges the model board CSV with
the composite/consensus board CSV (see consensus.py) so the artifact can
offer both a "Our Board" (VBD) view and a "Consensus" view without a second
data fetch.

Run build_draft_rankings.py and build_composite_board.py first (for both
scoring formats) - this script only reads their already-written CSVs.

Usage:
    uv run scripts/build_board_artifact_data.py
"""

import json
from pathlib import Path

import pandas as pd

from ffmodel.consensus import add_join_key
from ffmodel.paid_data import _normalize_name

BOARD_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"

MODEL_COLS = [
    "player_id", "player_display_name", "position", "team", "ppg_pred", "games_est",
    "total_points_pred", "vbd", "depth_chart_rank", "is_rookie", "current_injury_status",
    "sim_p10", "sim_median", "sim_p90", "sim_bust_prob", "sim_boom_prob", "sim_full_season_prob",
    "coachspeak_notes", "coachspeak_last_date", "reliability_injury", "reliability_depth_chart",
    "reliability_usage_workload", "reliability_transactions", "manual_override_note",
]
COMPOSITE_COLS = [
    "player_id", "player_display_name", "our_overall_rank", "dataroma_overall_rank", "dataroma_tier",
    "dataroma_position_rank", "barrett_overall_rank", "barrett_position_rank", "hansen_overall_rank",
    "hansen_position_rank", "csi_position_rank", "csi_tier", "consensus_overall_rank",
    "consensus_position_pct", "n_sources",
]


def build_one(scoring: str) -> list[dict]:
    board = pd.read_csv(BOARD_DIR / f"draft_rankings_2026_{scoring}.csv")[MODEL_COLS]
    composite = pd.read_csv(BOARD_DIR / f"composite_board_2026_{scoring}.csv")[COMPOSITE_COLS]
    # Both sides carry a handful of NaN player_id rows (the already-documented
    # 2026 UDFA-rookie gsis_id gap - De'Zhaun Stribling, Colbie Young, etc.).
    # Merging on raw player_id silently throws these players' real composite
    # data away entirely (their own board row has player_id=NaN, so a left
    # join keyed on player_id alone never matches them to their own composite
    # row, which also has player_id=NaN). Same fix as consensus.py's own
    # match_to_board/build_composite_board: merge on a join_key that falls
    # back to normalized display name only when player_id is null.
    board["normalized_name"] = board["player_display_name"].map(_normalize_name)
    composite["normalized_name"] = composite["player_display_name"].map(_normalize_name)
    board = add_join_key(board)
    composite = add_join_key(composite).drop_duplicates(subset="join_key", keep="first")
    composite = composite.drop(columns=["player_id", "player_display_name", "normalized_name"])
    merged = board.merge(composite, on="join_key", how="left")
    merged = merged.drop(columns=["join_key", "normalized_name"])
    merged["consensus_position_pct"] = merged["consensus_position_pct"].round(4)
    merged["consensus_overall_rank"] = merged["consensus_overall_rank"].round(2)
    # NaN -> null via a JSON round-trip (pandas' own to_json handles this correctly,
    # unlike a naive df.to_dict() which leaves float('nan') - not valid JSON).
    return json.loads(merged.to_json(orient="records"))


def main() -> None:
    data = {"half_ppr": build_one("half_ppr"), "ppr": build_one("ppr")}
    out_path = BOARD_DIR / "board_data.json"
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB, "
          f"{len(data['half_ppr'])} half_ppr / {len(data['ppr'])} ppr rows)")


if __name__ == "__main__":
    main()
