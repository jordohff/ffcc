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

BOARD_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"

MODEL_COLS = [
    "player_id", "player_display_name", "position", "team", "ppg_pred", "games_est",
    "total_points_pred", "vbd", "depth_chart_rank", "is_rookie", "current_injury_status",
    "sim_p10", "sim_median", "sim_p90", "sim_bust_prob", "sim_boom_prob", "sim_full_season_prob",
    "coachspeak_notes", "coachspeak_last_date", "reliability_injury", "reliability_depth_chart",
    "reliability_usage_workload", "reliability_transactions", "manual_override_note",
]
COMPOSITE_COLS = [
    "player_id", "our_overall_rank", "dataroma_overall_rank", "dataroma_tier", "dataroma_position_rank",
    "barrett_overall_rank", "barrett_position_rank", "hansen_overall_rank", "hansen_position_rank",
    "csi_position_rank", "csi_tier", "consensus_overall_rank", "consensus_position_pct", "n_sources",
]


def build_one(scoring: str) -> list[dict]:
    board = pd.read_csv(BOARD_DIR / f"draft_rankings_2026_{scoring}.csv")[MODEL_COLS]
    composite = pd.read_csv(BOARD_DIR / f"composite_board_2026_{scoring}.csv")[COMPOSITE_COLS]
    # Both sides carry a handful of NaN player_id rows (the already-documented
    # 2026 UDFA-rookie gsis_id gap) - a left key of NaN fans out against every
    # NaN on the right in a pandas merge (the same bug class this project has
    # hit and fixed repeatedly elsewhere), so drop them from the right side
    # before merging rather than let them multiply.
    composite = composite.dropna(subset=["player_id"])
    merged = board.merge(composite, on="player_id", how="left")
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
