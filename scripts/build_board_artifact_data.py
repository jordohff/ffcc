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
from datetime import datetime
from pathlib import Path

import pandas as pd

from ffmodel.consensus import add_join_key
from ffmodel.paid_data import _normalize_name

BOARD_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"

# Coachspeak quotes/coach-reliability columns dropped 2026-09-01 at the
# user's request (no longer shown in the row-detail panel - only the
# simulated season range and bust/boom probability are) - trimmed from the
# artifact's payload for the same reason the per-source consensus columns
# were: no reason to embed data the page no longer displays. The
# coachspeak overlay itself stays in build_draft_rankings.py/coachspeak.py
# and on the CSV outputs, untouched - only this artifact-facing slice
# changed.
MODEL_COLS = [
    "player_id", "player_display_name", "position", "team", "ppg_pred", "prev_ppg", "games_est",
    "total_points_pred", "vbd", "depth_chart_rank", "is_rookie", "team_changed", "current_injury_status",
    "sim_p10", "sim_median", "sim_p90", "sim_bust_prob", "sim_boom_prob", "sim_full_season_prob",
    "manual_override_note",
]
# Trimmed 2026-09-01 (was one column per external source) at the user's
# request - the Consensus view no longer shows a column per ranking, just
# the blended Consensus number, so there's no reason to embed the rest in
# the artifact's payload. The full per-source breakdown still lives in
# composite_board_2026_*.csv (untouched, still every source's own rank) for
# any future spot-check/investigation - only this artifact-facing slice
# is narrower.
COMPOSITE_COLS = [
    "player_id", "player_display_name", "consensus_overall_rank", "consensus_position_pct", "n_sources",
]


# K/DST rows have no model prediction at all (our model doesn't project
# kickers/defenses - see consensus.build_kdst_consensus_board) - every
# MODEL_COLS field except identity (player_id/name/position/team) is left
# null/0 so these rows share the exact same shape the artifact's JS already
# expects, rather than needing special-case handling there.
KDST_MODEL_COL_DEFAULTS = {
    "ppg_pred": None, "prev_ppg": None, "games_est": None, "total_points_pred": None, "vbd": None,
    "depth_chart_rank": None, "is_rookie": 0, "team_changed": 0, "current_injury_status": None,
    "sim_p10": None, "sim_median": None, "sim_p90": None, "sim_bust_prob": None, "sim_boom_prob": None,
    "sim_full_season_prob": None, "manual_override_note": None,
}


def build_kdst_rows(scoring: str) -> list[dict]:
    path = BOARD_DIR / f"composite_board_kdst_2026_{scoring}.csv"
    if not path.exists():
        print(f"  No {path.name} - skipping K/DST rows (run build_composite_board.py first)")
        return []
    kdst = pd.read_csv(path)[["player_display_name", "position", "team", "consensus_overall_rank",
                               "consensus_position_pct", "n_sources"]].copy()
    kdst["consensus_position_pct"] = kdst["consensus_position_pct"].round(4)
    kdst["consensus_overall_rank"] = kdst["consensus_overall_rank"].round(2)
    for col, default in KDST_MODEL_COL_DEFAULTS.items():
        kdst[col] = default
    # No real gsis_id for K/DST at all (our board never carries them) - the
    # JS side's rowKey() already falls back to player_display_name whenever
    # player_id is null (see the artifact's own comment on that function),
    # same convention as the handful of UDFA-rookie null-id rows it already
    # handles, so leaving this null (not synthesizing a fake id) is safe.
    kdst["player_id"] = None
    return json.loads(kdst[MODEL_COLS + ["consensus_overall_rank", "consensus_position_pct", "n_sources"]].to_json(orient="records"))


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


SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def build_replacement_constants(scoring: str) -> dict:
    """Per-position (replacement_points, replacement_rank) so the artifact's
    JS can recompute live VBD as players are drafted, instead of the static
    pre-draft VBD baked into the CSV. `compute_vbd` (season.py) subtracts a
    per-position constant baseline from total_points_pred to get vbd - that
    constant is recoverable directly from any player's own row
    (points - vbd), and its RANK (how many players sit at/above it) is the
    "how many players deep does league-wide demand run" figure the live
    recompute needs. Median (not mean/first) guards against float noise
    across ~100-250 rows per position landing on a robust, real number.
    """
    board = pd.read_csv(BOARD_DIR / f"draft_rankings_2026_{scoring}.csv")
    out = {}
    for pos in SKILL_POSITIONS:
        sub = board[(board["position"] == pos) & board["vbd"].notna() & board["total_points_pred"].notna()]
        if sub.empty:
            continue
        points = float((sub["total_points_pred"] - sub["vbd"]).median())
        rank = int((sub["total_points_pred"] >= points).sum())
        out[pos] = {"points": round(points, 2), "rank": rank}
    return out


def main() -> None:
    # Displayed in the artifact's masthead - reflects when this SCRIPT was
    # last run, i.e. when the embedded data was actually last regenerated,
    # not when the page was last published (those can differ if a
    # publish-only change, like a CSS fix, happens without new data).
    generated_at = datetime.now().strftime("%b %d, %Y, %I:%M %p")
    data = {
        "generated_at": generated_at,
        "half_ppr": build_one("half_ppr") + build_kdst_rows("half_ppr"),
        "ppr": build_one("ppr") + build_kdst_rows("ppr"),
        "replacement": {
            "half_ppr": build_replacement_constants("half_ppr"),
            "ppr": build_replacement_constants("ppr"),
        },
    }
    out_path = BOARD_DIR / "board_data.json"
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB, "
          f"{len(data['half_ppr'])} half_ppr / {len(data['ppr'])} ppr rows)")


if __name__ == "__main__":
    main()
