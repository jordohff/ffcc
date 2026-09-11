"""Lock in one week's opponent-adjusted point projections BEFORE that week's
games are played, and save a durable, committed snapshot.

Why this is a separate, committed artifact and not just another gitignored
output: every other file under output/ is fully regenerable from data/raw/
at any time. A weekly projection is NOT regenerable once that week's games
have been played - the live data sources (injuries, depth chart, rosters)
get overwritten by later pulls, and re-running the pipeline after the fact
would silently bake in hindsight (updated depth charts reflecting how that
week actually went, in-game injuries, etc.). Locking a real snapshot BEFORE
kickoff each week is the only way to have an honest, bias-free record of
what was actually projected going in - both for the artifact's Weekly
Rankings page and for later accuracy tracking against realized results.

Reuses project_weekly_points (season.py) - the same opponent-adjusted
redistribution machinery already used inside build_draft_rankings.py -
rather than reimplementing anything. The schedule/defense-strength inputs
are leak-safe by construction (season+1 offsets, never the target season's
own in-progress results), so the only thing that can leak hindsight into a
projection is a stale/contaminated BOARD csv - which is exactly why
--board-csv exists, to point at a specific point-in-time snapshot instead
of always trusting whatever's currently on disk.

Usage:
    uv run scripts/build_weekly_projections.py --week 2
    uv run scripts/build_weekly_projections.py --week 1 --board-csv path/to/prekickoff_board.csv \
        --note "reconstructed from the last pre-kickoff data pull (2026-09-04), not a literal snapshot"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features import compute_routes_run
from ffmodel.season import build_enriched_weekly, build_weekly_matchups, compute_defense_strength, project_weekly_points

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DRAFT_DIR = ROOT / "output" / "draft_rankings"
LOCK_DIR = ROOT / "data" / "weekly_locks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--scoring", choices=["half_ppr", "ppr", "both"], default="both")
    parser.add_argument(
        "--board-csv", type=str, default=None,
        help="Override board CSV path (e.g. a saved pre-kickoff snapshot). Defaults to the "
             "current output/draft_rankings/draft_rankings_{season}_{scoring}.csv for each format.",
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="Free-text provenance note stored in the output (e.g. what data this board reflects, "
             "and as-of when) - always fill this in so a future session knows how trustworthy the "
             "snapshot is.",
    )
    args = parser.parse_args()

    weekly = pd.read_parquet(RAW_DIR / "weekly_stats.parquet")
    pbp = pd.read_parquet(RAW_DIR / "pbp_dropbacks.parquet")
    participation = pd.read_parquet(RAW_DIR / "participation.parquet")
    ngs = pd.read_parquet(RAW_DIR / "nextgen_receiving.parquet")
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    routes = compute_routes_run(pbp, participation, weekly)
    weekly_matchups = build_weekly_matchups(schedules, args.season)

    scorings = ["half_ppr", "ppr"] if args.scoring == "both" else [args.scoring]
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    locked_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for scoring in scorings:
        board_path = Path(args.board_csv) if args.board_csv else DRAFT_DIR / f"draft_rankings_{args.season}_{scoring}.csv"
        board = pd.read_csv(board_path)

        enriched = build_enriched_weekly(weekly, routes, ngs, scoring=scoring)
        defense_strength = compute_defense_strength(enriched)
        weekly_board = project_weekly_points(board, weekly_matchups, defense_strength, args.season)

        week_board = weekly_board[weekly_board["week"] == args.week].merge(
            board.dropna(subset=["player_id"])[["player_id", "player_display_name", "position", "team", "vbd"]],
            on="player_id", how="left",
        )
        week_board = week_board.sort_values("weekly_points_pred", ascending=False).round(2)
        week_board["season"] = args.season
        week_board["scoring"] = scoring
        week_board["locked_at"] = locked_at
        week_board["source_note"] = args.note
        week_board["source_board"] = str(board_path)

        cols = [
            "season", "week", "player_id", "player_display_name", "position", "team", "opponent",
            "is_bye", "matchup_factor", "weekly_points_pred", "scoring", "locked_at", "source_note", "source_board",
        ]
        out_path = LOCK_DIR / f"week{args.week}_{args.season}_{scoring}.csv"
        week_board[cols].to_csv(out_path, index=False)
        print(f"{scoring}: wrote {len(week_board)} rows -> {out_path}")


if __name__ == "__main__":
    main()
