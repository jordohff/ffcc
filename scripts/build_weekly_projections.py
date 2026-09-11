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

Also runs a real Monte Carlo simulation per player-week (simulate_weekly_
outcomes, season.py) - a SEPARATE residual pool from the season-level sim
(compute_weekly_walk_forward_residuals), since a season average smooths out
week-to-week variance a single game doesn't have. Only conditioned on
position, not matchup difficulty - tested first (2026-09-11), not assumed:
real weekly residual variance barely moves across matchup_factor quartiles
at any position, so bucketing by matchup would add complexity the data
doesn't support (see that function's own docstring for the full test).

Real games within one "week" don't all kick off at once (Wed/Thu openers,
a full Sunday slate, a Monday closer) - a game already played needs the
bias-free historical reconstruction, but a game still days away can (and
should) use the freshest live data available, since using stale data there
would just be needlessly worse for no bias-avoidance benefit. --played-teams
+ --historical-board-csv handle exactly this split: the (fresh, default)
board is the base, and any team listed in --played-teams gets its players'
rows swapped in from --historical-board-csv instead (matched by player_id).

Usage:
    uv run scripts/build_weekly_projections.py --week 2
    uv run scripts/build_weekly_projections.py --week 1 --board-csv path/to/prekickoff_board.csv \
        --note "reconstructed from the last pre-kickoff data pull (2026-09-04), not a literal snapshot"
    uv run scripts/build_weekly_projections.py --week 1 \
        --played-teams NE,SEA,SF,LA --historical-board-csv path/to/prekickoff_board.csv \
        --note "hybrid: NE/SEA/SF/LA (already played) use the 2026-09-04 pre-kickoff snapshot; \
every other team uses live data as of lock time (games not yet played)"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features import compute_routes_run
from ffmodel.season import (
    aggregate_healthy_season_stats,
    aggregate_season_stats,
    apply_manual_status_overrides,
    build_enriched_weekly,
    build_season_training_table,
    build_weekly_matchups,
    compute_defense_strength,
    compute_strength_of_schedule,
    compute_weekly_walk_forward_residuals,
    project_weekly_points,
    simulate_weekly_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DRAFT_DIR = ROOT / "output" / "draft_rankings"
LOCK_DIR = ROOT / "data" / "weekly_locks"

# Real, verified news (via web search, not yet reflected in Sleeper's live
# feed as of lock time) that a specific player will NOT play a SPECIFIC
# upcoming week - distinct from MANUAL_STATUS_OVERRIDES in season.py, which
# is for a real SEASON-LONG absence. This only zeroes the one (player_id,
# week) pair listed, via the same WEEKLY_DEFINITE_OUT_STATUSES check
# project_weekly_points already does for a real current_injury_status - it
# does not touch the player's season-long board ranking at all, since a
# short absence shouldn't move their full-season total_points_pred/VBD.
# Remove an entry once Sleeper's own feed catches up and reflects it.
MANUAL_WEEKLY_OUT: dict[tuple[str, int], str] = {
    ("00-0039338", 1): "Brock Bowers (LV TE) - meniscus trim 9/9/26, officially listed as a non-"
    "participant on LV's Week 1 injury report, expected to miss the LV@MIA game 9/13. Not yet "
    "reflected in Sleeper's live feed as of lock time (2026-09-11).",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--scoring", choices=["half_ppr", "ppr", "both"], default="both")
    parser.add_argument(
        "--board-csv", type=str, default=None,
        help="Override board CSV path (e.g. a saved pre-kickoff snapshot). Defaults to the "
             "current output/draft_rankings/draft_rankings_{season}_{scoring}.csv for each format. "
             "This is the BASE board - see --played-teams/--historical-board-csv for a hybrid.",
    )
    parser.add_argument(
        "--played-teams", type=str, default=None,
        help="Comma-separated team codes whose game for this week has ALREADY been played (e.g. "
             "NE,SEA,SF,LA). Those teams' players' rows are swapped in from --historical-board-csv "
             "instead of the base board, so an already-decided game stays bias-free while a still-"
             "upcoming one gets the freshest live data. Requires --historical-board-csv.",
    )
    parser.add_argument(
        "--historical-board-csv", type=str, default=None,
        help="Pre-kickoff board snapshot to source --played-teams' players from.",
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="Free-text provenance note stored in the output (e.g. what data this board reflects, "
             "and as-of when) - always fill this in so a future session knows how trustworthy the "
             "snapshot is.",
    )
    parser.add_argument("--n-sims", type=int, default=10000)
    args = parser.parse_args()
    if args.played_teams and not args.historical_board_csv:
        parser.error("--played-teams requires --historical-board-csv")
    played_teams = set(t.strip() for t in args.played_teams.split(",")) if args.played_teams else set()

    weekly = pd.read_parquet(RAW_DIR / "weekly_stats.parquet")
    pbp = pd.read_parquet(RAW_DIR / "pbp_dropbacks.parquet")
    participation = pd.read_parquet(RAW_DIR / "participation.parquet")
    ngs = pd.read_parquet(RAW_DIR / "nextgen_receiving.parquet")
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    rosters = pd.read_parquet(RAW_DIR / "rosters.parquet")
    snap_share = pd.read_parquet(RAW_DIR / "snap_share.parquet")
    contract_history = pd.read_parquet(RAW_DIR / "contract_history.parquet")
    injuries = pd.read_parquet(RAW_DIR / "injuries.parquet")
    ecr_history = pd.read_parquet(RAW_DIR / "market_ecr_history.parquet")
    routes = compute_routes_run(pbp, participation, weekly)
    weekly_matchups = build_weekly_matchups(schedules, args.season)

    scorings = ["half_ppr", "ppr"] if args.scoring == "both" else [args.scoring]
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    locked_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for scoring in scorings:
        board_path = Path(args.board_csv) if args.board_csv else DRAFT_DIR / f"draft_rankings_{args.season}_{scoring}.csv"
        board = pd.read_csv(board_path)

        if played_teams:
            historical_path = Path(args.historical_board_csv)
            historical = pd.read_csv(historical_path)
            swap_ids = set(historical.loc[historical["team"].isin(played_teams), "player_id"].dropna())
            n_swapped = int(board["player_id"].isin(swap_ids).sum())
            board = board[~board["player_id"].isin(swap_ids)]
            board = pd.concat([board, historical[historical["player_id"].isin(swap_ids)]], ignore_index=True)
            print(f"{scoring}: hybrid board - swapped {n_swapped} players on {sorted(played_teams)} "
                  f"in from {historical_path.name} (already played); everyone else uses {board_path.name}")

        # Re-applying CURRENT MANUAL_STATUS_OVERRIDES here (idempotent for
        # rows that already have them from a fresh pipeline run) matters
        # specifically for rows swapped in from an OLDER snapshot, which was
        # built by whatever season.py code existed at snapshot time - it can
        # be missing an override added since (e.g. Ricky Pearsall's real Aug
        # 1 IR placement, which predates even the 9/4 snapshot but wasn't
        # coded as an override until 9/11). This isn't hindsight: these are
        # real facts already true before ANY of this week's games were
        # played, just not yet reflected in the pipeline at snapshot time.
        board = apply_manual_status_overrides(board)

        board = board.copy()
        for (player_id, wk), note in MANUAL_WEEKLY_OUT.items():
            if wk != args.week:
                continue
            mask = board["player_id"] == player_id
            if mask.any():
                board.loc[mask, "current_injury_status"] = "Out"
                print(f"{scoring}: applied manual weekly-out for {player_id} (week {wk}): {note}")

        enriched = build_enriched_weekly(weekly, routes, ngs, scoring=scoring)
        defense_strength = compute_defense_strength(enriched)
        weekly_board = project_weekly_points(board, weekly_matchups, defense_strength, args.season)

        week_board = weekly_board[weekly_board["week"] == args.week].merge(
            board.dropna(subset=["player_id"])[["player_id", "player_display_name", "position", "team", "vbd"]],
            on="player_id", how="left",
        )

        print(f"{scoring}: building weekly residual pool for the Monte Carlo sim (walk-forward, 2019-2026)...")
        season_stats = aggregate_season_stats(enriched)
        healthy_season_stats = aggregate_healthy_season_stats(enriched, injuries)
        sos = compute_strength_of_schedule(schedules, defense_strength)
        training_table = build_season_training_table(
            season_stats, rosters, schedules, snap_share, contract_history, sos, ecr_history, healthy_season_stats
        )
        weekly_residuals = compute_weekly_walk_forward_residuals(training_table, enriched, defense_strength)
        sim = simulate_weekly_outcomes(week_board, weekly_residuals, n_sims=args.n_sims)
        week_board = week_board.merge(sim, on="player_id", how="left")

        week_board = week_board.sort_values("weekly_points_pred", ascending=False).round(2)
        week_board["season"] = args.season
        week_board["scoring"] = scoring
        week_board["locked_at"] = locked_at
        week_board["source_note"] = args.note
        source_board_desc = str(board_path)
        if played_teams:
            source_board_desc += f" (base) + {args.historical_board_csv} for {sorted(played_teams)}"
        week_board["source_board"] = source_board_desc

        cols = [
            "season", "week", "player_id", "player_display_name", "position", "team", "opponent",
            "is_bye", "matchup_factor", "weekly_points_pred",
            "sim_p10", "sim_p25", "sim_median", "sim_p75", "sim_p90", "sim_bust_prob", "sim_boom_prob",
            "scoring", "locked_at", "source_note", "source_board",
        ]
        out_path = LOCK_DIR / f"week{args.week}_{args.season}_{scoring}.csv"
        week_board[cols].to_csv(out_path, index=False)
        print(f"{scoring}: wrote {len(week_board)} rows -> {out_path}")


if __name__ == "__main__":
    main()
