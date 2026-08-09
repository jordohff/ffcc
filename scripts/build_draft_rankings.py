"""Build pre-draft season-long rankings and write a draft board CSV.

Different from build_projections.py (which predicts a single week from
in-season form): this predicts a full season's per-game rate from each
returning player's PRIOR season stats + age + team change, estimates games
played separately (durability), and projects incoming rookies from their
draft slot instead (no prior NFL stats to anchor on). Backtests the veteran
model on a recent holdout season using rank correlation - not MAE - since
what matters for a draft is getting players in the right ORDER.

Usage:
    uv run scripts/build_draft_rankings.py
    uv run scripts/build_draft_rankings.py --draft-season 2026 --scoring ppr
    uv run scripts/build_draft_rankings.py --teams 10 --flex-slots 2
"""

import argparse
from pathlib import Path

import pandas as pd

from ffmodel.features import compute_routes_run
from ffmodel.season import (
    aggregate_season_stats,
    build_enriched_weekly,
    build_prediction_features,
    build_rookie_training_table,
    build_season_training_table,
    compute_vbd,
    estimate_games_played,
    evaluate_rankings,
    fit_rookie_averages,
    fit_vet_models_by_position,
    predict_vet_ppg,
    project_rookies,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"

# Players on injured reserve/exempt list are still worth ranking (could come
# back); only exclude players who are definitively gone from the league.
EXCLUDED_STATUSES = {"RET", "CUT"}


def load_raw():
    weekly = pd.read_parquet(RAW_DIR / "weekly_stats.parquet")
    rosters = pd.read_parquet(RAW_DIR / "rosters.parquet")
    draft_picks = pd.read_parquet(RAW_DIR / "draft_picks.parquet")
    pbp = pd.read_parquet(RAW_DIR / "pbp_dropbacks.parquet")
    participation = pd.read_parquet(RAW_DIR / "participation.parquet")
    ngs = pd.read_parquet(RAW_DIR / "nextgen_receiving.parquet")
    depth_chart = pd.read_parquet(RAW_DIR / "current_depth_chart.parquet")
    return weekly, rosters, draft_picks, pbp, participation, ngs, depth_chart


def backtest(training_table: pd.DataFrame, test_season: int, top_n: int) -> None:
    """Train on everything before test_season, predict test_season, and
    compare against what ACTUALLY happened that season - the honest measure
    of whether this approach works, using the same mechanics (prior-season
    features, durability-capped games estimate) as the real 2026 projection.
    """
    train = training_table[training_table["season"] < test_season]
    test = training_table[training_table["season"] == test_season].copy()
    if test.empty:
        print(f"  no data for test season {test_season}, skipping backtest")
        return

    models = fit_vet_models_by_position(train)
    test["ppg_pred"] = predict_vet_ppg(models, test)
    test["games_est"] = estimate_games_played(test["prev_games_played"])
    test["total_points_pred"] = test["ppg_pred"] * test["games_est"]
    test["total_points"] = test["ppg"] * test["games_played"]

    print(f"Backtest: trained on seasons < {test_season}, evaluated on {test_season} "
          f"(actual results already known)")
    results = evaluate_rankings(test, top_n=top_n)
    print(results.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring", choices=["half_ppr", "ppr"], default="half_ppr")
    parser.add_argument("--draft-season", type=int, default=2026, help="Season to project (default: 2026)")
    parser.add_argument(
        "--backtest-season", type=int, default=2025,
        help="Recent completed season to backtest the veteran model against (default: 2025)",
    )
    parser.add_argument("--top-n", type=int, default=24, help="Top-N hit rate cutoff for backtest eval")
    parser.add_argument("--teams", type=int, default=12, help="League size, for VBD replacement level")
    parser.add_argument("--qb-slots", type=int, default=1)
    parser.add_argument("--rb-slots", type=int, default=2)
    parser.add_argument("--wr-slots", type=int, default=2)
    parser.add_argument("--te-slots", type=int, default=1)
    parser.add_argument("--flex-slots", type=int, default=1)
    args = parser.parse_args()

    weekly, rosters, draft_picks, pbp, participation, ngs, depth_chart = load_raw()

    print("Building season-level stats...")
    routes = compute_routes_run(pbp, participation, weekly)
    enriched = build_enriched_weekly(weekly, routes, ngs, scoring=args.scoring)
    season_stats = aggregate_season_stats(enriched)
    training_table = build_season_training_table(season_stats, rosters)

    print()
    backtest(training_table, args.backtest_season, args.top_n)

    print()
    print(f"Fitting final veteran model on all seasons through {args.draft_season - 1}...")
    models = fit_vet_models_by_position(training_table)
    vet_board = build_prediction_features(season_stats, args.draft_season, rosters)
    vet_board["ppg_pred"] = predict_vet_ppg(models, vet_board)
    vet_board["games_est"] = estimate_games_played(vet_board["prev_games_played"])
    vet_board["total_points_pred"] = vet_board["ppg_pred"] * vet_board["games_est"]
    vet_board = vet_board[
        ["player_id", "player_display_name", "position", "team", "age", "team_changed",
         "ppg_pred", "games_est", "total_points_pred"]
    ]
    vet_board["is_rookie"] = 0
    print(f"  {len(vet_board):,} returning players projected")

    print(f"Projecting {args.draft_season} rookie class from draft capital...")
    rookie_table = build_rookie_training_table(season_stats, draft_picks[draft_picks["season"] < args.draft_season])
    rookie_averages = fit_rookie_averages(rookie_table)
    current_picks = draft_picks[draft_picks["season"] == args.draft_season]
    rookie_board = project_rookies(current_picks, rookie_averages)
    rookie_board["total_points_pred"] = rookie_board["ppg_pred"] * rookie_board["games_est"]
    rookie_board = rookie_board.rename(columns={"pfr_player_name": "player_display_name"})
    rookie_board["age"] = pd.NA
    rookie_board["team_changed"] = 0
    rookie_board["is_rookie"] = 1
    rookie_board = rookie_board[
        ["player_id", "player_display_name", "position", "team", "age", "team_changed",
         "ppg_pred", "games_est", "total_points_pred", "is_rookie"]
    ]
    print(f"  {len(rookie_board):,} rookies projected")

    board = pd.concat([vet_board, rookie_board], ignore_index=True)
    board = board.dropna(subset=["ppg_pred", "total_points_pred"])

    print("Merging current roster status and depth chart context...")
    status = (
        rosters[rosters["season"] == args.draft_season][["gsis_id", "status"]]
        .rename(columns={"gsis_id": "player_id"})
        .drop_duplicates("player_id")
    )
    board = board.merge(status, on="player_id", how="left")
    board = board[~board["status"].isin(EXCLUDED_STATUSES)]

    # pos_rank is each player's overall depth order at their position on
    # their team (1 = starter) - pos_slot instead distinguishes different
    # depth-chart "columns" (e.g. X/Z/slot WR) and isn't what we want here.
    depth = depth_chart[["gsis_id", "pos_rank"]].rename(
        columns={"gsis_id": "player_id", "pos_rank": "depth_chart_rank"}
    )
    board = board.merge(depth, on="player_id", how="left")

    print("Computing value-based rankings...")
    board = compute_vbd(
        board,
        teams=args.teams,
        qb_slots=args.qb_slots,
        rb_slots=args.rb_slots,
        wr_slots=args.wr_slots,
        te_slots=args.te_slots,
        flex_slots=args.flex_slots,
    )
    board = board.sort_values("vbd", ascending=False)
    board = board.round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"draft_rankings_{args.draft_season}_{args.scoring}.csv"
    board.to_csv(out_path, index=False)
    print(f"Wrote {len(board):,} ranked players -> {out_path}")


if __name__ == "__main__":
    main()
