"""Build the projection model and write a backtested projections CSV.

Trains a simple regression on earlier seasons, evaluates it on a recent
holdout season, and writes predicted-vs-actual weekly fantasy points to
output/projections/ so you can review how well the model does.

Usage:
    uv run scripts/build_projections.py
    uv run scripts/build_projections.py --scoring ppr --test-season 2025
"""

import argparse
from pathlib import Path

import pandas as pd

from ffmodel.features import FEATURE_COLUMNS, build_features, compute_routes_run
from ffmodel.model import fit_and_evaluate_by_position, train_test_split_by_season

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "projections"

DISPLAY_COLUMNS = [
    "season",
    "week",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "fantasy_points_target",
    "projected_points",
    *FEATURE_COLUMNS,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", choices=["half_ppr", "ppr"], default="half_ppr")
    parser.add_argument(
        "--test-season",
        type=int,
        default=None,
        help="Season to hold out for backtesting (default: most recent season in the data)",
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "gbm"],
        default="ridge",
        help="ridge = linear regression (default); gbm = gradient-boosted trees",
    )
    args = parser.parse_args()

    required_files = [
        "weekly_stats.parquet",
        "schedules.parquet",
        "injuries.parquet",
        "pbp_dropbacks.parquet",
        "participation.parquet",
        "nextgen_receiving.parquet",
    ]
    missing = [f for f in required_files if not (RAW_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {RAW_DIR} - run `uv run scripts/pull_data.py` first.")

    raw = pd.read_parquet(RAW_DIR / "weekly_stats.parquet")
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    injuries = pd.read_parquet(RAW_DIR / "injuries.parquet")
    pbp_dropbacks = pd.read_parquet(RAW_DIR / "pbp_dropbacks.parquet")
    participation = pd.read_parquet(RAW_DIR / "participation.parquet")
    ngs_receiving = pd.read_parquet(RAW_DIR / "nextgen_receiving.parquet")
    test_season = args.test_season or int(raw["season"].max())

    print(f"Building features (scoring={args.scoring})...")
    routes = compute_routes_run(pbp_dropbacks, participation, raw)
    featured = build_features(raw, schedules, injuries, routes, ngs_receiving, scoring=args.scoring)

    train, test = train_test_split_by_season(featured, test_season=test_season)
    print(
        f"Training on seasons < {test_season} ({len(train):,} rows), "
        f"testing on season {test_season} ({len(test):,} rows)"
    )

    _, test_with_preds = fit_and_evaluate_by_position(train, test, kind=args.model)

    out = test_with_preds[DISPLAY_COLUMNS].sort_values(
        ["week", "projected_points"], ascending=[True, False]
    )
    out = out.round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"projections_{args.scoring}_{args.model}_{test_season}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
