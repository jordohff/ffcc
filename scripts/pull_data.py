"""Pull raw data and cache it locally under data/raw/.

Usage:
    uv run scripts/pull_data.py
    uv run scripts/pull_data.py --seasons 2021 2022 2023 2024 2025
    uv run scripts/pull_data.py --force   # re-fetch, e.g. after changing --seasons
"""

import argparse
from pathlib import Path

import pandas as pd

from ffmodel.data import (
    fetch_sleeper_players,
    load_injury_reports,
    load_nextgen_receiving,
    load_participation,
    load_pbp_dropbacks,
    load_schedules,
    load_weekly_stats,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Next Gen Stats and play-by-play participation charting only exist from 2016 on.
NGS_MIN_SEASON = 2016


def _pull_and_cache(filename: str, label: str, fetch, force: bool) -> None:
    path = RAW_DIR / filename
    if force or not path.exists():
        print(f"Pulling {label}...")
        df: pd.DataFrame = fetch()
        df.to_parquet(path, index=False)
        print(f"  saved {len(df):,} rows -> {path}")
    else:
        print(f"  {path} already exists, skipping (use --force to re-fetch)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2010, 2026)),
        help="Seasons to pull weekly stats for (default: 2010-2025)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if a cached file already exists",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    _pull_and_cache(
        "weekly_stats.parquet",
        f"weekly player stats for seasons {args.seasons}",
        lambda: load_weekly_stats(args.seasons),
        args.force,
    )
    _pull_and_cache(
        "schedules.parquet",
        f"schedules for seasons {args.seasons}",
        lambda: load_schedules(args.seasons),
        args.force,
    )
    _pull_and_cache(
        "injuries.parquet",
        f"injury reports for seasons {args.seasons}",
        lambda: load_injury_reports(args.seasons),
        args.force,
    )
    _pull_and_cache(
        "sleeper_players.parquet",
        "Sleeper player metadata",
        fetch_sleeper_players,
        args.force,
    )

    ngs_seasons = [s for s in args.seasons if s >= NGS_MIN_SEASON]
    if not ngs_seasons:
        print(f"  no requested seasons are >= {NGS_MIN_SEASON}, skipping NGS/participation pulls")
    else:
        _pull_and_cache(
            "pbp_dropbacks.parquet",
            f"play-by-play dropback plays for seasons {ngs_seasons}",
            lambda: load_pbp_dropbacks(ngs_seasons),
            args.force,
        )
        _pull_and_cache(
            "participation.parquet",
            f"play participation for seasons {ngs_seasons}",
            lambda: load_participation(ngs_seasons),
            args.force,
        )
        _pull_and_cache(
            "nextgen_receiving.parquet",
            f"Next Gen Stats receiving for seasons {ngs_seasons}",
            lambda: load_nextgen_receiving(ngs_seasons),
            args.force,
        )


if __name__ == "__main__":
    main()
