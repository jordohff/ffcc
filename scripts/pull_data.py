"""Pull raw data and cache it locally under data/raw/.

Usage:
    uv run scripts/pull_data.py
    uv run scripts/pull_data.py --seasons 2021 2022 2023 2024 2025
    uv run scripts/pull_data.py --force
"""

import argparse
from pathlib import Path

from ffmodel.data import fetch_sleeper_players, load_weekly_stats

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2021, 2026)),
        help="Seasons to pull weekly stats for (default: 2021-2025)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if a cached file already exists",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    weekly_path = RAW_DIR / "weekly_stats.parquet"
    if args.force or not weekly_path.exists():
        print(f"Pulling weekly player stats for seasons {args.seasons} from nflreadpy...")
        weekly = load_weekly_stats(args.seasons)
        weekly.to_parquet(weekly_path, index=False)
        print(f"  saved {len(weekly):,} rows -> {weekly_path}")
    else:
        print(f"  {weekly_path} already exists, skipping (use --force to re-fetch)")

    players_path = RAW_DIR / "sleeper_players.parquet"
    if args.force or not players_path.exists():
        print("Pulling Sleeper player metadata...")
        players = fetch_sleeper_players()
        players.to_parquet(players_path, index=False)
        print(f"  saved {len(players):,} players -> {players_path}")
    else:
        print(f"  {players_path} already exists, skipping (use --force to re-fetch)")


if __name__ == "__main__":
    main()
