"""Pull raw data and cache it locally under data/raw/.

Usage:
    uv run scripts/pull_data.py
    uv run scripts/pull_data.py --seasons 2021 2022 2023 2024 2025
    uv run scripts/pull_data.py --force        # re-fetch everything, e.g. after changing --seasons
    uv run scripts/pull_data.py --refresh-live # re-fetch only in-season-live sources (depth chart,
                                                # rosters, Sleeper, injuries) - fast, for a mid-season refresh
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import requests

from ffmodel.data import (
    fetch_sleeper_players,
    load_contract_history,
    load_current_depth_chart,
    load_draft_pick_capital,
    load_injury_reports,
    load_market_ecr_history,
    load_nextgen_receiving,
    load_participation,
    load_pbp_dropbacks,
    load_roster_info,
    load_schedules,
    load_snap_share,
    load_team_play_volume,
    load_weekly_stats,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# A committed (NOT gitignored - see .gitignore) last-known-good snapshot of
# sleeper_players.parquet, used only as a fallback when the live Sleeper API
# is unreachable. Exists because the scheduled cloud routine's sandbox
# blocks outbound requests to api.sleeper.app at the network-policy level
# (a 403 "connect_rejected" from its egress proxy - confirmed 2026-09-16,
# not transient, and not something exposed as a configurable per-repo
# setting) while every other source here (all nflreadpy/nflverse-backed)
# pulls fine from that same sandbox. Refreshed automatically by
# _pull_sleeper_players every time a real live pull succeeds (i.e. whenever
# a human session with normal network access runs this script), so it
# stays reasonably fresh over the season without needing a dedicated
# maintenance step.
SLEEPER_FALLBACK_PATH = Path(__file__).resolve().parents[1] / "data" / "fallback" / "sleeper_players_snapshot.parquet"

# Next Gen Stats and play-by-play participation charting only exist from 2016 on.
NGS_MIN_SEASON = 2016

# Snap count charting only exists from 2012 on.
SNAP_COUNTS_MIN_SEASON = 2012


def _pull_and_cache(filename: str, label: str, fetch, force: bool) -> None:
    path = RAW_DIR / filename
    if force or not path.exists():
        print(f"Pulling {label}...")
        df: pd.DataFrame = fetch()
        df.to_parquet(path, index=False)
        print(f"  saved {len(df):,} rows -> {path}")
    else:
        print(f"  {path} already exists, skipping (use --force to re-fetch)")


def _pull_sleeper_players(force: bool) -> None:
    """Like _pull_and_cache, but falls back to a committed last-known-good
    snapshot (SLEEPER_FALLBACK_PATH) if api.sleeper.app is unreachable,
    instead of crashing the whole pipeline - see that constant's own
    comment for why this exists. A successful live pull also refreshes the
    fallback snapshot, so it drifts only as stale as "since the last time
    someone with real network access ran this".
    """
    path = RAW_DIR / "sleeper_players.parquet"
    if not force and path.exists():
        print(f"  {path} already exists, skipping (use --force to re-fetch)")
        return
    print("Pulling Sleeper player metadata...")
    try:
        df = fetch_sleeper_players()
    except requests.exceptions.RequestException as exc:
        print(f"  live pull failed ({exc!r}) - falling back to the last committed snapshot")
        if not SLEEPER_FALLBACK_PATH.exists():
            raise RuntimeError(
                "Sleeper API unreachable and no fallback snapshot exists at "
                f"{SLEEPER_FALLBACK_PATH} - nothing to fall back to."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SLEEPER_FALLBACK_PATH, path)
        print(f"  copied fallback snapshot -> {path} (may be stale - see data/fallback/README or pull_data.py)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  saved {len(df):,} rows -> {path}")
    SLEEPER_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SLEEPER_FALLBACK_PATH, index=False)
    print(f"  refreshed fallback snapshot -> {SLEEPER_FALLBACK_PATH}")


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
        "--current-season",
        type=int,
        default=2026,
        help="The in-progress season to pull current rosters/draft class/depth chart for (default: 2026)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if a cached file already exists",
    )
    parser.add_argument(
        "--refresh-live",
        action="store_true",
        help=(
            "Re-fetch only the sources that actually change during a live season "
            "(sleeper_players, current_depth_chart, rosters, injuries) - skips the "
            "heavy historical-only pulls (weekly_stats, pbp, participation, NGS, "
            "snap_share, contract_history, team_play_volume, schedules), which don't "
            "change once a season is over. Use this for an in-season refresh instead "
            "of --force, which re-pulls everything."
        ),
    )
    args = parser.parse_args()
    live_sources = {
        "sleeper_players.parquet", "current_depth_chart.parquet", "rosters.parquet", "injuries.parquet",
        # The CURRENT season's own preseason ECR keeps accumulating new
        # scrape dates through early September (see
        # data.load_market_ecr_history) - a mid-season refresh should pick
        # up the freshest available preseason read the same way it does
        # for depth chart/rosters/injuries.
        "market_ecr_history.parquet",
    }

    def is_forced(filename: str) -> bool:
        return args.force or (args.refresh_live and filename in live_sources)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Draft-rankings pipeline needs the CURRENT season too (not just history) -
    # rosters/schedules/draft class for 2026 reflect this year's actual situation
    # (schedules specifically carries home_coach/away_coach, used for head-coach
    # lineage tracking - see season.build_head_coach_history).
    roster_seasons = sorted(set(args.seasons) | {args.current_season})

    _pull_and_cache(
        "weekly_stats.parquet",
        f"weekly player stats for seasons {args.seasons}",
        lambda: load_weekly_stats(args.seasons),
        args.force,
    )
    _pull_and_cache(
        "schedules.parquet",
        f"schedules for seasons {roster_seasons}",
        lambda: load_schedules(roster_seasons),
        args.force,
    )
    _pull_and_cache(
        "injuries.parquet",
        f"injury reports for seasons {args.seasons}",
        lambda: load_injury_reports(args.seasons),
        is_forced("injuries.parquet"),
    )
    _pull_sleeper_players(is_forced("sleeper_players.parquet"))

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

    _pull_and_cache(
        "rosters.parquet",
        f"roster info for seasons {roster_seasons}",
        lambda: load_roster_info(roster_seasons),
        is_forced("rosters.parquet"),
    )
    _pull_and_cache(
        "draft_picks.parquet",
        f"draft picks for seasons {roster_seasons}",
        lambda: load_draft_pick_capital(roster_seasons),
        args.force,
    )
    _pull_and_cache(
        "current_depth_chart.parquet",
        f"current ({args.current_season}) depth chart",
        lambda: load_current_depth_chart(args.current_season),
        is_forced("current_depth_chart.parquet"),
    )

    snap_seasons = [s for s in args.seasons if s >= SNAP_COUNTS_MIN_SEASON]
    if not snap_seasons:
        print(f"  no requested seasons are >= {SNAP_COUNTS_MIN_SEASON}, skipping snap share pull")
    else:
        _pull_and_cache(
            "snap_share.parquet",
            f"weekly snap share for seasons {snap_seasons}",
            lambda: load_snap_share(snap_seasons),
            args.force,
        )
    _pull_and_cache(
        "contract_history.parquet",
        "contract history (year-by-year cap details, full career)",
        load_contract_history,
        args.force,
    )
    _pull_and_cache(
        "team_play_volume.parquet",
        f"team offensive play volume for seasons {args.seasons}",
        lambda: load_team_play_volume(args.seasons),
        args.force,
    )
    _pull_and_cache(
        "market_ecr_history.parquet",
        "historical FantasyPros redraft ECR (position rank) archive, 2019-present",
        load_market_ecr_history,
        is_forced("market_ecr_history.parquet"),
    )


if __name__ == "__main__":
    main()
