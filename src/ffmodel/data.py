"""Functions for pulling raw data from nflreadpy and the Sleeper API.

nflreadpy is the source of truth for historical stats. Sleeper is used only for
player ID/metadata (name, position, injury status) - see CLAUDE.md for why.
"""

import pandas as pd
import requests

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Pull weekly player stat lines for the given seasons from nflreadpy.

    Each row is one player's stat line for one week of one season: targets,
    receptions, rushing yards, etc. nflreadpy already computes fantasy_points
    (standard scoring) and fantasy_points_ppr (PPR scoring) for us, so we don't
    need to hand-roll a scoring formula.

    Only keeps QB/RB/WR/TE rows since those are the positions we're projecting.
    """
    import nflreadpy as nfl

    weekly = nfl.load_player_stats(seasons, summary_level="week")
    df = weekly.to_pandas()
    return df[df["position"].isin(FANTASY_POSITIONS)].reset_index(drop=True)


def fetch_sleeper_players() -> pd.DataFrame:
    """Pull the full Sleeper NFL player list (~11,000 players, a few MB).

    Sleeper's docs ask consumers to cache this locally rather than fetch it
    often, since the full list rarely changes. The caller is expected to save
    the result and reuse it (see scripts/pull_data.py).
    """
    resp = requests.get(SLEEPER_PLAYERS_URL, timeout=30)
    resp.raise_for_status()
    players = resp.json()  # dict keyed by Sleeper's internal player_id
    df = pd.DataFrame.from_dict(players, orient="index")
    df.index.name = "sleeper_player_id"
    return df.reset_index()
