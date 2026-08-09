"""Functions for pulling raw data from nflreadpy and the Sleeper API.

nflreadpy is the source of truth for historical stats, including historical
weekly injury reports (load_injuries). Sleeper is used only for player
ID/metadata - see CLAUDE.md for why. Sleeper's own injury_status field is a
live/current snapshot, not a historical time series, so it can't be used to
inform past weeks in a backtest (nflreadpy's load_injuries is what's used
for that instead - see add_injury_features in features.py).
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


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """Pull game schedule/context data for the given seasons from nflreadpy.

    One row per game: home/away teams, days of rest each team had going into
    the game, point spread, etc. Used to add home/away and rest-day context to
    each player's week.
    """
    import nflreadpy as nfl

    schedules = nfl.load_schedules(seasons)
    return schedules.to_pandas()


def load_injury_reports(seasons: list[int]) -> pd.DataFrame:
    """Pull historical weekly injury reports for the given seasons.

    One row per player per team-report update: their official pre-game injury
    designation (Questionable/Doubtful/Out) for that week, identified by
    `gsis_id` - the same player ID system used in load_weekly_stats/player_id.
    """
    import nflreadpy as nfl

    injuries = nfl.load_injuries(seasons)
    return injuries.to_pandas()


def load_pbp_dropbacks(seasons: list[int]) -> pd.DataFrame:
    """Pull just enough play-by-play data to identify pass dropbacks.

    The full nflverse play-by-play file has 300+ columns; we only need a
    handful to flag which plays were a QB dropback (attempt, sack, or
    scramble - any play where a receiver would have run a route), so we
    select those columns down immediately rather than caching the full file.
    """
    import nflreadpy as nfl

    pbp = nfl.load_pbp(seasons).select(["game_id", "play_id", "season", "week", "qb_dropback"])
    return pbp.to_pandas()


def load_participation(seasons: list[int]) -> pd.DataFrame:
    """Pull play-level participation data: which offensive players (by
    gsis_id) were on the field for each play.

    Only pulls the columns needed to estimate routes run (see
    features.compute_routes_run) - the full participation file also has
    personnel groupings and pass-rush/coverage charting we're not using yet.
    Available from 2016 onward.

    Note: participation also has an `offense_positions` column that looks
    like the obvious way to filter to WR/RB/TE, but it's only populated from
    2023 onward (100% null before that) - so we deliberately don't pull it,
    and instead tag each player's position using our own weekly_stats data,
    which has full coverage back to 2016 (see compute_routes_run).
    """
    import nflreadpy as nfl

    participation = nfl.load_participation(seasons).select(
        ["nflverse_game_id", "play_id", "offense_players"]
    )
    return participation.to_pandas()


def load_nextgen_receiving(seasons: list[int]) -> pd.DataFrame:
    """Pull weekly Next Gen Stats receiving metrics: average separation from
    the nearest defender, average cushion at the snap, share of the team's
    intended air yards, and yards-after-catch over expectation.

    Available from 2016 onward. Excludes the season-total rows nflreadpy
    includes at week=0 - we only want the per-week rows.
    """
    import nflreadpy as nfl

    ngs = nfl.load_nextgen_stats(seasons, stat_type="receiving").to_pandas()
    return ngs[ngs["week"] > 0].reset_index(drop=True)


def load_roster_info(seasons: list[int]) -> pd.DataFrame:
    """Pull roster info: team, roster status, birth date, years of
    experience, and draft slot, for the given seasons.

    Used two ways: (1) historically, to compute each player's age and team
    in past seasons for training the draft-rankings model, and (2) for the
    current season, to get each player's CURRENT team/status - catching
    offseason trades/free agency/retirements that last season's stats alone
    wouldn't reflect. `status` is worth filtering on for a final draft board
    (ACT = active roster; also RES/E14/RET/CUT for injured reserve/exempt/
    retired/released - not startable).
    """
    import nflreadpy as nfl

    rosters = nfl.load_rosters(seasons).select(
        ["season", "gsis_id", "team", "position", "status", "birth_date", "years_exp"]
    )
    return rosters.to_pandas()


def load_draft_pick_capital(seasons: list[int]) -> pd.DataFrame:
    """Pull NFL draft picks (round/pick/position/college) for the given
    draft-class seasons.

    Used both to train the rookie projection model (historical draft slot ->
    historical rookie-season production) and to project this year's actual
    incoming rookie class - `seasons` should include the current year, whose
    draft has already happened by the time this project cares about it
    (NFL draft is held every April, well before fantasy drafts in August).
    """
    import nflreadpy as nfl

    picks = nfl.load_draft_picks(seasons).select(
        ["season", "round", "pick", "team", "position", "gsis_id", "pfr_player_name"]
    )
    return picks.to_pandas()


def load_current_depth_chart(season: int) -> pd.DataFrame:
    """Pull the MOST RECENT live depth chart snapshot for the current season
    (team, position, depth rank within that position).

    The live feed actually contains many repeated snapshots over time (one
    per `dt` timestamp, taken as the depth chart gets updated through camp/
    the season) - naively dropping `dt` collapses those into duplicate rows
    per player, so we explicitly keep only the latest timestamp.

    NOTE: this only works for the CURRENT in-progress season - nflverse's
    depth chart archive for past seasons uses a different schema entirely
    (keyed by week/depth_team, not a live timestamp/pos_rank), so this isn't
    comparable across years. Because of that mismatch, this is used ONLY as
    current-context info for the final draft board (e.g. "is this player
    currently listed as the starter"), not as a trained historical feature.
    """
    import nflreadpy as nfl

    dc = nfl.load_depth_charts([season]).to_pandas()
    dc = dc[dc["pos_abb"].isin(FANTASY_POSITIONS)]
    latest = dc["dt"].max()
    return dc[dc["dt"] == latest].reset_index(drop=True)


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
