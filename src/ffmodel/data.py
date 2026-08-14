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

# Canonical team codes are whatever load_weekly_stats/load_schedules use
# (ARI, LA, LAC, LV, ...) since those are the tables everything else here
# gets joined against. A couple of other nflreadpy sources drift from that:
# load_rosters started returning "AZ" for Arizona starting with the 2026
# snapshot (still "ARI" in every past season), and load_draft_picks uses an
# entirely different 3-letter scheme (GNB, KAN, LVR, ...) plus some
# pre-relocation codes (OAK/SD/STL) for historical franchises. Left
# unnormalized, joins against schedules/weekly stats for the affected teams
# silently drop rows - see CLAUDE.md for the Arizona SOS case that surfaced
# this.
TEAM_CODE_FIXES = {
    "AZ": "ARI",
    "GNB": "GB",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NOR": "NO",
    "NWE": "NE",
    "OAK": "LV",
    "SDG": "LAC",
    "SD": "LAC",
    "SFO": "SF",
    "STL": "LA",
    "TAM": "TB",
}


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


def load_team_play_volume(seasons: list[int]) -> pd.DataFrame:
    """Pull team-level offensive play volume per game: total offensive plays
    (pass attempts + sacks + scrambles + rush attempts, i.e. every play that
    consumes a real offensive down - excludes kickoffs/punts/FGs/PATs/no-plays),
    split into pass plays vs. rush plays, aggregated to one row per team per
    season per game.

    This is the foundation for a "plays per game x player's share of that
    volume" opportunity model (see compute_team_pace in season.py) - the
    current draft-rankings pipeline fits every player's production
    independently, with nothing tying a team's players together, which lets
    physically-impossible team totals slip through (see
    apply_team_opportunity_cap's docstring for the concrete case that
    surfaced this). Aggregated down to team/game immediately rather than
    caching full play-level detail - only the per-game play counts are
    needed here, and the full nflverse pbp file has 300+ columns most of
    which aren't relevant to this.
    """
    import nflreadpy as nfl

    pbp = nfl.load_pbp(seasons).select(
        ["game_id", "season", "week", "season_type", "posteam", "rush_attempt", "pass_attempt"]
    )
    df = pbp.to_pandas()
    df = df[df["season_type"] == "REG"]
    df = df[(df["rush_attempt"] == 1) | (df["pass_attempt"] == 1)]
    df = df.dropna(subset=["posteam"])

    per_game = (
        df.groupby(["posteam", "season", "week", "game_id"])
        .agg(pass_plays=("pass_attempt", "sum"), rush_plays=("rush_attempt", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    per_game["total_plays"] = per_game["pass_plays"] + per_game["rush_plays"]
    per_game["team"] = per_game["team"].replace(TEAM_CODE_FIXES)
    return per_game


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
    df = rosters.to_pandas()
    df["team"] = df["team"].replace(TEAM_CODE_FIXES)
    return df


def load_draft_pick_capital(seasons: list[int]) -> pd.DataFrame:
    """Pull NFL draft picks (round/pick/position/college) for the given
    draft-class seasons.

    Used both to train the rookie projection model (historical draft slot ->
    historical rookie-season production) and to project this year's actual
    incoming rookie class - `seasons` should include the current year, whose
    draft has already happened by the time this project cares about it
    (NFL draft is held every April, well before fantasy drafts in August).

    load_draft_picks()' own `gsis_id` column is NOT a real gsis_id for the
    most recent draft class: nflreadpy hasn't back-filled it into the
    league's official ID system yet by draft season's end (that seems to
    happen once a player is actually in the league's official stats
    pipeline). Confirmed this affects 100% of the 2026 class (257/257 rows
    fail to match the standard "00-XXXXXXX" gsis format) - it's not a
    handful of edge cases. This silently broke every merge keyed on
    player_id for a true rookie (Sleeper roster info, current depth chart,
    strength-of-schedule) and was the root cause of every 2026 rookie
    showing a null depth_chart_rank on the board despite depth chart data
    for them existing. Fixed by crosswalking through `pfr_player_id`
    (a real PFR ID, e.g. "LoveJe00" - present and correct even for rookies)
    against load_players()' own pfr_id<->gsis_id mapping, same crosswalk
    pattern already used in load_snap_share. Recovers a real gsis_id for
    231/257 (90%) of the 2026 class - the rest are mostly non-skill
    positions (OL/DL/LB/DB) with no fantasy relevance; falls back to the
    original (broken) id for anyone the crosswalk can't resolve rather than
    dropping them, so non-fantasy positions and truly unmapped players don't
    silently disappear from the table.
    """
    import nflreadpy as nfl

    picks = nfl.load_draft_picks(seasons).select(
        ["season", "round", "pick", "team", "position", "gsis_id", "pfr_player_id", "pfr_player_name"]
    )
    df = picks.to_pandas()
    df["team"] = df["team"].replace(TEAM_CODE_FIXES)

    crosswalk = (
        nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls("pfr_id").to_pandas()
        .rename(columns={"gsis_id": "real_gsis_id", "pfr_id": "pfr_player_id"})
    )
    df = df.merge(crosswalk, on="pfr_player_id", how="left")
    df["gsis_id"] = df["real_gsis_id"].fillna(df["gsis_id"])
    return df.drop(columns=["pfr_player_id", "real_gsis_id"])


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


def load_snap_share(seasons: list[int]) -> pd.DataFrame:
    """Pull weekly offensive snap share (percent of team's offensive snaps
    played) for the given seasons.

    load_snap_counts() only has PFR-style player IDs (e.g. "WillKy00"), not
    the gsis_id used everywhere else in this pipeline, so this crosswalks
    through load_players()' pfr_id<->gsis_id mapping - verified a 99.8%
    match rate for QB/RB/WR/TE rows before relying on this. Available from
    2012 onward. Includes both REG and POST season_type rows (game_type
    column) - callers that care about the regular-season-only distinction
    (see aggregate_season_stats) need to filter it themselves.
    """
    import nflreadpy as nfl

    snaps = nfl.load_snap_counts(seasons).select(
        ["season", "week", "game_type", "pfr_player_id", "team", "position", "offense_snaps", "offense_pct"]
    )
    crosswalk = nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls("pfr_id")
    merged = snaps.join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    df = merged.to_pandas().rename(columns={"gsis_id": "player_id"}).drop(columns=["pfr_player_id"])
    return df


def load_contract_history() -> pd.DataFrame:
    """Pull each active/historical contract's year-by-year cap details
    (`season_history` - a nested per-year breakdown including `cap_percent`,
    that specific year's cap hit as a share of the total cap, already
    comparable across seasons without further normalization) and flatten it
    into one row per player per year.

    Not season-parameterized like other load_ functions here - this is a
    full-history pull (OverTheCap-sourced via nflreadpy), covering a
    player's whole known contract history in one call.
    """
    import nflreadpy as nfl

    contracts = nfl.load_contracts().select(["gsis_id", "season_history"]).to_pandas()
    contracts = contracts.dropna(subset=["gsis_id", "season_history"])
    exploded = contracts.explode("season_history").dropna(subset=["season_history"])
    detail = pd.json_normalize(exploded["season_history"])
    detail["player_id"] = exploded["gsis_id"].to_numpy()
    detail = detail.dropna(subset=["year", "cap_percent"])
    detail["season"] = detail["year"].astype(int)
    # A handful of years have multiple near-identical entries per player
    # (restructures/renegotiations tagged separately in the source) -
    # collapse to one row per player per year.
    return (
        detail.groupby(["player_id", "season"])["cap_percent"]
        .mean()
        .reset_index()
    )


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
