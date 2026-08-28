"""Season-long aggregation and modeling for pre-draft rankings.

This is a DIFFERENT pipeline than the weekly model (features.py/model.py).
The weekly model predicts a single week from THIS season's trailing form -
which doesn't exist before the season starts, so it can't produce a draft
ranking. This module instead predicts a full season's per-game rate from the
PRIOR season's stats plus age/situation, then separately estimates games
played - the right shape for "who should I draft," not "who should I start
this week."
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ffmodel.features import add_fantasy_target, add_next_gen_features, add_route_features

SEASON_STAT_COLUMNS = [
    "games_played",
    "ppg",
    "targets_pg",
    "carries_pg",
    "target_share",
    "attempts_pg",
    "passing_yards_pg",
    "passing_tds_pg",
    "passing_epa_pg",
    "routes_run_pg",
    "tprr",
    "yprr",
    "separation",
    "cushion",
    "ngs_air_yards_share",
    "yac_above_exp",
]

# Recency weights for blending the last N seasons of each stat into one
# feature (see add_weighted_history_features) - most recent season first.
HISTORY_WEIGHTS = [0.5, 0.3, 0.2]

# Every SEASON_STAT_COLUMNS entry except games_played - the "how good is this
# player" rate stats, as opposed to the "how often do they play" durability
# stat, which is tracked separately (see add_weighted_history_features and
# aggregate_healthy_season_stats for why they use different source data).
RATE_STAT_COLUMNS = [c for c in SEASON_STAT_COLUMNS if c != "games_played"]

# Age where each position's production typically starts falling off a cliff -
# RBs decline earliest and most sharply, QBs latest/least abruptly (physical
# decline matters far less for a passer than for a player taking hits).
DECLINE_AGE = {"RB": 27, "WR": 30, "TE": 30, "QB": 38}

# RB next-season PPG decline by prior-season touch volume (2010-2024 seasons,
# this project's own data): roughly stable at -12% to -14% from 150 up
# through 349 touches, then ACCELERATES sharply past 350 touches (-21.9%) -
# holds even controlling for age (24-28 prime-years-only subset shows the
# same acceleration), so it's not just an age proxy. Two hinge points
# reproduce that shape: one where the general high-usage effect starts,
# one where it visibly steepens. See add_touch_volume_features.
TOUCH_VOLUME_HINGES = [150, 350]

COMMON_VET_FEATURES = [
    "prev_games_played",
    "prev_made_playoffs",
    "wavg_ppg",
    "years_past_decline_age",
    "team_changed",
    "new_head_coach",
    "new_hc_prior_team_ppg",
    "cap_percent",
    "sos_pts_allowed_pg",
]

SKILL_VET_FEATURES = COMMON_VET_FEATURES + [
    "wavg_targets_pg",
    "wavg_carries_pg",
    "wavg_target_share",
    "wavg_routes_run_pg",
    "wavg_tprr",
    "wavg_yprr",
    "wavg_separation",
    "wavg_cushion",
    "wavg_ngs_air_yards_share",
    "wavg_yac_above_exp",
    "vacated_targets_pg",
    "vacated_carries_pg",
    "vacated_routes_run_pg",
    "prev_snap_share_trend",
    "prev_snap_share_level",
]

QB_VET_FEATURES = COMMON_VET_FEATURES + [
    "wavg_carries_pg",
    "wavg_attempts_pg",
    "wavg_passing_yards_pg",
    "wavg_passing_tds_pg",
    "wavg_passing_epa_pg",
]

# RB-only: touch-volume hinge features (see add_touch_volume_features) -
# not meaningful for WR/TE, which essentially never reach these touch totals.
RB_VET_FEATURES = SKILL_VET_FEATURES + [f"touches_over_{h}" for h in TOUCH_VOLUME_HINGES]

POSITION_VET_FEATURES = {
    "QB": QB_VET_FEATURES,
    "RB": RB_VET_FEATURES,
    "WR": SKILL_VET_FEATURES,
    "TE": SKILL_VET_FEATURES,
}


def build_enriched_weekly(
    weekly_stats: pd.DataFrame,
    routes: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    scoring: str = "half_ppr",
) -> pd.DataFrame:
    """Attach fantasy points, routes run, TPRR/YPRR, and Next Gen Stats to
    each raw weekly stat line - the same per-game enrichment the weekly
    model uses, reused here as the input to season-level aggregation.
    """
    df = add_fantasy_target(weekly_stats, scoring=scoring)
    df = add_route_features(df, routes)
    df = add_next_gen_features(df, ngs_receiving)
    return df


def aggregate_season_stats(enriched_weekly: pd.DataFrame) -> pd.DataFrame:
    """Collapse weekly stat lines into one row per player per season:
    games played, and PER-GAME rates for everything else (not season totals -
    per-game rates are what carry over player-to-player and year-to-year;
    games played is handled separately as a durability question, not folded
    into the rate stats).

    Only REGULAR SEASON games (season_type == "REG") count toward
    games_played and the per-game rate stats - none of this project's
    target leagues play fantasy through the NFL playoffs, so postseason
    performance shouldn't be blended into what's meant to represent "points
    per regular season game." This was a real, confirmed bug before this
    filter existed: games_played could run past the 17-game regular season
    (up to 20+) for players whose team made a deep playoff run, quietly
    inflating their per-game averages with extra, non-representative games -
    2,281 player-seasons in this dataset have playoff games mixed in.

    Playoff participation is still informative, though, just not folded into
    the rate stats the same way - a team trusting a player with real snaps
    in January says something about their role/health/team quality heading
    into next season. Captured separately below as `made_playoffs`/
    `playoff_games`/`playoff_ppg` rather than either discarded entirely or
    blended into the regular-season averages.
    """
    reg = enriched_weekly[enriched_weekly["season_type"] == "REG"].copy()
    post = enriched_weekly[enriched_weekly["season_type"] == "POST"]
    reg["touches"] = reg["carries"].fillna(0) + reg["receptions"].fillna(0)

    games_played = (
        reg.groupby(["player_id", "player_display_name", "position", "season"])["week"]
        .nunique()
        .reset_index(name="games_played")
    )

    per_game = (
        reg.groupby(["player_id", "season"])
        .agg(
            ppg=("fantasy_points_target", "mean"),
            targets_pg=("targets", "mean"),
            carries_pg=("carries", "mean"),
            target_share=("target_share", "mean"),
            attempts_pg=("attempts", "mean"),
            passing_yards_pg=("passing_yards", "mean"),
            passing_tds_pg=("passing_tds", "mean"),
            passing_epa_pg=("passing_epa", "mean"),
            routes_run_pg=("routes_run", "mean"),
            tprr=("tprr", "mean"),
            yprr=("yprr", "mean"),
            separation=("avg_separation", "mean"),
            cushion=("avg_cushion", "mean"),
            ngs_air_yards_share=("ngs_air_yards_share", "mean"),
            yac_above_exp=("avg_yac_above_expectation", "mean"),
            touches=("touches", "sum"),
        )
        .reset_index()
    )

    playoff_stats = (
        post.groupby(["player_id", "season"])
        .agg(playoff_games=("week", "nunique"), playoff_ppg=("fantasy_points_target", "mean"))
        .reset_index()
    )

    result = games_played.merge(per_game, on=["player_id", "season"], how="left")
    result = result.merge(playoff_stats, on=["player_id", "season"], how="left")
    result["made_playoffs"] = result["playoff_games"].notna().astype(int)
    result["playoff_games"] = result["playoff_games"].fillna(0)
    return result


def flag_injury_affected_weeks(injuries: pd.DataFrame) -> pd.DataFrame:
    """Flag player-weeks that are part of a PERSISTING injury: the player
    was Questionable or Doubtful for the same body part in both this week
    and the week before (consecutive weeks, same season).

    Deliberately requires 2+ consecutive weeks of the SAME body part, not
    just any single injury-report appearance - a one-off Friday game-time-
    decision tag that resolves by itself is normal and shouldn't suppress a
    whole season's numbers; a nagging issue a player is visibly playing
    through week after week is what we actually want to catch. The week
    that starts a 2-week streak already counts (it's the second consecutive
    week of the same issue), not just later weeks in a longer streak.
    """
    reports = (
        injuries.dropna(subset=["gsis_id"])
        .sort_values("date_modified")
        .drop_duplicates(subset=["season", "week", "gsis_id"], keep="last")
        .rename(columns={"gsis_id": "player_id"})
    ).sort_values(["player_id", "season", "week"])

    reports = reports.copy()
    reports["is_hurt"] = reports["report_status"].isin(["Questionable", "Doubtful"])
    reports["body_part"] = reports["report_primary_injury"].fillna("")

    grouped = reports.groupby("player_id")
    prev_hurt = grouped["is_hurt"].shift(1).fillna(False)
    prev_body_part = grouped["body_part"].shift(1)
    prev_week = grouped["week"].shift(1)
    prev_season = grouped["season"].shift(1)

    reports["injury_affected"] = (
        reports["is_hurt"]
        & prev_hurt
        & (reports["body_part"] == prev_body_part)
        & (reports["body_part"] != "")
        & (reports["season"] == prev_season)
        & (reports["week"] == prev_week + 1)
    )
    return reports[["player_id", "season", "week", "injury_affected"]]


def aggregate_healthy_season_stats(enriched_weekly: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Same as aggregate_season_stats, but excluding weeks flagged as part of
    a persisting injury (see flag_injury_affected_weeks) from the per-game
    RATE stats - a player's production while playing through a lingering
    injury understates their true talent level, so those weeks shouldn't
    drag down the historical signal fed to the model (see
    add_weighted_history_features, which uses this as the source for rate
    stats specifically, NOT for games played/durability).
    """
    flags = flag_injury_affected_weeks(injuries)
    healthy = enriched_weekly.merge(flags, on=["player_id", "season", "week"], how="left")
    healthy["injury_affected"] = healthy["injury_affected"].fillna(False).astype(bool)
    healthy = healthy[~healthy["injury_affected"]]
    return aggregate_season_stats(healthy)


def add_age_feature(table: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Add age (in years) as of September 1 of the season in `table`.

    Birth date doesn't change, so one lookup (deduped, keeping any available
    non-null birth_date per player) covers every season - no need to match
    the roster pull's own season to the row's season.
    """
    birth = (
        rosters.dropna(subset=["birth_date"])[["gsis_id", "birth_date"]]
        .drop_duplicates(subset="gsis_id")
        .rename(columns={"gsis_id": "player_id"})
    )
    table = table.merge(birth, on="player_id", how="left")
    season_start = pd.to_datetime(table["season"].astype(str) + "-09-01")
    table["age"] = (season_start - pd.to_datetime(table["birth_date"])).dt.days / 365.25
    return table.drop(columns="birth_date")


def add_age_curve_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add `years_past_decline_age`: a position-specific "hinge" that's zero
    until a player passes the age where THAT position's production typically
    falls off a cliff (see DECLINE_AGE), then grows 1-for-1 after - e.g. a
    35-year-old RB gets years_past_decline_age = 8.

    NOTE on things tried and reverted here, kept for anyone revisiting this:
    1. Plain `age`/`age_squared` were tried alongside this hinge and removed.
       The isolated age-term contribution swung ~28-30 points across the RB
       age range while the actual empirical ppg-by-age in this project's
       training data is close to FLAT (age 22 avg 7.8 ppg vs. age 33 avg 6.5
       ppg, no clean monotonic decline). That's a multicollinearity symptom:
       age, age^2, and years_past_decline_age are all highly correlated with
       each other, which let Ridge produce large, unstable coefficients that
       partially canceled out in-sample but distorted predictions at the
       age-30+ tail where training data is sparse (46 rows at 31, 15 at 33).
    2. An interaction term (`years_past_decline_age * wavg_ppg`) was tried to
       let elite players decline more gently than replacement-level players
       at the same age - a real, documented pattern in principle. It came
       back with the WRONG sign (penalized elite-and-aging players MORE, not
       less) and made a known test case (McCaffrey) rank worse, not better.
       Likely cause: the "old AND elite" cell of the training data is tiny
       (very few RB-seasons combine age 30+ with elite production), so the
       interaction coefficient is probably fitting noise in that sparse
       corner rather than a real signal. Reverted rather than shipped on a
       result moving the wrong direction. If revisited, this likely needs
       either much more data at that intersection or a non-linear model
       family that handles sparse interactions more gracefully than Ridge.
    """
    table = table.copy()
    decline_age = table["position"].map(DECLINE_AGE).fillna(30)
    table["years_past_decline_age"] = (table["age"] - decline_age).clip(lower=0)
    return table


def add_touch_volume_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add RB-specific touch-volume hinge features from `prev_touches` (last
    season's total carries + receptions, NOT blended across years - this is
    a single-season-lookback effect per the research, not a career-average
    one). `touches_over_150` and `touches_over_350` are each zero until
    `prev_touches` passes that threshold, then grow 1-for-1 - lets Ridge fit
    a steeper slope specifically above 350 instead of one straight line
    across the whole range.

    Only meaningful for RB (see RB_VET_FEATURES) - WRs/TEs essentially never
    approach these touch totals (their workload is overwhelmingly receptions,
    not combined carries+receptions), so this wasn't researched or intended
    for those positions.
    """
    table = table.copy()
    for hinge in TOUCH_VOLUME_HINGES:
        table[f"touches_over_{hinge}"] = (table["prev_touches"] - hinge).clip(lower=0)
    return table


def compute_snap_share_trend(snap_share: pd.DataFrame) -> pd.DataFrame:
    """For every player-season, compute a PROGRESSIVELY recency-weighted
    average offensive snap share across the whole regular season - each
    week is weighted by its own week number (week 17 counts ~17x as much as
    week 1), so the result reflects where a player's role ENDED UP rather
    than treating every week of the season equally.

    This replaces an earlier first-half/second-half split, which only had
    two discrete buckets and an arbitrary cutoff week - a player whose role
    changed gradually and continuously (the common case) is better captured
    by weighting every week's distance from the end of the season, not by
    which side of one cutoff each week happened to fall on.

    Also returns `snap_share_trend`: the weighted average MINUS the plain
    (unweighted) season average. Positive means the recency-weighted view
    is higher than a flat average would suggest - the role GREW as the
    season progressed (gaining trust, a committee-mate declining); negative
    means it SHRANK. A shrinking-but-still-dominant role reads very
    differently from a shrinking-and-now-shared one, which is why the level
    itself is returned alongside the trend, not just the trend alone.

    REG season only (`game_type == "REG"` in snap_share).
    """
    reg = snap_share[snap_share["game_type"] == "REG"].copy()
    reg["weighted_pct"] = reg["week"] * reg["offense_pct"]

    grouped = reg.groupby(["player_id", "season"])
    weighted_avg = grouped["weighted_pct"].sum() / grouped["week"].sum()
    plain_avg = grouped["offense_pct"].mean()

    result = pd.DataFrame(
        {"snap_share_level": weighted_avg, "snap_share_trend": weighted_avg - plain_avg}
    ).reset_index()
    return result


def add_snap_share_trend_features(table: pd.DataFrame, snap_share_trend: pd.DataFrame) -> pd.DataFrame:
    """Add `prev_snap_share_trend`/`prev_snap_share_level` from the single
    most recent season (NOT blended across years, like prev_touches -
    in-season momentum is inherently a recent-trajectory signal, not
    something to average with two-year-old trends).
    """
    prior = snap_share_trend.rename(
        columns={"snap_share_trend": "prev_snap_share_trend", "snap_share_level": "prev_snap_share_level"}
    )
    prior = prior.assign(season=prior["season"] + 1)
    table = table.merge(prior, on=["player_id", "season"], how="left")
    return table


def add_contract_signal_features(table: pd.DataFrame, contract_history: pd.DataFrame) -> pd.DataFrame:
    """Add `cap_percent`: the player's cap hit (as a share of that year's
    total cap - already comparable across seasons, see
    data.load_contract_history) for the season BEING PREDICTED itself - NOT
    lagged by a season the way performance stats are.

    This is deliberately NOT a `prev_` single-season-lookback feature like
    touches/games_played/snap_share_trend, even though it looks similar.
    Those are lagged because they only EXIST after a season is played - you
    can't know a player's 2026 touches until 2026 happens. A contract is the
    opposite: it's signed BEFORE the season starts and is fully known
    information at prediction time, same as current team or a new head
    coach. Lagging it by a season would mean a player who just got a huge
    new deal this offseason (e.g. a rookie-scale player extended into a
    market-setting contract) shows their OLD, much smaller cap hit instead
    of the new one - exactly backwards for what this feature is trying to
    capture ("how much has the team invested in this player RIGHT NOW").

    This is a "how much has this team invested in / committed to this
    player" signal, distinct from recent performance - a big, guaranteed
    contract reflects the team's own belief in a player's role security
    (and gives them less incentive to bench/replace him), which can matter
    independent of last season's raw stat line.
    """
    table = table.merge(contract_history, on=["player_id", "season"], how="left")
    table["cap_percent"] = table["cap_percent"].fillna(0)
    return table


def _team_by_season(rosters: pd.DataFrame) -> pd.DataFrame:
    """Each player's team for each season they were rostered - the most
    common team that season, in the rare case a mid-season trade means more
    than one row. Shared by add_team_change_feature and
    compute_vacated_opportunity, which both need "who was on what team when."
    """
    return (
        rosters.groupby(["gsis_id", "season"])["team"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
        .rename(columns={"gsis_id": "player_id"})
    )


def add_team_change_feature(table: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Add `team_changed`: did this player switch teams between the season
    used as features (season - 1) and the season being predicted?

    A team change (trade, free agency, cut-and-signed) can meaningfully
    change a player's role independent of their own talent/health, so it's
    worth flagging even though we don't model exactly which direction it'll
    push a given player.
    """
    team_by_season = _team_by_season(rosters)
    prev_team = team_by_season.rename(columns={"team": "prev_team"})
    prev_team = prev_team.assign(season=prev_team["season"] + 1)

    table = table.merge(team_by_season, on=["player_id", "season"], how="left")
    table = table.merge(prev_team, on=["player_id", "season"], how="left")
    table["team_changed"] = (
        table["team"].notna() & table["prev_team"].notna() & (table["team"] != table["prev_team"])
    ).astype(int)
    return table


def compute_vacated_opportunity(season_stats: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """For every team and season, sum the PRIOR season's per-game usage
    (targets, carries, routes run) of every player who was on that team last
    season but isn't this season - left via free agency, trade, retirement,
    or release.

    This is a TEAM-level "how much opportunity is now up for grabs" signal,
    not a prediction of who specifically absorbs it - that's left to each
    remaining/incoming player's own features (target share, depth chart
    role, etc.) to sort out. Computed for every season in the data (not just
    the season being projected) so the model can actually learn whether
    vacated opportunity predicts anything, rather than having it applied
    only at prediction time with no historical grounding.
    """
    team_by_season = _team_by_season(rosters)

    prior = team_by_season.rename(columns={"team": "prior_team", "season": "prior_season"})
    prior = prior.assign(season=prior["prior_season"] + 1)

    compare = prior.merge(team_by_season, on=["player_id", "season"], how="left")
    departed = compare[compare["team"].isna() | (compare["team"] != compare["prior_team"])]

    prior_stats = season_stats[["player_id", "season", "targets_pg", "carries_pg", "routes_run_pg"]].rename(
        columns={"season": "prior_season"}
    )
    departed = departed.merge(prior_stats, on=["player_id", "prior_season"], how="inner")

    return (
        departed.groupby(["prior_team", "season"])
        .agg(
            vacated_targets_pg=("targets_pg", "sum"),
            vacated_carries_pg=("carries_pg", "sum"),
            vacated_routes_run_pg=("routes_run_pg", "sum"),
        )
        .reset_index()
        .rename(columns={"prior_team": "team"})
    )


def add_vacated_opportunity_features(table: pd.DataFrame, vacated: pd.DataFrame) -> pd.DataFrame:
    """Merge in the team-level vacated-opportunity signal (see
    compute_vacated_opportunity) for the player's CURRENT team/season.
    Teams with no departures (or players not matched to a team) get 0,
    not NaN - no vacancy is a real, informative value here, not missing data.
    """
    table = table.merge(vacated, on=["team", "season"], how="left")
    for col in ["vacated_targets_pg", "vacated_carries_pg", "vacated_routes_run_pg"]:
        table[col] = table[col].fillna(0)
    return table


def build_head_coach_history(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per season with that team's head coach, built from
    schedules' `home_coach`/`away_coach` (one row per game - nflreadpy has
    no OC-level equivalent anywhere; see the separately maintained OC
    dataset in data/coaching/ for that piece, which only covers 2026 since
    it required manual research, not a structured data source).

    Takes the most common coach that season per team, in the rare case of
    an in-season firing/interim change.
    """
    home = schedules[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "head_coach"}
    )
    away = schedules[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "head_coach"}
    )
    games = pd.concat([home, away], ignore_index=True).dropna(subset=["head_coach"])
    return (
        games.groupby(["team", "season"])["head_coach"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
    )


def compute_team_offensive_output(season_stats: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive output per season: average combined PPG across
    all QB/RB/WR/TE on that team's roster. A simple, objective proxy for
    "how productive was this offense," computed entirely from data already
    in this pipeline - not a subjective coach rating.
    """
    team_by_season = _team_by_season(rosters)
    with_team = season_stats.merge(team_by_season, on=["player_id", "season"], how="inner")
    return with_team.groupby(["team", "season"])["ppg"].mean().reset_index(name="team_offensive_ppg")


def compute_team_season_pace(play_volume: pd.DataFrame) -> pd.DataFrame:
    """Roll load_team_play_volume's per-game rows up to one row per team per
    season: offensive plays per game, and the pass/rush split.

    First piece of the plays-per-game opportunity model (see
    apply_team_opportunity_cap's docstring for the problem this is meant to
    eventually replace - a proportional cap is a safety net, not a real fix,
    since it can't tell a true bell-cow rookie apart from a genuine
    committee back, it just trims a team's group evenly). This function
    only computes the HISTORICAL side; see project_team_pace for turning it
    into a forward-looking estimate for a season that hasn't happened yet.
    """
    season_totals = (
        play_volume.groupby(["team", "season"])
        .agg(games=("game_id", "nunique"), total_plays=("total_plays", "sum"), pass_plays=("pass_plays", "sum"))
        .reset_index()
    )
    season_totals["plays_per_game"] = season_totals["total_plays"] / season_totals["games"]
    season_totals["pass_rate"] = season_totals["pass_plays"] / season_totals["total_plays"]
    return season_totals[["team", "season", "plays_per_game", "pass_rate"]]


HC_SCHEME_IMPORT_WEIGHT = 0.7


def _incoming_coach_pass_rate(
    coach_history: pd.DataFrame, season_pace: pd.DataFrame, target_season: int
) -> pd.DataFrame:
    """For every team with a NEW head coach in `target_season` who was
    already a head coach elsewhere in `target_season - 1`, look up that
    coach's own team's pass_rate from that prior season - the scheme
    they're bringing with them. One row per team with a real prior-HC
    scheme to import; teams with no coaching change, a first-time HC, or an
    internal promotion (no "elsewhere" team to look up) simply don't appear.
    """
    this_year = coach_history[coach_history["season"] == target_season]
    last_year = coach_history[coach_history["season"] == target_season - 1].rename(
        columns={"head_coach": "prev_head_coach"}
    )
    changes = this_year.merge(last_year[["team", "prev_head_coach"]], on="team", how="left")
    changes = changes[changes["prev_head_coach"].notna() & (changes["head_coach"] != changes["prev_head_coach"])]

    prior_team = last_year.rename(columns={"team": "prior_team", "prev_head_coach": "coach_lookup"})
    changes = changes.merge(
        prior_team, left_on="head_coach", right_on="coach_lookup", how="inner"
    )
    changes = changes[changes["prior_team"] != changes["team"]]

    prior_pace = season_pace[season_pace["season"] == target_season - 1].rename(
        columns={"team": "prior_team", "pass_rate": "import_pass_rate"}
    )
    result = changes.merge(prior_pace[["prior_team", "import_pass_rate"]], on="prior_team", how="inner")
    return result[["team", "import_pass_rate"]]


def project_team_pace(
    season_pace: pd.DataFrame, target_season: int, coach_history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Project each team's plays-per-game and pass-rate for `target_season`
    as a recency-weighted average of their last 3 seasons (same 50/30/20
    HISTORY_WEIGHTS used throughout this pipeline, renormalized for teams
    with less history - e.g. a franchise that relocated only 2 seasons ago).
    Pace and scheme tendency are fairly sticky team/coaching-staff traits
    year to year, but do shift (a new coordinator can speed up or slow down
    an offense), so recent seasons count for more rather than an unweighted
    average.

    If `coach_history` is given, a team getting a NEW head coach who was
    already a head coach somewhere else last season blends in that coach's
    own prior team's pass_rate (their scheme identity) at
    HC_SCHEME_IMPORT_WEIGHT, rather than relying solely on THIS team's own
    trailing history (which reflects the OLD coach's scheme, not the
    incoming one). Tested against every real coaching change 2011-2025
    (n=8 - genuinely rare events, so treat the exact weight as directional
    rather than finely tuned): guessing the incoming coach's own prior-team
    pass_rate cut error nearly in half vs. pure team continuity (mean abs
    error 0.040 vs 0.077), and error dropped monotonically as more weight
    shifted toward the import guess, all the way to 100%. Deliberately
    chose 70%, not the in-sample-best 100% - hedges against personnel
    constraints capping how much of an old scheme a new coach can truly
    import, given how small and noisy this sample necessarily is (real HC
    changes are infrequent). This does NOT extend to plays_per_game (raw
    tempo): the same test showed the import guess was WORSE than
    continuity there (not significant, p=0.83) - tempo is much less
    cleanly a "scheme the coach brings with them" trait than pass/run
    identity is, so plays_per_game is left as pure continuity regardless of
    coaching changes.

    Deliberately does NOT scale plays_per_game by team offensive quality,
    despite the intuitive "better offenses stay on the field more" case for
    it: tested directly (2010-2025) and while offensive quality DOES
    correlate with plays_per_game in the same season (r=0.40) and even
    lagged a season for projection use (r=0.22), that entire relationship
    turned out to already be captured by a team's own pace continuity -
    once continuity is already known, the ADDITIONAL signal from offensive
    quality on top of it is small, statistically significant, and actually
    slightly NEGATIVE (r=-0.135, p=0.004). Adding it as an extra positive
    scaling factor would have been redundant with what continuity already
    captures, and directionally wrong on top of that.
    """
    import_rates = (
        _incoming_coach_pass_rate(coach_history, season_pace, target_season).set_index("team")["import_pass_rate"]
        if coach_history is not None
        else pd.Series(dtype=float)
    )

    rows = []
    for team, team_history in season_pace.groupby("team"):
        by_season = team_history.set_index("season")
        weighted_plays = weighted_pass_rate = weight_total = 0.0
        for lag, weight in enumerate(HISTORY_WEIGHTS, start=1):
            season = target_season - lag
            if season not in by_season.index:
                continue
            weighted_plays += by_season.loc[season, "plays_per_game"] * weight
            weighted_pass_rate += by_season.loc[season, "pass_rate"] * weight
            weight_total += weight
        if weight_total == 0:
            continue

        continuity_pass_rate = weighted_pass_rate / weight_total
        pass_rate_pred = continuity_pass_rate
        if team in import_rates.index:
            pass_rate_pred = (
                HC_SCHEME_IMPORT_WEIGHT * import_rates[team] + (1 - HC_SCHEME_IMPORT_WEIGHT) * continuity_pass_rate
            )

        rows.append(
            {
                "team": team,
                "season": target_season,
                "plays_per_game_pred": weighted_plays / weight_total,
                "pass_rate_pred": pass_rate_pred,
            }
        )
    return pd.DataFrame(rows)


def compute_defense_strength(enriched_weekly: pd.DataFrame) -> pd.DataFrame:
    """For every (team, season, position), the average fantasy points that
    team's DEFENSE allowed per game to that position across the whole
    regular season - e.g. "the 2025 Broncos allowed 9.2 PPG to opposing
    RBs." A season-level measure, unlike the weekly model's trailing-5-week
    version (features.add_matchup_features) - this pipeline projects a
    whole season, so it needs a whole-season baseline for the opponents,
    not an in-season rolling one.

    Needs the WEEKLY-grain `opponent_team` field (not in season_stats, which
    has no notion of a single game's opponent), so this takes the enriched
    weekly table, not season_stats.
    """
    reg = enriched_weekly[enriched_weekly["season_type"] == "REG"]
    points_allowed_by_week = (
        reg.groupby(["opponent_team", "season", "week", "position"])["fantasy_points_target"]
        .sum()
        .reset_index()
    )
    return (
        points_allowed_by_week.groupby(["opponent_team", "season", "position"])["fantasy_points_target"]
        .mean()
        .reset_index()
        .rename(columns={"opponent_team": "team", "fantasy_points_target": "pts_allowed_pg"})
    )


def compute_strength_of_schedule(schedules: pd.DataFrame, defense_strength: pd.DataFrame) -> pd.DataFrame:
    """For every team and season, a position-specific strength-of-schedule
    score: the average points that season's ACTUAL opponents allowed to
    that position in the PRIOR season (the most recent complete season
    knowable going into any given season - defense_strength itself has no
    "future" leakage since it's real historical results, but using the
    opponent's OWN current-season defense would be circular/unknowable in
    advance).

    Weighted naturally by how many times each opponent is actually played -
    a division rival faced twice contributes two rows to the average, not
    one, correctly reflecting that you really do play them twice.

    Computed for every season present in both `schedules` and
    `defense_strength` (not just the season being projected), so the model
    can learn whether SOS actually predicts anything rather than having it
    applied only at prediction time with no historical grounding.
    """
    home = schedules[["season", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = schedules[["season", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    matchups = pd.concat([home, away], ignore_index=True)

    prior_defense = defense_strength.rename(columns={"team": "opponent"})
    prior_defense = prior_defense.assign(season=prior_defense["season"] + 1)

    merged = matchups.merge(prior_defense, on=["opponent", "season"], how="inner")
    return (
        merged.groupby(["team", "season", "position"])["pts_allowed_pg"]
        .mean()
        .reset_index()
        .rename(columns={"pts_allowed_pg": "sos_pts_allowed_pg"})
    )


def add_strength_of_schedule_features(table: pd.DataFrame, sos: pd.DataFrame) -> pd.DataFrame:
    """Merge in `sos_pts_allowed_pg` for the player's own team/season/
    position - each position can face a very different schedule difficulty
    even on the same team's slate (a team's opponents might be tough against
    the run but weak against the pass, for instance), so this is matched on
    position too, not just team.
    """
    table = table.merge(sos, on=["team", "season", "position"], how="left")
    return table


def build_weekly_matchups(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per team per REG-season week for `season`: who they play that
    week. A team with a bye simply has no row for that week - callers that
    need every week explicitly represented (e.g. a 0-point bye-week row)
    should reindex against the full week range themselves.
    """
    reg = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    home = reg[["week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"})
    away = reg[["week", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"})
    return pd.concat([home, away], ignore_index=True)


def project_weekly_points(
    board: pd.DataFrame, weekly_matchups: pd.DataFrame, defense_strength: pd.DataFrame, target_season: int
) -> pd.DataFrame:
    """Distribute each player's season-level `ppg_pred` across the actual
    2026 schedule, week by week, scaled up or down by how tough that
    specific week's opponent is at the player's position - rather than
    assuming a flat, identical points total every week.

    Deliberately does NOT re-predict each week from scratch (that would be
    a much bigger, separately-validated model). Instead it redistributes
    the ALREADY-validated season total: `weekly_points = ppg_pred *
    matchup_factor * (games_est / 17)`, where `matchup_factor` is that
    week's opponent's points-allowed-per-game at the player's position
    (from the most recent completed season - `target_season` itself hasn't
    happened yet, same leak-safe convention as
    compute_strength_of_schedule) divided by the LEAGUE-AVERAGE points
    allowed to that position that same season. A factor of 1.0 means a
    perfectly average matchup that week; >1 means an easier-than-average
    matchup (weaker defense); <1 means tougher. Bye weeks get an explicit
    0. The `games_est / 17` term is essential, not optional: `ppg_pred` is
    a rate (points PER GAME PLAYED), and games_est already reflects that
    plenty of players aren't expected to suit up every single week (bench
    depth, injury-prone players, a rookie buried on a depth chart) -
    applying the raw per-game rate to all 17 non-bye weeks silently assumed
    every player plays every game, which is exactly wrong for anyone with
    games_est well below 17 (caught this from a deep-bench rookie QB whose
    weekly sum came out 10x his real season total before this fix - a
    games_est of ~2 applied at full rate across 17 weeks). There's no
    signal for WHICH specific weeks a player sits, so the discount is
    spread evenly across the whole schedule rather than guessed at.
    Matchup factors are NOT renormalized to force an exact reconciliation
    with the season's total_points_pred - real schedules aren't perfectly
    balanced (a team's bye and its specific 17 opponents are what they
    are), so a small residual drift between the weekly sum and the season
    total is expected and correct, not a bug to paper over.

    Drops any board row with a null player_id before building the weekly
    schedule - a handful of very-late-round/UDFA rookies have no resolvable
    gsis_id (see load_draft_pick_capital's crosswalk docstring) and a
    left-merge keyed on player_id treats every null as matching every other
    null, which would silently scramble weekly rows across unrelated
    players (the same class of bug already fixed once for the depth-chart
    merge in build_draft_rankings.py).
    """
    board = board.dropna(subset=["player_id"])

    league_avg = defense_strength.groupby(["season", "position"])["pts_allowed_pg"].mean().reset_index(
        name="league_avg_pts_allowed_pg"
    )
    opponent_defense = defense_strength.rename(columns={"team": "opponent"})
    opponent_defense = opponent_defense.assign(season=opponent_defense["season"] + 1)
    opponent_defense = opponent_defense.merge(
        league_avg.assign(season=league_avg["season"] + 1), on=["season", "position"], how="left"
    )
    opponent_defense = opponent_defense[opponent_defense["season"] == target_season]
    opponent_defense["matchup_factor"] = (
        opponent_defense["pts_allowed_pg"] / opponent_defense["league_avg_pts_allowed_pg"]
    )

    weeks = pd.DataFrame({"week": weekly_matchups["week"].unique()})
    n_weeks = len(weeks)
    schedule = board[["player_id", "team", "position", "ppg_pred", "games_est"]].merge(weeks, how="cross")
    schedule = schedule.merge(weekly_matchups, on=["team", "week"], how="left")

    schedule = schedule.merge(
        opponent_defense[["opponent", "position", "matchup_factor"]], on=["opponent", "position"], how="left"
    )
    schedule["matchup_factor"] = schedule["matchup_factor"].fillna(1.0)
    schedule["is_bye"] = schedule["opponent"].isna()
    availability = schedule["games_est"] / n_weeks
    schedule["weekly_points_pred"] = (schedule["ppg_pred"] * schedule["matchup_factor"] * availability).where(
        ~schedule["is_bye"], 0.0
    )
    return schedule[["player_id", "week", "opponent", "is_bye", "matchup_factor", "weekly_points_pred"]]


def add_head_coach_features(
    table: pd.DataFrame, coach_history: pd.DataFrame, team_output: pd.DataFrame
) -> pd.DataFrame:
    """Add `new_head_coach` (1/0: did this team change HC from last season to
    this one) and `new_hc_prior_team_ppg` (that incoming coach's team's
    offensive output in the season right before this one, from
    compute_team_offensive_output - 0 if the coach wasn't a HC anywhere the
    prior season, e.g. a first-time HC or someone promoted from within, or
    if there was no coaching change at all).

    This only tracks HEAD coaches, not coordinators (see
    build_head_coach_history for why) - a new OC running the same HC's
    system is a real, common case this feature won't catch, by design/data
    limitation, not an oversight.
    """
    prev_coach = coach_history.rename(columns={"head_coach": "prev_head_coach"})
    prev_coach = prev_coach.assign(season=prev_coach["season"] + 1)

    merged = coach_history.merge(prev_coach, on=["team", "season"], how="left")
    merged["new_head_coach"] = (
        merged["prev_head_coach"].notna() & (merged["head_coach"] != merged["prev_head_coach"])
    ).astype(int)

    # the new coach's own team+season from last year (wherever they were, if anywhere)
    prev_output = team_output.rename(columns={"team": "prev_hc_team", "team_offensive_ppg": "new_hc_prior_team_ppg"})
    prev_coach_team = coach_history.rename(columns={"team": "prev_hc_team", "head_coach": "coach_lookup"})
    prev_coach_team = prev_coach_team.assign(season=prev_coach_team["season"] + 1)
    merged = merged.merge(
        prev_coach_team, left_on=["head_coach", "season"], right_on=["coach_lookup", "season"], how="left"
    )
    merged = merged.merge(prev_output, on=["prev_hc_team", "season"], how="left")
    # only relevant when there WAS a coaching change and the new team differs from where they just were
    merged.loc[
        (merged["new_head_coach"] == 0) | (merged["prev_hc_team"] == merged["team"]), "new_hc_prior_team_ppg"
    ] = 0
    merged["new_hc_prior_team_ppg"] = merged["new_hc_prior_team_ppg"].fillna(0)

    table = table.merge(
        merged[["team", "season", "new_head_coach", "new_hc_prior_team_ppg"]], on=["team", "season"], how="left"
    )
    table["new_head_coach"] = table["new_head_coach"].fillna(0)
    table["new_hc_prior_team_ppg"] = table["new_hc_prior_team_ppg"].fillna(0)
    return table


def add_offensive_coordinator_context(
    board: pd.DataFrame, oc_data: pd.DataFrame, team_output: pd.DataFrame
) -> pd.DataFrame:
    """Merge in the maintained offensive-coordinator dataset (see
    data/coaching/offensive_coordinators_2026.csv) as INFORMATIONAL board
    columns - NOT a trained model feature. OC lineage has no structured
    historical data source anywhere (unlike head coaches - see
    build_head_coach_history), so this file only covers 2026, researched by
    hand. Once several seasons accumulate here, it could become a real
    trained feature the same way new_head_coach is; for now it's context for
    the human reading the board, refreshed by hand each offseason.

    `oc_prev_team_ppg` is only populated when `previous_role` was explicitly
    an "offensive coordinator" job at another identifiable NFL team (looked
    up via compute_team_offensive_output for the season right before this
    one) - internal promotions and non-OC prior roles (position coach, etc.)
    don't have a comparable external track record to look up.
    """
    oc = oc_data.copy()
    oc["lookup_season"] = oc["season"] - 1
    prev_output = team_output.rename(
        columns={"team": "previous_team", "season": "lookup_season", "team_offensive_ppg": "oc_prev_team_ppg"}
    )
    oc = oc.merge(prev_output, on=["previous_team", "lookup_season"], how="left")
    oc.loc[oc["previous_role"] != "offensive coordinator", "oc_prev_team_ppg"] = pd.NA

    return board.merge(
        oc[
            [
                "team",
                "offensive_coordinator",
                "previous_team",
                "previous_role",
                "is_internal_promotion",
                "oc_prev_team_ppg",
            ]
        ],
        on="team",
        how="left",
    )


def add_weighted_history_features(
    table: pd.DataFrame,
    season_stats: pd.DataFrame,
    healthy_season_stats: pd.DataFrame | None = None,
    weights: list[float] = HISTORY_WEIGHTS,
    full_season_games: int = 17,
) -> pd.DataFrame:
    """Blend the last `len(weights)` seasons of each stat into one
    recency-weighted feature (`wavg_<stat>`) instead of relying only on the
    single most recent season.

    This is what lets a proven player coming off one down/injury year still
    show up as good on paper: someone who put up 18 ppg for two years then
    had a 12-ppg down year lands around 15-16 here (weighted toward, but not
    solely anchored to, the down year) instead of being fully anchored to it
    the way a single prior-season feature would be.

    Weights default to 50/30/20 for the last 3 seasons, renormalized per
    player based on how many of those seasons actually exist - e.g. a
    2nd-year player with only 1 prior season just uses that season at full
    weight, not 50% weight against two missing/zero seasons.

    Each season's contribution to the RATE-stat blend (not games_played -
    see below) is ALSO scaled by how many games it contains relative to a
    full season (games_played / full_season_games, capped at 1), on top of
    the base recency weight. Without this, a heavily injury-SHORTENED
    season gets the SAME weight as a full healthy season purely because of
    when it happened, letting a small, unrepresentative sample dominate a
    player's history - e.g. McCaffrey's 4-game 2024 was getting full 30%
    recency weight in wavg_ppg despite being a tiny sample, even though his
    very next season (2025, a full 17 games) already showed he'd fully
    recovered. `games_played`'s own blend is NOT reliability-weighted this
    way - scaling games played by games played would be circular.

    `healthy_season_stats`, if given, is used as the source for every RATE
    stat (RATE_STAT_COLUMNS - ppg, targets_pg, etc.) INSTEAD of season_stats -
    i.e. built with weeks affected by a persisting injury excluded (see
    aggregate_healthy_season_stats), so a lingering injury doesn't drag down
    what we treat as a player's true talent level. `games_played`
    (durability) always comes from the unfiltered `season_stats` regardless -
    durability should reflect games ACTUALLY played, healthy or not.
    """
    if healthy_season_stats is None:
        healthy_season_stats = season_stats

    table = table.copy()
    lag_cols_by_stat = {stat: [] for stat in SEASON_STAT_COLUMNS}
    games_col_by_lag = {}

    for lag, _ in enumerate(weights, start=1):
        games_col = f"_lag{lag}_games_played"
        games_shifted = season_stats[["player_id", "season", "games_played"]].rename(
            columns={"games_played": games_col}
        )
        games_shifted = games_shifted.assign(season=games_shifted["season"] + lag)
        table = table.merge(games_shifted, on=["player_id", "season"], how="left")
        lag_cols_by_stat["games_played"].append(games_col)
        games_col_by_lag[lag] = games_col

        rate_cols = [f"_lag{lag}_{c}" for c in RATE_STAT_COLUMNS]
        rate_shifted = healthy_season_stats.rename(columns=dict(zip(RATE_STAT_COLUMNS, rate_cols)))
        rate_shifted = rate_shifted.assign(season=rate_shifted["season"] + lag)
        table = table.merge(
            rate_shifted[["player_id", "season", *rate_cols]], on=["player_id", "season"], how="left"
        )
        for stat, col in zip(RATE_STAT_COLUMNS, rate_cols):
            lag_cols_by_stat[stat].append(col)

    weight_arr = np.array(weights)
    # How "full" each lag's season was (0-1), 0 (not NaN) when that season
    # doesn't exist at all - avoids NaN propagating into the weighted sums.
    reliability_by_lag = np.column_stack(
        [
            np.nan_to_num(
                (table[games_col_by_lag[lag]].to_numpy(dtype=float) / full_season_games).clip(0, 1),
                nan=0.0,
            )
            for lag in range(1, len(weights) + 1)
        ]
    )

    for stat, cols in lag_cols_by_stat.items():
        values = table[cols].to_numpy(dtype=float)
        available = ~np.isnan(values)
        effective_weight = (
            np.broadcast_to(weight_arr, values.shape)
            if stat == "games_played"
            else weight_arr[np.newaxis, :] * reliability_by_lag
        )
        weighted_sum = np.nansum(np.nan_to_num(values, nan=0.0) * effective_weight, axis=1)
        weight_total = (available * effective_weight).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            table[f"wavg_{stat}"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
        table = table.drop(columns=cols)

    return table


def build_season_training_table(
    season_stats: pd.DataFrame,
    rosters: pd.DataFrame,
    schedules: pd.DataFrame,
    snap_share: pd.DataFrame,
    contract_history: pd.DataFrame,
    sos: pd.DataFrame,
    healthy_season_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the veteran training table: for each player-season Y where we
    also have that player's season Y-1 stats, attach recency-weighted
    multi-season history (`wavg_`, see add_weighted_history_features) plus
    last season's games played (`prev_games_played`, for durability) as
    features, and season Y's `ppg` as the target.

    Players with no season Y-1 row (true rookies, or anyone who missed the
    entire prior season) are excluded here - the model literally has nothing
    to condition on for them. They're handled by the separate rookie model
    (see build_rookie_training_table), which uses draft capital instead.

    `healthy_season_stats`, if given, is passed through to
    add_weighted_history_features so the `wavg_` rate features are built
    from injury-affected weeks excluded (see aggregate_healthy_season_stats).
    """
    prior_games = season_stats.rename(
        columns={"games_played": "prev_games_played", "made_playoffs": "prev_made_playoffs", "touches": "prev_touches"}
    )
    prior_games = prior_games.assign(season=prior_games["season"] + 1)
    prior_games = prior_games[["player_id", "season", "prev_games_played", "prev_made_playoffs", "prev_touches"]]

    table = season_stats.merge(prior_games, on=["player_id", "season"], how="inner")
    table = add_weighted_history_features(table, season_stats, healthy_season_stats)
    table = add_age_feature(table, rosters)
    table = add_age_curve_features(table)
    table = add_touch_volume_features(table)
    table = add_team_change_feature(table, rosters)
    table = add_vacated_opportunity_features(table, compute_vacated_opportunity(season_stats, rosters))
    coach_history = build_head_coach_history(schedules)
    team_output = compute_team_offensive_output(season_stats, rosters)
    table = add_head_coach_features(table, coach_history, team_output)
    table = add_snap_share_trend_features(table, compute_snap_share_trend(snap_share))
    table = add_contract_signal_features(table, contract_history)
    table = add_strength_of_schedule_features(table, sos)
    return table


def find_players_returning_from_lost_season(
    season_stats: pd.DataFrame, rosters: pd.DataFrame, target_season: int
) -> pd.DataFrame:
    """Find players who have NO season_stats row for `target_season - 1`
    (zero games played that season - hurt all year, on IR, suspended, etc.)
    but DO have real games in `target_season - 2` or `target_season - 3`,
    AND are on an active roster for `target_season` itself - i.e. a real,
    currently-rostered player returning from a fully lost season, not
    someone whose career simply ended.

    Found 2026-08-27 via the team-PPG-consistency check: Cleveland's real
    depth_chart_rank==1 QB is Deshaun Watson (confirmed 2026 Week 1 starter
    as of a real news check, after missing the entire 2025 season with a
    second Achilles tear) - but he was completely ABSENT from the board,
    not floored or discounted. Root cause: build_prediction_features only
    ever looked at season_stats[target_season - 1], and a player with ZERO
    games that season has no row there at all - a structural gap, not a
    calibration issue. Checked the scope before fixing: 42 real, currently
    active 2026 roster players are missing this same way (Watson, Will
    Levis, Tank Dell, Jonathon Brooks, and 38 much less relevant deep-bench
    names) - most would be correctly near-irrelevant even if included, but
    a few (Watson chief among them) are real, board-relevant misses.

    Validated this is a fittable population before adding it, not just
    patched in blind: the RAW "missed a season, had games before" cohort
    (2012-2025, unconditioned) is 90% players whose careers had simply
    ended (mean 0.59 games in the return season) - clearly not comparable
    to Watson. But conditioned the SAME way as every other role-transition
    check this session (current depth_chart_rank==1, from the
    contemporaneous week-1/2 snapshot): a completely different, much
    healthier population - mean 11.55 games played, median 14 (n=44). This
    is close enough to the already-validated 1-7-games-missed role-upgrade
    cohort's own outcome (QB healthy constant: 11.00 games) that it's
    reasonably treated as the SAME underlying phenomenon (a current starter
    with little-to-no recent track record) rather than needing its own
    separate calibration - once these players get a row at all, the
    existing role-upgrade machinery (prev_games_played < 8 already
    naturally includes 0) picks them up automatically.

    Deliberately does NOT extend build_season_training_table (the model-
    FITTING path) the same way - the Ridge model's own ppg_pred is not
    where the real accuracy comes from for this population anyway (the
    board-build-time role-upgrade replacement functions are), and training
    on this rare, thin-signal population risked destabilizing the model's
    coefficients for the much larger, well-behaved normal population with
    little benefit. This function only feeds the PREDICTION path.
    """
    had_prev = set(season_stats[season_stats["season"] == target_season - 1]["player_id"])
    had_earlier = set(
        season_stats[season_stats["season"].isin([target_season - 2, target_season - 3])]["player_id"]
    )
    active_roster = set(
        rosters[(rosters["season"] == target_season) & (rosters["status"] == "ACT")]["gsis_id"]
    )
    returning_ids = (had_earlier - had_prev) & active_roster
    if not returning_ids:
        return pd.DataFrame(columns=["player_id", "player_display_name", "position", "games_played",
                                      "made_playoffs", "touches"])

    latest = (
        season_stats[season_stats["player_id"].isin(returning_ids)]
        .sort_values("season")
        .drop_duplicates(subset="player_id", keep="last")[["player_id", "player_display_name", "position"]]
    )
    latest["games_played"] = 0
    latest["made_playoffs"] = 0
    latest["touches"] = 0
    return latest


def build_prediction_features(
    season_stats: pd.DataFrame,
    target_season: int,
    rosters: pd.DataFrame,
    schedules: pd.DataFrame,
    snap_share: pd.DataFrame,
    contract_history: pd.DataFrame,
    sos: pd.DataFrame,
    healthy_season_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature rows for predicting `target_season`, which hasn't been
    played yet (so it has no row of its own in season_stats) - from each
    returning player's recency-weighted recent history.

    Same construction as build_season_training_table, just for a single
    target season that doesn't need to already exist in the data. Includes
    players who have a season_stats row for target_season - 1 (i.e. played
    last season), PLUS players returning from a fully lost season (see
    find_players_returning_from_lost_season) - true rookies with zero NFL
    history at all are handled separately (see build_rookie_training_table/
    project_rookies).
    """
    prior = season_stats[season_stats["season"] == target_season - 1].copy()
    returning = find_players_returning_from_lost_season(season_stats, rosters, target_season)
    cols = ["player_id", "player_display_name", "position", "games_played", "made_playoffs", "touches"]
    table = pd.concat([prior[cols], returning[cols]], ignore_index=True).rename(
        columns={"games_played": "prev_games_played", "made_playoffs": "prev_made_playoffs", "touches": "prev_touches"}
    )
    table["season"] = target_season
    table = add_weighted_history_features(table, season_stats, healthy_season_stats)
    table = add_age_feature(table, rosters)
    table = add_age_curve_features(table)
    table = add_touch_volume_features(table)
    table = add_team_change_feature(table, rosters)
    table = add_vacated_opportunity_features(table, compute_vacated_opportunity(season_stats, rosters))
    coach_history = build_head_coach_history(schedules)
    team_output = compute_team_offensive_output(season_stats, rosters)
    table = add_head_coach_features(table, coach_history, team_output)
    table = add_snap_share_trend_features(table, compute_snap_share_trend(snap_share))
    table = add_contract_signal_features(table, contract_history)
    table = add_strength_of_schedule_features(table, sos)
    return table


def make_vet_pipeline() -> Pipeline:
    """Same shape as the weekly model's pipeline (median-impute + Ridge) -
    simple and explainable, and this dataset is much smaller (one row per
    player per season, not per week), so a heavier model has even less to
    work with here than it did for the weekly model.
    """
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=1.0))])


def fit_vet_models_by_position(train: pd.DataFrame) -> dict[str, Pipeline]:
    """Fit one Ridge model per position predicting next season's PPG from
    recency-weighted multi-season history (`wavg_`) + age + team_changed.
    """
    models = {}
    for position, feature_cols in POSITION_VET_FEATURES.items():
        pos_train = train[train["position"] == position].dropna(subset=["wavg_ppg"])
        if pos_train.empty:
            continue
        pipeline = make_vet_pipeline()
        pipeline.fit(pos_train[feature_cols], pos_train["ppg"])
        models[position] = pipeline
    return models


def predict_vet_ppg(models: dict[str, Pipeline], rows: pd.DataFrame) -> pd.Series:
    """Predict next-season PPG for each row using its position's model."""
    preds = pd.Series(index=rows.index, dtype=float)
    for position, feature_cols in POSITION_VET_FEATURES.items():
        mask = rows["position"] == position
        if not mask.any() or position not in models:
            continue
        preds.loc[mask] = models[position].predict(rows.loc[mask, feature_cols])
    return preds


def build_rookie_training_table(season_stats: pd.DataFrame, draft_picks: pd.DataFrame) -> pd.DataFrame:
    """Find each drafted player's actual rookie season (their first season
    with a recorded stat line matching their draft class year) and attach
    their draft round/pick.

    Drafted skill-position players who never recorded a single game that
    season (hurt all year, buried on the depth chart, etc.) are added back
    in as zero production, not silently excluded - otherwise the historical
    average is quietly conditioned on "drafted AND played at least once,"
    which overstates the true expected value of a given draft slot. This
    means a round's average, e.g., 8.5 ppg, is a true expected value across
    everyone drafted at that slot historically - including the real chance
    of a rookie contributing nothing at all - not "8.5 ppg if they play."
    """
    picks = draft_picks.dropna(subset=["gsis_id"]).rename(
        columns={"gsis_id": "player_id", "season": "draft_season", "position": "draft_position"}
    )
    picks = picks[picks["draft_position"].isin(POSITION_VET_FEATURES)]

    # Use season_stats' own charted position (actual NFL usage) for players
    # who DID play, in case it ever differs from their draft-listed position
    # (e.g. a college DB drafted as a WR project) - keep draft_position
    # around separately as the fallback for players who never played at all.
    rookie_stats = season_stats.merge(picks, on="player_id", how="inner")
    rookie_stats = rookie_stats[rookie_stats["season"] == rookie_stats["draft_season"]]

    played_keys = rookie_stats[["player_id", "draft_season"]].drop_duplicates()
    never_played = picks.merge(played_keys, on=["player_id", "draft_season"], how="left", indicator=True)
    never_played = never_played[never_played["_merge"] == "left_only"].drop(columns="_merge")
    never_played = never_played.assign(
        position=never_played["draft_position"], season=never_played["draft_season"], games_played=0, ppg=0.0
    )

    combined = pd.concat([rookie_stats.drop(columns="draft_position"), never_played], ignore_index=True)
    return combined


def fit_rookie_curve(rookie_table: pd.DataFrame) -> pd.DataFrame:
    """Historical rookie-season PPG and games played as a smooth function of
    OVERALL DRAFT PICK, fit per position - the whole "model" for projecting
    incoming rookies, since there's no prior-NFL-season data to anchor a
    regression on.

    Replaces a coarser round-bucket average. That approach gave every rookie
    in the same (position, round) bucket an IDENTICAL projection, which
    broke down badly at the top of round 1: the 2026 class has Jeremiyah
    Love (RB, pick 3 overall - a true top-3-overall selection, extraordinary
    for a running back) and Jadarian Price (RB, pick 32 - the very last pick
    of the same round) landing in the same bucket and getting the same
    ppg_pred/games_est, which is obviously wrong given how differently those
    two draft slots are actually valued.

    Fit is a simple log-linear regression, `ppg ~ a + b*log(pick)` (and the
    same shape for games_played), per position - not a black-box model,
    matching this project's preference for something easily inspected and
    second-guessed. log(pick) rather than raw pick because draft capital
    value decays roughly log-linearly (well documented in draft-value chart
    literature, e.g. the Jimmy Johnson/Rich Hill AV curves) - value drops
    fast from pick 1 to pick 30, then flattens out through the late rounds,
    which a straight linear fit in raw pick number would not capture.

    Historical fit quality (2010-2025 draft classes, R² of ppg ~ log(pick)):
    QB 0.45, RB 0.35, TE 0.32, WR 0.30 - real, usable signal (a flat
    within-bucket average has an effective R² of 0), though naturally
    noisier than the veteran models since a single college/combine profile
    says much less than a played NFL season does. Verified against the 2026
    class: Love (pick 3) now projects to ~16.8 ppg vs. Price (pick 32) at
    ~8.8 ppg - the two picks that were previously identical are now clearly
    differentiated, and this generalizes to every future draft class
    automatically (it's a function of pick number, not a hardcoded lookup
    for any specific player).
    """
    rows = []
    for position, pos_table in rookie_table.groupby("position"):
        log_pick = np.log(pos_table["pick"].to_numpy(dtype=float))
        ppg_slope, ppg_intercept = np.polyfit(log_pick, pos_table["ppg"].to_numpy(dtype=float), 1)
        games_slope, games_intercept = np.polyfit(log_pick, pos_table["games_played"].to_numpy(dtype=float), 1)
        rows.append(
            {
                "position": position,
                "ppg_intercept": ppg_intercept,
                "ppg_slope": ppg_slope,
                "games_intercept": games_intercept,
                "games_slope": games_slope,
                "n": len(pos_table),
            }
        )
    return pd.DataFrame(rows)


def project_rookies(current_draft_picks: pd.DataFrame, rookie_curve: pd.DataFrame) -> pd.DataFrame:
    """Apply each position's historical pick -> production curve (see
    fit_rookie_curve) to this year's actual draft class, giving each rookie
    a `ppg_pred` and `games_est` the same way veterans get one from the
    trained model - varying smoothly with the rookie's own pick number
    instead of only their draft round.
    """
    picks = current_draft_picks[current_draft_picks["position"].isin(POSITION_VET_FEATURES)].copy()
    picks = picks.merge(rookie_curve, on="position", how="left")
    log_pick = np.log(picks["pick"])
    picks["ppg_pred"] = (picks["ppg_intercept"] + picks["ppg_slope"] * log_pick).clip(lower=0)
    picks["games_est"] = (picks["games_intercept"] + picks["games_slope"] * log_pick).clip(lower=0, upper=17)
    picks = picks.rename(columns={"gsis_id": "player_id"})
    return picks.drop(columns=["ppg_intercept", "ppg_slope", "games_intercept", "games_slope", "n"])


def _round_bucket(round_num: pd.Series) -> pd.Series:
    """Collapse draft round into 1/2/3/4/'5-7' buckets - rounds 5-7 ("Day 3")
    grouped together since per-round sample sizes get noisy fast that deep.
    Used only for the outcome-range display below, where a real grouped
    sample of actual outcomes is what's wanted (unlike the point-estimate
    curve above, a percentile isn't something you want to extrapolate from
    a smooth fit).
    """
    return round_num.clip(upper=5).map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5-7"})


def compute_rookie_outcome_range(rookie_table: pd.DataFrame) -> pd.DataFrame:
    """Historical 10th/90th percentile rookie-season PPG by position and
    draft round - real spread, not a discount.

    Tested (and rejected) using a rookie-specific uncertainty discount on
    the point estimate itself: checked whether the outcome distribution at
    the top of the RB draft is right-skewed (a handful of stars like
    Barkley/Elliott inflating an unrepresentative mean) - it isn't. For RB
    picks 1-15 (2010-2025, n=12) mean and median are essentially identical
    (13.54 vs 13.60 ppg, skew -0.20, actually slightly LEFT-skewed) - the
    point estimate is already a fair summary of the typical outcome, not
    one a few outliers are dragging up. Also tested whether a competing
    established teammate on the roster (comp_max, a recency-weighted
    "who's the strongest returning competitor" signal) should scale the
    point estimate down: real and significant on pure opportunity share
    (r=-0.155, p=0.011) but adds ~nothing once you already know draft pick
    when tested on the actual predicted quantity, ppg (R² 0.392 -> 0.393,
    noise-level) - pick number already implicitly captures most of what
    roster crowding would tell you, so this was NOT shipped as a
    ppg_pred adjustment.

    Given both direct discount mechanisms came back unsupported, this
    exposes the real spread as CONTEXT instead: a low/high band a human
    can weigh against their own risk tolerance, rather than the model
    silently shrinking the number for everyone. Genuinely gappy real-world
    situational information this project doesn't have data for yet
    (offensive line quality, expected game script, beat-reporter camp
    intel) is exactly the kind of thing that SHOULD inform a call within
    this range - it just can't be systematically modeled from what's
    available here.
    """
    table = rookie_table.assign(round_bucket=_round_bucket(rookie_table["round"]))
    ranges = (
        table.groupby(["position", "round_bucket"])["ppg"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "ppg_outcome_low", 0.9: "ppg_outcome_high"})
        .reset_index()
    )
    return ranges


def add_rookie_outcome_range(rookies: pd.DataFrame, outcome_range: pd.DataFrame) -> pd.DataFrame:
    """Attach ppg_outcome_low/high (see compute_rookie_outcome_range) to a
    rookie board by position + draft round.
    """
    rookies = rookies.assign(round_bucket=_round_bucket(rookies["round"]))
    rookies = rookies.merge(outcome_range, on=["position", "round_bucket"], how="left")
    return rookies.drop(columns="round_bucket")


RECENT_INJURY_THRESHOLD = 10
BOUNCE_BACK_INTERCEPT = -0.58
BOUNCE_BACK_SLOPE = 0.40

QB_BOUNCE_BACK_INTERCEPT = -2.434
QB_BOUNCE_BACK_SLOPE = 0.466
"""QB-specific override of BOUNCE_BACK_INTERCEPT/SLOPE - see
estimate_games_played's docstring for why QB needed its own fit. Same
calibrate-on-2018-2022/validate-on-2023-2025 methodology as the original,
run separately for QB only.
"""


def estimate_games_played(
    weighted_games_played: pd.Series,
    prev_games_played: pd.Series,
    position: pd.Series | None = None,
    max_games: int = 17,
) -> pd.Series:
    """Durability estimate: recency-weighted average games played over the
    last few seasons (see HISTORY_WEIGHTS/add_weighted_history_features),
    capped at a full season, plus a bounce-back correction for a player
    coming off a recently shortened season.

    Uses multi-year history rather than just last season, so a player with a
    genuine injury-proneness PATTERN (e.g. 17/10/12 games the last 3 years)
    gets a lower estimate than someone who had one fluky bad-luck season
    (e.g. 17/17/10) - the same recency-weighted blend used for the rate
    stats, applied here to games played specifically. Still a simple,
    transparent heuristic rather than a trained model - it's easy to see
    (and second-guess) exactly what it's assuming for a given player, which
    matters more for a durability estimate than squeezing out a bit more
    accuracy from a black-box model on a genuinely hard problem (in-season
    injuries are close to unpredictable in advance).

    Bounce-back correction: prompted by Christian McCaffrey dropping out of
    the 2026 top-12 overall, then testing (not assuming) whether recent
    injury history should count AGAINST a player, per the user's explicit
    "draft for upside and situation, not scared of injuries" framing.
    Walk-forward tested (predict every season 2018-2025 using only data
    from before it) whether a plain recency-weighted games average is
    biased for players coming off a shortened season (prev_games_played <
    RECENT_INJURY_THRESHOLD). It is - badly - and in the OPPOSITE direction
    caution would suggest: calibrated the correction on 2018-2022 seasons,
    validated out-of-sample on 2023-2025, and the plain average
    underestimated this cohort's actual next-season games by +1.28 on
    average (well-centered at -0.13 after correcting). This held whether
    the recent injury was severe (0-4 games played: +2.60 underestimate) or
    moderate (5-9 games: +0.51), and whether it was a one-off (+1-year
    history clean: -0.13, i.e. already fine) or part of a chronic pattern
    (two bad seasons in a row: +1.32, underestimated even MORE) - there was
    no cut of this data where discounting further was justified. Final
    correction constants (BOUNCE_BACK_INTERCEPT/SLOPE) are refit on the
    full 2018-2025 pooled data after that validation.

    Deliberately does NOT extend to an OLDER injury that's now 2 seasons
    back with a full healthy season in between (McCaffrey's actual 2026
    setup: 2025 healthy, 2024's Achilles 2 seasons back still pulling his
    3-year wavg_games_played down) - that pattern was tested with the same
    calibrate/validate split and showed no statistically significant bias
    (validation mean -0.36, p=0.57), so no correction is applied there.
    McCaffrey's 2026 games_est is still discounted by 2024 sitting at 30%
    recency weight - the data doesn't currently support overriding that,
    and inventing a fix just to move one specific player would repeat the
    mistake already made and reverted once this project (the age x elite
    interaction term, fit on too sparse a slice of data to trust).

    QB-SPECIFIC CORRECTION (added 2026-08-27): the original BOUNCE_BACK_
    INTERCEPT/SLOPE were fit on data pooled across all four positions,
    dominated by RB/WR/TE (mean underestimate ~1.7-2.0 games for that
    cohort). Investigating why Jayden Daniels/Joe Burrow/Lamar Jackson
    ranked far below where real markets (FantasyCalc's trade-value data)
    place them found QB's OWN true bias is much smaller - recent-injury QBs
    are underestimated by only +0.38 games on average (vs +1.7-2.0 for the
    other three positions pooled, p=8e-10 that QB is genuinely different) -
    and isn't even uniform across severity: near-wipeout QB seasons (0-4
    games played) show a real +1.2 game underestimate, but MODERATE ones
    (5-9 games, e.g. Daniels' 2025) show no bias at all before any
    correction (-0.74, i.e. already fine or slightly generous). Applying
    the pooled correction to QB was measurably WRONG: on the 2023-2025
    holdout, the plain (uncorrected) estimate was already close to
    unbiased (mean resid 0.657, p=0.155 - not significant), but the
    shipped POOLED correction made it significantly biased the other way
    (mean resid -0.930, p=0.040). A QB-specific refit
    (QB_BOUNCE_BACK_INTERCEPT/SLOPE, same calibrate-2018-2022/validate-
    2023-2025 split) restores an unbiased estimate (mean resid 0.140,
    p=0.756) - it requires far more severe games-missed (breakeven ~5.2
    games missed vs the pooled formula's ~1.5) before any credit is added,
    matching the real shape found above.

    Also tested and REJECTED as an explanation for the Burrow/Daniels gap:
    an "oscillating health" pattern (last season AND the season 2 years
    back both shortened, with a healthy season between them - Burrow's
    actual 2023-short/2024-full/2025-short history). This IS a real,
    significant bias for RB/WR/TE (mean resid 1.64-2.06 games, p<3e-6 each)
    - a genuinely new pattern, distinct from both the single-recent-injury
    case (already corrected) and the single-old-injury-with-clean-recovery
    case (already tested and rejected, see above) - but for QB specifically
    it's small and NOT significant (mean resid 0.384, p=0.12, statistically
    indistinguishable from the general recent-injury QB bias already
    captured by the fix above). Not shipped as a QB correction; the
    remaining Burrow/Daniels/Lamar Jackson gap vs. real market value is not
    explained by a durability-estimate bug and is more likely inherent
    Ridge-model shrinkage for an unusual (elite-but-injury-interrupted)
    profile with few close training comps - a known, accepted limitation
    of this project's deliberately simple/interpretable model choice, not
    a bug with an identified fix.
    """
    games_est = weighted_games_played.clip(upper=max_games)
    games_missed = (RECENT_INJURY_THRESHOLD - prev_games_played).clip(lower=0)
    if position is not None and (position == "QB").any():
        intercept = pd.Series(BOUNCE_BACK_INTERCEPT, index=games_missed.index)
        slope = pd.Series(BOUNCE_BACK_SLOPE, index=games_missed.index)
        intercept = intercept.where(position != "QB", QB_BOUNCE_BACK_INTERCEPT)
        slope = slope.where(position != "QB", QB_BOUNCE_BACK_SLOPE)
        correction = (intercept + slope * games_missed).clip(lower=0)
    else:
        correction = (BOUNCE_BACK_INTERCEPT + BOUNCE_BACK_SLOPE * games_missed).clip(lower=0)
    correction = correction.where(prev_games_played < RECENT_INJURY_THRESHOLD, 0)
    return (games_est + correction).clip(upper=max_games)


# Sleeper and nflverse otherwise agree on team codes, but use different
# abbreviations for these two teams - normalize to nflverse's convention
# (used everywhere else in this pipeline) before comparing/using Sleeper's
# team field, or every Cardinals/Rams player falsely shows up as a "mismatch".
SLEEPER_TEAM_CODE_FIXES = {"LAR": "LA", "OAK": "LV"}


def apply_current_team_from_sleeper(board: pd.DataFrame, sleeper_players: pd.DataFrame) -> pd.DataFrame:
    """Override each player's team with Sleeper's, when available, and add
    current injury/depth-chart context from the same source.

    nflverse's roster snapshot (used everywhere else in this pipeline for
    historical team-by-season lookups) is a periodic pull and can lag real
    transactions by days to weeks. Sleeper is a live fantasy platform that
    needs to stay current for its own users, so it tends to reflect very
    recent moves faster - confirmed empirically: as of 2026-08-09, nflverse's
    roster pull still showed Stefon Diggs on NE (released back in March 2026)
    while Sleeper already had his correct signing with WAS from the prior
    week. `team` here is kept as the SLEEPER-preferred value used everywhere
    downstream; the original nflverse-derived value is kept as `team_nflverse`
    and `team_mismatch` flags any disagreement, so a mismatch is visible
    rather than silently overwritten.
    """
    sleeper = (
        sleeper_players.dropna(subset=["gsis_id"])
        .drop_duplicates(subset="gsis_id")
        # Sleeper's own "player_id" column is THEIR internal numeric ID, not
        # gsis_id - drop it first so renaming gsis_id -> player_id below
        # doesn't collide and produce two same-named columns.
        .drop(columns="player_id")
        .rename(
            columns={
                "gsis_id": "player_id",
                "team": "sleeper_team",
                "injury_status": "current_injury_status",
                "injury_body_part": "current_injury_body_part",
                "depth_chart_order": "sleeper_depth_chart_order",
            }
        )
    )
    sleeper["sleeper_team"] = sleeper["sleeper_team"].replace(SLEEPER_TEAM_CODE_FIXES)
    keep = [
        "player_id",
        "sleeper_team",
        "current_injury_status",
        "current_injury_body_part",
        "sleeper_depth_chart_order",
    ]
    board = board.rename(columns={"team": "team_nflverse"}).merge(sleeper[keep], on="player_id", how="left")
    board["team"] = board["sleeper_team"].fillna(board["team_nflverse"])
    board["team_mismatch"] = (
        board["team_nflverse"].notna() & board["sleeper_team"].notna() & (board["team_nflverse"] != board["sleeper_team"])
    )
    return board


def evaluate_rankings(
    df: pd.DataFrame,
    actual_col: str = "total_points",
    pred_col: str = "total_points_pred",
    top_n: int = 24,
) -> pd.DataFrame:
    """Evaluate draft-ranking quality per position: Spearman rank correlation
    (are players in the right ORDER, which is what a draft actually needs -
    not point-level precision) and top-N hit rate (of our predicted top N at
    a position, how many actually finished top N that season).
    """
    rows = []
    for position, group in df.groupby("position"):
        group = group.dropna(subset=[actual_col, pred_col])
        if len(group) < 2:
            continue
        spearman = group[actual_col].corr(group[pred_col], method="spearman")

        n = min(top_n, len(group))
        actual_top = set(group.sort_values(actual_col, ascending=False).head(n)["player_id"])
        pred_top = set(group.sort_values(pred_col, ascending=False).head(n)["player_id"])
        hit_rate = len(actual_top & pred_top) / n

        rows.append(
            {"position": position, "spearman": spearman, f"top{top_n}_hit_rate": hit_rate, "n": len(group)}
        )
    return pd.DataFrame(rows)


TEAM_POSITION_CEILING = {"QB": 444.6, "RB": 570.3, "WR": 782.2, "TE": 460.9}
"""The highest total fantasy points any single team's players at a position
have EVER combined for in a season (2010-2025): QB 444.6 (2024 MIN), RB
570.3 (2024 DET), WR 782.2 (2016 GB), TE 460.9 (2011 NE). Superseded by
compute_team_position_ceiling as the default ceiling source (see its
docstring) - kept only as the fallback for a team/position with no pace
projection available (e.g. a relocated/expansion franchise with no play
volume history to project from).
"""

EFFICIENCY_CEILING_PERCENTILE = 0.95


def compute_efficiency_ceiling(season_stats: pd.DataFrame, rosters: pd.DataFrame, play_volume: pd.DataFrame) -> dict:
    """The 95th-percentile historical points-per-team-attempt at each
    position (2010-2025): how many fantasy points a team's players at a
    position combined for, per rush attempt (RB) or per pass attempt
    (QB/WR/TE) that team actually ran that season. A generous but bounded
    real efficiency rate - the 95th percentile of real team-seasons, not an
    unbounded "best case."
    """
    team_by_season = _team_by_season(rosters)
    ss = season_stats.merge(team_by_season, on=["player_id", "season"], how="inner")
    ss = ss.assign(total_points=ss["ppg"] * ss["games_played"])
    team_totals = ss.groupby(["team", "season", "position"])["total_points"].sum().reset_index()

    season_totals = (
        play_volume.groupby(["team", "season"])
        .agg(total_pass_plays=("pass_plays", "sum"), total_rush_plays=("rush_plays", "sum"))
        .reset_index()
    )
    merged = team_totals.merge(season_totals, on=["team", "season"], how="inner")
    merged["attempts"] = merged["total_rush_plays"].where(merged["position"] == "RB", merged["total_pass_plays"])
    merged["pts_per_attempt"] = merged["total_points"] / merged["attempts"]
    return merged.groupby("position")["pts_per_attempt"].quantile(EFFICIENCY_CEILING_PERCENTILE).to_dict()


def compute_team_position_ceiling(
    pace_pred: pd.DataFrame, efficiency_ceiling: dict, games: int = 17
) -> pd.DataFrame:
    """Team- and position-specific opportunity ceiling: a team's own
    projected attempt volume (from project_team_pace) times a generous but
    real, bounded per-attempt efficiency rate (compute_efficiency_ceiling) -
    replaces the flat TEAM_POSITION_CEILING (the same historical all-time
    max applied to every team regardless of how many plays they're actually
    projected to run) with one that scales with a team's own real pace and
    scheme.

    This is what actually "wires in" the plays-per-game model (step 1) to
    predictions: previously it was pure infrastructure, validated but
    unused. Concretely fixes the flat cap's biggest blind spot: Arizona is a
    pass-heavy team (65%+ pass rate) with comparatively little rushing
    volume, so its real RB ceiling (~420 points) is well BELOW the flat
    all-time-max (570) that applied equally to every team regardless of
    tendency - while a run-heavy team like Baltimore or a run-committed
    incoming-coach case like the Giants (see project_team_pace's HC-scheme
    docstring) gets a HIGHER ceiling than the flat constant, since they
    actually have the volume to support more combined RB production.
    """
    rows = []
    for _, row in pace_pred.iterrows():
        rush_attempts = row["plays_per_game_pred"] * (1 - row["pass_rate_pred"]) * games
        pass_attempts = row["plays_per_game_pred"] * row["pass_rate_pred"] * games
        rows.append(
            {
                "team": row["team"],
                "RB": rush_attempts * efficiency_ceiling["RB"],
                "QB": pass_attempts * efficiency_ceiling["QB"],
                "WR": pass_attempts * efficiency_ceiling["WR"],
                "TE": pass_attempts * efficiency_ceiling["TE"],
            }
        )
    return pd.DataFrame(rows).melt(id_vars="team", var_name="position", value_name="ceiling")


ROLE_SECURITY_DEPTH_THRESHOLD = {"RB": 3, "WR": 3, "TE": 2}
"""Current depth_chart_rank at or above which a player gets the role-
security discount (ROLE_SECURITY_DISCOUNT). QB has no entry - not needed,
see apply_role_security_discount's docstring.
"""

ROLE_SECURITY_DISCOUNT = {"RB": 0.78, "WR": 0.78, "TE": 0.84}


def apply_role_security_discount(board: pd.DataFrame) -> pd.DataFrame:
    """Discount ppg_pred/total_points_pred for players with no real role
    security, using CURRENT depth_chart_rank - already a live-updatable
    signal in this pipeline (sourced from load_current_depth_chart's live
    snapshot, re-pullable anytime), not a static external scrape, so this
    stays correct as real depth charts change during the season.

    Root problem (found 2026-08-27, via user pushback that the TE board
    still looked wrong after the TE_MARKET_REPLACEMENT_RANK fix): cross-
    checked our board against real-market sources (FantasyPros ECR/ADP,
    FantasyFootballCalculator's real mock drafts, FantasyCalc's real trade-
    value market) - all showed bench-tier TE/RB/WR getting real positive
    total_points_pred and mid-board ranks here despite the market treating
    many of them as nearly worthless. TE_MARKET_REPLACEMENT_RANK only
    shifts the WHOLE position's baseline - it can't target individual
    under-secure players, which is what this function does.

    IMPORTANT METHODOLOGY CORRECTION, kept here as a record of a real
    mistake caught before shipping: the first version of this function used
    thresholds/discount calibrated by matching FantasyCalc's real-market
    RANKING (confusion-matrix precision against their ~193-player valued
    universe) - which produced a much more aggressive discount (0.375) and
    an extra TE-only secondary gate for low-volume nominal starters (e.g.
    Tommy Tremble, Pat Freiermuth). That was chasing MARKET OPINION, not
    validated against REAL, REALIZED outcomes - and when actually tested
    against real outcomes it did not hold up:
    1. A `routes_run_pg * yprr` interaction feature correlates well with
       FantasyCalc's real ranking (Spearman 0.75-0.85 for TE/WR, notably
       better than our own model's own ranking) but added as an explicit
       Ridge feature and walk-forward tested (2018-2025) against ACTUAL
       next-season ppg, it changes accuracy by a rounding error (WR
       Spearman 0.7594->0.7596, TE 0.7334->0.7331) - matching the market's
       opinion better does NOT mean predicting real outcomes better. Not
       shipped.
    2. The TE low-volume-starter secondary gate came back NULL against real
       outcomes: TE1 starters with wavg_targets_pg < 3.0 (the Tremble/
       Freiermuth case) actually score ppg 1.14x their own model's
       prediction on average (i.e. UNDER-predicted, not over-predicted),
       statistically indistinguishable from higher-volume TE1 starters
       (1.08x, p=0.87). The real market's near-zero valuation of these
       specific players is not supported by how they actually perform.
       Not shipped - Freiermuth/Tremble remain a known, honest gap between
       our board and market sentiment, and the evidence says the MARKET is
       the one overreacting here, not our model.

    What DID hold up, tested the right way: CONTEMPORANEOUS (current/same-
    season, not lagged) depth_chart_rank against REAL walk-forward ppg
    residuals (2018-2025, using nfl.load_depth_charts' week-1/2 snapshot
    each season as the historical analog of "the preseason depth chart for
    the season being predicted"). This is a different, and better-founded,
    test than the EARLIER null result from this same session (depth-chart-
    rank vs. ppg residual, p=0.16-0.86) - that test used LAST season's
    depth chart to predict a bias in the NEXT season, which mostly
    duplicates information the trailing performance features already
    capture. THIS test uses the CURRENT season's own depth chart (known at
    prediction time, same treatment as contracts/coaching elsewhere in this
    pipeline) - a genuine role signal the trailing-stats-only model has no
    other way to see (a player promoted or buried on the CURRENT depth
    chart hasn't necessarily had that show up in last year's rate stats
    yet). Result: real, large, highly significant overprediction bias
    for backups at every position (RB p=5e-6, WR p=2e-6, TE p=1e-7).

    Thresholds and discount ratios both come directly from this real-
    outcomes test (mean actual ppg / mean model-predicted ppg for gated
    players, gate>=threshold, n=123-412 per position):
    - RB: depth_chart_rank>=3 (rank 2 alone wasn't a strong enough signal -
      ratio 0.937, barely biased; committee-share RB2s often keep real
      value) -> ratio 0.783
    - WR: depth_chart_rank>=3 -> ratio 0.779 (rank>=4 has too few real
      week-1/2 observations to validate at all - the position's bench
      stays too deep/inconsistently charted that far down to measure
      reliably, so this pipeline doesn't try to go deeper than rank 3)
    - TE: depth_chart_rank>=2 -> ratio 0.844
    - QB: no threshold - not tested here, and this whole investigation's
      earlier board-vs-FantasyPros check already found zero QB mismatches
      in the top 200, so there's no known problem to fix.

    Applied BEFORE apply_team_opportunity_cap in the pipeline - discounting
    a gated backup's points first means they contribute less to their
    team's summed total, so the team cap (a separate, team-level check)
    isn't needlessly triggered by a player who's already been individually
    corrected.
    """
    board = board.copy()
    depth_threshold = board["position"].map(ROLE_SECURITY_DEPTH_THRESHOLD)
    discount = board["position"].map(ROLE_SECURITY_DISCOUNT)
    gated = (board["depth_chart_rank"] >= depth_threshold).fillna(False)

    board.loc[gated, "ppg_pred"] = board.loc[gated, "ppg_pred"] * discount[gated]
    board["total_points_pred"] = board["ppg_pred"] * board["games_est"]
    return board


QB_ROLE_UPGRADE_MIN_GAMES = 8
QB_ROLE_UPGRADE_BOOST = 5.02


def apply_qb_role_upgrade_boost(board: pd.DataFrame) -> pd.DataFrame:
    """Boost ppg_pred/total_points_pred for a QB who is the CURRENT starter
    (depth_chart_rank == 1) despite a thin trailing track record
    (prev_games_played < QB_ROLE_UPGRADE_MIN_GAMES) - the complement of
    apply_role_security_discount, using the same live-updatable
    depth_chart_rank signal. Requires `had_real_starter_season` (see
    add_prior_starter_season_flag) already merged onto the board.

    Found 2026-08-27 investigating why Malik Willis (MIA's nominal 2026
    starter per the current depth chart, but a career backup with a thin,
    mostly-bad multi-year track record - 2/7/6/4 games played 2022-2025)
    ranked far below FantasyCalc's real trade-value market. The trailing-
    stats-only wavg_ features have no way to see that a player has just WON
    a starting job - a backup who takes over typically outperforms what
    their own limited-snap history alone would predict, since even a
    mediocre STARTING quarterback gets far more fantasy-relevant volume
    than a good backup ever does - QB is uniquely binary this way (RB/WR/TE
    roles are far more graduated/continuous, which is exactly why the same
    test came back null for those three positions).

    RECALIBRATED 2026-08-27, same day as the games_est redesign
    (apply_role_upgrade_durability_boost) - for the same reason. The
    original +3.45 constant was calibrated BEFORE add_prior_starter_season_
    flag existed, so it was fit on a cohort that still included players
    like Kyler Murray (an established starter with a full 2024 season, just
    hurt in 2025) mixed in with true never-started backups like Willis -
    the same conflation already found and fixed on the durability side.
    Re-derived on the correctly-narrowed cohort (excludes players with a
    real starter season - >=13 games - anywhere in their 3-year lookback):
    the bias is real and substantially BIGGER than the unrefined estimate,
    +6.63 ppg for a clean/no-recent-injury QB (p=0.002, n=12) - Murray-style
    cases, which need less correction, were diluting the original number.
    Calibrate(2018-2022)/validate(2023-2025): calibration mean +4.86,
    validation mean +5.32 (held up out-of-sample, validation p=0.061 -
    borderline given the now-smaller refined sample, but consistent in
    sign and magnitude, not shrinking toward zero). Final constant (5.02)
    is the full 2018-2025 pooled mean on the refined cohort.

    Unlike the durability side, did NOT split this by injury history
    (`had_real_injury`) - tested, and the split cohort's "had injury"
    sub-group (n=6) isn't independently significant (p=0.45) at a sample
    this small, so a single pooled constant is the honestly-supported
    choice here, not a forced split for consistency with the durability
    fix. Also confirmed unchanged from the original derivation: RB/WR/TE
    remain not significant for this pattern once refined the same way
    (WR p=0.46, TE p=0.15, both n<15) - still QB-only.

    This is exactly the kind of situation this pipeline needs to keep
    getting right automatically as real 2026 games are played: an in-
    season injury or benching that hands a backup the starting job should
    trigger this boost the next time depth charts/rosters are re-pulled,
    without needing another manual investigation.
    """
    board = board.copy()
    upgraded = (
        (board["position"] == "QB")
        & (board["depth_chart_rank"] == 1)
        & (board["prev_games_played"] < QB_ROLE_UPGRADE_MIN_GAMES)
        & (~board["had_real_starter_season"])
    ).fillna(False)

    board.loc[upgraded, "ppg_pred"] = board.loc[upgraded, "ppg_pred"] + QB_ROLE_UPGRADE_BOOST
    board["total_points_pred"] = board["ppg_pred"] * board["games_est"]
    return board


ROLE_UPGRADE_MIN_GAMES = 8

REAL_STARTER_SEASON_THRESHOLD = 13
"""A season with at least this many games played counts as real evidence
the player has actually held a starting-caliber role before - see
add_prior_starter_season_flag and apply_role_upgrade_durability_boost's
docstring for why this distinction matters.
"""

ROLE_UPGRADE_GAMES_EST_HEALTHY = {"QB": 11.00, "RB": 13.88, "WR": 12.00, "TE": 11.16}
ROLE_UPGRADE_GAMES_EST_INJURY_HISTORY = {"QB": 9.75, "RB": 13.06, "WR": 11.09, "TE": 9.36}


def add_recent_injury_history_flag(board: pd.DataFrame, injuries: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Add `had_real_injury`: did this player appear on the OFFICIAL injury
    report with a serious designation (Out, Doubtful, or IR) at any point in
    `target_season - 1` - i.e. the same season prev_games_played already
    looks back to. Deliberately excludes Questionable (a routine game-time
    tag, not evidence of a real injury) - see apply_role_upgrade_durability_
    boost's docstring for why this distinction matters for the role-upgrade
    correction specifically.
    """
    serious = injuries[injuries["report_status"].isin(["Out", "Doubtful", "IR"])].dropna(subset=["gsis_id"])
    had_injury = (
        serious[serious["season"] == target_season - 1]
        .drop_duplicates(subset="gsis_id")[["gsis_id"]]
        .rename(columns={"gsis_id": "player_id"})
    )
    had_injury["had_real_injury"] = True
    board = board.merge(had_injury, on="player_id", how="left")
    board["had_real_injury"] = board["had_real_injury"].fillna(False).astype(bool)
    return board


def add_prior_starter_season_flag(board: pd.DataFrame, season_stats: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Add `had_real_starter_season`: did this player have AT LEAST ONE
    season with >= REAL_STARTER_SEASON_THRESHOLD games played anywhere in
    the 3 seasons before `target_season` (the same lookback window used
    elsewhere in this pipeline, e.g. add_weighted_history_features).

    This is what separates two very different populations that a naive
    "prev_games_played < 8" filter alone conflates: a true backup who has
    never held a real starting role (e.g. Malik Willis: 2/7/6/4 games
    2022-2025, never above 7) versus an ESTABLISHED starter who just had
    one bad injury year (e.g. Kyler Murray: 11/8/17/5 - a real, full 17-
    game starter season as recently as 2 years before the thin one).
    Confirmed empirically why this split matters - see
    apply_role_upgrade_durability_boost's docstring.
    """
    recent = season_stats[season_stats["season"].between(target_season - 3, target_season - 1)]
    had_starter_season = (
        recent[recent["games_played"] >= REAL_STARTER_SEASON_THRESHOLD][["player_id"]]
        .drop_duplicates()
        .assign(had_real_starter_season=True)
    )
    board = board.merge(had_starter_season, on="player_id", how="left")
    board["had_real_starter_season"] = board["had_real_starter_season"].fillna(False).astype(bool)
    return board


def apply_role_upgrade_durability_boost(board: pd.DataFrame, max_games: int = 17) -> pd.DataFrame:
    """REPLACE games_est for a CURRENT starter (depth_chart_rank == 1) with a
    thin trailing games-played history (prev_games_played <
    ROLE_UPGRADE_MIN_GAMES) with the empirical outcome for players in
    exactly this situation historically - the durability counterpart to
    apply_qb_role_upgrade_boost's rate fix, general across all four
    positions, not QB-only. Requires `had_real_injury` (see
    add_recent_injury_history_flag) already merged onto the board.

    Found 2026-08-27 investigating user feedback that games_est is being
    over-weighted for this kind of player: Malik Willis, MIA's current
    nominal starter, has a games_played history of 2/7/6/4 across 2022-2025
    - a thin backup-era usage pattern that isn't really an "injury" signal
    at all, it just reflects that he wasn't the guy before. `wavg_games_
    played` treats a backup's own history the same way it treats an
    injury-prone starter's, which is a real category error: a backup's low
    games_played mostly reflects ROLE (didn't get the chance to play), not
    AVAILABILITY (couldn't play) - and once a player is actually the
    CURRENT starter, their own pre-promotion history says very little about
    how many games they'll get.

    ADDITIVE-BOOST VERSION REPLACED WITH DIRECT REPLACEMENT (2026-08-27,
    same day - the first version of this fix, an additive +N games boost
    on top of the player's own wavg_games_played, wasn't good enough and
    the user correctly kept pushing). The real problem with an additive
    patch: it assumes every player in this cohort needs the SAME fixed
    delta added to THEIR OWN starting point, but their own starting point
    (wavg_games_played, built from backup-era usage) turns out to carry
    essentially ZERO predictive signal for this cohort at all - tested
    directly: correlation(prev_games_played, actual future games_played)
    within the role-upgrade cohort is r=0.057, p=0.67, statistically
    indistinguishable from zero. A player who played 1 game last season and
    a player who played 7 end up with statistically the same real outcome
    once they're both the CURRENT starter. That means anchoring the
    estimate on the player's own history at all - even patched with a
    boost - is the wrong shape of fix, not just a miscalibrated one:
    confirmed directly by comparing prediction error on a held-out
    validation set (2023-2025) - a FLAT REPLACEMENT using the calibration-
    set (2018-2022) empirical mean games_played for this cohort has MAE
    3.906, vs MAE 6.979 using each player's own wavg_games_played - the
    flat replacement is nearly TWICE as accurate. This mirrors exactly why
    rookies get a completely separate curve (fit_rookie_curve) instead of a
    patched version of the veteran wavg_-based approach: when a player's
    own trailing history isn't a real signal for the question being asked,
    patching it is structurally wrong regardless of the patch size.

    IMPORTANT, checked before generalizing this insight: this does NOT mean
    games_est is broadly over-discounting durability for the whole player
    population - the opposite framing was tested directly and rejected.
    For CURRENT STARTERS as a whole (not just the narrow role-upgrade
    cohort), the existing shipped model is already close to unbiased (mean
    resid -0.24 games, MAE 2.94) and clearly beats a naive "assume every
    starter plays a full season" baseline (mean resid -3.74, MAE 3.74 -
    real starters really do miss real games on average, even before
    considering any specific injury risk). Broadly shrinking or removing
    the durability discount for all players would make the model WORSE,
    not better - this fix is deliberately scoped to the one specific,
    validated population (role-upgrade, thin-history-is-uninformative
    starters) where the population-level logic breaks down, not applied
    globally.

    Split by injury history (added after Willis's case still looked wrong
    even with the pooled additive boost): user asked specifically whether
    the model was dinging Willis for injury risk he doesn't actually have -
    checked his real injury report (`data/raw/injuries.parquet`) directly:
    essentially clean, one minor late-2025 "Questionable - Shoulder" tag,
    nothing serious ever. Built `had_real_injury` (a serious Out/Doubtful/
    IR tag - not the routine Questionable tag) and computed the direct
    empirical replacement mean separately for "healthy scratch" vs "had a
    real injury" within the role-upgrade cohort (2018-2025, all seasons,
    since a direct empirical mean - unlike a fitted regression coefficient
    - doesn't need a separate calibrate/validate split to be trustworthy;
    checked anyway and both cuts were stable across a 2018-2022/2023-2025
    split, before the starter-season exclusion below was added). Real,
    position-specific differences: RB and TE show a meaningful healthy-vs-
    injury gap, QB and WR show little to none - kept the split anyway since
    it's real where it matters and harmless where it's small. See
    ROLE_UPGRADE_GAMES_EST_HEALTHY/INJURY_HISTORY for the final constants,
    refit after the exclusion described next.

    EXCLUDES players with a real recent starter season (`had_real_starter_
    season`, see add_prior_starter_season_flag) - caught as a real
    regression while verifying this fix on the board: Kyler Murray (prev_
    games_played=5, from a 2025 injury) ALSO satisfies "thin last season,
    current starter," but he is nothing like Willis - Murray had a full,
    healthy 17-game starter season as recently as 2024 (games history 11/8/
    17/5). Lumping him into the same "own history is uninformative"
    treatment as a true never-started backup would have thrown away real,
    relevant signal about him specifically - and it's exactly the case the
    QB-specific bounce-back correction (estimate_games_played) already
    exists to handle correctly. Verified directly: for players excluded
    this way (a real starter season somewhere in the 3-year lookback,
    thin most-recent one), their OWN wavg_games_played DOES correlate with
    their real outcome (r=0.198, p=0.059 - much stronger than the true
    role-upgrade cohort's r=0.057, p=0.67), and their mean actual games
    (12.14) tracks reasonably close to their own wavg_games_played (9.08,
    still somewhat underestimated - which is exactly what the existing
    bounce-back correction is for). These players fall through to the
    standard estimate_games_played path unchanged, not this function.
    Constants refit on the correctly-narrowed cohort after adding this
    exclusion (values differ slightly from the pre-exclusion version).

    Concretely, for Malik Willis (clean injury history, QB, never had a
    real starter season): games_est is now set directly to 11.00 (the real
    empirical outcome for a healthy QB in his exact situation), not his own
    wavg_games_played (~4.2) plus a patch - a materially different, better-
    supported number, and one any player matching his same profile gets
    automatically, not a name-driven special case. Kyler Murray correctly
    falls through to the standard bounce-back-corrected path instead.

    Uses the same live-updatable depth_chart_rank/injury-report signals as
    every other role-transition function in this pipeline - an in-season
    promotion, or a new injury, will correctly change which constant
    applies the next time data is re-pulled.
    """
    board = board.copy()
    upgraded = (
        (board["depth_chart_rank"] == 1)
        & (board["prev_games_played"] < ROLE_UPGRADE_MIN_GAMES)
        & (~board["had_real_starter_season"])
    ).fillna(False)
    healthy_est = board["position"].map(ROLE_UPGRADE_GAMES_EST_HEALTHY)
    injury_est = board["position"].map(ROLE_UPGRADE_GAMES_EST_INJURY_HISTORY)
    replacement_est = healthy_est.where(~board["had_real_injury"], injury_est)

    board.loc[upgraded, "games_est"] = replacement_est[upgraded].clip(upper=max_games)
    board["total_points_pred"] = board["ppg_pred"] * board["games_est"]
    return board


def apply_team_opportunity_cap(board: pd.DataFrame, team_ceiling: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rescale a team's players at a position so their combined
    `total_points_pred` never exceeds a real ceiling: `team_ceiling` (one
    row per team/position, see compute_team_position_ceiling) if given,
    falling back to the flat historical-max TEAM_POSITION_CEILING for any
    team/position missing from it.

    Every player's projection (veteran or rookie) is fit independently, so
    nothing stops two players on the same team from each getting a real,
    plausible-looking projection that together add up to something no real
    team has ever produced - a team's offensive touches/targets are a
    shared, roughly fixed pie, not an unlimited resource each player draws
    from independently. Found via the 2026 board: Arizona's projected RB
    corps (Jeremiyah Love + James Conner + Tyler Allgeier + depth) summed to
    617 points - MORE than any team's RB group has ever scored in 15 years
    of data (previous record: 570, 2024 Lions). Root cause was the rookie
    curve (fit_rookie_curve) and the veteran model each producing a
    standalone "if this player gets a normal share of playing time" number
    with no awareness that a real, established teammate (freshly re-signed
    Allgeier, incumbent starter Conner) was also projected for real
    production on the very same roster.

    Fix is a proportional scale-down, not a judgment call about WHICH player
    is wrong: if a team/position group's summed total_points_pred exceeds
    the ceiling, every player in that group is scaled down by the same
    ratio, preserving each player's relative share of the (corrected) pie.
    `games_est` (durability) is left untouched - the correction is about
    diluted PER-GAME usage from real shared competition, not about how many
    games anyone plays - so `ppg_pred` absorbs the whole adjustment and
    `total_points_pred` is recomputed from the scaled `ppg_pred` * the
    original `games_est`.

    Real bug found and fixed here (2026-08-27): `groupby(["team",
    "position"])` drops rows with a NaN `team` by default (pandas' own
    groupby behavior), so `.transform("sum")` returned NaN - not a real
    total - for every player with no resolved team (real players neither
    nflverse nor Sleeper currently roster to a team, e.g. Tyreek Hill, Nick
    Chubb, Austin Ekeler as of this pull - a known, already-documented gap,
    see apply_current_team_from_sleeper). That NaN then multiplied straight
    through `ppg_pred`/`total_points_pred`, silently deleting a valid
    prediction that had already been computed - 66 players (27 WR, 22 RB,
    12 TE, 5 QB) were wiped to NaN and sorted dead last on the board. This
    was the actual root cause of an apparent "the model wildly underrates
    established veterans" pattern found via the FantasyPros comparison -
    not a modeling opinion at all, just this mechanical NaN propagation.
    Fixed by treating "no team to check a ceiling against" as "no cap
    applies" (scale 1.0) rather than an undefined ratio.
    """
    board = board.copy()
    ceiling = board["position"].map(TEAM_POSITION_CEILING)
    if team_ceiling is not None:
        specific = board.merge(team_ceiling, on=["team", "position"], how="left")["ceiling"]
        specific.index = board.index
        ceiling = specific.fillna(ceiling)
    team_totals = board.groupby(["team", "position"])["total_points_pred"].transform("sum")
    scale = (ceiling / team_totals).clip(upper=1.0).fillna(1.0)
    board["ppg_pred"] = board["ppg_pred"] * scale
    board["total_points_pred"] = board["ppg_pred"] * board["games_est"]
    return board


MAN_GAMES_DEPTH_MULTIPLIER = {"QB": 1.06, "RB": 1.11, "TE": 1.10, "WR": 1.08}

TE_MARKET_REPLACEMENT_RANK = 10
"""0-indexed replacement rank for TE only (position 10 = the 11th-best TE by
total_points_pred, i.e. "TE11") - overrides the man-games/flex formula for
TE specifically. See compute_vbd's docstring for the full derivation; the
short version is that TE11 is where FantasyPros' real overall ECR rank
(113) lines up with RB's own already-validated replacement-level bar (111),
which is a much shallower/shorter bench than the man-games formula alone
implies (~TE14, 0-indexed 13).
"""

FLEX_ALLOCATION = {"RB": 0.16, "WR": 0.79, "TE": 0.05}
"""How much of each flex slot's replacement-depth credit goes to RB/WR/TE.

Previously a flat 45/45/10 rule of thumb. Measured empirically instead:
for every REG-season week 2010-2025, locked in the top teams*slots players
at each of RB/WR/TE by that week's realized fantasy points as dedicated
starters, then looked at the next teams*flex_slots best remaining RB/WR/TE
players (who'd actually fill the flex spot that week) and tallied their
position. Over the last 10 seasons (2016+): WR takes ~79% of flex value,
RB ~16%, TE ~4% (rounded up slightly here since TE has grown post-2023) -
WR dominates because the position stays productive much deeper down the
list (WR40 can still have a real week), while usable RB and TE production
falls off a cliff right after the dedicated starter slots. This actually
reverses the naive 45/45/10 assumption for RB specifically: it shrinks
RB's flex credit rather than granting it a near-equal share, which lowers
RB's replacement rank (fewer effective bodies count) and evenly increases
every RB's VBD - consistent with RB being the scarcer, more front-loaded
position in redraft value, and with real ADP behavior (RBs go early
precisely because so few remain useful past the top of the position).
"""


def compute_vbd(
    board: pd.DataFrame,
    teams: int = 12,
    qb_slots: int = 1,
    rb_slots: int = 2,
    wr_slots: int = 2,
    te_slots: int = 1,
    flex_slots: int = 1,
) -> pd.DataFrame:
    """Value Based Drafting: rank each player within their position by
    projected total points, then subtract the projection of the "replacement
    level" player at that position - the best player who'd likely still be on
    the wire given your league's roster requirements. This is what actually
    drives draft order: the RB12 in a shallow class is worth more than a
    similarly-projected WR12 in a deep one, and raw points alone can't tell
    you that.

    Flex slots are split across RB/WR/TE using FLEX_ALLOCATION, an empirical
    split rather than a rule-of-thumb guess (see its own docstring below) -
    override the defaults if your league's roster requirements differ.

    Note on QB: a naive teams*qb_slots replacement rank (QB12 in a standard
    12-team league) already produces top-5 QB VBD in the 40-80 range cited by
    public VORP methodology (sticktothemodel.com/FantasyPros) - verified
    against this project's own 2026 predictions (Josh Allen 66.5, down to
    Drake Maye 45.5). A "deepen the replacement rank to QB17" adjustment was
    tried and reverted: it's mathematically backwards - a DEEPER replacement
    rank means a LOWER-scoring replacement player, which makes the subtracted
    baseline smaller and VBD LARGER, the opposite of the intended effect.
    Deliberately not "fixed" further since the naive formula already matches
    the reference range without adjustment.

    Man-games replacement depth (MAN_GAMES_DEPTH_MULTIPLIER): the static
    teams*slots rank assumes exactly that many players are needed all
    season, but real rosters churn through more bodies than that because
    even the best players at a position miss games (byes, injuries) - so the
    TRUE freely-available replacement level sits a bit deeper than the
    static count. Measured directly from this project's own season_stats
    (2010-2025): for each season, took the top teams*slots players at each
    position by realized total points, and averaged what fraction of a
    17-game season they actually played. RB players who make the cut still
    only average ~90% game availability (2016+: 89.9%), WR ~92%, TE ~91%,
    QB ~95% (QBs are hurt less and rarely committee'd) - i.e. even "the
    guys good enough to matter" miss real time, so a full season of
    starter-quality man-games requires roughly 1/availability as many
    rostered bodies as the naive slot count. Multipliers are that ratio
    (1/availability, 2016+ seasons): QB 1.06, RB 1.11, TE 1.10, WR 1.08 -
    RB deepens the most, matching RB's well-known injury volatility.
    Deliberately NOT the much larger (3-5x) ratio you'd get from counting
    every player who ever had one boom week in the starter tier - that
    conflates real rostered depth with one-off waiver-wire flukes and
    would blow the replacement bar out to an implausible depth. This
    correctly compounds with (not fights) the QB note above: QB's own
    multiplier is the smallest of the four, so QB's already-validated
    replacement level barely moves.

    TE market-depth override (TE_MARKET_REPLACEMENT_RANK): unlike QB, this
    one IS a real, evidenced gap, found via the 2026-08-27 FantasyPros
    top-250 comparison. The man-games formula above already lines up with
    FantasyPros' own points-based VORP methodology for TE (~TE16 - see
    fantasypros.com/nfl/rankings/ppr-vorp-te.php) - so "how many TEs get
    rostered" isn't the problem. The problem is that FantasyPros' actual
    aggregated expert DRAFT ORDER (redraft-overall ECR, a different FP
    product measuring real market behavior, not their own points formula)
    craters non-elite TE value far more steeply: Pat Freiermuth, this
    project's own TE15/replacement-level player (VBD=0, our overall rank
    93 - a plausible low-end starter), sits at FantasyPros' TE28, overall
    rank 241 - outside the top 240 fantasy-relevant players entirely. Their
    own writeup gives the mechanism: TE needs only one roster slot and is
    genuinely streamable off waivers week to week in a way RB/WR aren't (a
    mediocre bench RB/WR still holds flex/injury-fill value; a mediocre
    bench TE largely doesn't) - a real roster-construction effect a static
    points-above-replacement formula can't reproduce by itself, no matter
    how MAN_GAMES_DEPTH_MULTIPLIER/FLEX_ALLOCATION are tuned.

    Checked this wasn't a general "our replacement ranks are all wrong"
    problem first: found each position's own replacement-level player and
    compared OUR overall rank for them against FantasyPros' real overall
    rank for that same player. RB (rank 96 vs 111) and QB (94 vs 139) line
    up reasonably; only TE is wildly off (93 vs 241) - confirms this is
    TE-specific, not a formula-wide issue.

    TE_MARKET_REPLACEMENT_RANK is set to 11, not derived from a sharp
    "cliff" (there isn't one - FantasyPros' TE overall-rank-per-position-
    rank slope is a fairly steady ~7.8 ranks/position throughout, roughly
    3-4x steeper than RB's ~2.8 and WR's ~2.1, and even steeper than QB's
    ~6.4 - TE just declines faster throughout its whole range, not
    flat-then-cliff). Instead calibrated against RB's own replacement level
    as an anchor, since RB (unlike the TE formula) was already confirmed to
    track FantasyPros' real market well: RB's replacement-level player sits
    at FantasyPros overall rank 111; TE11 (FantasyPros overall rank 113,
    Dalton Kincaid at the time of this check) is the closest TE position-
    rank match to that same real-market bar. This directly overrides the
    man-games-formula rank for TE only (RB/WR/QB keep the man-games/flex
    formula, which the same evidence shows already works for them) - unlike
    the QB attempt (see above), this shallows the rank, which correctly
    LOWERS every TE's VBD (a shallower rank means a higher-scoring
    replacement baseline, matching the intended direction this time).
    """
    board = board.copy()
    board["position_rank"] = board.groupby("position")["total_points_pred"].rank(
        ascending=False, method="first"
    )

    replacement_rank = {
        "QB": teams * qb_slots * MAN_GAMES_DEPTH_MULTIPLIER["QB"],
        "RB": teams * (rb_slots + flex_slots * FLEX_ALLOCATION["RB"]) * MAN_GAMES_DEPTH_MULTIPLIER["RB"],
        "WR": teams * (wr_slots + flex_slots * FLEX_ALLOCATION["WR"]) * MAN_GAMES_DEPTH_MULTIPLIER["WR"],
        "TE": TE_MARKET_REPLACEMENT_RANK,
    }

    replacement_points = {}
    for position, rank in replacement_rank.items():
        pos_board = board[board["position"] == position].sort_values("total_points_pred", ascending=False)
        if pos_board.empty:
            replacement_points[position] = 0.0
            continue
        idx = min(int(round(rank)), len(pos_board) - 1)
        replacement_points[position] = pos_board["total_points_pred"].iloc[idx]

    board["replacement_points"] = board["position"].map(replacement_points)
    board["vbd"] = board["total_points_pred"] - board["replacement_points"]
    return board


QB_STARTER_FLOOR_PPG = 11.90
"""10th-percentile PPG among real historical QB seasons with 12+ games
started (2010-2025, n=416) - see apply_qb_starter_floor's docstring.
"""


def apply_qb_starter_floor(board: pd.DataFrame) -> pd.DataFrame:
    """Give any CURRENT QB1 (depth_chart_rank == 1) a minimum VBD, so a real,
    rostered starting quarterback can never rank below players at other
    positions who will genuinely never see the field - a mechanical flaw in
    plain cross-position VBD comparison, not a rate/durability estimation
    problem.

    Found 2026-08-27, user pushback on Malik Willis's ranking (~600th
    overall) even after both role-upgrade fixes above. Checked the actual
    absolute numbers first, since a rate/durability tweak had already been
    tried twice: Willis's total_points_pred (~107) is NOT unrealistic on
    its own - real historical QB seasons with 12+ games started have a 10th-
    percentile total of 168.8 points, and the worst ones on record (Jimmy
    Clausen 2010, Derek Anderson 2010) still cleared 58-90. The problem is
    entirely in the SUBTRACTION: QB's replacement level sits around ~254
    points (naive QB13, already separately validated for the TOP of the
    position against real VORP reference ranges - see compute_vbd's QB
    note) - so any below-replacement-but-real starter gets an enormous
    negative VBD (-147 for Willis), which lands him in the same overall-
    rank neighborhood as a WR11 who will never play a snap and has ~5-10
    total points - a 10x+ gap in ABSOLUTE production that VBD's linear,
    same-bar-for-everyone subtraction can't see once both players are
    "very below replacement."

    This is specific to QB, not a general VBD flaw, for a real structural
    reason: a real starting QB plays essentially 100% of offensive snaps
    whenever active (unlike RB/WR/TE, where even a nominal "starter" is
    often a committee/timeshare) - depth_chart_rank==1 at QB is a much
    stronger, more literal guarantee of a real, full role than the same
    rank at any other position. That's also why this fix does NOT
    generalize automatically to RB/WR/TE: their own "starter floor," if
    warranted, would need to be derived from and scaled by their OWN real
    snap share (a true bell-cow WR/RB plays a very different share of
    snaps than a committee "starter" at the same depth-chart rank) - not
    attempted here, flagged as a distinct follow-up research question, not
    assumed to carry over with the same logic or magnitude.

    Floor is expressed as a PER-GAME rate (QB_STARTER_FLOOR_PPG, the 10th-
    percentile PPG among real 12+ game QB seasons, 2010-2025) multiplied by
    the player's OWN games_est, not a flat season-total floor - this
    deliberately composes with whatever games_est the model has already
    (validly) settled on, rather than assuming a player will necessarily
    play a full season. A QB with a genuinely low games_est (real injury/
    competition uncertainty) still gets a proportionally smaller floor, not
    the full-season amount.

    Does NOT touch total_points_pred/ppg_pred/games_est themselves - this
    is a ranking/comparison-mechanism fix, not a claim that the underlying
    point estimate was wrong. Applied after compute_vbd (needs
    replacement_points, which compute_vbd attaches to the board).
    """
    board = board.copy()
    is_current_starter = (board["position"] == "QB") & (board["depth_chart_rank"] == 1)
    floor_vbd = QB_STARTER_FLOOR_PPG * board["games_est"] - board["replacement_points"]
    board.loc[is_current_starter, "vbd"] = np.maximum(
        board.loc[is_current_starter, "vbd"], floor_vbd[is_current_starter]
    )
    return board


def build_team_offense_summary(board: pd.DataFrame, games: int = 17) -> pd.DataFrame:
    """Roll the final per-player draft board up to a team-level 2026 offense
    projection: total projected fantasy points across every rostered
    QB/RB/WR/TE, and the same total expressed as points-per-game.

    This is a straight aggregation of the SAME per-player predictions
    already on the board (each player's own total_points_pred already
    reflects our estimate of their realistic role/games, so a low-usage
    backup contributes little and doesn't need to be filtered out by hand)
    - not a separately trained model. Meant as a "which offenses project
    best/worst in fantasy-point terms for 2026" summary, not a play-calling
    or real-world scoring projection.
    """
    skill = board[board["position"].isin(["QB", "RB", "WR", "TE"])]
    summary = skill.groupby("team")["total_points_pred"].sum().reset_index(name="team_total_points_pred")
    summary["team_ppg_pred"] = summary["team_total_points_pred"] / games
    return summary.sort_values("team_ppg_pred", ascending=False).reset_index(drop=True)
