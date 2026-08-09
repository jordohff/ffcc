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
    return table


def build_prediction_features(
    season_stats: pd.DataFrame,
    target_season: int,
    rosters: pd.DataFrame,
    schedules: pd.DataFrame,
    snap_share: pd.DataFrame,
    contract_history: pd.DataFrame,
    healthy_season_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature rows for predicting `target_season`, which hasn't been
    played yet (so it has no row of its own in season_stats) - from each
    returning player's recency-weighted recent history.

    Same construction as build_season_training_table, just for a single
    target season that doesn't need to already exist in the data. Only
    includes players who have a season_stats row for target_season - 1 (i.e.
    played last season) - true rookies with zero NFL history are handled
    separately (see build_rookie_training_table/project_rookies).
    """
    prior = season_stats[season_stats["season"] == target_season - 1].copy()
    table = prior[
        ["player_id", "player_display_name", "position", "games_played", "made_playoffs", "touches"]
    ].rename(
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


def _round_bucket(round_num: pd.Series) -> pd.Series:
    """Collapse draft round into 1/2/3/4/'5-7' buckets. Rounds 5-7 ("Day 3")
    are grouped together since per-round rookie sample sizes get noisy fast
    once you're only looking at, say, round-6 tight ends across 15 seasons.
    """
    return round_num.clip(upper=5).map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5-7"})


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
    combined = combined.assign(round_bucket=_round_bucket(combined["round"]))
    return combined


def fit_rookie_averages(rookie_table: pd.DataFrame) -> pd.DataFrame:
    """Historical average rookie-season PPG and games played, by position and
    draft-round bucket - the whole "model" for projecting incoming rookies,
    since there's no prior-season data to anchor a regression on.
    """
    return (
        rookie_table.groupby(["position", "round_bucket"])
        .agg(rookie_ppg=("ppg", "mean"), rookie_games=("games_played", "mean"), n=("ppg", "size"))
        .reset_index()
    )


def project_rookies(current_draft_picks: pd.DataFrame, rookie_averages: pd.DataFrame) -> pd.DataFrame:
    """Apply historical rookie-round averages to this year's actual draft
    class, giving each rookie a `ppg_pred` and `games_est` the same way
    veterans get one from the trained model.
    """
    picks = current_draft_picks[current_draft_picks["position"].isin(POSITION_VET_FEATURES)].copy()
    picks["round_bucket"] = _round_bucket(picks["round"])
    picks = picks.merge(rookie_averages, on=["position", "round_bucket"], how="left")
    picks = picks.rename(
        columns={"rookie_ppg": "ppg_pred", "rookie_games": "games_est", "gsis_id": "player_id"}
    )
    return picks


def estimate_games_played(weighted_games_played: pd.Series, max_games: int = 17) -> pd.Series:
    """Durability estimate: recency-weighted average games played over the
    last few seasons (see HISTORY_WEIGHTS/add_weighted_history_features),
    capped at a full season.

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
    """
    return weighted_games_played.clip(upper=max_games)


# Sleeper and nflverse otherwise agree on team codes, but use different
# abbreviations for these two teams - normalize to nflverse's convention
# (used everywhere else in this pipeline) before comparing/using Sleeper's
# team field, or every Cardinals/Rams player falsely shows up as a "mismatch".
SLEEPER_TEAM_CODE_FIXES = {"ARI": "AZ", "LAR": "LA"}


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

    Flex slots are split across RB/WR/TE using a standard rule of thumb (most
    flex starts go to RB/WR, TE less often) - override the defaults if your
    league's roster requirements differ.

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
    """
    board = board.copy()
    board["position_rank"] = board.groupby("position")["total_points_pred"].rank(
        ascending=False, method="first"
    )

    replacement_rank = {
        "QB": teams * qb_slots,
        "RB": teams * (rb_slots + flex_slots * 0.45),
        "WR": teams * (wr_slots + flex_slots * 0.45),
        "TE": teams * (te_slots + flex_slots * 0.10),
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
