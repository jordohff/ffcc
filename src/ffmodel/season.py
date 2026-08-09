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

COMMON_VET_FEATURES = [
    "prev_games_played",
    "wavg_ppg",
    "age",
    "age_squared",
    "years_past_decline_age",
    "team_changed",
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
]

QB_VET_FEATURES = COMMON_VET_FEATURES + [
    "wavg_carries_pg",
    "wavg_attempts_pg",
    "wavg_passing_yards_pg",
    "wavg_passing_tds_pg",
    "wavg_passing_epa_pg",
]

POSITION_VET_FEATURES = {
    "QB": QB_VET_FEATURES,
    "RB": SKILL_VET_FEATURES,
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
    """
    games_played = (
        enriched_weekly.groupby(["player_id", "player_display_name", "position", "season"])["week"]
        .nunique()
        .reset_index(name="games_played")
    )

    per_game = (
        enriched_weekly.groupby(["player_id", "season"])
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
        )
        .reset_index()
    )

    return games_played.merge(per_game, on=["player_id", "season"], how="left")


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
    """Add non-linear age features on top of the plain `age` column, so a
    linear model (Ridge) can still fit a realistic career-arc shape instead
    of a straight line.

    `age_squared` lets the model fit a general rise-then-fall parabola.
    `years_past_decline_age` is a position-specific "hinge" that's zero until
    a player passes the age where THAT position's production typically falls
    off a cliff (see DECLINE_AGE), then grows 1-for-1 after - e.g. a
    35-year-old RB gets years_past_decline_age = 8, a strong, explicit
    "this specific player is well past the cliff" signal.

    This exists specifically to counteract a side effect of
    add_weighted_history_features: blending in a strong season from a few
    years back correctly helps a player mean-reverting from a fluky down
    year, but WRONGLY inflates an aging player whose decline is real and
    structural, not noise - the model needs some way to tell those two cases
    apart, and age is the only signal available for that distinction.
    """
    table = table.copy()
    table["age_squared"] = table["age"] ** 2
    decline_age = table["position"].map(DECLINE_AGE).fillna(30)
    table["years_past_decline_age"] = (table["age"] - decline_age).clip(lower=0)
    return table


def add_team_change_feature(table: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Add `team_changed`: did this player switch teams between the season
    used as features (season - 1) and the season being predicted?

    A team change (trade, free agency, cut-and-signed) can meaningfully
    change a player's role independent of their own talent/health, so it's
    worth flagging even though we don't model exactly which direction it'll
    push a given player.
    """
    team_by_season = (
        rosters.groupby(["gsis_id", "season"])["team"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
        .rename(columns={"gsis_id": "player_id"})
    )
    prev_team = team_by_season.rename(columns={"team": "prev_team"})
    prev_team = prev_team.assign(season=prev_team["season"] + 1)

    table = table.merge(team_by_season, on=["player_id", "season"], how="left")
    table = table.merge(prev_team, on=["player_id", "season"], how="left")
    table["team_changed"] = (
        table["team"].notna() & table["prev_team"].notna() & (table["team"] != table["prev_team"])
    ).astype(int)
    return table


def add_weighted_history_features(
    table: pd.DataFrame,
    season_stats: pd.DataFrame,
    healthy_season_stats: pd.DataFrame | None = None,
    weights: list[float] = HISTORY_WEIGHTS,
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

    for lag, _ in enumerate(weights, start=1):
        games_col = f"_lag{lag}_games_played"
        games_shifted = season_stats[["player_id", "season", "games_played"]].rename(
            columns={"games_played": games_col}
        )
        games_shifted = games_shifted.assign(season=games_shifted["season"] + lag)
        table = table.merge(games_shifted, on=["player_id", "season"], how="left")
        lag_cols_by_stat["games_played"].append(games_col)

        rate_cols = [f"_lag{lag}_{c}" for c in RATE_STAT_COLUMNS]
        rate_shifted = healthy_season_stats.rename(columns=dict(zip(RATE_STAT_COLUMNS, rate_cols)))
        rate_shifted = rate_shifted.assign(season=rate_shifted["season"] + lag)
        table = table.merge(
            rate_shifted[["player_id", "season", *rate_cols]], on=["player_id", "season"], how="left"
        )
        for stat, col in zip(RATE_STAT_COLUMNS, rate_cols):
            lag_cols_by_stat[stat].append(col)

    weight_arr = np.array(weights)
    for stat, cols in lag_cols_by_stat.items():
        values = table[cols].to_numpy(dtype=float)
        available = ~np.isnan(values)
        weighted_sum = np.nansum(values * weight_arr, axis=1)
        weight_total = (available * weight_arr).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            table[f"wavg_{stat}"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
        table = table.drop(columns=cols)

    return table


def build_season_training_table(
    season_stats: pd.DataFrame, rosters: pd.DataFrame, healthy_season_stats: pd.DataFrame | None = None
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
    prior_games = season_stats.rename(columns={"games_played": "prev_games_played"})
    prior_games = prior_games.assign(season=prior_games["season"] + 1)
    prior_games = prior_games[["player_id", "season", "prev_games_played"]]

    table = season_stats.merge(prior_games, on=["player_id", "season"], how="inner")
    table = add_weighted_history_features(table, season_stats, healthy_season_stats)
    table = add_age_feature(table, rosters)
    table = add_age_curve_features(table)
    table = add_team_change_feature(table, rosters)
    return table


def build_prediction_features(
    season_stats: pd.DataFrame,
    target_season: int,
    rosters: pd.DataFrame,
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
    table = prior[["player_id", "player_display_name", "position", "games_played"]].rename(
        columns={"games_played": "prev_games_played"}
    )
    table["season"] = target_season
    table = add_weighted_history_features(table, season_stats, healthy_season_stats)
    table = add_age_feature(table, rosters)
    table = add_age_curve_features(table)
    table = add_team_change_feature(table, rosters)
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
    drives draft order: the RB QB12 in a shallow class is worth more than a
    similarly-projected WR12 in a deep one, and raw points alone can't tell
    you that.

    Flex slots are split across RB/WR/TE using a standard rule of thumb (most
    flex starts go to RB/WR, TE less often) - override the defaults if your
    league's roster requirements differ.
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
