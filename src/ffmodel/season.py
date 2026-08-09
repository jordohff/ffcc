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

COMMON_VET_FEATURES = ["prev_games_played", "prev_ppg", "age", "team_changed"]

SKILL_VET_FEATURES = COMMON_VET_FEATURES + [
    "prev_targets_pg",
    "prev_carries_pg",
    "prev_target_share",
    "prev_routes_run_pg",
    "prev_tprr",
    "prev_yprr",
    "prev_separation",
    "prev_cushion",
    "prev_ngs_air_yards_share",
    "prev_yac_above_exp",
]

QB_VET_FEATURES = COMMON_VET_FEATURES + [
    "prev_carries_pg",
    "prev_attempts_pg",
    "prev_passing_yards_pg",
    "prev_passing_tds_pg",
    "prev_passing_epa_pg",
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


def build_season_training_table(season_stats: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Build the veteran training table: for each player-season Y where we
    also have that player's season Y-1 stats, use season Y-1 (prefixed
    `prev_`) as features and season Y's `ppg` as the target.

    Players with no season Y-1 row (true rookies, or anyone who missed the
    entire prior season) are excluded here - the model literally has nothing
    to condition on for them. They're handled by the separate rookie model
    (see build_rookie_training_table), which uses draft capital instead.
    """
    prior = season_stats.rename(columns={c: f"prev_{c}" for c in SEASON_STAT_COLUMNS})
    prior = prior.assign(season=prior["season"] + 1)
    prior = prior[["player_id", "season", *[f"prev_{c}" for c in SEASON_STAT_COLUMNS]]]

    table = season_stats.merge(prior, on=["player_id", "season"], how="inner")
    table = add_age_feature(table, rosters)
    table = add_team_change_feature(table, rosters)
    return table


def build_prediction_features(
    season_stats: pd.DataFrame, target_season: int, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Build feature rows for predicting `target_season`, which hasn't been
    played yet (so it has no row of its own in season_stats) - from each
    returning player's most recent prior season of stats.

    Same shifted-column construction as the `prev_` half of
    build_season_training_table, just for a single target season that
    doesn't need to already exist in the data. Only includes players who
    have a season_stats row for target_season - 1 (i.e. played last season) -
    true rookies with zero NFL history are handled separately (see
    build_rookie_training_table/project_rookies).
    """
    prior = season_stats[season_stats["season"] == target_season - 1].copy()
    prior = prior.rename(columns={c: f"prev_{c}" for c in SEASON_STAT_COLUMNS})
    keep = ["player_id", "player_display_name", "position", *[f"prev_{c}" for c in SEASON_STAT_COLUMNS]]
    table = prior[keep].copy()
    table["season"] = target_season
    table = add_age_feature(table, rosters)
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
    this season's (prefixed `prev_`) stats + age + team_changed.
    """
    models = {}
    for position, feature_cols in POSITION_VET_FEATURES.items():
        pos_train = train[train["position"] == position].dropna(subset=["prev_ppg"])
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

    Note: this only includes rookies who recorded at least one game that
    season - a rookie who was hurt all year and never played contributes
    nothing to the average, which means the historical average is implicitly
    "conditional on playing at least one game," not a true expected value
    across the whole drafted class. A reasonable v1 simplification, but worth
    knowing when reading the numbers.
    """
    picks = draft_picks.dropna(subset=["gsis_id"]).rename(
        columns={"gsis_id": "player_id", "season": "draft_season"}
    )
    # Use season_stats' own charted position (actual NFL usage), not the
    # draft position, in case they ever differ (e.g. a college DB drafted
    # as a WR project) - drop draft_picks' copy to avoid a silent
    # position_x/position_y column collision on merge.
    picks = picks.drop(columns="position")
    rookie_stats = season_stats.merge(picks, on="player_id", how="inner")
    rookie_stats = rookie_stats[rookie_stats["season"] == rookie_stats["draft_season"]]
    rookie_stats = rookie_stats.assign(round_bucket=_round_bucket(rookie_stats["round"]))
    return rookie_stats


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


def estimate_games_played(prev_games_played: pd.Series, max_games: int = 17) -> pd.Series:
    """Durability estimate: assume similar games played to last season,
    capped at a full season.

    Deliberately a simple, transparent heuristic rather than a trained model -
    it's easy to see (and second-guess) exactly what it's assuming for a
    given player, which matters more for a durability estimate than squeezing
    out a bit more accuracy from a black-box model on a genuinely hard
    problem (in-season injuries are close to unpredictable in advance).
    """
    return prev_games_played.clip(upper=max_games)


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
