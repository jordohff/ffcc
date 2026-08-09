"""Feature engineering: turn raw weekly stat lines into model-ready features.

The rule that matters most in this file: every feature for a player's week-W row
must only use information available BEFORE week W (earlier games). If we let a
feature peek at the very performance we're trying to predict, the model looks
great in testing and is useless in real projections. We enforce this with
`shift(1)` before every rolling average below - "look at past rows only."
"""

import pandas as pd

WINDOWS = (3, 5)
MATCHUP_WINDOW = 5

# Raw stat column -> short name used in the generated feature (avg_<name>_last{N}).
# Applies to every position; a stat that's meaningless for a position (e.g.
# receiving targets for a QB) just comes out ~0 and that position's model
# simply doesn't use that column (see POSITION_FEATURE_COLUMNS below).
ROLLING_STATS = {
    "fantasy_pts": "fantasy_points_target",
    "targets": "targets",
    "carries": "carries",
    "target_share": "target_share",
    "attempts": "attempts",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_epa": "passing_epa",
    "routes_run": "routes_run",
    "tprr": "tprr",
    "yprr": "yprr",
    "separation": "avg_separation",
    "cushion": "avg_cushion",
    "ngs_air_yards_share": "ngs_air_yards_share",
    "yac_above_exp": "avg_yac_above_expectation",
}

COMMON_FEATURES = [
    "avg_fantasy_pts_last3",
    "avg_fantasy_pts_last5",
    "opp_avg_pts_allowed_last5",
    "is_home",
    "rest_days",
    "implied_team_total",
    "injury_severity",
]

# Ordinal encoding of that week's official pre-game injury report status.
INJURY_SEVERITY = {"Questionable": 1, "Doubtful": 2, "Out": 3}

# RB/WR/TE scoring is driven by receiving/rushing volume, plus route
# participation and separation, which are receiver-specific (not meaningful
# for QBs, who don't run routes or get charted for separation).
SKILL_POSITION_FEATURES = COMMON_FEATURES + [
    "avg_targets_last3",
    "avg_targets_last5",
    "avg_carries_last3",
    "avg_carries_last5",
    "avg_target_share_last3",
    "avg_target_share_last5",
    "avg_routes_run_last3",
    "avg_routes_run_last5",
    "avg_tprr_last3",
    "avg_tprr_last5",
    "avg_yprr_last3",
    "avg_yprr_last5",
    "avg_separation_last3",
    "avg_separation_last5",
    "avg_cushion_last3",
    "avg_cushion_last5",
    "avg_ngs_air_yards_share_last3",
    "avg_ngs_air_yards_share_last5",
    "avg_yac_above_exp_last3",
    "avg_yac_above_exp_last5",
]

# QB scoring is driven by passing volume/efficiency, plus rushing floor for
# mobile QBs (targets/target_share don't apply - QBs aren't targeted).
QB_FEATURES = COMMON_FEATURES + [
    "avg_carries_last3",
    "avg_carries_last5",
    "avg_attempts_last3",
    "avg_attempts_last5",
    "avg_passing_yards_last3",
    "avg_passing_yards_last5",
    "avg_passing_tds_last3",
    "avg_passing_tds_last5",
    "avg_passing_epa_last3",
    "avg_passing_epa_last5",
]

POSITION_FEATURE_COLUMNS = {
    "QB": QB_FEATURES,
    "RB": SKILL_POSITION_FEATURES,
    "WR": SKILL_POSITION_FEATURES,
    "TE": SKILL_POSITION_FEATURES,
}

# Union of every feature used by any position - only used for the output CSV
# and for cache-warming; each position's model only sees its own subset above.
FEATURE_COLUMNS = sorted(set(QB_FEATURES) | set(SKILL_POSITION_FEATURES))


def add_fantasy_target(df: pd.DataFrame, scoring: str = "half_ppr") -> pd.DataFrame:
    """Add a `fantasy_points_target` column: the thing we're trying to predict.

    nflreadpy gives us `fantasy_points` (standard, 0 pts/reception) and
    `fantasy_points_ppr` (full PPR, 1 pt/reception) directly. Half-PPR (0.5
    pts/reception) sits exactly halfway between those two, so we can derive it
    by averaging them instead of re-deriving a scoring formula ourselves.
    """
    df = df.copy()
    if scoring == "ppr":
        df["fantasy_points_target"] = df["fantasy_points_ppr"]
    elif scoring == "half_ppr":
        df["fantasy_points_target"] = (df["fantasy_points"] + df["fantasy_points_ppr"]) / 2
    else:
        raise ValueError(f"Unknown scoring format: {scoring!r} (use 'half_ppr' or 'ppr')")
    return df


def _trailing_mean(series: pd.Series, window: int) -> pd.Series:
    """Average of the previous `window` rows, excluding the current row."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_recent_performance_features(
    df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS, stats: dict[str, str] = ROLLING_STATS
) -> pd.DataFrame:
    """Add trailing rolling averages of fantasy points and usage per player.

    E.g. `avg_fantasy_pts_last3` = that player's average fantasy points over
    their previous 3 games (not counting the game in that row). Runs this for
    every (name, raw_column) pair in `stats`, across every window.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id")

    for name, column in stats.items():
        for w in windows:
            df[f"avg_{name}_last{w}"] = grouped[column].transform(lambda s, w=w: _trailing_mean(s, w))
    return df


def add_matchup_features(df: pd.DataFrame, window: int = MATCHUP_WINDOW) -> pd.DataFrame:
    """Add how many fantasy points a player's opponent has recently allowed
    to that position - a simple stand-in for "is this a good or bad matchup."
    """
    df = df.copy()

    # Points allowed by (team, position) in a given week = total fantasy points
    # scored by all QB/RB/WR/TE who played against that team that week.
    points_allowed = (
        df.groupby(["opponent_team", "position", "season", "week"])["fantasy_points_target"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team", "fantasy_points_target": "points_allowed"})
        .sort_values(["defense_team", "position", "season", "week"])
    )
    points_allowed[f"opp_avg_pts_allowed_last{window}"] = points_allowed.groupby(
        ["defense_team", "position"]
    )["points_allowed"].transform(lambda s: _trailing_mean(s, window))

    df = df.merge(
        points_allowed[
            ["defense_team", "position", "season", "week", f"opp_avg_pts_allowed_last{window}"]
        ],
        left_on=["opponent_team", "position", "season", "week"],
        right_on=["defense_team", "position", "season", "week"],
        how="left",
    ).drop(columns="defense_team")
    return df


def _team_week_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape schedules (one row per game) into one row per team per week,
    with that team's home/away status, days of rest, and Vegas-implied point
    total going into that specific game.

    `spread_line` is the closing spread relative to the home team - positive
    means the home team was favored by that many points (confirmed by
    checking its correlation with actual home-minus-away results). `total_line`
    is the over/under for the combined score. Splitting the total using the
    spread gives each team its own implied score:
        home_implied_total = total/2 + spread/2   (favored team gets more)
        away_implied_total = total/2 - spread/2
    This is a sharper, game-specific "is this a good matchup" signal than a
    trailing average of points allowed, since it reflects that week's actual
    Vegas expectations (blowout vs. shootout vs. defensive grind).
    """
    s = schedules.copy()
    s["home_implied_total"] = s["total_line"] / 2 + s["spread_line"] / 2
    s["away_implied_total"] = s["total_line"] / 2 - s["spread_line"] / 2

    home = s[["season", "week", "home_team", "home_rest", "home_implied_total"]].rename(
        columns={
            "home_team": "team",
            "home_rest": "rest_days",
            "home_implied_total": "implied_team_total",
        }
    )
    home["is_home"] = 1
    away = s[["season", "week", "away_team", "away_rest", "away_implied_total"]].rename(
        columns={
            "away_team": "team",
            "away_rest": "rest_days",
            "away_implied_total": "implied_team_total",
        }
    )
    away["is_home"] = 0
    return pd.concat([home, away], ignore_index=True)


def add_schedule_context(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Add `is_home`, `rest_days`, and `implied_team_total` for each player's
    team/week, from the schedule rather than the play-by-play stat lines.
    """
    context = _team_week_context(schedules)
    return df.merge(context, on=["season", "week", "team"], how="left")


def _player_position_lookup(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Each player's most common charted position across all their weekly
    rows - used to tag participation data with a position, since
    participation's own `offense_positions` field is only populated from
    2023 onward (see data.load_participation) and we need this back to 2016.
    """
    return (
        weekly_stats.groupby("player_id")["position"]
        .agg(lambda s: s.mode().iat[0])
        .reset_index()
    )


def compute_routes_run(
    pbp_dropbacks: pd.DataFrame, participation: pd.DataFrame, weekly_stats: pd.DataFrame
) -> pd.DataFrame:
    """Estimate routes run per player per game from play-by-play participation.

    Public data doesn't have a per-player "ran a route" flag, so we use the
    standard proxy: for every play where the QB actually dropped back to pass
    (attempt, sack, or scramble - not just completed passes), count every
    offensive player at WR/RB/TE who was on the field for that play as having
    run a route. This slightly overcounts players who stayed in to pass-block
    on some of those plays rather than running a route, since the public data
    doesn't distinguish the two - but it's a real, play-level count, not a
    rough stand-in like snap share (which also counts running plays).
    """
    dropbacks = pbp_dropbacks[pbp_dropbacks["qb_dropback"] == 1][
        ["game_id", "play_id", "season", "week"]
    ]

    part = participation.rename(columns={"nflverse_game_id": "game_id"}).dropna(
        subset=["offense_players"]
    )
    plays = dropbacks.merge(part, on=["game_id", "play_id"], how="inner")
    plays = plays.assign(offense_players=plays["offense_players"].str.split(";"))
    exploded = plays.explode("offense_players").rename(columns={"offense_players": "player_id"})

    positions = _player_position_lookup(weekly_stats)
    exploded = exploded.merge(positions, on="player_id", how="inner")
    exploded = exploded[exploded["position"].isin(["WR", "RB", "TE"])]

    return exploded.groupby(["player_id", "season", "week"]).size().reset_index(name="routes_run")


def add_route_features(df: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Merge in that game's routes run, and derive per-game TPRR/YPRR
    (targets and receiving yards per route run) - efficiency metrics that
    strip out pure volume. Both then get rolled into trailing averages the
    same leak-safe way as every other stat in ROLLING_STATS.
    """
    df = df.merge(routes, on=["player_id", "season", "week"], how="left")
    df["tprr"] = df["targets"] / df["routes_run"]
    df["yprr"] = df["receiving_yards"] / df["routes_run"]
    return df


def add_next_gen_features(df: pd.DataFrame, ngs_receiving: pd.DataFrame) -> pd.DataFrame:
    """Merge in that game's Next Gen Stats receiving metrics: separation,
    cushion, share of the team's intended air yards, and YAC over expectation.
    """
    ngs = ngs_receiving.rename(
        columns={
            "player_gsis_id": "player_id",
            "percent_share_of_intended_air_yards": "ngs_air_yards_share",
        }
    )
    keep = [
        "player_id",
        "season",
        "week",
        "avg_cushion",
        "avg_separation",
        "ngs_air_yards_share",
        "avg_yac_above_expectation",
    ]
    return df.merge(ngs[keep], on=["player_id", "season", "week"], how="left")


def add_injury_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add that week's official pre-game injury report status as
    `injury_severity` (0 = no report/healthy, 1 = Questionable, 2 = Doubtful,
    3 = Out).

    This uses the SAME week's report, not a lagged/trailing one - that's not
    leakage, since the injury report is published and finalized before
    kickoff, same as home/away or the Vegas line. A player's status can be
    updated multiple times through the week (Wed/Thu/Fri practice reports);
    we keep only the last one per player per week, i.e. the final call.
    """
    reports = (
        injuries.dropna(subset=["gsis_id"])
        .sort_values("date_modified")
        .drop_duplicates(subset=["season", "week", "gsis_id"], keep="last")
    ).copy()
    reports["injury_severity"] = reports["report_status"].map(INJURY_SEVERITY).fillna(0)

    df = df.merge(
        reports[["season", "week", "gsis_id", "injury_severity"]],
        left_on=["season", "week", "player_id"],
        right_on=["season", "week", "gsis_id"],
        how="left",
    ).drop(columns="gsis_id")
    df["injury_severity"] = df["injury_severity"].fillna(0)
    return df


def build_features(
    df: pd.DataFrame,
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    routes: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    scoring: str = "half_ppr",
) -> pd.DataFrame:
    """Run the full feature pipeline: target, routes/NGS, recent performance,
    matchup, schedule context, injury status.

    Routes run and Next Gen Stats are merged in BEFORE the rolling-average
    step, since that step is what turns a raw per-game number into a
    leak-safe trailing feature (see ROLLING_STATS) - order matters here.

    Rows for a player's first few tracked games will have NaN features (there's
    no prior history yet to average) - that's expected, not a bug, and those
    rows get dropped before training. Routes/NGS data only exists from 2016
    on, so rows before that will always be NaN for those specific features -
    also expected, handled the same way as any other missing feature.
    """
    df = add_fantasy_target(df, scoring=scoring)
    df = add_route_features(df, routes)
    df = add_next_gen_features(df, ngs_receiving)
    df = add_recent_performance_features(df)
    df = add_matchup_features(df)
    df = add_schedule_context(df, schedules)
    df = add_injury_features(df, injuries)
    return df
