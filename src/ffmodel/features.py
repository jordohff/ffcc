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

FEATURE_COLUMNS = [
    "avg_fantasy_pts_last3",
    "avg_fantasy_pts_last5",
    "avg_targets_last3",
    "avg_targets_last5",
    "avg_carries_last3",
    "avg_carries_last5",
    "avg_target_share_last3",
    "avg_target_share_last5",
    "opp_avg_pts_allowed_last5",
]


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


def add_recent_performance_features(df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> pd.DataFrame:
    """Add trailing rolling averages of fantasy points and usage per player.

    E.g. `avg_fantasy_pts_last3` = that player's average fantasy points over
    their previous 3 games (not counting the game in that row).
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id")

    for w in windows:
        df[f"avg_fantasy_pts_last{w}"] = grouped["fantasy_points_target"].transform(
            lambda s, w=w: _trailing_mean(s, w)
        )
        df[f"avg_targets_last{w}"] = grouped["targets"].transform(lambda s, w=w: _trailing_mean(s, w))
        df[f"avg_carries_last{w}"] = grouped["carries"].transform(lambda s, w=w: _trailing_mean(s, w))
        df[f"avg_target_share_last{w}"] = grouped["target_share"].transform(
            lambda s, w=w: _trailing_mean(s, w)
        )
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


def build_features(df: pd.DataFrame, scoring: str = "half_ppr") -> pd.DataFrame:
    """Run the full feature pipeline: target, recent performance, matchup.

    Rows for a player's first few tracked games will have NaN features (there's
    no prior history yet to average) - that's expected, not a bug, and those
    rows get dropped before training.
    """
    df = add_fantasy_target(df, scoring=scoring)
    df = add_recent_performance_features(df)
    df = add_matchup_features(df)
    return df
