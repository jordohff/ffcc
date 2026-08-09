"""Train and evaluate regression models for weekly fantasy points."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from ffmodel.features import FEATURE_COLUMNS


def make_pipeline(kind: str = "ridge") -> Pipeline:
    """Build a model pipeline. `kind` is "ridge" (linear, regularized) or
    "gbm" (gradient-boosted trees, captures non-linear effects/interactions
    that a linear model can't - e.g. "high target share only matters when
    volume is also high").
    """
    if kind == "ridge":
        model = Ridge(alpha=1.0)
    elif kind == "gbm":
        model = HistGradientBoostingRegressor(random_state=0)
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (use 'ridge' or 'gbm')")
    return Pipeline([("impute", SimpleImputer(strategy="median")), (kind, model)])


def train_test_split_by_season(
    df: pd.DataFrame, test_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (seasons before test_season) and test (test_season).

    This is a time-respecting split, not a random shuffle - we only ever train
    on data from before the games we're evaluating on, which mirrors how the
    model would actually be used (predicting future weeks from past ones).

    Rows with missing features (a player's first few tracked games, with no
    prior history to average) are dropped, since the model has nothing to
    learn from or predict on for those rows.
    """
    train = df[df["season"] < test_season].dropna(subset=FEATURE_COLUMNS)
    test = df[df["season"] == test_season].dropna(subset=FEATURE_COLUMNS)
    return train, test


def fit_and_evaluate_by_position(
    train: pd.DataFrame, test: pd.DataFrame, kind: str = "ridge"
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit one model per position (QB/RB/WR/TE) instead of a single shared model.

    Fantasy scoring is driven by different things per position - a QB's points
    come mostly from passing yards/TDs, a WR's from targets/air yards - so
    letting each position have its own coefficients fits noticeably better
    than forcing one model to average across all of them.

    Prints accuracy per position plus an overall number, and returns the
    dict of fitted models (one per position) plus `test` with predictions
    filled in.
    """
    models: dict[str, Pipeline] = {}
    test = test.copy()
    test["projected_points"] = float("nan")

    for position in sorted(train["position"].unique()):
        pos_train = train[train["position"] == position]
        pos_test_mask = test["position"] == position
        if pos_test_mask.sum() == 0:
            continue

        pipeline = make_pipeline(kind=kind)
        pipeline.fit(pos_train[FEATURE_COLUMNS], pos_train["fantasy_points_target"])
        test.loc[pos_test_mask, "projected_points"] = pipeline.predict(
            test.loc[pos_test_mask, FEATURE_COLUMNS]
        )
        models[position] = pipeline

        pos_mae = mean_absolute_error(
            test.loc[pos_test_mask, "fantasy_points_target"],
            test.loc[pos_test_mask, "projected_points"],
        )
        pos_r2 = r2_score(
            test.loc[pos_test_mask, "fantasy_points_target"],
            test.loc[pos_test_mask, "projected_points"],
        )
        print(f"  {position}: MAE {pos_mae:.2f}, R^2 {pos_r2:.3f} (n={pos_test_mask.sum():,})")

    overall_mae = mean_absolute_error(test["fantasy_points_target"], test["projected_points"])
    overall_r2 = r2_score(test["fantasy_points_target"], test["projected_points"])
    print(f"Overall: MAE {overall_mae:.2f}, R^2 {overall_r2:.3f}")

    return models, test
