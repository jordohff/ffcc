"""Train and evaluate a simple regression model for weekly fantasy points."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from ffmodel.features import FEATURE_COLUMNS


def make_pipeline() -> Pipeline:
    """A simple, explainable model: median-impute missing features, then a
    lightly-regularized linear regression (Ridge).

    Ridge is plain linear regression with a small penalty that keeps
    coefficients from swinging wildly when features are correlated with each
    other (e.g. targets and target_share move together). It's a standard,
    boring, easy-to-explain starting point.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


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


def fit_and_evaluate(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit on `train`, predict on `test`, print accuracy, return both."""
    pipeline = make_pipeline()
    pipeline.fit(train[FEATURE_COLUMNS], train["fantasy_points_target"])

    test = test.copy()
    test["projected_points"] = pipeline.predict(test[FEATURE_COLUMNS])

    mae = mean_absolute_error(test["fantasy_points_target"], test["projected_points"])
    r2 = r2_score(test["fantasy_points_target"], test["projected_points"])
    print(f"Holdout MAE: {mae:.2f} fantasy points (average error per player-week)")
    print(f"Holdout R^2: {r2:.3f} (share of week-to-week variance explained, 1.0 = perfect)")

    return pipeline, test
