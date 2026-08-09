"""Train and evaluate regression models for weekly fantasy points."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from ffmodel.features import POSITION_FEATURE_COLUMNS


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

    Doesn't drop NaN feature rows here, since QB and skill-position models use
    different feature columns (see POSITION_FEATURE_COLUMNS) - that filtering
    happens per-position in fit_and_evaluate_by_position instead.
    """
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    return train, test


def fit_and_evaluate_by_position(
    train: pd.DataFrame, test: pd.DataFrame, kind: str = "ridge"
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit one model per position (QB/RB/WR/TE), each on its own feature set.

    Fantasy scoring is driven by different things per position - a QB's points
    come mostly from passing yards/TDs, a WR's from targets/air yards - so
    each position gets its own feature columns (POSITION_FEATURE_COLUMNS) and
    its own model coefficients rather than sharing one generic feature set.

    Rows where the player has no prior game history at all (`avg_fantasy_pts_last3`
    is NaN - their first tracked game or so) are dropped, since the model has
    nothing to learn from or predict on for those rows. Individual sparser
    features (e.g. Next Gen Stats separation, which isn't tracked for every
    player-week - see features.add_next_gen_features) are allowed to be
    missing and are median-imputed instead of gating the whole row out -
    otherwise a single sparse column would silently discard most of the
    dataset for no good reason.

    Prints accuracy per position plus an overall number, and returns the
    dict of fitted models (one per position) plus a combined dataframe of all
    test rows that got a prediction, with the new `projected_points` column.
    """
    models: dict[str, Pipeline] = {}
    predictions = []

    for position in sorted(train["position"].unique()):
        feature_cols = POSITION_FEATURE_COLUMNS.get(position)
        if feature_cols is None:
            continue

        has_history = ["avg_fantasy_pts_last3"]
        pos_train = train[train["position"] == position].dropna(subset=has_history)
        pos_test = test[test["position"] == position].dropna(subset=has_history).copy()
        if pos_train.empty or pos_test.empty:
            continue

        pipeline = make_pipeline(kind=kind)
        pipeline.fit(pos_train[feature_cols], pos_train["fantasy_points_target"])
        pos_test["projected_points"] = pipeline.predict(pos_test[feature_cols])
        models[position] = pipeline

        pos_mae = mean_absolute_error(pos_test["fantasy_points_target"], pos_test["projected_points"])
        pos_r2 = r2_score(pos_test["fantasy_points_target"], pos_test["projected_points"])
        print(f"  {position}: MAE {pos_mae:.2f}, R^2 {pos_r2:.3f} (n={len(pos_test):,})")
        predictions.append(pos_test)

    all_predictions = pd.concat(predictions)
    overall_mae = mean_absolute_error(all_predictions["fantasy_points_target"], all_predictions["projected_points"])
    overall_r2 = r2_score(all_predictions["fantasy_points_target"], all_predictions["projected_points"])
    print(f"Overall: MAE {overall_mae:.2f}, R^2 {overall_r2:.3f}")

    return models, all_predictions
