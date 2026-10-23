"""LightGBM wrappers shared by every experiment (same hyper-parameters for every feature set)."""
import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED

COMMON = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=20, seed=SEED,
              deterministic=True, force_col_wise=True)
RANK = dict(objective="lambdarank", metric="ndcg", eval_at=[10], lambdarank_truncation_level=30)
BINARY = dict(objective="binary", metric="binary_logloss")
QUICK = dict(learning_rate=0.1)  # used inside the feature search loop


def _xy(df: pl.DataFrame, cols):
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    return X, df["label"].to_numpy()


def _groups(df: pl.DataFrame):
    return df.group_by("uid", maintain_order=True).len()["len"].to_numpy()


def fit(train: pl.DataFrame, val: pl.DataFrame, cols, objective="rank", params=None, rounds=2000, es=100):
    """train/val must be sorted by uid."""
    p = {**COMMON, **(RANK if objective == "rank" else BINARY), **(params or {})}
    Xt, yt = _xy(train, cols)
    Xv, yv = _xy(val, cols)
    gt = _groups(train) if objective == "rank" else None
    gv = _groups(val) if objective == "rank" else None
    dt = lgb.Dataset(Xt, yt, group=gt, feature_name=[_safe(c) for c in cols], free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, group=gv, reference=dt)
    return lgb.train(p, dt, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(es, verbose=False)])


def predict(model, df: pl.DataFrame, cols) -> np.ndarray:
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    return model.predict(X, num_iteration=model.best_iteration)


def importance(model, cols) -> dict[str, float]:
    g = model.feature_importance("gain")
    return dict(sorted(zip(cols, map(float, g)), key=lambda t: -t[1]))


def keep_users_with_pos(df: pl.DataFrame) -> pl.DataFrame:
    """Groups without positives give no lambdarank gradient."""
    return df.filter(pl.col("label").max().over("uid") > 0)


def _safe(c: str) -> str:
    return c.replace(".", "_").replace("/", "_per_")


def prior_correction(p: np.ndarray, neg_rate: float) -> np.ndarray:
    """Undo negative down-sampling (rate r of negatives kept) for calibrated probabilities."""
    return p / (p + (1 - p) / neg_rate)
