"""Ranking and classification metrics.

Ranking metrics are computed over *all* listen+ items of a user in the label window, not only
those that made it into the candidates, so they are comparable to full-catalogue numbers:
  recall@k = hits@k / min(n_pos, k),  ndcg@k = dcg@k / idcg@k with idcg over min(n_pos, k) hits.
(The official benchmark's NDCG divides by the DCG of the predicted mask, which is a bug.)
"""
import numpy as np
import polars as pl
from scipy.stats import rankdata


def ranking_metrics(df: pl.DataFrame, users: pl.DataFrame, score: str = "score", ks=(10, 100)) -> dict:
    """df: uid, label, <score>; users: uid, n_pos (every evaluated user, even without candidates)."""
    # stable sort: tied scores keep the input order (uid, item), so metrics are reproducible
    r = df.select("uid", "label", score).sort(["uid", score], descending=[False, True], maintain_order=True)
    r = r.with_columns(pl.int_range(pl.len()).over("uid").alias("rank"))
    disc = 1.0 / np.log2(np.arange(max(ks)) + 2)
    aggs = []
    for k in ks:
        top = pl.col("rank") < k
        aggs.append(pl.col("label").filter(top).sum().alias(f"hits{k}"))
        aggs.append((pl.col("label") / (pl.col("rank") + 2).log(2)).filter(top).sum().alias(f"dcg{k}"))
    per = users.select("uid", "n_pos").join(r.group_by("uid").agg(aggs), on="uid", how="left").fill_null(0)
    n_pos = per["n_pos"].to_numpy()
    out = {}
    for k in ks:
        idcg = np.cumsum(disc[:k])[np.minimum(n_pos, k) - 1]
        out[f"ndcg@{k}"] = float(np.mean(per[f"dcg{k}"].to_numpy() / idcg))
        out[f"recall@{k}"] = float(np.mean(per[f"hits{k}"].to_numpy() / np.minimum(n_pos, k)))
    return out


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
