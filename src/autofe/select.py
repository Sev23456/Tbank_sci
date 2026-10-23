"""Automatic feature selection over the DSL space (uses train + val only, never test).

Pipeline:
  1. screen     - drop near-empty / constant features, score each feature by univariate AUC
                  (global and within-user), keep the top N, de-duplicate by Spearman |rho|.
  2. importance - one LightGBM ranker on all survivors, order them by gain.
  3. forward    - greedy forward selection in importance order, a feature is kept only if it
                  improves val NDCG@10 by more than eps (wrapper method).
  4. windows    - for every kept feature try the neighbouring time windows (e.g. 7d -> 3d / 14d)
                  and keep the swap if val NDCG@10 improves.
"""
import time

import numpy as np
import polars as pl

from autofe.dsl import SPACE, neighbours, parse
from metrics import ranking_metrics, roc_auc
from train import QUICK, fit, importance, keep_users_with_pos, predict


def _auc_score(x: np.ndarray, y: np.ndarray) -> float:
    x = np.where(np.isnan(x), np.nanmin(x) - 1 if np.isfinite(np.nanmin(x)) else 0, x)
    return abs(roc_auc(y, x) - 0.5)


def screen(train: pl.DataFrame, names: list[str], n_keep=300, max_null=0.99, rho_max=0.95,
           sample=300_000, seed=0) -> tuple[list[str], pl.DataFrame]:
    df = train.sample(min(sample, train.height), seed=seed).sort("uid")
    y = df["label"].to_numpy()
    rows = []
    for n in names:
        s = df[n]
        null = s.null_count() / len(s)
        if null > max_null or s.drop_nulls().n_unique() <= 1:
            rows.append((n, null, 0.0, 0.0))
            continue
        x = s.to_numpy().astype(np.float64)
        within = df.select((pl.col(n) - pl.col(n).mean().over("uid")).fill_null(0))[n].to_numpy()
        rows.append((n, null, _auc_score(x, y), _auc_score(within, y)))
    sc = pl.DataFrame(rows, schema=["feature", "null_rate", "auc_global", "auc_within"], orient="row")
    sc = sc.with_columns(pl.max_horizontal("auc_global", "auc_within").alias("score")).sort("score", descending=True)
    top = sc.filter(pl.col("score") > 0.005).head(n_keep)["feature"].to_list()

    # Spearman de-duplication, greedy in score order
    R = df.select([pl.col(n).rank("average").fill_null(-1).cast(pl.Float32) for n in top]).to_numpy()
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    C = np.abs(R.T @ R / len(R))
    kept = []
    for i in range(len(top)):
        if all(C[i, j] <= rho_max for j in kept):
            kept.append(i)
    return [top[i] for i in kept], sc


class Evaluator:
    """Quick train-on-train / score-on-val loop used by the wrapper search."""

    def __init__(self, train, val, val_users, base_cols, metric="ndcg@10"):
        self.train, self.val, self.val_users = keep_users_with_pos(train), val, val_users
        self.base, self.metric = list(base_cols), metric
        self.n_evals, self.time = 0, 0.0

    def __call__(self, cols) -> float:
        t = time.time()
        cols = self.base + [c for c in cols if c not in self.base]
        m = fit(self.train, self.val, cols, params=QUICK, rounds=400, es=30)
        v = self.val.select("uid", "label").with_columns(pl.Series("score", predict(m, self.val, cols)))
        self.n_evals += 1
        self.time += time.time() - t
        return ranking_metrics(v, self.val_users, ks=(10,))[self.metric]


def forward(ev: Evaluator, ordered: list[str], max_try=150, eps=5e-4, log=print) -> tuple[list[str], list[dict]]:
    cur, best = [], ev([])
    traj = [{"step": 0, "feature": None, "score": best, "accepted": True}]
    for i, f in enumerate(ordered[:max_try], 1):
        s = ev(cur + [f])
        acc = s > best + eps
        if acc:
            cur, best = cur + [f], s
        traj.append({"step": i, "feature": f, "score": s, "accepted": acc, "best": best})
        log(f"  fwd {i:3d} {'+' if acc else ' '} {f:40s} {s:.5f} best={best:.5f} n={len(cur)}")
    return cur, traj


def refine_windows(ev: Evaluator, cols: list[str], available: set[str], eps=5e-4, log=print):
    cur, best = list(cols), ev(cols)
    swaps = []
    for f in list(cur):
        for nb in neighbours(parse(f)):
            n = nb.name
            if n not in available or n in cur:
                continue
            cand = [n if c == f else c for c in cur]
            s = ev(cand)
            if s > best + eps:
                log(f"  window {f} -> {n}: {best:.5f} -> {s:.5f}")
                swaps.append((f, n, best, s))
                cur, best, f = cand, s, n
    return cur, best, swaps


def run(train, val, val_users, base_cols, budget, n_screen=300, max_try=150, log=print) -> dict:
    t0 = time.time()
    names = [n for n in SPACE if n in train.columns]
    screened, scores = screen(train, names, n_keep=n_screen)
    t_screen = time.time() - t0
    log(f"screen: {len(names)} -> {len(screened)} features ({t_screen:.0f}s)")

    tr = keep_users_with_pos(train)
    m = fit(tr, val, base_cols + screened, params=QUICK, rounds=400, es=30)
    imp = importance(m, base_cols + screened)
    ordered = [f for f in imp if f in screened]
    t_imp = time.time() - t0 - t_screen
    log(f"importance model: {m.best_iteration} trees ({t_imp:.0f}s)")

    ev = Evaluator(train, val, val_users, base_cols)
    sel, traj = forward(ev, ordered, max_try=max_try, log=log)
    budgeted = [t["feature"] for t in traj if t["accepted"] and t["feature"]][:budget]
    refined, refined_score, swaps = refine_windows(ev, sel, set(names), log=log)
    return {
        "screened": screened, "importance_order": ordered, "screen_scores": scores.to_dicts(),
        "auto_top": ordered[:budget], "auto_fwd_budget": budgeted, "auto_fwd": sel,
        "auto_fwd_win": refined, "window_swaps": swaps, "trajectory": traj,
        "n_space": len(names), "n_evals": ev.n_evals,
        "time_screen_s": t_screen, "time_importance_s": t_imp, "time_wrapper_s": ev.time,
        "time_total_s": time.time() - t0,
    }
