"""End-to-end experiment: candidates -> features -> auto selection -> final comparison.

    python run_all.py                       # full run on Yambda-500M (all stages)
    python run_all.py --stage final         # re-run one stage (earlier ones are cached on disk)
    python run_all.py --smoke               # tiny run on 2 buckets to check the pipeline
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import psutil

import config
from autofe import select as autosel
from autofe.dsl import SPACE, parse
from candidates import SOURCE_COLS, build_split, split_dir
from features import compute_split_features, feat_path, load_features
from features_manual import MANUAL
from metrics import logloss, ranking_metrics, roc_auc
from splits import SPLITS
from train import fit, importance, keep_users_with_pos, predict, prior_correction

RES = config.RESULTS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def peak_gb():
    m = psutil.Process().memory_info()
    return getattr(m, "peak_wset", m.rss) / 2**30


def dump(obj, name):
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(RES / name, "w"), indent=1, default=str)


def load(name):
    return json.load(open(RES / name))


def timings_update(**kw):
    p = RES / "timings.json"
    t = json.load(open(p)) if p.exists() else {}
    t.update(kw)
    dump(t, "timings.json")


# ------------------------------------------------------------------ stages
def stage_cands(cfg):
    out = {}
    for name, (n_users, neg) in {"train": (cfg.train_users, cfg.neg), "val": (cfg.val_users, None),
                                 "test": (cfg.test_users, None)}.items():
        out[name] = build_split(SPLITS[name], n_users, neg, buckets=cfg.buckets)
    dump(out, "cand_stats.json")


def stage_feats(cfg):
    st = {}
    for name in ("train", "val"):
        st[name] = compute_split_features(SPLITS[name], list(SPACE), "all", buckets=cfg.buckets)
        st[name]["peak_gb"] = peak_gb()
    timings_update(features_full_space=st)


def stage_select(cfg):
    train = load_features(SPLITS["train"], "all", buckets=cfg.buckets)
    val = load_features(SPLITS["val"], "all", buckets=cfg.buckets)
    vu = pl.read_parquet(split_dir(SPLITS["val"]) / "users.parquet")
    log(f"selection data: train {train.shape}, val {val.shape}")
    res = autosel.run(train, val, vu, SOURCE_COLS, budget=len(MANUAL), n_screen=cfg.n_screen,
                      max_try=cfg.max_try, log=log)
    res["peak_gb"] = peak_gb()
    dump(res, "selection.json")
    timings_update(selection={k: v for k, v in res.items() if k.startswith(("time", "n_"))})


def feature_sets() -> dict[str, list[str]]:
    s = load("selection.json")
    union = list(dict.fromkeys(MANUAL + s["auto_fwd_win"]))
    return {
        "candidate scores only": [],
        "manual": MANUAL,
        "auto: top-B by importance": s["auto_top"],
        "auto: forward (budget B)": s["auto_fwd_budget"],
        "auto: forward": s["auto_fwd"],
        "auto: forward + window refine": s["auto_fwd_win"],
        "auto: all screened": s["screened"],
        "manual + auto (fwd+win)": union,
    }


def stage_test_feats(cfg):
    names = sorted({n for fs in feature_sets().values() for n in fs} | set(ablation_sets()["manual"]))
    st = compute_split_features(SPLITS["test"], names, "final", buckets=cfg.buckets)
    # cost of the hand-crafted set alone, for the cost comparison
    st_m = compute_split_features(SPLITS["test"], MANUAL, "manual", buckets=cfg.buckets)
    timings_update(features_test_final=st, features_test_manual=st_m)


def tune(train, val, vu, cols):
    grid = [dict(num_leaves=nl, min_data_in_leaf=md) for nl in (31, 63, 127) for md in (50, 200)]
    best, rows = None, []
    for g in grid:
        m = fit(train, val, cols, params=g)
        v = val.select("uid", "label").with_columns(pl.Series("score", predict(m, val, cols)))
        s = ranking_metrics(v, vu, ks=(10,))["ndcg@10"]
        rows.append({**g, "val_ndcg@10": s, "trees": m.best_iteration})
        log(f"tune {g}: {s:.5f}")
        if best is None or s > best[1]:
            best = (g, s)
    return best[0], rows


def ablation_sets():
    groups = {"user": "u", "item": "i", "artist": "a", "user-item (direct)": "ui",
              "user-artist/album (indirect)": ("ua", "ual")}
    out = {"manual": MANUAL}
    for g, keys in groups.items():
        keys = keys if isinstance(keys, tuple) else (keys,)
        out[f"manual - {g}"] = [n for n in MANUAL if parse(n).key not in keys]
    return out


def predict_test(models: dict, cfg, tag="final"):
    """Score test bucket by bucket; returns uid,item,label + one column per model."""
    parts = []
    for b in cfg.buckets:
        if not feat_path(SPLITS["test"], tag, b).exists():
            continue
        df = pl.read_parquet(feat_path(SPLITS["test"], tag, b)).sort("uid", "item")
        cols = {"uid": df["uid"], "item": df["item"], "label": df["label"]}
        for name, (m, fcols, kind) in models.items():
            cols[name] = predict(m, df, fcols) if kind != "heur" else m(df)
        parts.append(pl.DataFrame(cols))
    return pl.concat(parts)


def heuristics():
    big = 1e6
    return {
        "heur: repeat (rep_rank)": lambda d: -d["rep_rank"].fill_null(big).to_numpy(),
        "heur: item2item score": lambda d: d["i2i_score"].fill_null(0).to_numpy(),
        "heur: popularity 1d": lambda d: d["i.cnt.listen.1d"].fill_null(0).to_numpy(),
        "heur: rep > i2i > artist > pop": lambda d: (
            -(d["rep_rank"].fill_null(big) * 1e6 + d["i2i_rank"].fill_null(big) * 1e3
              + d["art_rank"].fill_null(big)).to_numpy()
        ),
    }


def stage_final(cfg):
    sets = feature_sets()
    abl = ablation_sets()
    need = sorted({n for fs in list(sets.values()) + list(abl.values()) for n in fs})
    train = load_features(SPLITS["train"], "all", columns=SOURCE_COLS + need, buckets=cfg.buckets)
    val = load_features(SPLITS["val"], "all", columns=SOURCE_COLS + need, buckets=cfg.buckets)
    vu = pl.read_parquet(split_dir(SPLITS["val"]) / "users.parquet")
    tu = pl.read_parquet(split_dir(SPLITS["test"]) / "users.parquet")
    tr_rank = keep_users_with_pos(train)
    cs = load("cand_stats.json")["train"]
    pos = cs["pos_in_cands"]
    neg_rate = (cs["rows_saved"] - pos) / max(cs["rows"] - pos, 1)

    params, tune_rows = tune(tr_rank, val, vu, SOURCE_COLS + MANUAL)
    dump({"best": params, "grid": tune_rows}, "tuning.json")

    models, info = {}, {}
    for name, fs in {**sets, **{k: v for k, v in abl.items() if k != "manual"}}.items():
        cols = SOURCE_COLS + fs
        t = time.time()
        m = fit(tr_rank, val, cols, params=params)
        t_rank = time.time() - t
        t = time.time()
        mb = fit(train, val, cols, objective="binary", params=params)
        t_bin = time.time() - t
        models[name] = (m, cols, "rank")
        models[name + " #bin"] = (mb, cols, "bin")
        v = val.select("uid", "label").with_columns(pl.Series("score", predict(m, val, cols)))
        info[name] = {"n_features": len(fs), "trees": m.best_iteration, "fit_rank_s": t_rank, "fit_bin_s": t_bin,
                      "val": ranking_metrics(v, vu), "is_ablation": name in abl}
        pl.DataFrame({"feature": list(importance(m, cols)), "gain": list(importance(m, cols).values())}) \
            .write_csv(RES / f"importance_{_fname(name)}.csv")
        log(f"{name}: {len(fs)} feats, val {info[name]['val']}, {t_rank:.0f}s")
    for h, fn in heuristics().items():
        models[h] = (fn, None, "heur")

    t = time.time()
    pred = predict_test(models, cfg)
    t_pred = time.time() - t
    y = pred["label"].to_numpy()
    rows = []
    for name, (_, _, kind) in models.items():
        if kind == "bin":
            continue
        r = {"model": name, **ranking_metrics(pred.select("uid", "label", pl.col(name).alias("score")), tu)}
        if kind == "rank":
            pb = prior_correction(pred[name + " #bin"].to_numpy(), neg_rate)
            r.update({"auc": roc_auc(y, pb), "logloss": logloss(y, pb), **info[name]})
            r.update({f"val_{k}": v for k, v in r.pop("val").items()})
        rows.append(r)
        log(f"TEST {name}: ndcg@10={r['ndcg@10']:.5f} recall@100={r['recall@100']:.5f}")
    pl.DataFrame(rows).write_csv(RES / "test_metrics.csv")
    timings_update(final={"predict_test_s": t_pred, "test_rows": pred.height, "neg_rate": neg_rate,
                          "peak_gb": peak_gb()})
    # bootstrap over users: is the difference vs manual significant?
    dump(bootstrap(pred, tu, [n for n in sets] + list(heuristics())), "bootstrap.json")


def bootstrap(pred, users, names, ref="manual", n=200, seed=0):
    """Paired user-level bootstrap of NDCG@10 differences against the manual set."""
    per = {}
    disc = 1.0 / np.log2(np.arange(10) + 2)
    nposd = dict(zip(users["uid"].to_list(), users["n_pos"].to_list()))
    uids = users["uid"].to_numpy()
    for name in names:
        r = pred.select("uid", "label", pl.col(name).alias("s")).sort(["uid", "s"], descending=[False, True],
                                                                      maintain_order=True)
        r = r.with_columns(pl.int_range(pl.len()).over("uid").alias("rank")).filter(pl.col("rank") < 10)
        g = r.group_by("uid").agg((pl.col("label") / (pl.col("rank") + 2).log(2)).sum().alias("dcg"))
        d = dict(zip(g["uid"].to_list(), g["dcg"].to_list()))
        per[name] = np.array([d.get(u, 0.0) / np.cumsum(disc)[min(nposd[u], 10) - 1] for u in uids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uids), size=(n, len(uids)))
    out = {}
    for name in names:
        diff = per[name] - per[ref]
        bs = diff[idx].mean(1)
        out[name] = {"mean_diff": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                float(np.percentile(bs, 97.5))]}
    return out


def stage_scaling(cfg):
    s = feature_sets()
    sets = {"manual": s["manual"], "auto: forward + window refine": s["auto: forward + window refine"],
            "manual + auto (fwd+win)": s["manual + auto (fwd+win)"]}
    need = sorted({n for fs in sets.values() for n in fs})
    train = keep_users_with_pos(load_features(SPLITS["train"], "all", columns=SOURCE_COLS + need,
                                              buckets=cfg.buckets))
    val = load_features(SPLITS["val"], "all", columns=SOURCE_COLS + need, buckets=cfg.buckets)
    tu = pl.read_parquet(split_dir(SPLITS["test"]) / "users.parquet")
    params = load("tuning.json")["best"]
    users = train["uid"].unique().sort()
    rows, models = [], {}
    for frac in (0.125, 0.25, 0.5, 1.0):
        sub_users = users.sample(fraction=frac, seed=1) if frac < 1 else users
        sub = train.filter(pl.col("uid").is_in(sub_users.implode()))
        for name, fs in sets.items():
            t = time.time()
            m = fit(sub, val, SOURCE_COLS + fs, params=params)
            key = f"{name} @{len(sub_users)}"
            models[key] = (m, SOURCE_COLS + fs, "rank")
            rows.append({"set": name, "train_users": len(sub_users), "train_rows": sub.height,
                         "fit_s": time.time() - t, "trees": m.best_iteration})
    pred = predict_test(models, cfg)
    for r, key in zip(rows, models):
        r.update(ranking_metrics(pred.select("uid", "label", pl.col(key).alias("score")), tu))
        log(f"SCALING {key}: {r['ndcg@10']:.5f}")
    pl.DataFrame(rows).write_csv(RES / "scaling.csv")


def _fname(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


STAGES = {"cands": stage_cands, "feats": stage_feats, "select": stage_select, "test_feats": stage_test_feats,
          "final": stage_final, "scaling": stage_scaling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = argparse.Namespace(buckets=list(range(config.N_BUCKETS)), train_users=20_000, val_users=5_000,
                             test_users=None, neg=30, n_screen=300, max_try=150)
    if a.smoke:
        cfg = argparse.Namespace(buckets=[0, 1], train_users=600, val_users=300, test_users=400, neg=30,
                                 n_screen=60, max_try=15)
    RES.mkdir(parents=True, exist_ok=True)
    config.DATA.mkdir(parents=True, exist_ok=True)
    log(f"config {vars(cfg)} data={config.DATA} results={RES}")
    for name in (STAGES if a.stage == "all" else [a.stage]):
        t = time.time()
        log(f"=== stage {name}")
        STAGES[name](cfg)
        log(f"=== stage {name} done in {time.time() - t:.0f}s, peak RSS {peak_gb():.1f} GB")
        timings_update(**{f"stage_{name}_s": time.time() - t})


if __name__ == "__main__":
    main()
