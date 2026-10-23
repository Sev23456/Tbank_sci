"""Compute any list of DSL features for the candidates of a split."""
import time

import polars as pl

from autofe.dsl import (GLOBAL_KEYS, KEYS, USER_KEYS, add_derived, base_closure, combine_partials,
                        compute_partial, fill_missing, finalize, parse)
from candidates import SOURCE_COLS, load_history, split_dir
from config import N_BUCKETS
from splits import Split


def feat_path(split: Split, tag: str, b: int):
    return split_dir(split) / f"feat_{tag}_b{b:02d}.parquet"


def compute_split_features(split: Split, names: list[str], tag: str, buckets=range(N_BUCKETS)) -> dict:
    t0 = time.time()
    d = split_dir(split)
    cutoff = split.cutoff
    feats = [parse(n) for n in names]
    base = base_closure(feats)
    all_base = [f for fs in base.values() for f in fs]
    cands = {b: pl.read_parquet(d / f"cands_b{b:02d}.parquet") for b in buckets}

    # 1) item / artist / album aggregates over *all* users, summed bucket by bucket
    gkeys = [k for k in GLOBAL_KEYS if k in base]
    restrict = {k: pl.concat([c.select(KEYS[k]) for c in cands.values()]).drop_nulls().unique() for k in gkeys}
    acc = {k: None for k in gkeys}
    for b in range(N_BUCKETS) if gkeys else []:
        h = load_history(b, split).lazy()
        for k in gkeys:
            p = compute_partial(h, k, base[k], cutoff, restrict[k])
            acc[k] = p if acc[k] is None else combine_partials([acc[k], p], k, base[k], cutoff)
    glob = {k: finalize(acc[k], base[k], cutoff) for k in gkeys}
    t_global = time.time() - t0

    # 2) user-side aggregates, local to each bucket
    rows = 0
    for b in buckets:
        c = cands[b]
        h = load_history(b, split).lazy()
        out = c
        for k in USER_KEYS:
            if k not in base:
                continue
            cols = list(KEYS[k])
            p = compute_partial(h, k, base[k], cutoff, c.select(cols).drop_nulls().unique())
            out = out.join(finalize(p, base[k], cutoff), on=cols, how="left")
        for k, g in glob.items():
            out = out.join(g, on=list(KEYS[k]), how="left")
        out = add_derived(fill_missing(out, all_base), feats, cutoff)
        out = out.select(["uid", "item", "label"] + SOURCE_COLS + [pl.col(n).cast(pl.Float32) for n in names])
        out.write_parquet(feat_path(split, tag, b))
        rows += out.height
    t = time.time() - t0
    print(f"[{split.name}/{tag}] {len(names)} features x {rows:,} rows: global {t_global:.0f}s, total {t:.0f}s",
          flush=True)
    return {"split": split.name, "tag": tag, "n_features": len(names), "rows": rows,
            "time_global_s": t_global, "time_total_s": t}


def load_features(split: Split, tag: str, columns: list[str] | None = None, buckets=range(N_BUCKETS),
                  users: pl.Series | None = None) -> pl.DataFrame:
    paths = [feat_path(split, tag, b) for b in buckets if feat_path(split, tag, b).exists()]
    lf = pl.scan_parquet(paths)
    if columns is not None:
        lf = lf.select(["uid", "item", "label"] + [c for c in columns if c not in ("uid", "item", "label")])
    if users is not None:
        lf = lf.filter(pl.col("uid").is_in(users.implode()))
    return lf.collect().sort("uid", "item")
