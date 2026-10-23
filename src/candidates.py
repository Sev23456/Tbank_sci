"""Candidate generation shared by every ranker.

Sources (all computed from history ts < cutoff only):
  rep  - tracks the user listened (listen+) in the last 30 days (repeat consumption)
  i2i  - co-listening neighbours of the user's recent tracks
  art  - currently popular tracks of the user's favourite artists
  pop  - globally most popular tracks of the last day
Global statistics (popularity, co-occurrence) are additive over user buckets, so they are
computed bucket by bucket and summed.
"""
import json
import time

import numpy as np
import polars as pl

from config import DATA, DAY, LISTEN, LPLUS_PCT, N_BUCKETS, SEED
from prepare import bucket_path, item_meta
from splits import Split, check_no_leak, history, labels

N_REP, N_I2I, N_ART, N_POP = 100, 100, 60, 30
I2I_WINDOW, I2I_LAG, I2I_NBRS, I2I_SEEDS = 7 * DAY, 3, 30, 30
USER_WINDOW = 30 * DAY
TOP_ARTISTS, ARTIST_TRACKS = 5, 20

SOURCE_COLS = ["rep_rank", "rep_cnt", "i2i_score", "i2i_rank", "art_rank", "pop_rank", "n_src"]


def split_dir(split: Split):
    d = DATA / split.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_history(b: int, split: Split, columns=None) -> pl.DataFrame:
    h = history(pl.scan_parquet(bucket_path(b)), split)
    if columns:
        h = h.select(columns)
    h = h.collect()
    check_no_leak(h, split)
    return h


def lplus(h: pl.DataFrame | pl.LazyFrame, since: int):
    return h.filter((pl.col("etype") == LISTEN) & (pl.col("ratio") >= LPLUS_PCT) & (pl.col("ts") >= since))


# ---------------------------------------------------------------- global statistics
def global_stats(split: Split, buckets) -> dict[str, pl.DataFrame]:
    c = split.cutoff
    pop1, pop7, pairs = [], [], []
    for b in buckets:
        h = lplus(load_history(b, split, ["uid", "ts", "item", "artist", "etype", "ratio"]), c - I2I_WINDOW)
        pop1.append(h.filter(pl.col("ts") >= c - DAY).group_by("item").len("n"))
        pop7.append(h.group_by("item", "artist").len("n"))
        seq = h.select("uid", "item")  # already sorted by uid, ts
        bp = []
        for lag in range(1, I2I_LAG + 1):
            p = seq.with_columns(
                pl.col("item").shift(-lag).alias("nbr"), pl.col("uid").shift(-lag).alias("uid2")
            ).filter((pl.col("uid") == pl.col("uid2")) & (pl.col("item") != pl.col("nbr")))
            p = p.select("item", "nbr")
            bp += [p, p.select(pl.col("nbr").alias("item"), pl.col("item").alias("nbr"))]
        pairs.append(pl.concat(bp).group_by("item", "nbr").len("c"))
        pairs = [pl.concat(pairs).group_by("item", "nbr").agg(pl.col("c").sum())]

    pop1 = pl.concat(pop1).group_by("item").agg(pl.col("n").sum())
    pop = pop1.sort("n", "item", descending=[True, False]).head(N_POP).with_row_index("pop_rank", 1)
    pop7 = pl.concat(pop7).group_by("item", "artist").agg(pl.col("n").sum())
    # ranks: ties are broken by id (group_by output order is not deterministic)
    art = (
        pop7.drop_nulls("artist")
        .sort(["artist", "n", "item"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("artist").alias("art_item_rank"))
        .filter(pl.col("art_item_rank") <= ARTIST_TRACKS)
        .select("artist", "item", "art_item_rank")
    )
    cnt = pop7.group_by("item").agg(pl.col("n").sum())
    i2i = (
        pairs[0].filter(pl.col("c") >= 2)
        .join(cnt, on="item")
        .join(cnt.rename({"item": "nbr", "n": "n_nbr"}), on="nbr")
        .with_columns((pl.col("c") / (pl.col("n") * pl.col("n_nbr")).sqrt()).alias("s"))
        .sort(["item", "s", "nbr"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("item").alias("r"))
        .filter(pl.col("r") <= I2I_NBRS)
        .select("item", "nbr", "s")
    )
    return {"pop": pop.select("item", "pop_rank"), "art": art, "i2i": i2i}


# ---------------------------------------------------------------- per-user candidates
def user_candidates(h: pl.DataFrame, users: pl.Series, g: dict, cutoff: int) -> pl.DataFrame:
    h = lplus(h.filter(pl.col("uid").is_in(users.implode())), cutoff - USER_WINDOW)
    ui = h.group_by("uid", "item").agg(pl.len().alias("rep_cnt"), pl.col("ts").max().alias("last"))

    # every ranking breaks ties by id: group_by / join output order is not deterministic
    rep = (
        ui.sort(["uid", "rep_cnt", "last", "item"], descending=[False, True, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("rep_rank"))
        .filter(pl.col("rep_rank") <= N_REP)
        .select("uid", "item", "rep_rank", "rep_cnt")
    )

    seeds = (
        ui.sort(["uid", "last", "item"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("sr"))
        .filter(pl.col("sr") <= I2I_SEEDS)
        .select("uid", "item", (1.0 / pl.col("sr")).alias("w"))
    )
    i2i = (
        seeds.join(g["i2i"], on="item")
        .group_by("uid", pl.col("nbr").alias("item_n"))
        # float sums depend on summation order; rounding keeps ties stable across runs
        .agg((pl.col("w") * pl.col("s")).sum().round(9).alias("i2i_score"))
        .rename({"item_n": "item"})
        .sort(["uid", "i2i_score", "item"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("i2i_rank"))
        .filter(pl.col("i2i_rank") <= N_I2I)
    )

    ua = (
        h.drop_nulls("artist").group_by("uid", "artist").len("ua")
        .sort(["uid", "ua", "artist"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("ar"))
        .filter(pl.col("ar") <= TOP_ARTISTS)
    )
    art = (
        ua.join(g["art"], on="artist")
        .with_columns((pl.col("ar") * 100 + pl.col("art_item_rank")).alias("k"))
        .group_by("uid", "item").agg(pl.col("k").min())
        .with_columns(pl.col("k").rank("ordinal").over("uid").alias("art_rank"))
        .filter(pl.col("art_rank") <= N_ART)
        .select("uid", "item", "art_rank")
    )

    pop = pl.DataFrame({"uid": users}).join(g["pop"], how="cross")

    keys = pl.concat([d.select("uid", "item") for d in (rep, i2i, art, pop)]).unique()
    out = (
        keys.join(rep, on=["uid", "item"], how="left")
        .join(i2i, on=["uid", "item"], how="left")
        .join(art, on=["uid", "item"], how="left")
        .join(pop, on=["uid", "item"], how="left")
    )
    n_src = sum(pl.col(c).is_not_null().cast(pl.UInt8) for c in ("rep_rank", "i2i_rank", "art_rank", "pop_rank"))
    return out.with_columns(n_src.alias("n_src")).with_columns(
        [pl.col(c).cast(pl.Float32) for c in SOURCE_COLS]
    ).sort("uid", "item")


# ---------------------------------------------------------------- split builder
def label_users(split: Split, buckets) -> pl.DataFrame:
    out = []
    for b in buckets:
        lab = labels(pl.scan_parquet(bucket_path(b)), split).collect()
        out.append(lab.group_by("uid").len("n_pos").with_columns(pl.lit(b).alias("bucket")))
    return pl.concat(out)


def build_split(split: Split, n_users: int | None, neg_per_user: int | None, buckets=range(N_BUCKETS)):
    """Candidates + labels for a split. n_users=None keeps every user with a positive."""
    t0 = time.time()
    d = split_dir(split)
    users = label_users(split, buckets).sort("uid")  # sample() depends on row order
    if n_users is not None and n_users < users.height:
        users = users.sample(n_users, seed=SEED).sort("uid")
    users.write_parquet(d / "users.parquet")
    g = global_stats(split, buckets)
    # static catalogue: a bucket's history misses parents of tracks none of its users heard
    meta = item_meta().rename({"item_id": "item", "artist_id": "artist", "album_id": "album"})
    t_global = time.time() - t0
    rng = np.random.default_rng(SEED)
    stats = {"n_users": users.height, "n_pos": int(users["n_pos"].sum()), "rows": 0, "pos_in_cands": 0,
             "rows_saved": 0, "by_source": {}}
    for b in buckets:
        ub = users.filter(pl.col("bucket") == b)["uid"]
        h = load_history(b, split)
        cand = user_candidates(h, ub, g, split.cutoff)
        lab = labels(pl.scan_parquet(bucket_path(b)), split).collect()
        cand = cand.join(lab.with_columns(pl.lit(1, pl.UInt8).alias("label")), on=["uid", "item"], how="left")
        # join does not keep row order, and negative sampling below depends on it
        cand = cand.with_columns(pl.col("label").fill_null(0)).sort("uid", "item")
        stats["rows"] += cand.height
        stats["pos_in_cands"] += int(cand["label"].sum())
        for src in ("rep_rank", "i2i_rank", "art_rank", "pop_rank"):
            s = stats["by_source"].setdefault(src, [0, 0])
            sub = cand.filter(pl.col(src).is_not_null())
            s[0] += sub.height
            s[1] += int(sub["label"].sum())
        if neg_per_user is not None:
            # keep all positives + a random subset of negatives (training only)
            cand = cand.with_columns(pl.Series("rnd", rng.random(cand.height)))
            cand = cand.with_columns(
                pl.col("rnd").rank("ordinal").over("uid", "label").alias("rr")
            ).filter((pl.col("label") == 1) | (pl.col("rr") <= neg_per_user)).drop("rnd", "rr")
        # attach parents for indirect features
        cand = cand.join(meta, on="item", how="left")
        stats["rows_saved"] += cand.height
        cand.write_parquet(d / f"cands_b{b:02d}.parquet")
        print(f"[{split.name}] bucket {b}: users={len(ub)} cands={cand.height:,} t={time.time() - t0:.0f}s", flush=True)
    stats["cand_recall"] = stats["pos_in_cands"] / max(stats["n_pos"], 1)
    stats["avg_cands"] = stats["rows"] / max(stats["n_users"], 1)
    stats["time_global_s"] = t_global
    stats["time_total_s"] = time.time() - t0
    json.dump(stats, open(d / "cand_stats.json", "w"), indent=1)
    print(f"[{split.name}] cand recall={stats['cand_recall']:.4f} avg cands={stats['avg_cands']:.1f}")
    return stats
