"""Convert Yambda-500M multi_event into compact per-user-bucket parquet files.

Users are split into N_BUCKETS contiguous uid ranges (raw file is sorted by uid, so the
range filter is pushed down to parquet row groups). Every bucket holds the *full* history of
its users, which makes all per-user computations local to one bucket, and item-level
aggregates additive across buckets (users are disjoint).
"""
import json
import time

import numpy as np
import polars as pl

from config import EVENTS_DIR, N_BUCKETS, RAW_ALBUM, RAW_ARTIST, RAW_EVENTS


def item_meta() -> pl.DataFrame:
    # a few items have several artists/albums: keep the smallest id to get a single parent
    artist = pl.read_parquet(RAW_ARTIST).group_by("item_id").agg(pl.col("artist_id").min())
    album = pl.read_parquet(RAW_ALBUM).group_by("item_id").agg(pl.col("album_id").min())
    return artist.join(album, on="item_id", how="full", coalesce=True)


def bucket_path(b: int):
    return EVENTS_DIR / f"b{b:02d}.parquet"


def main():
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = pl.scan_parquet(RAW_EVENTS)
    uids = raw.select(pl.col("uid").unique()).collect(engine="streaming")["uid"].sort().to_numpy()
    parts = np.array_split(uids, N_BUCKETS)
    bounds = [(int(p[0]), int(p[-1])) for p in parts]
    meta = item_meta().lazy()
    total = 0
    for b, (lo, hi) in enumerate(bounds):
        (
            raw.filter(pl.col("uid").is_between(lo, hi))
            .join(meta, on="item_id", how="left")
            .select(
                pl.col("uid"),
                pl.col("timestamp").alias("ts"),
                pl.col("item_id").alias("item"),
                pl.col("artist_id").alias("artist"),
                pl.col("album_id").alias("album"),
                pl.col("event_type").to_physical().cast(pl.UInt8).alias("etype"),
                pl.col("is_organic").alias("organic"),
                pl.col("played_ratio_pct").fill_null(0).alias("ratio"),
            )
            .sort("uid", "ts")
            .sink_parquet(bucket_path(b))
        )
        n = pl.scan_parquet(bucket_path(b)).select(pl.len()).collect().item()
        total += n
        print(f"bucket {b}: uids [{lo},{hi}] rows={n:,} t={time.time() - t0:.0f}s", flush=True)
    raw_n = raw.select(pl.len()).collect().item()
    assert total == raw_n, (total, raw_n)
    json.dump({"bounds": bounds, "rows": total}, open(EVENTS_DIR / "buckets.json", "w"))
    print(f"done: {total:,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
