"""Time-based label windows and leakage-safe history access.

Layout (same test/val boundaries as the official Yambda benchmark):

    ... history ... | gap | train day | gap | val day | gap | test day ]
                                                                   LAST

For every split the features/candidates see only events with ts < cutoff = start - GAP,
labels are listen+ events inside [start, end).
"""
from dataclasses import dataclass

import polars as pl

from config import DAY, GAP, LAST, LISTEN, LPLUS_PCT


@dataclass(frozen=True)
class Split:
    name: str
    start: int
    end: int

    @property
    def cutoff(self) -> int:
        return self.start - GAP


def make_splits() -> dict[str, Split]:
    test = Split("test", LAST - DAY, LAST + 1)
    val = Split("val", LAST - 2 * DAY - GAP, LAST - DAY - GAP)
    train = Split("train", val.start - DAY - GAP, val.start - GAP)
    return {s.name: s for s in (train, val, test)}


SPLITS = make_splits()


def history(ev: pl.LazyFrame, split: Split) -> pl.LazyFrame:
    return ev.filter(pl.col("ts") < split.cutoff)


def labels(ev: pl.LazyFrame, split: Split) -> pl.LazyFrame:
    """Unique (uid, item) pairs with a listen+ inside the label window."""
    return (
        ev.filter(
            pl.col("ts").is_between(split.start, split.end, closed="left")
            & (pl.col("etype") == LISTEN)
            & (pl.col("ratio") >= LPLUS_PCT)
        )
        .select("uid", "item")
        .unique()
    )


def check_no_leak(hist: pl.DataFrame, split: Split):
    mx = hist["ts"].max()
    assert mx is None or mx < split.cutoff, f"{split.name}: history ts {mx} >= cutoff {split.cutoff}"


def check_layout():
    s = SPLITS
    assert s["train"].end + GAP == s["val"].start
    assert s["val"].end + GAP == s["test"].start
    # label windows never overlap the history of the same split
    for sp in s.values():
        assert sp.cutoff < sp.start
    # later splits may use earlier labels as history, never the reverse
    assert s["train"].end <= s["val"].cutoff and s["val"].end <= s["test"].cutoff


check_layout()
