"""A small feature language over the event log.

Every feature is `key . kind . event . window` (plus a second event/window for derived kinds),
e.g. `ui.cnt.lplus.30d`  = how many times the user listened (>=50%) to this track in 30 days,
     `ua.share.listen.7d` = share of the user's 7-day listens that went to the track's artist,
     `i.trend.listen.7d/30d` = item popularity growth.

Base kinds are computed as *partial* aggregates that can be summed / maxed over user buckets,
so item/artist/album-level features over the whole 500M log are computed bucket by bucket.
Both the hand-crafted set and the automatic search use this language, so they draw on
exactly the same information.
"""
from dataclasses import dataclass

import polars as pl

from config import DAY, DISLIKE, HOUR, LIKE, LISTEN, LPLUS_PCT, SKIP_PCT, UNLIKE

_is_listen = pl.col("etype") == LISTEN
EVENTS = {
    "listen": _is_listen,
    "lplus": _is_listen & (pl.col("ratio") >= LPLUS_PCT),
    "skip": _is_listen & (pl.col("ratio") < SKIP_PCT),
    "org": _is_listen & (pl.col("organic") == 1),
    "reco": _is_listen & (pl.col("organic") == 0),
    "like": pl.col("etype") == LIKE,
    "dislike": pl.col("etype") == DISLIKE,
    "unlike": pl.col("etype") == UNLIKE,
}
WINDOWS = {"6h": 6 * HOUR, "1d": DAY, "3d": 3 * DAY, "7d": 7 * DAY, "14d": 14 * DAY,
           "30d": 30 * DAY, "90d": 90 * DAY, "all": None}
WIN_ORDER = list(WINDOWS)
KEYS = {"u": ("uid",), "i": ("item",), "a": ("artist",), "al": ("album",),
        "ui": ("uid", "item"), "ua": ("uid", "artist"), "ual": ("uid", "album")}
GLOBAL_KEYS = ("i", "a", "al")  # aggregated over all users
USER_KEYS = ("u", "ui", "ua", "ual")
NUNIQ_OF = {"u": "item", "ua": "item", "ual": "item", "i": "uid", "a": "uid", "al": "uid"}

BASE_KINDS = ("cnt", "mratio", "nuniq", "since_last", "since_first")
RATE_PAIRS = [("lplus", "listen"), ("skip", "listen"), ("like", "listen"), ("dislike", "listen"),
              ("org", "listen"), ("unlike", "like")]
TREND_PAIRS = [("6h", "1d"), ("1d", "7d"), ("7d", "30d"), ("30d", "90d"), ("90d", "all")]


@dataclass(frozen=True)
class Feat:
    key: str
    kind: str
    ev: str
    w: str = ""
    ev2: str = ""
    w2: str = ""

    @property
    def name(self) -> str:
        if self.kind in ("since_last", "since_first"):
            return f"{self.key}.{self.kind}.{self.ev}"
        if self.kind == "rate":
            return f"{self.key}.rate.{self.ev}/{self.ev2}.{self.w}"
        if self.kind == "trend":
            return f"{self.key}.trend.{self.ev}.{self.w}/{self.w2}"
        return f"{self.key}.{self.kind}.{self.ev}.{self.w}"

    @property
    def is_base(self) -> bool:
        return self.kind in BASE_KINDS

    def deps(self) -> list["Feat"]:
        if self.kind == "rate":
            return [Feat(self.key, "cnt", self.ev, self.w), Feat(self.key, "cnt", self.ev2, self.w)]
        if self.kind == "share":
            return [Feat(self.key, "cnt", self.ev, self.w), Feat("u", "cnt", self.ev, self.w)]
        if self.kind == "trend":
            return [Feat(self.key, "cnt", self.ev, self.w), Feat(self.key, "cnt", self.ev, self.w2)]
        return [self]

    def with_window(self, w: str) -> "Feat":
        return Feat(self.key, self.kind, self.ev, w, self.ev2, self.w2)

    @property
    def group(self) -> str:
        return {"u": "user", "i": "item", "a": "artist", "al": "album",
                "ui": "user-item (direct)", "ua": "user-artist (indirect)", "ual": "user-album (indirect)"}[self.key]


def parse(name: str) -> Feat:
    key, kind, rest = name.split(".", 2)
    if kind in ("since_last", "since_first"):
        return Feat(key, kind, rest)
    ev, w = rest.rsplit(".", 1)
    if kind == "rate":
        e1, e2 = ev.split("/")
        return Feat(key, kind, e1, w, ev2=e2)
    if kind == "trend":
        w1, w2 = w.split("/")
        return Feat(key, kind, ev, w1, w2=w2)
    return Feat(key, kind, ev, w)


def generate_space() -> list[Feat]:
    out = []
    for k in KEYS:
        for e in EVENTS:
            for w in WINDOWS:
                out.append(Feat(k, "cnt", e, w))
            out.append(Feat(k, "since_last", e))
        for e in ("listen", "like"):
            out.append(Feat(k, "since_first", e))
        for e in ("listen", "org", "reco"):
            for w in WINDOWS:
                out.append(Feat(k, "mratio", e, w))
        if k in NUNIQ_OF:
            for e in ("listen", "lplus"):
                for w in ("1d", "7d", "30d", "all"):
                    out.append(Feat(k, "nuniq", e, w))
        for e1, e2 in RATE_PAIRS:
            for w in WINDOWS:
                out.append(Feat(k, "rate", e1, w, ev2=e2))
        for e in ("listen", "lplus"):
            for w1, w2 in TREND_PAIRS:
                out.append(Feat(k, "trend", e, w1, w2=w2))
        if k in ("ui", "ua", "ual"):
            for e in ("listen", "lplus", "skip", "like"):
                for w in WINDOWS:
                    out.append(Feat(k, "share", e, w))
    return out


def base_closure(feats) -> dict[str, list[Feat]]:
    """All base features needed to compute `feats`, grouped by key."""
    need = {}
    for f in feats:
        for d in f.deps():
            need.setdefault(d.key, {})[d.name] = d
    return {k: list(v.values()) for k, v in need.items()}


# ------------------------------------------------------------------ computation
def _cond(f: Feat, cutoff: int):
    c = EVENTS[f.ev]
    sec = WINDOWS.get(f.w)
    if sec is not None:
        c = c & (pl.col("ts") >= max(cutoff - sec, 0))
    return c


def _partials(f: Feat, cutoff: int):
    """[(column, expr, combine)] partial aggregates for a base feature."""
    c = _cond(f, cutoff)
    n = f.name
    if f.kind == "cnt":
        return [(n, c.sum().cast(pl.UInt32), "sum")]
    if f.kind == "mratio":
        return [(n + "#s", pl.col("ratio").filter(c).cast(pl.UInt64).sum(), "sum"),
                (n + "#n", c.sum().cast(pl.UInt32), "sum")]
    if f.kind == "nuniq":
        return [(n, pl.col(NUNIQ_OF[f.key]).filter(c).n_unique().cast(pl.UInt32), "sum")]
    if f.kind == "since_last":
        return [(n, pl.col("ts").filter(c).max(), "max")]
    if f.kind == "since_first":
        return [(n, pl.col("ts").filter(c).min(), "min")]
    raise ValueError(f.kind)


def compute_partial(h: pl.LazyFrame, key: str, feats: list[Feat], cutoff: int,
                    restrict: pl.DataFrame | None = None) -> pl.DataFrame:
    cols = list(KEYS[key])
    h = h.drop_nulls(cols)
    if restrict is not None:
        h = h.join(restrict.lazy(), on=cols, how="semi")
    exprs = [e.alias(n) for f in feats for n, e, _ in _partials(f, cutoff)]
    return h.group_by(cols).agg(exprs).collect()


def combine_partials(parts: list[pl.DataFrame], key: str, feats: list[Feat], cutoff: int) -> pl.DataFrame:
    cols = list(KEYS[key])
    ops = {"sum": pl.Expr.sum, "max": pl.Expr.max, "min": pl.Expr.min}
    aggs = [ops[op](pl.col(n)).alias(n) for f in feats for n, _, op in _partials(f, cutoff)]
    return pl.concat(parts).group_by(cols).agg(aggs)


def finalize(df: pl.DataFrame, feats: list[Feat], cutoff: int) -> pl.DataFrame:
    exprs = []
    for f in feats:
        n = f.name
        if f.kind == "mratio":
            exprs.append((pl.col(n + "#s") / pl.col(n + "#n")).alias(n))
        elif f.kind in ("since_last", "since_first"):
            exprs.append((pl.lit(cutoff, pl.Int64) - pl.col(n).cast(pl.Int64)).alias(n))
    return df.with_columns(exprs).drop([c for c in df.columns if "#" in c])


def fill_missing(df: pl.DataFrame, feats: list[Feat]) -> pl.DataFrame:
    """After left-joins: absent key = zero events (counts), recency stays null."""
    cols = [f.name for f in feats if f.kind in ("cnt", "nuniq") and f.name in df.columns]
    return df.with_columns([pl.col(c).fill_null(0) for c in cols])


def derived_expr(f: Feat, cutoff: int) -> pl.Expr:
    a, b = f.deps()
    num, den = pl.col(a.name).cast(pl.Float64), pl.col(b.name).cast(pl.Float64)
    if f.kind == "trend":
        num = num / (WINDOWS[f.w] or cutoff)
        den = den / (WINDOWS[f.w2] or cutoff)
    return pl.when(den > 0).then(num / den).otherwise(None).alias(f.name)


def add_derived(df: pl.DataFrame, feats: list[Feat], cutoff: int) -> pl.DataFrame:
    exprs = [derived_expr(f, cutoff) for f in feats if not f.is_base]
    return df.with_columns(exprs) if exprs else df


def neighbours(f: Feat) -> list[Feat]:
    """Same feature with adjacent time windows (used by window refinement)."""
    if not f.w or f.kind == "trend":
        return []
    i = WIN_ORDER.index(f.w)
    return [f.with_window(WIN_ORDER[j]) for j in (i - 1, i + 1) if 0 <= j < len(WIN_ORDER)]


SPACE = {f.name: f for f in generate_space()}
assert all(parse(n) == f for n, f in SPACE.items())
