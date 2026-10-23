"""Hand-crafted feature set: what a recsys engineer would typically write for a music ranker."""
from autofe.dsl import SPACE

MANUAL = [
    # user: activity level, taste for finishing / skipping / liking, organic vs recommendations
    "u.cnt.listen.7d", "u.cnt.listen.30d", "u.rate.lplus/listen.30d", "u.rate.skip/listen.7d",
    "u.rate.like/listen.all", "u.rate.org/listen.30d", "u.nuniq.listen.30d", "u.since_last.listen",
    # item: popularity now and its trend, quality signals
    "i.cnt.listen.1d", "i.cnt.listen.7d", "i.cnt.listen.30d", "i.trend.listen.7d/30d",
    "i.rate.lplus/listen.7d", "i.rate.skip/listen.30d", "i.rate.like/listen.30d", "i.rate.dislike/listen.all",
    "i.nuniq.listen.7d", "i.mratio.listen.30d", "i.since_first.listen",
    # direct user-item: repeat consumption
    "ui.cnt.listen.7d", "ui.cnt.lplus.30d", "ui.cnt.lplus.all", "ui.cnt.skip.30d", "ui.since_last.listen",
    "ui.since_last.lplus", "ui.since_last.like", "ui.since_last.dislike", "ui.mratio.listen.30d",
    "ui.since_first.listen",
    # indirect user-artist / user-album
    "ua.cnt.lplus.30d", "ua.share.listen.30d", "ua.share.listen.7d", "ua.rate.skip/listen.30d",
    "ua.since_last.listen", "ua.cnt.like.all", "ua.nuniq.listen.all",
    "ual.cnt.lplus.30d", "ual.share.listen.30d", "ual.since_last.listen",
    # artist popularity
    "a.cnt.listen.7d", "a.trend.listen.7d/30d",
]
assert len(set(MANUAL)) == len(MANUAL) and all(n in SPACE for n in MANUAL)
