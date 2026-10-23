"""Paths and global constants of the experiment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "hub" / "datasets--yandex--yambda" / "snapshots" / "dd6f3a19eef5866e346c3270e098baa641a44948"
RAW_EVENTS = RAW / "flat" / "500m" / "multi_event.parquet"
RAW_ARTIST = RAW / "artist_item_mapping.parquet"
RAW_ALBUM = RAW / "album_item_mapping.parquet"

EVENTS_DIR = ROOT / "data" / "events"
# per-run artefacts: a smoke run (`--smoke`) never touches the full run's folders
SMOKE = "--smoke" in sys.argv
DATA = Path(os.environ.get("EXP_DATA", ROOT / ("data_smoke" if SMOKE else "data")))
RESULTS = Path(os.environ.get("EXP_RESULTS", ROOT / ("results_smoke" if SMOKE else "results")))

# time is in seconds (binned to 5s), same convention as the official Yambda benchmark
HOUR = 3600
DAY = 24 * HOUR
GAP = HOUR // 2
LAST = 26_000_000

N_BUCKETS = 16

# event type codes (physical order of the Enum in multi_event.parquet)
LISTEN, DISLIKE, LIKE, UNDISLIKE, UNLIKE = 0, 1, 2, 3, 4

LPLUS_PCT = 50  # "listen+" threshold from the Yambda paper
SKIP_PCT = 30

SEED = 42
