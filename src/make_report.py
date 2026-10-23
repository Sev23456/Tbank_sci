"""Turn results/*.json|csv into markdown tables and figures used in README.md."""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from autofe.dsl import parse  # noqa: E402
from config import RESULTS  # noqa: E402

FIG = RESULTS / "fig"


def md(df: pl.DataFrame, floats=4) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append("| " + " | ".join(f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    parts = []
    tm = pl.read_csv(RESULTS / "test_metrics.csv")
    main_cols = ["model", "n_features", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "auc", "logloss",
                 "val_ndcg@10", "trees", "fit_rank_s"]
    tm = tm.select([c for c in main_cols if c in tm.columns])
    non_abl = tm.filter(~pl.col("model").str.starts_with("manual - "))
    parts.append("## Test metrics\n\n" + md(non_abl))
    parts.append("## Ablation (manual set minus a group)\n\n" + md(
        tm.filter(pl.col("model").str.starts_with("manual - ") | (pl.col("model") == "manual"))
        .select("model", "n_features", "ndcg@10", "recall@100")))

    bs = json.load(open(RESULTS / "bootstrap.json"))
    parts.append("## Paired bootstrap, test NDCG@10 vs manual\n\n" + md(pl.DataFrame(
        [{"model": k, "diff": v["mean_diff"], "ci95_lo": v["ci95"][0], "ci95_hi": v["ci95"][1]}
         for k, v in bs.items()]), 5))

    sel = json.load(open(RESULTS / "selection.json"))
    for key in ("auto_fwd_win",):
        fs = [parse(n) for n in sel[key]]
        comp = pl.DataFrame({"group": [f.group for f in fs], "kind": [f.kind for f in fs],
                             "window": [f.w or "-" for f in fs]})
        parts.append(f"## Composition of `{key}` ({len(fs)} features)\n\n"
                     + md(comp.group_by("group").len().sort("len", descending=True)) + "\n\n"
                     + md(comp.group_by("kind").len().sort("len", descending=True)) + "\n\n"
                     + md(comp.group_by("window").len().sort("len", descending=True)))
        parts.append("Selected features: " + ", ".join(f"`{n}`" for n in sel[key]))
    parts.append("Window swaps: " + "; ".join(f"`{a}` -> `{b}` ({s0:.4f} -> {s1:.4f})"
                                             for a, b, s0, s1 in sel["window_swaps"]))

    traj = sel["trajectory"]
    plt.figure(figsize=(7, 3.5))
    plt.plot([t["step"] for t in traj], [t["score"] for t in traj], ".", alpha=.4, label="tried")
    plt.plot([t["step"] for t in traj if t.get("best")], [t["best"] for t in traj if t.get("best")], "-",
             label="best so far")
    plt.xlabel("candidate feature (importance order)"); plt.ylabel("val NDCG@10"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "forward.png", dpi=120)

    windows = Counter(parse(n).w or "-" for n in sel["screened"])
    plt.figure(figsize=(6, 3))
    order = ["6h", "1d", "3d", "7d", "14d", "30d", "90d", "all", "-"]
    plt.bar(order, [windows.get(w, 0) for w in order])
    plt.title("windows among screened features"); plt.tight_layout(); plt.savefig(FIG / "windows.png", dpi=120)

    sc = pl.read_csv(RESULTS / "scaling.csv")
    plt.figure(figsize=(6, 3.5))
    for name in sc["set"].unique(maintain_order=True):
        s = sc.filter(pl.col("set") == name)
        plt.plot(s["train_users"], s["ndcg@10"], "o-", label=name)
    plt.xscale("log"); plt.xlabel("train users"); plt.ylabel("test NDCG@10"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "scaling.png", dpi=120)
    parts.append("## Scaling\n\n" + md(sc.select("set", "train_users", "train_rows", "ndcg@10", "recall@100",
                                                  "fit_s")))

    t = json.load(open(RESULTS / "timings.json"))
    parts.append("## Timings\n\n```json\n" + json.dumps(t, indent=1) + "\n```")
    tun = json.load(open(RESULTS / "tuning.json"))
    parts.append("## Tuning (manual set, val)\n\n" + md(pl.DataFrame(tun["grid"])))
    cs = json.load(open(RESULTS / "cand_stats.json"))
    parts.append("## Candidates\n\n```json\n" + json.dumps(cs, indent=1) + "\n```")
    (RESULTS / "report_tables.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("written", RESULTS / "report_tables.md")


if __name__ == "__main__":
    main()
