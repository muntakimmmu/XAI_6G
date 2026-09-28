"""10-seed SPAN report: paired comparisons against Sentinel and Constrained GRPO.

    python scripts/span_report.py results/full    # reads protocol{A,B,C}_by_seed.csv there
Writes <dir>/SPAN_RESULTS.md.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd.stats import ci95, dominates, holm, paired  # noqa: E402

VARIANTS = ["span", "span_cgrpo_safe", "span_cgrpo", "span_nosafe", "span_abscost",
            "span_entropy", "maxmc_only", "nscsd", "ppo"]
NAMES = {"span": "SPAN", "span_cgrpo_safe": "SPAN + channel-normalised cost",
         "span_cgrpo": "Constrained GRPO (channel-normalised cost)", "span_nosafe": "SPAN w/o safe-side member",
         "span_abscost": "uncentred cost advantage", "span_entropy": "entropy bonus x6 (no safe member)",
         "maxmc_only": "MaxMC curriculum, no budget", "nscsd": "NS-CSD (original)", "ppo": "PPO+shield control",
         "Sentinel": "Sentinel (released ckpt)"}


def f(m, h, d=3):
    return f"{m:.{d}f} ± {h:.{d}f}"


def compare(df, a, b, scen, metrics=("benchmark", "quality", "leakage", "severe")):
    rows = []
    for sc in scen:
        x = df[(df.defender == a) & (df.scenario == sc)].set_index("seed")
        y = df[(df.defender == b) & (df.scenario == sc)].set_index("seed")
        s = x.index.intersection(y.index)
        if len(s) < 2:
            continue
        for m in metrics:
            r = paired(x.loc[s, m].values, y.loc[s, m].values)
            rows.append(dict(scenario=sc, metric=m, a=x.loc[s, m].mean(), b=y.loc[s, m].mean(), n=len(s), **r))
    t = pd.DataFrame(rows)
    if not t.empty:
        for m in t.metric.unique():
            i = t.metric == m
            t.loc[i, "holm"] = holm(t.loc[i, "t_p"].values)
    return t


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "results/full"
    md = ["# SPAN: 10-seed results\n", "Seeds 42-51, paired by seed; mean ± 95% t-CI; severe = steps per 500. "
          "Holm correction across scenarios per metric.\n"]
    for P in ("B", "A"):
        p = os.path.join(D, f"protocol{P}_by_seed.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        scen = [s for s in df.scenario.unique() if s != "benign"]
        md.append(f"\n## Protocol {P}: overview (mean over scenarios, then over seeds)\n")
        md.append("| Defender | composite | severe | leakage | quality | n |\n|---|---|---|---|---|---|")
        for d in ["Sentinel"] + VARIANTS:
            s = df[(df.defender == d) & df.scenario.isin(scen)]
            if s.empty:
                continue
            g = s.groupby("seed")[["benchmark", "severe", "leakage", "quality"]].mean()
            md.append(f"| {NAMES.get(d, d)} | {f(*ci95(g.benchmark))} | {f(*ci95(g.severe), 1)} | "
                      f"{f(*ci95(g.leakage))} | {f(*ci95(g.quality))} | {len(g)} |")
        for other in ("Sentinel", "span_cgrpo"):
            t = compare(df, "span", other, scen)
            if t.empty:
                continue
            t.to_csv(os.path.join(D, f"tests_{P}_span_vs_{other}.csv"), index=False)
            md.append(f"\n### Protocol {P}: SPAN vs {NAMES[other]} (paired)\n")
            md.append("| scenario | metric | SPAN | other | diff | p (t) | Holm p | Wilcoxon p | d_z |\n|---|---|---|---|---|---|---|---|---|")
            for _, r in t.iterrows():
                md.append(f"| {r.scenario} | {r.metric} | {r.a:.3f} | {r.b:.3f} | {r['diff']:+.3f} | {r.t_p:.2g} | "
                          f"{r.holm:.2g} | {r.w_p:.2g} | {r.d_z:+.2f} |")
            wins = t[(t.metric == "benchmark") & (t.holm < 0.05)]
            md.append(f"\nComposite: significant wins {int((wins['diff'] > 0).sum())}, significant losses "
                      f"{int((wins['diff'] < 0).sum())}, n.s. {int((t.metric == 'benchmark').sum() - len(wins))} of "
                      f"{int((t.metric == 'benchmark').sum())} scenarios.")
        md.append(f"\n### Protocol {P}: Pareto relation to Sentinel (seed means; quality up, leakage down, severe down)\n")
        md.append("| scenario | SPAN (q, leak, sev) | Sentinel (q, leak, sev) | relation |\n|---|---|---|---|")
        for sc in scen:
            a = df[(df.defender == "span") & (df.scenario == sc)][["quality", "leakage", "severe"]].mean().values
            b = df[(df.defender == "Sentinel") & (df.scenario == sc)][["quality", "leakage", "severe"]].mean().values
            rel = "dominates" if dominates(a, b) else "dominated" if dominates(b, a) else "trade-off"
            md.append(f"| {sc} | {a[0]:.3f}, {a[1]:.3f}, {a[2]:.1f} | {b[0]:.3f}, {b[1]:.3f}, {b[2]:.1f} | {rel} |")
    p = os.path.join(D, "protocolC_by_seed.csv")
    if os.path.exists(p):
        C = pd.read_csv(p)
        md.append("\n## Protocol C: CEM red team, worst composite found (higher = more robust)\n")
        md.append("| Defender | worst composite | worst severe | n |\n|---|---|---|---|")
        for d in ["Sentinel"] + VARIANTS:
            s = C[C.defender == d]
            if not s.empty:
                md.append(f"| {NAMES.get(d, d)} | {f(*ci95(s.worst_benchmark))} | {f(*ci95(s.worst_severe), 1)} | {len(s)} |")
        for other in ("Sentinel", "span_cgrpo"):
            x = C[C.defender == "span"].set_index("seed").worst_benchmark
            y = C[C.defender == other].set_index("seed").worst_benchmark
            s = x.index.intersection(y.index)
            if len(s) > 1:
                r = paired(x[s].values, y[s].values)
                md.append(f"\nSPAN vs {NAMES[other]}: diff {r['diff']:+.3f}, paired-t p={r['t_p']:.2g}, Wilcoxon p={r['w_p']:.2g}")
    # mechanism: training-time span rate, validation severe rate, final multiplier
    rows = []
    for m in VARIANTS:
        for h in glob.glob(f"runs/inline/{m}/seed*/history.csv"):
            d = pd.read_csv(h)
            if "span" not in d:
                continue
            late = d[d.iter > d.iter.max() - 100]
            rows.append(dict(method=m, span_late=late.span_rate.mean(), val_sev_final=d.val_sev_mean.iloc[-1],
                             lam_final=d.lam.iloc[-1], champion=d.champion.iloc[-1]))
    if rows:
        M = pd.DataFrame(rows)
        md.append("\n## Mechanism (training logs): group span rate, validation severe rate, final multiplier\n")
        md.append("| Variant | span rate (last 100 it) | val severe rate | final lambda | budget met (<=0.05) | n |\n|---|---|---|---|---|---|")
        for m, g in M.groupby("method"):
            md.append(f"| {NAMES.get(m, m)} | {f(*ci95(g.span_late))} | {f(*ci95(g.val_sev_final))} | "
                      f"{f(*ci95(g.lam_final), 2)} | {int((g.val_sev_final <= 0.05).sum())}/{len(g)} | {len(g)} |")
    out = os.path.join(D, "SPAN_RESULTS.md")
    with open(out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
