"""Compact single-column tables for the IEEE Networking Letters version (paper/nl).

Reads only the committed 10-seed CSVs; the mechanism figure is reused from paper/icc.
    python scripts/nl_assets.py
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd.stats import ci95, holm, paired  # noqa: E402

OUT = "paper/nl"
SCEN = ["mild", "strong", "chaos", "icmp", "pulse", "polymorph", "boundary", "icmp_chaos"]
LABEL = {"mild": "Mild", "strong": "Strong", "chaos": "Chaos/flash", "icmp": "ICMP (zero-shot)",
         "pulse": "Pulsing$^\\ast$", "polymorph": "Polymorphic$^\\ast$", "boundary": "Boundary$^\\ast$",
         "icmp_chaos": "ICMP-chaos$^\\ast$"}
DEF = ("Sentinel", "span_cgrpo", "span")


def load(P):
    df = pd.read_csv(f"results/full/protocol{P}_by_seed.csv")
    return df[df.scenario != "benign"]


def table_main():
    B = load("B")
    pv = []
    for sc in SCEN:
        a = B[(B.defender == "span") & (B.scenario == sc)].set_index("seed").sort_index()
        b = B[(B.defender == "Sentinel") & (B.scenario == sc)].set_index("seed").sort_index()
        pv.append(paired(a.benchmark.values, b.benchmark.values)["t_p"])
    adj = dict(zip(SCEN, holm(pv)))
    rows = []
    for sc in SCEN:
        v = {d: B[(B.defender == d) & (B.scenario == sc)][["benchmark", "severe"]].mean() for d in DEF}
        bc, bs = max(x.benchmark for x in v.values()), min(x.severe for x in v.values())
        cells = []
        for d in DEF:
            t = f"{v[d].benchmark:.3f}"
            t = f"\\textbf{{{t}}}" if np.isclose(v[d].benchmark, bc) else t
            cells.append(t + ("\\textsuperscript{\\dag}" if d == "span" and adj[sc] < 0.05 else ""))
        for d in DEF:
            t = f"{v[d].severe:.1f}"
            cells.append(f"\\textbf{{{t}}}" if np.isclose(v[d].severe, bs) else t)
        rows.append(f"{LABEL[sc]} & " + " & ".join(cells) + " \\\\")
    g = B.groupby(["defender", "seed"])[["benchmark", "severe"]].mean()
    mean = [f"{g.loc[d].benchmark.mean():.3f}" for d in DEF] + [f"{g.loc[d].severe.mean():.1f}" for d in DEF]
    rows += ["\\midrule", "Average & " + " & ".join(mean) + " \\\\"]
    tex = r"""\begin{table}[t]
\centering
\caption{Protocol B over 10 Seeds: Composite Score ($\uparrow$) and Severe Outages per 500 Windows ($\downarrow$)}
\label{tab:main}
\setlength{\tabcolsep}{2.4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{Composite score} & \multicolumn{3}{c}{Severe outages} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Scenario & Sentinel & C-GRPO & MIZAN & Sentinel & C-GRPO & MIZAN \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}

\vspace{2pt}
\parbox{\columnwidth}{\scriptsize $^\ast$Held-out stress test. \dag\,MIZAN above Sentinel, paired $t$-test, Holm-corrected, $p<0.05$. Bold: best.}
\end{table}
"""
    open(os.path.join(OUT, "tab_main.tex"), "w").write(tex)


def table_ablation():
    B, C = load("B"), pd.read_csv("results/full/protocolC_by_seed.csv")
    budget = {}
    for m in ("span", "span_cgrpo_safe", "span_cgrpo", "span_nosafe", "span_entropy", "span_abscost"):
        hs = [pd.read_csv(h) for h in glob.glob(f"runs/inline/{m}/seed*/history.csv")]
        budget[m] = f"{sum(h.val_sev_mean.iloc[-1] <= 0.05 for h in hs)}/{len(hs)}"
    names = [("Sentinel", "Sentinel~\\cite{alfatemi2026sentinel}"), ("ppo", "PPO + shield (equal budget)"),
             ("nscsd", "GRPO, no budget"), ("span_nosafe", "GRPO + budget"),
             ("span_entropy", "\\;\\; + entropy bonus $\\times6$"), ("span_abscost", "\\;\\; + uncentred cost"),
             ("span_cgrpo", "C-GRPO~\\cite{girgis2026cgrpo}"), ("span", "\\textbf{MIZAN}"),
             ("span_cgrpo_safe", "MIZAN + C-GRPO normalisation")]
    rows = []
    for d, lab in names:
        g = B[B.defender == d].groupby("seed")[["benchmark", "severe"]].mean()
        w = C[C.defender == d].worst_benchmark.mean()
        rows.append(f"{lab} & {g.benchmark.mean():.3f} & {g.severe.mean():.1f} & {w:.2f} & {budget.get(d, '--')} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Ablation and Alternative Remedies (10 Seeds). Worst: Red-Team Worst Case; Budget: Seeds Meeting $\sigma\le0.05$}
\label{tab:ablation}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Score $\uparrow$ & Severe $\downarrow$ & Worst $\uparrow$ & Budget \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write(tex)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for fig in ("mechanism.pdf", "tradeoff.pdf"):
        shutil.copy(os.path.join("paper/icc/figures", fig), os.path.join(OUT, "figures", fig))
    shutil.copy("paper/icc/refs.bib", os.path.join(OUT, "refs.bib"))
    table_main()
    table_ablation()
    print("wrote paper/nl tables")


if __name__ == "__main__":
    main()
