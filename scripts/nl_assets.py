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
Scenario & Sentinel & C-GRPO & IMBANG & Sentinel & C-GRPO & IMBANG \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}

\vspace{2pt}
\parbox{\columnwidth}{\scriptsize $^\ast$Held-out stress test. \dag\,IMBANG above Sentinel, paired $t$-test, Holm-corrected, $p<0.05$. Bold: best.}
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
             ("span_cgrpo", "C-GRPO~\\cite{girgis2026cgrpo}"), ("span", "\\textbf{IMBANG}"),
             ("span_cgrpo_safe", "IMBANG + C-GRPO normalisation")]
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


def fig_dynamics():
    """Validation outage rate and Lagrange multiplier over training (mean and 95% CI over 10 seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "scripts")
    from icc_assets import AQUA, BLUE, GRAY, style
    style()
    series = [("span_nosafe", "budget only", GRAY, "--"), ("span_cgrpo", "C-GRPO", AQUA, "-"), ("span", "IMBANG", BLUE, "-")]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.45))
    for m, lab, c, ls in series:
        hs = pd.concat([pd.read_csv(h).assign(seed=h) for h in glob.glob(f"runs/inline/{m}/seed*/history.csv")])
        for k, (col, a) in enumerate((("val_sev_mean", ax[0]), ("lam", ax[1]))):
            g = hs.groupby("iter")[col]
            mu, se = g.mean(), g.std() / np.sqrt(g.count())
            a.plot(mu.index, mu.values, color=c, ls=ls, lw=1.2, label=lab)
            a.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    ax[0].axhline(0.05, color="#0b0b0b", lw=0.8, ls=":", label="budget $\\kappa$")
    ax[0].set_ylabel("Val. outage rate")
    ax[1].set_ylabel("Multiplier $\\lambda$")
    ax[0].set_xlabel("(a) Iteration")
    ax[1].set_xlabel("(b) Iteration")
    h, lab = ax[1].get_legend_handles_labels()
    h0, l0 = ax[0].get_legend_handles_labels()
    h, lab = h + h0[-1:], lab + l0[-1:]
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.84))
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               columnspacing=1.0, handlelength=1.8)
    fig.savefig(os.path.join(OUT, "figures", "dynamics.pdf"))
    plt.close(fig)


def fig_intensity():
    """Severe outages near the overload cliff and leakage over the full intensity range (10 seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "scripts")
    from icc_assets import AQUA, BLUE, ORANGE, style
    style()
    df = pd.read_csv("results/intensity_sweep_by_seed.csv")
    series = [("Sentinel", "Sentinel", ORANGE, "D"), ("span_cgrpo", "C-GRPO", AQUA, "^"), ("span", "IMBANG", BLUE, "o")]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.45))
    for m, lab, c, mk in series:
        for col, a, lo in (("severe", ax[0], 0.85), ("leakage", ax[1], 0.0)):
            g = df[(df.defender == m) & (df.intensity >= lo)].groupby("intensity")[col]
            mu, se = g.mean(), g.std() / np.sqrt(g.count())
            a.plot(mu.index, mu.values, color=c, lw=1.2, marker=mk, ms=2.5, label=lab)
            a.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    ax[0].set_ylabel("Severe / 500 windows")
    ax[1].set_ylabel("Attack leakage")
    ax[0].set_xlabel("(a) Attack intensity $i$")
    ax[1].set_xlabel("(b) Attack intensity $i$")
    h, lab = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.84))
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               columnspacing=1.2, handlelength=1.8)
    fig.savefig(os.path.join(OUT, "figures", "intensity.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_dynamics()
    fig_intensity()
