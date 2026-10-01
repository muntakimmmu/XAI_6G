"""Figures and tables for the IEEE ICC paper (paper/icc), generated only from committed CSVs.

    python scripts/icc_assets.py
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd.stats import ci95, holm, paired  # noqa: E402

OUT = "paper/icc"
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # SPAN, Sentinel, Constrained GRPO
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8c8b86"
SCEN = ["mild", "strong", "chaos", "icmp", "pulse", "polymorph", "boundary", "icmp_chaos"]
SCEN_TEX = {"mild": "Mild", "strong": "Strong", "chaos": "Chaos/flash", "icmp": "ICMP (zero-shot)",
            "pulse": "Pulsing$^\\ast$", "polymorph": "Polymorphic$^\\ast$", "boundary": "Boundary-rate$^\\ast$",
            "icmp_chaos": "ICMP-chaos$^\\ast$"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def fig_mechanism():
    d = pd.read_csv("results/cliff_probe_final_by_seed.csv")
    s = pd.read_csv("results/cliff_probe_by_seed.csv")
    d = pd.concat([d, s[s.method == "Sentinel"]]).drop_duplicates(["method", "seed", "scenario"])
    d = d[d.scenario == "strong"]
    groups = [("grpo_vanilla", "GRPO", GRAY, "x"), ("nscsd", "NS-CSD (GRPO)", GRAY, "v"),
              ("ppo", "PPO", GRAY, "s"), ("Sentinel", "Sentinel", ORANGE, "D"),
              ("span_nosafe", "budget only", GRAY, "<"), ("span_cgrpo", "Constrained GRPO", AQUA, "^"),
              ("span_cgrpo_safe", "IMBANG + C-GRPO", BLUE, "s"), ("span", "IMBANG", BLUE, "o")]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85))
    for m, lab, c, mk in groups:
        g = d[d.method == m]
        kw = dict(color=c, marker=mk, s=12 if mk != "x" else 14, lw=0.8 if mk == "x" else 0.4,
                  edgecolor="white" if mk not in ("x",) else None, label=lab, zorder=3 if c != GRAY else 2)
        ax[0].scatter(g.drop_std_in_group, g.span_rate, **kw)
        ax[1].scatter(g.span_rate, g.severe_rate, **kw)
    ax[0].set_ylabel("Span rate")
    ax[1].set_ylabel("Severe-outage rate")
    ax[0].set_xlabel("(a) Within-group drop-rate std")
    ax[1].set_xlabel("(b) Span rate")
    h, lab = ax[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.80))
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.1,
               columnspacing=0.9, borderaxespad=0.0)
    fig.savefig(os.path.join(FIG, "mechanism.pdf"))
    plt.close(fig)
    r1 = np.corrcoef(d.drop_std_in_group, d.span_rate)[0, 1]
    r2 = np.corrcoef(d.span_rate, d.severe_rate)[0, 1]
    return dict(n=len(d), r_std_span=r1, r_span_sev=r2)


def load(P):
    df = pd.read_csv(f"results/full/protocol{P}_by_seed.csv")
    return df[df.scenario != "benign"]


def fig_tradeoff():
    B = load("B")
    agg = B.groupby(["defender", "seed"])[["leakage", "severe"]].mean().reset_index()
    show = [("Sentinel", "Sentinel", ORANGE, "D"), ("span_cgrpo", "C-GRPO", AQUA, "^"),
            ("span", "IMBANG", BLUE, "o"), ("ppo", "PPO+shield", GRAY, "s"),
            ("span_nosafe", None, GRAY, "<"), ("span_entropy", None, GRAY, ">"), ("span_abscost", None, GRAY, "P")]
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    offs = {"Sentinel": (0.002, 4, "left"), "span_cgrpo": (0.003, 3, "left"), "span": (0.003, -6, "left"),
            "ppo": (0.002, -7.5, "left")}
    for m, lab, c, mk in show:
        g = agg[agg.defender == m]
        (mx, hx), (my, hy) = ci95(g.leakage), ci95(g.severe)
        ax.errorbar(mx, my, xerr=hx, yerr=hy, fmt=mk, color=c, ms=4.8 if c != GRAY else 3.8, mec="white", mew=0.5,
                    elinewidth=0.8, capsize=1.5, zorder=3 if c != GRAY else 2)
        if lab:
            dx, dy, ha = offs[m]
            ax.annotate(lab, (mx + dx, my + dy), fontsize=7, color=c if c != GRAY else MUTED, ha=ha)
    ax.annotate("budget only (w/o safe member,\nentropy bonus, uncentred cost)", (0.812, 57), fontsize=6.5,
                color=MUTED, ha="center")
    ax.set_ylim(0, 65)
    ax.set_xlabel("Attack leakage (lower is better)")
    ax.set_ylabel("Severe outages / 500 steps")
    ax.set_xlim(0.785, 0.895)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.pdf"))
    plt.close(fig)


def mark(p):
    return "\\textsuperscript{\\dag}" if p < 0.05 else ""


def table_main():
    B = load("B")
    rows = []
    t_sen = {}
    for sc in SCEN:
        a = B[(B.defender == "span") & (B.scenario == sc)].set_index("seed")
        b = B[(B.defender == "Sentinel") & (B.scenario == sc)].set_index("seed")
        t_sen[sc] = paired(a.benchmark.values, b.loc[a.index].benchmark.values)["t_p"]
    adj = dict(zip(SCEN, holm([t_sen[s] for s in SCEN])))
    lines = []
    for sc in SCEN:
        vals = {}
        for d in ("Sentinel", "span_cgrpo", "span"):
            g = B[(B.defender == d) & (B.scenario == sc)]
            vals[d] = (g.benchmark.mean(), g.severe.mean(), g.leakage.mean())
        best_c = max(v[0] for v in vals.values())
        best_s = min(v[1] for v in vals.values())
        best_l = min(v[2] for v in vals.values())
        cells = []
        for k in (0, 1, 2):
            for d in ("Sentinel", "span_cgrpo", "span"):
                v = vals[d][k]
                txt = f"{v:.1f}" if k == 1 else f"{v:.3f}"
                if (k == 0 and np.isclose(v, best_c)) or (k == 1 and np.isclose(v, best_s)) or (k == 2 and np.isclose(v, best_l)):
                    txt = f"\\textbf{{{txt}}}"
                if k == 0 and d == "span":
                    txt += mark(adj[sc])
                cells.append(txt)
        lines.append(f"{SCEN_TEX[sc]} & " + " & ".join(cells) + " \\\\")
    agg = B.groupby(["defender", "seed"])[["benchmark", "severe", "leakage"]].mean().reset_index()
    cells = []
    for k in ("benchmark", "severe", "leakage"):
        for d in ("Sentinel", "span_cgrpo", "span"):
            m, h = ci95(agg[agg.defender == d][k])
            cells.append(f"{m:.1f}$\\pm${h:.1f}" if k == "severe" else f"{m:.3f}$\\pm${h:.3f}")
    lines.append("\\midrule")
    lines.append("Mean$\\pm$CI & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    tex = r"""\begin{table*}[t]
\centering
\caption{Attacker-agnostic Battery (Protocol B): Composite Score $\mathcal{B}$ ($\uparrow$), Severe Outages per 500 Steps ($\downarrow$) and Leakage ($\downarrow$), Mean over 10 Seeds}
\label{tab:main}
\setlength{\tabcolsep}{4.5pt}
\footnotesize
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Composite $\mathcal{B}$ ($\uparrow$)} & \multicolumn{3}{c}{Severe outages / 500 steps ($\downarrow$)} & \multicolumn{3}{c}{Attack leakage ($\downarrow$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Scenario & Sentinel~\cite{alfatemi2026sentinel} & C-GRPO~\cite{girgis2026cgrpo} & IMBANG & Sentinel & C-GRPO & IMBANG & Sentinel & C-GRPO & IMBANG \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{\textwidth}{\scriptsize $^\ast$Held-out stress test (never used for training or model selection). Bold: best per metric. \dag\,IMBANG better than Sentinel (paired $t$-test over seeds, Holm-corrected across scenarios, $p<0.05$). Last row: mean $\pm$ 95\% $t$-interval over seeds of the scenario average.}
\end{table*}
"""
    open(os.path.join(TAB, "main.tex"), "w").write(tex)


def table_ablation():
    B, A, C = load("B"), load("A"), pd.read_csv("results/full/protocolC_by_seed.csv")
    hist = {}
    for m in ["span", "span_cgrpo_safe", "span_cgrpo", "span_nosafe", "span_entropy", "span_abscost", "maxmc_only"]:
        vals = []
        for h in glob.glob(f"runs/inline/{m}/seed*/history.csv"):
            d = pd.read_csv(h)
            vals.append((d.val_sev_mean.iloc[-1] <= 0.05, d.lam.iloc[-1] if "lam" in d else 0.0))
        hist[m] = vals
    names = [("Sentinel", "Sentinel~\\cite{alfatemi2026sentinel}"), ("ppo", "PPO + shield"),
             ("nscsd", "GRPO (no budget)"), ("maxmc_only", "\\;+ MaxMC levels"),
             ("span_nosafe", "\\;+ budget"), ("span_entropy", "\\;+ budget, entropy$\\times$6"),
             ("span_abscost", "\\;+ budget, uncentred cost"), ("span_cgrpo", "\\;+ budget, C-GRPO~\\cite{girgis2026cgrpo}"),
             ("span", "\\textbf{IMBANG} (+ safe member)"), ("span_cgrpo_safe", "IMBANG + C-GRPO norm.")]
    lines = []
    for d, lab in names:
        gb = B[B.defender == d].groupby("seed")[["benchmark", "severe"]].mean()
        ga = A[A.defender == d].groupby("seed").benchmark.mean()
        gc = C[C.defender == d].worst_benchmark
        budget = f"{sum(v[0] for v in hist[d])}/{len(hist[d])}" if d in hist and hist[d] else "--"
        lam = f"{np.mean([v[1] for v in hist[d]]):.1f}" if d in hist and hist[d] else "--"
        lines.append(f"{lab} & {gb.benchmark.mean():.3f} & {gb.severe.mean():.1f} & {ga.mean():.3f} & "
                     f"{gc.mean():.2f} & {budget} & {lam} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Ablation and Competitors (10 Seeds). B/A: Protocol B/A Composite; Sev.: Severe Outages per 500 Steps (B); Worst: Red-team Worst Case (C); Budget: Seeds Meeting the 5\% Validation Outage Budget; $\bar\lambda$: Final Multiplier}
\label{tab:ablation}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
Method & B $\uparrow$ & Sev. $\downarrow$ & A $\uparrow$ & Worst $\uparrow$ & Budget & $\bar\lambda$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(TAB, "ablation.tex"), "w").write(tex)


def numbers():
    """Key statistics quoted in the text."""
    B, A, C = load("B"), load("A"), pd.read_csv("results/full/protocolC_by_seed.csv")
    out = {}
    for P, df in (("B", B), ("A", A)):
        g = df.groupby(["defender", "seed"])[["benchmark", "severe", "leakage", "quality"]].mean()
        for a, b in (("span", "Sentinel"), ("span", "span_cgrpo"), ("span_cgrpo_safe", "span_cgrpo")):
            for m in ("benchmark", "severe", "leakage", "quality"):
                r = paired(g.loc[a][m].values, g.loc[b][m].values)
                out[f"{P}:{a}-{b}:{m}"] = (round(g.loc[a][m].mean(), 3), round(g.loc[b][m].mean(), 3), round(r["diff"], 3),
                                          float(f"{r['w_p']:.2g}"), float(f"{r['t_p']:.2g}"))
    c = C.set_index(["defender", "seed"]).worst_benchmark
    for a, b in (("span", "Sentinel"), ("span", "span_cgrpo")):
        r = paired(c.loc[a].values, c.loc[b].values)
        out[f"C:{a}-{b}"] = (round(c.loc[a].mean(), 3), round(c.loc[b].mean(), 3), float(f"{r['t_p']:.2g}"))
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    style()
    mech = fig_mechanism()
    fig_tradeoff()
    table_main()
    table_ablation()
    nums = numbers()
    with open(os.path.join(OUT, "numbers.txt"), "w") as f:
        f.write(f"mechanism (strong, all checkpoints): {mech}\n")
        for k, v in nums.items():
            f.write(f"{k}: {v}\n")
    print(open(os.path.join(OUT, "numbers.txt")).read())


if __name__ == "__main__":
    main()
