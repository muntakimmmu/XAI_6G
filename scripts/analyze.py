"""Turn per-seed evaluation CSVs and training logs into tables, tests and figures.

    python scripts/analyze.py            # reads results/, runs/, writes results/*.md and paper/figures/*.pdf
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd.stats import ci95, dominates, holm, paired  # noqa: E402

RES, FIG = "results", "paper/figures"
from nscsd.paths import SENTINEL_RUN  # noqa: E402
# Reference categorical palette (fixed order): NS-CSD, Sentinel, PPO; everything else is gray ink.
C = {"nscsd": "#2a78d6", "Sentinel": "#eb6834", "ppo": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {
    "nscsd": "NS-CSD (ours)", "nscsd-final": "NS-CSD final", "nscsd-tree": "NS-CSD tree program",
    "nscsd_safe": "NS-CSD safety-weighted", "Sentinel": "Sentinel", "Sentinel-final": "Sentinel final",
    "PPO-NoShield (Sentinel)": "PPO NoShield", "ppo": "PPO+shield (ours, no co-evo.)", "ppo-final": "PPO+shield final",
    "grpo_vanilla": "GRPO+shield", "no_shield_discovery": "- shield discovery", "no_discovery": "- scenario curriculum",
    "no_elite": "- elite injection", "no_anchor": "- champion anchor", "no_crn": "- common random numbers",
    "no_shield": "- shield (GRPO NoShield)", "ShieldOnly": "ShieldOnly", "Static": "Static", "Adaptive": "Adaptive",
    "Random": "Random",
}
MAIN = ["Random", "Static", "Adaptive", "ShieldOnly", "PPO-NoShield (Sentinel)", "Sentinel", "ppo", "nscsd",
        "nscsd-tree", "nscsd_safe"]
ABL = ["nscsd", "no_shield_discovery", "no_discovery", "no_elite", "no_anchor", "no_crn", "no_shield",
       "grpo_vanilla", "ppo"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def fmt(m, h, d=3):
    return f"{m:.{d}f} ± {h:.{d}f}"


def seed_table(df, scen_order, defenders, metrics, leak_note=""):
    lines = []
    for sc in scen_order:
        sub = df[df.scenario == sc]
        lines.append(f"\n**{sc}**\n\n| Defender | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics))
        for d in defenders:
            s = sub[sub.defender == d]
            if s.empty:
                continue
            cells = []
            for m in metrics:
                mu, h = ci95(s[m])
                cells.append(fmt(mu, h, 1 if m in ("severe", "degraded", "sla_miss") else 3))
            lines.append(f"| {NAMES.get(d, d)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + leak_note


def tests(df, a, b, scen_order, metrics=("benchmark", "quality", "leakage", "severe")):
    rows = []
    for sc in scen_order:
        sub = df[df.scenario == sc]
        x = sub[sub.defender == a].sort_values("seed")
        y = sub[sub.defender == b].sort_values("seed")
        if x.empty or y.empty:
            continue
        for m in metrics:
            r = paired(x[m].values, y[m].values)
            rows.append(dict(scenario=sc, metric=m, a=a, b=b, mean_a=x[m].mean(), mean_b=y[m].mean(), **r))
    t = pd.DataFrame(rows)
    if not t.empty:
        t["holm_p"] = np.nan
        for m in t.metric.unique():
            idx = t.metric == m
            t.loc[idx, "holm_p"] = holm(t.loc[idx, "t_p"].values)
    return t


def pareto_rows(df, a, b, scen_order):
    out = []
    for sc in scen_order:
        sub = df[df.scenario == sc]
        pa = sub[sub.defender == a][["quality", "leakage", "severe"]].mean().values
        pb = sub[sub.defender == b][["quality", "leakage", "severe"]].mean().values
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        rel = "dominates" if dominates(pa, pb) else "dominated" if dominates(pb, pa) else "trade-off"
        out.append(f"| {sc} | {NAMES.get(a, a)} {rel} {NAMES.get(b, b)} | q {pa[0]:.3f}/{pb[0]:.3f} | "
                   f"leak {pa[1]:.3f}/{pb[1]:.3f} | severe {pa[2]:.1f}/{pb[2]:.1f} |")
    return out


def histories(physics="inline"):
    h = []
    for p in glob.glob(f"runs/{physics}/*/seed*/history.csv"):
        m, s = p.split("/")[-3], int(p.split("/")[-2][4:])
        d = pd.read_csv(p)
        d["method"], d["seed"] = m, s
        h.append(d)
    return pd.concat(h) if h else pd.DataFrame()


def fig_training(H):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    ax = axes[0]
    for m, lab in (("nscsd", "NS-CSD live policy"), ("ppo", "PPO+shield live policy"),
                   ("grpo_vanilla", "GRPO+shield live policy")):
        d = H[H.method == m]
        if d.empty:
            continue
        g = d.groupby("iter")
        x = g.env_steps.mean() / 1e6
        mu, sd = g.val.mean(), g.val.std() / np.sqrt(g.val.count())
        col = C.get(m, MUTED)
        ls = "-" if m != "grpo_vanilla" else "--"
        ax.plot(x, mu, color=col, ls=ls, label=lab)
        ax.fill_between(x, mu - 1.96 * sd, mu + 1.96 * sd, color=col, alpha=0.15, lw=0)
        if m == "nscsd":
            ax.plot(x, g.champion.mean(), color=col, ls=":", label="NS-CSD archive champion")
    ax.set_xlabel("Training environment steps (millions)")
    ax.set_ylabel("Validation composite score")
    ax.set_title("(a) NS-CSD / controls, validation battery (no ICMP)")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    cs = os.path.join(SENTINEL_RUN, "checkpoint_selection.csv")
    if os.path.exists(cs):
        s = pd.read_csv(cs)
        g = s.groupby("generation").benchmark_score
        mu, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.plot(mu.index, mu.values, color=C["Sentinel"], label="Sentinel (released run, 10 seeds)")
        ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=C["Sentinel"], alpha=0.15, lw=0)
        ax.set_xlabel("Co-evolution generation")
        ax.set_ylabel("Sentinel benchmark score")
        ax.set_title("(b) Sentinel co-evolution, its own battery")
        ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_training.pdf"))
    plt.close(fig)


def fig_pareto(B, scen_order):
    defs = [("Sentinel", C["Sentinel"], "s"), ("ppo", C["ppo"], "^"), ("nscsd", C["nscsd"], "o")]
    fig, axes = plt.subplots(1, len(scen_order), figsize=(7.16, 1.9), sharey=False)
    for ax, sc in zip(axes, scen_order):
        sub = B[B.scenario == sc]
        for d in ("Static", "Adaptive", "ShieldOnly", "Random"):
            s = sub[sub.defender == d]
            ax.scatter(s.leakage.mean(), s.quality.mean(), s=12, color="#b7b6b0", zorder=1)
            ax.annotate(d, (s.leakage.mean(), s.quality.mean()), fontsize=5.5, color=MUTED,
                        xytext=(2, 2), textcoords="offset points")
        for d, col, mk in defs:
            s = sub[sub.defender == d]
            if s.empty:
                continue
            ax.scatter(s.leakage, s.quality, s=6, color=col, alpha=0.35, marker=mk, lw=0, zorder=2)
            ax.scatter(s.leakage.mean(), s.quality.mean(), s=36, color=col, marker=mk, edgecolor="white",
                       lw=1.2, zorder=3, label=NAMES[d])
        ax.set_title(sc)
        ax.set_xlabel("Attack leakage")
    axes[0].set_ylabel("Service quality")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pareto.pdf"))
    plt.close(fig)


def fig_ablation(B, scen):
    base = B[(B.defender == "nscsd") & B.scenario.isin(scen)].groupby("seed").benchmark.mean()
    rows = []
    for d in ABL[1:]:
        s = B[(B.defender == d) & B.scenario.isin(scen)].groupby("seed").benchmark.mean()
        common = base.index.intersection(s.index)
        if len(common) < 2:
            continue
        diff = s[common] - base[common]
        mu, h = ci95(diff.values)
        rows.append((NAMES.get(d, d), mu, h))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color="#9aa6b8", height=0.6,
            error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Δ composite vs. full NS-CSD (mean over 9 scenarios)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf"))
    plt.close(fig)


def fig_redteam(Cdf):
    order = ["Random", "Static", "Adaptive", "ShieldOnly", "PPO-NoShield (Sentinel)", "Sentinel-final", "Sentinel",
             "ppo", "grpo_vanilla", "no_shield_discovery", "nscsd", "nscsd-tree", "nscsd_safe"]
    order = [o for o in order if o in set(Cdf.defender)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for i, d in enumerate(order):
        mu, h = ci95(Cdf[Cdf.defender == d].worst_benchmark)
        col = C.get(d, "#b7b6b0")
        ax.barh(i, mu, xerr=h, color=col, height=0.6, error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax.set_yticks(range(len(order)), [NAMES.get(o, o) for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("Worst-case composite found by CEM red team (higher = more robust)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_redteam.pdf"))
    plt.close(fig)


def fig_shield(H):
    d = H[H.method == "nscsd"]
    if d.empty or "shield_panic_util" not in d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.8))
    for ax, k, lab, dflt in zip(axes, ("panic_util", "panic_drop", "panic_thr"),
                                ("PanicGuard utilisation gate", "PanicGuard drop cap", "PanicGuard threshold floor"),
                                (0.6, 0.1, 0.4)):
        for s, g in d.groupby("seed"):
            ax.plot(g.env_steps / 1e6, g[f"shield_{k}"], color=C["nscsd"], alpha=0.35, lw=0.8)
        g = d.groupby("iter")
        ax.plot(g.env_steps.mean() / 1e6, g[f"shield_{k}"].mean(), color=C["nscsd"], lw=1.8, label="mean of 10 seeds")
        ax.axhline(dflt, color=C["Sentinel"], ls="--", lw=1, label="Sentinel default")
        ax.set_title(lab)
        ax.set_xlabel("Env steps (M)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_shield.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    style()
    md = ["# NS-CSD results (auto-generated by scripts/analyze.py)\n",
          "All values: mean ± 95% t-interval over seeds 42-51 (n=10 unless noted). Counts are per 500 steps.\n"]
    A = pd.read_csv(os.path.join(RES, "protocolA_by_seed.csv")) if os.path.exists(os.path.join(RES, "protocolA_by_seed.csv")) else None
    B = pd.read_csv(os.path.join(RES, "protocolB_by_seed.csv"))
    Cdf = pd.read_csv(os.path.join(RES, "protocolC_by_seed.csv")) if os.path.exists(os.path.join(RES, "protocolC_by_seed.csv")) else None
    core = ["benign", "mild", "strong", "chaos", "icmp"]
    stress = ["pulse", "polymorph", "boundary", "icmp_chaos"]
    metrics = ["quality", "leakage", "severe", "degraded", "benchmark"]

    md.append("## Protocol B - attacker-agnostic battery (core scenarios)\n")
    md.append(seed_table(B, core, MAIN, metrics))
    md.append("\n\n## Protocol B - held-out stress scenarios\n")
    md.append(seed_table(B, stress, MAIN, metrics))
    if A is not None:
        md.append("\n\n## Protocol A - Sentinel-native (Sentinel's trained attacker drives the traffic)\n")
        A = A.replace({"scenario": {"mild_attack": "mild", "strong_attack": "strong", "chaos_flash_crowd": "chaos",
                                    "zero_shot_icmp": "icmp"}})
        md.append(seed_table(A, core, MAIN, metrics, "\n\n(Leakage here is Sentinel's all-step mean.)"))

    md.append("\n\n## Paired tests: NS-CSD vs Sentinel and vs PPO+shield (Holm across scenarios per metric)\n")
    for proto, df in (("B", B), ("A", A)):
        if df is None:
            continue
        for other in ("Sentinel", "ppo"):
            t = tests(df, "nscsd", other, core + (stress if proto == "B" else []))
            if t.empty:
                continue
            t.to_csv(os.path.join(RES, f"tests_{proto}_nscsd_vs_{other}.csv"), index=False)
            md.append(f"\n**Protocol {proto}: NS-CSD vs {NAMES[other]}**\n\n| scenario | metric | NS-CSD | other | diff | "
                      "paired-t p | Holm p | Wilcoxon p | d_z |\n|---|---|---|---|---|---|---|---|---|")
            for _, r in t.iterrows():
                md.append(f"| {r.scenario} | {r.metric} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r['diff']:+.3f} | "
                          f"{r.t_p:.3g} | {r.holm_p:.3g} | {r.w_p:.3g} | {r.d_z:+.2f} |")

    md.append("\n\n## Pareto relations (seed means; quality up, leakage down, severe down)\n")
    for proto, df, sc in (("B", B, core + stress), ("A", A, core)):
        if df is None:
            continue
        md.append(f"\n**Protocol {proto}**\n\n| scenario | relation | quality | leakage | severe |\n|---|---|---|---|---|")
        md += pareto_rows(df, "nscsd", "Sentinel", sc)
        md += pareto_rows(df, "nscsd", "ppo", sc)

    md.append("\n\n## Ablations (Protocol B, composite averaged over all 9 scenarios; paired vs full NS-CSD)\n")
    md.append("| Variant | composite (9 scen.) | core-5 | stress-4 | ICMP | Δ vs full | paired-t p | n |\n|---|---|---|---|---|---|---|---|")
    base = B[B.defender == "nscsd"].groupby("seed").benchmark.mean()
    for d in ABL + ["nscsd_safe", "Sentinel", "Sentinel-final"]:
        s = B[B.defender == d]
        if s.empty:
            continue
        allm = s.groupby("seed").benchmark.mean()
        c5 = s[s.scenario.isin(core)].groupby("seed").benchmark.mean()
        s4 = s[s.scenario.isin(stress)].groupby("seed").benchmark.mean()
        ic = s[s.scenario == "icmp"].groupby("seed").benchmark.mean()
        common = base.index.intersection(allm.index)
        p = paired(allm[common].values, base[common].values) if d != "nscsd" else dict(diff=0, t_p=1)
        md.append(f"| {NAMES.get(d, d)} | {fmt(*ci95(allm))} | {fmt(*ci95(c5))} | {fmt(*ci95(s4))} | {fmt(*ci95(ic))} | "
                  f"{p['diff']:+.3f} | {p['t_p']:.3g} | {len(allm)} |")

    md.append("\n\n## Final vs selected policy (late-stage degradation), Protocol B composite over 9 scenarios\n")
    md.append("| Method | selected | final | final - selected | paired-t p |\n|---|---|---|---|---|")
    for sel, fin in (("Sentinel", "Sentinel-final"), ("nscsd", "nscsd-final"), ("ppo", "ppo-final")):
        a = B[B.defender == sel].groupby("seed").benchmark.mean()
        b = B[B.defender == fin].groupby("seed").benchmark.mean()
        if a.empty or b.empty:
            continue
        p = paired(b.values, a.values)
        md.append(f"| {NAMES.get(sel, sel)} | {fmt(*ci95(a))} | {fmt(*ci95(b))} | {p['diff']:+.3f} | {p['t_p']:.3g} |")

    H = histories()
    if not H.empty:
        md.append("\n\n## Training dynamics (validation battery)\n")
        md.append("| Method | peak live val | final live val | peak - final | rollbacks | champion iter |\n|---|---|---|---|---|---|")
        for m in ABL + ["nscsd_safe"]:
            d = H[H.method == m]
            if d.empty:
                continue
            per = d.groupby("seed").agg(peak=("val", "max"), final=("val", "last"),
                                        rb=("rollbacks", "last") if "rollbacks" in d else ("val", "size"),
                                        ci=("champion_iter", "last"))
            gap = per.peak - per.final
            md.append(f"| {NAMES.get(m, m)} | {fmt(*ci95(per.peak))} | {fmt(*ci95(per.final))} | {fmt(*ci95(gap))} | "
                      f"{per.rb.mean():.1f} | {per.ci.mean():.0f} |")
        cs = os.path.join(SENTINEL_RUN, "checkpoint_selection.csv")
        if os.path.exists(cs):
            s = pd.read_csv(cs)
            per = s.sort_values("generation").groupby("seed").agg(peak=("benchmark_score", "max"),
                                                                  final=("benchmark_score", "last"))
            md.append(f"| Sentinel (its own battery, released run) | {fmt(*ci95(per.peak))} | {fmt(*ci95(per.final))} | "
                      f"{fmt(*ci95(per.peak - per.final))} | - | - |")
        fig_training(H)
        fig_shield(H)

    metas = []
    for p in glob.glob("runs/inline/nscsd/seed*/meta.json"):
        m = json.load(open(p))
        metas.append(dict(seed=int(p.split("/")[-2][4:]), **{f"sh_{k}": v for k, v in (m.get("champion_shield") or {}).items()},
                          accepted=len(m.get("shield_log") or []), tree_score=m.get("tree_score"),
                          champion_score=m.get("champion_score"), **(m.get("tree_fidelity") or {}),
                          **{f"arch_{k}": v for k, v in (m.get("archive") or {}).items()}))
    if metas:
        M = pd.DataFrame(metas).sort_values("seed")
        M.to_csv(os.path.join(RES, "nscsd_meta_by_seed.csv"), index=False)
        md.append("\n\n## Discovered shields, crystallised programs and archive composition (NS-CSD, per seed)\n")
        md.append(M.round(3).to_markdown(index=False))
        md.append("\n\nMeans: " + ", ".join(f"{c}={M[c].mean():.3f}" for c in M.columns if c != "seed" and M[c].dtype != object))

    R = os.path.join(RES, "shield_rules_by_seed.csv")
    if os.path.exists(R):
        rr = pd.read_csv(R)
        g = rr.groupby(["defender", "scenario"]).mean(numeric_only=True).drop(columns="seed").round(1)
        g = g.loc[:, (g > 0).any()]
        md.append("\n\n## Shield rule activations per 500 steps (Protocol B)\n")
        md.append(g.to_markdown())

    if Cdf is not None:
        md.append("\n\n## Protocol C - CEM red team (worst case found; full scenario space incl. ICMP)\n")
        md.append("| Defender | worst composite | quality | leakage | severe | most common worst family |\n|---|---|---|---|---|---|")
        for d in Cdf.defender.unique():
            s = Cdf[Cdf.defender == d]
            md.append(f"| {NAMES.get(d, d)} | {fmt(*ci95(s.worst_benchmark))} | {fmt(*ci95(s.worst_quality))} | "
                      f"{fmt(*ci95(s.worst_leakage))} | {fmt(*ci95(s.worst_severe), 1)} | {s.worst_family.mode().iat[0]} |")
        t = []
        for other in ("Sentinel", "ppo"):
            x = Cdf[Cdf.defender == "nscsd"].sort_values("seed").worst_benchmark.values
            y = Cdf[Cdf.defender == other].sort_values("seed").worst_benchmark.values
            if len(x) == len(y) and len(x) > 1:
                r = paired(x, y)
                t.append(f"NS-CSD vs {NAMES[other]}: diff {r['diff']:+.3f}, paired-t p={r['t_p']:.3g}, Wilcoxon p={r['w_p']:.3g}")
        md.append("\n\n" + "\n\n".join(t))
        fig_redteam(Cdf)

    fig_pareto(B, ["mild", "strong", "chaos", "icmp", "polymorph"])
    fig_ablation(B, core + stress)
    with open(os.path.join(RES, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/RESULTS.md and figures")


if __name__ == "__main__":
    main()
