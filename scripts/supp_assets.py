"""Tables and figures of the supplementary material (paper/nl/supp/).

Reads only committed results (results/) and training logs (runs/inline/*/seed*/), so every
number in the supplement is traceable to a file shipped with it.
    python scripts/supp_assets.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import scenarios as S  # noqa: E402
from nscsd.grpo import Config  # noqa: E402
from nscsd.stats import ci95, paired  # noqa: E402

OUT = "paper/nl/supp"
FIG = os.path.join(OUT, "figures")
ATT = ["mild", "strong", "chaos", "icmp", "pulse", "polymorph", "boundary", "icmp_chaos"]
SCEN = ["benign"] + ATT
SLAB = {"benign": "Benign", "mild": "Mild", "strong": "Strong", "chaos": "Chaos/flash",
        "icmp": "ICMP (zero-shot)", "pulse": "Pulsing$^\\ast$", "polymorph": "Polymorphic$^\\ast$",
        "boundary": "Boundary$^\\ast$", "icmp_chaos": "ICMP-chaos$^\\ast$"}
ALAB = {"benign": "Benign", "mild_attack": "Mild", "strong_attack": "Strong",
        "chaos_flash_crowd": "Chaos/flash crowd", "zero_shot_icmp": "Zero-shot ICMP"}
MAIN = ["Sentinel", "ppo", "span_cgrpo", "span"]
MLAB = {"Sentinel": "Sentinel", "ppo": "PPO + shield", "nscsd": "GRPO, no budget",
        "grpo_vanilla": "Plain GRPO", "maxmc_only": "MaxMC curriculum, no budget",
        "span_nosafe": "GRPO + budget", "span_entropy": "\\;\\;+ entropy bonus $\\times6$",
        "span_abscost": "\\;\\;+ uncentred cost", "span_cgrpo": "C-GRPO",
        "span": "\\textbf{IMBANG}", "span_cgrpo_safe": "IMBANG + C-GRPO norm."}
SHORT = {"Sentinel": "Sentinel", "ppo": "PPO", "span_cgrpo": "C-GRPO", "span": "IMBANG"}
HEUR = {"Adaptive": "Adaptive heuristic", "Static": "Static rule", "ShieldOnly": "Random actions + shield",
        "Random": "Random actions"}
ALL = ["Sentinel", "ppo", "nscsd", "maxmc_only", "span_nosafe", "span_entropy",
       "span_abscost", "span_cgrpo", "span", "span_cgrpo_safe"]


def pm(m, h, d=3):
    return f"${m:.{d}f}\\pm{h:.{d}f}$"


def pfmt(p):
    if p >= 0.01:
        return f"{p:.2f}" if p < 0.995 else "1.00"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def write(name, tex):
    open(os.path.join(OUT, name), "w").write(tex)


def table(name, caption, label, colspec, header, rows, note=None, size="\\small"):
    body = "\n".join(rows)
    foot = f"\n\\vspace{{2pt}}\n\\parbox{{\\textwidth}}{{\\footnotesize {note}}}" if note else ""
    write(name, f"""\\begin{{table}}[!htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{size}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colspec}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}{foot}
\\end{{table}}
""")


def seedmean(df, d, cols):
    return df[df.defender == d].groupby("seed")[cols].mean()


# ------------------------------------------------------------------ settings
def tab_scenarios():
    fams = ["none", "SYN", "UDP", "HTTP", "ICMP", "mixed"]
    role = {"benign": "Sentinel scenario", "mild": "Sentinel scenario", "strong": "Sentinel scenario",
            "chaos": "Sentinel scenario", "icmp": "Sentinel scenario (unseen family)"}
    rows = []
    for k in SCEN:
        v = S.EVAL[k]
        w = v[S.W]
        act = [(fams[i], w[i]) for i in range(6) if w[i] > 0]
        fam = ", ".join(f for f, _ in act) if np.allclose([x for _, x in act], act[0][1]) else \
            ", ".join(f"{f} ({x:.2f})" for f, x in act)
        pulse = f"{int(v[S.PER])} / {v[S.DUTY]:.1f}" if v[S.PER] > 0 else "--"
        rows.append(f"{SLAB[k]} & {fam} & {v[S.STAY]:.2f} & [{v[S.ILO]:.1f}, {v[S.IHI]:.1f}] & "
                    f"[{v[S.MLO]:.1f}, {v[S.MHI]:.1f}] & {pulse} & {v[S.FLASH]:.2f} & {role.get(k, 'held-out stress test')} \\\\")
    table("tab_scenarios.tex", "Protocol~B Evaluation Scenarios", "tab:s-scen", "@{}lp{4.2cm}cccccl@{}",
          "Scenario & Attack families (weights) & Persistence & Intensity & Mutation & Pulse period / duty & Flash prob. & Role \\\\",
          rows, note="Families switch with probability $1-$persistence per window; intensity and mutation are "
                     "redrawn uniformly from their ranges on each switch. $^\\ast$Never used for training or model selection.")


def tab_hyper():
    c = Config()
    rows = [
        "\\multicolumn{3}{@{}l}{\\emph{GRPO learner (shared by all GRPO variants)}} \\\\",
        f"Iterations & {c.iters} & validation every {c.eval_every} iterations \\\\",
        f"Contexts per iteration $B$ / group size $G$ / horizon $H$ & {c.B} / {c.G} / {c.H} & warm-up 0--{c.warmup_max} windows before the fork \\\\",
        "Group composition (IMBANG) & 4 + 1 + 2 + 1 & policy samples, safe-side member, shield probes, archive elite \\\\",
        f"Optimiser & Adam, lr ${c.lr:g}$ & {c.epochs} epochs, minibatch {c.minibatch}, gradient-norm clip 0.5 \\\\",
        f"Clipping / entropy / KL-anchor weight & {c.clip} / {c.ent_coef} / {c.kl_coef} & KL to the best validated checkpoint \\\\",
        f"Weight of the likelihood term $c$ & {c.elite_coef} & safe-side member and elite, $-c\\,[\\hat A]^+\\log\\pi_\\theta$ \\\\",
        f"Objective & composite score $b_t$ & Sentinel's benchmark weights (severe 0.5, degraded 0.25) \\\\",
        "\\midrule",
        "\\multicolumn{3}{@{}l}{\\emph{IMBANG-specific}} \\\\",
        "Safe-side shift $u$ & $\\mathcal{U}(0.2, 0.8)$ & drawn once per rollout \\\\",
        "Outage budget $\\kappa$ & 0.05 & severe-outage rate on the validation battery \\\\",
        f"Dual step $\\eta_\\lambda$ & {c.lam_lr} & per validation round, $\\lambda_0=0$ \\\\",
        "Champion selection & feasibility first & then composite validation score \\\\",
        "Curriculum & MaxMC, 360 levels & 5 families $\\times$ 6 intensity bands $\\times$ 3 mutation bands $\\times$ pulse $\\times$ flash \\\\",
        f"Shield probes / perturbation & {c.shield_probes} / $\\sigma={c.shield_sigma}$ & certified PanicGuard tuning \\\\",
        "\\midrule",
        "\\multicolumn{3}{@{}l}{\\emph{Network and evaluation}} \\\\",
        "Actor & 12--64--64 MLP & Beta heads for $\\tau$ and $d$, categorical head for $f$ (Sentinel's architecture) \\\\",
        "Seeds & 42--51 & paired with Sentinel's released checkpoints \\\\",
        "Validation battery & benign, mild, strong, chaos & ICMP never seen before testing \\\\",
        "Protocol B / A & 20 / 5 episodes & 100 windows each \\\\",
        "Protocol C (red team) & CEM, 48 $\\times$ 8 & 4 episodes of 60 windows per candidate; worst scenario re-scored on 20 fresh episodes of 100 \\\\",
    ]
    table("tab_hyper.tex", "Hyperparameters (Fixed Before the Ten-Seed Runs)", "tab:s-hyper",
          "@{}lll@{}", "Setting & Value & Note \\\\", rows)


def tab_reproduction():
    r = pd.read_csv("results/reproduction.csv")
    r = r[r.defender_type == "Sentinel"]
    rows = [f"{ALAB[x.scenario]} & {x.quality_pub:.3f} & {x.quality:.3f} & {x.leakage_pub:.3f} & {x.leakage:.3f} & "
            f"{x.severe_pub:.1f} & {x.severe:.1f} \\\\" for x in r.itertuples()]
    table("tab_repro.tex", "Sentinel's Released Checkpoints: Published Results and Our Port",
          "tab:s-repro", "@{}lcccccc@{}",
          " & \\multicolumn{2}{c}{Quality} & \\multicolumn{2}{c}{Leakage} & \\multicolumn{2}{c}{Severe / 500} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\nScenario & Published & Port & Published & Port & Published & Port \\\\",
          rows, note="Published: Sentinel's evaluation tables. Port: our batched simulator with Sentinel's evaluation "
                     "protocol and its released seed-42--51 checkpoints.")


# ------------------------------------------------------------------ main results
def tab_protB():
    B = pd.read_csv("results/full/protocolB_by_seed.csv")
    blocks = [((("benchmark", 3), ("severe", 1)), "tab_protB1.tex",
               "Protocol~B: Composite Score and Severe Outages per 500 Windows (Mean $\\pm$ 95\\% CI over Ten Seeds)",
               "tab:s-b1", ("Composite score $\\uparrow$", "Severe outages $\\downarrow$")),
              ((("quality", 3), ("leakage", 3)), "tab_protB2.tex",
               "Protocol~B: Service Quality and Attack Leakage (Mean $\\pm$ 95\\% CI over Ten Seeds)",
               "tab:s-b2", ("Service quality $\\uparrow$", "Attack leakage $\\downarrow$"))]
    for (m1, m2), name, cap, lab, heads in blocks:
        rows = []
        for sc in SCEN + ["avg"]:
            cells = []
            for col, d in (m1, m2):
                for dfn in MAIN:
                    x = B[(B.defender == dfn) & (B.scenario != "benign")].groupby("seed")[col].mean() if sc == "avg" \
                        else B[(B.defender == dfn) & (B.scenario == sc)].set_index("seed")[col]
                    cells.append(pm(*ci95(x), d))
            if sc == "avg":
                rows.append("\\midrule")
            rows.append(f"{'Average (attacks)' if sc == 'avg' else SLAB[sc]} & " + " & ".join(cells) + " \\\\")
        hdr = (f" & \\multicolumn{{4}}{{c}}{{{heads[0]}}} & \\multicolumn{{4}}{{c}}{{{heads[1]}}} \\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\nScenario & " + " & ".join([SHORT[d] for d in MAIN] * 2) + " \\\\")
        table(name, cap, lab, "@{}l" + "c" * 8 + "@{}", hdr, rows,
              note="$^\\ast$Held-out stress test. Severe outages are windows with service quality below 0.5. "
                   "PPO: PPO + shield trained with the same budget of simulator windows.")


def tab_protA():
    A = pd.read_csv("results/full/protocolA_by_seed.csv")
    rows = []
    for sc in ALAB:
        cells = []
        for col, dd in (("benchmark", 3), ("severe", 1), ("leakage", 3), ("quality", 3)):
            for d in ("Sentinel", "span"):
                x = A[(A.defender == d) & (A.scenario == sc)].set_index("seed")[col]
                cells.append(pm(*ci95(x), dd) if col in ("benchmark", "severe") else f"{x.mean():.{dd}f}")
        rows.append(f"{ALAB[sc]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    cells, Aa, ps = [], A[A.scenario != "benign"], {}
    for col, dd in (("benchmark", 3), ("severe", 1), ("leakage", 3), ("quality", 3)):
        g = {d: Aa[Aa.defender == d].groupby("seed")[col].mean() for d in ("Sentinel", "span")}
        ps[col] = paired(g["span"].values, g["Sentinel"].loc[g["span"].index].values)["t_p"]
        for d in ("Sentinel", "span"):
            cells.append(pm(*ci95(g[d]), dd) if col in ("benchmark", "severe") else f"{g[d].mean():.{dd}f}")
    rows.append("Average (attacks) & " + " & ".join(cells) + " \\\\")
    hdr = (" & \\multicolumn{2}{c}{Composite $\\uparrow$} & \\multicolumn{2}{c}{Severe / 500 $\\downarrow$} & "
           "\\multicolumn{2}{c}{Leakage $\\downarrow$} & \\multicolumn{2}{c}{Quality $\\uparrow$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
           "Scenario & Sentinel & IMBANG & Sentinel & IMBANG & Sentinel & IMBANG & Sentinel & IMBANG \\\\")
    table("tab_protA.tex", "Protocol~A: Sentinel's Own Trained Attackers (Ten Seeds, Five Episodes Each)",
          "tab:s-a", "@{}l" + "c" * 8 + "@{}", hdr, rows,
          note="For each seed, the attacker that Sentinel co-trained with its defender drives the traffic, as in "
               "Sentinel's own evaluation script; for IMBANG this is a transfer test. Leakage is averaged over all windows. "
               f"Paired $t$-tests on the attack average: composite $p={ps['benchmark']:.3f}$, severe $p={ps['severe']:.2f}$, "
               f"leakage $p={ps['leakage']:.3f}$, quality $p={ps['quality']:.2f}$.")


def tab_tests():
    rows = []
    for other, lab in (("Sentinel", "IMBANG vs Sentinel"), ("span_cgrpo", "IMBANG vs C-GRPO")):
        t = pd.read_csv(f"results/full/tests_B_span_vs_{other}.csv")
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        for sc in ATT:
            cells = []
            for m, dd in (("benchmark", 3), ("severe", 1)):
                r = t[(t.scenario == sc) & (t.metric == m)].iloc[0]
                cells += [f"{r['diff']:+.{dd}f}", pfmt(r.holm), pfmt(r.w_p), f"{r.d_z:+.2f}"]
            rows.append(f"{SLAB[sc]} & " + " & ".join(cells) + " \\\\")
        if other == "Sentinel":
            rows.append("\\midrule")
    hdr = (" & \\multicolumn{4}{c}{Composite score} & \\multicolumn{4}{c}{Severe outages / 500} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
           "Scenario & $\\Delta$ & Holm $p$ & Wilcoxon $p$ & $d_z$ & $\\Delta$ & Holm $p$ & Wilcoxon $p$ & $d_z$ \\\\")
    table("tab_tests.tex", "Paired Tests over Ten Seeds (Protocol~B)", "tab:s-tests", "@{}l" + "c" * 8 + "@{}",
          hdr, rows, note="$\\Delta$: IMBANG minus comparator, averaged over seeds. Holm: paired $t$-test, "
                          "Holm-corrected across the eight scenarios per metric. Wilcoxon: signed-rank test "
                          "(two-sided, uncorrected; its smallest attainable value with ten pairs is 0.002). "
                          "$d_z$: standardised paired effect size.")


def runs_summary(m):
    out = []
    for h in sorted(glob.glob(f"runs/inline/{m}/seed*/history.csv")):
        d = pd.read_csv(h).rename(columns={"straddle": "span_rate"})
        late = d[d.iter > d.iter.max() - 100]
        out.append(dict(val=d.val_sev_mean.iloc[-1] if "val_sev_mean" in d else np.nan,
                        lam=d.lam.iloc[-1] if "lam" in d else np.nan,
                        span=late.span_rate.mean() if "span_rate" in d else np.nan))
    return pd.DataFrame(out)


def tab_methods():
    A, B = pd.read_csv("results/full/protocolA_by_seed.csv"), pd.read_csv("results/full/protocolB_by_seed.csv")
    C = pd.read_csv("results/full/protocolC_by_seed.csv")
    B = B[B.scenario != "benign"]
    rows = []
    for m in ALL:
        g = seedmean(B, m, ["benchmark", "severe", "leakage", "quality"])
        a = seedmean(A[A.scenario != "benign"], m, ["benchmark"]).benchmark
        c = C[C.defender == m].worst_benchmark
        rs = runs_summary(m) if m != "Sentinel" else pd.DataFrame()
        bud = f"{int((rs.val <= 0.05).sum())}/{len(rs)}" if len(rs) and rs.val.notna().any() else "--"
        lam = f"{rs.lam.mean():.2f}" if len(rs) and rs.lam.notna().any() else "--"
        span = f"{rs.span.mean():.2f}" if len(rs) and rs.span.notna().any() else "--"
        rows.append(f"{MLAB[m]} & {pm(*ci95(g.benchmark))} & {pm(*ci95(g.severe), 1)} & {g.leakage.mean():.3f} & "
                    f"{g.quality.mean():.3f} & {a.mean():.3f} & {pm(*ci95(c), 2)} & {bud} & {lam} & {span} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{10}{@{}l}{\\emph{Reference heuristics from Sentinel's code (no learning)}} \\\\")
    for m, lab in HEUR.items():
        g = seedmean(B, m, ["benchmark", "severe", "leakage", "quality"])
        a = seedmean(A[A.scenario != "benign"], m, ["benchmark"]).benchmark
        c = C[C.defender == m].worst_benchmark
        rows.append(f"{lab} & {pm(*ci95(g.benchmark))} & {pm(*ci95(g.severe), 1)} & {g.leakage.mean():.3f} & "
                    f"{g.quality.mean():.3f} & {a.mean():.3f} & {pm(*ci95(c), 2)} & -- & -- & -- \\\\")
    hdr = ("Method & Composite (B) & Severe (B) & Leakage (B) & Quality (B) & Composite (A) & Worst (C) & "
           "Budget met & Final $\\lambda$ & Span rate \\\\")
    table("tab_methods.tex", "All Trained Methods (Ten Seeds; Protocol~B Averaged over the Eight Attack Scenarios)",
          "tab:s-methods", "@{}l" + "c" * 9 + "@{}", hdr, rows,
          note="Worst (C): composite score of the worst scenario found by the red team. Budget met: seeds whose final "
               "validation outage rate is at most $\\kappa=0.05$. Span rate: share of training groups whose members "
               "differ in outage count, averaged over the last 100 iterations. --: not applicable or not logged. "
               "Adaptive: Sentinel's traffic-responsive heuristic; Static: fixed protocol filtering (threshold 0.8, drop 0.5).")


def tab_redteam():
    C = pd.read_csv("results/full/protocolC_by_seed.csv")
    rows = []
    for m in ALL:
        s = C[C.defender == m]
        fam = s.worst_family.value_counts()
        rows.append(f"{MLAB[m]} & {pm(*ci95(s.worst_benchmark), 2)} & {pm(*ci95(s.worst_severe), 1)} & "
                    f"{s.worst_quality.mean():.3f} & {s.worst_leakage.mean():.3f} & {s.worst_intensity.mean():.2f} & "
                    + ", ".join(f"{k} {v}" for k, v in fam.items()) + " \\\\")
    rows.append("\\midrule")
    for m, lab in HEUR.items():
        s_ = C[C.defender == m]
        fam = s_.worst_family.value_counts()
        rows.append(f"{lab} & {pm(*ci95(s_.worst_benchmark), 2)} & {pm(*ci95(s_.worst_severe), 1)} & "
                    f"{s_.worst_quality.mean():.3f} & {s_.worst_leakage.mean():.3f} & {s_.worst_intensity.mean():.2f} & "
                    + ", ".join(f"{k} {v}" for k, v in fam.items()) + " \\\\")
    table("tab_redteam.tex", "Protocol~C: Worst Case Found by the Cross-Entropy Red Team (Ten Seeds)", "tab:s-redteam",
          "@{}lcccccl@{}", "Method & Worst composite & Severe / 500 & Quality & Leakage & Intensity & Worst family (seeds) \\\\",
          rows, note="The red team searches the full 14-dimensional scenario space, ICMP included, for the scenario "
                     "that minimises each defender's composite score.")


def tab_mechanism():
    d = pd.read_csv("results/cliff_probe_final_by_seed.csv")
    order = ["Sentinel", "ppo", "grpo_vanilla", "nscsd", "span_nosafe", "span_cgrpo", "span", "span_cgrpo_safe"]
    rows = []
    for m in order:
        cells = []
        for sc in ("strong", "polymorph"):
            x = d[(d.method == m) & (d.scenario == sc)]
            cells += [pm(*ci95(x.span_rate), 2), f"{x.drop_std_in_group.mean():.3f}", pm(*ci95(x.severe_rate), 2)]
        rows.append(f"{MLAB[m]} & " + " & ".join(cells) + " \\\\")
    hdr = (" & \\multicolumn{3}{c}{Strong flood} & \\multicolumn{3}{c}{Polymorphic flood} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
           "Method & Span rate & Drop spread & Severe rate & Span rate & Drop spread & Severe rate \\\\")
    table("tab_mechanism.tex", "Constraint-Cancellation Probe on the 80 Trained Checkpoints of Fig.~2 of the Letter",
          "tab:s-mech", "@{}lcccccc@{}", hdr, rows,
          note="From overload states reached after 20 deterministic windows, 64 contexts are forked into $G=8$ "
               "stochastic members sharing traffic and rolled out for $H=8$ windows. Span rate: share of groups "
               "whose members differ in outage count. Drop spread: within-group standard deviation of the sampled "
               "drop probability. Severe rate: share of member-windows in severe outage.")


def tab_safecheck():
    d = pd.read_csv("results/safe_member_check_by_seed.csv")
    rows = []
    for sc in SCEN:
        x = d[d.scenario == sc]
        rows.append(f"{SLAB[sc]} & {x.severe_policy.mean():.2f} & {x.severe_safe.mean():.2f} & {x.cd_policy.mean():.4f} & "
                    f"{x.cd_safe.mean():.4f} & {x.leak_policy.mean():.3f} & {x.leak_safe.mean():.3f} & "
                    f"{100 * x.ep_fewer.mean():.1f} & {100 * x.ep_more.mean():.1f} \\\\")
    hdr = (" & \\multicolumn{2}{c}{Severe / 500} & \\multicolumn{2}{c}{Collateral damage} & "
           "\\multicolumn{2}{c}{Leakage} & \\multicolumn{2}{c}{Episodes (\\%)} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
           "Scenario & Policy & Safe & Policy & Safe & Policy & Safe & Safe fewer & Safe more \\\\")
    table("tab_safecheck.tex", "Multi-Step Check of the Safe-Side Direction (IMBANG, Ten Seeds, 20 Episodes of 100 Windows)",
          "tab:s-safe", "@{}l" + "c" * 8 + "@{}", hdr, rows,
          note="Each trained IMBANG policy and its safe-shifted copy ($\\tilde d=u\\,d$, "
               "$\\tilde\\tau=\\tau+(1-u)(1-\\tau)$, $u\\sim\\mathcal{U}(0.2,0.8)$ per episode) face identical "
               "traffic with the seeds of Protocol~B; the policy columns therefore equal Table~\\ref{tab:s-b1}. "
               "Safe fewer / more: episodes in which the safe copy has strictly fewer / more severe windows.")


def tab_human():
    h = pd.read_csv("results/human/human_in_loop_by_seed.csv")
    h = h[h.operator.isin(["none", "icmp-only"])]
    rows = []
    for d, lab in (("Sentinel", "Sentinel"), ("SPAN", "IMBANG")):
        for op, acc in (("none", None), ("icmp-only", 0.7), ("icmp-only", 0.9), ("icmp-only", 1.0)):
            x = h[(h.defender == d) & (h.operator == op)]
            x = x[x.acc.isna()] if acc is None else x[np.isclose(x.acc, acc)]
            cells = []
            for sc in ("icmp", "icmp_chaos", "strong"):
                y = x[x.scenario == sc]
                cells += [f"{y.severe.mean():.1f}", f"{y.benchmark.mean():.3f}"]
            sw = x[x.scenario == "icmp"].switches_per_ep.mean()
            rows.append(f"{lab} & {'none' if acc is None else f'ICMP playbook, {acc:.1f}'} & " + " & ".join(cells)
                        + f" & {sw:.1f} \\\\")
        if d == "Sentinel":
            rows.append("\\midrule")
    hdr = (" & & \\multicolumn{2}{c}{ICMP} & \\multicolumn{2}{c}{ICMP-chaos} & \\multicolumn{2}{c}{Strong} & \\\\\n"
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
           "Defender & Operator (accuracy) & Severe & Composite & Severe & Composite & Severe & Composite & Switches \\\\")
    table("tab_human.tex", "Simulated Operator Applying Sentinel's ICMP Constants (Ten Seeds, 20 Episodes)",
          "tab:s-human", "@{}ll" + "c" * 7 + "@{}", hdr, rows,
          note="Every ten windows a simulated operator recognises the active family with the stated accuracy and, "
               "after three windows, applies protocol filtering with drop 0.5 and threshold 0.5 during ICMP floods. "
               "This study draws its own traffic, so its no-operator rows differ slightly from Table~\\ref{tab:s-b1}. "
               "Severe per 500 windows; switches: playbook changes per ICMP episode.")


def tab_seeds():
    B = pd.read_csv("results/full/protocolB_by_seed.csv")
    B = B[(B.defender == "span") & (B.scenario != "benign")].groupby("seed")[["benchmark", "severe"]].mean()
    rows = []
    for p in sorted(glob.glob("runs/inline/span/seed*")):
        s = int(p.split("seed")[-1])
        meta = json.load(open(f"{p}/meta.json"))
        h = pd.read_csv(f"{p}/history.csv")
        tf, sh = meta["tree_fidelity"], meta["champion_shield"]
        v = h.val_sev_mean.iloc[-1]
        rows.append(f"{s} & {B.loc[s, 'benchmark']:.3f} & {B.loc[s, 'severe']:.1f} & {v:.3f} & "
                    f"{'yes' if v <= 0.05 else 'no'} & {h.lam.iloc[-1]:.2f} & {100 * tf['focus_acc']:.1f} & "
                    f"{tf['thr_mae']:.3f} & {tf['drop_mae']:.3f} & {tf['leaves']:.0f} & "
                    f"{sh['panic_util']:.2f} / {sh['panic_drop']:.2f} / {sh['panic_thr']:.2f} \\\\")
    hdr = ("Seed & Composite (B) & Severe (B) & Val.\\ outage rate & Budget met & Final $\\lambda$ & Focus acc.\\ (\\%) & "
           "Threshold MAE & Drop MAE & Leaves & PanicGuard gate / cap / floor \\\\")
    table("tab_seeds.tex", "IMBANG per Seed: Results, Budget, Rule Distillation and Certified Shield",
          "tab:s-seeds", "@{}c" + "c" * 10 + "@{}", hdr, rows,
          note="Rule distillation: depth-4 decision trees fitted to the final policy, evaluated on held-out states. "
               "PanicGuard defaults are 0.60 / 0.10 / 0.40.")


def tab_intensity():
    d = pd.read_csv("results/intensity_sweep_by_seed.csv")
    rows = []
    for i in sorted(d.intensity.unique()):
        cells = []
        for col, dd in (("severe", 1), ("leakage", 3), ("quality", 3)):
            for m in ("Sentinel", "span_cgrpo", "span"):
                cells.append(f"{d[(d.defender == m) & np.isclose(d.intensity, i)][col].mean():.{dd}f}")
        rows.append(f"{i:.2f} & " + " & ".join(cells) + " \\\\")
    hdr = (" & \\multicolumn{3}{c}{Severe / 500 $\\downarrow$} & \\multicolumn{3}{c}{Leakage $\\downarrow$} & "
           "\\multicolumn{3}{c}{Quality $\\uparrow$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}\n"
           "Intensity & Sentinel & C-GRPO & IMBANG & Sentinel & C-GRPO & IMBANG & Sentinel & C-GRPO & IMBANG \\\\")
    table("tab_intensity.tex", "Operating Curve over Attack Intensity (Data of Fig.~3 of the Letter)", "tab:s-int",
          "@{}c" + "c" * 9 + "@{}", hdr, rows,
          note="Training attack mix (SYN, UDP, HTTP, mixed), 20 episodes of 100 windows per point and seed, "
               "identical traffic for all defenders; means over ten seeds.")


# ------------------------------------------------------------------ figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "scripts")
    from icc_assets import AQUA, BLUE, GRAY, ORANGE, style
    style()
    col = {"Sentinel": ORANGE, "ppo": GRAY, "span_cgrpo": AQUA, "span": BLUE}
    B = pd.read_csv("results/full/protocolB_by_seed.csv")
    Ba = B[B.scenario != "benign"]

    # S1: per-seed paired comparison
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for k, (c, lab) in enumerate((("benchmark", "Composite score (higher is better)"),
                                  ("severe", "Severe outages / 500 (lower is better)"))):
        g = {m: Ba[Ba.defender == m].groupby("seed")[c].mean() for m in ("Sentinel", "span_cgrpo", "span")}
        for s in g["span"].index:
            ax[k].plot([0, 1, 2], [g[m][s] for m in ("Sentinel", "span_cgrpo", "span")], color="#bbbbbb", lw=0.7, zorder=1)
        for j, m in enumerate(("Sentinel", "span_cgrpo", "span")):
            ax[k].scatter(np.full(len(g[m]), j), g[m].values, color=col[m], s=14, zorder=2)
        ax[k].set_xticks([0, 1, 2], ["Sentinel", "C-GRPO", "IMBANG"])
        ax[k].set_ylabel(lab)
        ax[k].set_xlim(-0.4, 2.4)
        ax[k].set_xlabel(f"({'ab'[k]})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "seeds.pdf"))
    plt.close(fig)

    # S2: span rate during training
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    for m, lab, c, ls in (("span_nosafe", "GRPO + budget", GRAY, "--"), ("span_cgrpo", "C-GRPO", AQUA, "-"),
                          ("span", "IMBANG", BLUE, "-")):
        hs = pd.concat([pd.read_csv(h).rename(columns={"straddle": "span_rate"})
                        for h in glob.glob(f"runs/inline/{m}/seed*/history.csv")])
        g = hs.groupby("iter").span_rate
        mu, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.plot(mu.index, mu.values, color=c, ls=ls, label=lab)
        ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Span rate")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "spanrate.pdf"))
    plt.close(fig)

    # S3: severe outages per scenario
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    w = 0.2
    for j, m in enumerate(MAIN):
        mu, hw = zip(*[ci95(B[(B.defender == m) & (B.scenario == sc)].set_index("seed").severe) for sc in ATT])
        ax.bar(np.arange(len(ATT)) + (j - 1.5) * w, mu, w, yerr=hw, color=col[m], label=SHORT[m],
               error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(np.arange(len(ATT)), [SLAB[s].replace("$^\\ast$", "*") for s in ATT], fontsize=7)
    ax.set_ylabel("Severe outages / 500")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scenarios.pdf"))
    plt.close(fig)


def numbers():
    d = pd.read_csv("results/safe_member_check_by_seed.csv")
    n_ep = len(d) * 20
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write(f"\\newcommand{{\\nSafeEpisodes}}{{{n_ep:,}}}\n".replace(",", "{,}"))
        f.write(f"\\newcommand{{\\safeMorePct}}{{{100 * d.ep_more.mean():.1f}}}\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    for fn in (tab_scenarios, tab_hyper, tab_reproduction, tab_protB, tab_protA, tab_tests, tab_methods,
               tab_redteam, tab_mechanism, tab_safecheck, tab_human, tab_seeds, tab_intensity, figures, numbers):
        fn()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
