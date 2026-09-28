"""Simulated human-in-the-loop discrete decisions on top of an autonomous defender.

A (simulated) operator reviews the link every `period` control steps, recognises the
active attack family correctly with probability `acc` (otherwise picks a random family),
and after `delay` steps applies one of three discrete playbooks:

  AUTO          leave the autonomous defender in control
  ICMP          protocol filtering, drop 0.5, threshold 0.5   (Sentinel's ICMP override constants)
  CONSERVATIVE  drop <= 0.1, threshold >= 0.4                 (Sentinel's PanicGuard constants)

Rule: ICMP family -> ICMP playbook; a low-rate flood (intensity < 0.6) under the
"quality-first" policy -> CONSERVATIVE; otherwise AUTO.  Playbook constants are fixed a priori
from Sentinel's shield, not tuned on these results.  The same operator runs on top of every
defender.  All humans here are simulated; this is a sensitivity analysis, not a user study.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import env as E  # noqa: E402
from nscsd import scenarios as S  # noqa: E402
from nscsd.agents import NetDefender  # noqa: E402
from nscsd.policy import DefenderNet, load_sentinel_defender  # noqa: E402
from nscsd.rollout import benchmark_score, deploy  # noqa: E402
from nscsd.stats import ci95, paired  # noqa: E402

torch.set_num_threads(1)
SM = "/home/user/alialfatemi/sentinel-ddos/outputs/run_20260523_200520_paper/models"
AUTO, ICMP, CONS = 0, 1, 2


def run(defender, scen, seed, operator=None, n=20, T=100, period=10, delay=3, acc=1.0, quality_first=True):
    rng = np.random.default_rng(seed)
    (typ, inten, mut), tape = S.contexts(rng, np.tile(scen, (n, 1)), T)
    orng = np.random.default_rng(seed + 12345)
    obs = E.initial_obs(tape.legit0)
    mode, pending = np.full(n, AUTO), [None] * n
    acc_sum = {k: np.zeros(n) for k in ("sq", "leak_att", "severe", "degraded", "attacked")}
    switches = 0
    for t in range(T):
        if operator and t % period == 0:
            seen = np.where(orng.random(n) < acc, typ[:, t], orng.choice([0, 1, 2, 3, 4, 5], n))
            want = np.full(n, AUTO)
            want = np.where(seen == 4, ICMP, want)
            if quality_first:
                want = np.where((seen > 0) & (seen != 4) & (inten[:, t] < 0.6), CONS, want)
            for i in range(n):
                pending[i] = (t + delay, want[i])
        if operator:
            for i in range(n):
                if pending[i] is not None and pending[i][0] == t:
                    switches += int(mode[i] != pending[i][1])
                    mode[i] = pending[i][1]
        (thr, focus, drop), _, _ = deploy(defender, obs)
        thr, focus, drop = thr.copy(), focus.copy(), drop.copy()
        ic, co = mode == ICMP, mode == CONS
        thr[ic], focus[ic], drop[ic] = 0.5, 1, 0.5
        drop[co] = np.minimum(drop[co], 0.1)
        thr[co] = np.maximum(thr[co], 0.4)
        obs, info = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), (thr, focus, drop), np.ones(n, bool))
        for k in ("sq", "severe", "degraded", "attacked"):
            acc_sum[k] += info[k]
        acc_sum["leak_att"] += info["leak"] * info["attacked"]
    sq = acc_sum["sq"].mean() / T
    leak = float(np.mean(np.where(acc_sum["attacked"] > 0, acc_sum["leak_att"] / np.maximum(acc_sum["attacked"], 1), 0)))
    sev, deg = acc_sum["severe"].mean() / T, acc_sum["degraded"].mean() / T
    return dict(quality=sq, leakage=leak, severe=500 * sev, benchmark=benchmark_score(sq, leak, sev, deg),
                switches_per_ep=switches / n)


def main():
    scen = ["icmp", "mild", "boundary", "icmp_chaos", "strong", "chaos", "pulse", "polymorph"]
    rows = []
    for seed in range(42, 52):
        p = f"runs/inline/span/seed{seed}"
        net = DefenderNet(); net.load_state_dict(torch.load(f"{p}/champion.pt", weights_only=True))
        kw = json.load(open(f"{p}/meta.json"))["champion_shield"]
        defs = {"SPAN": NetDefender(net, True, shield_kw=kw),
                "Sentinel": NetDefender(load_sentinel_defender(os.path.join(SM, f"best_benchmark_defender_seed{seed}.pt")), True)}
        for j, sc in enumerate(scen):
            for dn, d in defs.items():
                s = seed * 100 + 70 + j
                rows.append(dict(seed=seed, scenario=sc, defender=dn, operator="none", acc=np.nan, **run(d, S.EVAL[sc], s)))
                for acc in (1.0, 0.9, 0.7):
                    for qf in (True, False):
                        rows.append(dict(seed=seed, scenario=sc, defender=dn, acc=acc,
                                         operator="quality-first" if qf else "icmp-only",
                                         **run(d, S.EVAL[sc], s, operator=True, acc=acc, quality_first=qf)))
        print("seed", seed, "done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results/human", exist_ok=True)
    df.to_csv("results/human/human_in_loop_by_seed.csv", index=False)
    md = ["# Simulated human-in-the-loop discrete decisions (10 seeds, 20 episodes each)\n",
          "Operator reviews every 10 steps, acts after 3 steps, recognises the family with accuracy `acc`. "
          "`icmp-only`: ICMP playbook only; `quality-first`: also CONSERVATIVE on low-rate floods. Severe per 500 steps.\n"]
    for sc in scen:
        md.append(f"\n## {sc}\n\n| defender | operator | acc | composite | quality | leakage | severe | switches/ep |\n|---|---|---|---|---|---|---|---|")
        sub = df[df.scenario == sc]
        for (dn, op, acc), g in sub.groupby(["defender", "operator", "acc"], dropna=False):
            md.append(f"| {dn} | {op} | {acc} | {ci95(g.benchmark)[0]:.3f} ± {ci95(g.benchmark)[1]:.3f} | {g.quality.mean():.3f} | "
                      f"{g.leakage.mean():.3f} | {g.severe.mean():.1f} | {g.switches_per_ep.mean():.1f} |")
        # paired: SPAN+operator(icmp-only, acc .9) vs Sentinel without operator
        a = sub[(sub.defender == "SPAN") & (sub.operator == "icmp-only") & (sub.acc == 0.9)].sort_values("seed")
        b = sub[(sub.defender == "Sentinel") & (sub.operator == "none")].sort_values("seed")
        for m in ("benchmark", "severe", "quality", "leakage"):
            r = paired(a[m].values, b[m].values)
            md.append(f"\nSPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), {m}: diff {r['diff']:+.3f}, p={r['t_p']:.2g}")
    with open("results/human/HUMAN_IN_LOOP.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/human/HUMAN_IN_LOOP.md")


if __name__ == "__main__":
    main()
