"""Testbed diagnostics reported in the paper.

1. Reproduction: Sentinel's released checkpoints re-evaluated on the vectorised port
   (Protocol A) next to the numbers published in Sentinel's evaluation summary.
2. Headroom: a privileged per-step oracle (knows attack family, intensity and
   mutation; searches a 21x3x20 action grid each step) with and without the default
   shield, versus Sentinel, on the Protocol B core scenarios.
3. PanicGuard activity of Sentinel under mild attacks.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import env as E  # noqa: E402
from nscsd import scenarios as S  # noqa: E402
from nscsd.agents import NetDefender  # noqa: E402
from nscsd.policy import load_sentinel_attacker, load_sentinel_defender  # noqa: E402
from nscsd.rollout import benchmark_score, run_closed_loop, run_open_loop, summarize  # noqa: E402
from nscsd.shield import R, repair  # noqa: E402

torch.set_num_threads(1)
RUN = "/home/user/alialfatemi/sentinel-ddos/outputs/run_20260523_200520_paper"
M = os.path.join(RUN, "models")
GRID = np.array([(t, f, d) for t in np.linspace(0, 1, 21) for f in range(3) for d in np.linspace(0, 0.95, 20)])


def reproduction():
    rows = []
    for seed in range(42, 52):
        att = load_sentinel_attacker(os.path.join(M, f"best_benchmark_attacker_seed{seed}.pt"))
        net = load_sentinel_defender(os.path.join(M, f"best_benchmark_defender_seed{seed}.pt"))
        for j, sc in enumerate(["benign", "mild_attack", "strong_attack", "chaos_flash_crowd", "zero_shot_icmp"]):
            for name, d in (("Sentinel", NetDefender(net, True)), ("PPO_Ablation_NoShield", NetDefender(net, False))):
                r = summarize(run_closed_loop(d, att, sc, 5, np.random.default_rng(seed * 100 + j)), "leak")
                rows.append(dict(seed=seed, scenario=sc, defender_type=name, **r))
    ours = pd.DataFrame(rows).groupby(["scenario", "defender_type"])[["quality", "leakage", "severe"]].mean()
    pub = pd.read_csv(os.path.join(RUN, "analysis", "evaluation_benchmark_summary.csv"))
    pub = pub.set_index(["scenario", "defender_type"])[["service_quality_mean", "attack_leakage_mean",
                                                        "severe_outage_mean"]]
    pub.columns = ["quality_pub", "leakage_pub", "severe_pub"]
    return ours.join(pub, how="left").round(3)


def oracle(scen_vec, n_eps, seed, shielded, T=100):
    rng = np.random.default_rng(seed)
    (typ, inten, mut), tape = S.contexts(rng, np.tile(scen_vec, (n_eps, 1)), T)
    obs = E.initial_obs(tape.legit0)
    k = len(GRID)
    tot = dict(sq=0, leak=0, severe=0, degraded=0)
    for t in range(T):
        o = np.repeat(obs, k, 0)
        thr, foc, drp = np.tile(GRID[:, 0], n_eps), np.tile(GRID[:, 1], n_eps).astype(int), np.tile(GRID[:, 2], n_eps)
        if shielded:
            thr, foc, drp, _, _ = repair(o, thr, foc, drp)
        rep = np.repeat(np.arange(n_eps), k)
        sub = E.Tape(tape.legit[rep], tape.proto_noise[rep], tape.ent_noise[rep], tape.pkt_noise[rep],
                     tape.reroll[rep], tape.legit0[rep])
        _, info = E.step(o, t, sub, (typ[rep, t], inten[rep, t], mut[rep, t]), (thr, foc, drp),
                         np.ones(len(rep), bool))
        best = info["score"].reshape(n_eps, k).argmax(1) + np.arange(n_eps) * k
        a = (thr[best], foc[best], drp[best])
        obs, inf = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), a, np.ones(n_eps, bool))
        for key in tot:
            tot[key] = tot[key] + inf[key if key != "leak" else "leak"] * (1 if key != "leak" else 1)
    sq, leak = tot["sq"].mean() / T, tot["leak"].mean() / T
    sev, deg = tot["severe"].mean() / T, tot["degraded"].mean() / T
    return dict(quality=sq, leakage=leak, severe=500 * sev, benchmark=benchmark_score(sq, leak, sev, deg))


def headroom():
    rows = []
    for sc in ["mild", "boundary", "strong", "chaos", "icmp"]:
        for seed in range(42, 45):
            net = load_sentinel_defender(os.path.join(M, f"best_benchmark_defender_seed{seed}.pt"))
            out = run_open_loop(NetDefender(net, True), np.tile(S.EVAL[sc], (10, 1)), np.random.default_rng(seed), 100)
            s = summarize(out, "leak")
            pg = out["fired"][:, R["PanicGuard_Drop"]].mean() + 0
            rows.append(dict(scenario=sc, seed=seed, who="Sentinel", panicguard_drop_per_500=5 * pg,
                             **{k: s[k] for k in ("quality", "leakage", "severe", "benchmark")}))
            for sh in (False, True):
                o = oracle(S.EVAL[sc], 10, seed, sh)
                rows.append(dict(scenario=sc, seed=seed, who=f"oracle{'+shield' if sh else ''}", **o))
    return pd.DataFrame(rows).groupby(["scenario", "who"]).mean(numeric_only=True).drop(columns="seed").round(3)


def main():
    os.makedirs("results", exist_ok=True)
    rep = reproduction()
    hr = headroom()
    with open("results/DIAGNOSTICS.md", "w") as f:
        f.write("# Testbed diagnostics\n\n## 1. Reproduction of Sentinel's published results on the vectorised port\n\n")
        f.write("Protocol A, released benchmark checkpoints, 10 seeds; `_pub` = Sentinel's evaluation_benchmark_summary.csv.\n\n")
        f.write(rep.to_markdown() + "\n\n")
        f.write("## 2. Headroom: privileged per-step oracle vs Sentinel (Protocol B scenarios, seeds 42-44, 10 eps)\n\n")
        f.write("Leakage here is the all-step mean. The oracle knows the attack family, intensity and mutation.\n\n")
        f.write(hr.to_markdown() + "\n\n")
        f.write("Physics ceiling: with inline (pre-filter) overload the service quality of a full-intensity flood is at most "
                f"{1000/1800:.3f} (1000/(1500+300)), independent of the defender.\n")
    rep.to_csv("results/reproduction.csv")
    hr.to_csv("results/headroom.csv")
    print(rep)
    print(hr)


if __name__ == "__main__":
    main()
