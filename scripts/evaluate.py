"""Evaluate all defenders under three protocols and write per-seed CSVs.

Protocol A  Sentinel-native: Sentinel's own trained attacker (per seed) drives the
            five paper scenarios, exactly as in its ``evaluate_system.py``
            (5 episodes x 100 steps; leakage averaged over all steps).
Protocol B  Attacker-agnostic battery: the five core scenarios plus four held-out
            stress tests (20 episodes x 100 steps; leakage over attacked steps).
Protocol C  Adaptive red team: cross-entropy search over the full scenario space
            (ICMP included) for the attack that minimises each defender's
            composite score.

All defenders see identical traffic for a given (seed, scenario) (common random numbers).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import scenarios as S  # noqa: E402
from nscsd.agents import AdaptiveDefender, NetDefender, RandomDefender, StaticDefender  # noqa: E402
from nscsd.policy import DefenderNet, load_sentinel_attacker, load_sentinel_defender  # noqa: E402
from nscsd.rollout import benchmark_score, run_closed_loop, run_open_loop, summarize  # noqa: E402
from nscsd.paths import SENTINEL_MODELS  # noqa: E402
from nscsd.shield import RULES  # noqa: E402

torch.set_num_threads(1)
SEEDS = list(range(42, 52))
A_SCEN = ["benign", "mild_attack", "strong_attack", "chaos_flash_crowd", "zero_shot_icmp"]
OURS = ["nscsd", "no_shield_discovery", "no_discovery", "no_elite", "no_anchor", "no_crn", "no_shield",
        "grpo_vanilla", "nscsd_safe", "ppo", "span", "span_nosafe", "span_cgrpo", "maxmc_only",
        "span_archive", "span_entropy", "span_abscost", "span_cgrpo_safe"]


def load_net(path):
    net = DefenderNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def defenders(seed, runs, sentinel_models, physics):
    d = {}
    if sentinel_models:
        sb = load_sentinel_defender(os.path.join(sentinel_models, f"best_benchmark_defender_seed{seed}.pt"))
        sf = load_sentinel_defender(os.path.join(sentinel_models, f"final_defender_seed{seed}.pt"))
        d["Sentinel"] = NetDefender(sb, True)
        d["Sentinel-final"] = NetDefender(sf, True)
        d["PPO-NoShield (Sentinel)"] = NetDefender(sb, False)
    d["ShieldOnly"] = RandomDefender(np.random.default_rng(seed), shielded=True)
    d["Static"] = StaticDefender()
    d["Adaptive"] = AdaptiveDefender()
    d["Random"] = RandomDefender(np.random.default_rng(seed + 1))
    for m in OURS:
        p = os.path.join(runs, physics, m, f"seed{seed}")
        if not os.path.exists(os.path.join(p, "meta.json")):
            continue
        shielded = m != "no_shield"
        meta = json.load(open(os.path.join(p, "meta.json")))
        d[f"{m}"] = NetDefender(load_net(os.path.join(p, "champion.pt")), shielded,
                                shield_kw=meta.get("champion_shield"))
        d[f"{m}-final"] = NetDefender(load_net(os.path.join(p, "final.pt")), shielded,
                                      shield_kw=meta.get("final_shield"))
        tp = os.path.join(p, "tree.pkl")
        if m == "nscsd" and os.path.exists(tp):
            with open(tp, "rb") as f:
                d["nscsd-tree"] = pickle.load(f)
    return d


def cem_redteam(defender, seed, physics, pop=48, iters=8, elite=0.2, eps=4, T=60):
    """Cross-entropy search for the scenario that minimises the defender's composite score."""
    rng = np.random.default_rng(777 + seed)
    dim = 6 + 1 + 2 + 2 + 3 + 1
    mu, sd = np.zeros(dim), np.full(dim, 1.5)
    sig = lambda x: 1 / (1 + np.exp(-x))

    def decode(z):
        s = np.zeros((len(z), S.DIM))
        w = np.exp(z[:, :6] - z[:, :6].max(1, keepdims=True))
        s[:, S.W] = w / w.sum(1, keepdims=True)
        s[:, S.STAY] = 0.3 + 0.7 * sig(z[:, 6])
        a, b = sig(z[:, 7]), sig(z[:, 8])
        s[:, S.ILO], s[:, S.IHI] = np.minimum(a, b), np.maximum(a, b)
        a, b = sig(z[:, 9]), sig(z[:, 10])
        s[:, S.MLO], s[:, S.MHI] = np.minimum(a, b), np.maximum(a, b)
        s[:, S.PER] = np.where(z[:, 11] > 0, np.round(4 + 16 * sig(z[:, 12])), 0)
        s[:, S.DUTY] = np.where(s[:, S.PER] > 0, 0.2 + 0.6 * sig(z[:, 13]), 1.0)
        s[:, S.FLASH] = np.where(z[:, 14] > 0, 0.05, 0.0)
        return s

    best, best_s = np.inf, None
    for i in range(iters):
        z = mu + sd * rng.standard_normal((pop, dim))
        scen = decode(z)
        out = run_open_loop(defender, np.repeat(scen, eps, 0), np.random.default_rng(1000 * seed + i), T, physics)
        b = benchmark_score(out["sq"], out["leak_att"], out["severe"], out["degraded"]).reshape(pop, eps).mean(1)
        order = np.argsort(b)
        k = max(2, int(elite * pop))
        mu, sd = z[order[:k]].mean(0), z[order[:k]].std(0) + 0.05
        if b[order[0]] < best:
            best, best_s = b[order[0]], scen[order[0]]
    # re-score the worst scenario found on fresh traffic
    out = run_open_loop(defender, np.tile(best_s, (20, 1)), np.random.default_rng(99 + seed), 100, physics)
    r = summarize(out)
    fam = ["none", "syn", "udp", "http", "icmp", "mixed"][int(best_s[S.W].argmax())]
    return dict(worst_benchmark=r["benchmark"], worst_quality=r["quality"], worst_leakage=r["leakage"],
                worst_severe=r["severe"], worst_family=fam, worst_intensity=0.5 * (best_s[S.ILO] + best_s[S.IHI]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--physics", default="inline", help="simulator physics used for evaluation")
    ap.add_argument("--train_physics", default="inline", help="which trained runs to load")
    ap.add_argument("--sentinel_models",
                    default=SENTINEL_MODELS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--protocols", default="ABC")
    ap.add_argument("--eps_b", type=int, default=20)
    ap.add_argument("--methods", nargs="*", default=None, help="restrict our trained variants (default: all)")
    args = ap.parse_args()
    if args.methods is not None:
        OURS[:] = args.methods
    if not os.path.isdir(args.sentinel_models):
        args.sentinel_models = None
    os.makedirs(args.out, exist_ok=True)
    rows = {"A": [], "B": [], "C": []}
    rules = []
    for seed in args.seeds:
        defs = defenders(seed, args.runs, args.sentinel_models, args.train_physics)
        if "A" in args.protocols and args.sentinel_models and args.physics == "inline":
            att = load_sentinel_attacker(os.path.join(args.sentinel_models, f"best_benchmark_attacker_seed{seed}.pt"))
            for j, sc in enumerate(A_SCEN):
                for name, d in defs.items():
                    out = run_closed_loop(d, att, sc, 5, np.random.default_rng(seed * 100 + j), physics=args.physics)
                    rows["A"].append(dict(seed=seed, scenario=sc, defender=name, **summarize(out, "leak")))
        if "B" in args.protocols:
            for j, sc in enumerate(S.CORE + S.STRESS):
                scen = np.tile(S.EVAL[sc], (args.eps_b, 1))
                for name, d in defs.items():
                    out = run_open_loop(d, scen, np.random.default_rng(seed * 100 + 50 + j), 100, args.physics)
                    rows["B"].append(dict(seed=seed, scenario=sc, defender=name, **summarize(out)))
                    if name in ("nscsd", "Sentinel", "nscsd-tree"):
                        f = out["fired"].sum(0) / args.eps_b * 5  # per 500 steps
                        rules.append(dict(seed=seed, scenario=sc, defender=name, **dict(zip(RULES, f))))
        if "C" in args.protocols:
            for name, d in defs.items():
                if name.endswith("-final") and name != "Sentinel-final":
                    continue
                rows["C"].append(dict(seed=seed, defender=name, **cem_redteam(d, seed, args.physics)))
        print(f"seed {seed} done", flush=True)
    tag = "" if args.physics == "inline" else f"_{args.physics}"
    for p, r in rows.items():
        if r:
            pd.DataFrame(r).to_csv(os.path.join(args.out, f"protocol{p}{tag}_by_seed.csv"), index=False)
    if rules:
        pd.DataFrame(rules).to_csv(os.path.join(args.out, f"shield_rules{tag}_by_seed.csv"), index=False)


if __name__ == "__main__":
    main()
