"""Mechanism test: does group-relative normalisation hide the severe-outage cliff?

For a trained policy we fork CRN groups of G stochastic members from overload
states (full-intensity floods) and measure
  * span rate: fraction of groups whose members differ in severe-outage count,
  * the policy's sampled drop-rate spread (std) in those states,
  * the severe-outage rate.
If every member of a group suffers the same number of severe steps, the severe
penalty is identical across the group and vanishes from A = (R - mean)/std.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import env as E  # noqa: E402
from nscsd import scenarios as S  # noqa: E402
from nscsd.policy import DefenderNet, load_sentinel_defender  # noqa: E402
from nscsd.shield import repair  # noqa: E402

torch.set_num_threads(1)
from nscsd.paths import SENTINEL_MODELS as SM  # noqa: E402


def probe(net, kw, seed, scen="strong", B=64, G=8, H=8, W=20):
    rng = np.random.default_rng(seed)
    (typ, inten, mut), tape = S.contexts(rng, np.tile(S.EVAL[scen], (B, 1)), W + H)
    obs = E.initial_obs(tape.legit0)
    for t in range(W):
        thr, f, d = net.act(obs, deterministic=True)
        a = repair(obs, thr, f, d, **(kw or {}))[:3]
        obs, _ = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), a, np.ones(B, bool))
    idx = np.repeat(np.arange(B), G)
    tg, og = tape.take(idx), obs[idx]
    sev = np.zeros(B * G)
    drops = []
    for h in range(H):
        t = W + h
        thr, f, d = net.act(og, deterministic=False)
        drops.append(d.reshape(B, G))
        a = repair(og, thr, f, d, **(kw or {}))[:3]
        og, info = E.step(og, t, tg, (typ[idx, t], inten[idx, t], mut[idx, t]), a, np.ones(B * G, bool))
        sev += info["severe"]
    sev = sev.reshape(B, G)
    span_rate = float((sev.max(1) != sev.min(1)).mean())
    return dict(span_rate=span_rate, drop_std_in_group=float(np.mean([x.std(1).mean() for x in drops])),
                severe_rate=float(sev.mean() / H))


def main():
    rows = []
    full = os.environ.get("FULL") == "1"
    tag = "" if len(sys.argv) == 1 else ("_final" if full else "_pilot")
    seeds = range(42, 45) if tag == "_pilot" else range(42, 52)
    methods = sys.argv[1:] or ["nscsd", "grpo_vanilla", "ppo", "nscsd_safe"]
    for m in methods:
        for p in sorted(glob.glob(f"runs/inline/{m}/seed*/champion.pt")):
            seed = int(p.split("/")[-2][4:])
            if seed not in seeds:
                continue
            net = DefenderNet(); net.load_state_dict(torch.load(p, weights_only=True))
            meta = json.load(open(p.replace("champion.pt", "meta.json")))
            for sc in ("strong", "polymorph"):
                rows.append(dict(method=m, seed=seed, scenario=sc, **probe(net, meta.get("champion_shield"), seed, sc)))
    for seed in seeds:
        net = load_sentinel_defender(os.path.join(SM, f"best_benchmark_defender_seed{seed}.pt"))
        for sc in ("strong", "polymorph"):
            rows.append(dict(method="Sentinel", seed=seed, scenario=sc, **probe(net, None, seed, sc)))
    df = pd.DataFrame(rows)
    df.to_csv(f"results/cliff_probe{tag}_by_seed.csv", index=False)
    g = df.groupby(["scenario", "method"])[["span_rate", "drop_std_in_group", "severe_rate"]].agg(["mean", "std"]).round(3)
    print(g)
    with open(f"results/CLIFF_PROBE{tag.upper()}.md", "w") as f:
        f.write("# Cliff-visibility probe (overload states, G=8 CRN members, H=8)\n\n" + g.to_markdown() + "\n")


if __name__ == "__main__":
    main()
