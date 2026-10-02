"""Operating curve: severe outages and leakage as a function of flood intensity.

For each seed, Sentinel's released checkpoint, Constrained GRPO and IMBANG face the same
traffic (common random numbers) at fixed intensities from 0.1 to 1.0 (finer above 0.85, where outages begin), with the
attack mix of the training scenarios (ICMP excluded) and the default mutation range.
    python scripts/intensity_sweep.py   # -> results/intensity_sweep_by_seed.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import scenarios as S  # noqa: E402
from nscsd.agents import NetDefender  # noqa: E402
from nscsd.policy import DefenderNet, load_sentinel_defender  # noqa: E402
from nscsd.rollout import run_open_loop, summarize  # noqa: E402

torch.set_num_threads(1)
SM = "/home/user/alialfatemi/sentinel-ddos/outputs/run_20260523_200520_paper/models"
GRID = np.round(np.r_[np.arange(0.1, 0.85, 0.1), np.arange(0.86, 1.001, 0.02)], 2)
EPS = 20


def ours(m, seed):
    p = f"runs/inline/{m}/seed{seed}"
    net = DefenderNet()
    net.load_state_dict(torch.load(f"{p}/champion.pt", map_location="cpu", weights_only=True))
    net.eval()
    return NetDefender(net, True, shield_kw=json.load(open(f"{p}/meta.json")).get("champion_shield"))


def main():
    rows = []
    for seed in range(42, 52):
        defs = {"Sentinel": NetDefender(load_sentinel_defender(f"{SM}/best_benchmark_defender_seed{seed}.pt"), True),
                "span_cgrpo": ours("span_cgrpo", seed), "span": ours("span", seed)}
        for k, x in enumerate(GRID):
            scen = np.tile(S.make(S.ATTACKS, i=(x, x)), (EPS, 1))
            for name, d in defs.items():
                out = run_open_loop(d, scen, np.random.default_rng(10_000 * seed + k))
                rows.append(dict(defender=name, seed=seed, intensity=x, **summarize(out)))
        print("seed", seed, "done", flush=True)
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/intensity_sweep_by_seed.csv", index=False)
    print(df.groupby(["defender", "intensity"])[["severe", "leakage", "quality"]].mean().round(3).unstack(0))


if __name__ == "__main__":
    main()
