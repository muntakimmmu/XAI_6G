"""Empirical multi-step check of the safe-side direction (Proposition 2).

Prop. 2 holds window by window: for the same observation and traffic, the safe-side action
(d * u, tau + (1 - u)(1 - tau)) never yields more collateral damage after shielding.  Over a
rollout the observations of the two copies can drift apart, because the served attack share
enters the telemetry.  This script measures whether whole episodes keep the ordering: each
trained IMBANG policy and its safe-shifted copy (u ~ U(0.2, 0.8) per episode) face identical
traffic, with the Protocol B scenarios and random-number seeds of scripts/evaluate.py.
    python scripts/safe_member_check.py     # -> results/safe_member_check_by_seed.csv
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
from nscsd.policy import DefenderNet  # noqa: E402
from nscsd.rollout import run_open_loop  # noqa: E402

torch.set_num_threads(1)
EPS, T = 20, 100


class SafeShift:
    """Applies the safe-side transformation to a defender's raw action before the shield."""

    def __init__(self, base, u):
        self.base, self.u = base, u
        self.shielded, self.shield_kw = base.shielded, base.shield_kw

    def reset(self, n):
        self.base.reset(n)

    def raw(self, obs):
        thr, focus, drop = (np.asarray(x) for x in self.base.raw(obs))
        return thr + (1 - thr) * (1 - self.u), focus, drop * self.u


def main():
    rows = []
    for seed in range(42, 52):
        p = f"runs/inline/span/seed{seed}"
        net = DefenderNet()
        net.load_state_dict(torch.load(f"{p}/champion.pt", map_location="cpu", weights_only=True))
        net.eval()
        base = NetDefender(net, True, shield_kw=json.load(open(f"{p}/meta.json")).get("champion_shield"))
        u = np.random.default_rng(900 + seed).uniform(0.2, 0.8, EPS)
        for j, sc in enumerate(S.CORE + S.STRESS):
            scen = np.tile(S.EVAL[sc], (EPS, 1))
            res = {name: run_open_loop(d, scen, np.random.default_rng(seed * 100 + 50 + j), T)
                   for name, d in (("policy", base), ("safe", SafeShift(base, u)))}
            ks = {k: res[k]["severe"] * T for k in res}            # severe windows per episode
            att = {k: np.maximum(res[k]["attacked"], 1e-9) for k in res}
            rows.append(dict(seed=seed, scenario=sc,
                             severe_policy=500 * res["policy"]["severe"].mean(),
                             severe_safe=500 * res["safe"]["severe"].mean(),
                             cd_policy=res["policy"]["cd"].mean(), cd_safe=res["safe"]["cd"].mean(),
                             leak_policy=np.mean(res["policy"]["leak_att"]), leak_safe=np.mean(res["safe"]["leak_att"]),
                             ep_more=float(np.mean(ks["safe"] > ks["policy"])),
                             ep_fewer=float(np.mean(ks["safe"] < ks["policy"])),
                             win_cd_higher=float(np.mean(res["safe"]["cd"] > res["policy"]["cd"] + 1e-12))))
        print("seed", seed, "done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/safe_member_check_by_seed.csv", index=False)
    print(df.groupby("scenario")[["severe_policy", "severe_safe", "cd_policy", "cd_safe", "leak_policy",
                                  "leak_safe", "ep_more", "ep_fewer"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
