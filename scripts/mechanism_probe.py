"""Mechanism test for observation self-masking.

Observations are computed from SERVED traffic, so a defender that filters well
removes the attack's signature from its own next observation.  We measure how
well the hidden attack family can be recovered from (a) o_t alone and
(b) o_t plus the previously deployed action a_{t-1}, under defenders that
filter weakly or strongly.  Identifiability is measured by a held-out probe
classifier (gradient-boosted trees) trained per defender.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd import env as E  # noqa: E402
from nscsd import scenarios as S  # noqa: E402
from nscsd.agents import NetDefender, StaticDefender  # noqa: E402
from nscsd.archive import ConstantDefender  # noqa: E402
from nscsd.policy import DefenderNet, load_sentinel_defender  # noqa: E402
from nscsd.rollout import deploy  # noqa: E402

torch.set_num_threads(1)
SM = "/home/user/alialfatemi/sentinel-ddos/outputs/run_20260523_200520_paper/models"


def collect(defender, scen, rng, n=200, T=100):
    (typ, inten, mut), tape = S.contexts(rng, np.tile(scen, (n, 1)), T)
    obs = E.initial_obs(tape.legit0)
    prev = np.zeros((n, 3))
    rows_o, rows_p, fam, served = [], [], [], []
    defender.reset(n)
    for t in range(T):
        # the label is the family active at the step that produced this observation (t-1)
        if t > 0:
            rows_o.append(obs.copy()); rows_p.append(prev.copy()); fam.append(typ[:, t - 1])
        a, _, _ = deploy(defender, obs)
        obs, info = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), a, np.ones(n, bool))
        prev = np.stack([a[0], a[1], a[2]], 1)
        served.append(info["leak"])
    return np.concatenate(rows_o), np.concatenate(rows_p), np.concatenate(fam), float(np.mean(served))


def probe(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def main():
    scen = {"strong": S.EVAL["strong"], "polymorph": S.EVAL["polymorph"], "mild": S.EVAL["mild"]}
    net = DefenderNet(); net.load_state_dict(torch.load("runs/inline/nscsd/seed42/champion.pt", weights_only=True))
    kw = json.load(open("runs/inline/nscsd/seed42/meta.json"))["champion_shield"]
    defenders = {
        "no-mitigation": ConstantDefender(1.0, 1, 0.0, shielded=False),
        "Static": StaticDefender(),
        "Sentinel": NetDefender(load_sentinel_defender(os.path.join(SM, "best_benchmark_defender_seed42.pt")), True),
        "NS-CSD": NetDefender(net, True, shield_kw=kw),
        "max-mitigation (proto, d=.95)": ConstantDefender(0.3, 1, 0.95, shielded=False),
    }
    rows = []
    for sc, vec in scen.items():
        for name, d in defenders.items():
            O, P, y, leak = collect(d, vec, np.random.default_rng(7))
            rows.append(dict(scenario=sc, defender=name, mean_leakage=leak,
                             acc_obs=probe(O, y), acc_obs_prevaction=probe(np.hstack([O, P]), y),
                             chance=float(np.bincount(y).max() / len(y))))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).round(3)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/mechanism_probe.csv", index=False)
    with open("results/MECHANISM.md", "w") as f:
        f.write("# Self-masking mechanism probe\n\nHeld-out accuracy of a gradient-boosted probe recovering the "
                "attack family that generated o_t, from o_t alone or o_t plus the previous deployed action.\n\n")
        f.write(df.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
