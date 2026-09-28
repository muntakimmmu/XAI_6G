"""Rule crystallisation: distil a (neural + shield) defender into depth-bounded trees."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from . import scenarios as S
from .agents import TreeDefender
from .env import FEATURES
from .rollout import run_open_loop


def collect(defender, rng, n_random=24, eps_per_eval=4, T=100, physics="inline"):
    """(obs, deployed action) pairs over the validation battery plus random training scenarios."""
    scen = [np.tile(v, (eps_per_eval, 1)) for v in S.VALIDATION.values()]
    scen.append(S.random_scenarios(rng, n_random))
    out = run_open_loop(defender, np.concatenate(scen), rng, T, physics, icmp_allowed=False, return_obs=True)
    return out["obs"], out["act"]


def fit(X, A, max_depth=4, seed=0):
    Xtr, Xte, Atr, Ate = train_test_split(X, A, test_size=0.3, random_state=seed)
    tf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(Xtr, Atr[:, 1].astype(int))
    tt = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(Xtr, Atr[:, 0])
    td = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(Xtr, Atr[:, 2])
    fidelity = dict(
        focus_acc=accuracy_score(Ate[:, 1].astype(int), tf.predict(Xte)),
        thr_mae=mean_absolute_error(Ate[:, 0], tt.predict(Xte)),
        drop_mae=mean_absolute_error(Ate[:, 2], td.predict(Xte)),
        leaves=int(tf.get_n_leaves() + tt.get_n_leaves() + td.get_n_leaves()),
    )
    return TreeDefender(tf, tt, td, shielded=True), fidelity


def crystallize(defender, rng, max_depth=4, physics="inline"):
    X, A = collect(defender, rng, physics=physics)
    tree, fid = fit(X, A, max_depth=max_depth, seed=int(rng.integers(1 << 30)))
    tree.shield_kw = getattr(defender, "shield_kw", None)
    return tree, fid


def describe(tree: TreeDefender) -> str:
    parts = []
    for name, t in (("focus", tree.tf), ("threshold", tree.tt), ("drop", tree.td)):
        parts.append(f"## {name}\n" + export_text(t, feature_names=FEATURES, decimals=3))
    return "\n".join(parts)
