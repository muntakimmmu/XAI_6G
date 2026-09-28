"""Defender wrappers with a common batched interface.

Every defender exposes ``reset(n)`` and ``raw(obs) -> (threshold, focus, drop)``
(the action *proposed* before the shield) plus a ``shielded`` flag.  The
rollout engine applies the symbolic shield when ``shielded`` is True.
"""
from __future__ import annotations

import numpy as np

from .policy import DefenderNet


class NetDefender:
    def __init__(self, net: DefenderNet, shielded=True, deterministic=True, name="net", shield_kw=None):
        self.net, self.shielded, self.deterministic, self.name = net, shielded, deterministic, name
        self.shield_kw = shield_kw

    def reset(self, n):
        pass

    def raw(self, obs):
        return self.net.act(obs, deterministic=self.deterministic)


class RandomDefender:
    def __init__(self, rng: np.random.Generator, shielded=False, name="Random"):
        self.rng, self.shielded, self.name = rng, shielded, name

    def reset(self, n):
        pass

    def raw(self, obs):
        n = obs.shape[0]
        return self.rng.random(n), self.rng.integers(0, 3, n), self.rng.random(n)


class StaticDefender:
    shielded = False
    name = "Static"

    def reset(self, n):
        pass

    def raw(self, obs):
        n = obs.shape[0]
        return np.full(n, 0.8), np.ones(n, dtype=int), np.full(n, 0.5)


class AdaptiveDefender:
    """Sentinel's traffic-responsive heuristic (10-step moving average)."""

    shielded = False
    name = "Adaptive"

    def reset(self, n):
        self.hist = [[] for _ in range(n)]

    def raw(self, obs):
        n = obs.shape[0]
        thr, drop = np.empty(n), np.empty(n)
        for i in range(n):
            h = self.hist[i]
            h.append(obs[i, 0])
            if len(h) > 10:
                h.pop(0)
            avg = sum(h) / len(h)
            if obs[i, 0] > avg * 1.5:
                thr[i], drop[i] = min(1.0, avg), 0.8
            else:
                thr[i], drop[i] = 0.9, 0.1
        return thr, np.ones(n, dtype=int), drop


class TreeDefender:
    """Crystallised symbolic program: three depth-bounded decision trees."""

    def __init__(self, tree_focus, tree_thr, tree_drop, shielded=True, name="Tree", shield_kw=None):
        self.tf, self.tt, self.td = tree_focus, tree_thr, tree_drop
        self.shielded, self.name, self.shield_kw = shielded, name, shield_kw

    def reset(self, n):
        pass

    def raw(self, obs):
        return (np.clip(self.tt.predict(obs), 1e-4, 1 - 1e-4), self.tf.predict(obs).astype(int),
                np.clip(self.td.predict(obs), 1e-4, 1 - 1e-4))
