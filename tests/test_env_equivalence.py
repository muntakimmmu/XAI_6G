"""Check the vectorised port against Sentinel's reference simulator and shield.

The reference code is looked up at $SENTINEL_REPO (default: a sibling clone of
github.com/AliAlfatemi/sentinel-ddos); the test is skipped when it is absent.
"""
import copy
import os
import sys

import numpy as np
import pytest

from nscsd import env as E
from nscsd.shield import repair

REPO = os.environ.get("SENTINEL_REPO", "/home/user/alialfatemi/sentinel-ddos")
pytestmark = pytest.mark.skipif(not os.path.isdir(REPO), reason="Sentinel reference repo not available")


class _TapeRNG:
    """Stands in for the reference env's np_random and replays one Tape row."""

    def __init__(self, tape, t):
        self.tape, self.t = tape, t

    def poisson(self, lam):
        return self.tape.legit[0, self.t]

    def normal(self, mu, sd, size=None):
        if size == 4:
            return self.tape.proto_noise[0, self.t]
        if size == 3:
            return self.tape.ent_noise[0, self.t]
        return self.tape.pkt_noise[0, self.t]

    def choice(self, options):
        return self.tape.reroll[0, self.t]

    def random(self):
        return 1.0


def _reference():
    sys.path.insert(0, REPO)
    from simulation.env import DDoSEnv
    from agents.shield import SymbolicShield
    return DDoSEnv, SymbolicShield


def test_step_matches_reference():
    DDoSEnv, _ = _reference()
    rng = np.random.default_rng(0)
    ref = DDoSEnv(seed=0, shield_penalty_weight=0.0)
    ref.reset(seed=0)
    T = 400
    tape = E.Tape.sample(rng, 1, T)
    obs = E.initial_obs(tape.legit0)
    for t in range(T):
        a = (np.array([rng.integers(0, 6)]), rng.random(1), rng.random(1))
        d = (rng.random(1), np.array([rng.integers(0, 3)]), rng.random(1))
        icmp_ok = np.array([bool(rng.integers(0, 2))])
        ref.zero_shot_eval, ref.chaos_mode = bool(icmp_ok[0]), False
        ref.np_random = _TapeRNG(tape, t)
        att = {"type": int(a[0][0]), "intensity": a[1].astype(np.float32), "mutation": a[2].astype(np.float32)}
        dfn = {"threshold": d[0].astype(np.float32), "focus": int(d[1][0]), "drop_prob": d[2].astype(np.float32)}
        r_obs, _, r_rew, _, _, info = ref.step(att, dfn)
        # compare in float32 precision, as the reference casts actions to float32
        a32 = (a[0], a[1].astype(np.float32).astype(float), a[2].astype(np.float32).astype(float))
        d32 = (d[0].astype(np.float32).astype(float), d[1], d[2].astype(np.float32).astype(float))
        nxt, mine = E.step(obs, t, tape, a32, d32, icmp_ok)
        assert np.isclose(mine["sq"][0], info["service_quality"], atol=1e-5)
        assert np.isclose(mine["leak"][0], info["attack_leakage"], atol=1e-5)
        assert np.isclose(mine["reward"][0], r_rew, atol=1e-4)
        assert mine["severe"][0] == info["severe_outage"]
        np.testing.assert_allclose(nxt[0], r_obs, atol=1e-5)
        obs = nxt


def test_shield_matches_reference():
    _, SymbolicShield = _reference()
    rng = np.random.default_rng(1)
    shield = SymbolicShield()
    n = 5000
    obs = rng.random((n, 12))
    # bias half the samples into the regions where rules fire
    obs[: n // 2, 2] = rng.uniform(0.55, 1.0, n // 2)
    obs[: n // 4, 5] = rng.uniform(0.2, 1.0, n // 4)
    thr, focus, drop = rng.random(n), rng.integers(0, 3, n), rng.random(n)
    s_thr, s_focus, s_drop, rep, _ = repair(obs, thr, focus, drop)
    for i in range(n):
        raw = {"threshold": np.array([thr[i]], np.float32), "focus": int(focus[i]),
               "drop_prob": np.array([drop[i]], np.float32)}
        safe, _ = shield.repair(obs[i], copy.deepcopy(raw))
        assert safe["focus"] == s_focus[i]
        assert np.isclose(safe["threshold"][0], s_thr[i], atol=1e-6)
        assert np.isclose(safe["drop_prob"][0], s_drop[i], atol=1e-6)
