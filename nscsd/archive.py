"""Continuous solution discovery: a quality-diversity Solution Archive and a
regret-prioritised Scenario Curriculum.

* The **Solution Archive** stores, for every scenario niche, the best solution
  found so far.  Solutions are heterogeneous: neural snapshots (+shield),
  crystallised decision-tree programs (+shield) and seed symbolic programs.
  An elite is only ever replaced by a candidate that scores strictly higher on
  the same scenarios with the same common random numbers, so every niche's
  score is monotonically non-decreasing on its evaluation set.
* The **Scenario Curriculum** replaces Sentinel's co-evolving RL attacker.  It
  proposes scenarios by mutation (ACCEL-style) or from the prior and ranks them
  by *regret against the archive*: how far the live policy is below the best
  solution the archive already knows for that scenario.  Unlike an adversary
  that maximises the defender's loss, this signal is zero on scenarios that
  nobody can solve better, so the curriculum cannot run away into unwinnable
  attacks -- the failure mode behind late-stage co-evolutionary collapse.
"""
from __future__ import annotations

import numpy as np

from . import scenarios as S
from .rollout import benchmark_score, run_open_loop


class ConstantDefender:
    """Seed symbolic program: a fixed (threshold, focus, drop) rule behind the shield."""

    def __init__(self, thr, focus, drop, shielded=True):
        self.a = (thr, focus, drop)
        self.shielded = shielded
        self.name = f"Const(t={thr},f={focus},d={drop})"

    def reset(self, n):
        pass

    def raw(self, obs):
        n = obs.shape[0]
        return np.full(n, self.a[0]), np.full(n, self.a[1], dtype=int), np.full(n, self.a[2])


def seed_programs(shielded=True):
    return [ConstantDefender(t, f, d, shielded) for t in (0.3, 0.6, 0.9) for f in (0, 1, 2)
            for d in (0.3, 0.9)]


def scenario_scores(defender, scen, seed, eps=2, T=50, physics="inline", sev_w=0.5):
    """Composite score of ``defender`` on each scenario row (fixed CRN per call seed)."""
    n = scen.shape[0]
    rng = np.random.default_rng(seed)
    out = run_open_loop(defender, np.repeat(scen, eps, axis=0), rng, T, physics, icmp_allowed=False)
    b = benchmark_score(out["sq"], out["leak_att"], out["severe"], out["degraded"], sev_w)
    return b.reshape(n, eps).mean(1)


class SolutionArchive:
    def __init__(self, n_niches=18):
        self.elites = [None] * n_niches  # (defender, kind, score, born_iter)
        self.inserts = 0

    def elite(self, niche):
        e = self.elites[niche]
        return None if e is None else e[0]

    def consider(self, defender, kind, scen, scores_by_row, it, current_by_row=None):
        """Offer ``defender`` to every niche present in ``scen``.

        ``scores_by_row`` are the candidate's per-scenario scores;
        ``current_by_row`` the current elites' scores on the same rows (same CRN).
        """
        niches = S.descriptor(scen)
        changed = 0
        for k in np.unique(niches):
            rows = niches == k
            cand = scores_by_row[rows].mean()
            cur = -np.inf if self.elites[k] is None else current_by_row[rows].mean()
            if cand > cur + 1e-9:
                self.elites[k] = (defender, kind, cand, it)
                changed += 1
        self.inserts += changed
        return changed

    def composition(self):
        kinds = [e[1] for e in self.elites if e is not None]
        return {k: kinds.count(k) for k in set(kinds)}


class ScenarioCurriculum:
    def __init__(self, rng, size=64, temperature=0.3, staleness=0.3):
        self.rng = rng
        self.size, self.temperature, self.staleness = size, temperature, staleness
        self.scen = S.random_scenarios(rng, size)
        self.regret = np.zeros(size)
        self.last_seen = np.zeros(size)
        self.clock = 0

    def probs(self):
        order = np.argsort(np.argsort(-self.regret))
        h = 1.0 / (order + 1) ** (1.0 / self.temperature)
        h /= h.sum()
        stale = self.clock - self.last_seen
        stale = stale / stale.sum() if stale.sum() > 0 else np.full(self.size, 1.0 / self.size)
        return (1 - self.staleness) * h + self.staleness * stale

    def sample(self, n, replay_p):
        replay = self.rng.random(n) < replay_p
        idx = self.rng.choice(self.size, size=n, p=self.probs())
        out = np.where(replay[:, None], self.scen[idx], S.random_scenarios(self.rng, n))
        self.clock += 1
        self.last_seen[idx[replay]] = self.clock
        return out

    def update(self, new_scen, new_regret, all_regret_refreshed=None):
        if all_regret_refreshed is not None:
            self.regret = all_regret_refreshed
        scen = np.concatenate([self.scen, new_scen])
        reg = np.concatenate([self.regret, new_regret])
        seen = np.concatenate([self.last_seen, np.full(len(new_scen), self.clock)])
        keep = np.argsort(-reg, kind="stable")[: self.size]
        self.scen, self.regret, self.last_seen = scen[keep], reg[keep], seen[keep]

    def propose(self, n):
        k = n // 2
        parents = self.scen[self.rng.choice(self.size, size=k, p=self.probs())]
        return np.concatenate([S.mutate(self.rng, parents), S.random_scenarios(self.rng, n - k)])


def discrete_levels():
    """Discrete level grid for minimax-regret training (ICMP held out):
    dominant family x intensity band x mutation band x pulsing x flash crowds = 5*6*3*2*2 = 360 levels."""
    fams = {"syn": [0, 1, 0, 0, 0, 0], "udp": [0, 0, 1, 0, 0, 0], "http": [0, 0, 0, 1, 0, 0],
            "mixed": [0, 0, 0, 0, 0, 1], "switching": [1, 1, 1, 1, 0, 1]}
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.0)]
    muts = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    out, names = [], []
    for fn, w in fams.items():
        for ib in bands:
            for mb in muts:
                for pulse in (0, 1):
                    for flash in (0.0, 0.05):
                        out.append(S.make(w, stay=0.6 if fn == "switching" else 0.95, i=ib, m=mb,
                                          period=10 if pulse else 0, duty=0.5 if pulse else 1.0, flash=flash))
                        names.append(f"{fn}|i{ib[0]:.1f}-{ib[1]:.1f}|m{mb[0]:.1f}|p{pulse}|f{flash}")
    return np.array(out), names


class DiscreteMaxMCCurriculum:
    """Minimax-regret curriculum over a fixed discrete level set, scored with the MaxMC
    regret estimator of Jiang et al. (2021): reg(l) = max_{k<=now} J_k(l) - J_now(l), the best
    score ever achieved on level l by any past policy snapshot minus the current policy's score.
    Levels are replayed with rank-prioritised regret mixed with staleness (PLR sampling)."""

    def __init__(self, rng, temperature=0.3, staleness=0.3):
        self.rng = rng
        self.scen, self.names = discrete_levels()
        self.size = len(self.scen)
        self.temperature, self.staleness = temperature, staleness
        self.best = np.full(self.size, -np.inf)
        self.regret = np.zeros(self.size)
        self.last_seen = np.zeros(self.size)
        self.clock = 0

    probs = ScenarioCurriculum.probs

    def sample(self, n, replay_p):
        idx = self.rng.choice(self.size, size=n, p=self.probs())
        uniform = self.rng.integers(0, self.size, n)
        replay = self.rng.random(n) < replay_p
        idx = np.where(replay, idx, uniform)
        self.clock += 1
        self.last_seen[idx] = self.clock
        return self.scen[idx]

    def refresh(self, defender, seed, physics="inline", sev_w=0.5):
        """Score the current policy on every level (fixed CRN per level) and update MaxMC regret."""
        cur = scenario_scores(defender, self.scen, seed, eps=2, T=50, physics=physics, sev_w=sev_w)
        self.best = np.maximum(self.best, cur)
        self.regret = self.best - cur
        return cur

    def subset(self, n):
        return self.scen[self.rng.choice(self.size, size=min(n, self.size), replace=False)]
