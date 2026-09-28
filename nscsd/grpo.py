"""Neuro-Symbolic Continuous Solution Discovery (NS-CSD) with GRPO.

One training iteration:

1. The scenario curriculum supplies B contexts (replayed high-regret scenarios
   or fresh draws from the prior).  Each context is warmed up for W steps under
   the current policy to reach a realistic mid-episode state.
2. The context is forked G times.  All G members replay the *same* exogenous
   tape (common random numbers); G-1 members sample actions from the policy
   and, when the archive holds an elite for the context's niche, the last
   member is driven by that elite (neural snapshot or crystallised program).
3. Each member runs H steps through the symbolic shield; its return is the
   Sentinel defender reward minus the shield-repair penalty.
4. Group-relative advantages A_i = (R_i - mean_G R) / std_G R replace a learned
   critic.  Policy members get the clipped GRPO surrogate; the elite member
   contributes an advantage-weighted likelihood term only when it beat its
   group (A > 0).  A KL term anchors the policy to the archive champion.

Every ``eval_every`` iterations the discovery step validates and snapshots
the policy, crystallises it into a tree program, offers both to the Solution
Archive, re-scores the curriculum by regret, re-anchors to the champion, and
rolls the policy back to the champion after sustained regression.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import env as E
from . import scenarios as S
from .agents import NetDefender
from .archive import ScenarioCurriculum, SolutionArchive, scenario_scores, seed_programs
from .crystallize import crystallize
from .policy import DefenderNet, kl_to
from .rollout import run_open_loop, validation_score
from .shield import BOUNDS, DEFAULT, repair


@dataclass
class Config:
    seed: int = 42
    iters: int = 500
    B: int = 32              # contexts per iteration
    G: int = 8               # group size
    H: int = 8               # rollout horizon per group member
    warmup_max: int = 40
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    ent_coef: float = 0.005
    kl_coef: float = 0.02
    elite_coef: float = 0.5
    shield_penalty: float = 0.0   # Sentinel uses 0.1; see discover_shield for why NS-CSD drops it
    reward: str = "score"    # "score": composite objective; "reward": Sentinel's shaped reward
    sev_w: float = 0.5       # severe-outage weight of the objective (0.5 = Sentinel's benchmark)
    eval_every: int = 20
    replay_p: float = 0.5
    n_candidates: int = 16
    rollback_patience: int = 3
    rollback_margin: float = 0.02
    physics: str = "inline"
    # component switches (ablations)
    shield: bool = True
    discovery: bool = True
    elite: bool = True
    anchor: bool = True
    crn: bool = True
    crystallize: bool = True
    shield_discovery: bool = True
    shield_candidates: int = 8
    shield_sigma: float = 0.08
    shield_margin: float = 0.002
    shield_probes: int = 2       # group members that deploy under a perturbed shield


class _Stepper:
    """Applies policy/elite actions, the shield and the simulator to a batch."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.shield_kw = dict(DEFAULT)

    def act(self, policy, obs, elite_rows=None, elite_actions=None):
        thr, focus, drop = policy.act(obs, deterministic=False)
        if elite_rows is not None and elite_rows.any():
            thr[elite_rows], focus[elite_rows], drop[elite_rows] = elite_actions
        return thr, focus, drop

    def deploy(self, obs, thr, focus, drop, kw=None):
        if self.cfg.shield:
            s_thr, s_focus, s_drop, rep, _ = repair(obs, thr, focus, drop, **(kw or self.shield_kw))
            return (s_thr, s_focus, s_drop), rep
        return (thr, focus, drop), np.zeros(len(thr), bool)


def train(cfg: Config, log=print):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    policy = DefenderNet()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    ref = copy.deepcopy(policy)
    champion, champion_score, champion_iter = copy.deepcopy(policy), -np.inf, 0
    champion_kw, shield_log = dict(DEFAULT), []
    probe_dir = np.zeros(len(BOUNDS))
    archive = SolutionArchive()
    curriculum = ScenarioCurriculum(rng)
    stepper = _Stepper(cfg)
    history, env_steps, rollbacks, bad = [], 0, 0, 0
    best_tree, best_tree_score, best_tree_fid = None, -np.inf, None
    t0 = time.time()

    # Seed the archive with symbolic programs so regret is informative from the start.
    if cfg.discovery or cfg.elite:
        cache = _ScoreCache(curriculum.scen, cfg.seed, cfg.physics, cfg.sev_w)
        for prog in seed_programs(shielded=cfg.shield):
            archive.consider(prog, "seed-program", curriculum.scen, cache(prog), 0, cache.elites(archive))

    for it in range(1, cfg.iters + 1):
        # ---------------- 1. contexts ----------------
        replay_p = cfg.replay_p if cfg.discovery else 0.0
        scen = curriculum.sample(cfg.B, replay_p)
        W = int(rng.integers(0, cfg.warmup_max + 1))
        T = W + cfg.H
        (typ, inten, mut), tape = S.contexts(rng, scen, T)
        icmp = np.zeros(cfg.B, bool)
        obs = E.initial_obs(tape.legit0)
        for t in range(W):
            thr, focus, drop = stepper.act(policy, obs)
            action, _ = stepper.deploy(obs, thr, focus, drop)
            obs, _ = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), action, icmp, cfg.physics)
        env_steps += cfg.B * W

        # ---------------- 2. fork into groups ----------------
        idx = np.repeat(np.arange(cfg.B), cfg.G)
        n = cfg.B * cfg.G
        if cfg.crn:
            tapeG, sched = tape.take(idx), (typ[idx], inten[idx], mut[idx])
        else:  # independent traffic for every member from the fork point on
            (t2, i2, m2), tapeG = S.contexts(rng, scen[idx], T)
            sched = (t2, i2, m2)
        obsG = obs[idx]
        # shield probes: a few members per group deploy under a perturbed PanicGuard
        member = np.tile(np.arange(cfg.G), cfg.B)
        probe_rows = np.zeros(n, bool)
        kw_rows, eps = None, None
        if cfg.shield and cfg.shield_discovery and cfg.shield_probes > 0:
            probe_rows = (member >= cfg.G - 1 - cfg.shield_probes) & (member < cfg.G - 1)
            eps = rng.normal(0, cfg.shield_sigma, (n, len(BOUNDS))) * probe_rows[:, None]
            kw_rows = {}
            for j, (k, (lo, hi)) in enumerate(BOUNDS.items()):
                kw_rows[k] = np.clip(stepper.shield_kw[k] + eps[:, j] * (hi - lo), lo, hi)
        elite_rows = np.zeros(n, bool)
        elites = {}
        if cfg.elite:
            niches = S.descriptor(scen)
            for b in range(cfg.B):
                e = archive.elite(niches[b])
                if e is not None:
                    elite_rows[b * cfg.G + cfg.G - 1] = True
                    elites.setdefault(id(e), (e, []))[1].append(b * cfg.G + cfg.G - 1)

        # ---------------- 3. group rollouts ----------------
        O, A_thr, A_focus, A_drop = [], [], [], []
        R = np.zeros(n)
        for h in range(cfg.H):
            t = W + h
            el_thr, el_focus, el_drop = np.zeros(n), np.zeros(n, int), np.zeros(n)
            for e, rows in elites.values():
                rows = np.asarray(rows)
                a = e.raw(obsG[rows])
                el_thr[rows], el_focus[rows], el_drop[rows] = a
            thr, focus, drop = stepper.act(policy, obsG, elite_rows,
                                           (el_thr[elite_rows], el_focus[elite_rows], el_drop[elite_rows]))
            O.append(obsG); A_thr.append(thr); A_focus.append(focus); A_drop.append(drop)
            action, rep = stepper.deploy(obsG, thr, focus, drop, kw_rows)
            obsG, info = E.step(obsG, t, tapeG, (sched[0][:, t], sched[1][:, t], sched[2][:, t]),
                                action, np.zeros(n, bool), cfg.physics)
            r = info[cfg.reward] if cfg.reward != "score" else info["score"] - (cfg.sev_w - 0.5) * info["severe"]
            R += r - cfg.shield_penalty * rep
        env_steps += n * cfg.H

        Rg = R.reshape(cfg.B, cfg.G)
        std = Rg.std(1, keepdims=True)
        adv = np.where(std > 1e-6, (Rg - Rg.mean(1, keepdims=True)) / (std + 1e-8), 0.0).reshape(-1)
        if eps is not None:  # group-relative evolution-strategy direction for the shield
            probe_dir += (adv[probe_rows, None] * eps[probe_rows]).sum(0)

        # ---------------- 4. GRPO update ----------------
        obs_t = torch.as_tensor(np.concatenate(O), dtype=torch.float32)
        thr_t = torch.as_tensor(np.concatenate(A_thr), dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        foc_t = torch.as_tensor(np.concatenate(A_focus), dtype=torch.long)
        drop_t = torch.as_tensor(np.concatenate(A_drop), dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        adv_t = torch.as_tensor(np.tile(adv, cfg.H), dtype=torch.float32)
        is_el = torch.as_tensor(np.tile(elite_rows, cfg.H))
        # elite actions may be deterministic extremes; soften them for the likelihood term
        thr_t = torch.where(is_el, thr_t.clamp(0.01, 0.99), thr_t)
        drop_t = torch.where(is_el, drop_t.clamp(0.01, 0.99), drop_t)
        with torch.no_grad():
            old_lp = policy.log_prob(obs_t, thr_t, foc_t, drop_t)
        N = obs_t.shape[0]
        stats = dict(pg=0.0, kl=0.0, ent=0.0, el=0.0)
        for _ in range(cfg.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                lp = policy.log_prob(obs_t[mb], thr_t[mb], foc_t[mb], drop_t[mb])
                a, el = adv_t[mb], is_el[mb]
                ratio = torch.exp(lp - old_lp[mb])
                surr = torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                on = ~el
                pg = -(surr[on].mean() if on.any() else torch.zeros(()))
                el_loss = -(a[el].clamp(min=0) * lp[el]).mean() if el.any() else torch.zeros(())
                ent = policy.entropy(obs_t[mb]).mean()
                loss = pg - cfg.ent_coef * ent + cfg.elite_coef * el_loss
                if cfg.anchor and cfg.kl_coef > 0:
                    kl = kl_to(ref, policy, obs_t[mb]).mean()
                    loss = loss + cfg.kl_coef * kl
                    stats["kl"] += kl.item()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                stats["pg"] += pg.item(); stats["ent"] += ent.item(); stats["el"] += float(el_loss.detach())

        # ---------------- 5. discovery step ----------------
        if it % cfg.eval_every == 0 or it == cfg.iters:
            det = NetDefender(copy.deepcopy(policy), shielded=cfg.shield, shield_kw=dict(stepper.shield_kw))
            v, per = validation_score(det, cfg.seed, physics=cfg.physics, sev_w=cfg.sev_w)
            shield_acc = 0
            if cfg.shield and cfg.shield_discovery:
                kw, v_new, per_new = discover_shield(det, v, rng, cfg, probe_dir)
                probe_dir = np.zeros(len(BOUNDS))
                if kw is not None:
                    stepper.shield_kw, det.shield_kw, v, per, shield_acc = kw, dict(kw), v_new, per_new, 1
                    shield_log.append(dict(iter=it, val=v, **kw))
            improved = v > champion_score
            if improved:
                champion, champion_score, champion_iter, bad = copy.deepcopy(policy), v, it, 0
                champion_kw = dict(stepper.shield_kw)
            else:
                bad += 1
                if cfg.anchor and bad >= cfg.rollback_patience and v < champion_score - cfg.rollback_margin:
                    policy.load_state_dict(champion.state_dict())
                    stepper.shield_kw = dict(champion_kw)
                    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
                    rollbacks += 1
                    bad = 0
            if cfg.anchor:
                ref = copy.deepcopy(champion)

            tree_v = np.nan
            if cfg.crystallize:
                tree, fid = crystallize(det, rng, physics=cfg.physics)
                if not cfg.shield:
                    tree.shielded = False
                tree_v, _ = validation_score(tree, cfg.seed, physics=cfg.physics, sev_w=cfg.sev_w)
                if tree_v > best_tree_score:
                    best_tree, best_tree_score, best_tree_fid = tree, tree_v, fid

            n_ins, regret_mean = 0, 0.0
            if cfg.discovery or cfg.elite:
                pool = curriculum.scen
                cands = curriculum.propose(cfg.n_candidates) if cfg.discovery else np.zeros((0, S.DIM))
                allscen = np.concatenate([pool, cands])
                cache = _ScoreCache(allscen, cfg.seed * 1000 + it, cfg.physics, cfg.sev_w)
                pol_sc = cache(det)
                n_ins += archive.consider(det, "neural", allscen, pol_sc, it, cache.elites(archive))
                if cfg.crystallize:
                    n_ins += archive.consider(tree, "tree", allscen, cache(tree), it, cache.elites(archive))
                regret = np.maximum(0.0, cache.elites(archive) - pol_sc)
                regret_mean = float(regret[: len(pool)].mean())
                if cfg.discovery:
                    curriculum.update(cands, regret[len(pool):], regret[: len(pool)])

            rec = dict(iter=it, env_steps=env_steps, val=v, champion=champion_score, champion_iter=champion_iter,
                       shield_accept=shield_acc, **{f"shield_{k}": x for k, x in stepper.shield_kw.items()},
                       tree_val=tree_v, rollbacks=rollbacks, archive_inserts=n_ins, regret=regret_mean,
                       archive=str(archive.composition()), minutes=(time.time() - t0) / 60,
                       **{f"val_{k}": x for k, x in per.items()},
                       **{k: x / max(1, cfg.epochs) for k, x in stats.items()})
            history.append(rec)
            log(f"[seed {cfg.seed}] it {it:4d} steps {env_steps/1e6:.2f}M val {v:+.3f} "
                f"champ {champion_score:+.3f}@{champion_iter} tree {tree_v:+.3f} rb {rollbacks} "
                f"regret {regret_mean:.3f} shield {_fmt(stepper.shield_kw)} arch {archive.composition()}")

    return dict(policy=policy, champion=champion, champion_score=champion_score, champion_shield=champion_kw,
                final_shield=dict(stepper.shield_kw), shield_log=shield_log, best_tree=best_tree,
                best_tree_score=best_tree_score, best_tree_fid=best_tree_fid, history=history,
                archive=archive, config=asdict(cfg))


class _ScoreCache:
    """Per-scenario scores on one fixed scenario set with one CRN seed, cached per solution,
    so that candidates and incumbents are always compared on identical traffic."""

    def __init__(self, scen, seed, physics, sev_w=0.5):
        self.scen, self.seed, self.physics, self.sev_w = scen, seed, physics, sev_w
        self.niches = S.descriptor(scen)
        self.memo = {}

    def __call__(self, defender):
        key = id(defender)
        if key not in self.memo:
            self.memo[key] = (defender, scenario_scores(defender, self.scen, self.seed,
                                                                   physics=self.physics, sev_w=self.sev_w))
        return self.memo[key][1]

    def elites(self, archive):
        out = np.full(len(self.scen), -np.inf)
        for k in np.unique(self.niches):
            e = archive.elite(k)
            if e is not None:
                rows = self.niches == k
                out[rows] = self(e)[rows]
        return out


# ---------------------------------------------------------------------------
# Symbolic shield discovery
# ---------------------------------------------------------------------------
# Safety-certification battery: attack-free traffic with and without flash crowds.
CERT = {"benign": S.make([1, 0, 0, 0, 0, 0], i=(0, 0)),
        "flash": S.make([1, 0, 0, 0, 0, 0], i=(0, 0), flash=0.2)}


def _fmt(kw):
    return "(" + ",".join(f"{kw[k]:.2f}" for k in ("panic_util", "panic_drop", "panic_thr")) + ")"


def certify(defender, seed, physics, n_eps=10):
    """Service quality and outage counts on the certification battery (fixed CRN)."""
    res = {}
    for j, (name, vec) in enumerate(CERT.items()):
        out = run_open_loop(defender, np.tile(vec, (n_eps, 1)), np.random.default_rng(20_000 + 31 * seed + j),
                            100, physics, icmp_allowed=False)
        res[name] = (out["sq"].mean(), out["severe"].sum(), out["degraded"].sum())
    return res


def safe_wrt(res, ref, sq_tol=(0.002, 0.005)):
    """No safety regression relative to the operator-approved (default) shield."""
    for (name, (sq, sev, deg)), tol in zip(res.items(), sq_tol):
        r_sq, r_sev, r_deg = ref[name]
        if sq < r_sq - tol or sev > r_sev or deg > r_deg:
            return False
    return True


def discover_shield(det, v_cur, rng, cfg, direction=None):
    """Propose PanicGuard re-parameterisations around the current shield -- along the
    group-relative ES direction accumulated by the shield probes, plus random ones --
    and accept the best that improves validation and passes the safety certificate
    against the operator-approved default shield.

    Sentinel's shield-repair penalty is disabled in NS-CSD: it teaches the policy to
    propose only actions the current shield already allows, which removes any signal
    about whether a relaxed shield would help.  Safety is instead enforced by the
    certificate at adoption time.
    """
    ref = certify(NetDefender(det.net, True, shield_kw=dict(DEFAULT)), cfg.seed, cfg.physics)
    best_kw, best_v, best_per = None, v_cur + cfg.shield_margin, None
    cur = det.shield_kw
    steps = []
    if direction is not None and np.linalg.norm(direction) > 0:
        u = direction / np.linalg.norm(direction)
        steps = [u * cfg.shield_sigma * m for m in (0.5, 1.0, 2.0)]
    while len(steps) < cfg.shield_candidates:
        steps.append(rng.normal(0, cfg.shield_sigma, len(BOUNDS)))
    for step in steps:
        kw = {}
        for j, (k, (lo, hi)) in enumerate(BOUNDS.items()):
            kw[k] = float(np.clip(cur[k] + step[j] * (hi - lo), lo, hi))
        cand = NetDefender(det.net, True, shield_kw=kw)
        v, per = validation_score(cand, cfg.seed, physics=cfg.physics, sev_w=cfg.sev_w)
        if v > best_v and safe_wrt(certify(cand, cfg.seed, cfg.physics), ref):
            best_kw, best_v, best_per = kw, v, per
    return best_kw, best_v, best_per
