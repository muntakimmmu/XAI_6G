"""Batched evaluation rollouts and the Sentinel benchmark score."""
from __future__ import annotations

import numpy as np

from . import env as E
from . import scenarios as S
from .shield import RULES, repair


def benchmark_score(sq, leak, severe_rate, degraded_rate, sev_w=0.5):
    """Sentinel's composite checkpoint score, Eq. (15); ``sev_w`` lets an operator
    weight severe outages differently (0.5 reproduces Sentinel)."""
    return sq - leak - sev_w * severe_rate - 0.25 * degraded_rate


def deploy(defender, obs, shield_kw=None):
    shield_kw = shield_kw or getattr(defender, "shield_kw", None)
    thr, focus, drop = defender.raw(obs)
    thr, focus, drop = np.asarray(thr, float), np.asarray(focus, int), np.asarray(drop, float)
    if defender.shielded:
        s_thr, s_focus, s_drop, rep, fired = repair(obs, thr, focus, drop, **(shield_kw or {}))
        return (s_thr, s_focus, s_drop), rep, fired
    n = obs.shape[0]
    return (thr, focus, drop), np.zeros(n, bool), np.zeros((n, len(RULES)), bool)


def run_open_loop(defender, scen, rng, T=100, physics="inline", icmp_allowed=True, return_obs=False):
    """Roll ``defender`` on open-loop scenarios ``scen`` (n,14); one episode per row."""
    (typ, inten, mut), tape = S.contexts(rng, scen, T)
    return _run(defender, tape, T, physics, icmp_allowed,
                attack_fn=lambda obs, t: (typ[:, t], inten[:, t], mut[:, t]), return_obs=return_obs)


def run_closed_loop(defender, attacker, scenario, n_eps, rng, T=100, physics="inline"):
    """Replay Sentinel's evaluation protocol (``evaluate_system.evaluate_scenario``):
    the trained Sentinel attacker acts on the observation and the scenario overrides it."""
    flash = 0.05 if scenario == "chaos_flash_crowd" else 0.0
    zero_shot = scenario == "zero_shot_icmp"
    tape = E.Tape.sample(rng, n_eps, T, flash_prob=flash)

    def attack_fn(obs, t):
        typ, inten, mut = attacker.act(obs)
        if scenario == "benign":
            typ = np.zeros_like(typ)
        elif scenario == "mild_attack":
            typ = np.where(typ == 0, 1, typ)
            inten = np.clip(inten, 0.0, 0.3)
        elif scenario == "strong_attack":
            typ = np.where(typ == 0, 1, typ)
            inten = np.ones_like(inten)
        elif scenario == "zero_shot_icmp":
            typ = np.full_like(typ, 4)
            inten = np.ones_like(inten)
        return typ, inten, mut

    return _run(defender, tape, T, physics, zero_shot, attack_fn)


def _run(defender, tape, T, physics, icmp_allowed, attack_fn, return_obs=False):
    n = tape.legit.shape[0]
    obs = E.initial_obs(tape.legit0)
    icmp_allowed = np.broadcast_to(np.asarray(icmp_allowed, bool), (n,))
    defender.reset(n)
    acc = {k: np.zeros(n) for k in ("sq", "leak", "severe", "degraded", "miss", "reward", "attacked",
                                    "repairs", "cd")}
    leak_att = np.zeros(n)
    fired_tot = np.zeros((n, len(RULES)))
    obs_log, act_log = [], []
    for t in range(T):
        action, rep, fired = deploy(defender, obs)
        if return_obs:
            obs_log.append(obs.copy())
            act_log.append(np.stack([action[0], action[1], action[2]], 1))
        nxt, info = E.step(obs, t, tape, attack_fn(obs, t), action, icmp_allowed, physics)
        for k in ("sq", "leak", "severe", "degraded", "miss", "reward", "attacked", "cd"):
            acc[k] += info[k]
        leak_att += info["leak"] * info["attacked"]
        acc["repairs"] += rep
        fired_tot += fired
        obs = nxt
    out = {k: v / T for k, v in acc.items() if k not in ("repairs",)}
    out["leak_att"] = np.where(acc["attacked"] > 0, leak_att / np.maximum(acc["attacked"], 1), 0.0)
    out["repairs"] = acc["repairs"]
    out["fired"] = fired_tot
    if return_obs:
        out["obs"] = np.concatenate(obs_log)
        out["act"] = np.concatenate(act_log)
    return out


def summarize(out, leak_key="leak_att"):
    """Aggregate an episode batch into per-500-step counts (Sentinel's reporting unit)."""
    sq, leak = out["sq"].mean(), out[leak_key].mean()
    sev, deg, miss = out["severe"].mean(), out["degraded"].mean(), out["miss"].mean()
    return dict(quality=sq, leakage=leak, severe=500 * sev, degraded=500 * deg, sla_miss=500 * miss,
                benchmark=benchmark_score(sq, leak, sev, deg), repairs=out["repairs"].mean() * 5)


def validation_score(defender, seed, n_eps=10, T=100, physics="inline", battery=None, sev_w=0.5):
    """Composite score on the validation battery with fixed CRN tapes (ICMP excluded)."""
    battery = battery or S.VALIDATION
    scores = {}
    for j, (name, vec) in enumerate(battery.items()):
        rng = np.random.default_rng(10_000 + 97 * seed + j)
        out = run_open_loop(defender, np.tile(vec, (n_eps, 1)), rng, T, physics, icmp_allowed=False)
        r = summarize(out)
        scores[name] = benchmark_score(r["quality"], r["leakage"], r["severe"] / 500, r["degraded"] / 500, sev_w)
    return float(np.mean(list(scores.values()))), scores
