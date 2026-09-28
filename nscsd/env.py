"""Vectorised DDoS wargame simulator.

A batched NumPy port of the single-link simulator released with Sentinel
(Alfatemi et al., IEEE TNSM 2026, ``simulation/env.py`` in
github.com/AliAlfatemi/sentinel-ddos, MIT licence).  The per-step dynamics,
reward and observation model are reproduced term by term so that Sentinel's
published checkpoints can be evaluated on exactly the same testbed;
``tests/test_env_equivalence.py`` checks the port against the reference code.

Two deliberate extensions, both off by default:

* ``physics="upstream"`` computes link overload from the *post-filter* load
  (mitigation placed upstream of the bottleneck, e.g. an in-network scrubber on
  a 6G programmable data plane).  In the reference ("inline") physics the
  overload penalty uses the pre-filter load, which caps service quality at
  1000/1800 = 0.556 under a full-intensity flood whatever the defender does.
* All exogenous randomness is drawn from a :class:`Tape` so that several
  copies of the same context can be replayed with common random numbers (CRN),
  which is what makes the group-relative advantages in GRPO low-variance.

Attack types: 0 none, 1 SYN, 2 UDP, 3 HTTP, 4 ICMP, 5 mixed/polymorphic.
Defender focus: 0 source-IP, 1 protocol, 2 packet-size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_CAPACITY = 1000.0
LEGIT_MEAN = 300.0
OBS_DIM = 12
FEATURES = [
    "total_packet_rate", "total_byte_rate", "link_utilization",
    "tcp_ratio", "udp_ratio", "icmp_ratio", "http_ratio",
    "src_ip_entropy", "dst_port_entropy", "packet_size_entropy",
    "flow_count", "queue_signal",
]

# Reward weights of the released Sentinel code (alpha, beta, lambda, eta, shield).
ALPHA, BETA, LAMBDA, ETA, SHIELD_PENALTY = 1.0, 1.0, 1.0, 2.0, 0.1
SIGMA_MIN, DELTA_MAX = 0.90, 0.10

# match_score[attack_type, focus] from the reference implementation.
MATCH = np.zeros((6, 3))
MATCH[1] = [0.6, 0.8, 0.0]
MATCH[2] = [0.0, 0.9, 0.0]
MATCH[3] = [0.7, 0.0, 0.5]
MATCH[4] = [0.0, 0.9, 0.0]
MATCH[5] = [0.4, 0.4, 0.4]

INITIAL_OBS_TAIL = np.array([0.8, 0.1, 0.05, 0.05, 0.9, 0.9, 0.9, 0.5, 0.0], dtype=np.float64)


@dataclass
class Tape:
    """Pre-sampled exogenous randomness for ``n`` environments and ``T`` steps.

    Every quantity that the reference simulator samples inside ``step`` is
    stored here, so two environments that share a tape see identical traffic
    and identical observation noise; only the defender's actions differ.
    """

    legit: np.ndarray        # (n, T) Poisson legitimate arrivals (after flash crowds)
    proto_noise: np.ndarray  # (n, T, 4)
    ent_noise: np.ndarray    # (n, T, 3)
    pkt_noise: np.ndarray    # (n, T)
    reroll: np.ndarray       # (n, T) replacement type used when ICMP is disallowed
    legit0: np.ndarray       # (n,) legitimate arrivals used for the reset observation

    @staticmethod
    def sample(rng: np.random.Generator, n: int, T: int, flash_prob=0.0,
               flash_lo=2.5, flash_hi=4.0) -> "Tape":
        legit = rng.poisson(LEGIT_MEAN, size=(n, T)).astype(np.float64)
        flash_prob = np.broadcast_to(np.asarray(flash_prob, dtype=np.float64), (n,))
        flash = rng.random((n, T)) < flash_prob[:, None]
        legit = np.where(flash, legit * rng.uniform(flash_lo, flash_hi, size=(n, T)), legit)
        return Tape(
            legit=legit,
            proto_noise=rng.normal(0.0, 0.02, size=(n, T, 4)),
            ent_noise=rng.normal(0.0, 0.02, size=(n, T, 3)),
            pkt_noise=rng.normal(0.0, 0.1, size=(n, T)),
            reroll=rng.choice(np.array([0, 1, 2, 3, 5]), size=(n, T)),
            legit0=rng.poisson(LEGIT_MEAN, size=n).astype(np.float64),
        )

    def take(self, idx: np.ndarray) -> "Tape":
        """Select environments ``idx`` (used to replicate a context G times)."""
        return Tape(self.legit[idx], self.proto_noise[idx], self.ent_noise[idx],
                    self.pkt_noise[idx], self.reroll[idx], self.legit0[idx])


def initial_obs(legit0: np.ndarray) -> np.ndarray:
    byte = legit0 / MAX_CAPACITY
    n = legit0.shape[0]
    obs = np.empty((n, OBS_DIM), dtype=np.float64)
    obs[:, 0] = byte * 0.8
    obs[:, 1] = byte
    obs[:, 2] = np.minimum(1.0, byte)
    obs[:, 3:] = INITIAL_OBS_TAIL
    return obs


def step(obs_unused, t, tape: Tape, attack, defense, icmp_allowed, physics="inline"):
    """Advance ``n`` environments by one control window.

    ``attack`` = (type int[n], intensity[n], mutation[n]);
    ``defense`` = (threshold[n], focus int[n], drop[n]) -- the *deployed* action.
    ``icmp_allowed`` bool[n]: when False an ICMP choice is re-rolled, as in the
    reference simulator outside zero-shot evaluation.
    Returns (next_obs, info dict of per-env arrays).  The shield penalty is
    added by the caller because it depends on the raw (pre-shield) action.
    """
    a_type, intensity, mutation = attack
    thr_n, focus, drop = defense
    a_type = np.where((a_type == 4) & ~icmp_allowed, tape.reroll[:, t], a_type).astype(np.int64)
    legit = tape.legit[:, t]
    attack_traffic = np.where(a_type > 0, intensity * MAX_CAPACITY * 1.5, 0.0)

    threshold = thr_n * MAX_CAPACITY
    total_load = legit + attack_traffic
    act = np.clip((total_load - threshold) / np.maximum(total_load, 1e-8), 0.0, 1.0)

    match = MATCH[a_type, focus] * (1.0 - mutation * 0.5)
    attacked = a_type > 0
    below = threshold < legit
    legit_safe = np.maximum(legit, 1e-12)
    eff = np.where(attacked, match * drop * act, 0.0)
    cd = np.where(attacked,
                  drop * (1.0 - match) * 0.4 * act + np.where(below, 0.2 * (legit - threshold) / legit_safe, 0.0),
                  drop * 0.05 * act + np.where(below, 0.1 * (legit - threshold) / legit_safe, 0.0))
    cd = np.minimum(1.0, cd)
    eff = np.minimum(1.0, eff)

    legit_dropped = legit * cd
    attack_dropped = attack_traffic * eff
    legit_served = legit - legit_dropped
    attack_served = attack_traffic - attack_dropped

    sq = legit_served / np.maximum(legit, 1.0)
    cd_metric = legit_dropped / np.maximum(legit, 1.0)
    leak = np.where(attack_traffic > 0, attack_served / np.maximum(attack_traffic, 1.0), 0.0)
    mit = np.where(attack_traffic > 0, attack_dropped / np.maximum(attack_traffic, 1.0), 1.0)

    load_for_overload = total_load if physics == "inline" else legit_served + attack_served
    overload = np.maximum(0.0, load_for_overload - MAX_CAPACITY)
    sq = np.where(overload > 0, sq * (1.0 - overload / np.maximum(load_for_overload, 1e-8)), sq)
    sla = ((sq < SIGMA_MIN) | (cd_metric > DELTA_MAX)).astype(np.float64)
    reward = ALPHA * sq - BETA * cd_metric - LAMBDA * leak - ETA * sla

    # ---- observation model (aggregate telemetry only) ----
    n = legit.shape[0]
    proto = np.tile(np.array([0.8, 0.1, 0.05, 0.05]), (n, 1))
    ent = np.tile(np.array([0.9, 0.9, 0.9]), (n, 1))
    r = attack_served / np.maximum(1.0, total_load)
    on = r > 0.05
    low_mut_src = np.where(mutation < 0.5, 0.2, 0.7)

    def mix(base, target, mask):
        return np.where(mask, (1 - r) * base + r * target, base)

    m1, m2, m3, m4, m5 = [on & (a_type == k) for k in (1, 2, 3, 4, 5)]
    proto[:, 0] = mix(0.8, 1.0, m1 | m3)
    ent[:, 0] = mix(0.9, low_mut_src, m1)
    ent[:, 1] = mix(0.9, 0.1, m1)
    proto[:, 1] = mix(0.1, 1.0, m2)
    ent[:, 1] = np.where(m2, (1 - r) * 0.9 + r * 0.8, ent[:, 1])
    proto[:, 3] = mix(0.05, 1.0, m3)
    ent[:, 2] = mix(0.9, 0.2, m3)
    proto[:, 2] = mix(0.05, 1.0, m4)
    ent[:, 2] = np.where(m4, (1 - r) * 0.9 + r * 0.05, ent[:, 2])
    ent[:, 0] = np.where(m5, (1 - r) * 0.9 + r * 0.4, ent[:, 0])
    ent[:, 1] = np.where(m5, (1 - r) * 0.9 + r * 0.5, ent[:, 1])

    proto = proto / proto.sum(1, keepdims=True)
    proto = np.clip(proto + tape.proto_noise[:, t], 0.0, 1.0)
    proto = proto / proto.sum(1, keepdims=True)
    ent = np.clip(ent + tape.ent_noise[:, t], 0.0, 1.0)

    byte = np.clip(total_load / MAX_CAPACITY, 0.0, 1.0)
    pkt = np.clip(byte * (1.0 + tape.pkt_noise[:, t]), 0.0, 1.0)
    flow = np.clip(0.5 + pkt * ent[:, 0] * 0.5, 0.0, 1.0)
    queue = (overload > 0).astype(np.float64)
    nxt = np.concatenate([pkt[:, None], byte[:, None], byte[:, None], proto, ent,
                          flow[:, None], queue[:, None]], axis=1)
    severe = (sq < 0.50).astype(np.float64)
    degraded = (sq < 0.75).astype(np.float64)
    # per-step form of Sentinel's composite benchmark score, Eq. (15)
    score = sq - leak - 0.5 * severe - 0.25 * degraded
    info = dict(sq=sq, cd=cd_metric, leak=leak, mit=mit, sla=sla, reward=reward, score=score,
                severe=severe, degraded=degraded,
                miss=(sq < 0.90).astype(np.float64), attack_type=a_type,
                attacked=attacked.astype(np.float64))
    return nxt, info
