"""Parametric attack scenarios (the search space of scenario discovery).

A scenario is a 14-dim vector

    [w_none, w_syn, w_udp, w_http, w_icmp, w_mixed,   # attack-family weights
     stay,                                            # Markov persistence of the family
     i_lo, i_hi,                                      # intensity range (resampled on switch)
     m_lo, m_hi,                                      # polymorphic mutation range
     pulse_period, duty,                              # on/off pulsing (period 0 = always on)
     flash_prob]                                      # flash-crowd probability per step

and is turned into an open-loop attack schedule plus an exogenous-noise
:class:`~nscsd.env.Tape`.  Open-loop schedules make common random numbers exact:
all members of a GRPO group replay the same traffic.
"""
from __future__ import annotations

import numpy as np

from .env import Tape

DIM = 14
W = slice(0, 6)
STAY, ILO, IHI, MLO, MHI, PER, DUTY, FLASH = range(6, 14)
TRAIN_FAMILIES = np.array([0, 1, 2, 3, 5])  # ICMP (4) is held out of training


def make(w, stay=0.95, i=(0.0, 1.0), m=(0.0, 1.0), period=0, duty=1.0, flash=0.0):
    v = np.zeros(DIM)
    v[W] = np.asarray(w, dtype=float) / np.sum(w)
    v[STAY], v[ILO], v[IHI], v[MLO], v[MHI] = stay, i[0], i[1], m[0], m[1]
    v[PER], v[DUTY], v[FLASH] = period, duty, flash
    return v


ATTACKS = [0, 1, 1, 1, 0, 1]      # SYN, UDP, HTTP, mixed
ATTACKS_NONE = [1, 1, 1, 1, 0, 1]

# Standardised, attacker-agnostic evaluation battery (Protocol B).  The first
# five mirror the Sentinel scenarios; the rest are held-out stress tests that
# are never used for training or model selection.
EVAL = {
    "benign": make([1, 0, 0, 0, 0, 0], i=(0, 0)),
    "mild": make(ATTACKS, i=(0.0, 0.3)),
    "strong": make(ATTACKS, i=(1.0, 1.0)),
    "chaos": make(ATTACKS_NONE, i=(0.0, 1.0), flash=0.05),
    "icmp": make([0, 0, 0, 0, 1, 0], i=(1.0, 1.0)),
    "pulse": make(ATTACKS, i=(1.0, 1.0), period=10, duty=0.5),
    "polymorph": make(ATTACKS, stay=0.5, i=(0.5, 1.0), m=(0.7, 1.0)),
    "boundary": make(ATTACKS, i=(0.3, 0.6)),
    "icmp_chaos": make([1, 1, 1, 1, 3, 1], i=(0.5, 1.0), flash=0.05),
}
CORE = ["benign", "mild", "strong", "chaos", "icmp"]
STRESS = ["pulse", "polymorph", "boundary", "icmp_chaos"]
# Validation battery used for model selection (ICMP stays genuinely held out).
VALIDATION = {k: EVAL[k] for k in ["benign", "mild", "strong", "chaos"]}


def random_scenarios(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform prior over the training scenario space (no ICMP)."""
    out = np.zeros((n, DIM))
    k = rng.integers(1, 4, size=n)  # number of active families
    for j in range(n):
        fam = rng.choice(TRAIN_FAMILIES, size=k[j], replace=False)
        out[j, fam] = rng.dirichlet(np.ones(k[j]))
    out[:, STAY] = rng.uniform(0.5, 1.0, n)
    a, b = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    out[:, ILO], out[:, IHI] = np.minimum(a, b), np.maximum(a, b)
    a, b = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    out[:, MLO], out[:, MHI] = np.minimum(a, b), np.maximum(a, b)
    pulse = rng.random(n) < 0.25
    out[:, PER] = np.where(pulse, rng.integers(4, 21, n), 0)
    out[:, DUTY] = np.where(pulse, rng.uniform(0.2, 0.8, n), 1.0)
    out[:, FLASH] = np.where(rng.random(n) < 0.3, 0.05, 0.0)
    return out


def mutate(rng: np.random.Generator, s: np.ndarray, scale: float = 0.15) -> np.ndarray:
    """ACCEL-style local edit of a batch of scenarios, kept inside the space."""
    s = s.copy()
    n = s.shape[0]
    w = s[:, W] + rng.normal(0, scale, (n, 6)) * (rng.random((n, 6)) < 0.5)
    w[:, 4] = 0.0
    w = np.clip(w, 0, None)
    w[w.sum(1) == 0, 1] = 1.0
    s[:, W] = w / w.sum(1, keepdims=True)
    s[:, STAY] = np.clip(s[:, STAY] + rng.normal(0, scale, n), 0.3, 1.0)
    for lo, hi in ((ILO, IHI), (MLO, MHI)):
        a = np.clip(s[:, lo] + rng.normal(0, scale, n), 0, 1)
        b = np.clip(s[:, hi] + rng.normal(0, scale, n), 0, 1)
        s[:, lo], s[:, hi] = np.minimum(a, b), np.maximum(a, b)
    flip = rng.random(n) < 0.1
    s[:, PER] = np.where(flip, np.where(s[:, PER] > 0, 0, rng.integers(4, 21, n)), s[:, PER])
    s[:, DUTY] = np.where(s[:, PER] > 0, np.clip(s[:, DUTY] + rng.normal(0, scale, n), 0.2, 0.8), 1.0)
    s[:, FLASH] = np.where(rng.random(n) < 0.1, 0.05 - s[:, FLASH], s[:, FLASH])
    return s


def descriptor(s: np.ndarray) -> np.ndarray:
    """Coarse niche of a scenario: (dominant family, intensity bin)."""
    fam = s[:, W].argmax(1)
    imid = 0.5 * (s[:, ILO] + s[:, IHI])
    ibin = np.digitize(imid, [0.3, 0.7])
    return fam * 3 + ibin  # 18 niches


def schedule(rng: np.random.Generator, scen: np.ndarray, T: int):
    """Open-loop attack schedule (type, intensity, mutation) of shape (n, T)."""
    n = scen.shape[0]
    w = scen[:, W]
    cdf = np.cumsum(w, 1)
    typ = np.zeros((n, T), dtype=np.int64)
    inten = np.zeros((n, T))
    mut = np.zeros((n, T))
    u_type, u_stay = rng.random((n, T)), rng.random((n, T))
    u_i, u_m = rng.random((n, T)), rng.random((n, T))
    phase = rng.integers(0, 1000, n)
    cur = (u_type[:, 0:1] > cdf).sum(1).clip(0, 5)
    ci = scen[:, ILO] + u_i[:, 0] * (scen[:, IHI] - scen[:, ILO])
    cm = scen[:, MLO] + u_m[:, 0] * (scen[:, MHI] - scen[:, MLO])
    for t in range(T):
        if t > 0:
            sw = u_stay[:, t] > scen[:, STAY]
            new = (u_type[:, t:t + 1] > cdf).sum(1).clip(0, 5)
            cur = np.where(sw, new, cur)
            ci = np.where(sw, scen[:, ILO] + u_i[:, t] * (scen[:, IHI] - scen[:, ILO]), ci)
            cm = np.where(sw, scen[:, MLO] + u_m[:, t] * (scen[:, MHI] - scen[:, MLO]), cm)
        per = scen[:, PER]
        on = np.where(per > 0, ((t + phase) % np.maximum(per, 1)) < scen[:, DUTY] * per, True)
        typ[:, t] = np.where(on, cur, 0)
        inten[:, t] = np.where(on & (cur > 0), ci, 0.0)
        mut[:, t] = cm
    return typ, inten, mut


def contexts(rng: np.random.Generator, scen: np.ndarray, T: int):
    """Schedule + exogenous tape for a batch of scenarios."""
    typ, inten, mut = schedule(rng, scen, T)
    tape = Tape.sample(rng, scen.shape[0], T, flash_prob=scen[:, FLASH])
    return (typ, inten, mut), tape
