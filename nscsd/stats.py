"""Seed-level statistics: t-based 95% CIs, paired tests with Holm correction, Pareto dominance."""
from __future__ import annotations

import numpy as np
from scipy import stats


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(np.mean(x)), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def paired(a, b):
    """Paired t-test and Wilcoxon signed-rank test of a vs b (same seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return dict(diff=0.0, t_p=1.0, w_p=1.0, d_z=0.0)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = 1.0
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    return dict(diff=float(d.mean()), t_p=float(t.pvalue), w_p=float(w), d_z=float(dz))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def dominates(a, b, tol=1e-9):
    """a, b = (quality, leakage, severe); quality up, leakage and severe down."""
    ge = a[0] >= b[0] - tol and a[1] <= b[1] + tol and a[2] <= b[2] + tol
    gt = a[0] > b[0] + tol or a[1] < b[1] - tol or a[2] < b[2] - tol
    return ge and gt
