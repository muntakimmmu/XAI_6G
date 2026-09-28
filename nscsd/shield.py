"""Vectorised symbolic safety shield.

Batched port of Sentinel's ``SymbolicShield.repair`` (same rules, same order,
same constants).  The three PanicGuard constants are exposed as parameters
so that NS-CSD can re-discover them under a safety certificate; the defaults
reproduce Sentinel exactly.  ``repair`` returns the deployed action, a per-env flag telling
whether any field changed (used for the shield penalty), and a per-rule firing
matrix so rule activations can be audited.
"""
from __future__ import annotations

import numpy as np

RULES = [
    "PanicGuard_Drop", "PanicGuard_Threshold", "HTTP_SourceIP_Exception",
    "SpoofingGuard_Override_Protocol", "SpoofingGuard_Override_PacketSize",
    "SpoofingGuard_Fallback", "ProtocolAnomalyGuard", "ICMP_Protocol_Override",
    "ICMP_Drop_Override", "ICMP_Threshold_Override", "ExtremeDropCap",
]
R = {name: i for i, name in enumerate(RULES)}


DEFAULT = dict(panic_util=0.6, panic_drop=0.1, panic_thr=0.4)
# Bounds of the PanicGuard parameters that shield discovery may explore.
BOUNDS = dict(panic_util=(0.3, 0.9), panic_drop=(0.05, 0.95), panic_thr=(0.0, 0.6))


def repair(obs, thr, focus, drop, icmp_min_drop=0.5, icmp_max_thr=0.5,
           panic_util=0.6, panic_drop=0.1, panic_thr=0.4):
    obs = np.asarray(obs)
    thr0, focus0, drop0 = thr, focus, drop
    thr, focus, drop = thr.copy(), focus.copy(), drop.copy()
    n = obs.shape[0]
    fired = np.zeros((n, len(RULES)), dtype=bool)
    util, udp, icmp, http = obs[:, 2], obs[:, 4], obs[:, 5], obs[:, 6]
    src_ent, size_ent = obs[:, 7], obs[:, 9]

    # 1. PanicGuard: no aggressive filtering at normal utilisation.
    panic_util, panic_drop, panic_thr = (np.broadcast_to(np.asarray(x, float), (n,))
                                         for x in (panic_util, panic_drop, panic_thr))
    low = util < panic_util
    m = low & (drop > panic_drop); drop = np.where(m, panic_drop, drop); fired[:, R["PanicGuard_Drop"]] = m
    m = low & (thr < panic_thr); thr = np.where(m, panic_thr, thr); fired[:, R["PanicGuard_Threshold"]] = m

    # 2. SpoofingGuard: source-IP blocking is harmful when sources are diverse.
    sg = (src_ent > 0.85) & (focus0 == 0) & (util > 0.6)
    http_exc = sg & (http > 0.3)
    fired[:, R["HTTP_SourceIP_Exception"]] = http_exc
    rest = sg & ~http_exc
    proto_anom = (udp > 0.5) | (icmp > 0.5) | (http > 0.5)
    m = rest & proto_anom; focus = np.where(m, 1, focus); fired[:, R["SpoofingGuard_Override_Protocol"]] = m
    m = rest & ~proto_anom & (size_ent < 0.3); focus = np.where(m, 2, focus)
    fired[:, R["SpoofingGuard_Override_PacketSize"]] = m
    m = rest & ~proto_anom & ~(size_ent < 0.3); focus = np.where(m, 1, focus)
    fired[:, R["SpoofingGuard_Fallback"]] = m

    # 3. Protocol-anomaly guard (HTTP floods may keep source-IP focus).
    extreme = ((udp > 0.8) | (icmp > 0.8) | (http > 0.8)) & (util > 0.6) & (focus != 1)
    m = extreme & ~((http > 0.8) & (focus == 0))
    focus = np.where(m, 1, focus); fired[:, R["ProtocolAnomalyGuard"]] = m

    # 3.5 ICMP guard: protocol focus with a minimum mitigation strength.
    ic = (icmp > 0.3) & (util > 0.6)
    m = ic & (focus != 1); focus = np.where(m, 1, focus); fired[:, R["ICMP_Protocol_Override"]] = m
    m = ic & (drop < icmp_min_drop); drop = np.where(m, icmp_min_drop, drop); fired[:, R["ICMP_Drop_Override"]] = m
    m = ic & (thr > icmp_max_thr); thr = np.where(m, icmp_max_thr, thr); fired[:, R["ICMP_Threshold_Override"]] = m

    # 4. Cap extreme drops.
    m = drop > 0.95; drop = np.where(m, 0.95, drop); fired[:, R["ExtremeDropCap"]] = m

    repaired = (np.abs(thr0 - thr) > 1e-4) | (focus0 != focus) | (np.abs(drop0 - drop) > 1e-4)
    return thr, focus, drop, repaired, fired
