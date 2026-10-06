# NS-CSD: Neuro-Symbolic Continuous Solution Discovery with GRPO

This repository is a research codebase and paper draft. It replaces the **co-evolutionary** training in
*Sentinel: A Neuro-Symbolic Co-Evolutionary Framework for Trustworthy Network and Service
Management Against DDoS Attacks* (Alfatemi et al., IEEE TNSM 23, 2026) with **continuous
solution discovery** driven by **Group Relative Policy Optimization (GRPO)**.

Sentinel trains a PPO defender behind a symbolic safety shield against a co-evolving PPO attacker
with a Hall-of-Fame archive. Its evaluation reports that all 10 seeds degrade late in training. NS-CSD
keeps Sentinel's POMDP, action space and shield, and changes how the defender is trained:

| Sentinel | NS-CSD |
|---|---|
| PPO with a learned critic | **Critic-free GRPO**. Groups are forked from one simulator state and replay identical traffic (common random numbers). |
| Co-evolving RL attacker + Hall of Fame | **Regret-driven scenario curriculum** over a 14-d attack-scenario space (PLR/ACCEL-style). The regret signal is measured against an archive of known solutions. |
| Benchmark checkpoint chosen after training | **Monotone Solution Archive** (MAP-Elites over 18 scenario niches). Neural snapshots, crystallised tree programs and seed programs compete as elites. The champion anchors training through a KL term and roll-back. |
| Fixed shield | **Certificate-gated shield discovery**. Shield probes inside GRPO groups give a group-relative ES direction. A change is adopted only if it shows no safety regression versus the operator's default shield on benign and flash-crowd traffic. |
| Rule crystallisation for inspection | **Continuous crystallisation**. Depth-4 tree programs are archive members and can be deployed. |

## Layout

```
nscsd/        env.py (vectorised port of Sentinel's simulator), shield.py, scenarios.py,
              policy.py (Sentinel-compatible nets), grpo.py (NS-CSD), archive.py, crystallize.py,
              ppo.py (equal-budget PPO control), rollout.py, agents.py, stats.py
scripts/      train.py, run_all.sh, evaluate.py (protocols A/B/C), analyze.py, diagnostics.py
tests/        equivalence tests against Sentinel's reference simulator and shield
results/      per-seed CSVs, RESULTS.md, DIAGNOSTICS.md
paper/        LaTeX manuscript (main.tex) and figures
models/       NS-CSD champion policies, discovered shields and tree programs (10 seeds)
```

## Reproduce

```bash
pip install -r requirements.txt
git clone https://github.com/AliAlfatemi/sentinel-ddos ../sentinel-ddos   # reference + checkpoints
python -m pytest -q tests                  # port == reference (set SENTINEL_REPO for another location)
bash scripts/run_all.sh inline            # train all methods (4 workers, several CPU-hours)
python scripts/evaluate.py
python scripts/diagnostics.py             # reproduction of Sentinel's table + oracle headroom
python scripts/analyze.py                 # tables -> results/RESULTS.md, figures -> paper/figures
```

Single run: `python scripts/train.py --method nscsd --seed 42` (about 5-9 min on one CPU core).

## Scope

All results come from Sentinel's purpose-built simulator, not from production traces. The safety
certificate for shield discovery is empirical (fixed common random numbers on a certification battery).
It is not a formal proof. See the Limitations section of the paper.

The simulator and shield port are derived from Sentinel's MIT-licensed code
(https://github.com/AliAlfatemi/sentinel-ddos).

## Supplementary material (IEEE Networking Letters)

`bash scripts/make_supplementary.sh` builds `dist/IMBANG_supplementary_material.pdf` (from
`paper/nl/supp/`) and `dist/IMBANG_supplementary.zip` (code, results, checkpoints; see
`supplementary/README.md`).
