# IMBANG: supplementary material

Supplementary material for the letter *IMBANG: Balancing DDoS Mitigation and Service Outages with
Safety-Spanning Group-Relative Policy Optimisation* (M. Rahaman, A. Mahmud, A. Chehri),
submitted to IEEE Networking Letters.

IMBANG stands for **I**nterpretable, **M**onotone-safe, **B**udgeted, **A**dversary-free,
**N**euro-symbolic **G**RPO. The archive contains:

- the simulator, the training code and the evaluation code;
- the per-seed results behind every table and figure;
- the trained checkpoints (10 seeds) of every method reported in the letter;
- the supplementary document `IMBANG_supplementary_material.pdf`: full simulator and shield
  specification, complete proofs, all settings, and extended results.

Every number in the letter can be regenerated from these files.

## Contents

```
nscsd/              library: batched port of Sentinel's simulator (env.py) and shield (shield.py),
                    scenarios, policy networks, GRPO learner with the safe-side member (grpo.py),
                    PPO baseline (ppo.py), MaxMC curriculum (archive.py), tree distillation
                    (crystallize.py), rollouts and statistics
scripts/            training, evaluation, probes, and table/figure generation (see below)
tests/              equivalence tests of the port against Sentinel's reference implementation
results/            per-seed CSV files and summaries used in the letter
runs/inline/        trained checkpoints: <method>/seed<42..51>/{champion.pt, final.pt, meta.json,
                    history.csv, tree.pkl, tree_rules.txt}
reference_figures/  Figs. 2-5 as they appear in the letter
IMBANG_supplementary_material.pdf   supplementary document (Tables S1-S17, Figs. S1-S3)
NOTICE              attribution for the parts derived from Sentinel (MIT licence)
```

## Method names used in the code

The code predates the final name. In file names and CSV columns, the methods appear as follows.

| Code name | Name in the letter |
|---|---|
| `span` | **IMBANG** |
| `span_cgrpo` | Constrained GRPO (C-GRPO) |
| `span_cgrpo_safe` | IMBANG + C-GRPO normalisation |
| `span_nosafe` | GRPO + budget (no safe-side member) |
| `span_entropy` | GRPO + budget + entropy bonus ×6 |
| `span_abscost` | GRPO + budget + uncentred cost |
| `nscsd` | GRPO, no budget (NS-CSD in Fig. 2) |
| `grpo_vanilla` | GRPO (Fig. 2) |
| `ppo` | PPO + shield (equal budget) |
| `maxmc_only` | MaxMC curriculum without a budget (extra baseline, not in the letter's tables) |
| `Sentinel` | Sentinel's released `best_benchmark` checkpoints |

In the CSV files, `benchmark` is the composite score b, `severe` is severe outages per 500 windows,
`leakage` is the share of attack traffic delivered, and `quality` is service quality.

## Where each result comes from

| Item in the letter | Data | Produced by | Plotted / tabulated by |
|---|---|---|---|
| Fig. 2 (constraint cancellation, 80 checkpoints) | `results/cliff_probe_final_by_seed.csv` | `scripts/cliff_probe.py` (command below) | `scripts/icc_assets.py` |
| Table II (Protocol B, 8 scenarios) | `results/full/protocolB_by_seed.csv`; tests in `results/full/tests_B_span_vs_*.csv` | `scripts/evaluate.py` | `scripts/nl_assets.py`, `scripts/span_report.py` |
| Fig. 3 (operating curve over intensity) | `results/intensity_sweep_by_seed.csv` | `scripts/intensity_sweep.py` | `scripts/nl_assets.py` |
| Table III (ablation, red team, budget) | `results/full/protocol{B,C}_by_seed.csv`, `runs/inline/*/seed*/history.csv` | `scripts/evaluate.py`, `scripts/train.py` | `scripts/nl_assets.py` |
| Fig. 4 (training dynamics) | `runs/inline/{span,span_cgrpo,span_nosafe}/seed*/history.csv` | `scripts/train.py` | `scripts/nl_assets.py` |
| Fig. 5 (outages vs leakage) | `results/full/protocolB_by_seed.csv` | `scripts/evaluate.py` | `scripts/icc_assets.py` |
| Protocol A (Sentinel's own attackers) | `results/full/protocolA_by_seed.csv`; `results/full/tests_A_span_vs_*.csv` | `scripts/evaluate.py` | `scripts/span_report.py` |
| Red-team worst case (Protocol C) | `results/full/protocolC_by_seed.csv` | `scripts/evaluate.py` | `scripts/span_report.py` |
| ICMP operator playbook (limitations) | `results/human/` | `scripts/human_in_loop.py` | `results/human/HUMAN_IN_LOOP.md` |
| Port check against Sentinel's published table | `results/reproduction.csv`, `results/DIAGNOSTICS.md` | `scripts/diagnostics.py` | — |
| Tree fidelity and shield tuning (Interpretability) | `runs/inline/span/seed*/meta.json` (`tree_fidelity`, `champion_shield`) | `scripts/train.py` | `scripts/supp_assets.py` (Table S16) |
| Multi-step check of Proposition 2 (supplement, Table S5) | `results/safe_member_check_by_seed.csv` | `scripts/safe_member_check.py` | `scripts/supp_assets.py` |
| All supplementary tables and figures | files above | — | `scripts/supp_assets.py` |

`results/full/SPAN_RESULTS.md` collects all paired comparisons. It reports means with 95%
t-intervals, paired t-tests with Holm correction, and Wilcoxon tests.

## Setup

Python 3.10 or later; tested with Python 3.11, PyTorch 2.x (CPU only), NumPy 2.4, pandas 3.0,
scikit-learn 1.9, SciPy 1.17 and Matplotlib 3.11.

```bash
pip install -r requirements.txt
git clone https://github.com/AliAlfatemi/sentinel-ddos ../sentinel-ddos   # reference code and checkpoints
```

Sentinel's checkpoints are not redistributed here. The scripts expect a clone of its public
repository next to this folder; to use another location, set `SENTINEL_REPO=/path/to/sentinel-ddos`.
The comparison uses the released paper-mode run `outputs/run_20260523_200520_paper`.

## Reproducing the results

Run all commands from the top of this folder.

**1. Check the simulator port** (a few seconds):
```bash
python -m pytest -q tests
```

**2. Regenerate the tables and figures from the included results** (under a minute; no training):
```bash
python scripts/icc_assets.py      # Fig. 2, Fig. 5 -> paper/icc/figures/
python scripts/nl_assets.py       # Tables II-III, Figs. 3-4 -> paper/nl/
python scripts/span_report.py     # all paired tests -> results/full/SPAN_RESULTS.md
python scripts/supp_assets.py     # Tables S1-S17, Figs. S1-S3 -> paper/nl/supp/
```

**3. Re-evaluate the included checkpoints** (no training):
```bash
python scripts/evaluate.py --out results/full --methods span span_nosafe span_cgrpo span_cgrpo_safe \
       span_entropy span_abscost nscsd ppo maxmc_only      # Protocols A, B, C
python scripts/intensity_sweep.py                           # Fig. 3
python scripts/human_in_loop.py                             # ICMP operator playbook
python scripts/safe_member_check.py                         # multi-step check of Proposition 2
FULL=1 python scripts/cliff_probe.py span span_nosafe span_cgrpo span_cgrpo_safe \
       nscsd grpo_vanilla ppo                                # Fig. 2 (adds Sentinel automatically)
```

All evaluations use common random numbers, so for a given seed and scenario every defender faces
identical traffic and the results are deterministic.

**4. Retrain from scratch** (optional). One run of 500 iterations takes 3–5 minutes on a single CPU
core (the logged training time is 3.1–4.3 min):
```bash
python scripts/train.py --method span --seed 42             # one IMBANG run -> runs/inline/span/seed42
MAIN="span span_nosafe span_cgrpo span_cgrpo_safe span_entropy span_abscost nscsd ppo grpo_vanilla maxmc_only" \
ABL="" JOBS=4 bash scripts/run_all.sh                        # all reported methods, seeds 42-51
```
Retraining overwrites the checkpoints in `runs/inline/`. The full grid is 100 runs, about 6–8
CPU-hours.

## Settings

All settings were fixed before the ten-seed runs:
- **Seeds:** 42–51.
- **Training:** 500 iterations; B = 32 contexts and G = 8 members per group; horizon H = 8;
  validation every K = 20 iterations.
- **Optimiser:** Adam, learning rate 3e-4; clipping 0.2; KL weight 0.02; entropy weight 0.005.
- **Outage budget:** κ = 0.05; multiplier step η_λ = 4.
- **Safe-side member:** u ~ U(0.2, 0.8).
- **Held-out traffic:** ICMP is never seen in training or model selection.

## Scope

All results come from Sentinel's simulator, not from production traffic. The port of the simulator
and shield is derived from Sentinel's MIT-licensed code; see `NOTICE`.
