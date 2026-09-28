# Testbed diagnostics

## 1. Reproduction of Sentinel's published results on the vectorised port

Protocol A, released benchmark checkpoints, 10 seeds; `_pub` = Sentinel's evaluation_benchmark_summary.csv.

|                                                |   quality |   leakage |   severe |   quality_pub |   leakage_pub |   severe_pub |
|:-----------------------------------------------|----------:|----------:|---------:|--------------:|--------------:|-------------:|
| ('benign', 'PPO_Ablation_NoShield')            |     1     |     0     |      0   |         1     |         0     |          0   |
| ('benign', 'Sentinel')                         |     1     |     0     |      0   |         1     |         0     |          0   |
| ('chaos_flash_crowd', 'PPO_Ablation_NoShield') |     0.701 |     0.834 |     22   |         0.699 |         0.834 |         25.3 |
| ('chaos_flash_crowd', 'Sentinel')              |     0.701 |     0.836 |     22   |         0.699 |         0.836 |         25.3 |
| ('mild_attack', 'PPO_Ablation_NoShield')       |     0.971 |     0.92  |      0   |         0.971 |         0.92  |          0   |
| ('mild_attack', 'Sentinel')                    |     0.971 |     0.921 |      0   |         0.971 |         0.921 |          0   |
| ('strong_attack', 'PPO_Ablation_NoShield')     |     0.518 |     0.814 |     79.3 |         0.519 |         0.814 |         76.1 |
| ('strong_attack', 'Sentinel')                  |     0.519 |     0.815 |     78.2 |         0.519 |         0.815 |         75   |
| ('zero_shot_icmp', 'PPO_Ablation_NoShield')    |     0.514 |     0.833 |    146.8 |         0.514 |         0.833 |        145.7 |
| ('zero_shot_icmp', 'Sentinel')                 |     0.529 |     0.747 |      0   |         0.529 |         0.747 |          0   |

## 2. Headroom: privileged per-step oracle vs Sentinel (Protocol B scenarios, seeds 42-44, 10 eps)

Leakage here is the all-step mean. The oracle knows the attack family, intensity and mutation.

|                               |   panicguard_drop_per_500 |   quality |   leakage |   severe |   benchmark |
|:------------------------------|--------------------------:|----------:|----------:|---------:|------------:|
| ('boundary', 'Sentinel')      |                     5     |     0.913 |     0.905 |    0     |       0.007 |
| ('boundary', 'oracle')        |                   nan     |     0.84  |     0.661 |    0     |       0.179 |
| ('boundary', 'oracle+shield') |                   nan     |     0.84  |     0.671 |    0     |       0.169 |
| ('chaos', 'Sentinel')         |                   185.667 |     0.848 |     0.737 |   10.667 |       0.016 |
| ('chaos', 'oracle')           |                   nan     |     0.782 |     0.536 |    6.333 |       0.161 |
| ('chaos', 'oracle+shield')    |                   nan     |     0.806 |     0.591 |    6.333 |       0.13  |
| ('icmp', 'Sentinel')          |                     5     |     0.526 |     0.733 |   13.667 |      -0.47  |
| ('icmp', 'oracle')            |                   nan     |     0.504 |     0.522 |    0     |      -0.268 |
| ('icmp', 'oracle+shield')     |                   nan     |     0.505 |     0.527 |    0.5   |      -0.273 |
| ('mild', 'Sentinel')          |                   332.667 |     0.992 |     0.988 |    0     |       0.004 |
| ('mild', 'oracle')            |                   nan     |     0.86  |     0.697 |    0     |       0.163 |
| ('mild', 'oracle+shield')     |                   nan     |     0.953 |     0.896 |    0     |       0.057 |
| ('strong', 'Sentinel')        |                     5     |     0.51  |     0.835 |  142.833 |      -0.718 |
| ('strong', 'oracle')          |                   nan     |     0.502 |     0.689 |    0     |      -0.437 |
| ('strong', 'oracle+shield')   |                   nan     |     0.502 |     0.698 |    0     |      -0.446 |

Physics ceiling: with inline (pre-filter) overload the service quality of a full-intensity flood is at most 0.556 (1000/(1500+300)), independent of the defender.
