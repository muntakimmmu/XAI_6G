# SPAN: 10-seed results

Seeds 42-51, paired by seed; mean ± 95% t-CI; severe = steps per 500. Holm correction across scenarios per metric.


## Protocol B: overview (mean over scenarios, then over seeds)

| Defender | composite | severe | leakage | quality | n |
|---|---|---|---|---|---|
| Sentinel (released ckpt) | -0.290 ± 0.055 | 32.7 ± 37.2 | 0.862 ± 0.079 | 0.738 ± 0.010 | 3 |
| SPAN | -0.242 ± 0.023 | 17.5 ± 14.1 | 0.828 ± 0.021 | 0.736 ± 0.004 | 3 |
| Constrained GRPO (channel-normalised cost) | -0.240 ± 0.009 | 18.2 ± 15.3 | 0.818 ± 0.015 | 0.730 ± 0.007 | 3 |
| SPAN w/o safe-side member | -0.249 ± 0.027 | 30.6 ± 43.8 | 0.811 ± 0.019 | 0.728 ± 0.002 | 3 |
| MaxMC curriculum, no budget | -0.242 ± 0.017 | 33.7 ± 37.9 | 0.799 ± 0.048 | 0.725 ± 0.015 | 3 |
| NS-CSD (original) | -0.251 ± 0.038 | 92.9 ± 90.0 | 0.728 ± 0.129 | 0.710 ± 0.040 | 3 |
| PPO+shield control | -0.276 ± 0.008 | 17.3 ± 32.7 | 0.869 ± 0.044 | 0.742 ± 0.013 | 3 |

### Protocol B: SPAN vs Sentinel (released ckpt) (paired)

| scenario | metric | SPAN | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild | benchmark | 0.037 | 0.007 | +0.031 | 0.0072 | 0.057 | 0.25 | +6.79 |
| mild | quality | 0.982 | 0.994 | -0.011 | 0.019 | 0.16 | 0.25 | -4.09 |
| mild | leakage | 0.945 | 0.987 | -0.042 | 0.0087 | 0.069 | 0.25 | -6.15 |
| mild | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong | benchmark | -0.570 | -0.699 | +0.128 | 0.081 | 0.45 | 0.25 | +1.91 |
| strong | quality | 0.520 | 0.512 | +0.008 | 0.11 | 0.66 | 0.25 | +1.59 |
| strong | leakage | 0.810 | 0.826 | -0.015 | 0.71 | 1 | 0.75 | -0.25 |
| strong | severe | 30.417 | 135.417 | -105.000 | 0.087 | 0.7 | 0.25 | -1.83 |
| chaos | benchmark | -0.136 | -0.170 | +0.035 | 0.076 | 0.45 | 0.25 | +1.98 |
| chaos | quality | 0.825 | 0.827 | -0.002 | 0.2 | 0.82 | 0.5 | -1.07 |
| chaos | leakage | 0.860 | 0.893 | -0.033 | 0.13 | 0.75 | 0.25 | -1.47 |
| chaos | severe | 12.250 | 14.583 | -2.333 | 0.4 | 1 | 0.5 | -0.61 |
| icmp | benchmark | -0.454 | -0.468 | +0.014 | 0.41 | 0.72 | 0.5 | +0.60 |
| icmp | quality | 0.522 | 0.527 | -0.005 | 0.26 | 0.82 | 0.5 | -0.89 |
| icmp | leakage | 0.681 | 0.731 | -0.049 | 0.25 | 1 | 0.5 | -0.91 |
| icmp | severe | 44.083 | 14.500 | +29.583 | 0.39 | 1 | 0.5 | +0.63 |
| pulse | benchmark | -0.227 | -0.285 | +0.058 | 0.16 | 0.49 | 0.25 | +1.25 |
| pulse | quality | 0.762 | 0.759 | +0.003 | 0.14 | 0.71 | 0.25 | +1.36 |
| pulse | leakage | 0.846 | 0.861 | -0.015 | 0.69 | 1 | 0.75 | -0.26 |
| pulse | severe | 17.667 | 57.333 | -39.667 | 0.21 | 1 | 0.25 | -1.07 |
| polymorph | benchmark | -0.438 | -0.454 | +0.015 | 0.04 | 0.28 | 0.25 | +2.80 |
| polymorph | quality | 0.657 | 0.653 | +0.004 | 0.64 | 1 | 0.75 | +0.31 |
| polymorph | leakage | 0.888 | 0.896 | -0.009 | 0.73 | 1 | 0.75 | -0.23 |
| polymorph | severe | 16.500 | 15.917 | +0.583 | 0.96 | 1 | 1 | +0.03 |
| boundary | benchmark | 0.080 | 0.015 | +0.064 | 0.079 | 0.45 | 0.25 | +1.93 |
| boundary | quality | 0.899 | 0.912 | -0.013 | 0.034 | 0.24 | 0.25 | -3.04 |
| boundary | leakage | 0.819 | 0.896 | -0.078 | 0.071 | 0.5 | 0.25 | -2.05 |
| boundary | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| icmp_chaos | benchmark | -0.230 | -0.269 | +0.038 | 0.36 | 0.72 | 0.5 | +0.68 |
| icmp_chaos | quality | 0.719 | 0.717 | +0.002 | 0.65 | 1 | 1 | +0.31 |
| icmp_chaos | leakage | 0.776 | 0.809 | -0.033 | 0.44 | 1 | 0.5 | -0.56 |
| icmp_chaos | severe | 19.417 | 23.750 | -4.333 | 0.4 | 1 | 0.5 | -0.61 |

Composite: significant wins 0, significant losses 0, n.s. 8 of 8 scenarios.

### Protocol B: SPAN vs Constrained GRPO (channel-normalised cost) (paired)

| scenario | metric | SPAN | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild | benchmark | 0.037 | 0.059 | -0.022 | 0.012 | 0.1 | 0.25 | -5.12 |
| mild | quality | 0.982 | 0.968 | +0.015 | 0.0056 | 0.039 | 0.25 | +7.67 |
| mild | leakage | 0.945 | 0.908 | +0.037 | 0.0091 | 0.073 | 0.25 | +6.01 |
| mild | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong | benchmark | -0.570 | -0.572 | +0.002 | 0.82 | 1 | 1 | +0.15 |
| strong | quality | 0.520 | 0.517 | +0.003 | 0.021 | 0.1 | 0.25 | +3.93 |
| strong | leakage | 0.810 | 0.798 | +0.012 | 0.3 | 0.74 | 0.5 | +0.81 |
| strong | severe | 30.417 | 41.000 | -10.583 | 0.034 | 0.27 | 0.25 | -3.05 |
| chaos | benchmark | -0.136 | -0.121 | -0.014 | 0.17 | 1 | 0.25 | -1.20 |
| chaos | quality | 0.825 | 0.817 | +0.008 | 0.01 | 0.062 | 0.25 | +5.62 |
| chaos | leakage | 0.860 | 0.834 | +0.025 | 0.065 | 0.39 | 0.25 | +2.15 |
| chaos | severe | 12.250 | 14.417 | -2.167 | 0.058 | 0.41 | 0.25 | -2.29 |
| icmp | benchmark | -0.454 | -0.478 | +0.024 | 0.27 | 1 | 0.5 | +0.87 |
| icmp | quality | 0.522 | 0.526 | -0.005 | 0.22 | 0.44 | 0.5 | -1.02 |
| icmp | leakage | 0.681 | 0.727 | -0.046 | 0.25 | 0.74 | 0.5 | -0.94 |
| icmp | severe | 44.083 | 26.583 | +17.500 | 0.23 | 0.93 | 0.5 | +1.00 |
| pulse | benchmark | -0.227 | -0.224 | -0.003 | 0.32 | 1 | 0.5 | -0.76 |
| pulse | quality | 0.762 | 0.761 | +0.001 | 0.055 | 0.17 | 0.25 | +2.36 |
| pulse | leakage | 0.846 | 0.835 | +0.011 | 0.079 | 0.39 | 0.25 | +1.93 |
| pulse | severe | 17.667 | 24.250 | -6.583 | 0.17 | 0.93 | 0.25 | -1.23 |
| polymorph | benchmark | -0.438 | -0.441 | +0.003 | 0.27 | 1 | 0.5 | +0.86 |
| polymorph | quality | 0.657 | 0.647 | +0.010 | 0.042 | 0.17 | 0.25 | +2.73 |
| polymorph | leakage | 0.888 | 0.871 | +0.016 | 0.1 | 0.41 | 0.25 | +1.67 |
| polymorph | severe | 16.500 | 19.250 | -2.750 | 0.16 | 0.93 | 0.25 | -1.29 |
| boundary | benchmark | 0.080 | 0.095 | -0.016 | 0.11 | 0.76 | 0.25 | -1.61 |
| boundary | quality | 0.899 | 0.888 | +0.011 | 0.001 | 0.0084 | 0.25 | +17.86 |
| boundary | leakage | 0.819 | 0.791 | +0.027 | 0.037 | 0.26 | 0.25 | +2.94 |
| boundary | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| icmp_chaos | benchmark | -0.230 | -0.238 | +0.008 | 0.62 | 1 | 0.75 | +0.34 |
| icmp_chaos | quality | 0.719 | 0.718 | +0.001 | 0.61 | 0.61 | 0.75 | +0.34 |
| icmp_chaos | leakage | 0.776 | 0.781 | -0.005 | 0.78 | 0.78 | 0.75 | -0.18 |
| icmp_chaos | severe | 19.417 | 19.833 | -0.417 | 0.3 | 0.93 | 0.5 | -0.80 |

Composite: significant wins 0, significant losses 0, n.s. 8 of 8 scenarios.

### Protocol B: Pareto relation to Sentinel (seed means; quality up, leakage down, severe down)

| scenario | SPAN (q, leak, sev) | Sentinel (q, leak, sev) | relation |
|---|---|---|---|
| mild | 0.982, 0.945, 0.0 | 0.994, 0.987, 0.0 | trade-off |
| strong | 0.520, 0.810, 30.4 | 0.512, 0.826, 135.4 | dominates |
| chaos | 0.825, 0.860, 12.2 | 0.827, 0.893, 14.6 | trade-off |
| icmp | 0.522, 0.681, 44.1 | 0.527, 0.731, 14.5 | trade-off |
| pulse | 0.762, 0.846, 17.7 | 0.759, 0.861, 57.3 | dominates |
| polymorph | 0.657, 0.888, 16.5 | 0.653, 0.896, 15.9 | trade-off |
| boundary | 0.899, 0.819, 0.0 | 0.912, 0.896, 0.0 | trade-off |
| icmp_chaos | 0.719, 0.776, 19.4 | 0.717, 0.809, 23.8 | dominates |

## Protocol A: overview (mean over scenarios, then over seeds)

| Defender | composite | severe | leakage | quality | n |
|---|---|---|---|---|---|
| Sentinel (released ckpt) | -0.294 ± 0.129 | 6.0 ± 5.9 | 0.796 ± 0.186 | 0.676 ± 0.087 | 3 |
| SPAN | -0.219 ± 0.025 | 8.2 ± 15.3 | 0.711 ± 0.216 | 0.667 ± 0.116 | 3 |
| Constrained GRPO (channel-normalised cost) | -0.204 ± 0.057 | 6.8 ± 5.2 | 0.692 ± 0.136 | 0.662 ± 0.099 | 3 |
| SPAN w/o safe-side member | -0.206 ± 0.082 | 6.1 ± 6.1 | 0.695 ± 0.171 | 0.663 ± 0.096 | 3 |
| MaxMC curriculum, no budget | -0.189 ± 0.033 | 24.4 ± 52.8 | 0.652 ± 0.110 | 0.655 ± 0.086 | 3 |
| NS-CSD (original) | -0.216 ± 0.086 | 97.8 ± 251.5 | 0.599 ± 0.098 | 0.648 ± 0.071 | 3 |
| PPO+shield control | -0.304 ± 0.172 | 6.5 ± 7.6 | 0.810 ± 0.045 | 0.680 ± 0.091 | 3 |

### Protocol A: SPAN vs Sentinel (released ckpt) (paired)

| scenario | metric | SPAN | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild_attack | benchmark | 0.209 | 0.067 | +0.142 | 0.05 | 0.2 | 0.25 | +2.47 |
| mild_attack | quality | 0.952 | 0.979 | -0.027 | 0.089 | 0.36 | 0.25 | -1.81 |
| mild_attack | leakage | 0.743 | 0.912 | -0.169 | 0.053 | 0.21 | 0.25 | -2.42 |
| mild_attack | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong_attack | benchmark | -0.436 | -0.490 | +0.054 | 0.45 | 0.9 | 0.5 | +0.54 |
| strong_attack | quality | 0.523 | 0.526 | -0.002 | 0.84 | 1 | 1 | -0.14 |
| strong_attack | leakage | 0.699 | 0.765 | -0.066 | 0.47 | 0.94 | 0.75 | -0.51 |
| strong_attack | severe | 10.000 | 0.000 | +10.000 | 0.42 | 1 | 1 | +0.58 |
| chaos_flash_crowd | benchmark | -0.253 | -0.311 | +0.058 | 0.46 | 0.9 | 0.75 | +0.52 |
| chaos_flash_crowd | quality | 0.669 | 0.671 | -0.001 | 0.93 | 1 | 1 | -0.06 |
| chaos_flash_crowd | leakage | 0.728 | 0.787 | -0.058 | 0.52 | 0.94 | 0.75 | -0.45 |
| chaos_flash_crowd | severe | 23.000 | 24.000 | -1.000 | 0.42 | 1 | 1 | -0.58 |
| zero_shot_icmp | benchmark | -0.399 | -0.441 | +0.043 | 0.26 | 0.78 | 0.5 | +0.89 |
| zero_shot_icmp | quality | 0.524 | 0.529 | -0.006 | 0.23 | 0.69 | 0.5 | -0.98 |
| zero_shot_icmp | leakage | 0.672 | 0.721 | -0.048 | 0.26 | 0.77 | 0.5 | -0.90 |
| zero_shot_icmp | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |

Composite: significant wins 0, significant losses 0, n.s. 4 of 4 scenarios.

### Protocol A: SPAN vs Constrained GRPO (channel-normalised cost) (paired)

| scenario | metric | SPAN | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild_attack | benchmark | 0.209 | 0.244 | -0.035 | 0.036 | 0.14 | 0.25 | -2.97 |
| mild_attack | quality | 0.952 | 0.941 | +0.011 | 0.013 | 0.051 | 0.25 | +5.06 |
| mild_attack | leakage | 0.743 | 0.697 | +0.046 | 0.019 | 0.077 | 0.25 | +4.09 |
| mild_attack | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong_attack | benchmark | -0.436 | -0.397 | -0.039 | 0.26 | 0.65 | 0.5 | -0.90 |
| strong_attack | quality | 0.523 | 0.519 | +0.004 | 0.47 | 0.81 | 0.5 | +0.51 |
| strong_attack | leakage | 0.699 | 0.665 | +0.035 | 0.45 | 0.84 | 0.5 | +0.53 |
| strong_attack | severe | 10.000 | 1.667 | +8.333 | 0.42 | 1 | 1 | +0.58 |
| chaos_flash_crowd | benchmark | -0.253 | -0.222 | -0.031 | 0.46 | 0.65 | 0.5 | -0.53 |
| chaos_flash_crowd | quality | 0.669 | 0.660 | +0.009 | 0.41 | 0.81 | 0.5 | +0.60 |
| chaos_flash_crowd | leakage | 0.728 | 0.685 | +0.043 | 0.42 | 0.84 | 0.5 | +0.58 |
| chaos_flash_crowd | severe | 23.000 | 25.667 | -2.667 | 0.42 | 1 | 1 | -0.58 |
| zero_shot_icmp | benchmark | -0.399 | -0.444 | +0.045 | 0.22 | 0.65 | 0.5 | +1.02 |
| zero_shot_icmp | quality | 0.524 | 0.528 | -0.004 | 0.24 | 0.72 | 0.5 | -0.95 |
| zero_shot_icmp | leakage | 0.672 | 0.722 | -0.049 | 0.22 | 0.66 | 0.5 | -1.02 |
| zero_shot_icmp | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |

Composite: significant wins 0, significant losses 0, n.s. 4 of 4 scenarios.

### Protocol A: Pareto relation to Sentinel (seed means; quality up, leakage down, severe down)

| scenario | SPAN (q, leak, sev) | Sentinel (q, leak, sev) | relation |
|---|---|---|---|
| mild_attack | 0.952, 0.743, 0.0 | 0.979, 0.912, 0.0 | trade-off |
| strong_attack | 0.523, 0.699, 10.0 | 0.526, 0.765, 0.0 | trade-off |
| chaos_flash_crowd | 0.669, 0.728, 23.0 | 0.671, 0.787, 24.0 | trade-off |
| zero_shot_icmp | 0.524, 0.672, 0.0 | 0.529, 0.721, 0.0 | trade-off |

## Protocol C: CEM red team, worst composite found (higher = more robust)

| Defender | worst composite | worst severe | n |
|---|---|---|---|
| Sentinel (released ckpt) | -0.889 ± 0.535 | 269.2 ± 488.0 | 3 |
| SPAN | -0.710 ± 0.033 | 57.8 ± 85.4 | 3 |
| Constrained GRPO (channel-normalised cost) | -0.707 ± 0.079 | 39.1 ± 38.2 | 3 |
| SPAN w/o safe-side member | -0.853 ± 0.562 | 193.6 ± 583.7 | 3 |
| MaxMC curriculum, no budget | -0.817 ± 0.243 | 175.5 ± 313.4 | 3 |
| NS-CSD (original) | -1.080 ± 0.281 | 489.7 ± 22.6 | 3 |
| PPO+shield control | -0.831 ± 0.535 | 167.0 ± 509.1 | 3 |

SPAN vs Sentinel (released ckpt): diff +0.179, paired-t p=0.31, Wilcoxon p=0.25

SPAN vs Constrained GRPO (channel-normalised cost): diff -0.003, paired-t p=0.9, Wilcoxon p=1

## Mechanism (training logs): group span rate, validation severe rate, final multiplier

| Variant | span rate (last 100 it) | val severe rate | final lambda | budget met (<=0.05) | n |
|---|---|---|---|---|---|
| MaxMC curriculum, no budget | 0.417 ± 0.071 | 0.137 ± 0.036 | 0.00 ± 0.00 | 0/10 | 10 |
| SPAN | 0.423 ± 0.046 | 0.029 ± 0.015 | 0.43 ± 0.81 | 9/10 | 10 |
| uncentred cost advantage | 0.441 ± 0.037 | 0.081 ± 0.029 | 7.38 ± 1.75 | 1/10 | 10 |
| Constrained GRPO (channel-normalised cost) | 0.363 ± 0.058 | 0.077 ± 0.031 | 4.67 ± 2.15 | 3/10 | 10 |
| SPAN + channel-normalised cost | 0.415 ± 0.035 | 0.040 ± 0.021 | 0.41 ± 0.57 | 7/10 | 10 |
| entropy bonus x6 (no safe member) | 0.430 ± 0.040 | 0.120 ± 0.030 | 7.54 ± 2.16 | 0/10 | 10 |
| SPAN w/o safe-side member | 0.427 ± 0.044 | 0.106 ± 0.039 | 7.69 ± 2.30 | 1/10 | 10 |
