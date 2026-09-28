# STRADDLE: 10-seed results

Seeds 42-51, paired by seed; mean ± 95% t-CI; severe = steps per 500. Holm correction across scenarios per metric.


## Protocol B: overview (mean over scenarios, then over seeds)

| Defender | composite | severe | leakage | quality | n |
|---|---|---|---|---|---|
| Sentinel (released ckpt) | -0.304 ± 0.014 | 34.4 ± 9.5 | 0.876 ± 0.013 | 0.738 ± 0.003 | 10 |
| STRADDLE | -0.246 ± 0.009 | 14.8 ± 3.3 | 0.839 ± 0.013 | 0.739 ± 0.003 | 10 |
| Constrained GRPO (channel-normalised cost) | -0.248 ± 0.009 | 26.6 ± 17.8 | 0.820 ± 0.016 | 0.731 ± 0.005 | 10 |
| STRADDLE w/o safe-side member | -0.253 ± 0.011 | 36.5 ± 19.9 | 0.812 ± 0.018 | 0.729 ± 0.004 | 10 |
| entropy bonus x6 (no safe member) | -0.249 ± 0.010 | 33.8 ± 19.4 | 0.810 ± 0.019 | 0.729 ± 0.005 | 10 |
| NS-CSD (original) | -0.250 ± 0.009 | 91.5 ± 24.2 | 0.729 ± 0.034 | 0.710 ± 0.009 | 10 |
| PPO+shield control | -0.276 ± 0.006 | 19.6 ± 6.8 | 0.867 ± 0.008 | 0.742 ± 0.002 | 10 |

### Protocol B: STRADDLE vs Sentinel (released ckpt) (paired)

| scenario | metric | STRADDLE | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild | benchmark | 0.040 | 0.009 | +0.031 | 1.3e-07 | 1.1e-06 | 0.002 | +4.65 |
| mild | quality | 0.981 | 0.991 | -0.010 | 5.3e-05 | 0.00042 | 0.002 | -2.27 |
| mild | leakage | 0.941 | 0.983 | -0.041 | 5.5e-07 | 4.4e-06 | 0.002 | -3.95 |
| mild | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong | benchmark | -0.583 | -0.746 | +0.164 | 6.7e-05 | 0.00027 | 0.002 | +2.20 |
| strong | quality | 0.520 | 0.511 | +0.010 | 0.0022 | 0.013 | 0.0059 | +1.34 |
| strong | leakage | 0.814 | 0.852 | -0.038 | 0.012 | 0.06 | 0.027 | -0.99 |
| strong | severe | 38.875 | 155.225 | -116.350 | 0.00044 | 0.0035 | 0.0039 | -1.71 |
| chaos | benchmark | -0.123 | -0.169 | +0.046 | 2.1e-05 | 0.00013 | 0.002 | +2.55 |
| chaos | quality | 0.838 | 0.837 | +0.002 | 0.44 | 0.44 | 0.49 | +0.26 |
| chaos | leakage | 0.870 | 0.908 | -0.038 | 2.4e-05 | 0.00017 | 0.002 | -2.51 |
| chaos | severe | 10.725 | 14.125 | -3.400 | 0.016 | 0.093 | 0.037 | -0.94 |
| icmp | benchmark | -0.458 | -0.474 | +0.016 | 0.036 | 0.036 | 0.037 | +0.78 |
| icmp | quality | 0.526 | 0.529 | -0.003 | 0.033 | 0.13 | 0.084 | -0.79 |
| icmp | leakage | 0.716 | 0.748 | -0.032 | 0.029 | 0.064 | 0.049 | -0.82 |
| icmp | severe | 18.000 | 4.575 | +13.425 | 0.14 | 0.57 | 0.078 | +0.51 |
| pulse | benchmark | -0.225 | -0.303 | +0.078 | 0.00017 | 0.00052 | 0.002 | +1.94 |
| pulse | quality | 0.763 | 0.759 | +0.004 | 0.00063 | 0.0044 | 0.0039 | +1.62 |
| pulse | leakage | 0.847 | 0.877 | -0.030 | 0.016 | 0.064 | 0.049 | -0.94 |
| pulse | severe | 16.050 | 60.550 | -44.500 | 0.00084 | 0.0059 | 0.0039 | -1.55 |
| polymorph | benchmark | -0.430 | -0.454 | +0.024 | 4.2e-05 | 0.00021 | 0.002 | +2.33 |
| polymorph | quality | 0.662 | 0.657 | +0.005 | 0.087 | 0.26 | 0.064 | +0.61 |
| polymorph | leakage | 0.890 | 0.905 | -0.015 | 0.046 | 0.064 | 0.064 | -0.73 |
| polymorph | severe | 14.225 | 14.875 | -0.650 | 0.85 | 1 | 0.98 | -0.06 |
| boundary | benchmark | 0.071 | 0.003 | +0.068 | 8.7e-06 | 6.1e-05 | 0.002 | +2.84 |
| boundary | quality | 0.907 | 0.914 | -0.006 | 0.17 | 0.33 | 0.23 | -0.48 |
| boundary | leakage | 0.835 | 0.908 | -0.072 | 3.2e-05 | 0.00019 | 0.002 | -2.42 |
| boundary | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| icmp_chaos | benchmark | -0.257 | -0.299 | +0.042 | 0.003 | 0.0059 | 0.0059 | +1.27 |
| icmp_chaos | quality | 0.715 | 0.710 | +0.006 | 0.0086 | 0.043 | 0.02 | +1.06 |
| icmp_chaos | leakage | 0.800 | 0.829 | -0.029 | 0.02 | 0.064 | 0.027 | -0.89 |
| icmp_chaos | severe | 20.875 | 25.450 | -4.575 | 0.024 | 0.12 | 0.02 | -0.86 |

Composite: significant wins 8, significant losses 0, n.s. 0 of 8 scenarios.

### Protocol B: STRADDLE vs Constrained GRPO (channel-normalised cost) (paired)

| scenario | metric | STRADDLE | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild | benchmark | 0.040 | 0.053 | -0.013 | 0.036 | 0.28 | 0.049 | -0.78 |
| mild | quality | 0.981 | 0.970 | +0.011 | 0.0003 | 0.0024 | 0.002 | +1.80 |
| mild | leakage | 0.941 | 0.917 | +0.025 | 0.0075 | 0.06 | 0.014 | +1.09 |
| mild | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong | benchmark | -0.583 | -0.612 | +0.029 | 0.083 | 0.37 | 0.02 | +0.62 |
| strong | quality | 0.520 | 0.513 | +0.008 | 0.021 | 0.077 | 0.014 | +0.88 |
| strong | leakage | 0.814 | 0.789 | +0.025 | 0.099 | 0.3 | 0.064 | +0.58 |
| strong | severe | 38.875 | 85.900 | -47.025 | 0.082 | 0.66 | 0.037 | -0.62 |
| chaos | benchmark | -0.123 | -0.114 | -0.009 | 0.074 | 0.37 | 0.049 | -0.64 |
| chaos | quality | 0.838 | 0.829 | +0.010 | 0.0014 | 0.0099 | 0.002 | +1.43 |
| chaos | leakage | 0.870 | 0.843 | +0.027 | 0.0092 | 0.064 | 0.02 | +1.04 |
| chaos | severe | 10.725 | 16.775 | -6.050 | 0.2 | 0.81 | 0.0098 | -0.43 |
| icmp | benchmark | -0.458 | -0.474 | +0.016 | 0.058 | 0.35 | 0.11 | +0.69 |
| icmp | quality | 0.526 | 0.526 | -0.001 | 0.7 | 0.7 | 0.49 | -0.13 |
| icmp | leakage | 0.716 | 0.727 | -0.011 | 0.62 | 0.62 | 0.49 | -0.16 |
| icmp | severe | 18.000 | 23.300 | -5.300 | 0.75 | 1 | 0.65 | -0.10 |
| pulse | benchmark | -0.225 | -0.228 | +0.003 | 0.69 | 1 | 0.7 | +0.13 |
| pulse | quality | 0.763 | 0.760 | +0.003 | 0.019 | 0.077 | 0.014 | +0.90 |
| pulse | leakage | 0.847 | 0.825 | +0.022 | 0.054 | 0.22 | 0.049 | +0.70 |
| pulse | severe | 16.050 | 37.700 | -21.650 | 0.1 | 0.7 | 0.0059 | -0.58 |
| polymorph | benchmark | -0.430 | -0.442 | +0.012 | 0.045 | 0.32 | 0.0039 | +0.73 |
| polymorph | quality | 0.662 | 0.647 | +0.015 | 0.0096 | 0.057 | 0.0039 | +1.04 |
| polymorph | leakage | 0.890 | 0.868 | +0.022 | 0.033 | 0.16 | 0.02 | +0.80 |
| polymorph | severe | 14.225 | 24.000 | -9.775 | 0.16 | 0.8 | 0.049 | -0.48 |
| boundary | benchmark | 0.071 | 0.084 | -0.013 | 0.079 | 0.37 | 0.11 | -0.63 |
| boundary | quality | 0.907 | 0.893 | +0.014 | 0.01 | 0.057 | 0.0059 | +1.02 |
| boundary | leakage | 0.835 | 0.807 | +0.028 | 0.024 | 0.15 | 0.02 | +0.85 |
| boundary | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| icmp_chaos | benchmark | -0.257 | -0.255 | -0.002 | 0.84 | 1 | 1 | -0.06 |
| icmp_chaos | quality | 0.715 | 0.710 | +0.006 | 0.061 | 0.12 | 0.037 | +0.68 |
| icmp_chaos | leakage | 0.800 | 0.783 | +0.017 | 0.31 | 0.62 | 0.43 | +0.34 |
| icmp_chaos | severe | 20.875 | 25.175 | -4.300 | 0.1 | 0.7 | 0.0039 | -0.57 |

Composite: significant wins 0, significant losses 0, n.s. 8 of 8 scenarios.

### Protocol B: Pareto relation to Sentinel (seed means; quality up, leakage down, severe down)

| scenario | STRADDLE (q, leak, sev) | Sentinel (q, leak, sev) | relation |
|---|---|---|---|
| mild | 0.981, 0.941, 0.0 | 0.991, 0.983, 0.0 | trade-off |
| strong | 0.520, 0.814, 38.9 | 0.511, 0.852, 155.2 | dominates |
| chaos | 0.838, 0.870, 10.7 | 0.837, 0.908, 14.1 | dominates |
| icmp | 0.526, 0.716, 18.0 | 0.529, 0.748, 4.6 | trade-off |
| pulse | 0.763, 0.847, 16.1 | 0.759, 0.877, 60.5 | dominates |
| polymorph | 0.662, 0.890, 14.2 | 0.657, 0.905, 14.9 | dominates |
| boundary | 0.907, 0.835, 0.0 | 0.914, 0.908, 0.0 | trade-off |
| icmp_chaos | 0.715, 0.800, 20.9 | 0.710, 0.829, 25.4 | dominates |

## Protocol A: overview (mean over scenarios, then over seeds)

| Defender | composite | severe | leakage | quality | n |
|---|---|---|---|---|---|
| Sentinel (released ckpt) | -0.332 ± 0.062 | 25.1 ± 30.5 | 0.830 ± 0.043 | 0.680 ± 0.020 | 10 |
| STRADDLE | -0.265 ± 0.039 | 6.3 ± 3.1 | 0.778 ± 0.057 | 0.678 ± 0.028 | 10 |
| Constrained GRPO (channel-normalised cost) | -0.262 ± 0.071 | 20.9 ± 24.9 | 0.749 ± 0.058 | 0.666 ± 0.021 | 10 |
| STRADDLE w/o safe-side member | -0.248 ± 0.060 | 10.2 ± 7.0 | 0.746 ± 0.062 | 0.666 ± 0.023 | 10 |
| entropy bonus x6 (no safe member) | -0.244 ± 0.063 | 13.7 ± 10.0 | 0.736 ± 0.068 | 0.664 ± 0.022 | 10 |
| NS-CSD (original) | -0.276 ± 0.076 | 117.0 ± 55.7 | 0.642 ± 0.047 | 0.648 ± 0.021 | 10 |
| PPO+shield control | -0.315 ± 0.042 | 14.2 ± 16.9 | 0.827 ± 0.020 | 0.684 ± 0.023 | 10 |

### Protocol A: STRADDLE vs Sentinel (released ckpt) (paired)

| scenario | metric | STRADDLE | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild_attack | benchmark | 0.141 | 0.051 | +0.090 | 0.00076 | 0.003 | 0.002 | +1.57 |
| mild_attack | quality | 0.950 | 0.971 | -0.022 | 0.0057 | 0.023 | 0.02 | -1.14 |
| mild_attack | leakage | 0.808 | 0.921 | -0.112 | 0.00029 | 0.0012 | 0.002 | -1.81 |
| mild_attack | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong_attack | benchmark | -0.509 | -0.625 | +0.116 | 0.13 | 0.26 | 0.13 | +0.52 |
| strong_attack | quality | 0.524 | 0.519 | +0.006 | 0.42 | 0.53 | 0.56 | +0.27 |
| strong_attack | leakage | 0.778 | 0.815 | -0.037 | 0.23 | 0.47 | 0.56 | -0.40 |
| strong_attack | severe | 5.500 | 78.200 | -72.700 | 0.22 | 0.67 | 0.62 | -0.42 |
| chaos_flash_crowd | benchmark | -0.251 | -0.288 | +0.037 | 0.19 | 0.26 | 0.43 | +0.44 |
| chaos_flash_crowd | quality | 0.711 | 0.701 | +0.010 | 0.27 | 0.53 | 0.38 | +0.37 |
| chaos_flash_crowd | leakage | 0.811 | 0.836 | -0.025 | 0.4 | 0.47 | 0.77 | -0.28 |
| chaos_flash_crowd | severe | 19.800 | 22.000 | -2.200 | 0.038 | 0.15 | 0.062 | -0.77 |
| zero_shot_icmp | benchmark | -0.439 | -0.468 | +0.028 | 0.033 | 0.098 | 0.049 | +0.80 |
| zero_shot_icmp | quality | 0.526 | 0.529 | -0.003 | 0.026 | 0.079 | 0.064 | -0.84 |
| zero_shot_icmp | leakage | 0.715 | 0.747 | -0.032 | 0.032 | 0.095 | 0.049 | -0.80 |
| zero_shot_icmp | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |

Composite: significant wins 1, significant losses 0, n.s. 3 of 4 scenarios.

### Protocol A: STRADDLE vs Constrained GRPO (channel-normalised cost) (paired)

| scenario | metric | STRADDLE | other | diff | p (t) | Holm p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|---|---|
| mild_attack | benchmark | 0.141 | 0.154 | -0.013 | 0.53 | 1 | 0.084 | -0.21 |
| mild_attack | quality | 0.950 | 0.929 | +0.020 | 0.0094 | 0.036 | 0.002 | +1.04 |
| mild_attack | leakage | 0.808 | 0.775 | +0.033 | 0.082 | 0.16 | 0.064 | +0.62 |
| mild_attack | severe | 0.000 | 0.000 | +0.000 | 1 | 1 | 1 | +0.00 |
| strong_attack | benchmark | -0.509 | -0.531 | +0.021 | 0.66 | 1 | 0.92 | +0.14 |
| strong_attack | quality | 0.524 | 0.514 | +0.010 | 0.015 | 0.036 | 0.0059 | +0.95 |
| strong_attack | leakage | 0.778 | 0.735 | +0.043 | 0.049 | 0.15 | 0.037 | +0.72 |
| strong_attack | severe | 5.500 | 59.900 | -54.400 | 0.25 | 0.76 | 0.31 | -0.39 |
| chaos_flash_crowd | benchmark | -0.251 | -0.220 | -0.031 | 0.12 | 0.47 | 0.13 | -0.55 |
| chaos_flash_crowd | quality | 0.711 | 0.693 | +0.018 | 0.009 | 0.036 | 0.0059 | +1.05 |
| chaos_flash_crowd | leakage | 0.811 | 0.759 | +0.052 | 0.038 | 0.15 | 0.037 | +0.77 |
| chaos_flash_crowd | severe | 19.800 | 23.300 | -3.500 | 0.021 | 0.085 | 0.062 | -0.88 |
| zero_shot_icmp | benchmark | -0.439 | -0.452 | +0.012 | 0.54 | 1 | 0.49 | +0.20 |
| zero_shot_icmp | quality | 0.526 | 0.527 | -0.001 | 0.73 | 0.73 | 0.49 | -0.11 |
| zero_shot_icmp | leakage | 0.715 | 0.728 | -0.012 | 0.59 | 0.59 | 0.49 | -0.18 |
| zero_shot_icmp | severe | 0.000 | 0.600 | -0.600 | 0.34 | 0.76 | 1 | -0.32 |

Composite: significant wins 0, significant losses 0, n.s. 4 of 4 scenarios.

### Protocol A: Pareto relation to Sentinel (seed means; quality up, leakage down, severe down)

| scenario | STRADDLE (q, leak, sev) | Sentinel (q, leak, sev) | relation |
|---|---|---|---|
| mild_attack | 0.950, 0.808, 0.0 | 0.971, 0.921, 0.0 | trade-off |
| strong_attack | 0.524, 0.778, 5.5 | 0.519, 0.815, 78.2 | dominates |
| chaos_flash_crowd | 0.711, 0.811, 19.8 | 0.701, 0.836, 22.0 | dominates |
| zero_shot_icmp | 0.526, 0.715, 0.0 | 0.529, 0.747, 0.0 | trade-off |

## Protocol C: CEM red team, worst composite found (higher = more robust)

| Defender | worst composite | worst severe | n |
|---|---|---|---|
| Sentinel (released ckpt) | -1.071 ± 0.166 | 359.6 ± 127.3 | 10 |
| STRADDLE | -0.716 ± 0.032 | 67.8 ± 46.3 | 10 |
| Constrained GRPO (channel-normalised cost) | -0.804 ± 0.128 | 155.5 ± 121.3 | 10 |
| STRADDLE w/o safe-side member | -0.870 ± 0.148 | 213.0 ± 151.4 | 10 |
| entropy bonus x6 (no safe member) | -0.848 ± 0.160 | 208.7 ± 161.1 | 10 |
| NS-CSD (original) | -1.072 ± 0.076 | 466.6 ± 37.7 | 10 |
| PPO+shield control | -0.792 ± 0.085 | 141.4 ± 84.4 | 10 |

STRADDLE vs Sentinel (released ckpt): diff +0.355, paired-t p=0.00072, Wilcoxon p=0.002

STRADDLE vs Constrained GRPO (channel-normalised cost): diff +0.088, paired-t p=0.14, Wilcoxon p=0.38

## Mechanism (training logs): group straddle rate, validation severe rate, final multiplier

| Variant | straddle rate (last 100 it) | val severe rate | final lambda | budget met (<=0.05) | n |
|---|---|---|---|---|---|
| MaxMC curriculum, no budget | 0.396 ± 0.131 | 0.147 ± 0.061 | 0.00 ± 0.00 | 0/6 | 6 |
| STRADDLE | 0.423 ± 0.046 | 0.029 ± 0.015 | 0.43 ± 0.81 | 9/10 | 10 |
| uncentred cost advantage | 0.441 ± 0.037 | 0.081 ± 0.029 | 7.38 ± 1.75 | 1/10 | 10 |
| Constrained GRPO (channel-normalised cost) | 0.363 ± 0.058 | 0.077 ± 0.031 | 4.67 ± 2.15 | 3/10 | 10 |
| entropy bonus x6 (no safe member) | 0.430 ± 0.040 | 0.120 ± 0.030 | 7.54 ± 2.16 | 0/10 | 10 |
| STRADDLE w/o safe-side member | 0.427 ± 0.044 | 0.106 ± 0.039 | 7.69 ± 2.30 | 1/10 | 10 |
