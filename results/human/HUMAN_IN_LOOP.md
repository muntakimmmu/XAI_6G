# Simulated human-in-the-loop discrete decisions (10 seeds, 20 episodes each)

Operator reviews every 10 steps, acts after 3 steps, recognises the family with accuracy `acc`. `icmp-only`: ICMP playbook only; `quality-first`: also CONSERVATIVE on low-rate floods. Severe per 500 steps.


## icmp

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.476 ± 0.006 | 0.528 | 0.750 | 4.2 | 4.2 |
| SPAN | icmp-only | 0.9 | -0.480 ± 0.006 | 0.529 | 0.756 | 2.1 | 2.2 |
| SPAN | icmp-only | 1.0 | -0.481 ± 0.006 | 0.529 | 0.760 | 0.5 | 1.0 |
| SPAN | none | nan | -0.463 ± 0.011 | 0.525 | 0.720 | 17.9 | 0.0 |
| SPAN | quality-first | 0.7 | -0.476 ± 0.006 | 0.528 | 0.750 | 4.2 | 4.2 |
| SPAN | quality-first | 0.9 | -0.480 ± 0.006 | 0.529 | 0.756 | 2.1 | 2.2 |
| SPAN | quality-first | 1.0 | -0.481 ± 0.006 | 0.529 | 0.760 | 0.5 | 1.0 |
| Sentinel | icmp-only | 0.7 | -0.481 ± 0.006 | 0.529 | 0.758 | 1.7 | 4.2 |
| Sentinel | icmp-only | 0.9 | -0.481 ± 0.006 | 0.529 | 0.760 | 0.7 | 2.2 |
| Sentinel | icmp-only | 1.0 | -0.482 ± 0.006 | 0.529 | 0.760 | 0.3 | 1.0 |
| Sentinel | none | nan | -0.480 ± 0.007 | 0.528 | 0.751 | 6.9 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.481 ± 0.006 | 0.529 | 0.758 | 1.7 | 4.2 |
| Sentinel | quality-first | 0.9 | -0.481 ± 0.006 | 0.529 | 0.760 | 0.7 | 2.2 |
| Sentinel | quality-first | 1.0 | -0.482 ± 0.006 | 0.529 | 0.760 | 0.3 | 1.0 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.000, p=0.86

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -4.775, p=0.49

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.001, p=0.45

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff +0.005, p=0.49

## mild

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | 0.034 ± 0.005 | 0.983 | 0.949 | 0.0 | 0.9 |
| SPAN | icmp-only | 0.9 | 0.035 ± 0.005 | 0.982 | 0.948 | 0.0 | 0.3 |
| SPAN | icmp-only | 1.0 | 0.035 ± 0.005 | 0.982 | 0.947 | 0.0 | 0.0 |
| SPAN | none | nan | 0.035 ± 0.005 | 0.982 | 0.947 | 0.0 | 0.0 |
| SPAN | quality-first | 0.7 | 0.008 ± 0.001 | 0.996 | 0.988 | 0.0 | 2.6 |
| SPAN | quality-first | 0.9 | 0.007 ± 0.001 | 0.997 | 0.990 | 0.0 | 1.6 |
| SPAN | quality-first | 1.0 | 0.006 ± 0.001 | 0.997 | 0.991 | 0.0 | 1.0 |
| Sentinel | icmp-only | 0.7 | 0.006 ± 0.002 | 0.991 | 0.985 | 0.0 | 0.9 |
| Sentinel | icmp-only | 0.9 | 0.006 ± 0.002 | 0.991 | 0.985 | 0.0 | 0.3 |
| Sentinel | icmp-only | 1.0 | 0.006 ± 0.002 | 0.991 | 0.985 | 0.0 | 0.0 |
| Sentinel | none | nan | 0.006 ± 0.002 | 0.991 | 0.985 | 0.0 | 0.0 |
| Sentinel | quality-first | 0.7 | 0.003 ± 0.002 | 0.997 | 0.994 | 0.0 | 2.6 |
| Sentinel | quality-first | 0.9 | 0.002 ± 0.001 | 0.998 | 0.995 | 0.0 | 1.6 |
| Sentinel | quality-first | 1.0 | 0.002 ± 0.001 | 0.998 | 0.996 | 0.0 | 1.0 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.028, p=2.8e-07

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff +0.000, p=1

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff -0.009, p=0.00014

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.037, p=6.3e-07

## boundary

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | 0.062 ± 0.014 | 0.905 | 0.842 | 0.0 | 0.8 |
| SPAN | icmp-only | 0.9 | 0.064 ± 0.015 | 0.905 | 0.840 | 0.0 | 0.3 |
| SPAN | icmp-only | 1.0 | 0.065 ± 0.014 | 0.905 | 0.840 | 0.0 | 0.0 |
| SPAN | none | nan | 0.065 ± 0.014 | 0.905 | 0.840 | 0.0 | 0.0 |
| SPAN | quality-first | 0.7 | -0.019 ± 0.006 | 0.945 | 0.964 | 0.0 | 2.5 |
| SPAN | quality-first | 0.9 | -0.023 ± 0.006 | 0.948 | 0.971 | 0.0 | 1.5 |
| SPAN | quality-first | 1.0 | -0.024 ± 0.006 | 0.950 | 0.974 | 0.0 | 1.0 |
| Sentinel | icmp-only | 0.7 | -0.001 ± 0.018 | 0.908 | 0.907 | 0.0 | 0.8 |
| Sentinel | icmp-only | 0.9 | -0.000 ± 0.019 | 0.908 | 0.907 | 0.0 | 0.3 |
| Sentinel | icmp-only | 1.0 | -0.001 ± 0.019 | 0.908 | 0.907 | 0.0 | 0.0 |
| Sentinel | none | nan | -0.001 ± 0.019 | 0.908 | 0.907 | 0.0 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.030 ± 0.007 | 0.943 | 0.973 | 0.0 | 2.5 |
| Sentinel | quality-first | 0.9 | -0.031 ± 0.007 | 0.946 | 0.977 | 0.0 | 1.5 |
| Sentinel | quality-first | 1.0 | -0.032 ± 0.007 | 0.947 | 0.980 | 0.0 | 1.0 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.065, p=8.1e-07

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff +0.000, p=1

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff -0.003, p=0.42

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.067, p=1.2e-06

## icmp_chaos

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.298 ± 0.031 | 0.704 | 0.817 | 22.4 | 2.8 |
| SPAN | icmp-only | 0.9 | -0.298 ± 0.031 | 0.705 | 0.818 | 22.0 | 2.3 |
| SPAN | icmp-only | 1.0 | -0.298 ± 0.031 | 0.705 | 0.819 | 21.9 | 2.0 |
| SPAN | none | nan | -0.286 ± 0.033 | 0.704 | 0.807 | 21.1 | 0.0 |
| SPAN | quality-first | 0.7 | -0.307 ± 0.029 | 0.706 | 0.829 | 22.2 | 3.7 |
| SPAN | quality-first | 0.9 | -0.306 ± 0.028 | 0.707 | 0.829 | 21.7 | 2.9 |
| SPAN | quality-first | 1.0 | -0.305 ± 0.028 | 0.707 | 0.829 | 21.6 | 2.5 |
| Sentinel | icmp-only | 0.7 | -0.334 ± 0.032 | 0.698 | 0.838 | 28.2 | 2.8 |
| Sentinel | icmp-only | 0.9 | -0.334 ± 0.032 | 0.699 | 0.839 | 28.2 | 2.3 |
| Sentinel | icmp-only | 1.0 | -0.334 ± 0.032 | 0.699 | 0.839 | 28.1 | 2.0 |
| Sentinel | none | nan | -0.332 ± 0.034 | 0.698 | 0.836 | 28.2 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.340 ± 0.031 | 0.701 | 0.849 | 27.9 | 3.7 |
| Sentinel | quality-first | 0.9 | -0.339 ± 0.030 | 0.702 | 0.848 | 27.9 | 2.9 |
| Sentinel | quality-first | 1.0 | -0.338 ± 0.030 | 0.702 | 0.848 | 27.7 | 2.5 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.035, p=0.00079

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -6.200, p=0.002

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.006, p=0.0015

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.019, p=0.058

## strong

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.599 ± 0.012 | 0.520 | 0.822 | 47.0 | 0.8 |
| SPAN | icmp-only | 0.9 | -0.591 ± 0.011 | 0.520 | 0.820 | 41.2 | 0.3 |
| SPAN | icmp-only | 1.0 | -0.586 ± 0.012 | 0.521 | 0.819 | 37.9 | 0.0 |
| SPAN | none | nan | -0.586 ± 0.012 | 0.521 | 0.819 | 37.9 | 0.0 |
| SPAN | quality-first | 0.7 | -0.599 ± 0.012 | 0.520 | 0.822 | 47.0 | 0.8 |
| SPAN | quality-first | 0.9 | -0.591 ± 0.011 | 0.520 | 0.820 | 41.2 | 0.3 |
| SPAN | quality-first | 1.0 | -0.586 ± 0.012 | 0.521 | 0.819 | 37.9 | 0.0 |
| Sentinel | icmp-only | 0.7 | -0.756 ± 0.055 | 0.510 | 0.854 | 161.8 | 0.8 |
| Sentinel | icmp-only | 0.9 | -0.754 ± 0.056 | 0.510 | 0.853 | 160.6 | 0.3 |
| Sentinel | icmp-only | 1.0 | -0.753 ± 0.058 | 0.510 | 0.853 | 159.6 | 0.0 |
| Sentinel | none | nan | -0.753 ± 0.058 | 0.510 | 0.853 | 159.6 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.756 ± 0.055 | 0.510 | 0.854 | 161.8 | 0.8 |
| Sentinel | quality-first | 0.9 | -0.754 ± 0.056 | 0.510 | 0.853 | 160.6 | 0.3 |
| Sentinel | quality-first | 1.0 | -0.753 ± 0.058 | 0.510 | 0.853 | 159.6 | 0.0 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.162, p=5.3e-05

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -118.350, p=0.00032

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.010, p=0.0009

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.033, p=0.013

## chaos

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.125 ± 0.034 | 0.838 | 0.868 | 12.1 | 0.8 |
| SPAN | icmp-only | 0.9 | -0.124 ± 0.034 | 0.838 | 0.868 | 11.8 | 0.3 |
| SPAN | icmp-only | 1.0 | -0.123 ± 0.034 | 0.838 | 0.867 | 11.8 | 0.0 |
| SPAN | none | nan | -0.123 ± 0.034 | 0.838 | 0.867 | 11.8 | 0.0 |
| SPAN | quality-first | 0.7 | -0.159 ± 0.029 | 0.851 | 0.918 | 10.7 | 3.4 |
| SPAN | quality-first | 0.9 | -0.160 ± 0.028 | 0.852 | 0.920 | 10.3 | 2.6 |
| SPAN | quality-first | 1.0 | -0.160 ± 0.028 | 0.852 | 0.921 | 10.3 | 2.2 |
| Sentinel | icmp-only | 0.7 | -0.177 ± 0.032 | 0.837 | 0.911 | 17.2 | 0.8 |
| Sentinel | icmp-only | 0.9 | -0.177 ± 0.032 | 0.837 | 0.911 | 17.2 | 0.3 |
| Sentinel | icmp-only | 1.0 | -0.177 ± 0.032 | 0.837 | 0.911 | 17.3 | 0.0 |
| Sentinel | none | nan | -0.177 ± 0.032 | 0.837 | 0.911 | 17.3 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.186 ± 0.030 | 0.847 | 0.934 | 15.4 | 3.4 |
| Sentinel | quality-first | 0.9 | -0.187 ± 0.030 | 0.848 | 0.936 | 15.2 | 2.6 |
| Sentinel | quality-first | 1.0 | -0.188 ± 0.030 | 0.848 | 0.937 | 15.4 | 2.2 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.053, p=1.8e-06

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -5.500, p=0.083

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.001, p=0.69

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.043, p=0.00019

## pulse

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.230 ± 0.011 | 0.762 | 0.848 | 19.6 | 0.9 |
| SPAN | icmp-only | 0.9 | -0.226 ± 0.011 | 0.763 | 0.847 | 16.4 | 0.3 |
| SPAN | icmp-only | 1.0 | -0.224 ± 0.011 | 0.763 | 0.847 | 15.1 | 0.0 |
| SPAN | none | nan | -0.224 ± 0.011 | 0.763 | 0.847 | 15.1 | 0.0 |
| SPAN | quality-first | 0.7 | -0.240 ± 0.011 | 0.763 | 0.861 | 18.0 | 2.4 |
| SPAN | quality-first | 0.9 | -0.230 ± 0.011 | 0.763 | 0.852 | 15.8 | 1.0 |
| SPAN | quality-first | 1.0 | -0.224 ± 0.011 | 0.763 | 0.847 | 15.1 | 0.0 |
| Sentinel | icmp-only | 0.7 | -0.303 ± 0.024 | 0.759 | 0.875 | 61.6 | 0.9 |
| Sentinel | icmp-only | 0.9 | -0.303 ± 0.024 | 0.759 | 0.876 | 60.7 | 0.3 |
| Sentinel | icmp-only | 1.0 | -0.302 ± 0.024 | 0.759 | 0.876 | 60.2 | 0.0 |
| Sentinel | none | nan | -0.302 ± 0.024 | 0.759 | 0.876 | 60.2 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.305 ± 0.022 | 0.760 | 0.885 | 54.8 | 2.4 |
| Sentinel | quality-first | 0.9 | -0.304 ± 0.023 | 0.759 | 0.879 | 58.5 | 1.0 |
| Sentinel | quality-first | 1.0 | -0.302 ± 0.024 | 0.759 | 0.876 | 60.2 | 0.0 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.077, p=4.6e-05

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -43.775, p=0.0003

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.004, p=0.0016

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.029, p=0.019

## polymorph

| defender | operator | acc | composite | quality | leakage | severe | switches/ep |
|---|---|---|---|---|---|---|---|
| SPAN | icmp-only | 0.7 | -0.425 ± 0.007 | 0.664 | 0.889 | 13.6 | 0.8 |
| SPAN | icmp-only | 0.9 | -0.424 ± 0.007 | 0.664 | 0.889 | 13.5 | 0.3 |
| SPAN | icmp-only | 1.0 | -0.424 ± 0.007 | 0.664 | 0.889 | 13.7 | 0.0 |
| SPAN | none | nan | -0.424 ± 0.007 | 0.664 | 0.889 | 13.7 | 0.0 |
| SPAN | quality-first | 0.7 | -0.426 ± 0.006 | 0.672 | 0.906 | 11.1 | 3.7 |
| SPAN | quality-first | 0.9 | -0.426 ± 0.006 | 0.673 | 0.907 | 10.8 | 3.4 |
| SPAN | quality-first | 1.0 | -0.425 ± 0.007 | 0.673 | 0.907 | 10.8 | 3.2 |
| Sentinel | icmp-only | 0.7 | -0.448 ± 0.007 | 0.659 | 0.904 | 14.9 | 0.8 |
| Sentinel | icmp-only | 0.9 | -0.449 ± 0.007 | 0.659 | 0.904 | 14.9 | 0.3 |
| Sentinel | icmp-only | 1.0 | -0.449 ± 0.007 | 0.659 | 0.904 | 15.0 | 0.0 |
| Sentinel | none | nan | -0.449 ± 0.007 | 0.659 | 0.904 | 15.0 | 0.0 |
| Sentinel | quality-first | 0.7 | -0.446 ± 0.007 | 0.668 | 0.918 | 12.4 | 3.7 |
| Sentinel | quality-first | 0.9 | -0.447 ± 0.007 | 0.669 | 0.920 | 12.3 | 3.4 |
| Sentinel | quality-first | 1.0 | -0.446 ± 0.007 | 0.669 | 0.920 | 11.9 | 3.2 |

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), benchmark: diff +0.025, p=5.7e-06

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), severe: diff -1.500, p=0.65

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), quality: diff +0.005, p=0.095

SPAN+operator(icmp-only, acc 0.9) vs Sentinel (no operator), leakage: diff -0.015, p=0.036
