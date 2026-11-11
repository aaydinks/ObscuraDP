# Phase 3 results: every way of adding the noise at equal total budgets

Built by `python -m benchmarks.phase3.report` from 960 stored runs. Every record was validated, every stored privacy report was recomputed from its recorded releases, and 120 runs of the local arm were compared with the stored Phase 2 runs.

All measures are on held-out rows at their natural mix. Recall, precision and F1 are given only at the cut-off chosen on validation rows. Settings were chosen on made-up tables (`tune.py`), never on these rows.

**Stroke table, AUC on held-out rows at their natural mix** (mean of 10 seeds ± sample standard deviation; columns are total budgets)

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | no protection (no budget) | 0.821 ± 0.036 |  |  |  |  |
| network | local noise (Phase 2 protocol) | 0.609 ± 0.130 | 0.689 ± 0.100 | 0.722 ± 0.081 | 0.769 ± 0.043 | 0.788 ± 0.038 |
| network | local noise, rounds tuned | 0.551 ± 0.100 | 0.663 ± 0.058 | 0.738 ± 0.032 | 0.752 ± 0.076 | 0.785 ± 0.050 |
| network | shared noise, ordinary | 0.656 ± 0.075 | 0.753 ± 0.030 | 0.785 ± 0.023 | 0.820 ± 0.020 | 0.828 ± 0.021 |
| network | shared noise, whole numbers | 0.676 ± 0.087 | 0.742 ± 0.063 | 0.746 ± 0.046 | 0.821 ± 0.017 | 0.827 ± 0.022 |
| network | trusted server (upper limit) | 0.686 ± 0.064 | 0.770 ± 0.037 | 0.797 ± 0.020 | 0.827 ± 0.019 | 0.831 ± 0.019 |
| network | local noise, secure sum, 12-bit values |  | 0.690 ± 0.101 |  |  | 0.788 ± 0.038 |
| network | local noise, secure sum, 8-bit values |  | 0.668 ± 0.119 |  |  | 0.788 ± 0.038 |
| network | whole participants protected |  | 0.504 ± 0.045 |  |  | 0.513 ± 0.043 |
| logistic | no protection (no budget) | 0.832 ± 0.027 |  |  |  |  |
| logistic | local noise (Phase 2 protocol) | 0.729 ± 0.048 | 0.774 ± 0.027 | 0.792 ± 0.022 | 0.804 ± 0.020 | 0.808 ± 0.019 |
| logistic | local noise, rounds tuned | 0.624 ± 0.122 | 0.714 ± 0.071 | 0.784 ± 0.031 | 0.804 ± 0.020 | 0.808 ± 0.019 |
| logistic | shared noise, ordinary | 0.755 ± 0.042 | 0.800 ± 0.033 | 0.816 ± 0.026 | 0.830 ± 0.023 | 0.832 ± 0.022 |
| logistic | shared noise, whole numbers | 0.763 ± 0.036 | 0.806 ± 0.015 | 0.817 ± 0.018 | 0.821 ± 0.018 | 0.821 ± 0.019 |
| logistic | trusted server (upper limit) | 0.756 ± 0.050 | 0.794 ± 0.033 | 0.811 ± 0.026 | 0.825 ± 0.023 | 0.829 ± 0.022 |
| logistic | local noise, secure sum, 12-bit values |  | 0.774 ± 0.027 |  |  | 0.808 ± 0.019 |
| logistic | local noise, secure sum, 8-bit values |  | 0.773 ± 0.028 |  |  | 0.807 ± 0.020 |
| logistic | whole participants protected |  | 0.465 ± 0.097 |  |  | 0.614 ± 0.107 |

**Stroke table, difference in AUC against the better local arm** (mean of the seed-by-seed differences, 95% interval in brackets; marked only if the interval excludes zero)

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | +0.047 (-0.062 to +0.157) | +0.063 (-0.010 to +0.137) | +0.047 (+0.025 to +0.069) **better** | +0.050 (+0.021 to +0.079) **better** | +0.040 (+0.014 to +0.065) **better** |
| network | shared noise, whole numbers | +0.067 (-0.055 to +0.189) | +0.053 (-0.041 to +0.147) | +0.008 (-0.033 to +0.049) | +0.052 (+0.020 to +0.084) **better** | +0.039 (+0.017 to +0.061) **better** |
| network | trusted server (upper limit) | +0.077 (-0.048 to +0.202) | +0.080 (-0.007 to +0.168) | +0.059 (+0.035 to +0.084) **better** | +0.057 (+0.028 to +0.087) **better** | +0.043 (+0.019 to +0.068) **better** |
| logistic | shared noise, ordinary | +0.027 (-0.014 to +0.067) | +0.026 (+0.010 to +0.043) **better** | +0.024 (+0.016 to +0.032) **better** | +0.026 (+0.017 to +0.036) **better** | +0.025 (+0.017 to +0.033) **better** |
| logistic | shared noise, whole numbers | +0.034 (-0.008 to +0.076) | +0.032 (+0.013 to +0.052) **better** | +0.024 (+0.012 to +0.036) **better** | +0.017 (+0.008 to +0.026) **better** | +0.013 (+0.006 to +0.021) **better** |
| logistic | trusted server (upper limit) | +0.028 (-0.007 to +0.062) | +0.020 (+0.004 to +0.036) **better** | +0.018 (+0.007 to +0.029) **better** | +0.021 (+0.012 to +0.030) **better** | +0.022 (+0.013 to +0.030) **better** |

**Loan table, AUC on held-out rows at their natural mix** (mean of 5 seeds ± sample standard deviation; columns are total budgets)

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | no protection (no budget) | 0.976 ± 0.007 |  |  |  |  |
| network | local noise (Phase 2 protocol) | 0.958 ± 0.007 | 0.960 ± 0.006 | 0.961 ± 0.006 | 0.966 ± 0.007 | 0.967 ± 0.007 |
| network | local noise, rounds tuned | 0.958 ± 0.006 | 0.960 ± 0.006 | 0.961 ± 0.006 | 0.967 ± 0.007 | 0.967 ± 0.007 |
| network | shared noise, ordinary | 0.965 ± 0.007 | 0.967 ± 0.007 | 0.967 ± 0.007 | 0.967 ± 0.007 | 0.967 ± 0.007 |
| network | shared noise, whole numbers | 0.962 ± 0.007 | 0.962 ± 0.007 | 0.963 ± 0.007 | 0.966 ± 0.008 | 0.966 ± 0.007 |
| network | trusted server (upper limit) | 0.963 ± 0.008 | 0.966 ± 0.007 | 0.967 ± 0.007 | 0.967 ± 0.007 | 0.967 ± 0.007 |
| network | local noise, secure sum, 12-bit values |  | 0.960 ± 0.006 |  |  | 0.967 ± 0.007 |
| network | local noise, secure sum, 8-bit values |  | 0.960 ± 0.006 |  |  | 0.966 ± 0.007 |
| network | whole participants protected |  | 0.470 ± 0.042 |  |  | 0.752 ± 0.097 |
| logistic | no protection (no budget) | 0.964 ± 0.006 |  |  |  |  |
| logistic | local noise (Phase 2 protocol) | 0.960 ± 0.008 | 0.962 ± 0.007 | 0.962 ± 0.007 | 0.963 ± 0.007 | 0.963 ± 0.007 |
| logistic | local noise, rounds tuned | 0.960 ± 0.008 | 0.962 ± 0.007 | 0.962 ± 0.007 | 0.962 ± 0.007 | 0.963 ± 0.007 |
| logistic | shared noise, ordinary | 0.962 ± 0.006 | 0.963 ± 0.006 | 0.963 ± 0.007 | 0.963 ± 0.007 | 0.963 ± 0.007 |
| logistic | shared noise, whole numbers | 0.962 ± 0.005 | 0.963 ± 0.006 | 0.963 ± 0.007 | 0.963 ± 0.007 | 0.963 ± 0.007 |
| logistic | trusted server (upper limit) | 0.961 ± 0.006 | 0.962 ± 0.006 | 0.962 ± 0.007 | 0.963 ± 0.007 | 0.963 ± 0.007 |
| logistic | local noise, secure sum, 12-bit values |  | 0.962 ± 0.007 |  |  | 0.963 ± 0.007 |
| logistic | local noise, secure sum, 8-bit values |  | 0.962 ± 0.007 |  |  | 0.963 ± 0.007 |
| logistic | whole participants protected |  | 0.535 ± 0.127 |  |  | 0.957 ± 0.010 |

**Loan table, difference in AUC against the better local arm** (mean of the seed-by-seed differences, 95% interval in brackets; marked only if the interval excludes zero)

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | +0.007 (+0.001 to +0.013) **better** | +0.007 (+0.004 to +0.009) **better** | +0.006 (+0.004 to +0.008) **better** | +0.000 (-0.001 to +0.001) | -0.000 (-0.001 to +0.000) |
| network | shared noise, whole numbers | +0.004 (-0.002 to +0.009) | +0.002 (-0.001 to +0.004) | +0.002 (+0.000 to +0.004) **better** | -0.001 (-0.003 to +0.001) | -0.001 (-0.002 to +0.001) |
| network | trusted server (upper limit) | +0.005 (-0.002 to +0.012) | +0.005 (+0.003 to +0.008) **better** | +0.006 (+0.004 to +0.007) **better** | -0.000 (-0.002 to +0.001) | -0.001 (-0.002 to +0.000) |
| logistic | shared noise, ordinary | +0.001 (-0.004 to +0.006) | +0.001 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.001 (-0.001 to +0.002) | +0.001 (-0.000 to +0.002) |
| logistic | shared noise, whole numbers | +0.002 (-0.003 to +0.007) | +0.001 (-0.002 to +0.003) | +0.001 (-0.001 to +0.002) | +0.000 (-0.001 to +0.001) | +0.000 (-0.001 to +0.001) |
| logistic | trusted server (upper limit) | +0.001 (-0.004 to +0.006) | +0.000 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.000 (-0.001 to +0.002) | +0.000 (-0.000 to +0.001) |

## Stroke table: further tables

### Average precision

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | no protection (no budget) | 0.185 ± 0.059 |  |  |  |  |
| network | local noise (Phase 2 protocol) | 0.090 ± 0.039 | 0.111 ± 0.036 | 0.129 ± 0.052 | 0.149 ± 0.054 | 0.144 ± 0.041 |
| network | local noise, rounds tuned | 0.065 ± 0.022 | 0.099 ± 0.048 | 0.117 ± 0.030 | 0.135 ± 0.034 | 0.144 ± 0.038 |
| network | shared noise, ordinary | 0.082 ± 0.023 | 0.122 ± 0.032 | 0.133 ± 0.031 | 0.163 ± 0.038 | 0.167 ± 0.037 |
| network | shared noise, whole numbers | 0.105 ± 0.036 | 0.121 ± 0.030 | 0.117 ± 0.027 | 0.154 ± 0.030 | 0.168 ± 0.040 |
| network | trusted server (upper limit) | 0.095 ± 0.032 | 0.120 ± 0.018 | 0.144 ± 0.026 | 0.175 ± 0.044 | 0.181 ± 0.047 |
| network | local noise, secure sum, 12-bit values |  | 0.112 ± 0.038 |  |  | 0.146 ± 0.039 |
| network | local noise, secure sum, 8-bit values |  | 0.116 ± 0.046 |  |  | 0.144 ± 0.040 |
| network | whole participants protected |  | 0.044 ± 0.004 |  |  | 0.045 ± 0.005 |
| logistic | no protection (no budget) | 0.186 ± 0.051 |  |  |  |  |
| logistic | local noise (Phase 2 protocol) | 0.128 ± 0.032 | 0.144 ± 0.037 | 0.155 ± 0.040 | 0.154 ± 0.034 | 0.155 ± 0.034 |
| logistic | local noise, rounds tuned | 0.096 ± 0.030 | 0.125 ± 0.032 | 0.143 ± 0.025 | 0.154 ± 0.034 | 0.155 ± 0.034 |
| logistic | shared noise, ordinary | 0.129 ± 0.048 | 0.155 ± 0.060 | 0.168 ± 0.056 | 0.180 ± 0.051 | 0.180 ± 0.046 |
| logistic | shared noise, whole numbers | 0.127 ± 0.020 | 0.151 ± 0.022 | 0.162 ± 0.033 | 0.165 ± 0.037 | 0.167 ± 0.043 |
| logistic | trusted server (upper limit) | 0.130 ± 0.028 | 0.147 ± 0.029 | 0.157 ± 0.032 | 0.170 ± 0.033 | 0.175 ± 0.037 |
| logistic | local noise, secure sum, 12-bit values |  | 0.145 ± 0.038 |  |  | 0.155 ± 0.034 |
| logistic | local noise, secure sum, 8-bit values |  | 0.146 ± 0.039 |  |  | 0.154 ± 0.033 |
| logistic | whole participants protected |  | 0.042 ± 0.009 |  |  | 0.094 ± 0.042 |

### Difference in average precision against the better local arm

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | -0.008 (-0.036 to +0.021) | +0.011 (-0.019 to +0.041) | +0.017 (+0.006 to +0.027) **better** | +0.014 (-0.005 to +0.034) | +0.022 (+0.006 to +0.039) **better** |
| network | shared noise, whole numbers | +0.014 (-0.023 to +0.052) | +0.010 (-0.022 to +0.042) | +0.000 (-0.029 to +0.030) | +0.006 (-0.034 to +0.045) | +0.024 (+0.000 to +0.048) **better** |
| network | trusted server (upper limit) | +0.005 (-0.029 to +0.039) | +0.009 (-0.018 to +0.036) | +0.028 (+0.009 to +0.047) **better** | +0.027 (-0.004 to +0.057) | +0.037 (+0.019 to +0.055) **better** |
| logistic | shared noise, ordinary | +0.002 (-0.036 to +0.040) | +0.011 (-0.029 to +0.050) | +0.013 (-0.016 to +0.043) | +0.026 (-0.003 to +0.055) | +0.025 (+0.004 to +0.047) **better** |
| logistic | shared noise, whole numbers | -0.000 (-0.029 to +0.029) | +0.006 (-0.015 to +0.028) | +0.007 (-0.009 to +0.024) | +0.010 (-0.007 to +0.027) | +0.012 (-0.005 to +0.029) |
| logistic | trusted server (upper limit) | +0.003 (-0.020 to +0.026) | +0.003 (-0.020 to +0.025) | +0.002 (-0.020 to +0.024) | +0.015 (-0.001 to +0.032) | +0.020 (+0.006 to +0.035) **better** |

### Difference in AUC against local noise with the Phase 2 protocol

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | local noise, rounds tuned | -0.057 (-0.186 to +0.071) | -0.026 (-0.113 to +0.061) | +0.016 (-0.035 to +0.068) | -0.018 (-0.065 to +0.029) | -0.003 (-0.031 to +0.025) |
| network | shared noise, ordinary | +0.047 (-0.062 to +0.157) | +0.063 (-0.010 to +0.137) | +0.063 (+0.006 to +0.120) **better** | +0.050 (+0.021 to +0.079) **better** | +0.040 (+0.014 to +0.065) **better** |
| network | shared noise, whole numbers | +0.067 (-0.055 to +0.189) | +0.053 (-0.041 to +0.147) | +0.024 (-0.050 to +0.099) | +0.052 (+0.020 to +0.084) **better** | +0.039 (+0.017 to +0.061) **better** |
| network | trusted server (upper limit) | +0.077 (-0.048 to +0.202) | +0.080 (-0.007 to +0.168) | +0.075 (+0.016 to +0.135) **better** | +0.057 (+0.028 to +0.087) **better** | +0.043 (+0.019 to +0.068) **better** |
| logistic | local noise, rounds tuned | -0.105 (-0.173 to -0.036) **worse** | -0.059 (-0.095 to -0.023) **worse** | -0.009 (-0.030 to +0.013) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| logistic | shared noise, ordinary | +0.027 (-0.014 to +0.067) | +0.026 (+0.010 to +0.043) **better** | +0.024 (+0.016 to +0.032) **better** | +0.026 (+0.017 to +0.036) **better** | +0.025 (+0.017 to +0.033) **better** |
| logistic | shared noise, whole numbers | +0.034 (-0.008 to +0.076) | +0.032 (+0.013 to +0.052) **better** | +0.024 (+0.012 to +0.036) **better** | +0.017 (+0.008 to +0.026) **better** | +0.013 (+0.006 to +0.021) **better** |
| logistic | trusted server (upper limit) | +0.028 (-0.007 to +0.062) | +0.020 (+0.004 to +0.036) **better** | +0.018 (+0.007 to +0.029) **better** | +0.021 (+0.012 to +0.030) **better** | +0.022 (+0.013 to +0.030) **better** |

### Difference in AUC against local noise with tuned rounds

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | +0.105 (+0.015 to +0.195) **better** | +0.089 (+0.040 to +0.139) **better** | +0.047 (+0.025 to +0.069) **better** | +0.068 (+0.014 to +0.122) **better** | +0.042 (+0.007 to +0.078) **better** |
| network | shared noise, whole numbers | +0.124 (+0.011 to +0.238) **better** | +0.079 (+0.004 to +0.154) **better** | +0.008 (-0.033 to +0.049) | +0.070 (+0.012 to +0.128) **better** | +0.042 (+0.007 to +0.077) **better** |
| network | trusted server (upper limit) | +0.135 (+0.040 to +0.229) **better** | +0.106 (+0.053 to +0.160) **better** | +0.059 (+0.035 to +0.084) **better** | +0.075 (+0.021 to +0.129) **better** | +0.046 (+0.013 to +0.079) **better** |
| logistic | shared noise, ordinary | +0.131 (+0.049 to +0.213) **better** | +0.086 (+0.045 to +0.127) **better** | +0.033 (+0.015 to +0.050) **better** | +0.026 (+0.017 to +0.036) **better** | +0.025 (+0.017 to +0.033) **better** |
| logistic | shared noise, whole numbers | +0.139 (+0.043 to +0.235) **better** | +0.092 (+0.043 to +0.140) **better** | +0.033 (+0.014 to +0.052) **better** | +0.017 (+0.008 to +0.026) **better** | +0.013 (+0.006 to +0.021) **better** |
| logistic | trusted server (upper limit) | +0.132 (+0.066 to +0.199) **better** | +0.080 (+0.045 to +0.114) **better** | +0.027 (+0.011 to +0.043) **better** | +0.021 (+0.012 to +0.030) **better** | +0.022 (+0.013 to +0.030) **better** |

### Does rounding to fewer bits cost accuracy? (difference in AUC against local noise in the clear)

| Model | Arm | 1 | 8 |
|---|---|---|---|
| network | local noise, secure sum, 12-bit values | +0.000 (-0.001 to +0.002) | +0.000 (-0.000 to +0.001) |
| network | local noise, secure sum, 8-bit values | -0.022 (-0.041 to -0.003) **worse** | -0.000 (-0.005 to +0.004) |
| logistic | local noise, secure sum, 12-bit values | +0.000 (-0.000 to +0.000) | +0.000 (-0.000 to +0.000) |
| logistic | local noise, secure sum, 8-bit values | -0.001 (-0.003 to +0.001) | -0.001 (-0.002 to +0.001) |

### At the cut-off chosen on validation rows

| Model | Arm | Budget | Recall | Precision | F1 |
|---|---|---|---|---|---|
| network | no protection | none | 0.362 ± 0.165 | 0.158 ± 0.064 | 0.217 ± 0.089 |
| network | local noise (Phase 2 protocol) | 1 | 0.229 ± 0.150 | 0.135 ± 0.067 | 0.156 ± 0.084 |
| network | local noise (Phase 2 protocol) | 8 | 0.343 ± 0.199 | 0.156 ± 0.056 | 0.185 ± 0.036 |
| network | local noise, rounds tuned | 1 | 0.290 ± 0.213 | 0.097 ± 0.046 | 0.131 ± 0.063 |
| network | local noise, rounds tuned | 8 | 0.352 ± 0.192 | 0.135 ± 0.023 | 0.188 ± 0.043 |
| network | shared noise, ordinary | 1 | 0.327 ± 0.142 | 0.129 ± 0.041 | 0.171 ± 0.044 |
| network | shared noise, ordinary | 8 | 0.521 ± 0.193 | 0.154 ± 0.033 | 0.232 ± 0.051 |
| network | shared noise, whole numbers | 1 | 0.411 ± 0.238 | 0.107 ± 0.035 | 0.153 ± 0.025 |
| network | shared noise, whole numbers | 8 | 0.463 ± 0.177 | 0.149 ± 0.036 | 0.216 ± 0.045 |
| network | trusted server (upper limit) | 1 | 0.366 ± 0.113 | 0.124 ± 0.022 | 0.181 ± 0.030 |
| network | trusted server (upper limit) | 8 | 0.547 ± 0.197 | 0.145 ± 0.020 | 0.223 ± 0.045 |
| network | local noise, secure sum, 12-bit values | 1 | 0.235 ± 0.124 | 0.145 ± 0.071 | 0.166 ± 0.072 |
| network | local noise, secure sum, 12-bit values | 8 | 0.334 ± 0.188 | 0.158 ± 0.059 | 0.187 ± 0.039 |
| network | local noise, secure sum, 8-bit values | 1 | 0.257 ± 0.146 | 0.137 ± 0.066 | 0.164 ± 0.056 |
| network | local noise, secure sum, 8-bit values | 8 | 0.359 ± 0.208 | 0.170 ± 0.068 | 0.198 ± 0.048 |
| network | whole participants protected | 1 | 0.953 ± 0.089 | 0.045 ± 0.004 | 0.085 ± 0.006 |
| network | whole participants protected | 8 | 0.687 ± 0.430 | 0.038 ± 0.022 | 0.071 ± 0.039 |
| logistic | no protection | none | 0.382 ± 0.188 | 0.188 ± 0.119 | 0.227 ± 0.089 |
| logistic | local noise (Phase 2 protocol) | 1 | 0.296 ± 0.118 | 0.129 ± 0.039 | 0.175 ± 0.054 |
| logistic | local noise (Phase 2 protocol) | 8 | 0.334 ± 0.162 | 0.151 ± 0.047 | 0.193 ± 0.043 |
| logistic | local noise, rounds tuned | 1 | 0.277 ± 0.122 | 0.142 ± 0.020 | 0.178 ± 0.031 |
| logistic | local noise, rounds tuned | 8 | 0.334 ± 0.162 | 0.151 ± 0.047 | 0.193 ± 0.043 |
| logistic | shared noise, ordinary | 1 | 0.487 ± 0.117 | 0.141 ± 0.044 | 0.215 ± 0.053 |
| logistic | shared noise, ordinary | 8 | 0.520 ± 0.204 | 0.160 ± 0.046 | 0.226 ± 0.035 |
| logistic | shared noise, whole numbers | 1 | 0.432 ± 0.218 | 0.140 ± 0.015 | 0.200 ± 0.035 |
| logistic | shared noise, whole numbers | 8 | 0.465 ± 0.150 | 0.151 ± 0.046 | 0.219 ± 0.039 |
| logistic | trusted server (upper limit) | 1 | 0.353 ± 0.185 | 0.132 ± 0.043 | 0.179 ± 0.063 |
| logistic | trusted server (upper limit) | 8 | 0.582 ± 0.209 | 0.140 ± 0.037 | 0.223 ± 0.065 |
| logistic | local noise, secure sum, 12-bit values | 1 | 0.296 ± 0.118 | 0.130 ± 0.039 | 0.175 ± 0.054 |
| logistic | local noise, secure sum, 12-bit values | 8 | 0.334 ± 0.162 | 0.151 ± 0.047 | 0.193 ± 0.043 |
| logistic | local noise, secure sum, 8-bit values | 1 | 0.312 ± 0.108 | 0.138 ± 0.038 | 0.186 ± 0.045 |
| logistic | local noise, secure sum, 8-bit values | 8 | 0.338 ± 0.150 | 0.149 ± 0.039 | 0.194 ± 0.037 |
| logistic | whole participants protected | 1 | 0.876 ± 0.203 | 0.044 ± 0.006 | 0.083 ± 0.012 |
| logistic | whole participants protected | 8 | 0.374 ± 0.256 | 0.100 ± 0.055 | 0.135 ± 0.043 |

### Messages at a total budget of 1

| Model | Arm | Rounds | Bits per number | Per round, against 32-bit numbers | Whole run (kB) | Whole run, against the local arm |
|---|---|---|---|---|---|---|
| network | local noise (Phase 2 protocol) | 20 | 32 | 1.00 | 276.6 | 1.00 |
| network | local noise, rounds tuned | 60 | 32 | 1.00 | 829.7 | 3.00 |
| network | shared noise, ordinary | 200 | 13 | 0.41 | 1123.8 | 4.06 |
| network | shared noise, whole numbers | 200 | 22 | 0.69 | 1901.6 | 6.88 |
| network | trusted server (upper limit) | 200 | 32 | 1.00 | 2765.6 | 10.00 |
| network | local noise, secure sum, 12-bit values | 20 | 15 | 0.47 | 129.8 | 0.47 |
| network | local noise, secure sum, 8-bit values | 20 | 11 | 0.34 | 95.2 | 0.34 |
| network | whole participants protected | 20 | 10 | 0.31 | 86.6 | 0.31 |
| logistic | local noise (Phase 2 protocol) | 20 | 32 | 1.00 | 1.7 | 1.00 |
| logistic | local noise, rounds tuned | 60 | 32 | 1.00 | 5.0 | 3.00 |
| logistic | shared noise, ordinary | 600 | 12 | 0.38 | 19.4 | 11.52 |
| logistic | shared noise, whole numbers | 200 | 19 | 0.60 | 10.2 | 6.05 |
| logistic | trusted server (upper limit) | 600 | 32 | 1.00 | 50.4 | 30.00 |
| logistic | local noise, secure sum, 12-bit values | 20 | 15 | 0.48 | 1.0 | 0.57 |
| logistic | local noise, secure sum, 8-bit values | 20 | 11 | 0.35 | 0.7 | 0.44 |
| logistic | whole participants protected | 20 | 10 | 0.32 | 0.7 | 0.42 |

### Settings and noise

| Model | Arm | Budget | Rounds | Batch | Local passes | Learning rate | Clipping | Noise multiplier |
|---|---|---|---|---|---|---|---|---|
| network | local noise (Phase 2 protocol) | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 3.27 to 28.91 |
| network | local noise (Phase 2 protocol) | 8 | 20 | 32 | 3 | 0.2 | 1 | 0.82 to 4.65 |
| network | local noise, rounds tuned | 1 | 60 | 32 | 1 | 0.05 | 1 | 3.27 to 28.91 |
| network | local noise, rounds tuned | 8 | 60 | 32 | 1 | 0.2 | 1 | 0.82 to 4.65 |
| network | shared noise, ordinary | 1 | 200 | 640 | 1 | 1 | 1 | 9.95 to 9.97 |
| network | shared noise, ordinary | 8 | 600 | 640 | 1 | 1 | 1 | 2.86 to 2.87 |
| network | shared noise, whole numbers | 1 | 200 | all rows | 1 | 1 | 0.5 | 57.24 |
| network | shared noise, whole numbers | 8 | 200 | all rows | 1 | 3 | 1 | 9.02 |
| network | trusted server (upper limit) | 1 | 200 | 640 | 1 | 1 | 1 | 9.95 to 9.97 |
| network | trusted server (upper limit) | 8 | 600 | 640 | 1 | 1 | 1 | 2.86 to 2.87 |
| network | local noise, secure sum, 12-bit values | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 3.27 to 28.91 |
| network | local noise, secure sum, 12-bit values | 8 | 20 | 32 | 3 | 0.2 | 1 | 0.82 to 4.65 |
| network | local noise, secure sum, 8-bit values | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 3.27 to 28.91 |
| network | local noise, secure sum, 8-bit values | 8 | 20 | 32 | 3 | 0.2 | 1 | 0.82 to 4.65 |
| network | whole participants protected | 1 | 20 | 32 | 3 | 0.5 | 2 | 16.70 |
| network | whole participants protected | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 2.68 |
| logistic | local noise (Phase 2 protocol) | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 3.27 to 28.91 |
| logistic | local noise (Phase 2 protocol) | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 0.82 to 4.65 |
| logistic | local noise, rounds tuned | 1 | 60 | 32 | 1 | 0.5 | 0.5 | 3.27 to 28.91 |
| logistic | local noise, rounds tuned | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 0.82 to 4.65 |
| logistic | shared noise, ordinary | 1 | 600 | 640 | 1 | 1 | 1 | 17.08 to 17.12 |
| logistic | shared noise, ordinary | 8 | 600 | 640 | 1 | 3 | 1 | 2.86 to 2.87 |
| logistic | shared noise, whole numbers | 1 | 200 | all rows | 1 | 3 | 1 | 57.24 |
| logistic | shared noise, whole numbers | 8 | 200 | all rows | 1 | 3 | 1 | 9.02 |
| logistic | trusted server (upper limit) | 1 | 600 | 640 | 1 | 1 | 1 | 17.08 to 17.12 |
| logistic | trusted server (upper limit) | 8 | 600 | 640 | 1 | 3 | 1 | 2.86 to 2.87 |
| logistic | local noise, secure sum, 12-bit values | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 3.27 to 28.91 |
| logistic | local noise, secure sum, 12-bit values | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 0.82 to 4.65 |
| logistic | local noise, secure sum, 8-bit values | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 3.27 to 28.91 |
| logistic | local noise, secure sum, 8-bit values | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 0.82 to 4.65 |
| logistic | whole participants protected | 1 | 20 | 32 | 3 | 0.5 | 2 | 16.70 |
| logistic | whole participants protected | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 2.68 |

## Loan table: further tables

### Average precision

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | no protection (no budget) | 0.984 ± 0.006 |  |  |  |  |
| network | local noise (Phase 2 protocol) | 0.975 ± 0.008 | 0.976 ± 0.008 | 0.976 ± 0.009 | 0.979 ± 0.007 | 0.980 ± 0.007 |
| network | local noise, rounds tuned | 0.974 ± 0.009 | 0.975 ± 0.009 | 0.976 ± 0.009 | 0.980 ± 0.006 | 0.980 ± 0.006 |
| network | shared noise, ordinary | 0.979 ± 0.007 | 0.980 ± 0.007 | 0.980 ± 0.007 | 0.980 ± 0.007 | 0.980 ± 0.007 |
| network | shared noise, whole numbers | 0.977 ± 0.007 | 0.977 ± 0.008 | 0.977 ± 0.009 | 0.979 ± 0.007 | 0.980 ± 0.007 |
| network | trusted server (upper limit) | 0.978 ± 0.007 | 0.979 ± 0.007 | 0.980 ± 0.007 | 0.980 ± 0.007 | 0.980 ± 0.007 |
| network | local noise, secure sum, 12-bit values |  | 0.976 ± 0.008 |  |  | 0.980 ± 0.007 |
| network | local noise, secure sum, 8-bit values |  | 0.976 ± 0.008 |  |  | 0.979 ± 0.007 |
| network | whole participants protected |  | 0.612 ± 0.021 |  |  | 0.808 ± 0.084 |
| logistic | no protection (no budget) | 0.978 ± 0.008 |  |  |  |  |
| logistic | local noise (Phase 2 protocol) | 0.978 ± 0.007 | 0.978 ± 0.008 | 0.978 ± 0.008 | 0.978 ± 0.008 | 0.978 ± 0.008 |
| logistic | local noise, rounds tuned | 0.978 ± 0.007 | 0.978 ± 0.008 | 0.978 ± 0.008 | 0.977 ± 0.009 | 0.977 ± 0.009 |
| logistic | shared noise, ordinary | 0.976 ± 0.009 | 0.977 ± 0.009 | 0.977 ± 0.009 | 0.978 ± 0.008 | 0.978 ± 0.008 |
| logistic | shared noise, whole numbers | 0.977 ± 0.008 | 0.978 ± 0.008 | 0.978 ± 0.009 | 0.978 ± 0.009 | 0.978 ± 0.009 |
| logistic | trusted server (upper limit) | 0.977 ± 0.008 | 0.977 ± 0.008 | 0.977 ± 0.008 | 0.978 ± 0.008 | 0.978 ± 0.008 |
| logistic | local noise, secure sum, 12-bit values |  | 0.978 ± 0.008 |  |  | 0.978 ± 0.008 |
| logistic | local noise, secure sum, 8-bit values |  | 0.978 ± 0.008 |  |  | 0.978 ± 0.008 |
| logistic | whole participants protected |  | 0.658 ± 0.103 |  |  | 0.973 ± 0.012 |

### Difference in average precision against the better local arm

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | +0.005 (+0.000 to +0.010) **better** | +0.004 (-0.001 to +0.009) | +0.004 (-0.000 to +0.008) | -0.000 (-0.001 to +0.000) | -0.000 (-0.001 to +0.000) |
| network | shared noise, whole numbers | +0.003 (-0.002 to +0.009) | +0.002 (-0.001 to +0.004) | +0.001 (-0.001 to +0.002) | -0.001 (-0.002 to +0.000) | -0.000 (-0.001 to +0.000) |
| network | trusted server (upper limit) | +0.004 (-0.001 to +0.009) | +0.004 (-0.000 to +0.009) | +0.004 (-0.000 to +0.008) | -0.000 (-0.001 to +0.000) | -0.000 (-0.001 to -0.000) **worse** |
| logistic | shared noise, ordinary | -0.001 (-0.006 to +0.004) | -0.001 (-0.003 to +0.002) | -0.000 (-0.002 to +0.001) | -0.000 (-0.002 to +0.001) | +0.000 (-0.001 to +0.001) |
| logistic | shared noise, whole numbers | -0.000 (-0.004 to +0.004) | -0.000 (-0.002 to +0.002) | -0.000 (-0.002 to +0.002) | -0.000 (-0.002 to +0.001) | -0.000 (-0.002 to +0.001) |
| logistic | trusted server (upper limit) | -0.001 (-0.005 to +0.003) | -0.001 (-0.002 to +0.001) | -0.000 (-0.002 to +0.001) | -0.000 (-0.001 to +0.001) | -0.000 (-0.001 to +0.001) |

### Difference in AUC against local noise with the Phase 2 protocol

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | local noise, rounds tuned | +0.000 (-0.007 to +0.007) | +0.001 (-0.003 to +0.004) | +0.000 (-0.002 to +0.002) | +0.001 (-0.001 to +0.003) | +0.001 (-0.001 to +0.002) |
| network | shared noise, ordinary | +0.007 (+0.004 to +0.010) **better** | +0.007 (+0.004 to +0.010) **better** | +0.006 (+0.004 to +0.009) **better** | +0.001 (-0.001 to +0.003) | +0.000 (-0.001 to +0.001) |
| network | shared noise, whole numbers | +0.004 (+0.001 to +0.007) **better** | +0.002 (-0.000 to +0.005) | +0.002 (-0.000 to +0.004) | -0.000 (-0.003 to +0.003) | -0.000 (-0.002 to +0.001) |
| network | trusted server (upper limit) | +0.005 (+0.003 to +0.008) **better** | +0.006 (+0.003 to +0.008) **better** | +0.006 (+0.004 to +0.008) **better** | +0.001 (-0.002 to +0.003) | -0.000 (-0.001 to +0.001) |
| logistic | local noise, rounds tuned | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) | -0.000 (-0.002 to +0.001) | -0.000 (-0.001 to +0.001) |
| logistic | shared noise, ordinary | +0.001 (-0.004 to +0.006) | +0.001 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.001 (-0.001 to +0.002) | +0.001 (-0.000 to +0.002) |
| logistic | shared noise, whole numbers | +0.002 (-0.003 to +0.007) | +0.001 (-0.002 to +0.003) | +0.001 (-0.001 to +0.002) | +0.000 (-0.001 to +0.001) | +0.000 (-0.001 to +0.001) |
| logistic | trusted server (upper limit) | +0.001 (-0.004 to +0.006) | +0.000 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.000 (-0.001 to +0.002) | +0.000 (-0.000 to +0.001) |

### Difference in AUC against local noise with tuned rounds

| Model | Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| network | shared noise, ordinary | +0.007 (+0.001 to +0.013) **better** | +0.007 (+0.004 to +0.009) **better** | +0.006 (+0.004 to +0.008) **better** | +0.000 (-0.001 to +0.001) | -0.000 (-0.001 to +0.000) |
| network | shared noise, whole numbers | +0.004 (-0.002 to +0.009) | +0.002 (-0.001 to +0.004) | +0.002 (+0.000 to +0.004) **better** | -0.001 (-0.003 to +0.001) | -0.001 (-0.002 to +0.001) |
| network | trusted server (upper limit) | +0.005 (-0.002 to +0.012) | +0.005 (+0.003 to +0.008) **better** | +0.006 (+0.004 to +0.007) **better** | -0.000 (-0.002 to +0.001) | -0.001 (-0.002 to +0.000) |
| logistic | shared noise, ordinary | +0.001 (-0.004 to +0.006) | +0.001 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.001 (-0.000 to +0.002) | +0.001 (-0.000 to +0.001) |
| logistic | shared noise, whole numbers | +0.002 (-0.003 to +0.007) | +0.001 (-0.002 to +0.003) | +0.001 (-0.001 to +0.002) | +0.001 (-0.000 to +0.001) | +0.000 (-0.000 to +0.001) |
| logistic | trusted server (upper limit) | +0.001 (-0.004 to +0.006) | +0.000 (-0.002 to +0.003) | +0.000 (-0.001 to +0.002) | +0.001 (-0.001 to +0.002) | +0.000 (-0.000 to +0.001) |

### Does rounding to fewer bits cost accuracy? (difference in AUC against local noise in the clear)

| Model | Arm | 1 | 8 |
|---|---|---|---|
| network | local noise, secure sum, 12-bit values | -0.000 (-0.000 to +0.000) | +0.000 (-0.000 to +0.000) |
| network | local noise, secure sum, 8-bit values | +0.000 (-0.001 to +0.001) | -0.000 (-0.001 to +0.000) |
| logistic | local noise, secure sum, 12-bit values | -0.000 (-0.000 to +0.000) | -0.000 (-0.000 to +0.000) |
| logistic | local noise, secure sum, 8-bit values | -0.000 (-0.000 to +0.000) | +0.000 (-0.000 to +0.000) |

### At the cut-off chosen on validation rows

| Model | Arm | Budget | Recall | Precision | F1 |
|---|---|---|---|---|---|
| network | no protection | none | 0.922 ± 0.012 | 0.980 ± 0.004 | 0.950 ± 0.006 |
| network | local noise (Phase 2 protocol) | 1 | 0.916 ± 0.020 | 0.950 ± 0.025 | 0.932 ± 0.005 |
| network | local noise (Phase 2 protocol) | 8 | 0.915 ± 0.009 | 0.973 ± 0.009 | 0.943 ± 0.004 |
| network | local noise, rounds tuned | 1 | 0.922 ± 0.020 | 0.938 ± 0.020 | 0.930 ± 0.004 |
| network | local noise, rounds tuned | 8 | 0.913 ± 0.013 | 0.972 ± 0.016 | 0.941 ± 0.003 |
| network | shared noise, ordinary | 1 | 0.920 ± 0.011 | 0.962 ± 0.016 | 0.940 ± 0.004 |
| network | shared noise, ordinary | 8 | 0.914 ± 0.017 | 0.970 ± 0.012 | 0.941 ± 0.007 |
| network | shared noise, whole numbers | 1 | 0.926 ± 0.015 | 0.949 ± 0.023 | 0.937 ± 0.007 |
| network | shared noise, whole numbers | 8 | 0.916 ± 0.010 | 0.971 ± 0.011 | 0.943 ± 0.004 |
| network | trusted server (upper limit) | 1 | 0.913 ± 0.011 | 0.963 ± 0.009 | 0.937 ± 0.005 |
| network | trusted server (upper limit) | 8 | 0.919 ± 0.012 | 0.967 ± 0.016 | 0.942 ± 0.006 |
| network | local noise, secure sum, 12-bit values | 1 | 0.916 ± 0.020 | 0.950 ± 0.025 | 0.932 ± 0.005 |
| network | local noise, secure sum, 12-bit values | 8 | 0.914 ± 0.017 | 0.972 ± 0.010 | 0.942 ± 0.005 |
| network | local noise, secure sum, 8-bit values | 1 | 0.921 ± 0.015 | 0.940 ± 0.032 | 0.930 ± 0.009 |
| network | local noise, secure sum, 8-bit values | 8 | 0.914 ± 0.009 | 0.976 ± 0.012 | 0.944 ± 0.002 |
| network | whole participants protected | 1 | 1.000 ± 0.000 | 0.623 ± 0.001 | 0.768 ± 0.001 |
| network | whole participants protected | 8 | 0.925 ± 0.049 | 0.710 ± 0.065 | 0.800 ± 0.030 |
| logistic | no protection | none | 0.921 ± 0.020 | 0.940 ± 0.037 | 0.930 ± 0.012 |
| logistic | local noise (Phase 2 protocol) | 1 | 0.913 ± 0.012 | 0.967 ± 0.016 | 0.939 ± 0.007 |
| logistic | local noise (Phase 2 protocol) | 8 | 0.914 ± 0.009 | 0.967 ± 0.013 | 0.939 ± 0.005 |
| logistic | local noise, rounds tuned | 1 | 0.913 ± 0.012 | 0.967 ± 0.016 | 0.939 ± 0.007 |
| logistic | local noise, rounds tuned | 8 | 0.905 ± 0.009 | 0.967 ± 0.012 | 0.935 ± 0.009 |
| logistic | shared noise, ordinary | 1 | 0.915 ± 0.006 | 0.957 ± 0.012 | 0.935 ± 0.003 |
| logistic | shared noise, ordinary | 8 | 0.915 ± 0.006 | 0.964 ± 0.015 | 0.939 ± 0.006 |
| logistic | shared noise, whole numbers | 1 | 0.911 ± 0.005 | 0.969 ± 0.018 | 0.939 ± 0.010 |
| logistic | shared noise, whole numbers | 8 | 0.916 ± 0.005 | 0.965 ± 0.012 | 0.940 ± 0.005 |
| logistic | trusted server (upper limit) | 1 | 0.921 ± 0.005 | 0.962 ± 0.009 | 0.941 ± 0.005 |
| logistic | trusted server (upper limit) | 8 | 0.913 ± 0.007 | 0.970 ± 0.015 | 0.941 ± 0.007 |
| logistic | local noise, secure sum, 12-bit values | 1 | 0.914 ± 0.012 | 0.966 ± 0.018 | 0.939 ± 0.007 |
| logistic | local noise, secure sum, 12-bit values | 8 | 0.912 ± 0.006 | 0.968 ± 0.013 | 0.939 ± 0.004 |
| logistic | local noise, secure sum, 8-bit values | 1 | 0.915 ± 0.012 | 0.964 ± 0.017 | 0.938 ± 0.006 |
| logistic | local noise, secure sum, 8-bit values | 8 | 0.907 ± 0.008 | 0.971 ± 0.014 | 0.938 ± 0.004 |
| logistic | whole participants protected | 1 | 0.994 ± 0.006 | 0.625 ± 0.004 | 0.767 ± 0.002 |
| logistic | whole participants protected | 8 | 0.907 ± 0.039 | 0.921 ± 0.051 | 0.912 ± 0.026 |

### Messages at a total budget of 1

| Model | Arm | Rounds | Bits per number | Per round, against 32-bit numbers | Whole run (kB) | Whole run, against the local arm |
|---|---|---|---|---|---|---|
| network | local noise (Phase 2 protocol) | 20 | 32 | 1.00 | 79.4 | 1.00 |
| network | local noise, rounds tuned | 60 | 32 | 1.00 | 238.3 | 3.00 |
| network | shared noise, ordinary | 200 | 12 | 0.38 | 298.2 | 3.75 |
| network | shared noise, whole numbers | 200 | 20 | 0.63 | 496.8 | 6.25 |
| network | trusted server (upper limit) | 200 | 32 | 1.00 | 794.4 | 10.00 |
| network | local noise, secure sum, 12-bit values | 20 | 15 | 0.47 | 37.4 | 0.47 |
| network | local noise, secure sum, 8-bit values | 20 | 11 | 0.34 | 27.5 | 0.35 |
| network | whole participants protected | 20 | 10 | 0.31 | 25.0 | 0.31 |
| logistic | local noise (Phase 2 protocol) | 20 | 32 | 1.00 | 1.1 | 1.00 |
| logistic | local noise, rounds tuned | 20 | 32 | 1.00 | 1.1 | 1.00 |
| logistic | shared noise, ordinary | 600 | 12 | 0.38 | 12.8 | 11.39 |
| logistic | shared noise, whole numbers | 60 | 18 | 0.57 | 2.1 | 1.86 |
| logistic | trusted server (upper limit) | 600 | 32 | 1.00 | 33.6 | 30.00 |
| logistic | local noise, secure sum, 12-bit values | 20 | 15 | 0.48 | 0.7 | 0.62 |
| logistic | local noise, secure sum, 8-bit values | 20 | 11 | 0.36 | 0.6 | 0.50 |
| logistic | whole participants protected | 20 | 10 | 0.32 | 0.5 | 0.46 |

### Settings and noise

| Model | Arm | Budget | Rounds | Batch | Local passes | Learning rate | Clipping | Noise multiplier |
|---|---|---|---|---|---|---|---|---|
| network | local noise (Phase 2 protocol) | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 6.78 to 6.83 |
| network | local noise (Phase 2 protocol) | 8 | 20 | 32 | 3 | 0.2 | 0.5 | 1.30 to 1.31 |
| network | local noise, rounds tuned | 1 | 60 | 32 | 1 | 0.05 | 0.5 | 6.82 to 6.85 |
| network | local noise, rounds tuned | 8 | 60 | 32 | 1 | 0.2 | 0.5 | 1.30 to 1.31 |
| network | shared noise, ordinary | 1 | 200 | 640 | 1 | 1 | 0.5 | 11.41 |
| network | shared noise, ordinary | 8 | 200 | 640 | 1 | 1 | 0.5 | 2.00 |
| network | shared noise, whole numbers | 1 | 200 | all rows | 1 | 0.3 | 1 | 57.24 |
| network | shared noise, whole numbers | 8 | 200 | all rows | 1 | 1 | 0.5 | 9.02 |
| network | trusted server (upper limit) | 1 | 200 | 640 | 1 | 1 | 0.5 | 11.41 |
| network | trusted server (upper limit) | 8 | 200 | 640 | 1 | 1 | 0.5 | 2.00 |
| network | local noise, secure sum, 12-bit values | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 6.78 to 6.83 |
| network | local noise, secure sum, 12-bit values | 8 | 20 | 32 | 3 | 0.2 | 0.5 | 1.30 to 1.31 |
| network | local noise, secure sum, 8-bit values | 1 | 20 | 32 | 3 | 0.05 | 0.5 | 6.78 to 6.83 |
| network | local noise, secure sum, 8-bit values | 8 | 20 | 32 | 3 | 0.2 | 0.5 | 1.30 to 1.31 |
| network | whole participants protected | 1 | 20 | 32 | 3 | 0.02 | 2 | 16.70 |
| network | whole participants protected | 8 | 20 | 32 | 3 | 0.1 | 0.5 | 2.68 |
| logistic | local noise (Phase 2 protocol) | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 6.78 to 6.83 |
| logistic | local noise (Phase 2 protocol) | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 1.30 to 1.31 |
| logistic | local noise, rounds tuned | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 6.78 to 6.83 |
| logistic | local noise, rounds tuned | 8 | 60 | 32 | 1 | 0.5 | 1 | 1.30 to 1.31 |
| logistic | shared noise, ordinary | 1 | 600 | 640 | 1 | 1 | 0.5 | 19.64 |
| logistic | shared noise, ordinary | 8 | 600 | 640 | 1 | 1 | 1 | 3.26 |
| logistic | shared noise, whole numbers | 1 | 60 | all rows | 1 | 3 | 1 | 31.34 |
| logistic | shared noise, whole numbers | 8 | 60 | all rows | 1 | 3 | 1 | 4.94 |
| logistic | trusted server (upper limit) | 1 | 600 | 640 | 1 | 1 | 0.5 | 19.64 |
| logistic | trusted server (upper limit) | 8 | 600 | 640 | 1 | 1 | 1 | 3.26 |
| logistic | local noise, secure sum, 12-bit values | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 6.78 to 6.83 |
| logistic | local noise, secure sum, 12-bit values | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 1.30 to 1.31 |
| logistic | local noise, secure sum, 8-bit values | 1 | 20 | 32 | 3 | 0.2 | 0.5 | 6.78 to 6.83 |
| logistic | local noise, secure sum, 8-bit values | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 1.30 to 1.31 |
| logistic | whole participants protected | 1 | 20 | 32 | 3 | 0.5 | 0.5 | 16.70 |
| logistic | whole participants protected | 8 | 20 | 32 | 3 | 0.5 | 0.5 | 2.68 |
