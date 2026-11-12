# Phase 4 attack tests: what an attacker can tell about single records

Built by `python -m benchmarks.phase4.report` from 1080 stored runs. Every file was validated and every stored privacy report was recomputed from its recorded releases.

Each run trains once on a real table in which about a quarter of every participant's training rows (at most 250) are test records, each kept or left out by a fair coin. Advantage is the attacker's true-positive rate minus its false-positive rate at a cut-off chosen on other records. The lower limit on the budget pools the guesses of all runs of a cell (one-run method of Steinke, Nasr and Jagielski, 2023), at 95% confidence.

Share of the test records an attack guesses for, fixed on the pilot runs before these runs: 0.05 for the plan's attacker, 0.05 for the outsider.

**The website's target, read off the default plan** (measured advantage of the plan's attacker, mean of 30 runs; the interval is 95%, from resampling whole runs)

| Table | Model | Measured at a budget of 1 (the plan's attacker) | Upper end of the interval | Within 5%? | Largest budget whose upper end is within 5% |
|---|---|---|---|---|---|
| stroke | network | -1.1% | -0.1% | yes | 8 |
| stroke | logistic | 0.5% | 1.4% | yes | 8 |
| loan | network | 4.3% | 5.6% | **no** | 0.5 |
| loan | logistic | 5.3% | 6.3% | **no** | none tried |

**Stroke table, network: the default plan, three numbers side by side** (advantage = true-positive rate minus false-positive rate; measured values are means of 30 runs with a 95% interval)

| Total budget | Limit for any method | Limit for the noise used | Measured: the plan's attacker | Measured: outsider with the final model | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| 0.5 | 24.5% | 5.7% | -0.4% (-1.9% to 1.2%) | 0.4% (-0.9% to 1.7%) | 0.00 | 0.21 |
| 1 | 46.2% | 10.7% | -1.1% (-2.1% to -0.1%) | 0.1% (-1.0% to 1.2%) | 0.04 | 0.49 |
| 2 | 76.2% | 19.8% | 0.5% (-0.7% to 1.6%) | 1.2% (-0.1% to 2.5%) | 0.15 | 1.06 |
| 4 | 96.4% | 35.6% | 0.9% (-0.4% to 2.2%) | 1.0% (-0.7% to 2.8%) | 0.29 | 2.11 |
| 8 | 99.9% | 59.5% | 2.2% (1.0% to 3.5%) | 0.9% (-0.7% to 2.4%) | 0.53 | 4.01 |

**Stroke table, logistic: the default plan, three numbers side by side** (advantage = true-positive rate minus false-positive rate; measured values are means of 30 runs with a 95% interval)

| Total budget | Limit for any method | Limit for the noise used | Measured: the plan's attacker | Measured: outsider with the final model | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| 0.5 | 24.5% | 5.7% | 0.5% (-0.4% to 1.5%) | 0.2% (-1.2% to 1.8%) | 0.05 | 0.21 |
| 1 | 46.2% | 10.7% | 0.5% (-0.5% to 1.4%) | 0.3% (-1.2% to 1.8%) | 0.11 | 0.49 |
| 2 | 76.2% | 19.8% | 1.1% (0.3% to 1.8%) | 0.6% (-0.6% to 1.9%) | 0.24 | 1.06 |
| 4 | 96.4% | 35.6% | 1.5% (0.5% to 2.6%) | 0.9% (-0.6% to 2.3%) | 0.50 | 2.11 |
| 8 | 99.9% | 59.5% | 2.8% (2.0% to 3.7%) | 0.9% (-0.5% to 2.4%) | 0.73 | 4.01 |

**Loan table, network: the default plan, three numbers side by side** (advantage = true-positive rate minus false-positive rate; measured values are means of 30 runs with a 95% interval)

| Total budget | Limit for any method | Limit for the noise used | Measured: the plan's attacker | Measured: outsider with the final model | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| 0.5 | 24.5% | 5.6% | 2.8% (1.6% to 4.0%) | -0.0% (-1.2% to 1.2%) | 0.03 | 0.22 |
| 1 | 46.2% | 10.6% | 4.3% (3.1% to 5.6%) | 0.3% (-0.7% to 1.4%) | 0.23 | 0.51 |
| 2 | 76.2% | 19.5% | 8.0% (6.7% to 9.3%) | -0.2% (-1.1% to 0.7%) | 0.53 | 1.06 |
| 4 | 96.4% | 34.6% | 7.6% (6.1% to 9.1%) | -0.2% (-1.4% to 1.0%) | 0.62 | 2.05 |
| 8 | 99.9% | 56.4% | 10.4% (9.0% to 11.7%) | 0.2% (-1.1% to 1.3%) | 0.68 | 3.78 |

**Loan table, logistic: the default plan, three numbers side by side** (advantage = true-positive rate minus false-positive rate; measured values are means of 30 runs with a 95% interval)

| Total budget | Limit for any method | Limit for the noise used | Measured: the plan's attacker | Measured: outsider with the final model | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| 0.5 | 24.5% | 5.6% | 3.6% (2.2% to 5.0%) | -0.8% (-1.7% to 0.3%) | 0.12 | 0.22 |
| 1 | 46.2% | 10.6% | 5.3% (4.2% to 6.3%) | -0.5% (-1.6% to 0.7%) | 0.30 | 0.51 |
| 2 | 76.2% | 19.5% | 8.2% (7.1% to 9.3%) | -0.4% (-1.6% to 1.0%) | 0.57 | 1.06 |
| 4 | 96.4% | 34.6% | 6.4% (5.1% to 7.5%) | -0.4% (-1.6% to 1.0%) | 0.59 | 2.05 |
| 8 | 99.9% | 56.4% | 7.7% (6.3% to 9.1%) | 0.0% (-1.1% to 1.4%) | 0.52 | 3.78 |

**Stroke table: the other arms**

| Model | Arm | Stated budget | Limit for the noise used | Measured: the plan's attacker | Measured: outsider | Lower limit shown | Most the test could show | Flagged |
|---|---|---|---|---|---|---|---|---|
| network | no protection | none | n/a | 4.0% (2.3% to 5.6%) | 1.0% (-0.5% to 2.5%) | 0.05 | n/a | n/a |
| network | planted defect: default plan, noise switched off | 1 | 10.7% | 18.8% (17.1% to 20.5%) | 1.7% (0.4% to 3.1%) | 0.67 | 0.49 | **yes** (advantage above the limit for the stated noise) |
| logistic | no protection | none | n/a | 5.6% (4.1% to 7.3%) | 0.2% (-1.2% to 1.5%) | 0.83 | n/a | n/a |
| logistic | planted defect: default plan, noise switched off | 1 | 10.7% | 9.5% (8.1% to 10.9%) | -0.0% (-1.0% to 1.2%) | 2.31 | 0.49 | **yes** (lower limit above the stated budget) |
| logistic | shared noise, ordinary | 1 | 10.6% | -0.6% (-1.8% to 0.7%) | 0.3% (-1.2% to 1.8%) | 0.00 | 0.48 | no |
| logistic | shared noise, ordinary | 8 | 58.2% | 1.0% (-0.2% to 2.1%) | 0.9% (-0.7% to 2.7%) | 0.35 | 3.82 | no |
| logistic | shared noise, whole numbers | 1 | 15.1% | 3.0% (1.2% to 4.6%) | 1.6% (-0.2% to 3.5%) | 0.07 | 0.76 | no |
| logistic | shared noise, whole numbers | 8 | 95.6% | 2.0% (0.3% to 3.7%) | 1.0% (-1.0% to 3.0%) | 0.56 | 5.04 | no |

**Loan table: the other arms**

| Model | Arm | Stated budget | Limit for the noise used | Measured: the plan's attacker | Measured: outsider | Lower limit shown | Most the test could show | Flagged |
|---|---|---|---|---|---|---|---|---|
| network | no protection | none | n/a | 10.4% (9.0% to 11.8%) | -0.2% (-1.4% to 1.0%) | 0.29 | n/a | n/a |
| network | planted defect: default plan, noise switched off | 1 | 10.6% | 30.1% (28.7% to 31.7%) | -0.4% (-1.4% to 0.6%) | 0.87 | 0.51 | **yes** (advantage above the limit for the stated noise) |
| logistic | no protection | none | n/a | 6.9% (5.3% to 8.3%) | -0.3% (-1.5% to 1.0%) | 0.61 | n/a | n/a |
| logistic | planted defect: default plan, noise switched off | 1 | 10.6% | 15.3% (14.0% to 16.5%) | -0.4% (-1.5% to 0.7%) | 1.03 | 0.51 | **yes** (lower limit above the stated budget; advantage above the limit for the stated noise) |
| logistic | shared noise, ordinary | 1 | 10.6% | 2.8% (1.8% to 3.9%) | 0.8% (-0.4% to 2.1%) | 0.29 | 0.50 | no |
| logistic | shared noise, ordinary | 8 | 58.4% | 5.7% (4.2% to 7.2%) | 0.4% (-0.9% to 1.9%) | 0.60 | 3.88 | no |
| logistic | shared noise, whole numbers | 1 | 15.1% | 3.2% (1.9% to 4.6%) | 0.6% (-0.8% to 2.0%) | 0.21 | 0.77 | no |
| logistic | shared noise, whole numbers | 8 | 95.6% | 8.6% (7.4% to 9.8%) | 0.1% (-1.3% to 1.5%) | 0.59 | 5.10 | no |

## Every cell

| Table | Model | Arm | Budget | Attack | Runs | Test records | Guesses | Right | Lower limit (pooled) | Median of single runs | Most it could show | Advantage | Mean test AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stroke | network | no protection | none | an outsider with the final model | 30 | 19245 | 932 | 486 | 0.000 | 0.000 | n/a | 1.0% (-0.5% to 2.5%) | 0.816 |
| stroke | network | no protection | none | the plan's attacker, every round | 30 | 19245 | 932 | 505 | 0.053 | 0.000 | n/a | 4.0% (2.3% to 5.6%) | 0.816 |
| stroke | logistic | no protection | none | an outsider with the final model | 30 | 19245 | 932 | 490 | 0.000 | 0.000 | n/a | 0.2% (-1.2% to 1.5%) | 0.825 |
| stroke | logistic | no protection | none | the plan's attacker, every round | 30 | 19245 | 932 | 673 | 0.827 | 0.224 | n/a | 5.6% (4.1% to 7.3%) | 0.825 |
| stroke | network | default plan (local noise) | 0.5 | an outsider with the final model | 30 | 19245 | 932 | 469 | 0.000 | 0.000 | 0.21 | 0.4% (-0.9% to 1.7%) | 0.609 |
| stroke | network | default plan (local noise) | 0.5 | the plan's attacker, every round | 30 | 19245 | 932 | 485 | 0.000 | 0.000 | 0.21 | -0.4% (-1.9% to 1.2%) | 0.609 |
| stroke | network | default plan (local noise) | 1 | an outsider with the final model | 30 | 19245 | 932 | 484 | 0.000 | 0.000 | 0.49 | 0.1% (-1.0% to 1.2%) | 0.692 |
| stroke | network | default plan (local noise) | 1 | the plan's attacker, every round | 30 | 19245 | 932 | 503 | 0.044 | 0.000 | 0.49 | -1.1% (-2.1% to -0.1%) | 0.692 |
| stroke | network | default plan (local noise) | 2 | an outsider with the final model | 30 | 19245 | 932 | 501 | 0.035 | 0.000 | 1.06 | 1.2% (-0.1% to 2.5%) | 0.731 |
| stroke | network | default plan (local noise) | 2 | the plan's attacker, every round | 30 | 19245 | 932 | 528 | 0.152 | 0.000 | 1.06 | 0.5% (-0.7% to 1.6%) | 0.731 |
| stroke | network | default plan (local noise) | 4 | an outsider with the final model | 30 | 19245 | 932 | 486 | 0.000 | 0.000 | 2.11 | 1.0% (-0.7% to 2.8%) | 0.749 |
| stroke | network | default plan (local noise) | 4 | the plan's attacker, every round | 30 | 19245 | 932 | 560 | 0.292 | 0.000 | 2.11 | 0.9% (-0.4% to 2.2%) | 0.749 |
| stroke | network | default plan (local noise) | 8 | an outsider with the final model | 30 | 19245 | 932 | 498 | 0.022 | 0.000 | 4.01 | 0.9% (-0.7% to 2.4%) | 0.782 |
| stroke | network | default plan (local noise) | 8 | the plan's attacker, every round | 30 | 19245 | 932 | 613 | 0.532 | 0.037 | 4.01 | 2.2% (1.0% to 3.5%) | 0.782 |
| stroke | logistic | default plan (local noise) | 0.5 | an outsider with the final model | 30 | 19245 | 932 | 477 | 0.000 | 0.000 | 0.21 | 0.2% (-1.2% to 1.8%) | 0.692 |
| stroke | logistic | default plan (local noise) | 0.5 | the plan's attacker, every round | 30 | 19245 | 932 | 504 | 0.048 | 0.000 | 0.21 | 0.5% (-0.4% to 1.5%) | 0.692 |
| stroke | logistic | default plan (local noise) | 1 | an outsider with the final model | 30 | 19245 | 932 | 476 | 0.000 | 0.000 | 0.49 | 0.3% (-1.2% to 1.8%) | 0.750 |
| stroke | logistic | default plan (local noise) | 1 | the plan's attacker, every round | 30 | 19245 | 932 | 518 | 0.109 | 0.000 | 0.49 | 0.5% (-0.5% to 1.4%) | 0.750 |
| stroke | logistic | default plan (local noise) | 2 | an outsider with the final model | 30 | 19245 | 932 | 487 | 0.000 | 0.000 | 1.06 | 0.6% (-0.6% to 1.9%) | 0.775 |
| stroke | logistic | default plan (local noise) | 2 | the plan's attacker, every round | 30 | 19245 | 932 | 549 | 0.243 | 0.000 | 1.06 | 1.1% (0.3% to 1.8%) | 0.775 |
| stroke | logistic | default plan (local noise) | 4 | an outsider with the final model | 30 | 19245 | 932 | 495 | 0.010 | 0.000 | 2.11 | 0.9% (-0.6% to 2.3%) | 0.790 |
| stroke | logistic | default plan (local noise) | 4 | the plan's attacker, every round | 30 | 19245 | 932 | 607 | 0.504 | 0.000 | 2.11 | 1.5% (0.5% to 2.6%) | 0.790 |
| stroke | logistic | default plan (local noise) | 8 | an outsider with the final model | 30 | 19245 | 932 | 489 | 0.000 | 0.000 | 4.01 | 0.9% (-0.5% to 2.4%) | 0.797 |
| stroke | logistic | default plan (local noise) | 8 | the plan's attacker, every round | 30 | 19245 | 932 | 654 | 0.730 | 0.168 | 4.01 | 2.8% (2.0% to 3.7%) | 0.797 |
| stroke | network | planted defect: default plan, noise switched off | 1 | an outsider with the final model | 30 | 19245 | 932 | 476 | 0.000 | 0.000 | 0.49 | 1.7% (0.4% to 3.1%) | 0.749 |
| stroke | network | planted defect: default plan, noise switched off | 1 | the plan's attacker, every round | 30 | 19245 | 932 | 642 | 0.671 | 0.044 | 0.49 | 18.8% (17.1% to 20.5%) | 0.749 |
| stroke | logistic | planted defect: default plan, noise switched off | 1 | an outsider with the final model | 30 | 19245 | 932 | 476 | 0.000 | 0.000 | 0.49 | -0.0% (-1.0% to 1.2%) | 0.791 |
| stroke | logistic | planted defect: default plan, noise switched off | 1 | the plan's attacker, every round | 30 | 19245 | 932 | 864 | 2.315 | 1.481 | 0.49 | 9.5% (8.1% to 10.9%) | 0.791 |
| stroke | logistic | shared noise, ordinary | 1 | an outsider with the final model | 30 | 14938 | 708 | 366 | 0.000 | 0.000 | 0.48 | 0.3% (-1.2% to 1.8%) | 0.782 |
| stroke | logistic | shared noise, ordinary | 1 | the plan's attacker, every round | 30 | 14938 | 708 | 369 | 0.000 | 0.000 | 0.48 | -0.6% (-1.8% to 0.7%) | 0.782 |
| stroke | logistic | shared noise, ordinary | 8 | an outsider with the final model | 30 | 14938 | 708 | 384 | 0.038 | 0.000 | 3.82 | 0.9% (-0.7% to 2.7%) | 0.818 |
| stroke | logistic | shared noise, ordinary | 8 | the plan's attacker, every round | 30 | 14938 | 708 | 438 | 0.349 | 0.000 | 3.82 | 1.0% (-0.2% to 2.1%) | 0.818 |
| stroke | logistic | shared noise, whole numbers | 1 | an outsider with the final model | 30 | 14938 | 708 | 366 | 0.000 | 0.000 | 0.76 | 1.6% (-0.2% to 3.5%) | 0.779 |
| stroke | logistic | shared noise, whole numbers | 1 | the plan's attacker, every round | 30 | 14938 | 708 | 390 | 0.072 | 0.000 | 0.76 | 3.0% (1.2% to 4.6%) | 0.779 |
| stroke | logistic | shared noise, whole numbers | 8 | an outsider with the final model | 30 | 14938 | 708 | 380 | 0.015 | 0.000 | 5.04 | 1.0% (-1.0% to 3.0%) | 0.807 |
| stroke | logistic | shared noise, whole numbers | 8 | the plan's attacker, every round | 30 | 14938 | 708 | 473 | 0.560 | 0.000 | 5.04 | 2.0% (0.3% to 3.7%) | 0.807 |
| loan | network | no protection | none | an outsider with the final model | 30 | 22350 | 1080 | 542 | 0.000 | 0.000 | n/a | -0.2% (-1.4% to 1.0%) | 0.978 |
| loan | network | no protection | none | the plan's attacker, every round | 30 | 22350 | 1080 | 647 | 0.293 | 0.000 | n/a | 10.4% (9.0% to 11.8%) | 0.978 |
| loan | logistic | no protection | none | an outsider with the final model | 30 | 22350 | 1080 | 542 | 0.000 | 0.000 | n/a | -0.3% (-1.5% to 1.0%) | 0.968 |
| loan | logistic | no protection | none | the plan's attacker, every round | 30 | 22350 | 1080 | 728 | 0.613 | 0.031 | n/a | 6.9% (5.3% to 8.3%) | 0.968 |
| loan | network | default plan (local noise) | 0.5 | an outsider with the final model | 30 | 22350 | 1080 | 549 | 0.000 | 0.000 | 0.22 | -0.0% (-1.2% to 1.2%) | 0.961 |
| loan | network | default plan (local noise) | 0.5 | the plan's attacker, every round | 30 | 22350 | 1080 | 578 | 0.034 | 0.000 | 0.22 | 2.8% (1.6% to 4.0%) | 0.961 |
| loan | network | default plan (local noise) | 1 | an outsider with the final model | 30 | 22350 | 1080 | 565 | 0.000 | 0.000 | 0.51 | 0.3% (-0.7% to 1.4%) | 0.965 |
| loan | network | default plan (local noise) | 1 | the plan's attacker, every round | 30 | 22350 | 1080 | 630 | 0.228 | 0.000 | 0.51 | 4.3% (3.1% to 5.6%) | 0.965 |
| loan | network | default plan (local noise) | 2 | an outsider with the final model | 30 | 22350 | 1080 | 557 | 0.000 | 0.000 | 1.06 | -0.2% (-1.1% to 0.7%) | 0.966 |
| loan | network | default plan (local noise) | 2 | the plan's attacker, every round | 30 | 22350 | 1080 | 707 | 0.527 | 0.061 | 1.06 | 8.0% (6.7% to 9.3%) | 0.966 |
| loan | network | default plan (local noise) | 4 | an outsider with the final model | 30 | 22350 | 1080 | 533 | 0.000 | 0.000 | 2.05 | -0.2% (-1.4% to 1.0%) | 0.971 |
| loan | network | default plan (local noise) | 4 | the plan's attacker, every round | 30 | 22350 | 1080 | 730 | 0.621 | 0.119 | 2.05 | 7.6% (6.1% to 9.1%) | 0.971 |
| loan | network | default plan (local noise) | 8 | an outsider with the final model | 30 | 22350 | 1080 | 528 | 0.000 | 0.000 | 3.78 | 0.2% (-1.1% to 1.3%) | 0.971 |
| loan | network | default plan (local noise) | 8 | the plan's attacker, every round | 30 | 22350 | 1080 | 744 | 0.679 | 0.177 | 3.78 | 10.4% (9.0% to 11.7%) | 0.971 |
| loan | logistic | default plan (local noise) | 0.5 | an outsider with the final model | 30 | 22350 | 1080 | 541 | 0.000 | 0.000 | 0.22 | -0.8% (-1.7% to 0.3%) | 0.964 |
| loan | logistic | default plan (local noise) | 0.5 | the plan's attacker, every round | 30 | 22350 | 1080 | 601 | 0.119 | 0.000 | 0.22 | 3.6% (2.2% to 5.0%) | 0.964 |
| loan | logistic | default plan (local noise) | 1 | an outsider with the final model | 30 | 22350 | 1080 | 540 | 0.000 | 0.000 | 0.51 | -0.5% (-1.6% to 0.7%) | 0.966 |
| loan | logistic | default plan (local noise) | 1 | the plan's attacker, every round | 30 | 22350 | 1080 | 648 | 0.296 | 0.000 | 0.51 | 5.3% (4.2% to 6.3%) | 0.966 |
| loan | logistic | default plan (local noise) | 2 | an outsider with the final model | 30 | 22350 | 1080 | 535 | 0.000 | 0.000 | 1.06 | -0.4% (-1.6% to 1.0%) | 0.967 |
| loan | logistic | default plan (local noise) | 2 | the plan's attacker, every round | 30 | 22350 | 1080 | 717 | 0.568 | 0.061 | 1.06 | 8.2% (7.1% to 9.3%) | 0.967 |
| loan | logistic | default plan (local noise) | 4 | an outsider with the final model | 30 | 22350 | 1080 | 534 | 0.000 | 0.000 | 2.05 | -0.4% (-1.6% to 1.0%) | 0.967 |
| loan | logistic | default plan (local noise) | 4 | the plan's attacker, every round | 30 | 22350 | 1080 | 722 | 0.588 | 0.061 | 2.05 | 6.4% (5.1% to 7.5%) | 0.967 |
| loan | logistic | default plan (local noise) | 8 | an outsider with the final model | 30 | 22350 | 1080 | 533 | 0.000 | 0.000 | 3.78 | 0.0% (-1.1% to 1.4%) | 0.967 |
| loan | logistic | default plan (local noise) | 8 | the plan's attacker, every round | 30 | 22350 | 1080 | 705 | 0.519 | 0.031 | 3.78 | 7.7% (6.3% to 9.1%) | 0.967 |
| loan | network | planted defect: default plan, noise switched off | 1 | an outsider with the final model | 30 | 22350 | 1080 | 550 | 0.000 | 0.000 | 0.51 | -0.4% (-1.4% to 0.6%) | 0.967 |
| loan | network | planted defect: default plan, noise switched off | 1 | the plan's attacker, every round | 30 | 22350 | 1080 | 787 | 0.868 | 0.295 | 0.51 | 30.1% (28.7% to 31.7%) | 0.967 |
| loan | logistic | planted defect: default plan, noise switched off | 1 | an outsider with the final model | 30 | 22350 | 1080 | 539 | 0.000 | 0.000 | 0.51 | -0.4% (-1.5% to 0.7%) | 0.967 |
| loan | logistic | planted defect: default plan, noise switched off | 1 | the plan's attacker, every round | 30 | 22350 | 1080 | 821 | 1.028 | 0.418 | 0.51 | 15.3% (14.0% to 16.5%) | 0.967 |
| loan | logistic | shared noise, ordinary | 1 | an outsider with the final model | 30 | 17880 | 840 | 418 | 0.000 | 0.000 | 0.50 | 0.8% (-0.4% to 2.1%) | 0.967 |
| loan | logistic | shared noise, ordinary | 1 | the plan's attacker, every round | 30 | 17880 | 840 | 505 | 0.287 | 0.000 | 0.50 | 2.8% (1.8% to 3.9%) | 0.967 |
| loan | logistic | shared noise, ordinary | 8 | an outsider with the final model | 30 | 17880 | 840 | 417 | 0.000 | 0.000 | 3.88 | 0.4% (-0.9% to 1.9%) | 0.967 |
| loan | logistic | shared noise, ordinary | 8 | the plan's attacker, every round | 30 | 17880 | 840 | 566 | 0.597 | 0.020 | 3.88 | 5.7% (4.2% to 7.2%) | 0.967 |
| loan | logistic | shared noise, whole numbers | 1 | an outsider with the final model | 30 | 17880 | 840 | 420 | 0.000 | 0.000 | 0.77 | 0.6% (-0.8% to 2.0%) | 0.966 |
| loan | logistic | shared noise, whole numbers | 1 | the plan's attacker, every round | 30 | 17880 | 840 | 490 | 0.214 | 0.000 | 0.77 | 3.2% (1.9% to 4.6%) | 0.966 |
| loan | logistic | shared noise, whole numbers | 8 | an outsider with the final model | 30 | 17880 | 840 | 418 | 0.000 | 0.000 | 5.10 | 0.1% (-1.3% to 1.5%) | 0.967 |
| loan | logistic | shared noise, whole numbers | 8 | the plan's attacker, every round | 30 | 17880 | 840 | 565 | 0.591 | 0.020 | 5.10 | 8.6% (7.4% to 9.8%) | 0.967 |
