# Phase 4, further attacks: shadow models, a hidden column, reading records back

Built by `python -m benchmarks.phase4.report_more` from 384 shadow runs, 120 audited runs on the rare rows and one experiment on single updates.

**Stroke table: an outsider with the final model, with and without shadow models** (advantage = true-positive rate minus false-positive rate; mean of 32 runs with a 95% interval)

| Model | Protection | Limit for the noise used | One cut-off for all records | Shadow models | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| network | no protection | n/a | 0.9% (0.4% to 1.4%) | 0.3% (-0.2% to 0.8%) | 0.01 | n/a |
| network | default plan, budget 1 | 10.6% | 0.2% (-0.4% to 0.8%) | -0.2% (-0.8% to 0.4%) | 0.00 | 0.56 |
| network | default plan, budget 8 | 58.5% | 0.5% (-0.2% to 1.2%) | -0.0% (-0.5% to 0.5%) | 0.00 | 4.12 |
| logistic | no protection | n/a | 0.7% (0.1% to 1.2%) | -0.5% (-0.9% to -0.0%) | 0.00 | n/a |
| logistic | default plan, budget 1 | 10.6% | -0.0% (-0.5% to 0.5%) | -0.2% (-0.6% to 0.3%) | 0.00 | 0.56 |
| logistic | default plan, budget 8 | 58.5% | 0.8% (0.1% to 1.5%) | -0.2% (-0.6% to 0.3%) | 0.00 | 4.12 |

**Loan table: an outsider with the final model, with and without shadow models** (advantage = true-positive rate minus false-positive rate; mean of 32 runs with a 95% interval)

| Model | Protection | Limit for the noise used | One cut-off for all records | Shadow models | Lower limit on the budget shown | Most the test could show |
|---|---|---|---|---|---|---|
| network | no protection | n/a | -0.3% (-0.8% to 0.2%) | 0.1% (-0.5% to 0.7%) | 0.00 | n/a |
| network | default plan, budget 1 | 10.6% | 0.1% (-0.4% to 0.7%) | 0.6% (0.1% to 1.2%) | 0.00 | 0.55 |
| network | default plan, budget 8 | 56.4% | -0.1% (-0.6% to 0.4%) | 0.4% (-0.3% to 1.1%) | 0.00 | 3.90 |
| logistic | no protection | n/a | 0.0% (-0.5% to 0.6%) | 1.0% (0.3% to 1.7%) | 0.00 | n/a |
| logistic | default plan, budget 1 | 10.6% | 0.2% (-0.3% to 0.7%) | 0.1% (-0.5% to 0.7%) | 0.00 | 0.55 |
| logistic | default plan, budget 8 | 56.4% | 0.1% (-0.4% to 0.7%) | 1.0% (0.4% to 1.6%) | 0.00 | 3.90 |

**Stroke table: the rare class apart** (advantage by class; shadow models: mean of 32 runs; the plan's attacker: mean of 30 runs in which every row of the rare class is a test record)

| Model | Protection | Attack | Limit for the noise used | Rows of the rare class | Rows of the common class |
|---|---|---|---|---|---|
| network | no protection | shadow models (outsider) | n/a | 11.0% (8.5% to 13.8%) | -0.5% (-0.9% to 0.0%) |
| network | default plan, budget 1 | shadow models (outsider) | 10.6% | -2.2% (-4.7% to 0.3%) | 0.2% (-0.4% to 0.9%) |
| network | default plan, budget 8 | shadow models (outsider) | 58.5% | -0.9% (-3.7% to 2.1%) | 0.3% (-0.3% to 0.9%) |
| logistic | no protection | shadow models (outsider) | n/a | 12.4% (9.4% to 15.6%) | -0.4% (-0.9% to -0.0%) |
| logistic | default plan, budget 1 | shadow models (outsider) | 10.6% | -2.0% (-3.9% to -0.2%) | -0.2% (-0.7% to 0.2%) |
| logistic | default plan, budget 8 | shadow models (outsider) | 58.5% | 3.0% (0.2% to 6.1%) | -0.4% (-0.9% to 0.1%) |
| network | default plan, budget 1 | the plan's attacker, every round | 10.7% | 5.4% (2.4% to 8.6%) | 0.2% (-2.5% to 2.9%) |
| network | default plan, budget 8 | the plan's attacker, every round | 59.5% | 22.8% (19.2% to 26.5%) | 3.1% (0.4% to 6.3%) |
| logistic | default plan, budget 1 | the plan's attacker, every round | 10.7% | 9.0% (5.7% to 12.9%) | 0.4% (-2.1% to 3.1%) |
| logistic | default plan, budget 8 | the plan's attacker, every round | 59.5% | 29.9% (26.5% to 33.6%) | 0.9% (-1.4% to 3.4%) |

**Stroke table: guessing a hidden yes/no column** (balanced accuracy, 50% is guessing; mean of 32 runs)

| Hidden column | Model | Protection | Without any model | With the model, used rows | With the model, unused rows | Used minus unused (points) |
|---|---|---|---|---|---|---|
| hypertension | network | no protection | 52.1% | 51.4% | 51.3% | +0.1 (-0.1 to +0.4) |
| hypertension | network | default plan, budget 1 | 52.1% | 51.6% | 51.4% | +0.2 (-0.2 to +0.6) |
| hypertension | network | default plan, budget 8 | 52.1% | 54.3% | 53.9% | +0.5 (-0.0 to +1.0) |
| hypertension | logistic | no protection | 52.1% | 50.8% | 50.8% | +0.0 (-0.2 to +0.3) |
| hypertension | logistic | default plan, budget 1 | 52.1% | 52.5% | 52.6% | -0.1 (-0.6 to +0.3) |
| hypertension | logistic | default plan, budget 8 | 52.1% | 54.1% | 54.1% | -0.1 (-0.6 to +0.5) |
| heart_disease | network | no protection | 51.6% | 51.5% | 51.8% | -0.2 (-0.7 to +0.2) |
| heart_disease | network | default plan, budget 1 | 51.6% | 51.0% | 51.1% | -0.1 (-0.4 to +0.3) |
| heart_disease | network | default plan, budget 8 | 51.6% | 52.7% | 52.5% | +0.1 (-0.4 to +0.6) |
| heart_disease | logistic | no protection | 51.6% | 51.0% | 51.0% | -0.1 (-0.4 to +0.3) |
| heart_disease | logistic | default plan, budget 1 | 51.6% | 52.6% | 52.6% | +0.0 (-0.5 to +0.6) |
| heart_disease | logistic | default plan, budget 8 | 51.6% | 53.6% | 53.1% | +0.5 (-0.1 to +1.2) |
| ever_married | network | no protection | 84.2% | 84.5% | 84.1% | +0.3 (-0.1 to +0.8) |
| ever_married | network | default plan, budget 1 | 84.2% | 84.5% | 84.2% | +0.3 (-0.1 to +0.7) |
| ever_married | network | default plan, budget 8 | 84.2% | 84.5% | 84.1% | +0.4 (-0.1 to +0.8) |
| ever_married | logistic | no protection | 84.2% | 84.4% | 84.2% | +0.3 (-0.2 to +0.7) |
| ever_married | logistic | default plan, budget 1 | 84.2% | 84.4% | 84.1% | +0.3 (-0.1 to +0.8) |
| ever_married | logistic | default plan, budget 8 | 84.2% | 84.4% | 84.1% | +0.4 (-0.1 to +0.8) |

**Loan table: guessing a hidden yes/no column** (balanced accuracy, 50% is guessing; mean of 32 runs)

| Hidden column | Model | Protection | Without any model | With the model, used rows | With the model, unused rows | Used minus unused (points) |
|---|---|---|---|---|---|---|
| self_employed | network | no protection | 48.8% | 49.4% | 48.7% | +0.7 (+0.1 to +1.3) |
| self_employed | network | default plan, budget 1 | 48.8% | 49.6% | 48.9% | +0.7 (+0.1 to +1.2) |
| self_employed | network | default plan, budget 8 | 48.8% | 49.1% | 48.3% | +0.8 (+0.2 to +1.4) |
| self_employed | logistic | no protection | 48.8% | 49.1% | 48.6% | +0.5 (-0.1 to +1.1) |
| self_employed | logistic | default plan, budget 1 | 48.8% | 49.1% | 48.6% | +0.5 (-0.1 to +1.0) |
| self_employed | logistic | default plan, budget 8 | 48.8% | 49.0% | 48.5% | +0.5 (-0.1 to +1.0) |
| education | network | no protection | 48.8% | 49.8% | 49.5% | +0.3 (-0.3 to +0.8) |
| education | network | default plan, budget 1 | 48.8% | 49.5% | 49.7% | -0.2 (-0.7 to +0.2) |
| education | network | default plan, budget 8 | 48.8% | 49.6% | 49.7% | -0.1 (-0.6 to +0.4) |
| education | logistic | no protection | 48.8% | 49.6% | 49.5% | +0.1 (-0.4 to +0.7) |
| education | logistic | default plan, budget 1 | 48.8% | 49.7% | 49.6% | +0.1 (-0.5 to +0.6) |
| education | logistic | default plan, budget 8 | 48.8% | 49.7% | 49.5% | +0.2 (-0.3 to +0.7) |

**Reading a record back out of an update** (median distance between the record and what is read back, as a share of the record's length: 0 is a perfect copy, 1 is as far off as answering nothing)

| What the attacker holds | stroke, network | stroke, logistic | loan, network | loan, logistic |
|---|---|---|---|---|
| alone, no protection | 0.00 | 0.00 | 0.00 | 0.00 |
| one of 32, no protection | 0.63 | 0.64 | 0.94 | 0.99 |
| alone, default plan, budget 8 | 1.02 | 2.69 | 1.04 | 2.52 |
| alone, default plan, budget 1 | 1.02 | 2.38 | 1.03 | 2.34 |
| no update at all (the average record) | 0.65 | 0.65 | 0.78 | 0.78 |
