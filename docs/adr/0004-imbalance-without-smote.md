# 0004. Class imbalance without SMOTE inside private training

Status: accepted by the owner on 2026-11-10.

## Context

Stroke cases are 4.3% of the data. Experiment 1 and Trial 07/27 balance each participant's data with
SMOTETomek before training with privacy noise (G3). SMOTE builds each new row by blending real rows, so one
real patient influences many training rows, and the privacy budget no longer describes that patient. In the
largest participant, 2,460 positive rows are built from 22 real ones.

## Decision

1. Inside any run with a privacy budget, non-private oversampling is refused. It can be forced with
   `allow_unsafe=True`, which puts a flag in the report saying the stated budget does not hold for real
   records.
2. The supported alternative is a class-weighted loss whose weights come from a public prior (for example a
   published prevalence), not from private counts.
3. A second alternative, choosing the decision threshold on a validation split, is offered because it needs
   no extra information release beyond the trained model.
4. A private oversampling method may be added later only with a proof of how much budget it spends.

## Consequences

- The stroke results change and must be re-run (S11). The earlier observation that the private arm beat the
  non-private arm on stroke is expected to disappear, since it came from the resampling recipe.
- The Experiment 1 replication stays in the repository as a replication of the paper, with a note that its
  budget does not hold for real records.
- The rare-outcome task in the benchmark uses synthetic health data with class weighting (S17).
