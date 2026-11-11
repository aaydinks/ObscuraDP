# Phase 3 results: does sharing the noise give better models at the same budget?

Date: 2026-11-11. Written in step S14. Every number below comes from
`benchmarks/results/phase3/summary.json`, which is built from 960 stored runs by
`python -m benchmarks.phase3.report`. The full tables are in `benchmarks/results/phase3/summary.md`.

## In one paragraph

Yes on the small, hard table, and it makes no real difference on the easy one. On the stroke table,
letting the participants share the noise (steps S12 and S13) gives a better model than letting each
participant add all of its own noise, at the same total budget: by 0.01 to 0.03 in AUC for logistic
regression at every budget from 1 upwards, and by about 0.04 to 0.05 for the network from a budget of 2
(ordinary noise) or 4 (whole-number noise) upwards. At the strictest budgets the averages also favour
sharing, but the runs scatter too much to say so with confidence. On the loan table every arm is within
0.007 of every other. Protecting whole participants is useless with five participants, as expected. The
gain has a price: the shared-noise modes need many more rounds, so a participant uploads more in total,
and they rest on more assumptions than local noise does.

## What was run

- **Tables, participants, splits:** as in Phase 2 (`docs/GATE_PHASE2.md`): the stroke and loan tables,
  five participants, each keeping 70% of its rows for training, 15% for choosing the cut-off and 15% for
  the final score.
- **Seeds:** ten for stroke (42 to 51), five for loan (42 to 46). The same seed means the same split for
  every arm, so arms are compared seed by seed.
- **Budgets:** a total $\varepsilon$ of 0.5, 1, 2, 4 and 8 for the whole run at $\delta = 10^{-5}$.
- **Models:** the research network and logistic regression.
- **Arms** (ways of protecting a run):

| Arm | What it is |
|---|---|
| No protection | The baseline, with the Phase 2 protocol |
| Local noise (Phase 2 protocol) | Every participant adds all of its own noise; 20 rounds of 3 local passes. Repeats Phase 2 |
| Local noise, rounds tuned | The same, with the number of rounds and local passes chosen on made-up tables like the arms below |
| Shared noise, ordinary | Step S12: every participant adds a slice of the noise inside the secure sum; one assumed colluder |
| Shared noise, whole numbers | Step S13: the same with whole-number noise; every row in every round |
| Trusted server | The server adds the noise. The least noise any method needs; uses the shared arm's settings |
| Local noise through the secure sum, 12-bit and 8-bit values | For message sizes, at budgets 1 and 8 |
| Whole participants protected | The new mode of this step, at budgets 1 and 8 |

- **Settings** (learning rate, clipping, and for the new arms the number of rounds and the batch size)
  were chosen on made-up tables with the public shape of the real ones: 1,728 short runs, by mean
  validation AUC. No real row was used. The choices are in `benchmarks/phase3/settings.json`.
- **The yardstick is "the better local arm"**: of the two local arms, the one with the higher mean AUC at
  that table, model and budget. It is picked after looking at the held-out rows, which can only favour
  local noise.
- **When a difference counts:** only if the 95% interval of the seed-by-seed differences excludes zero.

## Results

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

What these numbers say:

1. **Stroke, logistic regression: sharing the noise is better from a budget of 1 upwards.** At a budget
   of 1 the AUC is 0.800 (ordinary noise) and 0.806 (whole-number noise) against 0.774 for local noise
   and 0.832 without protection. The gap to the unprotected model shrinks from 0.058 to about 0.03. At a
   budget of 8 ordinary shared noise reaches the unprotected model (0.832).
2. **Stroke, network: better from a budget of 2 (ordinary) or 4 (whole numbers) upwards**, by 0.04 to
   0.05. At budgets 0.5 and 1 the averages favour sharing by 0.05 to 0.07, but the intervals are wide:
   the network under local noise scatters a lot from seed to seed at strict budgets.
3. **Loan: no difference that matters.** Every arm that protects single rows is between 0.958 and 0.967. A few
   differences of 0.002 to 0.007 are marked for the network, far below anything a user would notice.
4. **Whole-number noise does about as well as ordinary noise**, although it needs 14% more noise and
   uses every row in every round. On logistic regression it is slightly behind at loose budgets (0.821
   against 0.832 at 8).
5. **The trusted server is not clearly ahead of shared noise.** It adds about a tenth less noise at the same
   settings, and its averages are within the scatter of the shared arm's. In other words: sharing the
   noise among five participants with one assumed colluder gives up almost nothing against a server one
   has to trust completely.
6. **Choosing rounds on made-up tables did not help local noise.** The "rounds tuned" arm is worse than
   the Phase 2 protocol at strict budgets on stroke (by 0.06 to 0.10 for logistic regression). This is
   why the yardstick is the better of the two.
7. **Whole participants cannot be protected in a group of five.** AUC 0.47 to 0.54 at a budget of 1, no
   better than guessing. That is what the mode's warning says, now with numbers.
8. **Rounding to 8 bits costs accuracy in one place.** Local noise through the secure sum with 12-bit
   values equals local noise in the clear everywhere (differences below 0.001). With 8-bit values the
   stroke network at a budget of 1 loses 0.022 (interval 0.003 to 0.041); elsewhere nothing shows. The
   default of 12 bits stays.

## Messages

Per round, against the same update as 32-bit numbers (the website's target is at most 0.5):

| Arm | Bits per number | Per round |
|---|---|---|
| Local noise in the clear | 32 | 1.00 |
| Local noise through the secure sum, 12-bit values | 15 | 0.47 to 0.48 |
| Local noise through the secure sum, 8-bit values | 11 | 0.34 to 0.36 |
| Shared noise, ordinary | 12 to 13 | 0.38 to 0.41 |
| Shared noise, whole numbers | 18 to 22 | 0.57 to 0.69 |
| Whole participants protected | 10 | 0.31 to 0.32 |
| Encryption (CKKS, research only) | n/a | 17 times plain for the stroke network, 59 times for the loan network |

Over the whole run the picture turns around, because the shared-noise arms need 60 to 600 rounds where
the local arms need 20. At a budget of 1 on the stroke network one participant uploads 1.1 MB with
ordinary shared noise and 1.9 MB with whole-number noise, against 0.28 MB with local noise in the clear
and 0.13 MB with local noise through the secure sum. So: **smaller per round, larger in total.** One
encrypted update is about 235 kB whatever the model size, because one ciphertext holds up to 4,096
numbers and is sent whole.

## Where the website targets stand after Phase 3

| Target | Now |
|---|---|
| At least 50% fewer bytes per round | Met by local noise through the secure sum (0.47) and by shared ordinary noise (0.38 to 0.41). **Not met** by shared whole-number noise (0.57 to 0.69). The target says "per round"; in total the shared-noise arms upload more |
| At most 5% attacker advantage at a total budget of 1 | Unchanged: measured in step S15. The proven limits are in each run's privacy report |
| Mathematically guaranteed privacy | All 930 protected runs stayed within budget and every stored report recomputes. The trusted-server runs carry their unsafe flag, by design. Shared ordinary noise still rests on one argued step (rounding); whole-number noise does not |

## Limits of this evidence

- **Two tables, five participants, one assumed colluder.** Nothing here says how the arms compare with
  fifty participants, with larger tables, or when more participants are assumed to collude (the noise
  grows as fewer slices can be counted on).
- **Small test sets.** Stroke has about 31 positive held-out rows per seed. Ten seeds make differences of
  0.02 to 0.03 visible for logistic regression; for the network at strict budgets they do not suffice.
- **Settings at the edge of what was tried.** Several chosen values are the largest on their grid (600
  rounds, a batch of 640, a learning rate of 3). The shared arms might do better still with more rounds,
  and so might others.
- **Made-up tables are an imperfect guide.** They made the tuned local arm worse, not better. The same
  may hold in the other direction for any arm.
- **The unprotected baseline keeps the Phase 2 protocol** and was not given more rounds. "Reaches the
  unprotected model" means that baseline.
- **More rounds mean more of everything else:** more messages, more opportunities for a participant to be
  missing (a failed round costs budget in the shared modes), and more time. None of that is in the AUC.
- **Noise came from the run's seed** so that runs can be repeated, as in Phase 2.

## What this means for the default

Recorded as a proposal in `docs/adr/0005-default-plan.md`. It is the owner's decision.
