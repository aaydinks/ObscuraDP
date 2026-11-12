# Phase 4, further attacks: a second opinion, a hidden column, and reading records back

Date: 2026-11-12. Written in step S16. Every number below comes from
`benchmarks/results/phase4/more/summary.json`, which is built by
`python -m benchmarks.phase4.report_more` from 384 runs on random halves of the tables, 120 audited runs
on the rare rows of the stroke table and one experiment on single updates. This memo continues
`docs/PHASE4_ATTACK_TESTS.md` (step S15) and **corrects one of its statements**.

## In one paragraph

Three further attacks were pointed at the default plan, and one group of rows was looked at apart.
**The correction first: on the stroke table the 5% target does not hold for the rows of the rare class.**
Step S15 measured an average over all rows and found no more than 0.5% at a total budget of 1. Read by
class, the attacker the plan is written for finds rows of the rare class (the stroke cases) with an
advantage of 5.4% (network) and 9.0% (logistic regression) at a budget of 1, and 23% to 30% at a
budget of 8, while rows of the common class stay at about 0%. All of it is within the proven limits
(10.7% and 59.5%), so nothing is broken; but the people the stroke table is about are far easier to
find than the average suggested. The other results are reassuring. The strongest known attack for an
outsider with the final model (shadow models) finds nothing above 1% in any protected cell. A hidden
yes/no column is guessed no better with the model than without it, and no better for rows that were
used than for rows that were not. A record that stands alone in an unprotected update can be read back
exactly; behind the default plan's noise, or among 31 other records, what is read back is no closer to
the record than the table's average row.

## What was asked

Step S15 left three things open: a second opinion on "was this record used?" from an attack that
judges every record against itself; whether a model gives away single columns of a record; and whether
whoever sees an update can read a record back out of it. It also left a caveat: the stroke runs give the
rare class more weight, which shortens every row of the common class to about 3% of the clipping
length, so the average over all rows might hide the rows that matter.

## How each test works

**Shadow models** (`obscuradp.audit.shadow`). The attacker is an outsider with the final model. Whether
a model fits a record well says little by itself, because some records are easy and others hard
whether or not they were used. So the whole pipeline is run 32 times, each time on a random half of
every participant's rows (public sizes held fixed, as in S15). Every row is then in the training rows
of about half of the runs. For a run under attack, each row's fit is compared with how the same row
fared in the other runs when it was in and when it was out (the likelihood-ratio attack of Carlini and
others, 2022). This is the strongest known attack of its kind, and nobody outside could run it without
the participants' rows; it is a measuring tool. For the lower limit on the budget, runs 0 to 15 are
attacked with runs 16 to 31 as shadows only, so that their guesses may be pooled (5% of the rows
guessed for, as fixed in S15).

**The rare rows apart.** On the stroke table the whole-run audit of S15 is repeated with every row of
the rare class as a test record (147 per run on average), beside equally many rows of the common class,
and the measured advantage is read by class. 30 runs per cell, new seeds (300 to 329).

**A hidden column** (`obscuradp.audit.attribute`). The attacker knows a record except for one yes/no
column, knows its outcome, and has the model. It tries both values and asks under which the known
outcome is more plausible, starting from what the other columns already say. Two yardsticks decide
whether that is a leak: an attacker **without any model**, who guesses the column from the other
columns and the outcome with nothing but rows of the same population; and the same attack on rows that
were **not used** in training. Quality is balanced accuracy: 50% is guessing, whatever the mix of
values. The models are those of the shadow runs, where it is known which rows were used.

**Reading a record back** (`obscuradp.audit.inversion`). For the models of this library there is an
exact way when a record stands alone in an update: in the first layer, the gradient for the weights is
the gradient for the bias times the record itself, so one divided by the other is the record. The
experiment takes 200 real rows per table and asks how far what is read back is from the row, for five
things an attacker might hold.

## Results

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

What these numbers say:

1. **Rows of the rare class are found far more easily than the average row.** By the plan's attacker,
   at a total budget of 1: 5.4% (network) and 9.0% (logistic regression) for the rare class, against
   0.2% and 0.4% for the common class. At a budget of 8: 22.8% and 29.9% against 3.1% and 0.9%. The
   proven limits are 10.7% and 59.5%, so the plan keeps its promise; it is the average that misled.
   The cause is the class weighting: a row of the rare class may pull with the full clipping length,
   a row of the common class with about 3% of it. The budget is the same for both, but only the rare
   rows come near what it allows.
2. **Part of the rise against S15 is the test itself.** With about 290 test records per run instead of
   640, the attacker has fewer unknowns to untangle and does better per record. That does not explain
   the gap between the classes, which are measured in the same runs.
3. **An outsider with the final model finds nothing behind the default plan, shadow models or not.** In
   every protected cell the advantage is between -0.2% and 1.0%, and the pooled lower limit on the
   budget is 0.00, where the test could have shown up to 0.56 (budget 1) or 4.1 (budget 8). For the
   rare class alone the outsider gets nothing at a budget of 1 and 3.0% (interval 0.2% to 6.1%) in one
   cell at a budget of 8.
4. **Without protection the outsider does find the rare rows**: 11.0% and 12.4% for the rare class of
   the stroke table, nothing for the common class and nothing on the loan table. One cut-off for all
   records (the S15 outsider) cannot see this; judging each record against itself can. This is what
   the default plan removes at a budget of 1.
5. **The model gives away no column.** Whether someone ever married is guessed right 84% of the time
   without any model, from age and the other columns; with the model it is the same 84%. The other
   columns cannot be guessed by anyone (49% to 54%). Rows that were used are guessed no better than
   rows that were not: every difference is below one point. The one difference whose interval excludes
   zero (self-employment on the loan table, +0.7 points at an accuracy of 49%) is the same with and
   without protection; the cells share their coins, so it reflects which rows the coins happened to
   pick, not the training.
6. **A record alone in an unprotected update is read back exactly**, all 200 of 200 in every cell. This
   is what a server sees of a participant that sends plain gradients of single records, and what a
   trusted server sees in the trusted-server plan.
7. **Behind the default plan's noise nothing is read back.** What comes out is 1.0 to 2.7 record lengths
   away from the record, further than the table's average row (0.65 and 0.78), at a budget of 8 as
   much as at 1. Not one of 200 records is within a tenth of its length. Among 31 other records
   without protection the exact method does no better than the average row either. In the shared-noise
   plans the server never holds less than the sum over all participants' rows plus noise.

## What changes in the S15 account

- `docs/PHASE4_ATTACK_TESTS.md` said the 5% target "holds on the stroke table". That stands for the
  average row only. **For the rows of the rare class it does not hold** (5.4% and 9.0% at a budget of
  1). A note was added there.
- The target is therefore met for no table as a whole: not on the loan table (4.3% and 5.3%), and not
  for the rare class of the stroke table.
- What can be said: at a total budget of 1 the strongest attacker's advantage was between 0% and 9%,
  depending on the table and on the kind of row, against a proven worst case of 10.7%; an outsider who
  only gets the finished model learned nothing measurable.

## Limits of this evidence

- **Two tables, five participants, two small models**, as before. Larger models memorize more; the
  shadow-model attack in particular is known to find much more there.
- **The 32 shadow runs of a cell share one split of the table and learn from each other.** Their
  intervals are a guide. The pooled lower limit uses a separate group of shadows and is valid.
- **The by-class figures rest on about 147 rare rows per run.** Their intervals are three points wide
  either way; the order of size is clear, the second digit is not.
- **Class weighting is a setting, not a law.** Without it the rare rows would be less exposed and, as
  Phase 2 showed, the model would be worse at finding them. That trade was not explored here.
- **The hidden-column attack is the simple one**: one column, two values, the model's fit as evidence.
  Columns with many values, or several hidden columns at once, were not tried.
- **Reading records back was tried with the exact method only.** Methods that search for a record by
  trial and error can do better among several records without protection; they cannot remove the
  default plan's noise, which is what the budget counts.
- **Noise came from the run's seed** so that runs can be repeated.
