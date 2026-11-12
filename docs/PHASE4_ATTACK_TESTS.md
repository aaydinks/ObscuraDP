# Phase 4 attack tests: what can an attacker tell about a single record?

Date: 2026-11-11. Written in step S15. Every number below comes from
`benchmarks/results/phase4/summary.json`, which is built from 1,080 stored runs by
`python -m benchmarks.phase4.report`. The full tables are in `benchmarks/results/phase4/summary.md`.

**Corrected in step S16 (2026-11-12).** Where this memo says the 5% target holds on the stroke table,
that is true of the average row only. Read by class, the rows of the rare class are found with an
advantage of 5.4% and 9.0% at a total budget of 1 (`docs/PHASE4_MORE_ATTACKS.md`). The numbers below are
unchanged; the sentences they support are marked.

## In one paragraph

The target "at most 5% attacker advantage at a budget of 1" **holds on the stroke table (for the
average row only, see the correction above) and does not hold on the loan table**, measured against the
stronger of the two attacks on whole runs. On the loan
table the attacker the default plan is written for reaches an advantage of 4.3% (network) and 5.3%
(logistic regression) at a total budget of 1, with intervals that reach 5.6% and 6.3%; on the stroke
table it does no better than a coin (-1.1% and 0.5%). All of this is below the proven limits (10.6% for
the noise that was added, 46.2% for any method with that budget), on the loan table at about half of
the first. An outsider who only gets the final model learns nothing measurable at any budget. No
protected run leaked more than its report states: in all 28 protected cells the lower limit on the
budget stays below the stated budget and the measured advantage below the proven limit. The test
catches what it has to catch: a planted defect (the noise switched off while the report still states a
budget of 1) is flagged in all four cells, and the auditor passes its self-test. Its reach is short,
though: with tables of this size it could prove at most about half of a budget of 1 even against a
perfect attacker, so a small lower limit here is not evidence of safety.

## What was asked

The website says "at most 5% attacker advantage at a budget of 1". "Attacker advantage" is how much more
often than chance an attacker correctly says "this record was used in training": the share of used
records it calls used, minus the share of unused records it calls used. 0 means a coin flip, 100% means
it is always right. The privacy budget (written $\varepsilon$, here always for a whole run at
$\delta = 10^{-5}$) puts a proven limit on that number, but at a budget of 1 the limit is far above 5%.
So the 5% can only be something that is measured against named attacks, and the decision of the Phase 0
gate was to publish the measured value beside the two proven limits, whatever it turns out to be
(`docs/KPI_MAPPING.md`). This step builds the attacks and measures.

Three numbers therefore stand side by side in every table:

1. **The limit for any method** with that budget: $\dfrac{e^{\varepsilon} - 1 + 2\delta}{e^{\varepsilon} + 1}$.
2. **The limit for the noise that was actually added**, from the run's own privacy report. It is lower,
   because it uses how the noise was added and not only the budget.
3. **The measured value**: what an attack achieved, with a 95% interval.

## How the test works

**One run, many test records.** In every run about a quarter of each participant's real training rows (at
most 250) are picked as test records. Each is kept in or left out by a fair coin, and the run trains once.
Afterwards an attacker gives every test record a score and has to say which way its coin fell. This is
the method of Steinke, Nasr and Jagielski (2023, "Privacy Auditing with One (1) Training Run"); their
routine was taken from the paper itself and is tested against the paper's own listing. A run has 745 test
records on the loan table and 641 on average on the stroke table (596 and 497 in the shared-noise arms,
where the participant that works with the server holds none).

**Sizes stay public and fixed.** A participant's declared number of rows is its full number whatever the
coins say. Otherwise the noise level and the chance of a row joining a step would move with the coins and
give them away through a side door the guarantee does not cover. This needed a new setting,
`Simulation(declared_rows=...)`.

**Two readings of the same scores.**

- *Measured advantage.* The cut-off between "in" and "out" is chosen on one fixed half of the test
  records and applied to the other half, and the other way round; the two results are averaged. The
  figure in the tables is the mean of 30 runs, and the interval comes from resampling whole runs.
- *A lower limit on the budget.* The attacker guesses only for the test records it is surest about. If it
  is right more often than a budget $\varepsilon$ allows, the true budget is larger than $\varepsilon$.
  The guesses of the 30 runs of a cell are pooled: thirty independent runs with their own coins are,
  taken together, one procedure in which a single record still enters only one run, so the paper's
  routine applies to the pooled counts. That step is this project's own (short) argument, not the
  paper's; `summary.md` also gives the median of the single runs.

**Three attackers, from weak to strong.**

| Attacker | Sees | Knows | Where |
|---|---|---|---|
| An outsider | the final model | the test records | `obscuradp.audit.membership`, "final model" |
| The attacker the plan is written for | what the plan lets it see in every round (table below) | the test records **and every other row** | `obscuradp.audit.membership`, "updates" |
| The strongest the guarantee allows | every single noisy step | everything but one hand-made worst-case record | `obscuradp.audit.mechanism` |

The second attacker works out what each round's messages would look like without the test records,
subtracts that, and solves for the coins that best explain what is left over (a least-squares fit with a
penalty). It measures the size of the noise from what it sees and does not take the pipeline's word for
it. The third is not run on real rows: it attacks the noise itself with a record built to be the worst
case, 1,500 to 2,000 times with and without that record. It is the check that existed inside the tests
since step S8 and is now a public tool.

**What the plan's attacker may see** (`obscuradp.audit.views`; the simulation hands out everything that
was sent, and this module is the only place that cuts it down):

| Plan | Attacker | Sees per round |
|---|---|---|
| Default plan, updates in the clear | the server | every participant's update |
| Default plan through the secure sum | the server | only the opened sum |
| Shared noise (ordinary or whole numbers) | the server and the assumed colluder | the opened sum, minus the colluder's own message |
| Trusted server | someone who only gets the models | by how much each round moved the model |

**What was fixed beforehand.** For how many test records an attack guesses decides how much the lower
limit can show, so it must not be picked after looking. A pilot on five other seeds (90 to 94, 120 runs)
chose it: the surest 5% of the records, for both attacks (`benchmarks/phase4/settings.json`). The
reported runs use seeds 100 to 129, which no earlier phase used.

## What was run

- **Tables, participants, settings:** as in Phases 2 and 3: the stroke and loan tables, five participants,
  the settings chosen there on made-up tables. What is attacked is what was measured there.
- **The default plan** (every participant adds all of its own noise, updates in the clear, 20 rounds of 3
  local passes) at total budgets 0.5, 1, 2, 4 and 8, with the research network and with logistic
  regression.
- **No protection**, for comparison.
- **A planted defect:** the default plan with its noise switched off while its ledger goes on counting
  the noise and its report states a budget of 1. This is the defect an audit exists to catch. It is
  planted in the benchmark script and nowhere in the library.
- **Shared noise**, ordinary and whole numbers, one assumed colluder, logistic regression, budgets 1
  and 8.
- 30 runs per cell, 36 cells, 1,080 runs, about seven and a half hours of computing spread over four
  cores.

## Results

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

What these numbers say:

1. **Loan table: about 4% to 5% at a budget of 1, so the 5% target is missed as it is worded.** The
   target is defined on the upper end of the interval (`docs/KPI_MAPPING.md`), and that is 5.6% for the
   network and 6.3% for logistic regression. At a budget of 0.5 the network is within 5% (2.8%, upper
   end 4.0%); logistic regression is at 3.6% with an upper end of 5.0%, a hair over. From a budget of 2
   upwards the measured advantage is 6% to 10%.
2. **Stroke table: within 5% at every budget tried**, between about 0% and 3%. This is an average over
   all test records, and it needs a caveat. The stroke runs weight the rare class (stroke cases, about
   4% of the rows) and shorten every row of the common class to about 3% of the clipping length, so 96%
   of the test records can barely move the model. The few rows of the rare class keep the full length
   and are probably as exposed as loan rows. The stored counts do not separate the classes; step S16
   should measure them apart. (It did: 5.4% and 9.0% for the rare class at a budget of 1, about 0%
   for the common class. See `docs/PHASE4_MORE_ATTACKS.md`.)
3. **The loan figures are a large share of what is possible at all.** At a budget of 0.5 the plan's
   attacker reaches 3.6% of a proven 5.6%; at a budget of 1, 5.3% of 10.6%. Loan rows pull harder than
   the clipping length allows, so each is shortened to exactly that length, which is the worst case the
   proven limit is computed for. The lower limit on the budget tells the same story: 0.30 shown where
   the most the test could have shown is 0.51.
4. **An outsider with the final model learns nothing measurable**, at any budget, on either table, and
   not even without protection (every one of its intervals contains zero, and its lower limit on the
   budget is never above 0.04). These models have 14 to about 3,500 numbers trained on thousands of
   rows; they do not memorize single rows. The leak that
   the budget has to cover here is in the updates sent during training, not in the finished model.
5. **No protected run was flagged.** Over all 28 protected cells and both attacks the lower limit on
   the budget is at most 30% of the stated budget, and no measured advantage exceeds the proven limit
   for the noise that was used (the closest is 64% of it).
6. **The planted defect is caught in all four cells**, by one of two signs. With logistic regression the
   lower limit on the budget rises above the stated budget of 1 (2.31 on stroke, 1.03 on loan). With
   the network the lower limit stays below 1 (0.67 and 0.87), but the measured advantage (18.8% and
   30.1%) stands far above the 10.6% to 10.7% that the stated noise allows, which cannot happen if the
   noise is there. Loan with logistic regression shows both signs.
7. **Shared noise at a budget of 1 measures lower than the default plan on the loan table** (2.8% with
   ordinary noise and 3.2% with whole numbers, against 5.3%). The two are not attacked through the same
   window: with shared noise the server sees one sum per round instead of every participant's update,
   and one participant is assumed to work with it. This is a measurement of each plan against its own
   attacker, not a ranking of the plans.
8. **Without protection the plan's attacker reaches 4% to 10%**, less than it reaches against the planted
   defect. Its arithmetic is exact for the protected plans, where every row's pull is shortened and
   known; ordinary training moves the model further within a round than the attack follows. The figure
   for "no protection" is therefore a floor for what a tailored attack would get, not a measure of it.
9. **The rows are not smooth curves.** Budgets up to 2 run with one group of settings and budgets from 4
   with another (as in Phases 2 and 3), which is why the loan table shows less at a budget of 4 than
   at 2. Every cell uses the same 30 seeds, so cells are also not independent of each other.

## How far the test can see

The column "most the test could show" plays an attacker that could do no better: every test record
exactly as exposed as the proven limit allows, the same number of records, runs and guesses. Its lower
limit is what the test would report at best.

| Stated budget | Most the test could show (default plan) | Shown on the loan table |
|---|---|---|
| 0.5 | 0.21 to 0.22 | 0.03 and 0.12 |
| 1 | 0.49 to 0.51 | 0.23 and 0.30 |
| 2 | 1.06 | 0.53 and 0.57 |
| 4 | 2.05 to 2.11 | 0.62 and 0.59 |
| 8 | 3.78 to 4.01 | 0.68 and 0.52 |

So risk 13 of `docs/RISK_REGISTER.md` is real: **on tables of this size the method cannot confirm a
budget, it can only catch a pipeline that is badly broken.** Thirty pooled runs with 640 to 750 test
records each can prove about half of the stated budget at the very best, and a single run proves next
to nothing (the median lower limit of the single runs is at most 0.18 in any protected cell, and 0.00
in most; pooling was necessary). A lower limit of 0.30 next to a stated budget of 1 must not be read as "the true budget is
about 0.30". It means: this attack, on these records, could show 0.30, and no attack could have shown
more than about 0.5.

What the whole-run test did not catch, and what covers the gap:

- **Noise that is merely too small.** A training step with half the stated noise is not flagged by the
  whole-run test. It is flagged by the single-step check, which attacks the noise itself with a
  hand-made worst-case record (it proves at least 1.53 where 1.00 is reported). That check runs for
  the private training step, both shared-noise modes and the whole-participant mode.
- **Missing noise that other randomness covers.** With the noise switched off, the random choice of rows
  in every step still hides a good part of a record from someone who only sees a participant's update
  once per round. That is why the planted defect shows an advantage of 10% to 30% and not 100%, and why
  two of the four cells needed the second sign. In the shared-noise mode with ordinary noise the
  rounding grid, which is sized to the stated noise, has the same effect; a test shows the defect on
  the whole-number path instead, where nothing is rounded after the noise.

## Does the auditor itself work?

The step's exit check is the auditor's own test (`obscuradp audit self-test`, also in the fast test
suite and, on ten more made-up tables, in the nightly one). On a small made-up table it shows both
auditors a correct and a broken pipeline:

| Pipeline | Must be flagged | Result |
|---|---|---|
| Whole run, default plan at a budget of 1 | no | not flagged (lower limit 0.06) |
| Whole run, no noise at all but a budget of 1 stated | yes | **flagged** (lower limit 3.33, 99% advantage) |
| Single step, the library's private training step | no | not flagged (proves at least 0.46 of a reported 1.00; best estimate 0.87) |
| Single step, half the noise its ledger is told about | yes | **flagged** (proves at least 1.53) |

Further planted defects are in the tests: the default plan with its noise switched off (flagged when
five runs are pooled), the whole-number shared sum without its noise (flagged in one run), a shared sum
with a third of its noise slice, and, as before, halved noise and gradients that are not shortened in
the private training step.

One lesson from building it is worth recording. The first version of the update attack took the size of
the noise from the pipeline's own settings. It then looked away exactly when the noise was missing and
did not flag the planted defect. The attack now measures the noise from what it sees. This was found on
the pilot, before any reported run; the first pilot was discarded and repeated.

## Where the website target stands

| Target | Now |
|---|---|
| At most 5% attacker advantage at a total budget of 1 | **Measured. Met on the stroke table for the average row** (upper ends -0.1% and 1.4%; not for its rare class, see the correction at the top), **not met on the loan table** (4.3% and 5.3%, upper ends 5.6% and 6.3%). On the loan table the network is within 5% at a budget of 0.5; logistic regression is at 3.6% there with an upper end of 5.0%. Proven limits at a budget of 1: 10.6% for the noise used, 46.2% for any method |
| Mathematically guaranteed privacy | Unchanged, and now tested from outside: no protected run exceeded what its report states, and a pipeline with its noise switched off is caught |

For the website (step S22, new item 33 of `docs/WEBSITE_MISMATCHES.md`): the sentence about 5% cannot
stay a single promise. What the measurements support is "in our attack tests the strongest attacker
reached 0% to 5% at a total budget of 1, depending on the table; the proven worst case is 10.6%", shown
with the table above.

## Limits of this evidence

- **Two tables, five participants, two small models.** Nothing here says what an attacker gets on larger
  models, which memorize more, or with more participants.
- **Real rows are not the worst case.** The test records are ordinary rows. A hand-made record can be
  worse; that is what the single-step check is for, and it shows the proven limit is nearly reached
  there.
- **One attack is not all attacks.** The plan's attacker is a strong one (it knows every other row and
  reaches up to 64% of the proven limit), but it follows local training only roughly, and it treats
  the random choice of rows as noise it cannot see through. A better attack can only raise the measured
  numbers, up to the proven limits. Step S16 adds shadow models as a second opinion.
- **The averages hide groups of rows.** See finding 2: rows of a rare class that is given more weight
  are more exposed than the average shows.
- **The reach of the lower limit is short** (section above). It is reported next to every result so that
  a small number is not mistaken for safety.
- **Pooling runs is this project's own step.** The argument is short (thirty independent runs are one
  procedure in which a record enters one run), and a test checks by brute force that pooled limits
  overstate no more often than single ones. It has not been reviewed by anyone outside.
- **The second sign of a broken pipeline was added after the reported runs were in.** The measured
  advantage standing above the proven limit for the stated noise is a valid sign whatever the data,
  but the decision to report it was made on seeing that the lower limit alone missed two of the four
  planted-defect cells. No protected cell shows either sign.
- **Whole participants were not tested this way.** A plan that protects a whole participant needs whole
  participants to flip coins over, and five are far too few. That part waits for the larger groups of
  steps S17 and S18; until then the single-step check covers that mode.
- **Noise came from the run's seed** so that runs can be repeated, as in Phases 2 and 3.
