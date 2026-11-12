# The benchmark's plan, written before any of its runs

Written in step S17 on 2026-11-12, before any run of the benchmark. It fixes what will be run, with which
seeds, how settings are chosen, what is measured and how it is judged, so that none of this can be bent
towards a result afterwards. Changes after this date go into the section "Amendments" at the end, with a
date and a reason; the text above it is not edited.

Words used: a **budget** is the total privacy budget $\varepsilon$ of a whole training run at
$\delta = 10^{-5}$. A **unit** is what a budget protects: one row of a table, or one participant with
everything it holds. An **arm** is one way of protecting a run. A **seed** is the number all random
choices of a run are derived from.

## 1. What the benchmark is for

To state, on five datasets and with numbers anyone can regenerate, what protection by a stated budget
costs in model quality and in messages, for the ways of protecting a run that this library offers. It
is not a comparison with other libraries.

## 2. Datasets and what a budget protects there

| Dataset | Participants | Rows | Unit | Model | Headline measure | A model that learned nothing |
|---|---|---|---|---|---|---|
| Made-up health table (`obscuradp.tables.synthetic_health`, table seed 2026) | 10 hospitals, 400 to 2,900 rows each | 20,000 | row | network with 1,321 numbers; logistic regression | AUC on held-out rows at their natural mix | 0.5 |
| Stroke table | 5 | 4,908 | row | research network (3,457 numbers); logistic regression | the same | 0.5 |
| Loan table | 5 | 4,269 | row | research network (993 numbers); logistic regression | the same | 0.5 |
| Handwriting (Federated EMNIST) | 3,400 writers | 671,585 training images | participant | small image network (44,318 numbers) | share of test images classified right | about 0.06 (the most common class) |
| Text (Shakespeare) | 644 roles | 253,529 training windows | participant | small text network (63,952 numbers) | share of next symbols predicted right | about 0.18 (always a blank) |

Sources, sizes, fingerprints and licenses are in `docs/DATASETS.md`. Rows are cut as follows. Tables: 70%
training, 15% validation, 15% test rows per participant, cut anew for every seed (for stroke and loan
also which rows a participant holds, as in Phases 2 and 3). Handwriting and text: the archive's own
training and test files; the last tenth of a participant's training rows are its validation rows; one
fixed cut for all seeds.

## 3. Arms and budgets

**Tables** (unit: a row), budgets 0.5, 1, 2, 4 and 8:

| Arm | What it is |
|---|---|
| none | no protection |
| local | the default plan: every participant adds all of its own noise (`LocalDPSGD`) |
| shared | shared noise, ordinary, one assumed colluder (`DistributedDPSGD`) |
| shared_whole | shared noise, whole numbers, one assumed colluder (`DistributedDPSGD(noise="skellam")`) |

**Handwriting and text** (unit: a participant):

| Arm | What it is |
|---|---|
| none | no protection; in every round each participant takes part with the same chance as below |
| participant | whole participants protected (`ClientLevelDP`), noise shared, a tenth of the participants assumed to work with the server |

Handwriting: about 200 of 3,400 writers per round, 100 rounds, budgets 2, 4 and 8, and one single run at
a budget of 1. Text: about 50 of 644 roles per round, 100 rounds, a budget of 8 only, three seeds
(section 9 says why).

## 4. Seeds

| Use | Seeds |
|---|---|
| Timing pilot (done, section 9) | 900 |
| Choosing settings (section 5) | 901, 902, 903 (the larger models: 901 only) |
| The reported runs | 1000, 1001, 1002, 1003, 1004 (text: 1000, 1001, 1002) |
| The single handwriting run at a budget of 1 | 1000 |

None of these was used in an earlier phase (42 to 51, 90 to 94, 100 to 129, 200 to 231, 300 to 329). No
seed is added, dropped or replaced after a reported run has been looked at.

## 5. How settings are chosen

- **Stroke and loan:** exactly the settings of Phases 2 and 3 (`benchmarks/phase2/settings.json`,
  `benchmarks/phase3/settings.json`), which were chosen on made-up tables. Nothing is chosen again.
- **Health and handwriting:** a search on **validation** rows with seeds that no reported run uses. The
  setting with the highest mean headline measure on validation rows is taken; on a tie the smaller
  learning rate, then the smaller clipping norm. Test rows are not scored during the search.
- **Text:** no search (section 9): learning rate 1, clipping norm 1.

| Dataset | Model | Arm | Fixed | Searched | Seeds |
|---|---|---|---|---|---|
| Health | logistic regression | none, local | 30 rounds, 1 local pass, batch 32 | learning rate 0.05, 0.1, 0.2, 0.5; for local also clipping norm 0.5, 1, 2 | 901 to 903 |
| Health | logistic regression | shared, shared_whole | 100 rounds; batch 640, or every row in every round | learning rate 0.3, 1, 3; clipping norm 0.5, 1, 2 | 901 to 903 |
| Health | network | none, local | 30 rounds, 1 local pass, batch 32 | none: learning rate 0.05, 0.1, 0.2, 0.5; local: learning rate 0.05, 0.1, 0.2 and clipping norm 0.5, 1 | 901 |
| Health | network | shared, shared_whole | as for logistic regression | learning rate 0.3, 1, 3; clipping norm 0.5, 1 | 901 |
| Handwriting | image network | none, participant | 100 rounds, 1 local pass, batch 20 | learning rate 0.05, 0.1, 0.2; for participant also clipping norm 0.5, 1 | 901 |

Protected arms are searched at two budgets, 1 and 8 (handwriting: 8 only, and its choice is used at
every budget). On the health table budgets up to 2 use what was chosen at 1, the others what was chosen
at 8, as in Phase 3.

**What this search costs in privacy:** nothing on the made-up table and on the two public datasets,
where no row is private. It would cost something on private rows, and the library does not yet count
that (planned after the first release). The benchmark's budgets therefore cover the reported training
runs, not a search.

## 6. What is expected (written down before the runs)

1. **Tables, small and hard (health, stroke):** at the same budget, shared noise gives a higher AUC than
   local noise, at every budget from 1 upwards. Phase 3 found this on the stroke table; the health table
   has not been run.
2. **Loan table:** no arm differs from another by more than 0.01 AUC at any budget.
3. **Tables, budget 8:** shared noise is within 0.03 AUC of no protection on all three tables.
4. **Handwriting:** protection of whole participants costs a great deal. At a budget of 8 the accuracy is
   expected clearly below the unprotected one; **at budgets of 2 and 1 it may be no better than a model
   that learned nothing (about 0.06)**. That would be a result, not a failure.
5. **Text:** with 644 roles, fewer than the thousand the library asks for, the protected model is
   expected at the level of a model that learned nothing (about 0.18), even at a budget of 8.
6. **Budgets:** every protected run stays within its budget and every stored privacy report recomputes.

Expectations 1 to 3 are judged by the rule of section 8. Nothing else is called a finding unless it
passes the same rule, and anything found that was not expected here is reported as "not planned for".

## 7. What is measured in every run

- The headline measure of section 2 on test rows; for tables also average precision, and recall,
  precision and F1 at the cut-off chosen on validation rows.
- What one participant uploads per round and over the whole run, against the same update as 32-bit
  numbers.
- The privacy report: the budget spent, both proven limits on the attacker's advantage, the table of
  budgets by number of colluders. Stored beside the scores of every protected run.
- Whether the secure sum was run or simulated (`crypto_in_loop`, section 10).
- Time, the commit of the code, the versions of the libraries, the machine.

Attack tests are part of the benchmark where they change what a reader should conclude:

- **Health table, by class:** the whole-run audit of step S15 at budgets 1 and 8 with every row of the
  rare class a test record, read by class, 30 runs (seeds 1100 to 1129). Step S16 showed that an average
  over rows hides the rare class when classes are weighted.
- **Handwriting, whole participants:** an audit in which whole writers are the test units, at a budget
  of 8. It needs an attack that does not exist yet. If it cannot be built within the step that runs the
  benchmark, the results say "not done" in its place.

## 8. How results are judged

- Every cell (dataset, model, arm, budget) is reported as the mean of its five seeds with the sample
  standard deviation (dividing by four, not five).
- Two arms are compared seed by seed, since the same seed means the same cut of the data. The mean of
  the five differences is given with a 95% interval (Student's t with four degrees of freedom). **A
  difference counts only if that interval does not contain zero.**
- For the yardstick "local noise" on stroke and loan, the benchmark has one local arm (the Phase 2
  protocol); Phase 3's second local arm is not repeated.
- With five seeds an interval is wide: differences below about 0.02 AUC will often not count. That is
  accepted; more seeds are not added afterwards.
- Many comparisons are made (three protected arms, five budgets, two models, three tables). No
  correction for their number is applied; every comparison is reported, counted or not, and only
  expectations 1 to 3 are called confirmed or not confirmed.
- No run is left out. A run that stops for a technical reason is run again with the same seed and the
  stop is recorded.

## 9. Time, and what is cut first

Measured by the timing pilot (`benchmarks/results/phase5/pilot.json`; seed 900; one core per run; times
of protected arms scaled up from three rounds, which overestimates):

| Dataset | Model | Arm | Rounds of a full run | Estimated time of one full run | Reported runs | Estimated hours of computing |
|---|---|---|---|---|---|---|
| Health | network | none | 30 | 6 s | 5 | 0.01 |
| Health | network | local | 30 | 6 min | 25 | 2.33 |
| Health | network | shared | 100 | 7 min | 25 | 2.83 |
| Health | network | shared_whole | 100 | 85 s | 25 | 0.59 |
| Health | logistic | none | 30 | 4 s | 5 | 0.00 |
| Health | logistic | local | 30 | 18 s | 25 | 0.13 |
| Health | logistic | shared | 100 | 6 s | 25 | 0.04 |
| Health | logistic | shared_whole | 100 | 5 s | 25 | 0.03 |
| Stroke | network | none | 20 | 3 s | 5 | 0.00 |
| Stroke | network | local | 20 | 64 s | 25 | 0.45 |
| Stroke | network | shared | 200 | 7 min | 25 | 2.97 |
| Stroke | network | shared_whole | 200 | 2 min | 25 | 1.00 |
| Stroke | logistic | none | 20 | 2 s | 5 | 0.00 |
| Stroke | logistic | local | 20 | 8 s | 25 | 0.05 |
| Stroke | logistic | shared | 600 | 24 s | 25 | 0.17 |
| Stroke | logistic | shared_whole | 200 | 7 s | 25 | 0.05 |
| Loan | network | none | 20 | 3 s | 5 | 0.00 |
| Loan | network | local | 20 | 42 s | 25 | 0.30 |
| Loan | network | shared | 200 | 6 min | 25 | 2.58 |
| Loan | network | shared_whole | 200 | 38 s | 25 | 0.26 |
| Loan | logistic | none | 20 | 1 s | 5 | 0.00 |
| Loan | logistic | local | 20 | 7 s | 25 | 0.05 |
| Loan | logistic | shared | 600 | 24 s | 25 | 0.17 |
| Loan | logistic | shared_whole | 60 | 2 s | 25 | 0.01 |
| Handwriting | cnn | none | 100 | 9 min | 5 | 0.72 |
| Handwriting | cnn | participant | 100 | 39 min | 16 | 10.53 |
| Text | gru | none | 100 | 30 min | 3 | 1.51 |
| Text | gru | participant | 100 | 70 min | 3 | 3.50 |

Together about 30 hours of computing for the reported runs, and about 5 more for the search of
section 5: some 9 hours on four cores. This is an upper estimate. The roadmap had guessed 17 hours of
computing.

What the pilot changed in this plan, before any reported run:

- **The text dataset is run smaller than the roadmap said.** One unprotected run takes 30 minutes and a
  protected one is estimated at over an hour. It is therefore run with three seeds (1000 to 1002), not
  five, and without a search: learning rate 1 and clipping norm 1, with which the unprotected pilot run
  predicted 44% of the next symbols on validation rows.
- **The searches for the two larger models are kept small** (one seed, fewer values), see section 5.
- The unprotected pilot runs learn: validation AUC 0.81 on the health table (both models), 0.75 to 0.76
  on stroke, 0.97 to 0.98 on loan; 72% of validation images right on handwriting. These are validation
  rows of the pilot's seed; no test row was scored.

If the reported runs take more than twice the estimate, cuts are made in this order and recorded: the
text dataset; handwriting down to 75 rounds and 150 writers per round; handwriting down to three seeds;
handwriting budgets down to 2 and 8. **Table runs are never cut.**

## 10. The secure sum at this size

Shared noise goes through a secure sum, in which every pair of participants shares a secret. Three
tables with five or ten participants run it for real, in every round. With 3,400 writers that is about
5.8 million shared secrets and as many masks of the model's length per round, which one machine cannot
play in reasonable time. For handwriting and text the secure sum is therefore **simulated**
(`Simulation(crypto_in_loop=False)`): the participants' shortened changes are added up in the clear and
the noise of all slices is drawn in one piece. What this changes and what it does not:

- The opened sum has the same content and the same amount of noise. A test checks this against the real
  secure sum on a small group (`tests/test_large_federations.py`). The rounding of each message to the
  secure sum's grid is left out; it is small against the noise.
- The budget and the privacy report are those of the real protocol. The report says in plain words that
  the run was simulated.
- Message sizes are computed from the real protocol's layout, not measured from sent messages.
- As a check, the reported runs include one small real case: the first 60 writers by name, 20 rounds, a
  budget of 8, three seeds (1000 to 1002), once with the real and once with the simulated secure sum.
  The two must agree within the scatter of the seeds.

## 11. What will be published

Every run's record, the settings chosen in section 5 with everything that was tried, a summary built by a
script from the records, and a memo in everyday words with the limits of the evidence. Results that do
not match section 6 are reported with the same weight as those that do.

## 12. How it can be checked that this plan came first

The commit that adds this file is the plan. Every record of a run stores the commit it was run at; a
check of the results requires that commit to come after the plan's commit and this file to be unchanged
above the "Amendments" line. The commit is meant to carry the tag `benchmark-plan-v1`; the working
session can only push its own branch, so the tag has to be set by the owner, on the merge commit of the
pull request that adds this file.

## Amendments

### 2026-11-12, before the first run of the search and of the benchmark

Written in step S18, after the plan's commit (`ad2289c`) and before any run named in sections 4 and 5.
None of these changes a dataset, an arm, a budget, a seed or the judging rule. They fix details the text
above left open, so that they cannot be chosen after results are seen.

1. **Attack tests by class on the health table (section 7).** They are run on the default plan (arm
   "local") with the settings the search chooses, for logistic regression and for the network, at budgets
   1 and 8, seeds 1100 to 1129. The test records of a hospital are all its training rows of the rare
   class and equally many of the other class, drawn at random, as in step S16. Both attacks of step S15
   are used, with the share of guesses fixed there (5%). Reason: section 7 named neither the arm nor the
   models.
2. **Attack test on whole writers (section 7).** One attempt, one run: handwriting, a budget of 8, seed
   1100, the settings the search chooses. 400 writers are drawn at random as test units; each is kept or
   left out of the whole run by a fair coin, and every other writer takes part as usual. The attacker
   sees the sum that is opened in every round, knows the images of the 400 test writers, and does not
   know who was drawn in which round. For every round it works out what each test writer would have
   sent and takes out the direction that all test writers share. The scores of all test writers are
   then worked out together, as the set of scores that best explains what is left of the opened sums
   (the decoding step S15 used for rows). Guesses are made for the 5% highest and the 5% lowest scores. The attack is built and tried on a small made-up
   group only, never on this run. If it cannot be built and run within about three hours of computing,
   the results say "not done". Reason: section 7 described no attack.
3. **The small real case of section 10.** All 60 writers take part in every round (a round would draw
   200, more than there are), 6 of them are assumed to work with the server, and the learning rate and
   the clipping norm are those the search chooses for handwriting. Real and simulated secure sum are
   compared seed by seed with the rule of section 8 (Student's t with two degrees of freedom): they
   agree if the interval of their difference contains zero. Reason: section 10 named neither the
   participation nor what "agree" means in numbers.
4. **A search run whose score is not a number** counts as the worst setting. Reason: section 5 did not
   say what happens when a learning rate is too large for a run to stay finite.
5. **What the search stores.** One small record per run with the scores on validation rows, the time
   and the budget spent; scores on test rows are not computed into any stored file. Everything tried
   and everything chosen goes to `benchmarks/results/phase5/settings.json`, which is committed before
   the first reported run.
6. **Where runs are made.** Every run is made from a clean copy of the repository at a recorded
   commit, so that the commit stored in a record names exactly the code that produced it.
7. **How "unchanged" is checked (section 12).** The text above the line "Amendments" has the sha256
   fingerprint `87e4c3b26b1240a69eb7fb1288a8889b8f89ed23357ec57d2a499ad93a197313`, computed on the file
   with Unix line ends. A test compares it on every change, and the results check requires every run's
   commit to come after the plan's commit `ad2289c`.
8. **How expectations 1 to 3 are read (section 6).** "Shared noise" means each of the two shared arms,
   and every statement is checked for both models.
   - Expectation 1 is confirmed if, on the health and the stroke table, for both shared arms, both
     models and every budget of 1, 2, 4 and 8, the difference "shared minus local" is above zero and
     counts by the rule of section 8 (32 comparisons). Otherwise it is not confirmed, and the results
     say in how many of the 32 it holds.
   - Expectation 2 is confirmed if, on the loan table, the mean AUC of any two of the three protected
     arms differs by at most 0.01, for both models at every budget (30 comparisons).
   - Expectation 3 is confirmed if, at a budget of 8, the mean AUC of each shared arm is at most 0.03
     below that of no protection, for both models on all three tables (12 comparisons).

   Reason: section 6 did not say which shared arm and which model a statement is about.
