# Shared noise: why it is safe, and what is not yet proven

Written in step S12 (2026-11-11). This note belongs to `obscuradp.DistributedDPSGD`
(`src/obscuradp/plan.py`, `src/obscuradp/mechanisms/shared_noise.py`, `src/obscuradp/compression/grid.py`).
It states the argument behind the budget of that mode and marks the one step that rests on a paper
argument and a numerical check, not on a proof the tests can run.

## The idea in plain words

In the local mode every participant adds all the random noise its records need. With five participants
the model receives five full doses. In the shared-noise mode every participant adds a slice, and the
slices add up to one full dose inside the secure sum, the step where the server can read the total of all
messages and no single message.

The price is trust in more things at once. A single message now carries only a slice, far too little to
protect anyone. The protection exists only as long as single messages stay hidden and enough of the
slices come from participants who keep them to themselves.

## One round

$N$ participants, a clipping norm $C$, a noise multiplier $\sigma$. At most $t$ participants pool what they
know with the server, at most $d$ are missing.

1. Every row of every participant joins the round independently with the same probability $q$.
2. Participant $i$ computes the gradient of each of its drawn rows at the shared model, shortens each to
   length at most $C$, and adds them up: $S_i$. Adding or removing one row changes $S_i$ by at most $C$.
3. It adds its slice $Z_i \sim \mathcal{N}(0, \tau^2 I)$ with $\tau = \sigma C / \sqrt{N - t - d}$ and sends
   $S_i + Z_i$, rounded to whole numbers, through the secure sum.
4. The server opens $\sum_i (S_i + Z_i)$, divides by the fixed number $q \cdot n$ ($n$ rows in total) and
   takes one optimizer step.

## Why the opened sum is protected

Take the worst case the plan allows: $t$ participants give their own messages to the server. Everything
the group can then compute comes from the sum over the other participants,

$$Y = \sum_{i \,\text{honest}} (S_i + Z_i).$$

When the sum opened, all $N$ messages were there, so at least $N - t$ slices are in $Y$ and its noise has
standard deviation at least $\tau \sqrt{N - t} \ge \sigma C$. One row moves $Y$ by at most $C$. That is the
Gaussian mechanism with noise multiplier at least $\sigma$, on rows drawn with probability $q$, once per
round. The ledger charges exactly that: `SubsampledGaussianEvent(sigma, q, 1)` per round and participant.
Because every row has the same $q$ and sees the same noise, every participant spends the same budget.

If more than $t$ collude, fewer slices remain. With $t'$ colluders the multiplier is

$$\sigma_{t'} = \sigma \sqrt{\frac{N - t'}{N - t - d}},$$

and the report prints the budget for every $t'$ from 0 to $N - 1$. Two consequences that are easy to miss:

- **A participant on its own counts as one colluder.** It knows its own slice and can subtract it from
  what the model reveals. With $t = 0$ the stated budget therefore holds against the server and outside
  observers, and the row $t' = 1$ holds against any single participant.
- **$d$ buys nothing yet.** The secure sum of this version cannot open with a message missing, so every
  opened sum has all $N$ slices. A value $d > 0$ makes every slice larger for no gain in this version. It
  is there for secure sums that tolerate missing participants (planned with the Flower adapter).

## Why a failed round is charged

The roadmap first said that a round with a dropout is "charged only if a sum was released". That is not
safe when $t \ge 1$. Every pair of participants shares a secret mask. If the missing participant is one of
the colluders, the group knows all masks that it shares with the others. It removes them from the messages
that did arrive, the remaining masks cancel among the honest participants, and the group reads their sum
$Y$. No sum was opened by the protocol, yet $Y$ was released to the group.

The test `test_a_missing_colluder_lets_the_group_read_the_others_sum` carries this attack out with real
keys, and `test_a_missing_honest_participant_keeps_everything_hidden` shows the other case. The rule in
the library is therefore: **whoever sent a message is charged for the round**, opened or not. A failed
round does not move the model, the next round draws fresh rows, fresh noise and fresh masks, and the
number of rounds given to the simulation is the number of rounds that are started. A run with dropouts
ends with fewer model updates and never over budget.

(If more participants are missing than the plan assumes colluders, at least one missing participant is
honest and nothing can be read. A participant cannot check that from where it sits, so the library does
not make an exception for it.)

## Why values are never cut off

The local mode cuts every number of a message off at a declared range before rounding. There that is
harmless: each message already carries all of its noise, and anything done to it afterwards cannot reveal
more.

With slices it is different. The sum of cut-off messages is not something one can compute from the true
sum $Y$. If the other participants' numbers are cut off, their slices are gone, and a record is left with
its own participant's slice only. So in this mode:

- Each number of the clean sum $S_i$ is limited to $\pm A$ **before** the noise is added, with $A$ computed
  from public numbers (clipping norm, sampling rate, size of the largest table). Limiting each number can
  only bring two sums closer together, so one row still moves $S_i$ by at most $C$.
- After the noise nothing is cut off. Numbers are written modulo $2^k$ and wrap around if they are too
  large. Wrapping is harmless: the wrapped sum of wrapped numbers equals the wrapped true sum, so it is
  computed from the true sum alone. The range covers $A$ plus ten standard deviations of a slice, so a
  wrap of the total practically never happens, and if it did, only accuracy would suffer.

## The step that is argued, not proven: rounding

Each participant rounds its noisy number to a grid before the secure sum, at random and right on average.
The grid step is a quarter of the slice's standard deviation $\tau$. The budget above was computed for
unrounded numbers. Is the sum of separately rounded numbers still computed from $Y$ alone?

Not exactly, but very nearly. In one coordinate, with the grid step as the unit:

- Rounding a number $x$ at random to a neighbouring whole number picks $k$ with probability
  $\max(0, 1 - |x - k|)$. The rounded value of $a + Z$ with Gaussian $Z$ therefore has the probabilities
  of the smooth curve $\varphi_\tau * \Lambda$ (Gaussian density blended with a hat of width 2), read off at
  the whole numbers.
- The sum of $h$ such independent whole numbers has, up to a tiny error, the probabilities of
  $\varphi_{\tau\sqrt{h}} * \Lambda * \dots * \Lambda$ ($h$ hats) read off at the whole numbers. That is
  exactly what one gets from the true noisy sum $Y$ by adding $h - 1$ small noises that depend on no data
  and rounding once.
- The error comes from the curves not being perfectly smooth at the scale of the grid. In the language of
  frequencies, the Gaussian part of each curve is $e^{-2 \pi^2 \tau^2 \xi^2}$, and the error terms contain
  two such factors whose frequencies differ by at least 1. Their product is at most $e^{-\pi^2 \tau^2}$. With
  $\tau = 4$ grid steps that is $e^{-158}$, about $10^{-69}$.

So in every coordinate what the group sees is, up to a difference of that order, something computed from
$Y$ alone, and the budget of $Y$ applies. Over all coordinates and rounds the differences add up to
something still far below any $\delta$ in use.

What supports this:

- The argument above, on paper.
- `tests/test_shared_noise.py` computes both distributions exactly in one dimension and finds them equal
  to the limit of double-precision arithmetic (difference below $10^{-10}$) at the planned grid, for
  several numbers of participants and clean values. On a coarse grid (step of two slice widths) the same
  computation shows a difference above $10^{-3}$, so the check can fail.
- The library refuses any other grid in this mode: the step is fixed at $\tau / 4$.

What is missing: a proof that the tests can run for all dimensions at once, and a treatment of
floating-point effects in the noise itself (finding G16). Step S13 added a second path with
whole-number noise (Skellam), where each row is rounded first, the noise lives on the grid, and the sum
of the slices is exact.
Every report of the Gaussian path says that the rounding step is argued and not machine-proven. The
whole-number path below does not have this step.

One correction to an earlier claim made while planning: the sum of separately rounded messages does
**not** have the same distribution as one rounding of the plain sum (they differ by about $10^{-3}$ in
total variation at this grid). It has the distribution of one rounding of the sum **plus small independent
noise**, which is just as good for privacy and is the statement tested here.

## What else the mode relies on

- **Unchanged public keys.** Keys are not authenticated. A server that swaps them could read single
  messages. In the local mode that exposes a fully noised update. Here it exposes a message with one
  slice, and the stated budget no longer holds.
- **Secret randomness.** The noise, and the choice of rows, must be unknown to an attacker. With
  `secure_noise=True` both come from the operating system. (Before S12 the choice of rows was drawn from
  the run's seed even then; that is fixed for all modes.)
- **Everyone follows the protocol.** A participant that sends something else, or a server that deviates,
  is outside what any report covers.

## The whole-number path (added in S13)

`DistributedDPSGD(noise="skellam")` turns the order around so that the step above needs no argument.

1. Every row takes part in every round.
2. Each row's clipped gradient is scaled to a grid of $c$ points per clipping norm and **rounded on its
   own**, at random and right on average.
3. The squared length of the rounded row is computed in whole-number arithmetic and compared with a fixed
   whole number $\Delta_2^2$. A row above it is rounded again; after 16 tries it is rounded towards zero,
   which cannot lengthen it; a row that is still too long counts for nothing. **So no row can move the sum
   by more than $\Delta_2$, whatever floating-point arithmetic did before.** One row moves the sum of
   absolute values by at most $\Delta_1 = \min(\Delta_2^2, \sqrt{m}\,\Delta_2)$ for $m$ model numbers.
4. The rounded rows are added up, each number is limited to a public range, and the participant adds a
   slice of Skellam noise (the difference of two Poisson counts) with variance $\mu / (N - t - d)$.
5. Slices of Skellam noise add up to Skellam noise with the variances added, exactly. The honest
   participants therefore supply Skellam noise of variance at least $\mu$, with nothing approximated.

The limit $\Delta_2^2$ is $\min\bigl((c + \sqrt{m})^2,\ c^2 + m/4 + \sqrt{2 \ln(1/\beta)}\,(c + \sqrt{m}/2)\bigr)$
with $\beta = e^{-1/2}$ (Kairouz, Liu and Steinke, 2021). The formula only decides how often a row has to
be rounded again; the guarantee comes from the exact check in step 3. The library chooses $c$ so that
$\Delta_2 \le 1.05\,c$: rounding lengthens a row by at most 5%.

**The count.** For whole numbers that one row changes by at most $\Delta_2$ in length and $\Delta_1$ in
summed absolute size, Skellam noise of variance $\mu$ per number, and every whole-number order
$\alpha > 1$ (Agarwal, Kairouz and Liu, 2021, Theorem 3.5 and Corollary 3.6, read on arXiv:2110.04995 on
2026-11-11):

$$\varepsilon(\alpha) \le \frac{\alpha \Delta_2^2}{2\mu}
+ \min\left(\frac{(2\alpha - 1)\Delta_2^2 + 6\Delta_1}{4\mu^2},\ \frac{3\Delta_1}{2\mu}\right).$$

The first part is what Gaussian noise of the same variance costs; the second is the price of whole
numbers and fades as the noise grows. Rounds add up by adding these curves, and the curve is turned into
a budget with the conversion of the `dp-accounting` library. Only whole-number orders are used, because
only those are proven.

**What the tests check** (`tests/test_skellam.py`):

- The formula is never below the exact value computed from the Skellam probabilities, over 42 pairs of
  noise level and shift at seven orders each, and for 200 many-dimensional cases (where the exact value
  is the sum over coordinates). For ample noise it is within 1% of exact.
- For one release, the exact leak at the reported budget is below $\delta$.
- A formula made 10% too small breaks 23 tests. Dropping only the second part breaks one test (the one
  that pins the paper's form): on every case tried the exact value lies even below the first part. The
  second part is kept because it is what the paper proves.
- No rounded row ever exceeds $\Delta_2$, over random sizes, scales and rows up to 50 times too long.

**What this path costs**, measured:

- **About 14% more noise** than Gaussian noise for the same budget: about 8% because only the looser
  counting method (RDP) is proven for this noise, 5% from the rounding.
- **Every row in every round.** The extra protection from drawing rows at random is not counted for this
  noise here. More noise per round, against a larger batch.
- **Larger messages**: 18 to 22 bits per number for the two research-sized groups (see
  `docs/KPI_MAPPING.md`).

**What remains.** The noise is drawn with numpy's Poisson routine, which is not a certified exact
sampler; every report of this path says so. Gradients are still computed in floating point, but nothing
about the guarantee depends on their exact value any more: whatever number comes out is rounded and
checked. The public-key and protocol assumptions below are unchanged.

## The trusted-server baseline

`CentralDPSGD` uses the same round, except that participants send $S_i$ without noise and the server adds
$\mathcal{N}(0, \sigma^2 C^2 I)$ once. It is the least noise any method needs for a given budget, which
makes it the upper limit for comparisons. The server sees clean sums, so records are not protected from
it at all. The mode needs `allow_unsafe=True`, its report is marked unsafe with the reason, and it takes
no secure sum in this version (rounding clean sums before the noise changes how far one row can move
them, which S13 handles).
