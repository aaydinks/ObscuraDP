# Threat model (draft, Phase 0)

This file says who the library protects, from whom, and under which assumptions. It is a draft: the choices
marked **decision** are confirmed at the Phase 0 gate, and the tables are rechecked whenever a mode or
backend is added. Every privacy report the library prints will quote the row that applies to that run.

## What is being protected

One **record**: one row of one participant's local training table (one patient visit, one loan application).
The default promise is about adding or removing one record. If one person can have several rows, the promise
weakens in proportion to the number of rows, and the report says so.

A second, optional unit is one whole **participant** (one phone, one writer). It only makes sense when there
are thousands of participants.

## Who might attack

| Actor | What they can see | Assumed behaviour |
|---|---|---|
| Model observer | The final trained model, and every published global model along the way | Anyone, unlimited computing power |
| Aggregation server | Every message a participant sends it, plus everything a model observer sees | Follows the protocol but tries to learn from what it sees ("honest but curious") |
| Other participants | The global model each round, their own data and randomness | Up to $t$ of them may pool what they know with each other and with the server |
| Network eavesdropper | Messages in transit | Passive; transport encryption (TLS) is assumed and is outside this library |
| Key holder | The decryption key, where an encryption backend is used | Must not be the aggregation server |

## What is not covered (non-goals for v0.1)

- A server that **deviates** from the protocol, for example by asking for a single participant's ciphertext
  to be decrypted, or by sending different models to different participants.
- Participants who send poisoned updates to damage the model.
- Leaks through timing, message sizes or which participants were online.
- Anything learned from the public facts the protocol needs: the number of records each participant
  declares, the feature layout, and the model architecture.
- Records that a participant shares through some other channel.
- Choices made by looking at private results outside the library, such as tuning settings on the real test
  set.

## Today's research code (as audited)

"Yes" means the stack gives a stated, bounded protection against that actor. Finding numbers refer to
`docs/AUDIT.md`.

| Stack (Trial 07/27) | Model observer | Aggregation server | Other participants | Notes |
|---|---|---|---|---|
| `plain` | No | No | No | Baseline, no protection claimed |
| `dp` | Not as labelled | Not as labelled | Not as labelled | Budget is per round and about 4 times larger over a run (G2); resampling voids it on stroke (G3) |
| `he` | No | No | No | The server process holds the secret key (G5); decrypted values are published (G15) |
| `dp_he` | Not as labelled | Not as labelled | Not as labelled | Sum of the two rows above; encryption adds nothing against the server while the server holds the key |
| `dp_spe` | Not as labelled | Not as labelled | Not as labelled | As `dp_he`, plus the mask leaks client 0's data (G4) and 75% of each update is sent in the clear |

None of today's stacks delivers the protection the website describes. The mathematics they are built on is
sound; the gaps are in how budgets are counted, how data is prepared and who holds keys.

## Planned modes

$N$ is the number of participants in a round, $t$ the assumed maximum number that collude, $d$ the assumed
maximum number that drop out.

| Mode | Unit | Model observer | Aggregation server | Other participants | Needs |
|---|---|---|---|---|---|
| `LocalDPSGD` | record | Yes, total budget | Yes, same budget | Yes, same budget | Nothing beyond the participant's own noise |
| `LocalDPSGD` + secure sum | record | Yes | Yes, and the server also cannot see individual updates | Yes | Secure sum that hides individual messages |
| `DistributedDPSGD(t, d)` + secure sum | record | Yes, total budget | Yes, if at most $t$ collude | Yes, if at most $t$ collude and $t \ge 1$ (a participant always knows its own slice of the noise); the budget degrades in a stated way beyond that | Secure sum; one step per round; refuses SPE |
| `CentralDPSGD` | record | Yes | **No** (the server sees clean updates and adds the noise itself) | Yes | A trusted server. Shipped as an upper-bound baseline only, behind `allow_unsafe=True` |
| `ClientLevelDP` | participant | Yes | Only with a secure sum (then the participants share the noise and the server sees only the total); without one the server adds the noise and must be trusted | Yes, if at most $t$ collude | Thousands of participants; warns otherwise |

Secure-sum backends:

| Backend | What the server learns | Exact? | Weak point | Status |
|---|---|---|---|---|
| Pairwise masking over integers | Only the sum | Exact | Dropouts stall a round in v0.1 (retry with fresh randomness) | **decision:** default |
| BFV encryption on integers | Only the sum, if it does not hold the key | Exact | The key holder must refuse to decrypt anything except the final sum | Optional |
| CKKS encryption | Only the sum, if it does not hold the key | Approximate | Publishing decrypted values can expose the key (G15) | `research_only`, behind an unsafe flag |

Selective encryption (SPE) is only allowed together with `LocalDPSGD`, where the clear part of each update is
already fully noised. The report states the clear fraction and that it is protected by the noise alone.

What the pairwise-masking backend assumes, as built in S10 (`obscuradp.crypto.pairwise_masking`):

- The server follows the protocol and passes the participants' public keys on unchanged. Keys are not
  authenticated in this version, so a server that swaps keys could read updates.
- Up to $N - 2$ participants may work with the server. With two participants, each can work out the
  other's update from the sum, as with any sum of two.
- Every participant's message is needed to open a sum. A missing one means the round is repeated with a
  new sum index; masks are never reused.
- It hides individual updates. It adds no privacy budget of its own: what the sum reveals about a record
  is limited only by the noise the participants added.

What the shared-noise mode assumes, as built in S12 (`obscuradp.DistributedDPSGD`, argument in
`docs/SHARED_NOISE.md`):

- Each participant adds a slice of the noise with standard deviation $\sigma C / \sqrt{N - t - d}$. A single
  message is protected by that slice alone, so the mode is refused without a secure sum and no setting can
  force it.
- The stated budget holds against the server, against anyone who sees a model, and against groups of up to
  $t$ participants working with the server. A participant on its own counts as one: with $t = 0$ the budget
  against a single curious participant is the report's table row for 1.
- A server that swaps public keys could read single messages. In the local mode that exposes a fully noised
  update; here it exposes a message with one slice of noise, and the stated budget no longer holds. Keys
  are still not authenticated (planned with the Flower adapter, S19).
- A round in which a participant is missing fails, and **everyone who sent a message is charged for it**.
  If the missing participant works with the server, the two can strip the masks from the other messages
  and read their sum, which carries at least the full noise but is a release all the same.
- $d$ only makes every slice larger in this version, because this secure sum cannot open with anyone
  missing.
- Values are never cut off before the secure sum. They are rounded on a grid a quarter of a slice wide and
  wrap around the modulus. That the rounding leaves the budget unchanged is argued and checked numerically,
  not machine-proven. With `noise="skellam"` (S13) the gap does not exist: each record is rounded before
  the noise is added, its length is checked exactly, and the noise is whole numbers whose slices add up
  exactly. That path counts full rounds only and its budget comes from one proven method (RDP).

What the whole-participant mode assumes, as built in S14 (`obscuradp.ClientLevelDP`):

- The budget covers everything one participant contributes to training: replacing all of its rows by
  nothing changes what anyone outside can learn by at most the stated amount. The number of participants
  is treated as public.
- Each participant decides by its own secret coin whether it takes part in a round. With the secure sum,
  every participant sends a message every round (its shortened change or nothing, plus a slice of the
  noise), so neither the server nor the other participants learn who took part. The assumptions of the
  shared-noise mode above apply unchanged: the stated number of colluders, unchanged public keys, a
  charge for failed rounds, and the argued rounding step.
- Without the secure sum the server sees every change and adds the noise. That run needs
  `allow_unsafe=True` and its report carries the "trusted server" flag.
- A participant's validation and test rows are outside the budget.
- With few participants the noise that hides one of them swamps the signal (measured in S14: no better
  than guessing with five participants at a budget of 1). The library warns below 1,000.

## What every privacy report will state

1. Unit and neighbouring rule ("one record, add or remove; declared dataset sizes are public").
2. Target and spent $(\varepsilon, \delta)$ per participant and the maximum, with the accountant used and a
   cross-check from a second accountant.
3. The list of charged events, so anyone can recompute the total (`obscuradp report verify`).
4. The mode, the backend, $t$, $d$, and the row of the table above that applies.
5. Unsafe flags, if any, in plain words.
6. A fixed list of caveats: floating-point noise where it applies, the random generator used, and the
   non-goals above.

## Open questions for later phases

- Whether a key-holder committee (several parties who must cooperate to decrypt) is worth supporting before
  v0.2. It needs threshold encryption, whose Python packaging is not portable today.
- Dropout-tolerant secure sums. Flower's SecAgg+ provides this; the adapter can map onto it (S19).
- How to state protection for people with many records in one dataset without asking participants to count
  them.
