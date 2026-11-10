# Audit of the research code and the website (Phase 0)

Baseline: `main` at `34071be` (2026-11-10). Scope: `Experiment 1/`, `Experiment 2/`, `Experiment 3/`,
`Trial 07/27/`, the repo root and `website/`. Nothing was changed while auditing.

Every finding has a verdict, evidence as `path:line` on the baseline commit, a severity and the session
that fixes it (session numbers refer to `docs/ROADMAP.md`). Line numbers were collected by script from the
baseline commit, not typed from memory.

## Words used in this file

- **Privacy budget ($\varepsilon$, with $\delta$):** the number that says how much any one person's data
  may change what an outside observer can see. Smaller means stronger privacy and more added noise.
- **DP-SGD:** the standard way to train with a privacy budget. Each record's influence on a training step is
  capped ("clipped"), then random noise is added.
- **Noise multiplier ($\sigma$):** how large the added noise is, relative to the cap.
- **Accountant:** the bookkeeping code that turns "this much noise, this many steps" into a privacy budget.
  Two kinds appear here: RDP (older, looser) and PLD (tighter).
- **Poisson sampling:** each record is picked for a training step independently with probability $q$. The
  accountants assume this.
- **Round:** one cycle in which every participant trains locally and the server combines the results.
- **CKKS:** an encryption scheme that lets a server add encrypted decimal numbers. Its results are
  approximate, not exact.
- **SPE (selective parameter encryption):** encrypting only a chosen fraction $p$ of the model's numbers.
  The **mask** is the list of which numbers get encrypted.
- **SMOTE / SMOTETomek:** a way to balance a dataset by inventing extra rows of the rare class, each one a
  blend of real rows.

Severity scale: **A** breaks or overstates a privacy guarantee. **B** misleads a reader. **C** is a
correctness or reproducibility defect. **D** is hygiene.

## Summary

| ID | Finding | Verdict | Sev. | Fixed in |
|---|---|---|---|---|
| G1 | There is no library yet | Confirmed | D | S4 to S6 |
| G2 | The privacy budget is per round, and accounting does not match what runs | Confirmed | A | S7, S8 |
| G3 | SMOTETomek before DP-SGD voids the per-record guarantee | Confirmed | A | S9 |
| G4 | The SPE mask is computed from one client's raw data and is not accounted | Confirmed | A | S10 |
| G5 | One object holds the secret key and does everything | Confirmed | A | S10 |
| G6 | DP and encryption are combined in the most expensive way | Confirmed, design corrected | C | S12 to S14 |
| G7 | The bytes target contradicts the measured encryption cost | Confirmed | B | S10, S14 |
| G8 | The attacker-advantage target is worded as a guarantee | Confirmed | B | S3, S15 |
| G9 | Other website and README mismatches | Confirmed | B | S3, S22 |
| G10 | Datasets are vendored twice with no license or source link | Confirmed | D | gate, S17 |
| G11 | The planned benchmark does not fit a CPU budget as written | Confirmed | C | S17, S18 |
| G12 | Preprocessing uses pooled data | New | A | S9 |
| G13 | No validation split; settings were chosen on test results | New | C | S9, S11 |
| G14 | The "DP" mask uses about 8 times too little noise | New | A | S10 |
| G15 | Decrypted CKKS values are published every round | New | A | S10 |
| G16 | Noise quality: ordinary random generator, floating-point noise | New | A | S8, S13 |
| G17 | Seed collisions and non-reproducible encrypted runs | New | C | S6 |
| G18 | Silent failures in the training loop and drivers | New | C | S6 |
| G19 | The verification harness prints fingerprints but never checks them | New | C | S6 |
| G20 | Experiment 3: overhead benchmark ignores the mask; some model state is not combined | New | C | S5 |
| G21 | Website promises with no counterpart in code | New | B | S3, S7 |
| G22 | Statistics: population standard deviation over 5 seeds, no intervals | New | C | S17, S20 |
| G23 | About 40 numbers typed into website copy by hand | New | D | S22 |
| G24 | Request logging is on while the README promises no analytics | New | B | S3 |
| G25 | The stroke test set is 84 rows | New | B | S11 |
| G26 | Committed results predate the committed code; no provenance | New | C | Phase 0, S6 |
| G27 | Empty sampled batches are skipped; dependency pins are behind | New | A | S8 |

## Findings from the prompt

### G1. There is no library yet. Confirmed (D)

- No `pyproject.toml`, `setup.py`, `LICENSE`, `tests/`, `CITATION.cff` or git tags. The only workflow is
  `.github/workflows/website-ci.yml`, added in aaydinks/ObscuraDP#7; nothing tests Python.
- Four separate packages with overlapping code: `Experiment 1/stroke_fl/`, `Experiment 2/credit_fl/`,
  `Experiment 3/fedml_he/`, `Trial 07/27/obscura/`. A duplication table is in "Reusable code" below.
- `Trial 07/27/obscura/__init__.py:28` claims `__version__ = "1.0.0"`.
- Root `README.md:2` stops mid-sentence ("It incorporates").
- The root `.gitignore` is the stock Python template and its rules are not anchored to the root:
  `.gitignore:11` (`build/`), `.gitignore:17` (`lib/`), `.gitignore:154` (`env/`). `git check-ignore -v`
  confirms that `src/obscuradp/lib/x.py`, `docs/build/x` and `src/obscuradp/env/x.py` would be silently
  ignored. This already happened once to `website/src/lib` (commit `c4925a3`).
- No file anywhere uses pytest or unittest. `Experiment 1/smoke_test.py` has no assertions at all.

### G2. The privacy budget is per round, and accounting does not match what runs. Confirmed (A)

Per round, not per run:
- `Trial 07/27/obscura/config.py:97` documents the per-round convention; the grid is at `Trial 07/27/obscura/config.py:100`.
- $\sigma$ is solved once per client for the local epochs of a single round:
  `Trial 07/27/obscura/orchestrator.py:206-207`, `Trial 07/27/obscura/dp.py:29`, `Trial 07/27/obscura/dp.py:46`.
- Every call to `make_private` builds a new Opacus `PrivacyEngine` (`Trial 07/27/obscura/dp.py:102`), once per client per round
  (`Trial 07/27/obscura/fedprox.py:115`), and discards it. Nothing tracks spend across rounds.
- The run-level number is only computed afterwards for reporting (`Trial 07/27/obscura/orchestrator.py:391`, `Trial 07/27/obscura/orchestrator.py:396`)
  and is never enforced. In `Trial 07/27/results/arms.json` the stroke `dp` arm labelled 9 has a composed
  budget of about 39.17. A labelled 1 composes to about 4.78 over 20 rounds.
- Experiment 1 uses the same per-round convention (`Experiment 1/stroke_fl/train.py:55`, `Experiment 1/stroke_fl/train.py:66`,
  `Experiment 1/README.md:24`, `Experiment 1/README.md:136-137`). Experiment 2 uses a total over all rounds
  (`Experiment 2/credit_fl/train.py:62`, `Experiment 2/credit_fl/train.py:67`, `Experiment 2/credit_fl/train.py:88`). The two conventions are mixed on the
  website (see `docs/WEBSITE_MISMATCHES.md`).

Accounted steps differ from executed steps:
- The accountant is given $q = B / n$ and $\lfloor E / q \rfloor$ steps (`Trial 07/27/obscura/dp.py:46`, `Trial 07/27/obscura/dp.py:71`).
- Training uses a shuffled loader (`Trial 07/27/obscura/fedprox.py:85`) that Opacus replaces with Poisson sampling at
  $q = 1 / \text{len(loader)}$. For a client with 484 rows the accountant assumes $q = 0.0661$ and 45 steps,
  while training runs $q = 0.0625$ and 48 steps. The two never agree exactly.
- `dp_noise_std` divides by the nominal batch size (`Trial 07/27/obscura/dp.py:125`) where Opacus divides by its expected batch
  size (29 to 32 here), so the reported noise level is off by up to about 10%. `VERIFICATION.md` already
  records this as a correction.

Fix: one total $(\varepsilon, \delta)$ per run, a ledger that is charged before every release, and a training
step whose sampling matches the accountant exactly (S7, S8).

### G3. SMOTETomek before DP-SGD voids the per-record guarantee. Confirmed (A)

- Resampling: `Trial 07/27/obscura/data.py:280`, `Trial 07/27/obscura/data.py:290`. It is on by default for stroke in every arm
  (`Trial 07/27/obscura/data.py:311`).
- The noise is calibrated on the resampled row count (`Trial 07/27/obscura/orchestrator.py:207` uses `len(yc)` of the resampled
  data). `Trial 07/27/obscura/config.py:61` notes the largest client builds 2,460 positive rows from 22 real ones.
- Consequence: clipping caps the influence of one training row, but one real patient now influences many
  rows. The stated budget is about synthetic rows and is not a valid guarantee for real patients. Published
  work reaches the same conclusion (arXiv:2411.05733; arXiv:2510.15083).
- Experiment 1 does the same (`Experiment 1/stroke_fl/train.py:40`, `Experiment 1/stroke_fl/train.py:86`).

Fix: refuse non-private oversampling inside a DP run unless `allow_unsafe=True`; offer class weighting from a
public prior instead (S9). The stroke results must be re-run (S11).

### G4. The SPE mask is computed from one client's raw data and is not accounted. Confirmed (A)

- Built once before round 1 from client 0's data: `Trial 07/27/obscura/orchestrator.py:218`, `Trial 07/27/obscura/orchestrator.py:222-223`.
- Default mode is `"clean"`, meaning gradients without any noise: `Trial 07/27/obscura/config.py:127`, applied at
  `Trial 07/27/obscura/orchestrator.py:132`; scoring at `Trial 07/27/obscura/sensitivity.py:47`.
- The mask is then used by every client, so information derived from client 0's raw records is released
  with no protection and no entry in any budget.
- Under SPE the unencrypted share of each update is averaged in the clear from each client's own vector
  (`Trial 07/27/obscura/he.py:184-188`, `Trial 07/27/obscura/he.py:293`). At the default $p = 0.25$ the server sees 75% of every
  client's update, protected only by the DP noise.

Fix: data-independent masks by default; any data-dependent mask is charged to the ledger (S10).

### G5. One object holds the secret key and does everything. Confirmed (A)

- `Trial 07/27/obscura/he.py:43-50` creates one TenSEAL context including the secret key. The same object
  encrypts each client's vector (`Trial 07/27/obscura/he.py:94`), sums, and decrypts (`Trial 07/27/obscura/he.py:162`, `Trial 07/27/obscura/he.py:286`). There is no
  public-only copy anywhere in the repo.
- Experiment 2 is the same (`Experiment 2/run_experiments.py:97`, `Experiment 2/run_experiments.py:116`), while its
  docstring says "the server holds no secret key and never decrypts" (`Experiment 2/credit_fl/he.py:49`).
  Experiment 3 too (`Experiment 3/fedml_he/he.py:24`).
- In a real deployment whoever holds that object can decrypt any single client's update.

Fix: separate `KeyHolder`, `ClientEncryptor` and `Aggregator` roles, and a test that the aggregator cannot
decrypt (S10).

### G6. DP and encryption are combined in the most expensive way. Confirmed, design corrected (C)

Each client adds the full noise locally, then the already-noised update is encrypted. If the server can only
ever see the sum of all updates, less noise per client is enough for the same budget.

Correction to the prompt: the "each client adds a share of the noise" idea is only sound when every client
sends **one** clipped gradient sum per round, computed at the shared model. With several local steps a
changed record alters every later local step, and those steps were protected by only a fraction of the noise.
So there are two sound designs for per-record privacy, and the library ships both (S8, S12):

1. Local mode: several local steps, full noise at each client. Safe against everyone, lowest accuracy.
2. Distributed mode: one step per round, a secure sum, and a noise share sized so that the honest clients
   alone supply enough noise, $\sigma C / \sqrt{N - t - d}$ for $N$ clients, at most $t$ colluding and $d$
   dropping out.

The expected gain at 5 sites is about 2 to 5 times less noise on the averaged gradient. That figure is an
estimate and is measured in S14.

### G7. The bytes target contradicts the measured encryption cost. Confirmed (B)

The website target is at least 50% fewer upload bytes per round than plain federated learning
(`website/src/sections/Impact.astro:14`). Measured CKKS payloads are 14.365 times larger at large model
sizes, 18.76 times at 3,137 parameters and about 63 times at 929 parameters (`Trial 07/27/README.md`,
section 7). No setting of the current path can meet the target. `docs/KPI_MAPPING.md` gives the bit budget
that can.

### G8. The attacker-advantage target is worded as a guarantee. Confirmed (B)

- `website/src/sections/Impact.astro:10` says an attacker "gets it right at most 52.5% of the time" at a
  budget of 1.
- `website/src/components/visuals/PrivacySlider.astro:90` shows 73% for the same budget, from
  `website/src/lib/laplace.ts:14-16` ($e^{\varepsilon} / (1 + e^{\varepsilon})$).
- For any $(\varepsilon, \delta)$ guarantee the best possible advantage (true-positive rate minus
  false-positive rate) is bounded by

$$
\mathrm{Adv} \le \frac{e^{\varepsilon} - 1 + 2\delta}{e^{\varepsilon} + 1},
$$

  which is about 0.462 at a budget of 1. A guaranteed 5% would need a budget near 0.1.
- `website/src/components/visuals/PrivacySlider.astro:95` says "Real attacks measured in audits land far lower". No audit has been run.

Fix: the 5% figure becomes a measured target for a named attack, shown beside the ceiling (S3, S15).

### G9. Other website and README mismatches. Confirmed (B)

The full list with proposed wording is in `docs/WEBSITE_MISMATCHES.md`. The main ones:
- "Open source" is claimed in `website/src/sections/Hero.astro:9`, `website/src/lib/site.ts:8`,
  `website/scripts/og/template.html:95` and the comparison table (`website/src/sections/Framework.astro:11`),
  but the repository has no license file.
- `website/src/sections/Framework.astro:109`: the server "never sees the unencrypted values". False under
  SPE (G4) and under the current key handling (G5).
- "Pre-registered" is stated as done in `website/src/sections/Roadmap.astro:12` and
  `website/src/sections/Research.astro:26`, with an empty link (`website/src/lib/site.ts:12`).
- Trial 07/27 is not on the site and `website/scripts/sync-data.mjs` copies none of its results.
- Targets 1 and 3 (records protected, organizations onboarded) have no way to be measured, and the project
  promises no telemetry.

### G10. Datasets are vendored twice with no license or source link. Confirmed (D)

- `Experiment 1/data/healthcare-dataset-stroke-data.csv` and `Trial 07/27/data/healthcare-dataset-stroke-data.csv`
  are byte-identical (md5 `3f25a4c2...`). Same for `loan_approval_dataset.csv` in `Experiment 2/data/` and
  `Trial 07/27/data/` (md5 `3d8ecb6a...`).
- The READMEs say both files came from mirrors because Kaggle needs a login
  (`Experiment 1/REPRODUCTION.md:74`, `Experiment 2/README.md:137-138`). No URL and no license statement is
  given for either.

Decision needed at the gate: may these files stay in the repository? The library itself will never ship
datasets (S17 downloads with checksums).

### G11. The planned benchmark does not fit a CPU budget as written. Confirmed (C)

Measured in this session on 4 CPU cores: the 20-run arm matrix on two small tabular models took 7 minutes
53 seconds (`docs/ROADMAP_STATUS.md`). The earlier 105-run stroke campaign took about 1.6 hours. A full FEMNIST and Shakespeare grid with encryption in the
loop would take days. `docs/ROADMAP.md` sizes the benchmark at about 17 CPU hours with an explicit cut order.

## New findings

### G12. Preprocessing uses pooled data (A)

Scalers are fit on the pooled training rows of all clients (`Trial 07/27/obscura/data.py:142`, `Trial 07/27/obscura/data.py:181`),
the category layout is derived from the full dataset (`Trial 07/27/obscura/data.py:49`), and the split into clients sees all
labels (`Trial 07/27/obscura/data.py:227`). A real federation has no party that can do this, and pooled statistics are
themselves a release about every record. Fix: preprocessing from declared feature bounds and categories
(S9).

### G13. No validation split; settings were chosen on test results (C)

The learning rate, round count, local epochs and weighting rule are each justified in `config.py` by their
effect on test metrics (`Trial 07/27/obscura/config.py:38`, `Trial 07/27/obscura/config.py:67`, `Trial 07/27/obscura/config.py:78`). There is no
validation set. Reported test numbers are therefore optimistic. Fix: a validation split inside each client and
tuning on a synthetic stand-in dataset (S9, S11).

### G14. The "DP" mask uses about 8 times too little noise (A)

`Trial 07/27/obscura/sensitivity.py:88` sets the batch size to the scoring batch (256 rows,
`Trial 07/27/obscura/sensitivity.py:349`) and `Trial 07/27/obscura/sensitivity.py:125` divides the noise by it. Training uses batches of about 32. So
the "DP-aware" mask results in Trial 07/27 section 6 were produced with roughly one eighth of the training
noise, and the release was never charged to any budget. Fix in S10; the section 6 numbers need a correction
note.

### G15. Decrypted CKKS values are published every round (A)

CKKS results carry a small error term. If a party sees both a ciphertext and its decrypted value, that error
can be used to recover the secret key (Li and Micciancio, Eurocrypt 2021). Here the decrypted average becomes
the public global model every round (`Trial 07/27/obscura/orchestrator.py:299`), decrypted as 64-bit floats
(`Trial 07/27/obscura/he.py:162`, `Trial 07/27/obscura/he.py:286`). The only thing in the way is the cast to 32-bit floats at
`Trial 07/27/obscura/models.py:152`, which was not designed as a defence. The DP noise does not help because
it is inside the encrypted message, not added at decryption. `Trial 07/27/obscura/he.py:50` also generates rotation keys that
are never used.

Fix: pairwise masking over integers as the default secure sum; CKKS kept as `research_only` behind an unsafe
flag; exact integer BFV as the encryption option (S10).

### G16. Noise quality (A)

`make_private` is called without Opacus's secure mode (`Trial 07/27/obscura/dp.py:103`), so noise comes
from the ordinary torch generator. Opacus's secure mode depends on a package that is no longer maintained, so
turning it on is not the answer. Separately, Gaussian noise in floating point and clipping in floating point
have published attacks. Fix: an operating-system-seeded generator for real runs, a seeded one only for tests
and marked in the report (S8); integer noise on the secure-sum path (S13).

### G17. Seed collisions and non-reproducible encrypted runs (C)

- Per-client seeds are `seed * 1000 + round_idx * 10 + client_idx` (`Trial 07/27/obscura/orchestrator.py:278`).
  Two different (round, client) pairs share a seed as soon as there are 10 or more clients.
- The encryption library's randomness cannot be seeded, and the decrypted average feeds the next round
  (`Trial 07/27/obscura/orchestrator.py:299`), so encrypted arms are not bit-for-bit reproducible.
- `Trial 07/27/run_multiseed.py:18` fixes 2 threads; other drivers do not.
- `Trial 07/27/run_smoke.py:114` and `Trial 07/27/run_smoke.py:188` draw unseeded random inputs.

### G18. Silent failures in the training loop and drivers (C)

- `Trial 07/27/obscura/fedprox.py:69-71`: `x = x or default` treats a legitimate 0 as "not given".
- `Trial 07/27/obscura/orchestrator.py:275`: an unknown dataset name silently gets stroke's learning rate; `Trial 07/27/obscura/orchestrator.py:169`
  silently defaults the weighting rule.
- `Trial 07/27/obscura/orchestrator.py:172`: passing `prepared=` silently overrides `use_smote` and is not checked against `seed`.
- `Trial 07/27/obscura/orchestrator.py:449`: `use_smote` is written to results as given, usually `None`, so a result file cannot
  say whether resampling ran.
- `Trial 07/27/run_multiseed.py:82`: the checkpoint condition `(eps_idx + 1) * (frac_idx + 1) % 7 == 0`
  almost never fires.
- `Trial 07/27/run_losslessness.py:121` hard-codes `weighted=True` on stroke, unlike the trained arms
  (`Trial 07/27/obscura/config.py:78`).
- `Experiment 1/stroke_fl/train.py:169` changes global configuration inside a function call.

### G19. The verification harness prints fingerprints but never checks them (C)

`Trial 07/27/verify_claims.py:128-151` computes sha256 prefixes of the results files and prints them. The
comparison with the values in `VERIFICATION.md` is left to the reader. The interpreter path in the docstring
is machine-specific (`Trial 07/27/verify_claims.py:45`). The verdict-table parser (`Trial 07/27/verify_claims.py:196`) is sound and
is reused.

### G20. Experiment 3 details (C)

- The overhead-versus-fraction benchmark encrypts the first $p \cdot n$ numbers, not the masked ones
  (`Experiment 3/fedml_he/bench.py:42-43`). That is fine for timing but it is not "SPE with a sensitivity
  mask" as the write-up implies.
- Flattening uses `named_parameters()` only (`Experiment 3/fedml_he/model_utils.py:14`), so running
  statistics of normalization layers are never combined across clients.
- `Experiment 3/fedml_he/models.py:4` says the small CNN has about 0.1M parameters; it has 282,250.

### G21. Website promises with no counterpart in code (B)

- "Personalized ε" (`website/src/sections/Framework.astro:67`) does not exist in any experiment.
- "Same answer either way" (`website/src/components/visuals/HeroDemo.astro:100`) overstates the guarantee:
  the two outputs are draws from two similar distributions, not the same answer.
- The pipeline picture (`website/src/components/visuals/Pipeline.astro:6-7`, `website/src/components/visuals/Pipeline.astro:26`) never says
  who holds the key or who decrypts.

### G22. Statistics (C)

`website/src/pages/research.astro:91` reports the population standard deviation over five seeds. With five
runs this understates spread by about 11% compared with the sample standard deviation, and there are no
confidence intervals or paired comparisons anywhere.

### G23. Numbers typed into website copy by hand (D)

About 40 figures in `website/src/sections/Research.astro` and `website/src/pages/research.astro` (for example
`website/src/sections/Research.astro:52`) are literals, not read from the result files. All of them match the data today (checked
by script during this audit). Nothing keeps them in sync. Fix: a claims registry checked at build time (S22).

### G24. Request logging versus "no analytics" (B)

`website/wrangler.jsonc:12` enables Cloudflare Workers observability (request logs), while
`website/README.md:125` says "No cookies, no third-party trackers, no analytics". Either switch it off or say
that request logs exist (S3).

### G25. The stroke test set is 84 rows (B)

The stroke test set is all 42 positive rows plus 42 sampled negatives (`Trial 07/27/obscura/data.py:149`,
`Experiment 1/stroke_fl/data.py:66`). One patient is 2.4 points of recall. The "DP beats non-private" result
is 29 of 42 against 24 of 42. A balanced set also hides the real 4.3% prevalence. Fix: report a
natural-prevalence test set beside it and state intervals (S11).

### G26. Committed results predate the committed code (C)

`Trial 07/27/results/arms.json` has no `wire_payload_ratio` field in its crypto block, but the current
`orchestrator.py` writes one, so the committed numbers were produced by an earlier version of the code. The
`meta` block records no commit hash, package versions or hardware. The outcome of regenerating the results in
this session: all 8 unencrypted runs reproduce bit for bit and the 12 encrypted runs agree within
$7 \times 10^{-6}$ (details in `docs/ROADMAP_STATUS.md`). So the numbers are reproducible, but the files
themselves cannot prove where they came from. Fix: a reproducibility manifest in every result file (S6).

### G27. Empty sampled batches are skipped; pins are behind (A)

- `Trial 07/27/obscura/fedprox.py:122` skips a step when Poisson sampling returns no rows. Opacus would take
  a noise-only step there. Skipping it means the number of noisy steps taken differs from the number the
  accountant assumed, and "this step had no data" becomes visible in the training path.
- `Trial 07/27/requirements.txt:13-14` pins `tenseal==0.3.16` and `torch==2.13.0`; current releases are
  0.3.18 and 2.15.0.

## Reusable code

Port these, do not rewrite them.

| Purpose | Source |
|---|---|
| Client splits | `Trial 07/27/obscura/data.py`: `dirichlet_partition`, `iid_partition` |
| Model to flat vector and back | `Trial 07/27/obscura/models.py`: `get_weights`, `set_weights`, `flatten_state`, `unflatten_state`, `param_index_map` |
| Chunked encryption and timing | `Trial 07/27/obscura/he.py`: `chunk_bounds`, `encrypt_vector`, `fedavg_he_timed`, `fedavg_plaintext` |
| Aggregation seam | `Trial 07/27/obscura/aggregate.py`: `fedavg_weights`, `aggregate_states`, `plaintext_reference` |
| Masks and agreement metrics | `Trial 07/27/obscura/sensitivity.py`: `build_mask`, `tensor_selection`, `tensor_selection_agreement`, `mask_agreement`, `random_mask` |
| Independent per-record gradient reference | `Trial 07/27/obscura/sensitivity.py`: `dp_sensitivity_scores` |
| Run summaries | `Trial 07/27/obscura/metrics.py`: `summarize` |
| Payload model | `Trial 07/27/obscura/bench.py`: `ciphertext_quantum`, `payload_ratio_model` |
| Model registry, generic flatten | `Experiment 3/fedml_he/models.py`: `BUILDERS`, `build_model`; `Experiment 3/fedml_he/model_utils.py`: `flatten_params` |
| FedProx term applied after the private step | `Experiment 1/stroke_fl/train.py:116`, `Trial 07/27/obscura/fedprox.py:137` |
| Test patterns | `Trial 07/27/run_smoke.py` (7 checks), `Trial 07/27/verify_claims.py` (`parse_documented_verdicts`, independent references) |
| Website charts | `website/src/lib/chart.ts`, `website/src/components/charts/Exp1RecallCurve.astro`, `Exp3OverheadScaling.astro` |
