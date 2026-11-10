# ObscuraDP: from research repository to installable library

> **How to use this file.** Paste everything below the line into a fresh Claude Code session opened on
> the `aaydinks/ObscuraDP` repository. The prompt is written to be re-entered many times: each session
> reads the progress tracker, finishes the next unfinished phase, updates the tracker, and stops. Do not
> edit the prompt between sessions unless the plan itself changes; record progress in
> `docs/ROADMAP_STATUS.md` instead.

---

## 0. Your role and how to work

You are the lead engineer turning ObscuraDP from a collection of paper replications into a real,
installable, documented Python library named **`obscuradp`** that combines differential privacy (DP),
federated learning (FL) and homomorphic encryption (HE) behind one coherent API, with an honest,
machine-checked privacy report for every training run.

Work rules for every session:

1. **Start by reading `docs/ROADMAP_STATUS.md`.** If it does not exist, you are in Phase 0: create it
   from the phase checklist in Section 5. Then do only the **next unfinished phase** (or the next
   unfinished part of it if the phase is large). Do not jump ahead.
2. **One phase, one branch, one draft PR.** Branch name `roadmap/phase-N-<slug>` unless the session
   assigns a branch, in which case use that one. End each session by updating `docs/ROADMAP_STATUS.md`
   (what was done, what was verified and how, what is left, open questions) in the same PR.
3. **Gates.** Phases marked **GATE** end with a short decision memo in the PR description and stop for
   human review. Do not start the next phase in the same session.
4. **Evidence over assertion.** Every number in docs, README files or the website must be generated
   from a results file produced by committed code. Never hand-edit a results JSON. Keep the culture of
   `Trial 07/27/verify_claims.py` (sha256-pinned results, a verdict table that is re-parsed on every
   run, retractions recorded rather than deleted).
5. **Never weaken a privacy guarantee to make a number look better.** If a fix makes results worse,
   publish the worse number next to the old one and explain why.
6. **Tests are not optional.** Never skip, xfail or delete a test to get CI green.
7. **Ask at most three questions per session**, and only for decisions that genuinely belong to the
   owner. Otherwise pick the sensible default, state it in the PR, and continue.

Writing rules (the owner's preferences, apply to every file you write and every message you send):

- Never use em dashes. Use commas, periods, parentheses or colons. (`website/scripts/lint-copy.mjs`
  already enforces this for the site; add the same check to the docs build.)
- Write math in LaTeX in Markdown docs (`$...$` inline, `$$...$$` display), except trivial expressions
  like `n`, `2n` or a single number with a unit.
- Website copy and user-facing explanations use everyday words. A specialized term may appear only after
  it has been explained in plain language.

---

## 1. Decisions already made (do not re-litigate)

| Decision | Choice | Consequence |
|---|---|---|
| Foundation | **Hybrid** | A standalone core (`obscuradp`: accounting, mechanisms, encryption, selection, auditing, single-process simulator) with **no hard dependency on Flower**, plus an optional extra `obscuradp[flower]` that provides a Flower strategy and client mods for real multi-machine deployments. |
| Budget | **About 3 months, part-time, CPU only** | Every benchmark must be sized to finish on a laptop-class CPU. State the wall-clock cost of every experiment. No GPU-only code paths in CI. |
| Website | **Propose, then edit** | When the site disagrees with the math or the code, list each mismatch with reasoning in the PR, then fix the copy in the same PR so the site and library stay in sync. |
| Package name | `obscuradp` on PyPI | Verified free as of 2026-11-10 (`obscura` is taken; `obscuradp` and `obscura-dp` normalize to the same free name). Reserve it on TestPyPI first. |
| License | Apache-2.0 (default) | Patent grant matters for a privacy and crypto library. The website promises "permissively licensed". If the owner prefers MIT, they will say so in review. |

---

## 2. What exists today (read these before writing code)

**Repository map**

- `Experiment 1/` (`stroke_fl/`): replication of arXiv:2508.10017. Non-IID Dirichlet clients,
  per-client SMOTETomek, FedProx, Opacus DP-SGD, stroke data. ε is a **per-round** budget.
- `Experiment 2/` (`credit_fl/`): replication of Sci. Rep. 10.1038/s41598-025-34536-9. Standard FL vs
  DP-FL vs HE-FL on loan approval. TenSEAL CKKS, `[60,40,40,60]` chain. ε reported cumulatively.
- `Experiment 3/` (`fedml_he/`): replication of FedML-HE, arXiv:2303.10837. Chunked CKKS, Selective
  Parameter Encryption (SPE), $O(n)$ overhead, losslessness. No DP.
- `Trial 07/27/` (`obscura/`): **the most advanced code and the seed of the library.** Composes all
  three into one stack with five arms (`plain`, `dp`, `he`, `dp_he`, `dp_spe`), weighted homomorphic
  aggregation, a joint $\varepsilon \times p$ frontier, losslessness under DP noise, DP-aware SPE masks,
  overhead modelling, and an independent verification harness. Read `README.md`, `DIAGNOSIS.md`,
  `VERIFICATION.md`, `obscura/config.py`, `obscura/orchestrator.py`, `obscura/aggregate.py`,
  `obscura/he.py`, `obscura/dp.py`, `obscura/sensitivity.py`.
- `website/`: Astro site deployed to Cloudflare Workers (obscuradp.org). Goals, KPIs and roadmap live in
  `src/sections/{Hero,Mission,Problem,Framework,Research,Roadmap,Impact}.astro`,
  `src/pages/research.astro`, `src/components/visuals/PrivacySlider.astro`, `src/lib/site.ts`.
  Result JSONs are copied in by `npm run sync-data`.

**Project goals and KPIs as published on the website**

- Mission: "Verifiable privacy for every innovator, not just big tech." Principles: **Free, Portable,
  Verifiable.** One privacy dial (ε) set per use case.
- Expected outcomes: (a) the first reproducible benchmark of the full DP + FL + HE stack under non-IID
  conditions (FEMNIST, Shakespeare, synthetic health data, several ε values, five seeds, pre-registered);
  (b) "a working library any developer can install, free to use across borders."
- Impact targets (labelled "targets, not results"):
  1. **1M records** protected under mathematical privacy guarantees.
  2. **$\le 5\%$ attacker advantage at $\varepsilon = 1$** (advantage = TPR minus FPR).
  3. **25 to 50 organizations** onboarded and running the framework.
  4. **$\ge 50\%$ fewer bytes per training round** than an uncompressed FL baseline.
- Roadmap 2026-27 ("Integrate and audit"): full security layer, privacy auditing against membership
  inference, attribute inference and gradient inversion, permissively licensed open-source release,
  first partners (small clinics or school systems without comprehensive data-protection law).
  2028+: low-bandwidth deployment, documentation for low-resource settings, hardware acceleration.
- Site promises: no telemetry, no trackers. The library must honour this too (no phone-home, ever).

**Current dependency landscape (checked 2026-11-10; re-check versions before pinning)**

`opacus 1.6.0`, `tenseal 0.3.18` (released 2026-09, but maintenance signals are weak: treat it as a
replaceable backend), `openfhe 1.6.1` (Python wrapper of OpenFHE, supports threshold FHE in C++; check
whether threshold CKKS is exposed in Python before relying on it), `flwr 1.43.0` (central and local DP
mods, SecAgg+; DP labelled "preview"), `flwr-datasets 0.6.1` (FEMNIST and Shakespeare with natural
partitions via `NaturalIdPartitioner`), `dp-accounting 0.6.0` (Google: RDP and PLD accountants).

---

## 3. Audit findings you must address (verify each one first)

These were found by reading the code and the site. **Phase 0 must confirm or refute each finding with
file and line evidence** before anything is fixed. Some may be wrong; say so if they are.

**G1. There is no library yet.** Four near-duplicate packages (`stroke_fl`, `credit_fl`, `fedml_he`,
`obscura`) live in folders whose names contain spaces and a slash-like date (`Trial 07/27`). All
configuration is module-level constants in `config.py`. There is no `pyproject.toml`, no `LICENSE`, no
CI (`.github/` does not exist), no unit tests (only smoke scripts), and `obscura/__init__.py` claims
`__version__ = "1.0.0"`. The root `README.md` has a truncated first paragraph ("It incorporates").
The root `.gitignore` is the stock Python template, whose `lib/` rule already silently excluded
`website/src/lib` once.

**G2. The privacy budget is reported per round, not end to end.** `config.EPSILON_GRID` is a per-round
budget (Experiment 1 convention). In Trial 07/27 a labelled $\varepsilon = 1$ composes to about
$\varepsilon \approx 4.78$ over 20 rounds, and $\varepsilon = 9$ to about $39$. The website quotes
Experiment 1 at "ε = 9" without saying it is per round. Also, `dp.get_sigma` solves $\sigma$ at sample
rate `batch_size / n_local` while Opacus accounts with `1 / len(loader)`, and `dp_noise_std` uses
`BATCH_SIZE` where Opacus uses `expected_batch_size` (see `VERIFICATION.md` claims 4 and the closing
corrections). The library must calibrate noise to a **total** $(\varepsilon, \delta)$ for the whole run,
state the **unit of privacy** (one record, or one client), and use the same sampling scheme in
calibration and in training (Poisson sampling, as DP-SGD analysis assumes).

**G3. SMOTETomek before DP-SGD breaks the record-level guarantee.** SMOTE builds synthetic minority rows
by interpolating real ones, so one real patient can influence many training rows, and DP-SGD's
accounting (which assumes each real record appears once) no longer bounds that patient's influence.
Literature: arXiv:2411.05733 (DP under class imbalance; SMOTE's sensitivity makes the effective ε
"meaninglessly large"), arXiv:2510.15083 ("SMOTE and Mirrors": attacks that reconstruct minority
records). Trial 07/27 even documents that the largest client interpolates 2,460 positives from 22 real
ones. The library must not offer non-private oversampling inside a DP pipeline without either
(a) refusing it, (b) accounting for it (group privacy or a proven sensitivity bound), or (c) replacing
it with a DP-safe alternative (class-weighted loss, DP-aware reweighting, or a DP-SMOTE variant). This
also explains the "DP beats non-private on stroke" finding in Trial 07/27 Section 8 and needs re-running.

**G4. The SPE mask leaks and is not accounted.** `orchestrator.run_stack` builds the mask once from
**client 0's raw data** with **clean, non-DP gradients** by default (`MASK_MODE_DEFAULT = "clean"`),
then shares it with every client. That is an unaccounted release of client 0's information. In
addition, with SPE the unencrypted fraction $1 - p$ of every update is sent to the server in the clear,
so the website's "the server never sees the unencrypted values" is false for SPE. SPE only makes sense
under a stated threat model in which the clear part is protected by DP. Options: data-independent masks
(layer or position based), masks from public or proxy data, or DP masks whose cost is charged to the
privacy ledger.

**G5. HE gives no protection against the server as built.** `he.make_context()` creates one TenSEAL
context including the secret key, used both to encrypt and to decrypt in a single process. In any real
deployment whoever holds that context can decrypt each client's individual ciphertext. A library needs
explicit key management: clients hold the secret key and the server only aggregates (decryption by a
client or a key-holder committee), or threshold/multiparty CKKS (OpenFHE), or secure aggregation by
pairwise masking (Bonawitz et al., SecAgg+ as in Flower) as the default and CKKS as an option.

**G6. DP and HE are composed in the most expensive way.** Each client adds the full DP noise locally
(local DP-SGD) and the updates are then also encrypted. If aggregation is secure (only the sum is
revealed), each client only needs to add a share of the noise: **distributed DP** with the Distributed
Discrete Gaussian (Kairouz, Liu, Steinke, arXiv:2102.06387) or Skellam mechanism (Agarwal, Kairouz,
Liu, arXiv:2110.04995) over secure aggregation gives central-DP utility with protection against an
honest-but-curious server. This is the single biggest lever for "better results" at the same ε, and it
also makes HE and DP genuinely complementary rather than redundant. Client-level DP (DP-FedAvg, clip
and noise whole client updates) is the standard unit for cross-device FL; record-level DP-SGD inside
each silo is the right unit for cross-silo hospitals. Support both, explicitly.

**G7. KPI 4 (bytes) contradicts the measured HE cost.** CKKS ciphertexts are about $14.4\times$ the
plaintext size asymptotically (measured: 14.365×), and up to $63\times$ on small models because of
whole-ciphertext padding. "$\ge 50\%$ fewer bytes than uncompressed FL" is impossible with the current
CKKS path. Achieving it needs a different design: quantized updates over modular secure aggregation
(which distributed DP already requires), smaller ring dimensions where precision allows, symmetric or
seeded ciphertext serialization, SPE that sends the clear part quantized, and update compression whose
privacy effect is accounted for. The KPI must be defined precisely (which baseline, which model, which
precision) and measured by the benchmark suite.

**G8. KPI 2 (attacker advantage) is stated as a guarantee it cannot be.** For an
$(\varepsilon, \delta)$-DP mechanism the best possible membership-inference advantage is

$$
\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR} \;\le\; \frac{e^{\varepsilon} - 1 + 2\delta}{e^{\varepsilon} + 1},
$$

which is about $0.462$ at $\varepsilon = 1$ (attacker accuracy about $73\%$, not $52.5\%$). Guaranteeing
$\le 5\%$ would need $\varepsilon \approx 2\,\operatorname{artanh}(0.05) \approx 0.1$. So
`website/src/sections/Impact.astro` ("gets it right at most 52.5% of the time") is wrong, while
`PrivacySlider.astro` correctly shows the theoretical ceiling. KPI 2 must be reframed as an **empirical
audit target**: measured advantage of a named, strong attack (for example LiRA-style membership
inference, or the one-run audit of Steinke, Nasr, Jagielski, NeurIPS 2023, arXiv:2305.08846), always
reported next to the theoretical ceiling above.

**G9. Other website and README mismatches.** Examples to check: the Framework compare table marks
Obscura DP "open source" while there is no license; "We never receive raw data ... the server never sees
the unencrypted values" (false under SPE and under the current key handling); the Roadmap says DP and HE
are integrated, but Trial 07/27 (where that happened) is not on the site at all; KPI 1 ("1M records")
and KPI 3 ("25 to 50 organizations") have no measurement method, and the site promises no telemetry, so
they can only be measured through partner reports, case studies, PyPI download statistics and public
dependents. Write the measurement method down rather than inventing a counter.

**G10. Data and reproducibility hygiene.** CSVs (Kaggle stroke and loan data) are vendored in several
folders. Check their licenses before redistributing; the wheel must never ship datasets (download with
checksums into a cache instead). Results JSONs are the source of truth for the site; keep it that way.

**G11. The planned benchmark does not fit a CPU budget as written.** Full FEMNIST (about 3,500 writers)
and Shakespeare (about 1,100 roles, LSTM) for many ε values, five seeds, five stacks, with CKKS, will not
finish on a CPU in this timeline. Size it explicitly (Section 5, Phase 5).

---

## 4. Target design of the library

### 4.1 Principles

1. **One dial, honestly defined.** The user sets a total $(\varepsilon, \delta)$ and a unit of privacy;
   the library calibrates every mechanism to that total over the whole run and refuses to run if it
   cannot.
2. **Every release is on the ledger.** Training noise, mask selection, any DP preprocessing, adaptive
   clipping, and hyperparameter tuning on private data (see Papernot and Steinke, "Hyperparameter tuning
   with Renyi DP", ICLR 2022) all spend from one `PrivacyLedger`. Anything not accounted is either
   refused or flagged `unsafe` in the report.
3. **Explicit threat model per configuration.** Each run's report states who is protected from whom:
   final-model observers, the aggregation server, other clients, eavesdroppers; and under which
   assumptions (honest-but-curious server, maximum colluding clients, dropouts).
4. **Pluggable backends.** Crypto backends behind an interface (TenSEAL default, OpenFHE optional,
   pairwise-masking secure aggregation built in). Accounting backends behind an interface (`dp-accounting`
   PLD and RDP, Opacus RDP and PRV). Swapping a backend never changes the public API.
5. **Safe by default, loud when not.** Unsafe combinations (non-private oversampling plus DP, clean SPE
   masks, a server-held secret key) require `allow_unsafe=True` and appear in red in the report.
6. **Free, portable, verifiable.** Pure Python plus wheels that install on Linux, macOS and Windows on CPU;
   no telemetry; every claim backed by a test or an audit.

### 4.2 Package layout (src layout)

```
src/obscuradp/
  __init__.py            public API re-exports, __version__ from package metadata
  policy.py              PrivacyPolicy(epsilon, delta, unit="record"|"client", accountant=...)
  ledger.py              PrivacyLedger: composes every mechanism, emits the privacy report
  report.py              PrivacyReport: JSON schema + human-readable summary + threat model
  accounting/            adapters over dp-accounting (PLD, RDP) and Opacus (RDP, PRV); sigma calibration to a TOTAL budget
  mechanisms/            gaussian, discrete_gaussian, skellam; clipping (fixed, adaptive)
  client/                LocalTrainer: plain or DP-SGD (Opacus, Poisson sampling), FedProx term
  aggregation/           fedavg (weighted/unweighted), secure_sum interface
  crypto/                backends: tenseal_ckks, openfhe_ckks (optional), pairwise_mask_secagg; key management
  selection/             SPE masks: data_independent, public_data, dp (ledger-charged)
  compression/           quantization for modular secure aggregation; byte accounting
  imbalance/             DP-safe options: class-weighted loss, DP reweighting; non-private SMOTE only behind allow_unsafe
  audit/                 membership (LiRA-lite, one-run canary audit), gradient_inversion (DLG/iDLG), attribute_inference
  sim/                   single-process Simulation (successor of Trial 07/27's orchestrator)
  datasets/              downloaders with checksums: stroke, loan, FEMNIST/Shakespeare via flwr-datasets, synthetic health
  adapters/flower/       optional extra: Strategy wrapper + ClientApp mods mapping the same Plan onto Flower
  bench/                 overhead harness and padding model (from Trial 07/27 bench.py)
tests/                   unit, property-based (hypothesis), golden regression, slow/integration markers
benchmarks/              pre-registered benchmark configs + runner + results/ (not shipped in the wheel)
docs/                    mkdocs-material site: quickstart, concepts, threat model, choosing epsilon, API reference
research/                (later) the original experiments, unchanged, linked from docs
```

### 4.3 Public API sketch (aim for this shape, refine with tests)

```python
import obscuradp as odp

policy = odp.PrivacyPolicy(epsilon=1.0, delta=1e-5, unit="record")      # total budget for the whole run
plan = odp.Plan(
    dp=odp.dp.DPSGD(clip=1.0),                                           # or odp.dp.ClientLevel(...), odp.dp.Distributed(mechanism="skellam")
    secure_aggregation=odp.crypto.PairwiseMasking(bits=16),              # or odp.crypto.CKKS(backend="tenseal", key_holder="clients")
    selection=odp.selection.SPE(fraction=0.25, source="public"),         # optional
    imbalance=odp.imbalance.ClassWeighted(),
)
sim = odp.Simulation(model_fn=make_model, clients=client_datasets, plan=plan, policy=policy, rounds=20)
result = sim.run(seed=42)

print(result.report.summary())            # total epsilon spent, unit, mechanisms, threat model, unsafe flags
result.report.to_json("privacy_report.json")
audit = odp.audit.membership(result, attack="one_run", n_canaries=1000)
print(audit.advantage, audit.theoretical_ceiling)
```

The Flower adapter must accept the same `Plan` and `PrivacyPolicy` objects so moving from simulation to a
real deployment is a configuration change, not a rewrite.

---

## 5. Phased plan (about 12 weeks, CPU only)

Track progress in `docs/ROADMAP_STATUS.md` with this checklist. Each phase lists its exit criteria.

### Phase 0 (week 1): Audit and freeze. **GATE**

- Tag the current state as `research-baseline-v0` (annotated tag; push only if the session allows tags,
  otherwise note the commit hash in the status file).
- Write `docs/AUDIT.md`: for each of G1 to G11, verdict (confirmed, partly confirmed, refuted), evidence
  with `path:line`, severity, and the proposed fix. Add anything else you find (threats, silent bugs,
  missing seeds, nondeterminism).
- Write `docs/THREAT_MODEL.md` draft: actors, what each stack protects against, assumptions.
- Write `docs/KPI_MAPPING.md` (see Section 6) with a proposed precise definition for every KPI.
- Write the website mismatch list (G8, G9 and anything else) as a table: current copy, problem, proposed
  copy. **Do not edit the site yet.**
- Exit: PR with the four documents, a decision memo listing the choices the owner must confirm (license,
  default secure-aggregation backend, default unit of privacy, how to handle SMOTE). Stop.

### Phase 1 (weeks 1 to 3): Package skeleton with zero behaviour change

- `pyproject.toml` (hatchling or setuptools, PEP 621 metadata, `requires-python >= 3.11`), extras:
  `[he]` (tenseal), `[openfhe]`, `[flower]`, `[audit]`, `[bench]`, `[docs]`, `[dev]`. Torch from the CPU
  index in CI.
- `LICENSE` (Apache-2.0 unless the owner chose otherwise at the Phase 0 gate), `CITATION.cff`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` (how to report a privacy bug),
  `CHANGELOG.md` (Keep a Changelog), version `0.1.0.dev0` from package metadata.
- Tooling: ruff (lint and format), mypy or pyright on `src/`, pytest with markers `slow` and
  `integration`, hypothesis, coverage, pre-commit. Fix the root `.gitignore` (scope `lib/` and `build/`
  rules so they cannot swallow source folders).
- GitHub Actions: lint, type-check, unit tests on Python 3.11 to 3.13 on Linux, plus macOS and Windows
  for one version; a nightly job for `slow` tests; the website build and its `lint:copy` and
  `check:contrast` scripts on changes under `website/`.
- Port `Trial 07/27/obscura/` into `src/obscuradp/` with the layout in 4.2, replacing module constants
  with typed config objects (dataclasses or pydantic). Keep behaviour identical.
- **Golden regression tests:** re-run a reduced Trial 07/27 configuration (few rounds, one seed) through
  the new package and through the old code, and require identical metrics (bitwise where the old code is
  deterministic, documented tolerance where CKKS randomness enters). Pin the old results by sha256 as
  `verify_claims.py` does.
- Leave the `Experiment */` and `Trial 07/` folders in place (the website links to them). The package
  must never import from them.
- Exit: `pip install -e ".[dev,he]"` then `pytest` green in CI; golden tests pass; coverage of ported core
  modules at least 80%.

### Phase 2 (weeks 3 to 5): Privacy correctness. **GATE**

- `PrivacyPolicy`, `PrivacyLedger`, `PrivacyReport` with a versioned JSON schema.
- Calibrate $\sigma$ to the **total** budget over all rounds and local epochs, with Poisson sampling and
  the same sample rate in calibration and training (fixes G2). Cross-check two accountants (PLD and RDP)
  and fail if they disagree beyond a documented tolerance.
- Unit of privacy: record-level (DP-SGD in each silo) and client-level (clip and noise whole updates);
  the report states which.
- Fix G3: remove non-private SMOTE from DP pipelines; provide class-weighted loss and at least one
  DP-safe rebalancing option; non-private SMOTE only behind `allow_unsafe=True`.
- Fix G4: SPE masks from data-independent rules, public data, or DP selection charged to the ledger;
  report what the clear fraction exposes.
- Fix G5: explicit key management; a server-side aggregator object that cannot decrypt; decryption by a
  client-held key or a committee; document what happens with dropouts.
- Property tests: $\sigma$ decreases as $\varepsilon$ grows; total spend never exceeds the policy; ledger
  spend equals the sum of mechanism spends under composition; refusing unsafe configs.
- Re-run Trial 07/27's arms and frontier (stroke and loan) with corrected accounting and without
  non-private SMOTE. Publish old vs new numbers side by side in `benchmarks/results/` with an explanation.
- Exit and gate memo: corrected results, what changed and why, and any finding that reverses an earlier
  claim (for example "DP beats non-private on stroke").

### Phase 3 (weeks 5 to 7): Better results through better composition

- Distributed DP over secure aggregation: Skellam (or Distributed Discrete Gaussian) mechanism with
  quantization, modular clipping and a pairwise-masking secure-sum backend. Compare at **equal total ε**
  against local DP-SGD plus CKKS and against central DP (trusted server) as the upper bound.
- Utility levers, each one ledger-accounted: adaptive clipping, larger effective batch via more local
  steps or client sampling, server-side momentum, fewer but better-tuned rounds, public-data warm start
  where a public proxy exists. Tune hyperparameters on public or synthetic proxies, or charge the tuning
  to the ledger.
- Communication (KPI 4): quantized updates (16, 12, 8 bits), measured bytes per client per round
  versus an uncompressed float32 FedAvg baseline; CKKS improvements (ring dimension 4096 where precision
  allows, symmetric or seeded serialization if the backend supports it, quantized clear part under SPE).
- Exit: a results table with utility, total ε, bytes per round and wall-clock for every configuration;
  a recommendation for the default `Plan`.

### Phase 4 (weeks 6 to 8): Auditing (KPI 2)

- `obscuradp.audit.membership`: one-run canary audit (Steinke et al. 2023) as the CPU-friendly default,
  plus a small LiRA-style attack with a capped number of shadow models for small tabular models.
- `obscuradp.audit.gradient_inversion`: DLG and iDLG against what the server actually sees under each
  stack (plain, SPE clear part, masked sum), on a small model.
- `obscuradp.audit.attribute_inference`: a baseline attack on the tabular datasets.
- Every audit reports the measured advantage with a confidence interval next to the theoretical ceiling
  $\frac{e^{\varepsilon} - 1 + 2\delta}{e^{\varepsilon} + 1}$.
- Exit: audit results at total $\varepsilon \in \{0.5, 1, 2, 4, 8\}$ for the default `Plan` on stroke and
  loan, five seeds, with the KPI 2 status stated honestly.

### Phase 5 (weeks 7 to 10): Flower adapter and the pre-registered benchmark

- `obscuradp[flower]`: a Strategy and ClientApp mods that map `Plan` and `PrivacyPolicy` onto Flower's
  APIs (use Flower's own SecAgg+ where it matches our secure-sum interface). Integration test that runs a
  small federation with the Flower simulation engine on CPU and checks the privacy report equals the
  `Simulation` report for the same config.
- Write `benchmarks/PREREGISTRATION.md` **before** running anything: hypotheses, datasets, partitions,
  ε grid (total budget), seeds, metrics, stopping rules, and the CPU budget per run. Link it from the
  website once the owner publishes it.
- CPU-sized benchmark (adjust after a timing pilot, and record the pilot):
  - FEMNIST via `flwr-datasets` natural writer partition, subsample about 100 to 200 writers, a small CNN
    (under 100k parameters), client sampling of about 10 per round.
  - Shakespeare via natural role partition, subsample about 100 roles, a small GRU or LSTM with short
    sequences; drop it to a later release if the timing pilot shows it cannot finish.
  - Synthetic health data with a known generator (documented and seeded), non-IID by site, including a
    rare-outcome task that replaces the stroke rebalancing study without SMOTE.
  - Stacks: plain, central DP, local DP-SGD, distributed DP + secure aggregation, and the full stack with
    SPE. Total $\varepsilon \in \{0.5, 1, 2, 4, 8, \infty\}$, five seeds.
- Exit: the ε versus accuracy curves for the full stack under non-IID data, every number generated by
  committed code, every run's privacy report stored next to its metrics.

### Phase 6 (weeks 10 to 12): Documentation, release, website sync. **GATE before publishing**

- Docs (mkdocs-material, API reference via mkdocstrings): 5-minute CPU quickstart, "Choosing ε" in plain
  language, threat model, privacy report explained, recipes for clinics (cross-silo, record-level) and
  apps (cross-device, client-level), low-bandwidth settings, FAQ including what the library does **not**
  protect against. Every code snippet in the docs is executed in CI.
- Release: build sdist and wheel, `twine check`, publish to **TestPyPI** first, then PyPI with Trusted
  Publishing (OIDC from GitHub Actions, no API tokens), signed attestations, a GitHub release with notes.
  Version `0.1.0`. Do not publish to PyPI without the owner's explicit go-ahead at the gate.
- Website (propose, then edit): apply the Phase 0 mismatch table, add Trial 07/27 and the new benchmark
  to `/research` through `npm run sync-data`, reword KPI 2 as an audit target with the theoretical
  ceiling, mark every ε as total and state the unit, add install instructions and a link to the docs,
  and keep `npm run build`, `lint:copy`, `check:contrast` and `test:e2e` green.
- Optional: a JOSS paper draft (`paper/paper.md`) once the package has docs and tests.
- Exit: `pip install obscuradp` works on a clean CPU machine on Linux, macOS and Windows and the
  quickstart runs in under 5 minutes.

---

## 6. KPI mapping (fill in `docs/KPI_MAPPING.md` in Phase 0, keep it current)

| Website KPI | Precise definition to adopt | How it is measured | Where it is reported | Honest status today |
|---|---|---|---|---|
| 1M records protected | Records in training runs that used `obscuradp` with a privacy report, as reported by partners | Partner case studies; no telemetry | Website Impact, docs "Adopters" page | Not measurable yet |
| $\le 5\%$ attacker advantage at $\varepsilon = 1$ | Measured advantage of the named audit attack at **total** $\varepsilon = 1$, upper end of a 95% confidence interval, for the default `Plan` | `obscuradp.audit`, Phase 4 | Benchmark results, website Impact with the theoretical ceiling next to it | Not measured; theoretical ceiling is about 46% |
| 25 to 50 organizations | Organizations with a signed pilot or a public statement of use | Partner list, GitHub dependents, PyPI downloads as a secondary signal | Website, docs | Zero |
| $\ge 50\%$ fewer bytes per round | Mean bytes uploaded per client per round vs float32 FedAvg on the same model, at the default `Plan` | `obscuradp.bench`, Phase 3 | Benchmark results | Current CKKS path is about $14\times$ **more** |
| Reproducible full-stack benchmark | Pre-registered, five seeds, all numbers regenerated by CI-checked code | Phase 5 | `/research` | Planned |
| Installable library | `pip install obscuradp` + quickstart under 5 minutes on CPU | Release CI on three OSes | README, docs | Not started |

Engineering KPIs to add and track: CI green on `main`; core coverage at least 85%; zero unsafe flags in
the default `Plan`; accountant cross-check within tolerance; docs snippets executed in CI; median install
time; golden regression tests unchanged unless a PR explains why.

---

## 7. Definition of done for v0.1.0

- `pip install obscuradp` (and `obscuradp[he]`, `obscuradp[flower]`) works on CPU on three OSes.
- One `PrivacyPolicy` sets a **total** $(\varepsilon, \delta)$; every run emits a `PrivacyReport` with
  the unit of privacy, mechanisms, accountant, threat model and unsafe flags.
- No unaccounted release anywhere in the default configuration (G2 to G5 closed, with tests).
- Distributed DP over secure aggregation available and benchmarked against local DP + CKKS.
- Membership, gradient-inversion and attribute-inference audits shipped and run on the benchmark.
- The pre-registered CPU benchmark published with code, configs, results and privacy reports.
- Docs site live; website claims match the library and results; Trial 07/27 on `/research`.
- Apache-2.0 (or the owner's choice) license, CITATION, SECURITY policy, CHANGELOG, tagged release.

---

## 8. Starting this session

1. Read `docs/ROADMAP_STATUS.md`. If missing, create it with the Section 5 checklist and start Phase 0.
2. Re-check the dependency versions in Section 2 (PyPI JSON API) and note any change in the status file.
3. Do the next unfinished phase, open or update its draft PR, update the status file, and stop at the end
   of the phase (or at a GATE).
4. Finish with a short plain-language summary for the owner: what changed, what was verified and how, and
   what decision (if any) is needed from them.
