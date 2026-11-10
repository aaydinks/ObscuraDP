# ObscuraDP: from research repository to installable library

> **How to use this file.** Paste everything below the line into a fresh Claude Code session opened on
> the `aaydinks/ObscuraDP` repository. The prompt is written to be re-entered many times: each session
> reads the progress tracker, finishes the next unfinished phase, updates the tracker, and stops. Do not
> edit the prompt between sessions unless the plan itself changes; record progress in
> `docs/ROADMAP_STATUS.md` instead.
>
> **Revision 2 (2026-11-10, after Phase 0).** The audit confirmed G1 to G11, added G12 to G27 and corrected
> five points of the first version. The detailed session plan now lives in `docs/ROADMAP.md`, the evidence in
> `docs/AUDIT.md`, and the target definitions in `docs/KPI_MAPPING.md`. Sections 3, 5 and 6 below are
> summaries that point there.

---

## 0. Your role and how to work

You are the lead engineer turning ObscuraDP from a collection of paper replications into a real,
installable, documented Python library named **`obscuradp`** that combines differential privacy (DP),
federated learning (FL) and homomorphic encryption (HE) behind one coherent API, with an honest,
machine-checked privacy report for every training run.

Work rules for every session:

1. **Start by reading `docs/ROADMAP_STATUS.md`, then `docs/ROADMAP.md`.** Do only the **next unfinished
   session** in the status checklist. Do not jump ahead. If a gate is still open (the owner has not answered
   the "Decisions needed" list), stop and ask instead of starting the next session.
2. **One session, one branch, one draft PR.** Branch name `roadmap/sNN-<slug>` unless the session
   assigns a branch, in which case use that one. End each session by updating `docs/ROADMAP_STATUS.md`
   (what was done, what was verified and how, what is left, open questions) in the same PR.
3. **Gates.** Sessions marked **GATE** end with a short decision memo in the PR description and stop for
   human review. Do not start the next session in the same sitting.
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
| Session plan | `docs/ROADMAP.md` (approved 2026-11-10) | 22 sessions, cut order and never-cut list are fixed there. |
| License | Apache-2.0 (confirmed 2026-11-10) | Patent grant matters for a privacy and crypto library. |
| Secure sum | Pairwise masking over integers by default; exact BFV optional; CKKS `research_only` | `docs/adr/0003-secure-sum-default.md`. |
| Privacy unit | One record, one total budget per run, PLD accountant with RDP cross-check | `docs/adr/0002-privacy-unit-and-budget.md`. |
| Imbalance | SMOTE refused inside private training unless forced; class weighting instead | `docs/adr/0004-imbalance-without-smote.md`. |
| Datasets | The two CSV files stay in the repository for now; the package never ships data | Add source and license notes if they can be found (S17). |
| Targets | Attacker advantage is a measured target beside two ceilings; bytes are counted against 32-bit plain messages | `docs/KPI_MAPPING.md`. |
| Release label | "Prerun + novel research action" | Owner's wording for v0.1. |

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
partitions via `NaturalIdPartitioner`), `dp-accounting 0.6.0` (Google: RDP and PLD accountants; pins
`attrs<24`), `torch 2.15.0`. In the cloud environment the CPU-only torch download host is blocked, so a
plain `pip install torch` pulls several gigabytes of GPU libraries; see the status file.

---

## 3. Audit findings (summary; evidence in `docs/AUDIT.md`)

Phase 0 checked every finding against the code at `main` `34071be`. All of G1 to G11 are confirmed and 16
more were found. Read `docs/AUDIT.md` before touching the code it describes.

| ID | Finding | Fixed in |
|---|---|---|
| G1 | There is no library yet (no packaging, license, Python tests; unanchored `.gitignore` rules) | S4 to S6 |
| G2 | The privacy budget is per round, and accounted steps differ from executed steps | S7, S8 |
| G3 | SMOTETomek before DP-SGD voids the per-record guarantee | S9 |
| G4 | The SPE mask comes from one client's raw data and is not accounted | S10 |
| G5 | One object holds the secret key and encrypts, adds and decrypts | S10 |
| G6 | DP and encryption are combined in the most expensive way (**design corrected below**) | S12 to S14 |
| G7 | The bytes target contradicts the measured encryption cost | S10, S14 |
| G8 | The attacker-advantage target is worded as a guarantee | S3, S15 |
| G9 | Other website and README mismatches (`docs/WEBSITE_MISMATCHES.md`) | S3, S22 |
| G10 | Datasets vendored twice with no license or source link | gate, S17 |
| G11 | The planned benchmark does not fit a CPU budget as first written | S17, S18 |
| G12 | Preprocessing uses pooled data | S9 |
| G13 | No validation split; settings were chosen on test results | S9, S11 |
| G14 | The "DP" mask uses about 8 times too little noise | S10 |
| G15 | Decrypted CKKS values are published every round (key-recovery exposure) | S10 |
| G16 | Noise quality: ordinary random generator, floating-point noise | S8, S13 |
| G17 | Seed collisions and non-reproducible encrypted runs | S6 |
| G18 | Silent failures in the training loop and drivers | S6 |
| G19 | The verification harness prints fingerprints but never checks them | S6 |
| G20 | Experiment 3: overhead benchmark ignores the mask; buffers are not combined | S5 |
| G21 | Website promises with no counterpart in code | S3, S7 |
| G22 | Population standard deviation over 5 seeds, no intervals | S17, S20 |
| G23 | About 40 numbers typed into website copy by hand | S22 |
| G24 | Request logging is on while the README promises no analytics | S3 |
| G25 | The stroke test set is 84 rows | S11 |
| G26 | Committed results carry no provenance | S6 |
| G27 | Empty sampled batches are skipped; dependency pins are behind | S8 |

**Corrections to the first version of this prompt (binding):**

1. **Noise shares (G6).** "Each client adds a share of the noise" is only sound when every client sends
   **one** clipped gradient sum per round at the shared model. With several local steps it is not. Ship two
   per-record modes: `LocalDPSGD` (several local steps, full noise) and `DistributedDPSGD` (one step per
   round through a secure sum, share $\sigma C / \sqrt{N - t - d}$ for $t$ colluders and $d$ dropouts).
2. **Accountant cross-check.** PLD and RDP always differ because RDP is looser. Test ordering and a band,
   not equality.
3. **Default backend.** Pairwise masking over integers is the default secure sum, with value width
   $b \le 14 - \lceil \log_2 N \rceil$ bits. Exact BFV is the encryption option. CKKS is `research_only`
   behind an unsafe flag.
4. **FEMNIST.** Keep the full writer population and sample about 200 per round. Do not subsample writers.
5. **Audit sample size.** About 30 short runs per budget, not 5 seeds.

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
4. **Pluggable backends.** Secure-sum backends behind an interface: pairwise masking over integers
   (default, no TenSEAL needed), exact BFV through TenSEAL (optional), CKKS (`research_only`), OpenFHE
   (later). Accounting backends behind an interface (`dp-accounting` PLD sets the noise, RDP cross-checks).
   Swapping a backend never changes the public API.
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
  mechanisms/            dpsgd (own step, exact Poisson sampling), sampling, noise (gaussian, skellam), clipping
  client/                LocalTrainer: plain or private, FedProx term
  aggregation/           fedavg (weighted/unweighted), secure_sum interface
  crypto/                pairwise_masking (default), tenseal_bfv, tenseal_ckks (research_only); roles: KeyHolder, ClientEncryptor, Aggregator
  selection/             SPE masks: data_independent, public_data, dp (ledger-charged)
  compression/           quantization for modular secure aggregation; byte accounting
  imbalance/             DP-safe options: class-weighted loss, DP reweighting; non-private SMOTE only behind allow_unsafe
  audit/                 membership (LiRA-lite, one-run canary audit), gradient_inversion (DLG/iDLG), attribute_inference
  sim/                   Simulation (new loop) and legacy.py (Trial 07/27's loop, verbatim, for comparison tests)
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

policy = odp.PrivacyPolicy(epsilon=1.0, delta=1e-5, unit="record")  # total budget for the whole run
plan = odp.Plan(
    dp=odp.dp.LocalDPSGD(clip=1.0),  # or DistributedDPSGD(max_colluding=1), ClientLevelDP(...)
    # or odp.crypto.BFV(key_holder="committee")
    secure_sum=odp.crypto.PairwiseMasking(value_bits=8),
    selection=None,  # SPE only with an encryption backend, local mode
    imbalance=odp.imbalance.ClassWeighted(),
)
sim = odp.Simulation(
    model_fn=make_model, clients=client_datasets, plan=plan, policy=policy, rounds=20
)
result = sim.run(seed=42)

print(result.report.summary())  # total epsilon spent, unit, mechanisms, threat model, unsafe flags
result.report.to_json("privacy_report.json")
audit = odp.audit.membership(result, attack="one_run", n_canaries=1000)
print(audit.advantage, audit.theoretical_ceiling)
```

The Flower adapter must accept the same `Plan` and `PrivacyPolicy` objects so moving from simulation to a
real deployment is a configuration change, not a rewrite.

---

## 5. Session plan (summary; details and exit checks in `docs/ROADMAP.md`)

| Phase | Sessions | Content |
|---|---|---|
| 0 | S1 (S2 spare) | Audit, baseline reproduction, threat model, target definitions, website wording list. **GATE** |
| 0.5 | S3 | `LICENSE`, and the website wording that is false today. Only if approved at the gate |
| 1 | S4 to S6 | Packaging and CI; comparison harness against the old code; port with zero behaviour change |
| 2 | S7 to S11 | Total-budget ledger and report; own DP-SGD step; pipeline hygiene; secure sum and key roles; corrected re-runs. **GATE** |
| 3 | S12 to S14 | Distributed and central modes; integer noise path; equal-budget comparison and measured bytes |
| 4 | S15, S16 | Membership audit with three reference numbers; shadow models, attribute inference, gradient inversion |
| 5 | S17 to S20 | Datasets, written research plan tagged before runs, benchmark, thin Flower adapter, analysis |
| 6 | S21, S22 | Documentation and partner kit; release engineering and website sync. **GATE** before publishing |
| 7 | later | Adoption, upkeep, v0.2 scope |

---

## 6. Targets (summary; definitions and reference tables in `docs/KPI_MAPPING.md`)

- **1M records, 25 to 50 organizations:** counted only from what partners report. The software sends nothing
  back.
- **At most 5% attacker advantage at a budget of 1:** a measured target for a named attack, over about 30
  runs, always shown beside the general ceiling (46.2%) and the ceiling for the actual noise method (about
  10.6%).
- **At least 50% fewer bytes per round:** upload bytes per participant per round against the same model sent
  as 32-bit numbers. Reachable with masked 8 to 12 bit values; not reachable with CKKS.
- **Benchmark and library:** as in `docs/KPI_MAPPING.md`.

---

## 7. Definition of done for v0.1.0

- `pip install obscuradp` (and `obscuradp[he]`, `obscuradp[flower]`) works on CPU on three OSes.
- One `PrivacyPolicy` sets a **total** $(\varepsilon, \delta)$; every run emits a `PrivacyReport` with
  the unit of privacy, mechanisms, accountant, threat model and unsafe flags.
- No unaccounted release anywhere in the default configuration (G2 to G5 closed, with tests).
- Local and distributed per-record modes available and compared at equal total budget; pairwise masking is
  the default secure sum; CKKS is flagged `research_only`.
- Membership, gradient-inversion and attribute-inference audits shipped and run on the benchmark.
- The pre-registered CPU benchmark published with code, configs, results and privacy reports.
- Docs site live; website claims match the library and results; Trial 07/27 on `/research`.
- Apache-2.0 license, CITATION, SECURITY policy, CHANGELOG, tagged release, labelled
  "Prerun + novel research action" (the owner's wording), with a plain statement that there has been no
  independent audit.

---

## 8. Starting this session

1. Read `docs/ROADMAP_STATUS.md` and `docs/ROADMAP.md`. If an open gate is listed, ask the owner for the
   pending decisions and stop.
2. Re-check the dependency versions in Section 2 (PyPI JSON API) and note any change in the status file.
3. Do the next unfinished session, open or update its draft PR, update the status file, and stop at the end
   of the session (or at a GATE).
4. Finish with a short plain-language summary for the owner: what changed, what was verified and how, and
   what decision (if any) is needed from them.
