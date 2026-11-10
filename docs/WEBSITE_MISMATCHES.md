# Website wording that disagrees with the math or the code (Phase 0)

Baseline: `main` at `34071be`. Line numbers and "current wording" below describe that commit.

**Status (2026-11-10):** items 1 to 12 were applied in aaydinks/ObscuraDP#9 (merged as `37b924c`), together
with the `LICENSE` file. The final wording differs slightly from the proposals below where the copy lint or
the layout required it; the pull request lists old and new wording for each item. Items 13 to 28 wait for
session S22, once the library and audits exist.

Proposed wording follows the site's own rules: no em dashes, everyday words, and it must pass
`npm run lint:copy`.

## Fixed in aaydinks/ObscuraDP#9 (the wording was false or contradicted another page)

| # | Where | Current wording | Problem | Proposed wording |
|---|---|---|---|---|
| 1 | `website/src/sections/Impact.astro:10` | "An attacker trying to tell whether your record was used gets it right at most 52.5% of the time, versus 50% for a coin flip." | Worded as a guarantee. At this privacy setting the provable limit is about 73%, which the privacy dial on the same site shows. | "Our goal: in our own attack tests, an attacker trying to tell whether your record was used should be right no more than 52.5% of the time, against 50% for a coin flip. The proven worst case at this setting is 73%. We have not run these tests yet." |
| 2 | `website/src/components/visuals/PrivacySlider.astro:95` | "This is the mathematical ceiling. Real attacks measured in audits land far lower; our target is at most 5% attacker advantage at ε = 1." | No audit has been run, so "measured in audits land far lower" has no backing. | "This is the proven worst case. Our target, to be measured in attack tests, is at most 5% attacker advantage at ε = 1." |
| 3 | `website/src/sections/Framework.astro:109` | "The server adds them together while they stay encrypted and never sees the unencrypted values." | In the current code the same program that adds also holds the key. With partial encryption, most of each update is sent unencrypted. | "The server adds them together while they stay encrypted. The key that unlocks the total is kept away from the server. This separation is designed and is being built." |
| 4 | `website/src/sections/Framework.astro:84` | "We never receive raw data, so it can't be stolen from us or intercepted on the way." | "We" suggests Obscura DP runs a service. It is a library that others run. | "Whoever runs the training never receives raw data, so it can't be stolen from them or intercepted on the way." |
| 5 | `website/src/sections/Framework.astro:115` | "the data can't be read in transit, isn't sitting on our servers to steal, ..." | Same "our servers" issue. | "the updates can't be read in transit, the raw data never sits on a central server, ..." |
| 6 | `website/src/components/visuals/Pipeline.astro:7` | "Adds encrypted updates together" | Never says who can unlock the total. | "Adds encrypted updates together. Cannot unlock them." plus a sixth step or note: "Only the combined total is ever unlocked." |
| 7 | `website/src/sections/Hero.astro:9`, `website/src/lib/site.ts:8`, `website/scripts/og/template.html:95` | "Open-source privacy framework for AI" | There is no license file, so nobody has the legal right to reuse the code yet. | Add the `LICENSE` file (Apache-2.0 unless the owner chooses otherwise). Wording can then stay. |
| 8 | `website/src/sections/Framework.astro:11` | Comparison row marks Obscura DP as open source | Same as 7. | Resolved by adding the license. |
| 9 | `website/src/sections/Roadmap.astro:12`, `website/src/sections/Research.astro:26` | "Pre-registered research design." / "Our research design is pre-registered." | The link is empty (`website/src/lib/site.ts:12`). A plan nobody can read is not pre-registered in the usual sense. | Until a link exists: "Research plan written before running the benchmark (to be published)." Move the roadmap item from "Done" to "In progress". |
| 10 | `website/src/sections/Research.astro:100`, `website/src/components/charts/Exp1RecallCurve.astro:41` | "(recall 0.0 at ε = 9, all five seeds)" | In Experiment 1 this number is a budget for each round. Over the 20 rounds the total is several times larger. The detail page says so; the home page and chart do not. | "(recall 0.0 at a per-round ε of 9, all five seeds)" |
| 11 | `website/src/components/visuals/HeroDemo.astro:100` | "Same answer either way, so the output can't reveal whether you were included." | The two outputs are not the same; they are random draws that look alike. | "Nearly the same answer either way, so the output gives almost nothing away about whether you were included." |
| 12 | `website/wrangler.jsonc:12` with `website/README.md:125` | Request logs are enabled; README says "no analytics" | The two statements conflict. | Either set observability to off, or add to the README: "Cloudflare keeps short-lived request logs for debugging." Owner's choice. |

## Fix at release (needs the library, audits or new results first)

| # | Where | Current wording | Problem | Proposed change |
|---|---|---|---|---|
| 13 | `website/src/sections/Impact.astro:8` | "attacker advantage at ε = 1.0." | Does not say whether the setting covers one round or the whole training run, or what counts as one person. | "attacker advantage at ε = 1.0 for the whole training run, measured per record." Then show the measured value once S15 has it. |
| 14 | `website/src/sections/Impact.astro:14` | "fewer bytes per training round compared with an uncompressed federated learning baseline." | Not precise enough to check, and the current encryption path is about 14 times larger, not smaller. | "fewer upload bytes per participant per round than sending the same model as 32-bit numbers." Show the measured ratio per backend once S14 has it. |
| 15 | `website/src/sections/Impact.astro:5`, `website/src/sections/Impact.astro:13` | "1M records protected", "25–50 organizations" | No way to count either, and the project promises no telemetry. | Add a line: "Counted from what partners choose to report. The software sends nothing back." |
| 16 | `website/src/sections/Framework.astro:66-67` | "Hospital records can run below ε = 1 ... With personalized ε, the most sensitive users get ε < 1 ..." | No experiment has a total budget below 1, and per-person settings do not exist yet. | After S7: keep, and link to the documentation page that shows how each site sets its own budget. Before then: "Planned: each site can choose its own setting." |
| 17 | `website/src/sections/Framework.astro:41` | "any two datasets that differ by one person." | The code protects one record, not one person. They differ when a person has several rows. | "any two datasets that differ by one record (for example one patient visit)." |
| 18 | `website/src/sections/Hero.astro:10`, `website/src/lib/site.ts:6`, `website/src/components/Footer.astro:13`, `website/scripts/og/template.html:91` | "Mathematically guaranteed privacy" | True of the underlying method. Not yet true of the code, whose budgets are miscounted. | Keep after S11 confirms correct accounting. Until then the status line under the hero should say "Research stage: the guarantees below describe the design, not a released product." |
| 19 | `website/src/sections/Mission.astro:40` | "The guarantee is a mathematical bound you can check, not a promise." | Nothing lets a user check it today. | Keep once `obscuradp report verify` exists (S7), and link to it. |
| 20 | `website/src/sections/Roadmap.astro:21` | "DP-SGD integrated with federated learning, and encryption implemented alongside differential privacy." | That work is Trial 07/27, which the site never shows. | Add Trial 07/27 to `/research` with its corrections (S22) and extend `website/scripts/sync-data.mjs`. |
| 21 | `website/src/sections/Roadmap.astro:55` | "Target: low attacker advantage at ε ≤ 1." | Says "at most 1" where the Impact section says "equal to 1". | Use one wording in both places. |
| 22 | `website/src/sections/Research.astro:52` and about 40 other literals | Numbers typed in by hand | All correct today (checked by script), but nothing keeps them in step with the data. | Build-time check against a claims file (S22). |
| 23 | `website/src/pages/research.astro:91` | "The reported std is the population standard deviation." | With five runs this understates the spread by about 11%. | Report the sample standard deviation and 95% intervals from S20 on; keep old tables labelled as they are. |
| 24 | `website/src/pages/research.astro:135` | "only the mean is decrypted" | Does not say by whom. In the code it is the same process. | "only the mean is decrypted. In this replication one program played every role; the library separates them." |
| 25 | `website/src/sections/Contact.astro:13` | "Add provable privacy to your training pipeline. Watch the repository for the first release." | Fine as a promise; needs the install line at release. | At release: "Install with `pip install obscuradp`." |
| 27 | Experiment 1 tab on the home page; the opening paragraph of `/research` | "FedAvgon average", "theGitHub repository" | A space is missing in the rendered page. Astro drops the space before an expression that starts on the next line. Found while checking screenshots for the fixes above. | Add an explicit space (`{' '}`) in both places. |
| 28 | Roadmap "Workstreams" table at phone width | The table scrolls sideways inside its own box | Not wrong, but easy to miss on a phone. The page itself does not scroll sideways. | Consider a stacked layout for narrow screens. |
| 26 | `website/src/lib/laplace.ts:16` | Ceiling formula ignores $\delta$ | The site states $\delta = 10^{-5}$ for experiments (`website/src/sections/Framework.astro:56`). The effect on the shown percentage is far below rounding. | Leave the number; add a code comment that the tiny extra term is omitted. |

## Checks to run after any edit

```bash
cd website && npm run build && npm run lint:copy && npm run check:contrast
```

```bash
cd website && npm run preview
```

Then, in a second terminal, against the preview address that `wrangler dev` prints:

```bash
cd website && npm run test:e2e
```

If the headline changes, re-render the share image with `npm run og`.
