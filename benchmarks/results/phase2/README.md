# Phase 2 results

The corrected re-runs of the research experiments (step S11 of `docs/ROADMAP.md`). The findings are written
up in `docs/GATE_PHASE2.md`.

| Path | What it is |
|---|---|
| `runs/` | One record per run (200 files). Each holds the settings and their fingerprint, the code version, the machine, the seed, the scores, the measured upload size and, for protected runs, the full privacy report with every recorded release. |
| `summary.json` | Means, sample standard deviations and 95% intervals over the five seeds for every setting, plus the earlier research numbers they are compared with. |
| `summary.md` | The same as tables. |

Nothing here is edited by hand. To regenerate, from the repository root:

```bash
python -m benchmarks.phase2.tune
```

```bash
python -m benchmarks.phase2.run
```

```bash
python -m benchmarks.phase2.report
```

The first command chooses the learning rate and clipping norm on made-up tables and writes
`benchmarks/phase2/settings.json`. The second runs the grid on the real tables and skips runs whose record
already exists. The third checks every record, recomputes every stored privacy report from its recorded
releases, and rebuilds the two summary files; it exits with status 1 if any check fails.

To check a single privacy report yourself, take the `privacy_report` entry of any run record, save it as a
file, and run `obscuradp report verify` on it.

The dataset files are read from the frozen research folder `Trial 07/27/data/`. They are not part of the
installed package, and neither is this folder.
