# Obscura DP

Obscura DP is a research project that is becoming a Python library, `obscuradp`. Its goal: let several
organizations train a shared machine-learning model on sensitive data without handing that data to anyone,
with a stated, checkable limit on how much the result can reveal about any one record.

It combines three ideas:

- **Federated learning.** Each participant trains on its own data and only sends model updates.
- **Differential privacy.** Each record's influence on training is capped and random noise is added, so the
  result changes very little whether or not any single record was used. One number, the privacy budget
  (written ε), says how little.
- **A secure sum.** The server only ever learns the combined update of all participants, never an
  individual one.

Website: [obscuradp.org](https://obscuradp.org)

## Status

**Stage: "Prerun + novel research action".** The library is under construction and is not on PyPI yet.

Read this before relying on anything here:

- The code has **not been independently audited**.
- The research folders (`Experiment 1` to `3`, `Trial 07/27`) are replications and experiments. An internal
  audit found that their privacy numbers do not mean what a reader would assume: for example, some budgets
  are counted per training round and not for the whole run. The findings are in
  [`docs/AUDIT.md`](docs/AUDIT.md). Do not reuse those folders for real data.
- The library is being rebuilt so that every run prints a privacy report whose total can be recomputed by
  anyone. Progress is tracked in [`docs/ROADMAP_STATUS.md`](docs/ROADMAP_STATUS.md).

## Install from source

Requires Python 3.11 or newer.

```bash
git clone https://github.com/aaydinks/ObscuraDP.git
```

```bash
cd ObscuraDP && bash tools/bootstrap.sh
```

The bootstrap script creates `.venv`, installs a CPU-only build of PyTorch when it can reach the download
server, and installs the package with its development tools. To do it by hand:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

## A first example

Three participants with made-up data train one shared model under a single privacy budget for the whole
run. Nothing here needs real data or a network connection.

```python
import numpy as np

import obscuradp as odp
from obscuradp.models import TabularMLP

# Made-up tables: 6 numeric columns and a yes/no outcome that is "yes" for about one row in five.
rng = np.random.default_rng(0)
clients = []
for name, rows in [("north", 900), ("south", 600), ("east", 400)]:
    x = rng.normal(size=(rows, 6)).astype(np.float32)
    y = (x[:, 0] + 0.6 * x[:, 1] + rng.normal(scale=0.5, size=rows) > 1.0).astype(np.float32)
    # Each participant keeps its own training, validation and test rows.
    clients.append(odp.data.make_client_data(name, x, y, seed=0))

simulation = odp.Simulation(
    model_fn=lambda: TabularMLP(6, hidden=(16,)),
    clients=clients,
    plan=odp.Plan(
        dp=odp.LocalDPSGD(clip_norm=1.0),  # every participant adds its own noise
        imbalance=odp.imbalance.ClassWeighted(positive_prior=0.2),  # a public figure, not a count
    ),
    policy=odp.PrivacyPolicy(epsilon=1.0, delta=1e-5),  # one budget for the whole run
    rounds=5,
    learning_rate=0.1,
)
result = simulation.run(seed=0)

print(result.report.summary())  # what was promised, what was spent, what is not covered
print(result.test)  # scores on rows that were never trained on
result.report.to_json("privacy_report.json")
```

The report stores every noisy release that was made, so anyone can recompute its total:

```bash
obscuradp report verify privacy_report.json
```

Settings that would quietly break the promise (for example making synthetic rows from real ones before
private training) are refused unless you pass `allow_unsafe=True`, and are then named in the report.

## Check that it works

```bash
pytest
```

```bash
ruff check . && ruff format --check . && mypy && python tools/check_gitignore.py
```

The tests run with network access switched off. The library sends nothing anywhere, and a test fails if any
code tries to.

## Where things are

| Path | What it is |
|---|---|
| `src/obscuradp/` | The library (first building blocks; no private training yet) |
| `tests/` | Its tests. `tests/golden/` compares the library with the research code, number for number |
| `docs/` | Roadmap, audit, threat model, target definitions, decision records |
| `constraints/` | Pinned package versions for repeatable installs |
| `tools/` | Small scripts used by developers and CI |
| `Experiment 1/` | Replication of SMOTETomek + FedProx ([arXiv:2508.10017](https://arxiv.org/abs/2508.10017)) on stroke data |
| `Experiment 2/` | Replication of standard, private and encrypted federated learning on loan data ([Sci. Rep. 10.1038/s41598-025-34536-9](https://doi.org/10.1038/s41598-025-34536-9)) |
| `Experiment 3/` | Replication of the FedML-HE encryption overhead benchmarks ([arXiv:2303.10837](https://arxiv.org/abs/2303.10837)) |
| `Trial 07/27/` | All three combined in one pipeline, with an independent verification harness |
| `website/` | The public website (Astro, Cloudflare Workers) |

Each experiment folder has its own README with setup, results and the commands to reproduce them. The
research folders are frozen: the library does not import them, and they are kept as a reference that the
library's behaviour is compared against.

## Documentation for contributors

- [`docs/ROADMAP.md`](docs/ROADMAP.md): the plan, session by session.
- [`docs/AUDIT.md`](docs/AUDIT.md): what is wrong with the research code today, with evidence.
- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md): who is protected from whom, and what is out of scope.
- [`docs/KPI_MAPPING.md`](docs/KPI_MAPPING.md): how each published target is defined and measured.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CHANGELOG.md`](CHANGELOG.md).

## Privacy of this project

No telemetry. The library never contacts a server of ours, and there is no counter of users or records.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Citing

See [`CITATION.cff`](CITATION.cff).
