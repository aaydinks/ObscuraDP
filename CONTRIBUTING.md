# Contributing

Thank you for helping. This project handles privacy claims, so the bar for evidence is high and the rules
below are short but firm.

## Set up

Python 3.11 or newer.

```bash
bash tools/bootstrap.sh
```

```bash
. .venv/bin/activate
```

## Before you open a pull request

```bash
pytest
```

```bash
ruff check . && ruff format --check . && mypy && python tools/check_gitignore.py
```

Slow tests and tests that need optional extras are marked and can be selected:

```bash
pytest -m "slow or integration"
```

The comparison tests in `tests/golden/` run the Trial 07/27 research code next to the library and require
identical results. They need the research code's pinned versions:

```bash
pip install -c constraints/legacy-py311.txt -e ".[dev,legacy]"
```

```bash
OBSCURADP_REQUIRE_GOLDEN=1 pytest -m integration
```

Without those packages the comparison tests are skipped; with `OBSCURADP_REQUIRE_GOLDEN=1` a skip is a
failure.

If you change anything under `website/`, also run its checks (see `website/README.md`).

## Ground rules

1. **Evidence over assertion.** Every number in documentation, a README or the website comes from a results
   file produced by committed code. Never edit a results file by hand.
2. **Never weaken a privacy guarantee to make a number look better.** If a fix makes results worse, publish
   the worse number next to the old one and say why.
3. **Tests are not optional.** Do not skip, mark as expected-to-fail, or delete a test to get a green run. If
   a test is wrong, fix the test and explain it in the pull request.
4. **Every release of private information is counted.** New code that looks at private data must charge the
   privacy ledger or be refused by default.
5. **The research folders are frozen.** `Experiment 1` to `3` and `Trial 07/27` are the reference the
   library is compared against. The library never imports them. Change them only to fix a documented error,
   record the correction next to the old value, and then run `python tools/freeze_research.py --write` so
   the fingerprint list in `tests/golden/research_manifest.json` matches. A test fails until you do.
6. **No real personal data** in issues, tests or examples. Use synthetic data.
7. **No telemetry.** The library never opens a network connection on its own. Tests run with sockets
   disabled.

## Writing style

- No em dashes. Use commas, periods, parentheses or colons.
- Math in LaTeX in Markdown (`$...$` inline, `$$...$$` on its own line), except trivial expressions.
- Explanations for users in everyday words. Introduce a specialized term only after explaining it.

## How work is organized

The plan is in `docs/ROADMAP.md` and its progress in `docs/ROADMAP_STATUS.md`. One step is one pull request.
Decisions that should not be reopened without new evidence are in `docs/adr/`.

## Dependencies

- Core dependencies stay small. Anything optional goes into an extra in `pyproject.toml`.
- `constraints/` holds pinned versions. Regenerate them with `bash tools/update_constraints.sh` and commit
  the result in its own pull request.

## Reporting a security problem

See `SECURITY.md`. Please do not open a public issue.
