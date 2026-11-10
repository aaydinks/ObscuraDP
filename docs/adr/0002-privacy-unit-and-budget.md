# 0002. Privacy unit and how the budget is counted

Status: accepted by the owner on 2026-11-10.

## Context

Experiment 1 and Trial 07/27 state the budget per round; Experiment 2 states it for the whole run (finding
G2 in `docs/AUDIT.md`). A labelled 9 in Trial 07/27 is really about 39 over the run. The website mixes both.
The code also never says what "one person" means.

## Decision

1. The user sets one total $(\varepsilon, \delta)$ for the whole training run. No per-round budgets appear
   in the public interface or in any new result.
2. The default unit is **one record**: one row of one participant's table. Two datasets are neighbours if
   one has a single record added or removed. A second unit, one whole participant, is available for large
   populations and is always named in the report.
3. The number of records each participant declares is treated as public, because it sets the sampling rate
   and the number of steps.
4. If one person can own up to $k$ rows, the report states that the protection for that person is weaker by
   a factor of $k$.
5. Every release of information that depends on private data is charged to one ledger **before** it
   happens. Anything that cannot be charged is refused, or allowed only with `allow_unsafe=True` and a flag
   in the report.
6. The tighter PLD accountant sets the noise; the RDP accountant is run as a cross-check and must fall
   inside an expected band above it.
7. Each participant may choose its own budget in local mode. The report lists every participant's spend and
   the maximum.

## Consequences

- Old results cannot be compared with new ones by their labels. S11 republishes them side by side.
- Noise levels rise by a factor of 4 to 20 at a total budget of 1 (table 1 in `docs/KPI_MAPPING.md`).
- The website must say "for the whole training run, per record" wherever a budget value appears.
