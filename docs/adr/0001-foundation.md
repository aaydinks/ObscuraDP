# 0001. Standalone core with an optional Flower adapter

Status: accepted by the owner on 2026-11-10.

## Context

The research code runs every participant and the server inside one process. A real deployment needs
networking between machines. Flower is the most widely used open framework for that, and it already has
building blocks for privacy noise and for hiding individual updates, but its privacy features are labelled
preview and change between releases.

## Decision

`obscuradp` is a standalone library. Budget accounting, noise mechanisms, secure sums, masks, audits and a
single-process simulator have no dependency on Flower. An optional extra, `obscuradp[flower]`, maps the same
`Plan` and `PrivacyPolicy` objects onto Flower so that moving from simulation to real machines is a
configuration change.

## Consequences

- The privacy-critical code can be tested without any networking.
- The adapter must stay thin, and a test must show that its privacy report equals the simulator's report
  for the same configuration (S19).
- If time runs out, the adapter is late in the cut order, and dropping it needs the owner's consent.
