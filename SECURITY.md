# Security policy

Obscura DP is privacy software, so a mistake here can expose people's data. Reports are welcome and are
treated as the highest priority.

## Status

The project is at the stage "Prerun + novel research action". **Nothing in this repository has been
independently audited.** The research folders (`Experiment 1` to `3`, `Trial 07/27`) have known privacy
defects, listed in `docs/AUDIT.md`, and must not be used with real data.

## What counts as a security problem

- Any way the library reveals more about a record than its privacy report states.
- A privacy budget that is counted wrongly, or a release of information that is not counted at all.
- A way for the aggregation server, or another participant, to read an individual participant's update when
  the chosen configuration says it cannot.
- Weak or predictable randomness in noise or key generation.
- Any network connection the library opens on its own.
- A dependency with a known vulnerability that affects the library.

What the library does not try to defend against is listed in `docs/THREAT_MODEL.md`. Reports about those
limits are still useful if the documentation states them unclearly.

## How to report

Please do not open a public issue for a vulnerability.

1. Use GitHub's private reporting: on the repository page, open the **Security** tab and choose
   **Report a vulnerability**.
2. If that is not available to you, use the contact form at <https://obscuradp.org/#contact> and say only that
   you have a security report. You will be given a private channel.

Please include what you ran, what you expected the privacy report to say, what you observed, and the
smallest example that shows it.

## What happens next

- You get an acknowledgement within 7 days.
- A confirmed problem gets a failing test first, then a fix, then a note in `CHANGELOG.md` under "Security".
- Any published result that the problem affects is corrected next to the old value, never silently replaced.
- You are credited unless you ask not to be.

## Supported versions

Only the latest release, once there is one. There are no releases yet.
