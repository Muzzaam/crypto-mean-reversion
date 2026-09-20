# Hypothesis log

Every configuration tested, including the ones that failed and the ones
abandoned halfway. **The dead ends are the point.**

Why this file is committed rather than kept locally: git history timestamps
each entry. That makes the record tamper-evident — you cannot retroactively
claim you only tried three configurations when the log says you tried forty.
The count in this file is the `n_trials` argument to `deflated_sharpe()`. A
strategy selected as the best of 40 needs a far higher bar than one tested
once, and without an honest count that correction cannot be computed.

Rules:
- Log **before** you look at the result. Write the hypothesis, then run it.
- Log abandoned runs too ("killed it, looked bad") — those still consumed a
  degree of freedom.
- Never edit or delete a past entry. Add a new one that supersedes it.

---

| # | Date | Hypothesis | Config | Result | Verdict |
|---|------|-----------|--------|--------|---------|
| 1 | 2026-09-21 | Mean reversion exists in 4h returns conditional on z-score | `feasibility.py --lookback 30 --interval 4h` | Monotone ladder h=1–16, t=5–17, null control clean | PASS |

---

## Running count

- Configurations tested: **0**
- Distinct hypotheses: **0**

This total — not the count of the ones that worked — is what goes into the
deflated Sharpe calculation.
