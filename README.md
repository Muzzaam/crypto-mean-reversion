# Cross-sectional mean reversion in crypto perpetuals

Research into whether a flow-conditioned mean-reversion signal survives
realistic costs and a full overfitting gauntlet on liquid USDT-margined
perpetual futures.

**Status: feasibility stage. No edge has been established. Nothing here is a
trading recommendation.**

---

## The question

Standard mean reversion fades a price deviation from a rolling mean. It fails
in trending regimes, where the deviation keeps widening. The hypothesis under
test is that **participation separates the two cases**: a price move without
broad participation behind it is liquidity-driven and reverts, while one
backed by real flow is a genuine move and does not.

Candidate measures of participation, in rough order of mechanism strength:

| Measure | Source | Mechanism |
|---|---|---|
| Funding rate extremes | exchange API | Crowded leverage paying to stay in |
| Open interest vs price | exchange API | Forced closure vs new positioning |
| Taker delta / CVD divergence | kline `taker_buy_base` | Aggressive flow vs price |
| Candle breadth vs return | OHLC | Move concentrated in few bars |
| ETF net flows | issuer disclosure | Disclosed institutional demand |
| Coinbase premium | cross-venue spot | US institutional demand proxy |
| Session structure | timestamps | TradFi participants are not always present |

## Method

Deliberately staged so the project can be killed cheaply if the premise is
wrong, rather than after weeks of modelling.

1. **Collect** — raw 4h klines and funding history for ~15 liquid perps.
2. **Feasibility** — does mean reversion exist at this horizon at all?
3. **Backtest** — cost-aware, with execution lag enforced structurally.
4. **Gauntlet** — the twelve gates below.
5. **Paper → shadow → live**, with pre-committed kill criteria.

## Methodological commitments

- **Costs first.** Fees, funding and slippage are subtracted inside the
  engine. A gross-return backtest is not evidence.
- **Causal features only.** Rolling windows, expanding ranks, no full-sample
  statistics. The in-progress bar is dropped at collection.
- **Execution lag is structural**, not a matter of discipline. Signals from
  bar *t* fill at bar *t+1*.
- **Newey-West t-stats.** Overlapping *h*-bar forward returns share *h−1*
  bars; naive t-stats on them are inflated by roughly √h.
- **Null controls everywhere.** A test that cannot distinguish a random walk
  from real data is not a test.
- **Multiple-testing discipline.** Every configuration tried is logged in
  [`HYPOTHESES.md`](HYPOTHESES.md), and that count feeds the deflated Sharpe.
- **Baselines are mandatory.** Added complexity must beat buy-and-hold *and*
  the ungated signal out of sample, or it gets removed.

## The validation gauntlet

Run in order. Each gate can kill the strategy; later gates are wasted effort
if an earlier one fails. **These criteria are fixed before results are seen** —
which is the reason they live in the repo rather than in someone's head.

| # | Gate | Tool | Kill criterion |
|---|---|---|---|
| 1 | Signal exists | `feasibility.py` | Non-monotone buckets, or \|t\| < 2 on both tails |
| 2 | Survives costs | `backtest.py` | Net Sharpe < 0.5 after fees, funding, slippage |
| 3 | No leakage | `leakage_audit` | Cheating version barely beats the honest one |
| 4 | Not luck | `stationary_bootstrap_sharpe` | 5th-percentile Sharpe ≤ 0 |
| 5 | Long enough | `min_track_record_length` | MinTRL exceeds available bars |
| 6 | Not search | `deflated_sharpe` | DSR < 0.95 at the true trial count |
| 7 | Selection works | `probability_of_backtest_overfitting` | PBO > 0.5 |
| 8 | Holds out of sample | `purged_walk_forward` | OOS Sharpe < half of in-sample |
| 9 | Plateau not spike | `parameter_plateau` | Neighbour mean < 60% of peak |
| 10 | Regime-independent | `regime_stability` | Profit confined to one regime |
| 11 | Survivable | `monte_carlo_drawdown`, `risk_of_ruin` | p95 drawdown beyond tolerance |
| 12 | Not just beta | correlation to buy-and-hold | Correlation > 0.7 with BTC |

Gates 6 and 7 are the ones retail projects skip, and they are the ones that
fail most real strategies.

## Setup

```bash
git clone https://github.com/Muzzaam/crypto-mean-reversion.git
cd crypto-mean-reversion

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Validate the tooling before trusting it on real data:

```bash
python selftest.py MR && python feasibility.py    # must detect reversion
python selftest.py RW && python feasibility.py    # must NOT detect it
python selftest_validation.py                     # validates the validators
```

Then collect and run:

```bash
python collect.py
python feasibility.py
python feasibility.py --control
```

`data/` is gitignored and fully regenerable from `collect.py`.

## Reading the feasibility output

The `MEAN-REVERTING` verdict line alone is **not** sufficient. With nine
buckets across five horizons there are 45 chances for noise to produce the
desired sign pattern, and a pure random walk does trigger it. Real signal
requires a **monotone** ladder across buckets, **|t| > 2** on both tails, and a
visible difference against `--control`.

## Path to live

1. All twelve gates pass.
2. **Paper trade** for at least the estimated MinTRL, logging intended fills
   and timestamps.
3. **Reconcile** realised slippage against the modelled figure. If realised is
   worse, re-run gate 2 with the true number and re-check gates 4–8.
4. **Shadow mode** — real signals, manual confirmation and execution. Verify
   the live return distribution matches the backtest distribution.
5. **Kill switch, written down before any capital is deployed**: maximum
   drawdown, maximum consecutive losing days, and a live-vs-backtest
   divergence threshold. Committed in writing, or it will be rationalised away
   at exactly the moment it matters.

## Files

| File | Purpose |
|---|---|
| `collect.py` | Raw klines + funding → Parquet. Append-only, de-duplicating. |
| `feasibility.py` | The go/no-go test. Z-score and funding conditioning. |
| `backtest.py` | Cost-aware engine. Execution lag enforced structurally. |
| `validation.py` | DSR, PSR, PBO, purged CV, bootstrap, Monte Carlo ruin. |
| `selftest.py` | Synthetic data with known properties. |
| `selftest_validation.py` | Validates the validators against known signal/noise. |
| `HYPOTHESES.md` | Every configuration tested. The multiple-testing record. |

## Not implemented yet

Signal generation, exit rules, expectancy grid, cross-sectional construction,
HMM regime gate, live signal reporting.

---

Research code. Not financial advice.
