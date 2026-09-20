"""
validation.py — the statistical gauntlet.

These are the tests that decide whether a backtest is evidence or a story.
Implemented from the source papers rather than pulled from a library, because
the failure modes live in the details.

References:
  Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
  Bailey, Borwein, Lopez de Prado & Zhu (2016), "The Probability of Backtest
      Overfitting" (CSCV)
  Lopez de Prado (2018), "Advances in Financial Machine Learning" (purging,
      embargo, combinatorial CV)
  Politis & Romano (1994), stationary bootstrap

Convention: `returns` is a pandas Series of PER-BAR strategy returns, net of
all costs. Sharpe ratios are per-bar unless explicitly annualised.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sharpe ratio, honestly
# ─────────────────────────────────────────────────────────────────────────────

def sharpe(returns: pd.Series, periods_per_year: int | None = None) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    return sr * np.sqrt(periods_per_year) if periods_per_year else sr


def sharpe_stderr(returns: pd.Series) -> float:
    """Standard error of the per-bar Sharpe, accounting for skew and kurtosis.

    The naive 1/sqrt(n) SE assumes normal returns. Crypto returns are skewed
    and fat-tailed, which inflates Sharpe uncertainty. Ignoring that is how a
    Sharpe of 1.2 gets reported as though it were established.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 30:
        return np.nan
    sr = sharpe(r)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)  # non-excess
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1)
    return np.sqrt(max(var, 0))


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), given observed higher moments.

    PSR < 0.95 means you cannot claim the strategy beats the benchmark, no
    matter how good the equity curve looks.
    """
    se = sharpe_stderr(returns)
    if not np.isfinite(se) or se == 0:
        return np.nan
    return float(stats.norm.cdf((sharpe(returns) - sr_benchmark) / se))


def min_track_record_length(returns: pd.Series, sr_benchmark: float = 0.0,
                            confidence: float = 0.95) -> float:
    """Bars of track record needed before PSR would clear `confidence`.

    If this exceeds the data you have, your backtest is too short to support
    the claim — regardless of the result.
    """
    r = pd.Series(returns).dropna()
    sr = sharpe(r)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return np.inf
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    z = stats.norm.ppf(confidence)
    return 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deflated Sharpe — the multiple-testing correction
# ─────────────────────────────────────────────────────────────────────────────

def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected MAXIMUM Sharpe under the null of zero true edge, across
    `n_trials` independent backtests.

    This is the number people never compute. Search 100 configurations of a
    worthless strategy and the best one will show a respectable Sharpe purely
    by construction. This says how respectable.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = stats.norm.ppf(1 - 1 / n_trials)
    b = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - g) * a + g * b))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    sharpe_variance: float) -> dict:
    """PSR benchmarked against the expected max Sharpe from searching.

    `n_trials`      — EVERY configuration you evaluated, including failures.
                      Read it off HYPOTHESES.md. Under-reporting it makes this
                      test meaningless.
    `sharpe_variance` — variance of the per-bar Sharpes across those trials.

    DSR < 0.95 means the result is indistinguishable from the best of N
    random searches. That is the verdict that matters.
    """
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    dsr = probabilistic_sharpe(returns, sr_benchmark=sr0)
    return {
        "observed_sharpe": sharpe(returns),
        "n_trials": n_trials,
        "expected_max_sharpe_under_null": sr0,
        "deflated_sharpe": dsr,
        "verdict": "PASS" if dsr > 0.95 else "FAIL — indistinguishable from search",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probability of Backtest Overfitting (CSCV)
# ─────────────────────────────────────────────────────────────────────────────

def probability_of_backtest_overfitting(perf: pd.DataFrame, n_splits: int = 16,
                                        max_combos: int = 1000) -> dict:
    """PBO via Combinatorial Symmetric Cross-Validation.

    `perf` — DataFrame of per-bar returns, one COLUMN PER CONFIGURATION you
             tried. Rows are time.

    Procedure: chop time into S blocks; for every way of splitting the blocks
    into equal train/test halves, pick the configuration that ranks best
    in-sample, then look at where it ranks out-of-sample. PBO is the fraction
    of splits where the in-sample winner lands below the OOS median.

    PBO > 0.5 means your selection procedure is worse than picking at random.
    It catches the failure that walk-forward alone misses: not "does this
    strategy work" but "does my way of choosing strategies work".
    """
    perf = perf.dropna()
    n, m = perf.shape
    if m < 2:
        raise ValueError("PBO needs >= 2 configurations to compare")
    if n_splits % 2:
        n_splits += 1

    blocks = np.array_split(np.arange(n), n_splits)
    half = n_splits // 2
    combos = list(combinations(range(n_splits), half))
    if len(combos) > max_combos:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in
                  rng.choice(len(combos), max_combos, replace=False)]

    logits = []
    for tr in combos:
        te = [i for i in range(n_splits) if i not in tr]
        tr_idx = np.concatenate([blocks[i] for i in tr])
        te_idx = np.concatenate([blocks[i] for i in te])

        is_sr = perf.iloc[tr_idx].apply(sharpe)
        oos_sr = perf.iloc[te_idx].apply(sharpe)
        if is_sr.isna().all() or oos_sr.isna().all():
            continue

        best = is_sr.idxmax()
        # relative rank of the IS winner within the OOS distribution
        rank = oos_sr.rank(pct=True)[best]
        rank = min(max(rank, 1 / (m + 1)), 1 - 1 / (m + 1))  # keep logit finite
        logits.append(np.log(rank / (1 - rank)))

    logits = np.asarray(logits)
    pbo = float((logits <= 0).mean()) if len(logits) else np.nan
    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "median_logit": float(np.median(logits)) if len(logits) else np.nan,
        "verdict": "PASS" if pbo < 0.5 else "FAIL — selection is worse than random",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Purged / embargoed splits
# ─────────────────────────────────────────────────────────────────────────────

def purged_walk_forward(n: int, n_folds: int = 6, embargo_frac: float = 0.01,
                        purge: int = 0):
    """Expanding walk-forward with purging and embargo.

    Purge   — drop training rows whose LABEL overlaps the test window. With
              h-bar forward returns, the last h training rows peek into test.
    Embargo — drop training rows immediately AFTER the test window too, since
              serial correlation leaks backwards across the boundary.

    Skipping these is the most common way a crypto backtest leaks. It shows up
    as an OOS Sharpe that is implausibly close to the in-sample one.

    Yields (train_idx, test_idx).
    """
    folds = np.array_split(np.arange(n), n_folds + 1)
    emb = int(n * embargo_frac)
    for k in range(1, n_folds + 1):
        test = folds[k]
        t0, t1 = test[0], test[-1]
        train = np.arange(0, max(t0 - purge, 0))
        after = np.arange(min(t1 + 1 + emb, n), n)
        yield np.concatenate([train, after]) if len(after) else train, test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bootstrap and Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def stationary_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000,
                                mean_block: int = 20, seed: int = 0) -> dict:
    """Confidence interval for the Sharpe via stationary bootstrap.

    Resamples geometrically-sized blocks, preserving serial dependence that an
    iid bootstrap would destroy. If the 5th percentile is below zero, you
    cannot reject "no edge".
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < 50:
        return {"error": "need >= 50 observations"}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        s = r[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd if sd > 0 else np.nan
    out = out[np.isfinite(out)]
    lo, hi = np.percentile(out, [5, 95])
    return {
        "sharpe_point": sharpe(returns),
        "ci_5": float(lo), "ci_95": float(hi),
        "p_sharpe_le_0": float((out <= 0).mean()),
        "verdict": "PASS" if lo > 0 else "FAIL — CI includes zero",
    }


def monte_carlo_drawdown(returns: pd.Series, n_sims: int = 5000,
                         seed: int = 0) -> dict:
    """Distribution of max drawdown from reshuffled trade order.

    Your backtest shows ONE drawdown path — one ordering of the same trades.
    Reshuffle and you see the drawdown you could plausibly have suffered.
    Size the account against the 95th percentile, not the realised one.
    """
    r = pd.Series(returns).dropna().to_numpy()
    if len(r) < 30:
        return {"error": "need >= 30 observations"}
    rng = np.random.default_rng(seed)

    def mdd(x):
        eq = np.cumprod(1 + x)
        return float((eq / np.maximum.accumulate(eq) - 1).min())

    sims = np.array([mdd(rng.permutation(r)) for _ in range(n_sims)])
    return {
        "realised_max_dd": mdd(r),
        "median_max_dd": float(np.median(sims)),
        "p95_max_dd": float(np.percentile(sims, 5)),   # 5th pct = worst 95%
        "p99_max_dd": float(np.percentile(sims, 1)),
        "note": "size the account against p95, not the realised path",
    }


def risk_of_ruin(returns: pd.Series, ruin_threshold: float = -0.5,
                 horizon: int = 500, n_sims: int = 5000, seed: int = 0) -> float:
    """P(equity falls below `ruin_threshold` within `horizon` bars).

    The constraint that actually binds on leveraged derivatives. A strategy
    with a great Sharpe and a 15% chance of ruin is not tradeable.
    """
    r = pd.Series(returns).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    ruined = 0
    for _ in range(n_sims):
        path = np.cumprod(1 + rng.choice(r, horizon, replace=True)) - 1
        if path.min() <= ruin_threshold:
            ruined += 1
    return ruined / n_sims


# ─────────────────────────────────────────────────────────────────────────────
# 6. Robustness
# ─────────────────────────────────────────────────────────────────────────────

def parameter_plateau(results: pd.DataFrame, param_cols: list[str],
                      metric: str = "sharpe") -> dict:
    """Is the edge a plateau or a spike?

    `results` — one row per parameter combination, with the metric achieved.

    Real edge degrades gracefully as you move a parameter. A sharp peak
    surrounded by poor neighbours is curve-fitting, even if the peak is high.
    Compares the best cell against its immediate neighbourhood.
    """
    df = results.dropna(subset=[metric]).copy()
    if df.empty:
        return {"error": "no valid rows"}
    best = df.loc[df[metric].idxmax()]

    norm = df[param_cols].apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1))
    bnorm = (best[param_cols] - df[param_cols].mean()) / df[param_cols].std(ddof=0).replace(0, 1)
    dist = np.sqrt(((norm - bnorm) ** 2).sum(axis=1))
    nb = df[(dist > 0) & (dist <= 1.5)]

    if nb.empty:
        return {"error": "no neighbours — widen the grid"}
    ratio = float(nb[metric].mean() / best[metric]) if best[metric] else np.nan
    return {
        "best_params": best[param_cols].to_dict(),
        "best_metric": float(best[metric]),
        "neighbour_mean": float(nb[metric].mean()),
        "neighbour_ratio": ratio,
        "verdict": "PASS — plateau" if ratio > 0.6 else "FAIL — isolated spike",
    }


def regime_stability(returns: pd.Series, benchmark: pd.Series,
                     n_regimes: int = 3) -> pd.DataFrame:
    """Performance split by benchmark regime (e.g. BTC trend tercile).

    A strategy that only works in one regime is a bet on that regime.
    """
    df = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    df["regime"] = pd.qcut(df["b"].rolling(30).mean(), n_regimes,
                           labels=[f"regime_{i}" for i in range(n_regimes)])
    return df.groupby("regime", observed=True)["r"].agg(
        n="count", mean_bps=lambda s: s.mean() * 1e4, sharpe=sharpe)


def full_report(returns: pd.Series, n_trials: int, sharpe_variance: float,
                periods_per_year: int = 2190) -> None:
    """Print the core battery. 2190 = 4h bars per year."""
    r = pd.Series(returns).dropna()
    print("=" * 70)
    print(f"  n={len(r)}  Sharpe(ann)={sharpe(r, periods_per_year):.3f}  "
          f"Sharpe(bar)={sharpe(r):.5f}")
    print("=" * 70)
    print(f"\nPSR vs 0            : {probabilistic_sharpe(r):.4f}")
    mtrl = min_track_record_length(r)
    print(f"Min track record    : {mtrl:,.0f} bars "
          f"({'HAVE ENOUGH' if mtrl <= len(r) else 'TOO SHORT'})")
    for k, v in deflated_sharpe(r, n_trials, sharpe_variance).items():
        print(f"  {k:<32}: {v}")
    print()
    for k, v in stationary_bootstrap_sharpe(r).items():
        print(f"  {k:<32}: {v}")
    print()
    for k, v in monte_carlo_drawdown(r).items():
        print(f"  {k:<32}: {v}")
    print(f"\nRisk of 50% loss (500 bars): {risk_of_ruin(r):.3%}")
