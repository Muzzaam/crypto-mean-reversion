"""Validate the validation tools against data with KNOWN properties.

A test suite that cannot tell a real edge from noise is worse than no test
suite, because it launders noise into confidence. Run this before trusting
anything in validation.py or backtest.py.

    python selftest_validation.py
"""
import numpy as np
import pandas as pd

import backtest as B
import validation as V


def test_sharpe_statistics():
    print("\n[1] SHARPE STATISTICS — null vs real edge")
    rng = np.random.default_rng(1)
    n = 4000
    cases = {"NULL (no edge)": pd.Series(rng.normal(0, 0.01, n)),
             "EDGE (real)":    pd.Series(rng.normal(0.0004, 0.01, n))}
    for name, r in cases.items():
        d = V.deflated_sharpe(r, n_trials=50, sharpe_variance=4e-4)
        b = V.stationary_bootstrap_sharpe(r, n_boot=300)
        print(f"  {name:16s} SR_ann={V.sharpe(r, 2190):6.2f}  "
              f"PSR={V.probabilistic_sharpe(r):.3f}  "
              f"DSR={d['deflated_sharpe']:.3f}  boot={b['verdict']}")
    print("  EXPECT: null fails everything; edge passes PSR/bootstrap but")
    print("          still FAILS DSR at 50 trials — that is the whole point.")


def test_pbo():
    print("\n[2] PBO — can it detect that selection is worthless?")
    rng = np.random.default_rng(3)
    n, m = 3000, 20
    noise = pd.DataFrame(rng.normal(0, 0.01, (n, m)),
                         columns=[f"c{i}" for i in range(m)])
    real = noise.copy()
    real["c7"] = rng.normal(0.0008, 0.01, n)
    for name, df in [("all configs noise", noise), ("one real edge", real)]:
        r = V.probability_of_backtest_overfitting(df, n_splits=10, max_combos=200)
        print(f"  {name:20s} PBO={r['pbo']:.2f}  {r['verdict']}")
    print("  EXPECT: noise PBO > 0.5 (fail), real edge PBO near 0 (pass).")


def test_purging():
    print("\n[3] PURGED SPLITS — no train/test contamination")
    sp = list(V.purged_walk_forward(1000, n_folds=4, embargo_frac=0.02, purge=16))
    overlap = any(set(a) & set(b) for a, b in sp)
    gap_ok = all(a.max() < b[0] - 15 for a, b in sp if len(a) and a.max() < b[0])
    print(f"  folds={len(sp)}  overlap={overlap}  purge_gap_respected={gap_ok}")
    print("  EXPECT: overlap False, purge_gap True.")


def test_engine():
    print("\n[4] ENGINE — costs bite, lag is enforced")
    rng = np.random.default_rng(5)
    n = 4000
    trend = np.cumsum(rng.normal(0, 0.004, n))
    dev = np.zeros(n)
    for t in range(1, n):
        dev[t] = 0.9 * dev[t - 1] + rng.normal(0, 0.012)
    close = np.exp(np.log(30000) + trend + dev)
    df = pd.DataFrame({"close": close,
                       "funding_rate": np.where(np.arange(n) % 2 == 0, 1e-4, 0.0)})
    logp = np.log(df.close)
    z = (logp - logp.rolling(30).mean()) / logp.rolling(30).std()
    sig = (-z / 2.5).clip(-1, 1)

    for label, c in [("zero cost", B.Costs(0, 0, 0)), ("real cost", B.Costs(5, 2, 2))]:
        s = B.run_backtest(df, sig, c)["stats"]
        print(f"  {label:10s} SR={s['sharpe_ann']:5.2f}  trades={s['n_trades']:5d}  "
              f"drag={s['cost_drag_ann_bps']:6.0f}bps")
    print("  NOTE: the synthetic series has an artificial AR(1) with no real")
    print("        analogue. These Sharpes say the ENGINE works, nothing more.")
    print("\n  leakage audit:")
    B.leakage_audit(df, sig)


if __name__ == "__main__":
    test_sharpe_statistics()
    test_pbo()
    test_purging()
    test_engine()
    print("\nAll checks printed. Compare against the EXPECT lines above.")
