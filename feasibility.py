"""
feasibility.py — the gate on whether this project continues.

One question: conditional on price being N sigma from its rolling mean, what is
the average forward return? If mean reversion exists, forward returns in the
high-|z| buckets must oppose the sign of z, and do so monotonically. If they
don't, no exit rule, gate or regime model will manufacture an edge.

Also tests funding as a conditioning variable, since if funding carries signal
(rather than being pure cost) the strategy design changes.

Look-ahead defences:
  - z-score uses a trailing rolling window (causal).
  - funding percentile uses an EXPANDING rank, not a full-sample rank. A
    full-sample percentile leaks the future distribution into every early row.
  - funding is joined backwards onto bar close (only settlements you'd have seen).
  - t-stats are Newey-West corrected. Overlapping h-bar forward returns are
    autocorrelated by construction; naive t-stats on them are inflated by
    roughly sqrt(h) and will tell you noise is significant.

Usage:
    python feasibility.py
    python feasibility.py --control      # same test on shuffled returns (null)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
HORIZONS = [1, 2, 4, 8, 16]
Z_EDGES = [-np.inf, -2.5, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 2.5, np.inf]


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """t-stat for mean(x) != 0, robust to the MA(h-1) structure that
    overlapping forward returns induce. Bartlett kernel."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    xc = x - x.mean()
    s = np.dot(xc, xc) / n
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * np.dot(xc[l:], xc[:-l]) / n
    if s <= 0:
        return np.nan
    return x.mean() / np.sqrt(s / n)


def build_features(symbol: str, interval: str, lookback: int,
                   control: bool) -> pd.DataFrame:
    kp = RAW / f"klines_{interval}" / f"{symbol}.parquet"
    if not kp.exists():
        return pd.DataFrame()
    k = pd.read_parquet(kp).sort_values("open_time").reset_index(drop=True)

    df = pd.DataFrame({
        "symbol": symbol,
        "open_time": k["open_time"],
        "close_time": k["close_time"],
        "close": k["close"].astype(float),
    })
    df["logp"] = np.log(df["close"])
    df["ret"] = df["logp"].diff()

    if control:
        # NULL CONTROL: shuffle returns, rebuild the price path. Destroys all
        # serial structure, keeps the return distribution. Anything that still
        # looks significant here is an artefact of the test, not the market.
        rng = np.random.default_rng(0)
        r = df["ret"].to_numpy().copy()
        mask = ~np.isnan(r)
        r[mask] = rng.permutation(r[mask])
        df["ret"] = r
        df["logp"] = df["logp"].iloc[0] + np.nan_to_num(r).cumsum()

    # --- causal z-score: trailing window ending at the current bar ---
    mu = df["logp"].rolling(lookback).mean()
    sd = df["logp"].rolling(lookback).std()
    df["z"] = (df["logp"] - mu) / sd

    # --- taker flow, straight out of the kline (no websocket needed) ---
    # delta = taker buys - taker sells = 2*taker_buy - total
    vol = k["volume"].astype(float).to_numpy()
    tb = k["taker_buy_base"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["delta_frac"] = np.where(vol > 0, (2 * tb - vol) / vol, np.nan)

    # --- funding, joined backwards so only settled payments are visible ---
    fp = RAW / "funding" / f"{symbol}.parquet"
    if fp.exists():
        f = pd.read_parquet(fp).sort_values("funding_time")
        df = pd.merge_asof(
            df.sort_values("close_time"), f,
            left_on="close_time", right_on="funding_time",
            direction="backward",
        )
        # expanding rank -> percentile using only history available at time t
        df["funding_pct"] = (
            df["funding_rate"].expanding(min_periods=200)
            .apply(lambda s: (s.iloc[-1] > s.iloc[:-1]).mean(), raw=False)
        )
    else:
        df["funding_rate"] = np.nan
        df["funding_pct"] = np.nan

    # --- forward returns (the only forward-looking columns; never features) ---
    for h in HORIZONS:
        df[f"fwd_{h}"] = df["logp"].shift(-h) - df["logp"]

    return df


def bucket_report(df: pd.DataFrame, by: str, edges, label: str) -> pd.DataFrame:
    df = df.copy()
    df["_bucket"] = pd.cut(df[by], bins=edges)
    out = []
    for b, g in df.groupby("_bucket", observed=True):
        row = {label: str(b), "n": len(g)}
        for h in HORIZONS:
            x = g[f"fwd_{h}"].to_numpy()
            row[f"fwd{h}_bps"] = np.nanmean(x) * 1e4
            row[f"fwd{h}_t"] = newey_west_t(x, lag=h)
        out.append(row)
    return pd.DataFrame(out)


def verdict(rep: pd.DataFrame, label: str) -> None:
    """Mean reversion requires the extreme buckets to oppose the deviation."""
    print(f"\n  VERDICT ({label}):")
    lo, hi = rep.iloc[0], rep.iloc[-1]
    for h in HORIZONS:
        lo_r, hi_r = lo[f"fwd{h}_bps"], hi[f"fwd{h}_bps"]
        lo_t, hi_t = lo[f"fwd{h}_t"], hi[f"fwd{h}_t"]
        ok = (lo_r > 0) and (hi_r < 0)
        sig = (abs(lo_t) > 2) and (abs(hi_t) > 2)
        tag = "MEAN-REVERTING" if ok else ("trending" if lo_r < 0 and hi_r > 0
                                           else "no clear sign")
        star = "  <-- significant" if (ok and sig) else ""
        print(f"    h={h:>2}: low={lo_r:+7.1f}bps (t={lo_t:+5.2f})  "
              f"high={hi_r:+7.1f}bps (t={hi_t:+5.2f})  {tag}{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--lookback", type=int, default=30,
                    help="bars in the rolling mean/std window")
    ap.add_argument("--control", action="store_true",
                    help="run on shuffled returns as a null control")
    args = ap.parse_args()

    kdir = RAW / f"klines_{args.interval}"
    symbols = sorted(p.stem for p in kdir.glob("*.parquet")) if kdir.exists() else []
    if not symbols:
        raise SystemExit(f"No data in {kdir}. Run collect.py first.")

    frames = [build_features(s, args.interval, args.lookback, args.control)
              for s in symbols]
    panel = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    mode = "NULL CONTROL (shuffled)" if args.control else "REAL DATA"
    print(f"\n{'=' * 78}")
    print(f"  {mode} | {args.interval} bars | lookback={args.lookback} | "
          f"{len(symbols)} symbols | {len(panel):,} rows")
    print(f"{'=' * 78}")

    pd.set_option("display.width", 200, "display.max_columns", 40,
                  "display.float_format", lambda v: f"{v:8.2f}")

    print("\n[1] FORWARD RETURNS BY Z-SCORE BUCKET (bps, pooled across symbols)")
    z_rep = bucket_report(panel.dropna(subset=["z"]), "z", Z_EDGES, "z_bucket")
    print(z_rep.to_string(index=False))
    verdict(z_rep, "z-score")

    if panel["funding_pct"].notna().any():
        print("\n[2] FORWARD RETURNS BY FUNDING PERCENTILE (expanding rank)")
        f_edges = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        f_rep = bucket_report(panel.dropna(subset=["funding_pct"]),
                              "funding_pct", f_edges, "funding_pct")
        print(f_rep.to_string(index=False))
        verdict(f_rep, "funding")

    print("\n" + "-" * 78)
    print("Read this as: if the extreme z buckets do not oppose the deviation")
    print("with |t| > 2, stop. Then re-run with --control: the real data must")
    print("look clearly different from the shuffled null or you have nothing.")
    print("Log every lookback value you try. That count feeds the deflated")
    print("Sharpe calculation later.")


if __name__ == "__main__":
    main()
