"""
scalp_lab.py - test scalping ideas fast, honestly.

For each signal, at each holding horizon H, it measures:
  - TAKER: enter at the next bar's open, exit at close of bar t+H.
  - MAKER: post a limit at the signal bar's close. Counts as FILLED only if
    the next bar trades THROUGH the limit (low < limit for a buy, high > limit
    for a sell). Merely touching it does not count - you'd be at the back of
    the queue. This captures adverse selection: limit orders fill when price
    is moving against you.

And reports the BREAKEVEN cost per side for each. Compare that to the real
fees of the exchange you'd use. That is the whole viability question.

Events are non-overlapping (after an event, the next H bars are skipped),
so t-stats are not inflated by overlap.

Every cell of the output table is a separate test. With 4 signals x 4
horizons x 2 entry types you are running 32 tests - expect 1-2 to look
"significant" by luck. Log the count in HYPOTHESES.md.

Usage:
    python scalp_lab.py --interval 5m
    python scalp_lab.py --interval 5m --control     # shuffled-bar null
    python scalp_lab.py --interval 1m --horizons 1 5 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load(symbol: str, interval: str) -> pd.DataFrame | None:
    p = RAW / f"klines_{interval}" / f"{symbol}.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p).sort_values("open_time").reset_index(drop=True)
    df = pd.DataFrame({c: k[c].astype(float)
                       for c in ("open", "high", "low", "close", "volume")})
    df["open_time"] = k["open_time"]
    tb = k["taker_buy_base"].astype(float) if "taker_buy_base" in k else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        df["delta_frac"] = np.where(df["volume"] > 0,
                                    (2 * tb - df["volume"]) / df["volume"], np.nan)
    return df


def shuffle_bars(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Null: permute whole bars (return, shape, volume, flow together),
    rebuild prices. Keeps every bar's character, destroys the ordering."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy()
    prev = np.r_[c[0], c[:-1]]
    idx = rng.permutation(len(df))
    ret = (c / prev)[idx]
    o_r = (df["open"].to_numpy() / prev)[idx]
    h_r = (df["high"].to_numpy() / c)[idx]
    l_r = (df["low"].to_numpy() / c)[idx]
    new_c = c[0] * np.cumprod(ret)
    new_prev = np.r_[c[0], new_c[:-1]]
    out = df.copy()
    out["close"], out["open"] = new_c, new_prev * o_r
    out["high"], out["low"] = new_c * h_r, new_c * l_r
    out["volume"] = df["volume"].to_numpy()[idx]
    out["delta_frac"] = df["delta_frac"].to_numpy()[idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Signals: each returns +1 (long), -1 (short), 0 (nothing), known at bar close
# ─────────────────────────────────────────────────────────────────────────────

def _ret_z(df: pd.DataFrame, win: int = 288) -> pd.Series:
    r = np.log(df["close"]).diff()
    return r / r.rolling(win).std().shift(1)      # vol from PRIOR bars only


def sig_overshoot_fade(df):
    """Single-bar move > 3 sigma -> fade it."""
    z = _ret_z(df)
    return pd.Series(np.select([z > 3, z < -3], [-1, 1], 0), index=df.index)


def sig_capitulation_fade(df):
    """Big move on a volume spike (forced flow / liquidations) -> fade."""
    z = _ret_z(df)
    v = np.log(df["volume"].replace(0, np.nan))
    vz = (v - v.rolling(288).mean().shift(1)) / v.rolling(288).std().shift(1)
    big = vz > 3
    return pd.Series(np.select([big & (z > 2.5), big & (z < -2.5)], [-1, 1], 0),
                     index=df.index)


def sig_flow_follow(df):
    """Extreme aggressive-buy or -sell imbalance in a bar -> follow it."""
    d = df["delta_frac"]
    hi = d.rolling(288).quantile(0.99).shift(1)
    lo = d.rolling(288).quantile(0.01).shift(1)
    return pd.Series(np.select([d > hi, d < lo], [1, -1], 0), index=df.index)


def sig_ha_streak3(df):
    """Reference: the Heikin Ashi 3-streak already shown to be noise."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    g = pd.Series(hc > ho, index=df.index)
    s3g = g.rolling(3).sum() == 3
    s3r = (~g).rolling(3).sum() == 3
    return pd.Series(np.select([s3g & ~s3g.shift(1, fill_value=False),
                                s3r & ~s3r.shift(1, fill_value=False)], [1, -1], 0),
                     index=df.index)


SIGNALS = {
    "overshoot_fade": sig_overshoot_fade,
    "capitulation_fade": sig_capitulation_fade,
    "flow_follow": sig_flow_follow,
    "ha_streak3 (ref)": sig_ha_streak3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Event study
# ─────────────────────────────────────────────────────────────────────────────

def events(df: pd.DataFrame, sig: pd.Series, H: int,
           through_bps: float = 1.0) -> dict:
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    s = sig.to_numpy()
    n = len(c)
    taker, maker, attempts = [], [], 0
    i = 0
    while i < n - H - 1:
        d = s[i]
        if d == 0:
            i += 1
            continue
        exit_px = c[i + H]
        # TAKER: fill at next open
        taker.append(d * (exit_px / o[i + 1] - 1) * 1e4)
        # MAKER: limit at signal close, must trade THROUGH on next bar
        attempts += 1
        limit = c[i]
        m = through_bps / 1e4        # price must trade THROUGH by this much
        filled = (l[i + 1] < limit * (1 - m)) if d > 0 else (h[i + 1] > limit * (1 + m))
        if filled:
            maker.append(d * (exit_px / limit - 1) * 1e4)
        i += H + 1                                  # non-overlapping
    return {"taker": np.array(taker), "maker": np.array(maker),
            "attempts": attempts}


def summarise(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 30:
        return np.nan, np.nan
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 6, 12])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--through-bps", type=float, default=1.0,
                    help="maker fill requires price to trade this far through the limit")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    kdir = RAW / f"klines_{args.interval}"
    symbols = args.symbols or sorted(p.stem for p in kdir.glob("*.parquet"))
    if not symbols:
        raise SystemExit(f"No data in {kdir}. Run collect.py --interval {args.interval}")

    frames = {}
    for s in symbols:
        df = load(s, args.interval)
        if df is not None and len(df) > 1000:
            frames[s] = shuffle_bars(df) if args.control else df

    mode = "NULL CONTROL (shuffled)" if args.control else "REAL DATA"
    print(f"\n{'=' * 96}\n  {mode} | {args.interval} | {len(frames)} symbols\n{'=' * 96}")
    print(f"{'signal':<20}{'H':>4}{'events':>9}{'taker bps':>11}{'t':>7}"
          f"{'fill%':>8}{'maker bps':>11}{'t':>7}{'BE taker':>10}{'BE maker':>10}")
    print("-" * 96)

    for name, fn in SIGNALS.items():
        sigs = {s: fn(df) for s, df in frames.items()}
        for H in args.horizons:
            tk, mk, att = [], [], 0
            for s, df in frames.items():
                e = events(df, sigs[s], H, args.through_bps)
                tk.append(e["taker"])
                mk.append(e["maker"])
                att += e["attempts"]
            tk, mk = np.concatenate(tk), np.concatenate(mk)
            tm, tt = summarise(tk)
            mm, mt = summarise(mk)
            fill = len(mk) / att * 100 if att else np.nan
            # taker: both sides taker. maker: entry maker, exit taker (market out)
            be_t = tm / 2 if np.isfinite(tm) else np.nan
            be_m = mm / 2 if np.isfinite(mm) else np.nan
            print(f"{name:<20}{H:>4}{len(tk):>9,}{tm:>11.2f}{tt:>7.1f}"
                  f"{fill:>8.1f}{mm:>11.2f}{mt:>7.1f}{be_t:>10.2f}{be_m:>10.2f}")
        print()

    print("BE = breakeven cost per side, in bps. A signal is only viable if its")
    print("BE is comfortably ABOVE your exchange's real all-in cost per side.")
    print("Typical retail: taker ~5-6 bps/side, maker ~0-2 bps/side.")
    print("Compare against --control: real-data edges must clearly exceed the null.")


if __name__ == "__main__":
    main()
