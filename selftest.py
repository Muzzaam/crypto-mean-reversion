"""Validate feasibility.py against data with KNOWN properties.

Generates two fake universes:
  MR  : log price = slow trend + stationary AR(1) deviation  -> must be detected
  RW  : pure random walk                                     -> must NOT be detected

If the script flags the random walk, the test is broken and would have sent
Muzzi chasing a phantom edge.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")


def make(symbol: str, kind: str, n: int = 6000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    sigma = 0.012

    if kind == "MR":
        trend = np.cumsum(rng.normal(0, sigma * 0.35, n))
        dev, phi = np.zeros(n), 0.90          # AR(1) deviation, half-life ~6.6 bars
        for t in range(1, n):
            dev[t] = phi * dev[t - 1] + rng.normal(0, sigma)
        logp = np.log(30000) + trend + dev
    else:
        logp = np.log(30000) + np.cumsum(rng.normal(0, sigma, n))

    close = np.exp(logp)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    open_time = t0 + pd.to_timedelta(np.arange(n) * 4, unit="h")
    close_time = open_time + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1)
    vol = rng.lognormal(6, 0.5, n)

    k = pd.DataFrame({
        "open_time": open_time,
        "open": close, "high": close * 1.004, "low": close * 0.996, "close": close,
        "volume": vol, "close_time": close_time,
        "quote_volume": vol * close, "trades": rng.integers(1e3, 1e5, n),
        "taker_buy_base": vol * rng.uniform(0.4, 0.6, n),
        "taker_buy_quote": vol * close * 0.5,
    })
    p = RAW / "klines_4h" / f"{symbol}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    k.to_parquet(p, index=False)

    f = pd.DataFrame({
        "funding_time": open_time[::2],
        "funding_rate": rng.normal(1e-4, 3e-4, len(open_time[::2])),
    })
    fp = RAW / "funding" / f"{symbol}.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(fp, index=False)


def build(kind: str) -> None:
    shutil.rmtree(RAW, ignore_errors=True)
    for i in range(6):
        make(f"{kind}{i}USDT", kind, seed=i)


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
    print(f"built synthetic {sys.argv[1]} universe")
