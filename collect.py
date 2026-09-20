"""
collect.py — pull raw 4h klines + funding-rate history for liquid USDT-M perps.

Design rules (these exist to prevent look-ahead bugs, do not relax them):
  1. RAW ONLY. Nothing derived is written here. No z-scores, no returns.
  2. Never overwrite. Re-running appends and de-duplicates on timestamp.
  3. Drop the in-progress bar. The final kline from the API is the current,
     unclosed bar. Keeping it is the single most common source of look-ahead
     bias in crypto backtests.
  4. All timestamps stored UTC, tz-aware.

Usage:
    python collect.py                 # default universe, 4h, from 2019
    python collect.py --interval 1d --start 2021-01-01
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
RAW = Path("data/raw")

# Liquid USDT-M perps with long history. Deliberately excludes anything that
# only listed recently — short history is worse than no history because it
# silently biases the sample toward one market regime.
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT",
]

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mr-research/1.0"})


def _get(path: str, params: dict, max_retries: int = 5) -> list:
    """GET with backoff. Binance 429s if you hammer it; 418 means you got banned."""
    for attempt in range(max_retries):
        r = SESSION.get(f"{BASE}{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 418):
            wait = 2 ** attempt * 5
            print(f"    rate limited ({r.status_code}), sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after {max_retries} retries: {path} {params}")


def fetch_klines(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    """Page forward through klines. Binance caps at 1500 rows per call."""
    rows, cursor = [], start_ms
    while True:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        print(f"    {symbol} klines: {len(rows)}", end="\r")
        if len(batch) < 1500:
            break
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore"])
    num = ["open", "high", "low", "close", "volume", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote"]
    df[num] = df[num].apply(pd.to_numeric)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # RULE 3: drop any bar whose close is in the future — it is still forming.
    df = df[df["close_time"] <= pd.Timestamp.utcnow()]
    return df.reset_index(drop=True)


def fetch_funding(symbol: str, start_ms: int) -> pd.DataFrame:
    """Realised funding payments. Binance settles every 8h on most perps."""
    rows, cursor = [], start_ms
    while True:
        batch = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        print(f"    {symbol} funding: {len(rows)}", end="\r")
        if len(batch) < 1000:
            break
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    return df[["funding_time", "funding_rate"]].reset_index(drop=True)


def save(df: pd.DataFrame, path: Path, time_col: str) -> None:
    """RULE 2: merge with whatever is already on disk, de-dup, never truncate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = (df.drop_duplicates(subset=[time_col], keep="last")
            .sort_values(time_col)
            .reset_index(drop=True))
    df.to_parquet(path, index=False)
    print(f"    -> {path}  ({len(df)} rows, {df[time_col].min().date()} .. "
          f"{df[time_col].max().date()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--start", default="2019-09-01")
    ap.add_argument("--symbols", nargs="*", default=UNIVERSE)
    args = ap.parse_args()

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

    for symbol in args.symbols:
        print(f"\n{symbol}")
        try:
            k = fetch_klines(symbol, args.interval, start_ms)
            if k.empty:
                print("    no klines, skipping")
                continue
            save(k, RAW / f"klines_{args.interval}" / f"{symbol}.parquet", "open_time")

            f = fetch_funding(symbol, start_ms)
            if not f.empty:
                save(f, RAW / "funding" / f"{symbol}.parquet", "funding_time")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("\nDone. Raw data in data/raw/ — treat it as read-only from here on.")


if __name__ == "__main__":
    main()
