"""
backtest.py — cost-aware backtest engine.

The only two things an engine must get right:
  1. EXECUTION LAG. A signal computed from bar t's close cannot be filled at
     bar t's close. It fills at bar t+1's open, at worst. This is enforced
     structurally here (`positions.shift(1)`) rather than left to discipline,
     because it is the leak that quietly doubles every backtest's Sharpe.
  2. COSTS BEFORE VERDICT. Fees, funding and slippage are subtracted inside
     the engine. There is no gross-return output to be tempted by.

Everything else — signal generation, sizing, exits — is yours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    """Set these from your venue's actual fee schedule, not from optimism."""
    taker_bps: float = 5.0        # per side, basis points of notional
    maker_bps: float = 2.0
    slippage_bps: float = 2.0     # per side; raise it for size or thin books
    use_maker: bool = False       # assume maker fills only if you truly post

    @property
    def per_side_bps(self) -> float:
        fee = self.maker_bps if self.use_maker else self.taker_bps
        return fee + self.slippage_bps


def run_backtest(df: pd.DataFrame, signal: pd.Series, costs: Costs = Costs(),
                 vol_target_ann: float | None = 0.20,
                 vol_lookback: int = 30, max_leverage: float = 2.0,
                 periods_per_year: int = 2190) -> dict:
    """
    df     — must contain 'close'; optionally 'funding_rate' (per settlement).
    signal — desired position in [-1, 1], indexed like df, computed from data
             available AT OR BEFORE each bar's close.

    Returns dict with the net return series, equity curve, and diagnostics.
    """
    df = df.copy()
    close = df["close"].astype(float)
    bar_ret = close.pct_change()

    pos = pd.Series(signal, index=df.index).clip(-1, 1).fillna(0.0)

    # ── volatility targeting ────────────────────────────────────────────────
    # Fixed notional means your risk swings with the vol regime. Scale so each
    # bar carries comparable risk. Uses TRAILING vol only.
    if vol_target_ann:
        realised = bar_ret.rolling(vol_lookback).std() * np.sqrt(periods_per_year)
        scale = (vol_target_ann / realised).replace([np.inf, -np.inf], np.nan)
        pos = (pos * scale.shift(1)).clip(-max_leverage, max_leverage).fillna(0.0)

    # ── THE LAG. Do not remove. ─────────────────────────────────────────────
    held = pos.shift(1).fillna(0.0)

    gross = held * bar_ret

    # ── costs ───────────────────────────────────────────────────────────────
    turnover = held.diff().abs().fillna(held.abs())
    trade_cost = turnover * costs.per_side_bps / 1e4

    if "funding_rate" in df.columns:
        # Funding is paid by longs when positive. Charged only on settlement
        # bars, on the position actually held.
        fr = df["funding_rate"].fillna(0.0)
        settle = fr.ne(fr.shift(1)) & fr.ne(0)
        funding_cost = np.where(settle, held * fr, 0.0)
    else:
        funding_cost = 0.0

    net = (gross - trade_cost - funding_cost).fillna(0.0)
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1

    n_trades = int((turnover > 1e-9).sum())
    return {
        "net_returns": net,
        "gross_returns": gross,
        "equity": equity,
        "drawdown": dd,
        "position": held,
        "stats": {
            "n_bars": len(net),
            "n_trades": n_trades,
            "exposure": float((held.abs() > 1e-9).mean()),
            "total_return": float(equity.iloc[-1] - 1),
            "max_drawdown": float(dd.min()),
            "sharpe_ann": float(net.mean() / net.std(ddof=1)
                                * np.sqrt(periods_per_year)) if net.std() else np.nan,
            "cost_drag_ann_bps": float(
                (trade_cost.mean() + np.mean(funding_cost)) * periods_per_year * 1e4),
            "gross_sharpe_ann": float(gross.mean() / gross.std(ddof=1)
                                      * np.sqrt(periods_per_year)) if gross.std() else np.nan,
        },
    }


def leakage_audit(df: pd.DataFrame, signal: pd.Series, costs: Costs = Costs()) -> None:
    """Shift the signal one bar EARLIER — i.e. deliberately cheat.

    If the cheating version is only slightly better than the honest one, your
    signal is weak. If it is dramatically better, fine — that is expected. But
    if the HONEST version already performs like the cheating one, you have a
    leak somewhere upstream in feature construction.
    """
    honest = run_backtest(df, signal, costs)["stats"]["sharpe_ann"]
    cheat = run_backtest(df, signal.shift(-1), costs)["stats"]["sharpe_ann"]
    print(f"  honest Sharpe : {honest:.3f}")
    print(f"  cheating Sharpe: {cheat:.3f}  (peeks one bar ahead)")
    ratio = cheat / honest if honest else np.inf
    if np.isfinite(ratio) and 0.8 < ratio < 1.25:
        print("  WARNING: cheating barely helps — suspect an existing leak.")
    else:
        print("  OK: lookahead materially changes results, as it should.")
