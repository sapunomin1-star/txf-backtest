#!/usr/bin/env python3
"""
TXF 1-minute EMA trend strategy backtest.

This script compares a simple long/short EMA trend strategy against buying and
holding one continuous TXF contract. PnL is measured in index points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("/Users/guichenxiang/Downloads/台指期-1分K-2006~20240515.csv")


@dataclass(frozen=True)
class Performance:
    name: str
    total_points: float
    max_drawdown_points: float
    trades: int
    win_days_pct: float
    start: str
    end: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest TXF EMA trend strategy.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to TXF 1-minute CSV.")
    parser.add_argument("--fast", type=int, default=360, help="Fast EMA span in minutes.")
    parser.add_argument("--slow", type=int, default=2880, help="Slow EMA span in minutes.")
    parser.add_argument("--band", type=float, default=5.0, help="Flat band around the slow EMA, in points.")
    parser.add_argument(
        "--cost",
        type=float,
        default=1.0,
        help="Transaction cost in points per side. Long to short costs two sides.",
    )
    parser.add_argument("--split-year", type=int, default=2019, help="Out-of-sample split year.")
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/guichenxiang/txf_backtest/output"))
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Date", "Time", "Open", "High", "Low", "Close", "TotalVolume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["Close"] = df["Close"].astype(float)
    return df


def ema_trend_strategy(df: pd.DataFrame, fast: int, slow: int, band: float, cost: float) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("--fast must be smaller than --slow")

    close = df["Close"]
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()

    raw_signal = np.where(
        fast_ema > slow_ema + band,
        1,
        np.where(fast_ema < slow_ema - band, -1, 0),
    )

    result = df[["datetime", "Close"]].copy()
    result["fast_ema"] = fast_ema
    result["slow_ema"] = slow_ema
    result["signal"] = raw_signal.astype(np.int8)

    # Trade on the next bar to avoid using information from the same close.
    result["position"] = result["signal"].shift(1).fillna(0).astype(np.int8)
    result["price_change"] = result["Close"].diff().fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["strategy_pnl"] = result["position"] * result["price_change"] - cost * result["turnover"]
    result["buy_hold_pnl"] = result["price_change"]
    result["strategy_equity"] = result["strategy_pnl"].cumsum()
    result["buy_hold_equity"] = result["buy_hold_pnl"].cumsum()
    return result


def summarize(name: str, bt: pd.DataFrame, pnl_col: str, position_col: str | None = None) -> Performance:
    equity = bt[pnl_col].cumsum()
    drawdown = equity - equity.cummax()
    daily_pnl = bt.groupby(bt["datetime"].dt.date)[pnl_col].sum()
    trades = 0
    if position_col is not None:
        trades = int(bt[position_col].diff().abs().fillna(bt[position_col].abs()).gt(0).sum())

    return Performance(
        name=name,
        total_points=float(equity.iloc[-1]),
        max_drawdown_points=float(drawdown.min()),
        trades=trades,
        win_days_pct=float((daily_pnl > 0).mean() * 100),
        start=str(bt["datetime"].iloc[0]),
        end=str(bt["datetime"].iloc[-1]),
    )


def print_performance(perfs: list[Performance]) -> None:
    rows = [
        {
            "name": p.name,
            "total_points": round(p.total_points, 1),
            "max_drawdown_points": round(p.max_drawdown_points, 1),
            "trades": p.trades,
            "win_days_pct": round(p.win_days_pct, 2),
            "start": p.start,
            "end": p.end,
        }
        for p in perfs
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def write_equity_chart(daily: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped equity chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = pd.to_datetime(daily["date"])
    ax.plot(x, daily["strategy_equity"], label="EMA trend", linewidth=1.5)
    ax.plot(x, daily["buy_hold_equity"], label="Buy & Hold", linewidth=1.2)
    ax.set_title("TXF EMA trend strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Points")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)
    bt = ema_trend_strategy(df, args.fast, args.slow, args.band, args.cost)

    all_perfs = [
        summarize("EMA trend", bt, "strategy_pnl", "position"),
        summarize("Buy & Hold", bt, "buy_hold_pnl"),
    ]

    train = bt[bt["datetime"].dt.year < args.split_year].copy()
    test = bt[bt["datetime"].dt.year >= args.split_year].copy()
    split_perfs = [
        summarize(f"EMA trend < {args.split_year}", train, "strategy_pnl", "position"),
        summarize(f"Buy & Hold < {args.split_year}", train, "buy_hold_pnl"),
        summarize(f"EMA trend >= {args.split_year}", test, "strategy_pnl", "position"),
        summarize(f"Buy & Hold >= {args.split_year}", test, "buy_hold_pnl"),
    ]

    print(f"CSV: {args.csv}")
    print(f"Strategy: fast EMA={args.fast}, slow EMA={args.slow}, band={args.band}, cost={args.cost} point/side")
    print("\nFull period")
    print_performance(all_perfs)
    print(f"\nSplit by {args.split_year}")
    print_performance(split_perfs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = (
        bt.assign(date=bt["datetime"].dt.date)
        .groupby("date", as_index=False)[["strategy_pnl", "buy_hold_pnl"]]
        .sum()
    )
    daily["strategy_equity"] = daily["strategy_pnl"].cumsum()
    daily["buy_hold_equity"] = daily["buy_hold_pnl"].cumsum()
    daily.to_csv(args.out_dir / "daily_equity.csv", index=False)
    write_equity_chart(daily, args.out_dir / "equity_curve.png")

    bt[
        [
            "datetime",
            "Close",
            "fast_ema",
            "slow_ema",
            "signal",
            "position",
            "strategy_pnl",
            "buy_hold_pnl",
            "strategy_equity",
            "buy_hold_equity",
        ]
    ].to_csv(args.out_dir / "minute_backtest.csv", index=False)
    print(f"\nWrote: {args.out_dir / 'daily_equity.csv'}")
    print(f"Wrote: {args.out_dir / 'equity_curve.png'}")
    print(f"Wrote: {args.out_dir / 'minute_backtest.csv'}")


if __name__ == "__main__":
    main()
