#!/usr/bin/env python3
"""
Current TXF EMA signal helper.

Use minute bars for the exact 360/2880-minute signal when available. If only
daily bars are available, print a clearly labelled daily proxy instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAILY_CACHE = Path("/Users/guichenxiang/quant_eval/data/cache/TXF_1998-07-21_2026-06-13_1d.csv")
OLD_MINUTE_CSV = Path("/Users/guichenxiang/Downloads/台指期-1分K-2006~20240515.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current TXF EMA trend signal.")
    parser.add_argument("--minute-csv", type=Path, default=None, help="CSV with Date, Time, Close columns.")
    parser.add_argument("--daily-csv", type=Path, default=DAILY_CACHE, help="Daily TXF CSV fallback.")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-06-14")
    parser.add_argument("--fast-minutes", type=int, default=360)
    parser.add_argument("--slow-minutes", type=int, default=2880)
    parser.add_argument("--band", type=float, default=5.0)
    return parser.parse_args()


def signal_name(diff: float, band: float) -> str:
    if diff > band:
        return "做多"
    if diff < -band:
        return "放空"
    return "空手"


def load_minute(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["Close"] = df["Close"].astype(float)
    return df


def exact_minute_signal(df: pd.DataFrame, args: argparse.Namespace) -> None:
    window_start = pd.Timestamp(args.start)
    window_end = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    if df["datetime"].max() < window_start:
        raise ValueError(f"minute CSV ends at {df['datetime'].max()}, before requested start {window_start}")

    fast = df["Close"].ewm(span=args.fast_minutes, adjust=False).mean()
    slow = df["Close"].ewm(span=args.slow_minutes, adjust=False).mean()
    df = df.assign(fast_ema=fast, slow_ema=slow)
    period = df[(df["datetime"] >= window_start) & (df["datetime"] < window_end)].copy()
    if period.empty:
        raise ValueError("no minute rows in requested period")

    last = period.iloc[-1]
    diff = float(last["fast_ema"] - last["slow_ema"])
    period["signal"] = np.where(
        period["fast_ema"] > period["slow_ema"] + args.band,
        1,
        np.where(period["fast_ema"] < period["slow_ema"] - args.band, -1, 0),
    )
    period["position"] = period["signal"].shift(1).fillna(0)
    period["pnl"] = period["position"] * period["Close"].diff().fillna(0)
    buy_hold = float(period["Close"].iloc[-1] - period["Close"].iloc[0])
    strategy = float(period["pnl"].sum())

    print("資料頻率: 1 分 K，精確 360/2880 分 EMA")
    print(f"期間: {period['datetime'].iloc[0]} ~ {period['datetime'].iloc[-1]}")
    print(f"最後收盤: {last['Close']:.0f}")
    print(f"EMA{args.fast_minutes}: {last['fast_ema']:.2f}")
    print(f"EMA{args.slow_minutes}: {last['slow_ema']:.2f}")
    print(f"快慢線差: {diff:.2f}")
    print(f"目前訊號: {signal_name(diff, args.band)}")
    print(f"期間策略點數粗估: {strategy:.1f}")
    print(f"期間 Buy & Hold 點數: {buy_hold:.1f}")


def daily_proxy_signal(path: Path, args: argparse.Namespace) -> None:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df[(df["Date"] >= pd.Timestamp(args.start)) & (df["Date"] <= pd.Timestamp(args.end))].copy()
    if df.empty:
        raise ValueError("no daily rows in requested period")

    # The old minute file's recent median is about 1129 bars/day. 360 and 2880
    # minutes therefore map roughly to sub-day and 2.6-day EMAs; use 2/5 daily
    # spans as a conservative, readable proxy.
    fast_days = 2
    slow_days = 5
    df["fast_ema"] = df["Close"].ewm(span=fast_days, adjust=False).mean()
    df["slow_ema"] = df["Close"].ewm(span=slow_days, adjust=False).mean()
    df["signal"] = np.where(
        df["fast_ema"] > df["slow_ema"] + args.band,
        1,
        np.where(df["fast_ema"] < df["slow_ema"] - args.band, -1, 0),
    )
    df["position"] = df["signal"].shift(1).fillna(0)
    df["pnl"] = df["position"] * df["Close"].diff().fillna(0)

    last = df.iloc[-1]
    diff = float(last["fast_ema"] - last["slow_ema"])
    buy_hold = float(df["Close"].iloc[-1] - df["Close"].iloc[0])
    strategy = float(df["pnl"].sum())

    print("資料頻率: 日線，這是近似判斷，不是精確 1 分 K 360/2880 EMA")
    print(f"期間: {df['Date'].iloc[0].date()} ~ {df['Date'].iloc[-1].date()}")
    print(f"最後收盤: {last['Close']:.0f}")
    print(f"日線 proxy EMA{fast_days}: {last['fast_ema']:.2f}")
    print(f"日線 proxy EMA{slow_days}: {last['slow_ema']:.2f}")
    print(f"快慢線差: {diff:.2f}")
    print(f"目前近似訊號: {signal_name(diff, args.band)}")
    print(f"期間策略點數粗估: {strategy:.1f}")
    print(f"期間 Buy & Hold 點數: {buy_hold:.1f}")
    print("\n最近 5 根日線:")
    print(df[["Date", "Open", "High", "Low", "Close", "Volume", "fast_ema", "slow_ema"]].tail().to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.minute_csv is not None:
        exact_minute_signal(load_minute(args.minute_csv), args)
        return

    if OLD_MINUTE_CSV.exists():
        old = load_minute(OLD_MINUTE_CSV)
        if old["datetime"].max() >= pd.Timestamp(args.start):
            exact_minute_signal(old, args)
            return

    daily_proxy_signal(args.daily_csv, args)


if __name__ == "__main__":
    main()
