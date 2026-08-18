#!/usr/bin/env python3
"""
Download TXF continuous near-month 1-minute Kbars from Shioaji.

Credentials are read from environment variables:

    SHIOAJI_API_KEY
    SHIOAJI_SECRET_KEY

This script only fetches historical market data. It does not place orders and
does not activate CA.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_OUT = Path("/Users/guichenxiang/txf_backtest/shioaji_data/TXFR1_1min.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Shioaji TXFR1 1-minute Kbars.")
    parser.add_argument("--start", default="2024-05-17", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-days", type=int, default=7, help="Download in date chunks; Shioaji requires <= 30 days.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk on timeout/transient errors.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Seconds to sleep between retries.")
    parser.add_argument("--simulation", action="store_true", help="Use Shioaji simulation mode.")
    return parser.parse_args()


def require_shioaji():
    try:
        import shioaji as sj
    except ModuleNotFoundError:
        print(
            "Missing package: shioaji\n"
            "Install it with:\n"
            "  python3 -m pip install -r /Users/guichenxiang/txf_backtest/requirements_shioaji.txt",
            file=sys.stderr,
        )
        raise
    return sj


def date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_days > 30:
        raise ValueError("Shioaji Kbars date range must not exceed 30 days; use --chunk-days <= 30.")
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("--start must be <= --end")

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_ts
    while current <= end_ts:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def kbars_to_frame(kbars) -> pd.DataFrame:
    if hasattr(kbars, "items"):
        data = {key: value for key, value in kbars.items()}
    else:
        data = {}
        for key in ("ts", "Open", "High", "Low", "Close", "Volume", "Amount"):
            if hasattr(kbars, key):
                data[key] = getattr(kbars, key)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "ts" not in df.columns:
        raise ValueError(f"Unexpected Shioaji kbars fields: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["ts"])
    # Shioaji may return timezone-naive local timestamps or nanosecond epoch-like
    # timestamps depending on SDK version; pandas handles both in current docs.
    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "TotalVolume",
        }
    )
    keep = ["datetime", "Open", "High", "Low", "Close", "TotalVolume"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"Missing Kbar columns: {missing}; got {list(df.columns)}")
    return df[keep].copy()


def login(api, api_key: str, secret_key: str) -> None:
    accounts = api.login(api_key=api_key, secret_key=secret_key)
    if not accounts:
        raise RuntimeError("Shioaji login returned no accounts.")


def fetch_kbars_with_retry(api, contract, start: str, end: str, retries: int, retry_sleep: float):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return api.kbars(contract=contract, start=start, end=end)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"  attempt {attempt}/{retries} failed: {exc}. retrying in {retry_sleep:g}s ...")
            time.sleep(retry_sleep)
    raise last_error


def main() -> None:
    args = parse_args()
    sj = require_shioaji()

    api_key = os.environ.get("SHIOAJI_API_KEY")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "Please set SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY environment variables first.\n"
            "Do not paste secrets into chat or commit them to files."
        )

    end = args.end or pd.Timestamp.today(tz="Asia/Taipei").strftime("%Y-%m-%d")
    api = sj.Shioaji(simulation=args.simulation)
    try:
        login(api, api_key, secret_key)
        contract = api.Contracts.Futures.TXF.TXFR1
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in date_chunks(args.start, end, args.chunk_days):
            print(f"Downloading TXFR1 {chunk_start.date()} ~ {chunk_end.date()} ...")
            kbars = fetch_kbars_with_retry(
                api=api,
                contract=contract,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            frame = kbars_to_frame(kbars)
            if not frame.empty:
                frames.append(frame)
                print(f"  rows: {len(frame):,}")
            else:
                print("  rows: 0")

        if not frames:
            raise RuntimeError("No Kbar rows returned.")

        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        out["Date"] = out["datetime"].dt.strftime("%Y/%-m/%-d")
        out["Time"] = out["datetime"].dt.strftime("%H:%M:%S")
        out = out[["Date", "Time", "Open", "High", "Low", "Close", "TotalVolume", "datetime"]]

        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}")
        print(f"Rows: {len(out):,}")
        print(f"Range: {out['datetime'].iloc[0]} ~ {out['datetime'].iloc[-1]}")
    finally:
        try:
            api.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
