#!/usr/bin/env python3
"""Shioaji 逐合約 TXF 1 分 K 下載器（使用者在自己終端機執行）。

用途：
  1. 探測目前掛牌的 TXF 合約與各自 kbar 可回溯深度（含 TXFR1/TXFR2 連續檔）。
  2. 把每口掛牌合約的 1 分 K 存成獨立檔案 shioaji_data/per_contract/TXF{YYYYMM}_1min.csv。
  3. 每月換約後重跑一次＝隨時間累積出「有 contract_id 的分鐘級檔案庫」（Shioaji 抓不到已下市合約，
     歷史部分由 FinMind 日線指紋法補：見 taifutures_strategy/contract_fingerprint_rebuild.py）。

金鑰：只從環境變數讀 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY（不硬寫、不列印、不存檔）。
用法（在你自己的終端機）：
    export SHIOAJI_API_KEY="..."; export SHIOAJI_SECRET_KEY="..."
    /Users/guichenxiang/txf_backtest/.venv-shioaji/bin/python shioaji_per_contract_download.py --probe   # 只探測
    /Users/guichenxiang/txf_backtest/.venv-shioaji/bin/python shioaji_per_contract_download.py           # 探測+下載
只抓歷史行情，不下單、不啟用 CA。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

OUT_DIR = Path("/Users/guichenxiang/txf_backtest/shioaji_data/per_contract")
CHUNK_DAYS = 25          # Shioaji kbars 單次區間上限 30 天
PROBE_BACK_MONTHS = 30   # 深度探測最多往回試 30 個月


def login():
    import shioaji as sj

    api_key = os.environ.get("SHIOAJI_API_KEY")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY")
    if not api_key or not secret_key:
        print("請先在『你自己的終端機』export SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY 再執行。", file=sys.stderr)
        sys.exit(1)
    api = sj.Shioaji()
    api.login(api_key=api_key, secret_key=secret_key, contracts_timeout=30000)
    return api


def txf_contracts(api) -> list:
    """掛牌中的 TXF 月合約 + 連續檔 R1/R2。"""
    out = []
    for c in api.Contracts.Futures.TXF:
        out.append(c)
    return out


def fetch_range(api, contract, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    frames = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur <= e:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), e)
        for attempt in range(retries):
            try:
                k = api.kbars(contract, start=cur.strftime("%Y-%m-%d"), end=chunk_end.strftime("%Y-%m-%d"))
                df = pd.DataFrame({**k})
                if len(df):
                    frames.append(df)
                break
            except Exception as exc:  # 網路/限流重試
                print(f"    retry {attempt+1}: {type(exc).__name__}", file=sys.stderr)
                time.sleep(3.0 * (attempt + 1))
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def probe_depth(api, contract) -> str:
    """往回逐月試抓 3 天，找出最早有資料的月份。"""
    today = pd.Timestamp.today().normalize()
    earliest = None
    for back in range(PROBE_BACK_MONTHS):
        t0 = (today - pd.DateOffset(months=back)).replace(day=1)
        try:
            k = api.kbars(contract, start=t0.strftime("%Y-%m-%d"), end=(t0 + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
            if len(pd.DataFrame({**k})):
                earliest = t0
            else:
                if earliest is not None:
                    break
        except Exception:
            break
        time.sleep(0.2)
    return earliest.strftime("%Y-%m") if earliest is not None else "無資料"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shioaji TXF per-contract 1min downloader")
    parser.add_argument("--probe", action="store_true", help="只探測深度，不下載")
    parser.add_argument("--start", default=None, help="下載起日（預設=探測到的最早月）")
    args = parser.parse_args()

    api = login()
    try:
        contracts = txf_contracts(api)
        print(f"掛牌 TXF 合約 {len(contracts)} 口：")
        rows = []
        for c in contracts:
            depth = probe_depth(api, c)
            rows.append((c.code, getattr(c, "delivery_month", ""), depth))
            print(f"  {c.code:10} 交割月 {getattr(c, 'delivery_month', ''):8} kbar 最早 ≈ {depth}")
        if args.probe:
            usage = api.usage()
            print(f"\nAPI 用量: {usage}")
            return

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        for c, (code, dm, depth) in zip(contracts, rows):
            if depth == "無資料":
                continue
            start = args.start or f"{depth}-01"
            out = OUT_DIR / f"{code}_1min.csv"
            if out.exists():  # append-safe：從既有檔尾之後續抓
                old = pd.read_csv(out, parse_dates=["ts"])
                start = (old["ts"].max().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"下載 {code} {start} → {today} ...")
            df = fetch_range(api, c, start, today)
            if df.empty:
                print("  （無新資料）")
                continue
            if out.exists():
                df = pd.concat([old, df], ignore_index=True).drop_duplicates("ts").sort_values("ts")
            df.to_csv(out, index=False)
            print(f"  存 {len(df):,} 根 → {out}")
        usage = api.usage()
        print(f"\nAPI 用量: {usage}")
    finally:
        api.logout()


if __name__ == "__main__":
    main()
