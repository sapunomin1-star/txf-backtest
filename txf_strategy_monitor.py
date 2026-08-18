#!/usr/bin/env python3
"""
TXF strategy monitor using TAIFEX recent tick downloads.

2026-07-10 稽核後的可執行核心：

    現最佳    = EMA480/2160 做多 x (20日波動>15% -> 0.5)

舊基差空手腿因跨連假訊號映射錯誤已撤銷；此工具只把基差列為診斷，不再納入目標曝險。
基差 = (期貨日盤收盤 - TAIEX 現貨收盤)/現貨; 現貨走 FinMind(免費)並快取。
舊 RULES 表 (0.5+1.0xEMA 世代) 保留作研究對照。監控/研究工具, 非下單引擎。
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TAIFEX_PREV30_URL = "https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData"
DAILY_CACHE = Path("/Users/guichenxiang/quant_eval/data/cache/TXF_1998-07-21_2026-06-13_1d.csv")
DEFAULT_WORK_DIR = Path("/Users/guichenxiang/txf_backtest/taifex_prev30")
SPOT_CACHE = Path("/Users/guichenxiang/txf_backtest/finmind_data/taiex_daily.csv")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


@dataclass(frozen=True)
class ExposureRule:
    name: str
    base: float
    addon: float


RULES = [
    ExposureRule("Buy & Hold", 1.0, 0.0),
    ExposureRule("純 EMA long/flat", 0.0, 1.0),
    ExposureRule("核心0.5 + EMA{0.5,1.5}", 0.5, 1.0),
    ExposureRule("原加碼版 1+0.5xEMA", 1.0, 0.5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor TXF EMA exposure strategies.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--minute-csv",
        type=Path,
        default=None,
        help="Optional existing 1-minute CSV. When set, skip TAIFEX download/parse.",
    )
    parser.add_argument("--daily-csv", type=Path, default=DAILY_CACHE)
    parser.add_argument("--fast", type=int, default=480, help="現最佳=480 (舊 360)。")
    parser.add_argument("--slow", type=int, default=2160, help="現最佳=2160 (舊 2880)。")
    parser.add_argument("--band", type=float, default=5.0)
    parser.add_argument("--cost", type=float, default=1.5, help="Cost in points per 1.0 exposure change.")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--output-tail-days", type=int, default=60, help="Only write recent signal rows to CSV.")
    parser.add_argument("--monthly-stop", type=float, default=-0.08, help="Research-only monthly stop threshold.")
    parser.add_argument("--skip-monthly-stop", action="store_true", help="Skip monthly stop research table.")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached TAIFEX zip files only.")
    parser.add_argument("--spot-cache", type=Path, default=SPOT_CACHE, help="TAIEX 現貨日線快取 CSV。")
    parser.add_argument("--basis-window", type=int, default=90, help="基差 z 滾動視窗(交易日)。")
    parser.add_argument("--basis-trigger", type=float, default=-2.0, help="基差 z 低於此值→空手 (O4c)。")
    parser.add_argument("--roll-clean-days", type=int, default=3, help="換月日起 N 交易日內不觸發(股利假訊號清洗)。")
    parser.add_argument("--vol-threshold", type=float, default=0.15, help="20日年化波動高於此→減半。")
    parser.add_argument("--skip-basis", action="store_true", help="跳過基差診斷區塊(離線且無現貨快取時)。")
    parser.add_argument("--max-data-age-days", type=int, default=4, help="最新 K 棒最多可落後幾個曆日。")
    parser.add_argument("--allow-stale-data", action="store_true", help="研究舊檔時允許過期資料；不應用於日常監控。")
    return parser.parse_args()


def fetch_taifex_zip_urls() -> list[str]:
    import requests

    html = requests.get(TAIFEX_PREV30_URL, timeout=30).text
    pattern = r"https://www\.taifex\.com\.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_\d{4}_\d{2}_\d{2}\.zip"
    return sorted(set(re.findall(pattern, html)))


def download_zips(raw_dir: Path, refresh: bool) -> list[Path]:
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = fetch_taifex_zip_urls()
    session = requests.Session()
    paths: list[Path] = []
    for url in urls:
        path = raw_dir / url.rsplit("/", 1)[1]
        if refresh and (not path.exists() or path.stat().st_size < 1000):
            response = session.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        if path.exists():
            paths.append(path)
    return sorted(paths)


def read_txf_ticks(zip_paths: list[Path]) -> pd.DataFrame:
    cols = ["成交日期", "商品代號", "到期月份(週別)", "成交時間", "成交價格", "成交數量(B+S)"]
    frames: list[pd.DataFrame] = []

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            day_frames: list[pd.DataFrame] = []
            with archive.open(csv_name) as file_obj:
                for chunk in pd.read_csv(file_obj, encoding="big5", usecols=cols, dtype=str, chunksize=200_000):
                    chunk["商品代號"] = chunk["商品代號"].str.strip()
                    tx = chunk[chunk["商品代號"].eq("TX")].copy()
                    if tx.empty:
                        continue
                    tx["到期月份(週別)"] = tx["到期月份(週別)"].str.strip()
                    tx = tx[tx["到期月份(週別)"].str.fullmatch(r"\d{6}", na=False)]
                    if not tx.empty:
                        day_frames.append(tx)

            if not day_frames:
                continue

            day = pd.concat(day_frames, ignore_index=True)
            nearest_month = sorted(day["到期月份(週別)"].unique())[0]
            frames.append(day[day["到期月份(週別)"].eq(nearest_month)].copy())

    if not frames:
        raise RuntimeError("No TX monthly-contract ticks were found in TAIFEX zip files.")

    ticks = pd.concat(frames, ignore_index=True)
    ticks["成交日期"] = ticks["成交日期"].str.strip()
    ticks["成交時間"] = ticks["成交時間"].str.strip().str.zfill(6)
    ticks["datetime"] = pd.to_datetime(
        ticks["成交日期"] + ticks["成交時間"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    ticks["price"] = pd.to_numeric(ticks["成交價格"].str.strip(), errors="coerce")
    ticks["volume"] = pd.to_numeric(ticks["成交數量(B+S)"].str.strip(), errors="coerce").fillna(0.0) / 2.0
    ticks = ticks.dropna(subset=["datetime", "price"]).sort_values("datetime")
    return ticks[~ticks.duplicated(subset=["datetime", "price", "volume", "到期月份(週別)"], keep="last")]


def ticks_to_minute_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bars = (
        ticks.set_index("datetime")
        .resample("1min")
        .agg(
            Open=("price", "first"),
            High=("price", "max"),
            Low=("price", "min"),
            Close=("price", "last"),
            Volume=("volume", "sum"),
        )
        .dropna(subset=["Close"])
        .reset_index()
    )
    return bars


def load_minute_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif {"Date", "Time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    else:
        raise ValueError("Minute CSV must contain either datetime or Date+Time columns.")

    required = ["datetime", "Open", "High", "Low", "Close", "TotalVolume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Minute CSV missing columns: {missing}")

    bars = df[required].copy()
    bars = bars.rename(columns={"TotalVolume": "Volume"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = bars.dropna(subset=["datetime", "Close"]).drop_duplicates(subset=["datetime"], keep="last")
    return bars.sort_values("datetime").reset_index(drop=True)


def add_ema_signal(bars: pd.DataFrame, fast: int, slow: int, band: float) -> pd.DataFrame:
    out = bars.copy()
    out["fast_ema"] = out["Close"].ewm(span=fast, adjust=False).mean()
    out["slow_ema"] = out["Close"].ewm(span=slow, adjust=False).mean()
    out["ema_long_signal"] = (out["fast_ema"] > out["slow_ema"] + band).astype(float)
    out["ema_position"] = out["ema_long_signal"].shift(1).fillna(0.0)
    out["price_change"] = out["Close"].diff().fillna(0.0)
    return out


def third_wednesday(ts: pd.Timestamp) -> pd.Timestamp:
    first = pd.Timestamp(ts.year, ts.month, 1)
    first_wed = first + pd.Timedelta(days=(2 - first.weekday()) % 7)
    return first_wed + pd.Timedelta(days=14)


def fetch_spot_daily(cache_path: Path, refresh: bool) -> tuple[pd.Series, str]:
    """TAIEX 現貨日收盤: 快取 + FinMind 增量更新; 離線退回快取。回傳 (日期→close, 來源註記)。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
    note = f"快取 (至 {cached['date'].max().date()})" if not cached.empty else "無快取"
    if refresh:
        try:
            import json
            import urllib.parse
            import urllib.request

            start = "2005-01-01" if cached.empty else (cached["date"].max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            query = urllib.parse.urlencode({
                "dataset": "TaiwanStockPrice",
                "data_id": "TAIEX",
                "start_date": start,
                "end_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
            })
            with urllib.request.urlopen(f"{FINMIND_URL}?{query}", timeout=30) as resp:
                data = json.loads(resp.read().decode()).get("data", [])
            if data:
                fresh = pd.DataFrame(data)
                fresh["date"] = pd.to_datetime(fresh["date"])
                cached = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
                cached = cached.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                cached.to_csv(cache_path, index=False)
                note = f"FinMind 更新至 {cached['date'].max().date()}"
        except Exception as exc:
            note = f"FinMind 失敗({type(exc).__name__}), 退回{note}"
    if cached.empty:
        raise RuntimeError("無 TAIEX 現貨資料 (快取空且抓取失敗); 可用 --skip-basis 略過。")
    spot = cached.set_index("date")["close"].astype(float).sort_index()
    return spot[~spot.index.duplicated(keep="last")], note


def futures_daily_series(bars: pd.DataFrame, daily_csv: Path) -> tuple[pd.Series, pd.Series]:
    """回傳 (13:45 日盤收盤 [算基差用], 日末收盤 [算20日波動用, 同回測 groupby-last 慣例])。
    以 bars 為準, 較舊歷史用日線快取補 (prev30 模式 z 視窗才夠長)。"""
    t = bars["datetime"].dt
    minutes = t.hour * 60 + t.minute
    dates = t.normalize()
    day_mask = (minutes >= 8 * 60 + 45) & (minutes <= 13 * 60 + 45)
    day_bars = bars.loc[day_mask, ["datetime", "Close"]].copy()
    day_dates = day_bars["datetime"].dt.normalize()
    close_1345 = day_bars.groupby(day_dates)["Close"].last()
    last_stamp = day_bars.groupby(day_dates)["datetime"].max()
    last_minute = last_stamp.dt.hour * 60 + last_stamp.dt.minute
    complete = pd.Series(False, index=last_stamp.index)
    for date, minute in last_minute.items():
        # 到期月最後交易日收至 13:30；一般日收至 13:45。容許末筆少 5 分鐘。
        required = 13 * 60 + (25 if date == third_wednesday(date) else 40)
        complete.loc[date] = minute >= required
    close_1345 = close_1345[complete]
    close_eod = bars.groupby(dates)["Close"].last()
    if daily_csv and Path(daily_csv).exists():
        daily = pd.read_csv(daily_csv, parse_dates=["Date"])
        hist = daily.set_index(daily["Date"].dt.normalize())["Close"].astype(float)
        hist = hist[~hist.index.duplicated(keep="last")]
        older = hist[~hist.index.isin(close_1345.index)]
        close_1345 = pd.concat([older, close_1345]).sort_index()
        older_eod = hist[~hist.index.isin(close_eod.index)]
        close_eod = pd.concat([older_eod, close_eod]).sort_index()
    return close_1345, close_eod


def vol20_from_bars(bars: pd.DataFrame) -> pd.Series:
    """20 日年化波動, bar 級複製回測慣例: 只中和換月『那一根』(session 首根且跨 3rd-Wed), 再取日末收盤。"""
    dt_col = bars["datetime"]
    dates = dt_col.dt.normalize()
    gap_minutes = dt_col.diff().dt.total_seconds().div(60).fillna(0)
    session_start = gap_minutes > 30
    tw_map = {d: third_wednesday(d) for d in dates.drop_duplicates()}
    twd = dates.map(tw_map)
    roll_bar = session_start & (dates >= twd) & (dates.shift(1) < twd.shift(1)).fillna(False) & dates.shift(1).notna()
    ret = bars["Close"].pct_change().fillna(0.0)
    ret[roll_bar] = 0.0
    adj_close = float(bars["Close"].iloc[0]) * (1 + ret).cumprod()
    daily_close = adj_close.groupby(dates).last()
    return daily_close.pct_change().rolling(20).std() * np.sqrt(252)


def basis_candidate_status(
    close_1345: pd.Series,
    close_eod: pd.Series,
    spot: pd.Series,
    window: int,
    trigger: float,
    roll_clean_days: int,
    vol_threshold: float,
    vol20_override: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """基差 z (O4c: z<trigger 且不在換月清洗窗 → 空手) + 20日波動 (優先用 bar 級回測慣例)。"""
    common = close_1345.index.intersection(spot.index)
    if len(common) < window + 5:
        raise RuntimeError(f"基差共同交易日僅 {len(common)} 天 < 視窗 {window}+5; 請用 --minute-csv 合併檔或補日線快取。")
    fut = close_1345[common]
    basis = (fut - spot[common]) / spot[common] * 100.0
    z = (basis - basis.rolling(window).mean()) / basis.rolling(window).std()
    idx = basis.index
    is_roll = pd.Series(False, index=idx)
    for _, grp in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        tw_day = third_wednesday(grp.iloc[0])
        after = grp[grp >= tw_day]
        if len(after):
            is_roll.loc[after.iloc[0]] = True
    in_roll_window = is_roll.rolling(roll_clean_days, min_periods=1).max().astype(bool)
    trig = (z < trigger) & ~in_roll_window
    if vol20_override is not None and vol20_override.notna().any():
        vol20 = vol20_override
    else:
        vol20 = close_eod.pct_change().rolling(20).std() * np.sqrt(252)  # 粗 fallback (bars 太短時)
    table = pd.DataFrame({
        "fut_1345": fut,
        "spot": spot[common],
        "basis_pct": basis,
        "basis_z": z,
        "in_roll_window": in_roll_window,
        "basis_flat": trig,
    })
    latest = {
        "date": idx[-1],
        "fut": float(fut.iloc[-1]),
        "spot": float(spot[common].iloc[-1]),
        "basis": float(basis.iloc[-1]),
        "z": float(z.iloc[-1]),
        "in_roll_window": bool(in_roll_window.iloc[-1]),
        "basis_flat": bool(trig.iloc[-1]),
        "vol20": float(vol20.dropna().iloc[-1]) if vol20.notna().any() else float("nan"),
        "spot_lag": bool(close_1345.index[-1] > idx[-1]),
        "fut_last_date": close_1345.index[-1],
    }
    return table, latest


def load_donchian_status(daily_csv: Path, last_dt: pd.Timestamp) -> tuple[str, float, float, float]:
    daily = pd.read_csv(daily_csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    daily = daily[daily["Date"] <= last_dt.normalize()].copy()
    if daily.empty:
        return "unknown", np.nan, np.nan, np.nan

    daily["hh50"] = daily["High"].shift(1).rolling(50).max()
    daily["ll50"] = daily["Low"].shift(1).rolling(50).min()
    in_market = True
    states = []
    for _, row in daily.iterrows():
        if pd.notna(row["ll50"]) and row["Close"] < row["ll50"]:
            in_market = False
        elif pd.notna(row["hh50"]) and row["Close"] > row["hh50"]:
            in_market = True
        states.append(in_market)

    daily["donchian_in_market"] = states
    last = daily.iloc[-1]
    status = "持有" if bool(last["donchian_in_market"]) else "避險/空手"
    return status, float(last["Close"]), float(last["hh50"]), float(last["ll50"])


def apply_monthly_stop(
    bars: pd.DataFrame,
    base: float,
    addon: float,
    threshold: float,
    start_price: float,
) -> tuple[pd.Series, pd.Series]:
    """Disable the EMA add-on after monthly return breaches threshold."""
    exposures = []
    stopped_flags = []
    stopped = False
    current_month = None
    month_pnl = 0.0

    for row in bars.itertuples(index=False):
        month = row.datetime.to_period("M")
        if month != current_month:
            current_month = month
            stopped = False
            month_pnl = 0.0

        exposure = base if stopped else base + addon * row.ema_position
        pnl = exposure * row.price_change
        month_pnl += pnl
        exposures.append(exposure)
        if month_pnl / start_price <= threshold:
            # 觸發當根損失必須保留；下一根才切到 base。
            stopped = True
        stopped_flags.append(stopped)

    return (
        pd.Series(exposures, index=bars.index, dtype=float),
        pd.Series(stopped_flags, index=bars.index, dtype=bool),
    )


def summarize_rule(
    bars: pd.DataFrame,
    rule: ExposureRule,
    cost: float,
    eval_mask: pd.Series,
    monthly_stop: float | None = None,
) -> dict[str, float | str]:
    # 當前目標曝險用「最新收盤的未 shift 訊號」, 與印出的 EMA 訊號一致;
    # shift 過的 exposure 只拿來算歷史 P&L (避免用當根訊號交易當根).
    current_signal = float(bars["ema_long_signal"].iloc[-1])
    if monthly_stop is None:
        exposure = rule.base + rule.addon * bars["ema_position"]
        stop_triggered = "N/A"
        current_exposure = rule.base + rule.addon * current_signal
    else:
        eval_bars = bars.loc[eval_mask].copy()
        start_price = float(eval_bars["Close"].iloc[0])
        exposure, stopped_flags = apply_monthly_stop(eval_bars, rule.base, rule.addon, monthly_stop, start_price)
        stop_triggered = "YES" if bool(stopped_flags.any()) else "NO"
        current_exposure = rule.base if bool(stopped_flags.iloc[-1]) else rule.base + rule.addon * current_signal
        gross_points = float((exposure * eval_bars["price_change"]).sum())
        turnover = exposure.diff().abs().fillna(exposure.abs())
        cost_points = float(turnover.sum() * cost)
        net_points = gross_points - cost_points
        equity = (exposure * eval_bars["price_change"] - turnover * cost).cumsum()
        return {
            "策略": rule.name,
            "今日曝險": float(current_exposure),
            "平均曝險": float(exposure.mean()),
            "換倉成本點數": cost_points,
            "淨報酬點數": net_points,
            "淨報酬率": net_points / start_price,
            "區間最大回撤點數": float((equity - equity.cummax()).min()) if len(equity) else 0.0,
            "月停損觸發": stop_triggered,
        }

    exposure = pd.Series(exposure, index=bars.index, dtype=float)
    gross_points = float((exposure * bars["price_change"])[eval_mask].sum())
    turnover = exposure.diff().abs().fillna(exposure.abs())
    cost_points = float(turnover[eval_mask].sum() * cost)
    net_points = gross_points - cost_points
    start_price = float(bars.loc[eval_mask, "Close"].iloc[0])
    equity = (exposure * bars["price_change"] - turnover * cost).loc[eval_mask].cumsum()
    max_drawdown = float((equity - equity.cummax()).min()) if len(equity) else 0.0
    return {
        "策略": rule.name,
        "今日曝險": float(current_exposure),
        "平均曝險": float(exposure[eval_mask].mean()),
        "換倉成本點數": cost_points,
        "淨報酬點數": net_points,
        "淨報酬率": net_points / start_price,
        "區間最大回撤點數": max_drawdown,
        "月停損觸發": stop_triggered,
    }


def write_outputs(
    work_dir: Path,
    bars: pd.DataFrame,
    summary: pd.DataFrame,
    stop_summary: pd.DataFrame | None,
    output_tail_days: int,
) -> None:
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_tail_days > 0:
        cutoff = pd.Timestamp(bars["datetime"].iloc[-1]) - pd.Timedelta(days=output_tail_days)
        bars_to_write = bars[bars["datetime"] >= cutoff].copy()
    else:
        bars_to_write = bars
    bars_to_write.to_csv(out_dir / "txf_recent_1min_with_signals.csv", index=False)
    summary.to_csv(out_dir / "strategy_monitor_summary.csv", index=False)
    if stop_summary is not None:
        stop_summary.to_csv(out_dir / "monthly_stop_research.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.minute_csv is not None:
        bars = add_ema_signal(load_minute_csv(args.minute_csv), args.fast, args.slow, args.band)
    else:
        raw_dir = args.work_dir / "raw"
        zip_paths = download_zips(raw_dir, refresh=not args.no_refresh)
        ticks = read_txf_ticks(zip_paths)
        bars = add_ema_signal(ticks_to_minute_bars(ticks), args.fast, args.slow, args.band)

    last_dt = pd.Timestamp(bars["datetime"].iloc[-1])
    data_age = (pd.Timestamp.now().normalize() - last_dt.normalize()).days
    if data_age > args.max_data_age_days and not args.allow_stale_data:
        raise RuntimeError(
            f"STALE DATA: 最新 K 棒 {last_dt} 已落後 {data_age} 天；"
            "先更新資料，研究舊檔才可加 --allow-stale-data。"
        )
    eval_start = last_dt - pd.Timedelta(days=args.lookback_days)
    eval_mask = bars["datetime"] >= eval_start
    if not eval_mask.any():
        raise RuntimeError("No bars in evaluation window.")

    rows = [summarize_rule(bars, rule, args.cost, eval_mask) for rule in RULES]
    summary = pd.DataFrame(rows)
    stop_summary = None
    if not args.skip_monthly_stop:
        stop_rows = [summarize_rule(bars, rule, args.cost, eval_mask, monthly_stop=args.monthly_stop) for rule in RULES]
        stop_summary = pd.DataFrame(stop_rows)

    donchian_status, daily_close, hh50, ll50 = load_donchian_status(args.daily_csv, last_dt)
    write_outputs(args.work_dir, bars, summary, stop_summary, args.output_tail_days)

    last = bars.iloc[-1]
    print(f"資料區間: {bars['datetime'].iloc[0]} ~ {last_dt}")
    print(f"評估區間: {bars.loc[eval_mask, 'datetime'].iloc[0]} ~ {last_dt}")
    print(f"最後 1 分 K close: {last['Close']:.0f}")
    print(f"EMA{args.fast}: {last['fast_ema']:.2f}")
    print(f"EMA{args.slow}: {last['slow_ema']:.2f}")
    print(f"EMA 訊號: {'多' if last['ema_long_signal'] > 0 else '空手'}")
    print(f"Donchian 50: {donchian_status} (日線 close={daily_close:.0f}, hh50={hh50:.0f}, ll50={ll50:.0f})")

    if not args.skip_basis:
        try:
            spot, spot_note = fetch_spot_daily(args.spot_cache, refresh=not args.no_refresh)
            close_1345, close_eod = futures_daily_series(bars, args.daily_csv)
            vol_override = vol20_from_bars(bars) if bars["datetime"].dt.normalize().nunique() >= 25 else None
            basis_table, bs = basis_candidate_status(
                close_1345, close_eod, spot,
                args.basis_window, args.basis_trigger, args.roll_clean_days, args.vol_threshold,
                vol20_override=vol_override,
            )
            ema_sig = float(last["ema_long_signal"])
            vol_mult = 0.5 if bs["vol20"] > args.vol_threshold else 1.0
            best = ema_sig * vol_mult
            print("\n=== 核心策略 + 基差診斷（基差腿已撤銷，不納入目標） ===")
            print(f"現貨資料: {spot_note}")
            if bs["spot_lag"]:
                print(f"⚠ 現貨落後期貨 (期貨最新 {bs['fut_last_date'].date()}, 基差只算到 {bs['date'].date()})")
            print(f"基差基準日 {bs['date'].date()}: 期貨13:45 {bs['fut']:.0f} vs 現貨 {bs['spot']:.0f} → 基差 {bs['basis']:+.2f}%")
            print(f"基差 z({args.basis_window}d): {bs['z']:+.2f} | 換月清洗窗: {'是' if bs['in_roll_window'] else '否'} | 舊研究條件(z<{args.basis_trigger:g}): {'是' if bs['basis_flat'] else '否'}（僅診斷）")
            print(f"20日波動(年化, 換月中和): {bs['vol20']*100:.1f}% ({'>' if vol_mult == 0.5 else '≤'}{args.vol_threshold*100:.0f}% → {'減半' if vol_mult == 0.5 else '全額'})")
            print(f"目標曝險(依最新收盤, 未shift): G4 核心 {best:.2f}x；基差不調整部位")
            recent = basis_table.tail(10).copy()
            recent.index = recent.index.strftime("%m-%d")
            print("近 10 日基差 z:")
            print(recent[["basis_pct", "basis_z", "in_roll_window", "basis_flat"]].to_string(
                float_format=lambda v: f"{v:+.2f}"))
            out_dir = args.work_dir / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            basis_table.tail(120).to_csv(out_dir / "basis_z_recent.csv")
        except Exception as exc:
            print(f"\n[基差z] 略過: {exc}")

    print("\n主策略表")
    print(summary.to_string(index=False, formatters={"淨報酬率": lambda value: f"{value * 100:.2f}%"}))
    if stop_summary is not None:
        print(f"\n月停損研究 only: threshold={args.monthly_stop * 100:.1f}%")
        print(stop_summary.to_string(index=False, formatters={"淨報酬率": lambda value: f"{value * 100:.2f}%"}))
    print(f"\nWrote: {args.work_dir / 'output' / 'strategy_monitor_summary.csv'}")
    if stop_summary is not None:
        print(f"Wrote: {args.work_dir / 'output' / 'monthly_stop_research.csv'}")
    print(f"Wrote: {args.work_dir / 'output' / 'txf_recent_1min_with_signals.csv'}")


if __name__ == "__main__":
    main()
