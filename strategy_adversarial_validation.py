#!/usr/bin/env python3
"""Adversarial validation for capped TXF EMA strategies.

The main candidate is deliberately simple:

* EMA480 > EMA2160 + 5 points: long, otherwise flat.
* Use yesterday's 20-trading-day realized volatility.
* If annualized volatility is above 15%, cut the active position in half.
* Never increase exposure when volatility is low.

This is a research harness, not an order execution engine.  The merged input is
still an unadjusted continuous series, and the post-2024 period has already been
inspected.  Results therefore remain provisional until roll-adjusted data and
genuinely unseen forward observations are available.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    "/Users/guichenxiang/txf_backtest/shioaji_data/"
    "TXF_2006_20260626_1min_merged_unadjusted.csv"
)
DEFAULT_OUT_DIR = Path("/Users/guichenxiang/txf_backtest/output/adversarial")
TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class BacktestResult:
    name: str
    exposure: np.ndarray
    daily_returns: np.ndarray
    bar_returns: np.ndarray
    turnover: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial TXF strategy validation.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cost", type=float, default=1.5)
    parser.add_argument("--include-partial-last-day", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260627)
    return parser.parse_args()


def load_bars(path: Path) -> pd.DataFrame:
    usecols = ["Date", "Time", "High", "Low", "Close", "datetime"]
    bars = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "datetime" in bars.columns:
        bars["dt"] = pd.to_datetime(bars["datetime"], errors="coerce")
    else:
        bars["dt"] = pd.to_datetime(
            bars["Date"].astype(str) + " " + bars["Time"].astype(str), errors="coerce"
        )
    for col in ["High", "Low", "Close"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = (
        bars.dropna(subset=["dt", "High", "Low", "Close"])
        .sort_values("dt")
        .drop_duplicates(subset=["dt"], keep="last")
        .reset_index(drop=True)
    )
    return bars


def assign_trading_day(dt: pd.Series) -> np.ndarray:
    """Map TXF evening and early-morning bars to the following day session."""
    calendar_day = dt.to_numpy(dtype="datetime64[D]")
    minute_of_day = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
    day_session = (minute_of_day >= 8 * 60) & (minute_of_day < 15 * 60)
    session_days = np.unique(calendar_day[day_session])
    if len(session_days) == 0:
        raise RuntimeError("No day-session bars found; cannot assign TXF trading days.")

    trading_day = calendar_day.copy()
    evening = minute_of_day >= 15 * 60
    outside_day = ~day_session

    # Evening bars belong to the next observed day session.  Early-morning bars
    # belong to the same date's session, or the next observed session on a holiday.
    evening_idx = np.searchsorted(session_days, calendar_day[evening], side="right")
    early = outside_day & ~evening
    early_idx = np.searchsorted(session_days, calendar_day[early], side="left")

    valid_evening = evening_idx < len(session_days)
    valid_early = early_idx < len(session_days)
    evening_positions = np.flatnonzero(evening)
    early_positions = np.flatnonzero(early)
    trading_day[evening_positions[valid_evening]] = session_days[evening_idx[valid_evening]]
    trading_day[early_positions[valid_early]] = session_days[early_idx[valid_early]]

    # Only possible at the very end of a partial file.
    trading_day[evening_positions[~valid_evening]] = calendar_day[evening_positions[~valid_evening]] + 1
    return trading_day


def group_starts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, starts = np.unique(labels, return_index=True)
    return unique, starts


def compound_groups(bar_returns: np.ndarray, starts: np.ndarray) -> np.ndarray:
    clipped = np.clip(bar_returns, -0.999999, None)
    return np.expm1(np.add.reduceat(np.log1p(clipped), starts))


def sharpe(returns: np.ndarray) -> float:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return np.nan
    std = np.std(clean, ddof=1)
    if std <= 0:
        return np.nan
    return float(np.mean(clean) / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return np.nan
    equity = np.cumprod(1.0 + np.clip(returns, -0.999999, None))
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def metrics(
    result: BacktestResult,
    trading_days: np.ndarray,
    bar_trading_days: np.ndarray,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, float | int | str]:
    day_mask = np.ones(len(trading_days), dtype=bool)
    bar_mask = np.ones(len(bar_trading_days), dtype=bool)
    if start:
        bound = np.datetime64(start)
        day_mask &= trading_days >= bound
        bar_mask &= bar_trading_days >= bound
    if end:
        bound = np.datetime64(end)
        day_mask &= trading_days <= bound
        bar_mask &= bar_trading_days <= bound

    daily = result.daily_returns[day_mask]
    intraday = result.bar_returns[bar_mask]
    days = trading_days[day_mask]
    if len(daily) == 0:
        return {
            "x": np.nan,
            "cagr": np.nan,
            "sharpe": np.nan,
            "daily_maxdd": np.nan,
            "intraday_maxdd": np.nan,
            "days": 0,
        }

    total_x = float(np.prod(1.0 + daily))
    span_days = max(1.0, float((days[-1] - days[0]) / np.timedelta64(1, "D")))
    return {
        "x": total_x,
        "cagr": float(total_x ** (365.25 / span_days) - 1.0),
        "sharpe": sharpe(daily),
        "daily_maxdd": max_drawdown(daily),
        "intraday_maxdd": max_drawdown(intraday),
        "days": int(len(daily)),
    }


def ema_signal(close: pd.Series, fast: int, slow: int, band: float) -> np.ndarray:
    fast_ema = close.ewm(span=fast, adjust=False).mean().to_numpy()
    slow_ema = close.ewm(span=slow, adjust=False).mean().to_numpy()
    raw = (fast_ema > slow_ema + band).astype(float)
    raw[: min(len(raw), slow)] = 0.0
    position = np.roll(raw, 1)
    position[0] = 0.0
    return position


def daily_close_values(close: np.ndarray, starts: np.ndarray) -> np.ndarray:
    last = np.r_[starts[1:] - 1, len(close) - 1]
    return close[last]


def realized_volatility(daily_close: np.ndarray, lookback: int) -> np.ndarray:
    returns = pd.Series(daily_close).pct_change()
    return (returns.rolling(lookback).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)).to_numpy()


def one_way_brake(
    daily_close: np.ndarray,
    starts: np.ndarray,
    bar_count: int,
    lookback: int,
    threshold: float,
    reduced_scale: float = 0.5,
) -> np.ndarray:
    vol = realized_volatility(daily_close, lookback)
    known_vol = np.roll(vol, 1)
    known_vol[0] = np.nan
    daily_scale = np.where(np.isfinite(known_vol) & (known_vol > threshold), reduced_scale, 1.0)
    counts = np.diff(np.r_[starts, bar_count])
    return np.repeat(daily_scale, counts)


def inverse_vol_scale(
    daily_close: np.ndarray,
    starts: np.ndarray,
    bar_count: int,
    lookback: int,
    target: float,
    upper: float | None,
) -> np.ndarray:
    vol = realized_volatility(daily_close, lookback)
    scale = target / vol
    scale = np.roll(scale, 1)
    scale[0] = np.nan
    scale = np.where(np.isfinite(scale), scale, 1.0)
    scale = np.maximum(scale, 0.5)
    if upper is not None:
        scale = np.minimum(scale, upper)
    counts = np.diff(np.r_[starts, bar_count])
    return np.repeat(scale, counts)


def run_backtest(
    name: str,
    exposure: np.ndarray,
    close: np.ndarray,
    price_returns: np.ndarray,
    starts: np.ndarray,
    cost: float,
) -> BacktestResult:
    turnover = np.abs(np.diff(exposure, prepend=0.0))
    bar_returns = exposure * price_returns - turnover * (cost / close)
    daily_returns = compound_groups(bar_returns, starts)
    return BacktestResult(
        name=name,
        exposure=exposure,
        daily_returns=daily_returns,
        bar_returns=bar_returns,
        turnover=float(np.sum(turnover)),
    )


def summarize_results(
    results: list[BacktestResult],
    trading_days: np.ndarray,
    bar_trading_days: np.ndarray,
) -> pd.DataFrame:
    periods = {
        "full": (None, None),
        "is_pre_2019": (None, "2018-12-31"),
        "oos_2019": ("2019-01-01", None),
        "design_pre_2024_05": (None, "2024-05-16"),
        "new_data_2024_05": ("2024-05-17", None),
    }
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        row: dict[str, float | int | str] = {
            "strategy": result.name,
            "min_exposure": float(np.min(result.exposure)),
            "max_exposure": float(np.max(result.exposure)),
            "avg_exposure": float(np.mean(result.exposure)),
            "turnover": result.turnover,
        }
        for label, (start, end) in periods.items():
            stat = metrics(result, trading_days, bar_trading_days, start=start, end=end)
            for key, value in stat.items():
                row[f"{label}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_period_results(
    results: list[BacktestResult],
    trading_days: np.ndarray,
    bar_trading_days: np.ndarray,
) -> pd.DataFrame:
    blocks = [
        ("2006-2009", "2006-01-01", "2009-12-31"),
        ("2010-2013", "2010-01-01", "2013-12-31"),
        ("2014-2017", "2014-01-01", "2017-12-31"),
        ("2018-2021", "2018-01-01", "2021-12-31"),
        ("2022-2026H1", "2022-01-01", "2026-12-31"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, start, end in blocks:
        for result in results:
            row = {"period": label, "strategy": result.name}
            row.update(metrics(result, trading_days, bar_trading_days, start=start, end=end))
            rows.append(row)
    return pd.DataFrame(rows)


def yearly_results(
    results: list[BacktestResult],
    trading_days: np.ndarray,
    bar_trading_days: np.ndarray,
) -> pd.DataFrame:
    years = pd.to_datetime(trading_days).year
    rows: list[dict[str, float | int | str]] = []
    for year in sorted(np.unique(years)):
        for result in results:
            row = {"year": int(year), "strategy": result.name}
            row.update(
                metrics(
                    result,
                    trading_days,
                    bar_trading_days,
                    start=f"{year}-01-01",
                    end=f"{year}-12-31",
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def leave_one_year_out_results(
    results: list[BacktestResult],
    trading_days: np.ndarray,
) -> pd.DataFrame:
    years = pd.to_datetime(trading_days).year.to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for excluded_year in sorted(np.unique(years)):
        keep = years != excluded_year
        for result in results:
            daily = result.daily_returns[keep]
            rows.append(
                {
                    "excluded_year": int(excluded_year),
                    "strategy": result.name,
                    "sharpe": sharpe(daily),
                    "x": float(np.prod(1.0 + daily)),
                }
            )
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="excluded_year", columns="strategy", values="sharpe")
    frame = frame.merge(
        (
            pivot["EMA480/2160 binary brake 1x"] - pivot["EMA480/2160 pure 1x"]
        ).rename("brake_minus_pure_sharpe"),
        on="excluded_year",
    )
    return frame


def return_concentration(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        daily = result.daily_returns
        log_returns = np.log1p(np.clip(daily, -0.999999, None))
        total_log = float(np.sum(log_returns))
        descending = np.sort(log_returns)[::-1]
        row: dict[str, float | int | str] = {
            "strategy": result.name,
            "full_sharpe": sharpe(daily),
            "full_x": float(np.prod(1.0 + daily)),
        }
        for count in [1, 5, 10, 20]:
            top_share = float(np.sum(descending[:count]) / total_log) if total_log != 0 else np.nan
            remove = np.ones(len(daily), dtype=bool)
            top_indices = np.argpartition(log_returns, -count)[-count:]
            remove[top_indices] = False
            row[f"top_{count}_days_log_return_share"] = top_share
            row[f"sharpe_without_top_{count}_days"] = sharpe(daily[remove])
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap_sharpe_difference(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    samples: int,
    block: int,
    seed: int,
) -> dict[str, float | int]:
    if len(candidate) != len(benchmark):
        raise ValueError("Bootstrap return arrays must be aligned.")
    n = len(candidate)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    block_count = math.ceil(n / block)
    max_start = max(1, n - block + 1)
    offsets = np.arange(block)
    for i in range(samples):
        starts = rng.integers(0, max_start, size=block_count)
        indices = (starts[:, None] + offsets).ravel()[:n]
        differences[i] = sharpe(candidate[indices]) - sharpe(benchmark[indices])
    q = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "samples": samples,
        "block_days": block,
        "observed_sharpe_diff": sharpe(candidate) - sharpe(benchmark),
        "probability_diff_gt_zero": float(np.mean(differences > 0)),
        "ci_2_5": float(q[0]),
        "median": float(q[1]),
        "ci_97_5": float(q[2]),
    }


def cscv_pbo(return_matrix: np.ndarray, segments: int = 12) -> dict[str, float | int]:
    """Estimate probability of backtest overfitting with contiguous CSCV blocks."""
    if segments % 2:
        raise ValueError("CSCV segment count must be even.")
    if return_matrix.ndim != 2 or return_matrix.shape[1] < 2:
        raise ValueError("PBO requires at least two strategy columns.")
    folds = np.array_split(np.arange(return_matrix.shape[0]), segments)
    logits: list[float] = []
    test_ranks: list[float] = []
    for train_folds in itertools.combinations(range(segments), segments // 2):
        train_set = set(train_folds)
        test_folds = [i for i in range(segments) if i not in train_set]
        train_idx = np.concatenate([folds[i] for i in train_folds])
        test_idx = np.concatenate([folds[i] for i in test_folds])
        train_sharpes = np.array([sharpe(return_matrix[train_idx, j]) for j in range(return_matrix.shape[1])])
        test_sharpes = np.array([sharpe(return_matrix[test_idx, j]) for j in range(return_matrix.shape[1])])
        winner = int(np.nanargmax(train_sharpes))
        ascending_rank = int(np.sum(test_sharpes <= test_sharpes[winner]))
        relative_rank = (ascending_rank - 0.5) / return_matrix.shape[1]
        relative_rank = float(np.clip(relative_rank, 1e-6, 1.0 - 1e-6))
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
        test_ranks.append(relative_rank)
    logit_array = np.asarray(logits)
    return {
        "strategies": int(return_matrix.shape[1]),
        "segments": segments,
        "combinations": int(len(logits)),
        "pbo": float(np.mean(logit_array <= 0.0)),
        "median_selected_test_percentile": float(np.median(test_ranks)),
    }


def metric_row(
    name: str,
    result: BacktestResult,
    trading_days: np.ndarray,
    bar_trading_days: np.ndarray,
) -> dict[str, float | int | str]:
    full = metrics(result, trading_days, bar_trading_days)
    oos19 = metrics(result, trading_days, bar_trading_days, start="2019-01-01")
    oos24 = metrics(result, trading_days, bar_trading_days, start="2024-05-17")
    row: dict[str, float | int | str] = {
        "strategy": name,
        "avg_exposure": float(np.mean(result.exposure)),
        "max_exposure": float(np.max(result.exposure)),
        "turnover": result.turnover,
    }
    for prefix, stat in [("full", full), ("oos_2019", oos19), ("new_data_2024_05", oos24)]:
        for key, value in stat.items():
            row[f"{prefix}_{key}"] = value
    return row


def write_report(
    path: Path,
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    ema_grid: pd.DataFrame,
    brake_grid: pd.DataFrame,
    cost_stress: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    jump_stress: pd.DataFrame,
    pbo: dict[str, float | int],
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> None:
    def selected(name: str) -> pd.Series:
        return summary.loc[summary["strategy"] == name].iloc[0]

    bh = selected("Buy & Hold")
    pure = selected("EMA480/2160 pure 1x")
    brake = selected("EMA480/2160 binary brake 1x")
    brake15 = selected("EMA480/2160 binary brake 1.5x")
    neighborhood = ema_grid[
        (ema_grid["fast"].isin([360, 480, 600]))
        & (ema_grid["slow"].isin([1800, 2160, 2520]))
    ]
    stable_blocks = periods.pivot(index="period", columns="strategy", values="sharpe")
    wins_vs_pure = int(
        (stable_blocks["EMA480/2160 binary brake 1x"] > stable_blocks["EMA480/2160 pure 1x"]).sum()
    )
    loo_brake = leave_one_out[leave_one_out["strategy"] == "EMA480/2160 binary brake 1x"]
    worst_loo_diff = float(loo_brake["brake_minus_pure_sharpe"].min())
    jump_2pct = jump_stress[
        (jump_stress["strategy"] == "Binary brake 1x") & (jump_stress["bar_return_clip"] == 0.02)
    ].iloc[0]

    lines = [
        "# TXF EMA 對抗式驗證報告",
        "",
        f"資料：{data_start} 至 {data_end}；成本假設每 1.0 曝險變動 {cost_stress['cost_points'].min():g} 點。",
        "",
        "## 固定候選",
        "",
        "EMA480/2160 + 5 點濾網；訊號成立才做多。以前一交易日可知的 20 日實現波動率判斷：",
        "波動率高於 15% 時只留一半部位，否則維持全額。這是單向煞車，不會在低波動時加槓桿。",
        "",
        "|策略|全期 Sharpe|盤中 MaxDD|2019+ Sharpe|2024-05+ Sharpe|報酬 x|最大曝險|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|B&H|{bh.full_sharpe:.3f}|{bh.full_intraday_maxdd:.1%}|{bh.oos_2019_sharpe:.3f}|{bh.new_data_2024_05_sharpe:.3f}|{bh.full_x:.2f}|{bh.max_exposure:.2f}|",
        f"|純 EMA 1x|{pure.full_sharpe:.3f}|{pure.full_intraday_maxdd:.1%}|{pure.oos_2019_sharpe:.3f}|{pure.new_data_2024_05_sharpe:.3f}|{pure.full_x:.2f}|{pure.max_exposure:.2f}|",
        f"|波動煞車 1x|{brake.full_sharpe:.3f}|{brake.full_intraday_maxdd:.1%}|{brake.oos_2019_sharpe:.3f}|{brake.new_data_2024_05_sharpe:.3f}|{brake.full_x:.2f}|{brake.max_exposure:.2f}|",
        f"|波動煞車 1.5x|{brake15.full_sharpe:.3f}|{brake15.full_intraday_maxdd:.1%}|{brake15.oos_2019_sharpe:.3f}|{brake15.new_data_2024_05_sharpe:.3f}|{brake15.full_x:.2f}|{brake15.max_exposure:.2f}|",
        "",
        "## 反駁測試",
        "",
        f"- 固定 4 年區段中，波動煞車 Sharpe 高於純 EMA：{wins_vs_pure}/{len(stable_blocks)}。",
        f"- EMA 鄰域全期 Sharpe 中位數：{neighborhood.full_sharpe.median():.3f}；最差：{neighborhood.full_sharpe.min():.3f}。",
        f"- 波動參數網格全期 Sharpe 中位數：{brake_grid.full_sharpe.median():.3f}；最差：{brake_grid.full_sharpe.min():.3f}。",
        f"- 逐年剔除後，相對純 EMA 最差 Sharpe 差：{worst_loo_diff:+.3f}。",
        f"- 單根價格報酬壓到 +/-2% 後，波動煞車 Sharpe：{jump_2pct.full_sharpe:.3f}。",
        f"- CSCV PBO：{float(pbo['pbo']):.1%}（策略家族 {int(pbo['strategies'])} 個）。",
        "- Bootstrap 與成本壓力結果另見 CSV；信賴區間包含 0 時，不能宣稱統計上已證明優於基準。",
        "",
        "## 判決限制",
        "",
        "資料仍未做正式換月 back-adjust，舊檔與 Shioaji 的 K 棒來源也不同；2024-05 後資料已被看過，",
        "因此它不是全新 OOS。這份驗證能否定明顯的 5x 槓桿假象，但不能保證未來獲利。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bars = load_bars(args.csv)
    if bars.empty:
        raise RuntimeError("No bars loaded.")

    bars["trading_day"] = assign_trading_day(bars["dt"])
    if not args.include_partial_last_day:
        last_trading_day = bars["trading_day"].iloc[-1]
        bars = bars[bars["trading_day"] < last_trading_day].reset_index(drop=True)
    if bars.empty:
        raise RuntimeError("No complete trading day remains after filtering.")

    close_s = bars["Close"].astype(float).reset_index(drop=True)
    close = close_s.to_numpy()
    bar_trading_days = bars["trading_day"].to_numpy(dtype="datetime64[D]")
    trading_days, starts = group_starts(bar_trading_days)
    daily_close = daily_close_values(close, starts)
    price_returns = np.zeros_like(close)
    price_returns[1:] = close[1:] / close[:-1] - 1.0

    signal_480 = ema_signal(close_s, 480, 2160, 5.0)
    signal_360 = ema_signal(close_s, 360, 2880, 5.0)
    brake_scale = one_way_brake(daily_close, starts, len(close), 20, 0.15)

    exposures = {
        "Buy & Hold": np.ones_like(close),
        "EMA360/2880 pure 1x": signal_360,
        "EMA480/2160 pure 1x": signal_480,
        "EMA480/2160 pure 1.5x": 1.5 * signal_480,
        "EMA480/2160 staircase 0.25/1.50": 0.25 + 1.25 * signal_480,
        "EMA480/2160 binary brake 1x": signal_480 * brake_scale,
        "EMA480/2160 binary brake 1.5x": 1.5 * signal_480 * brake_scale,
    }
    results = [
        run_backtest(name, exposure, close, price_returns, starts, args.cost)
        for name, exposure in exposures.items()
    ]
    result_map = {result.name: result for result in results}

    summary = summarize_results(results, trading_days, bar_trading_days)
    periods = contiguous_period_results(results, trading_days, bar_trading_days)
    yearly = yearly_results(results, trading_days, bar_trading_days)
    leave_one_out = leave_one_year_out_results(results, trading_days)
    concentration = return_concentration(results)

    # Reproduce the criticized inverse-volatility family.  The final exposure
    # cap, not merely the volatility scale, is what matters operationally.
    raw_staircase = 0.25 + 1.25 * signal_480
    uncapped_scale = inverse_vol_scale(daily_close, starts, len(close), 20, 0.15, upper=None)
    cap_rows: list[dict[str, float | int | str]] = []
    no_vt = run_backtest("No VT", raw_staircase, close, price_returns, starts, args.cost)
    cap_rows.append(metric_row("No VT", no_vt, trading_days, bar_trading_days))
    for final_cap in [1.0, 1.5, 2.0, None]:
        exposure = np.maximum(0.25, raw_staircase * uncapped_scale)
        label = "uncapped" if final_cap is None else f"cap {final_cap:.1f}x"
        if final_cap is not None:
            exposure = np.minimum(exposure, final_cap)
        result = run_backtest(label, exposure, close, price_returns, starts, args.cost)
        row = metric_row(label, result, trading_days, bar_trading_days)
        row["pct_bars_above_1_5"] = float(np.mean(exposure > 1.5))
        cap_rows.append(row)
    cap_sweep = pd.DataFrame(cap_rows)

    # Parameter-neighborhood tests are reported, not used to choose on OOS.
    ema_rows: list[dict[str, float | int | str]] = []
    pbo_results: list[BacktestResult] = []
    for fast in [360, 480, 600]:
        for slow in [1800, 2160, 2520, 2880]:
            if fast >= slow:
                continue
            for band in [0.0, 5.0, 10.0]:
                signal = ema_signal(close_s, fast, slow, band)
                exposure = signal * brake_scale
                name = f"EMA{fast}/{slow} band{band:g} brake L20 T15%"
                result = run_backtest(name, exposure, close, price_returns, starts, args.cost)
                row = metric_row(name, result, trading_days, bar_trading_days)
                row.update({"fast": fast, "slow": slow, "band": band})
                ema_rows.append(row)
                pbo_results.append(result)
    ema_grid = pd.DataFrame(ema_rows)

    brake_rows: list[dict[str, float | int | str]] = []
    for lookback in [10, 20, 40, 60]:
        for threshold in [0.125, 0.15, 0.175]:
            scale = one_way_brake(daily_close, starts, len(close), lookback, threshold)
            exposure = signal_480 * scale
            name = f"EMA480/2160 brake L{lookback} T{threshold:.3f}"
            result = run_backtest(name, exposure, close, price_returns, starts, args.cost)
            row = metric_row(name, result, trading_days, bar_trading_days)
            row.update({"lookback": lookback, "threshold": threshold})
            brake_rows.append(row)
            pbo_results.append(result)
    brake_grid = pd.DataFrame(brake_rows)

    cost_rows: list[dict[str, float | int | str]] = []
    stress_exposures = {
        "Buy & Hold": exposures["Buy & Hold"],
        "Pure EMA480/2160 1x": exposures["EMA480/2160 pure 1x"],
        "Binary brake 1x": exposures["EMA480/2160 binary brake 1x"],
        "Binary brake 1.5x": exposures["EMA480/2160 binary brake 1.5x"],
    }
    for cost in [1.5, 3.0, 5.0, 10.0]:
        for name, exposure in stress_exposures.items():
            result = run_backtest(name, exposure, close, price_returns, starts, cost)
            row = metric_row(name, result, trading_days, bar_trading_days)
            row["cost_points"] = cost
            cost_rows.append(row)
    cost_stress = pd.DataFrame(cost_rows)

    jump_rows: list[dict[str, float | int | str]] = []
    jump_exposures = {
        "Buy & Hold": exposures["Buy & Hold"],
        "Pure EMA480/2160 1x": exposures["EMA480/2160 pure 1x"],
        "Binary brake 1x": exposures["EMA480/2160 binary brake 1x"],
    }
    for clip in [None, 0.01, 0.02]:
        stressed_price_returns = price_returns if clip is None else np.clip(price_returns, -clip, clip)
        for name, exposure in jump_exposures.items():
            result = run_backtest(name, exposure, close, stressed_price_returns, starts, args.cost)
            row = metric_row(name, result, trading_days, bar_trading_days)
            row["bar_return_clip"] = np.nan if clip is None else clip
            jump_rows.append(row)
    jump_stress = pd.DataFrame(jump_rows)

    bootstrap_rows: list[dict[str, float | int | str]] = []
    candidate = result_map["EMA480/2160 binary brake 1x"]
    for period, start in [("full", None), ("oos_2019", "2019-01-01"), ("new_data_2024_05", "2024-05-17")]:
        mask = np.ones(len(trading_days), dtype=bool)
        if start:
            mask &= trading_days >= np.datetime64(start)
        for benchmark_name in ["Buy & Hold", "EMA480/2160 pure 1x"]:
            benchmark = result_map[benchmark_name]
            row: dict[str, float | int | str] = {
                "period": period,
                "candidate": candidate.name,
                "benchmark": benchmark_name,
            }
            row.update(
                moving_block_bootstrap_sharpe_difference(
                    candidate.daily_returns[mask],
                    benchmark.daily_returns[mask],
                    samples=args.bootstrap_samples,
                    block=20,
                    seed=args.seed + len(bootstrap_rows),
                )
            )
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    pbo_matrix = np.column_stack([result.daily_returns for result in pbo_results])
    # Exact duplicate variants would otherwise receive multiple votes in rank tests.
    pbo_matrix = np.unique(np.round(pbo_matrix.T, 14), axis=0).T
    pbo = cscv_pbo(pbo_matrix, segments=12)
    pbo_frame = pd.DataFrame([pbo])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "adversarial_strategy_summary.csv", index=False)
    periods.to_csv(args.out_dir / "adversarial_periods.csv", index=False)
    yearly.to_csv(args.out_dir / "adversarial_yearly.csv", index=False)
    leave_one_out.to_csv(args.out_dir / "adversarial_leave_one_year_out.csv", index=False)
    concentration.to_csv(args.out_dir / "adversarial_return_concentration.csv", index=False)
    cap_sweep.to_csv(args.out_dir / "adversarial_cap_sweep.csv", index=False)
    ema_grid.to_csv(args.out_dir / "adversarial_ema_neighborhood.csv", index=False)
    brake_grid.to_csv(args.out_dir / "adversarial_brake_grid.csv", index=False)
    cost_stress.to_csv(args.out_dir / "adversarial_cost_stress.csv", index=False)
    jump_stress.to_csv(args.out_dir / "adversarial_jump_stress.csv", index=False)
    bootstrap.to_csv(args.out_dir / "adversarial_bootstrap.csv", index=False)
    pbo_frame.to_csv(args.out_dir / "adversarial_pbo.csv", index=False)
    write_report(
        args.out_dir / "TXF_EMA_adversarial_report.md",
        summary,
        periods,
        ema_grid,
        brake_grid,
        cost_stress,
        bootstrap,
        leave_one_out,
        jump_stress,
        pbo,
        bars["dt"].iloc[0],
        bars["dt"].iloc[-1],
    )

    display = summary[
        [
            "strategy",
            "full_x",
            "full_sharpe",
            "full_intraday_maxdd",
            "oos_2019_sharpe",
            "new_data_2024_05_sharpe",
            "max_exposure",
        ]
    ].copy()
    print(f"Data: {bars['dt'].iloc[0]} -> {bars['dt'].iloc[-1]}")
    print(f"Trading days: {len(trading_days):,}; bars: {len(bars):,}; cost: {args.cost:g} points")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPBO")
    print(pbo_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
