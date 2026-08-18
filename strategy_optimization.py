#!/usr/bin/env python3
"""
Robust TXF EMA strategy optimization on the merged 1-minute dataset.

This is a research harness, not an execution engine.  The default split keeps
2006-01-02..2024-05-16 as the design/train period and treats 2024-05-17 onward
as the already-inspected OOS period.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    "/Users/guichenxiang/txf_backtest/shioaji_data/TXF_2006_20260626_1min_merged_unadjusted.csv"
)
DEFAULT_OUT_DIR = Path("/Users/guichenxiang/txf_backtest/output/optimization")


@dataclass(frozen=True)
class ExposureRule:
    name: str
    base: float
    addon: float


BASELINE_RULES = [
    ExposureRule("Buy & Hold", 1.0, 0.0),
    ExposureRule("Pure EMA 1x", 0.0, 1.0),
    ExposureRule("Pure EMA 1.5x", 0.0, 1.5),
    ExposureRule("Pure EMA 2x", 0.0, 2.0),
    ExposureRule("Always-in 0.25/1.50", 0.25, 1.25),
    ExposureRule("Always-in 0.50/1.50", 0.50, 1.00),
    ExposureRule("Always-in 0.75/1.50", 0.75, 0.75),
    ExposureRule("Old add-on 1.00/1.50", 1.00, 0.50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize TXF EMA exposure strategies.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--oos-start", type=str, default="2024-05-17")
    parser.add_argument("--cost", type=float, default=1.5, help="Index points per 1.0 exposure change.")
    parser.add_argument("--base-fast", type=int, default=360)
    parser.add_argument("--base-slow", type=int, default=2880)
    parser.add_argument("--base-band", type=float, default=5.0)
    parser.add_argument(
        "--include-last-date",
        action="store_true",
        help="Include the final calendar date even if it is likely partial.",
    )
    return parser.parse_args()


def load_bars(path: Path, include_last_date: bool) -> pd.DataFrame:
    usecols = ["Date", "Time", "High", "Low", "Close", "datetime"]
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"])
    else:
        df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
    for col in ["High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dt", "High", "Low", "Close"]).reset_index(drop=True)

    if not include_last_date and len(df):
        last_day = df["dt"].iloc[-1].normalize()
        df = df[df["dt"].dt.normalize() < last_day].reset_index(drop=True)
    return df


def build_day_index(dt: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = dt.values.astype("datetime64[D]")
    unique_days, starts = np.unique(days, return_index=True)
    return days, unique_days, starts


def daily_from_bar_returns(bar_returns: np.ndarray, starts: np.ndarray) -> np.ndarray:
    clipped = np.clip(bar_returns, -0.999999, None)
    daily_log = np.add.reduceat(np.log1p(clipped), starts)
    return np.expm1(daily_log)


def metrics_from_daily(daily_returns: np.ndarray, days: np.ndarray) -> dict[str, float]:
    if len(daily_returns) == 0:
        return {
            "x": np.nan,
            "cagr": np.nan,
            "sharpe": np.nan,
            "maxdd": np.nan,
            "days": 0,
        }

    equity = np.cumprod(1.0 + daily_returns)
    total_x = float(equity[-1])
    span_days = max(1.0, float((days[-1] - days[0]) / np.timedelta64(1, "D")))
    cagr = total_x ** (365.25 / span_days) - 1.0
    std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else np.nan
    sharpe = float(np.mean(daily_returns) / std * np.sqrt(252.0)) if std and std > 0 else np.nan
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "x": total_x,
        "cagr": float(cagr),
        "sharpe": sharpe,
        "maxdd": float(np.min(drawdown)),
        "days": int(len(daily_returns)),
    }


def year_return_map(daily_returns: np.ndarray, unique_days: np.ndarray) -> dict[int, float]:
    years = pd.to_datetime(unique_days).year.to_numpy()
    out: dict[int, float] = {}
    for year in sorted(set(years)):
        mask = years == year
        out[int(year)] = float(np.prod(1.0 + daily_returns[mask]) - 1.0)
    return out


def build_row(
    name: str,
    exposure: np.ndarray,
    close: np.ndarray,
    price_returns: np.ndarray,
    cost: float,
    starts: np.ndarray,
    unique_days: np.ndarray,
    oos_start: np.datetime64,
    bh_years: dict[int, float] | None = None,
    meta: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    turnover = np.abs(np.diff(exposure, prepend=0.0))
    bar_returns = exposure * price_returns - turnover * (cost / close)
    daily = daily_from_bar_returns(bar_returns, starts)

    train_day_mask = unique_days < oos_start
    oos_day_mask = unique_days >= oos_start
    dt_days = unique_days.astype("datetime64[D]")

    train = metrics_from_daily(daily[train_day_mask], dt_days[train_day_mask])
    oos = metrics_from_daily(daily[oos_day_mask], dt_days[oos_day_mask])
    full = metrics_from_daily(daily, dt_days)

    bar_days = np.repeat(unique_days, np.diff(np.r_[starts, len(close)]))
    train_bar_mask = bar_days < oos_start
    oos_bar_mask = bar_days >= oos_start

    years = year_return_map(daily, unique_days)
    wins = ""
    loses = ""
    if bh_years is not None:
        valid_years = sorted(set(years).intersection(bh_years))
        win_years = [year for year in valid_years if years[year] > bh_years[year]]
        lose_years = [year for year in valid_years if years[year] <= bh_years[year]]
        wins = f"{len(win_years)}/{len(valid_years)}"
        loses = " ".join(str(year) for year in lose_years)

    row: dict[str, float | int | str] = {
        "strategy": name,
        "full_x": full["x"],
        "full_cagr": full["cagr"],
        "full_sharpe": full["sharpe"],
        "full_maxdd": full["maxdd"],
        "train_x": train["x"],
        "train_cagr": train["cagr"],
        "train_sharpe": train["sharpe"],
        "train_maxdd": train["maxdd"],
        "oos_x": oos["x"],
        "oos_cagr": oos["cagr"],
        "oos_sharpe": oos["sharpe"],
        "oos_maxdd": oos["maxdd"],
        "avg_exposure_full": float(np.mean(exposure)),
        "avg_exposure_train": float(np.mean(exposure[train_bar_mask])) if train_bar_mask.any() else np.nan,
        "avg_exposure_oos": float(np.mean(exposure[oos_bar_mask])) if oos_bar_mask.any() else np.nan,
        "turnover_full": float(np.sum(turnover)),
        "win_years_vs_bh": wins,
        "lose_years_vs_bh": loses,
    }
    if meta:
        row.update(meta)
    return row


def ema_signal(close_s: pd.Series, fast_ema: pd.Series, slow_ema: pd.Series, band: float) -> np.ndarray:
    signal = (fast_ema.to_numpy() > slow_ema.to_numpy() + band).astype(float)
    signal[: int(min(len(signal), max(1, slow_ema.attrs.get("span", 1))))] = 0.0
    position_signal = np.roll(signal, 1)
    position_signal[0] = 0.0
    return position_signal


def precompute_emas(close_s: pd.Series, spans: list[int]) -> dict[int, pd.Series]:
    emas: dict[int, pd.Series] = {}
    for span in sorted(set(spans)):
        ema = close_s.ewm(span=span, adjust=False).mean()
        ema.attrs["span"] = span
        emas[span] = ema
    return emas


def baseline_and_grids(
    close_s: pd.Series,
    close: np.ndarray,
    high: pd.Series,
    low: pd.Series,
    price_returns: np.ndarray,
    starts: np.ndarray,
    unique_days: np.ndarray,
    oos_start: np.datetime64,
    cost: float,
    base_fast: int,
    base_slow: int,
    base_band: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fast_grid = [120, 240, 360, 480, 720, 960]
    slow_grid = [1440, 2160, 2880, 4320, 5760]
    band_grid = [0.0, 5.0, 10.0, 20.0, 40.0]
    spans = sorted(set(fast_grid + slow_grid + [base_fast, base_slow]))
    emas = precompute_emas(close_s, spans)

    base_signal = ema_signal(close_s, emas[base_fast], emas[base_slow], base_band)
    bh_exposure = np.ones_like(close)
    bh_row = build_row(
        "Buy & Hold",
        bh_exposure,
        close,
        price_returns,
        cost,
        starts,
        unique_days,
        oos_start,
        bh_years=None,
        meta={"fast": 0, "slow": 0, "band": 0.0, "base": 1.0, "addon": 0.0},
    )
    bh_daily = daily_from_bar_returns(price_returns, starts)
    bh_years = year_return_map(bh_daily, unique_days)

    baseline_rows = [bh_row]
    for rule in BASELINE_RULES[1:]:
        exposure = rule.base + rule.addon * base_signal
        baseline_rows.append(
            build_row(
                rule.name,
                exposure,
                close,
                price_returns,
                cost,
                starts,
                unique_days,
                oos_start,
                bh_years=bh_years,
                meta={"fast": base_fast, "slow": base_slow, "band": base_band, "base": rule.base, "addon": rule.addon},
            )
        )

    scan_rules = [
        ExposureRule("Pure EMA 1x", 0.0, 1.0),
        ExposureRule("Pure EMA 2x", 0.0, 2.0),
        ExposureRule("Always-in 0.25/1.50", 0.25, 1.25),
        ExposureRule("Always-in 0.50/1.50", 0.50, 1.0),
    ]
    ema_rows = []
    for fast in fast_grid:
        for slow in slow_grid:
            if fast >= slow:
                continue
            for band in band_grid:
                sig = ema_signal(close_s, emas[fast], emas[slow], band)
                for rule in scan_rules:
                    exposure = rule.base + rule.addon * sig
                    ema_rows.append(
                        build_row(
                            f"{rule.name} EMA{fast}/{slow} band{band:g}",
                            exposure,
                            close,
                            price_returns,
                            cost,
                            starts,
                            unique_days,
                            oos_start,
                            bh_years=bh_years,
                            meta={"fast": fast, "slow": slow, "band": band, "base": rule.base, "addon": rule.addon},
                        )
                    )

    exposure_rows = []
    caps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    bases = [0.0, 0.25, 0.5, 0.75, 1.0]
    for base in bases:
        for cap in caps:
            if cap <= base:
                continue
            addon = cap - base
            exposure = base + addon * base_signal
            exposure_rows.append(
                build_row(
                    f"Exposure {base:.2f}/{cap:.2f}",
                    exposure,
                    close,
                    price_returns,
                    cost,
                    starts,
                    unique_days,
                    oos_start,
                    bh_years=bh_years,
                    meta={"fast": base_fast, "slow": base_slow, "band": base_band, "base": base, "addon": addon, "cap": cap},
                )
            )

    overlay_specs = [
        ("EMA360/2880 band5", base_signal, base_fast, base_slow, base_band),
        ("EMA480/2160 band5", ema_signal(close_s, emas[480], emas[2160], 5.0), 480, 2160, 5.0),
        ("EMA120/5760 band0", ema_signal(close_s, emas[120], emas[5760], 0.0), 120, 5760, 0.0),
    ]
    overlay_rows: list[dict[str, float | int | str]] = []
    for signal_label, signal, fast, slow, band in overlay_specs:
        overlay_rows.extend(
            build_overlay_rows(
                high=high,
                low=low,
                close=close,
                price_returns=price_returns,
                signal=signal,
                signal_label=signal_label,
                starts=starts,
                unique_days=unique_days,
                oos_start=oos_start,
                cost=cost,
                bh_years=bh_years,
                fast=fast,
                slow=slow,
                band=band,
            )
        )

    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(ema_rows),
        pd.DataFrame(exposure_rows),
        pd.DataFrame(overlay_rows),
    )


def build_daily_donchian_state(
    high: pd.Series,
    low: pd.Series,
    close: np.ndarray,
    starts: np.ndarray,
    unique_days: np.ndarray,
) -> np.ndarray:
    dt_index = pd.to_datetime(unique_days)
    last_indices = np.r_[starts[1:] - 1, len(close) - 1]
    daily = pd.DataFrame(
        {
            "High": np.maximum.reduceat(high.to_numpy(dtype=float), starts),
            "Low": np.minimum.reduceat(low.to_numpy(dtype=float), starts),
            "Close": close[last_indices],
        },
        index=dt_index,
    )
    hh50 = daily["High"].shift(1).rolling(50).max()
    ll50 = daily["Low"].shift(1).rolling(50).min()

    in_market = True
    states = []
    for close, hh, ll in zip(daily["Close"], hh50, ll50):
        if pd.notna(ll) and close < ll:
            in_market = False
        elif pd.notna(hh) and close > hh:
            in_market = True
        states.append(float(in_market))
    # Trade on the next day after the daily signal.
    return pd.Series(states, index=daily.index).shift(1).fillna(1.0).to_numpy()


def build_vol_scale(close: np.ndarray, starts: np.ndarray, unique_days: np.ndarray, target_vol: float) -> np.ndarray:
    daily_close = np.array([close[np.r_[starts[1:] - 1, len(close) - 1][i]] for i in range(len(starts))], dtype=float)
    daily_ret = pd.Series(daily_close).pct_change()
    realized = daily_ret.rolling(20).std() * np.sqrt(252.0)
    scale = (target_vol / realized).shift(1).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = scale.clip(lower=0.50, upper=1.50).to_numpy(dtype=float)
    counts = np.diff(np.r_[starts, len(close)])
    return np.repeat(scale, counts)


def build_overlay_rows(
    high: pd.Series,
    low: pd.Series,
    close: np.ndarray,
    price_returns: np.ndarray,
    signal: np.ndarray,
    signal_label: str,
    starts: np.ndarray,
    unique_days: np.ndarray,
    oos_start: np.datetime64,
    cost: float,
    bh_years: dict[int, float],
    fast: int,
    slow: int,
    band: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    counts = np.diff(np.r_[starts, len(close)])
    donchian_daily = build_daily_donchian_state(high, low, close, starts, unique_days)
    donchian_bar = np.repeat(donchian_daily, counts)

    overlay_specs = [
        ("0.25/1.50", 0.25, 1.25),
        ("0.50/1.50", 0.50, 1.00),
        ("Pure 2x", 0.00, 2.00),
    ]
    for label, base, addon in overlay_specs:
        raw = base + addon * signal
        cap_to_base = np.where(donchian_bar > 0, raw, np.minimum(raw, base))
        flat_when_bad = np.where(donchian_bar > 0, raw, 0.0)
        for overlay_name, exposure in [
            ("Donchian cap-to-base", cap_to_base),
            ("Donchian flat-when-bad", flat_when_bad),
        ]:
            rows.append(
                build_row(
                    f"{label} {signal_label} + {overlay_name}",
                    exposure,
                    close,
                    price_returns,
                    cost,
                    starts,
                    unique_days,
                    oos_start,
                    bh_years=bh_years,
                    meta={
                        "fast": fast,
                        "slow": slow,
                        "band": band,
                        "base": base,
                        "addon": addon,
                        "signal": signal_label,
                        "overlay": overlay_name,
                    },
                )
            )

    for target_vol in [0.15, 0.20, 0.25]:
        scale = build_vol_scale(close, starts, unique_days, target_vol)
        for label, base, addon in [("0.25/1.50", 0.25, 1.25), ("0.50/1.50", 0.50, 1.0)]:
            raw = base + addon * signal
            vol_variants = [
                ("total-scale", raw * scale),
                ("floor-keep", np.maximum(base, raw * scale)),
                ("addon-only", base + addon * signal * scale),
            ]
            for vol_mode, exposure in vol_variants:
                rows.append(
                    build_row(
                        f"{label} {signal_label} + vol-target {target_vol:.0%} {vol_mode}",
                        exposure,
                        close,
                        price_returns,
                        cost,
                        starts,
                        unique_days,
                        oos_start,
                        bh_years=bh_years,
                        meta={
                            "fast": fast,
                            "slow": slow,
                            "band": band,
                            "base": base,
                            "addon": addon,
                            "signal": signal_label,
                            "overlay": f"vol-target {target_vol:.0%} {vol_mode}",
                        },
                    )
                )
    return rows


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["train_dd_abs"] = out["train_maxdd"].abs()
    # OOS is report-only.  Using oos_sharpe here previously changed the chosen
    # EMA family and contaminated the holdout.
    out["robust_score"] = (
        out["train_sharpe"].fillna(-99)
        - 0.50 * np.maximum(out["train_dd_abs"] - 0.45, 0.0)
        - 0.20 * np.maximum(out["avg_exposure_full"] - 1.30, 0.0)
    )
    out["oos_used_for_selection"] = False
    return out.sort_values(["robust_score", "train_sharpe"], ascending=False)


def write_outputs(
    out_dir: Path,
    baseline: pd.DataFrame,
    ema_grid: pd.DataFrame,
    exposure_grid: pd.DataFrame,
    overlays: pd.DataFrame,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out_dir / "strategy_optimization_baselines.csv", index=False)
    ema_grid.to_csv(out_dir / "strategy_optimization_ema_grid.csv", index=False)
    exposure_grid.to_csv(out_dir / "strategy_optimization_exposure_grid.csv", index=False)
    overlays.to_csv(out_dir / "strategy_optimization_overlays.csv", index=False)

    combined = pd.concat(
        [
            baseline.assign(group="baseline"),
            ema_grid.assign(group="ema_grid"),
            exposure_grid.assign(group="exposure_grid"),
            overlays.assign(group="overlay"),
        ],
        ignore_index=True,
        sort=False,
    )
    ranked = add_rank_columns(combined)
    ranked.to_csv(out_dir / "strategy_optimization_ranked.csv", index=False)
    return ranked


def print_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> None:
    view = df.loc[:, cols].head(n).copy()
    pct_cols = [col for col in view.columns if col.endswith(("cagr", "maxdd"))]
    for col in pct_cols:
        view[col] = view[col].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "")
    for col in ["full_x", "train_x", "oos_x", "avg_exposure_full", "full_sharpe", "train_sharpe", "oos_sharpe"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(view.to_string(index=False))


def main() -> None:
    args = parse_args()
    bars = load_bars(args.csv, include_last_date=args.include_last_date)
    if bars.empty:
        raise RuntimeError("No rows loaded.")

    close_s = bars["Close"].astype(float).reset_index(drop=True)
    close = close_s.to_numpy(dtype=float)
    high = bars["High"].astype(float).reset_index(drop=True)
    low = bars["Low"].astype(float).reset_index(drop=True)
    dt = bars["dt"].reset_index(drop=True)
    _, unique_days, starts = build_day_index(dt)
    price_returns = np.zeros_like(close)
    price_returns[1:] = close[1:] / close[:-1] - 1.0
    oos_start = np.datetime64(pd.Timestamp(args.oos_start).date())

    baseline, ema_grid, exposure_grid, overlays = baseline_and_grids(
        close_s=close_s,
        close=close,
        high=high,
        low=low,
        price_returns=price_returns,
        starts=starts,
        unique_days=unique_days,
        oos_start=oos_start,
        cost=args.cost,
        base_fast=args.base_fast,
        base_slow=args.base_slow,
        base_band=args.base_band,
    )
    ranked = write_outputs(args.out_dir, baseline, ema_grid, exposure_grid, overlays)

    print(f"Data: {dt.iloc[0]} -> {dt.iloc[-1]} ({len(bars):,} minute rows, {len(unique_days):,} days)")
    print(f"OOS start: {args.oos_start} | cost: {args.cost} point per 1.0 exposure change")
    if not args.include_last_date:
        print("Final calendar date was excluded to avoid partial-day research bias.")

    cols = [
        "strategy",
        "group",
        "full_x",
        "full_sharpe",
        "full_maxdd",
        "train_sharpe",
        "train_maxdd",
        "oos_x",
        "oos_sharpe",
        "oos_maxdd",
        "avg_exposure_full",
        "win_years_vs_bh",
    ]
    print("\nBaselines")
    print_table(baseline.assign(group="baseline"), cols, n=len(baseline))

    print("\nTop robust-score candidates")
    printable = ranked[ranked["strategy"] != "Buy & Hold"]
    print_table(printable, cols + ["fast", "slow", "band", "base", "addon"], n=15)

    print(f"\nWrote: {args.out_dir / 'strategy_optimization_baselines.csv'}")
    print(f"Wrote: {args.out_dir / 'strategy_optimization_ema_grid.csv'}")
    print(f"Wrote: {args.out_dir / 'strategy_optimization_exposure_grid.csv'}")
    print(f"Wrote: {args.out_dir / 'strategy_optimization_overlays.csv'}")
    print(f"Wrote: {args.out_dir / 'strategy_optimization_ranked.csv'}")


if __name__ == "__main__":
    main()
