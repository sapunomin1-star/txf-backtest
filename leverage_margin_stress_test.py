#!/usr/bin/env python3
"""Compare target-leverage and initial-margin-max TXF execution profiles."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_adversarial_validation import (
    assign_trading_day,
    daily_close_values,
    ema_signal,
    group_starts,
    load_bars,
    metrics,
    one_way_brake,
    run_backtest,
)


DEFAULT_CSV = Path(
    "/Users/guichenxiang/txf_backtest/shioaji_data/"
    "TXF_2006_20260626_1min_merged_unadjusted.csv"
)
DEFAULT_OUT = Path("/Users/guichenxiang/txf_backtest/output/leverage_margin")
MARGIN_SOURCE = "https://www.taifex.com.tw/cht/5/indexMarging"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TXF leverage and legal-margin stress test")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--equity", type=float, default=200_000)
    parser.add_argument("--target-leverage", type=float, default=7.23)
    parser.add_argument("--initial-margin", type=float, default=28_900)
    parser.add_argument("--maintenance-margin", type=float, default=22_150)
    parser.add_argument("--multiplier", type=float, default=10.0)
    parser.add_argument("--cost-points", type=float, default=1.5)
    return parser.parse_args()


def state_exposure(signal: np.ndarray, brake: np.ndarray, full: float, reduced: float) -> np.ndarray:
    active = np.where(brake < 1.0, reduced, full)
    return signal * active


def state_lots(signal: np.ndarray, brake: np.ndarray, full: int, reduced: int) -> np.ndarray:
    active = np.where(brake < 1.0, reduced, full)
    return (signal * active).astype(int)


def summary_row(name, exposure, close, price_returns, starts, days, bar_days, cost):
    result = run_backtest(name, exposure, close, price_returns, starts, cost)
    row = {
        "strategy": name,
        "min_exposure": float(np.min(exposure)),
        "max_exposure": float(np.max(exposure)),
        "avg_exposure": float(np.mean(exposure)),
    }
    for label, start in [("full", None), ("oos_2019", "2019-01-01"), ("new_data_2024_05", "2024-05-17")]:
        stat = metrics(result, days, bar_days, start=start)
        for key, value in stat.items():
            row[f"{label}_{key}"] = value
    return row


def yearly_exposure_results(exposures, close, price_returns, starts, days, bar_days, cost):
    years = sorted(np.unique(pd.to_datetime(days).year))
    rows = []
    for name, exposure in exposures.items():
        result = run_backtest(name, exposure, close, price_returns, starts, cost)
        for year in years:
            stat = metrics(
                result,
                days,
                bar_days,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
            )
            rows.append({"year": int(year), "strategy": name, **stat})
    frame = pd.DataFrame(rows)
    bh = frame[frame["strategy"] == "Buy & Hold 1x"][["year", "x", "sharpe"]].rename(
        columns={"x": "bh_x", "sharpe": "bh_sharpe"}
    )
    frame = frame.merge(bh, on="year", how="left")
    frame["beat_bh_return"] = frame["x"] > frame["bh_x"]
    frame["beat_bh_sharpe"] = frame["sharpe"] > frame["bh_sharpe"]
    return frame


def simulate_contract_period(
    name: str,
    close: np.ndarray,
    dt: np.ndarray,
    lots: np.ndarray,
    start_equity: float,
    maintenance_margin: float,
    multiplier: float,
    cost_points: float,
) -> dict[str, float | int | str | bool]:
    equity = float(start_equity)
    peak = equity
    maxdd = 0.0
    margin_call = False
    margin_call_time = ""
    min_margin_buffer = np.inf
    max_lots = 0
    turnover = 0
    previous_lots = 0

    for i in range(len(close)):
        current_lots = int(lots[i])
        delta = abs(current_lots - previous_lots)
        turnover += delta
        equity -= delta * cost_points * multiplier
        if i:
            equity += current_lots * (close[i] - close[i - 1]) * multiplier
        required = current_lots * maintenance_margin
        min_margin_buffer = min(min_margin_buffer, equity - required)
        max_lots = max(max_lots, current_lots)
        peak = max(peak, equity)
        maxdd = min(maxdd, equity / peak - 1.0)
        previous_lots = current_lots
        if current_lots and equity < required:
            margin_call = True
            margin_call_time = str(pd.Timestamp(dt[i]))
            break

    return {
        "strategy": name,
        "ending_equity": equity,
        "return": equity / start_equity - 1.0,
        "maxdd": maxdd,
        "margin_call": margin_call,
        "margin_call_time": margin_call_time,
        "min_margin_buffer": min_margin_buffer,
        "max_lots": max_lots,
        "turnover_lots": turnover,
        "bars_survived": i + 1,
    }


def contract_stress_rows(
    close: np.ndarray,
    dt: np.ndarray,
    bar_days: np.ndarray,
    profiles: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> pd.DataFrame:
    years = pd.to_datetime(bar_days).year.to_numpy()
    periods: list[tuple[str, np.ndarray]] = [
        ("full", np.ones(len(close), dtype=bool)),
        ("oos_2019", bar_days >= np.datetime64("2019-01-01")),
        ("new_data_2024_05", bar_days >= np.datetime64("2024-05-17")),
    ]
    periods.extend((str(year), years == year) for year in sorted(np.unique(years)))

    rows = []
    for period, mask in periods:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        for name, lots in profiles.items():
            row = {"period": period}
            row.update(
                simulate_contract_period(
                    name=name,
                    close=close[indices],
                    dt=dt[indices],
                    lots=lots[indices],
                    start_equity=args.equity,
                    maintenance_margin=args.maintenance_margin,
                    multiplier=args.multiplier,
                    cost_points=args.cost_points,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_dynamic_period(
    name: str,
    mode: str,
    close: np.ndarray,
    dt: np.ndarray,
    signal: np.ndarray,
    brake: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int | str | bool]:
    equity = float(args.equity)
    peak = equity
    maxdd = 0.0
    held_lots = 0
    max_lots = 0
    turnover = 0
    margin_call = False
    margin_call_time = ""
    min_margin_buffer = np.inf
    previous_state = None

    for i in range(len(close)):
        if i:
            equity += held_lots * (close[i] - close[i - 1]) * args.multiplier
        required_maintenance = held_lots * args.maintenance_margin
        min_margin_buffer = min(min_margin_buffer, equity - required_maintenance)
        peak = max(peak, equity)
        maxdd = min(maxdd, equity / peak - 1.0)
        if held_lots and equity < required_maintenance:
            margin_call = True
            margin_call_time = str(pd.Timestamp(dt[i]))
            break

        state = (bool(signal[i]), bool(brake[i] < 1.0))
        if state != previous_state:
            if not state[0]:
                target_lots = 0
            else:
                legal_max = max(0, int(math.floor(equity / args.initial_margin)))
                if mode == "target":
                    target_exposure = args.target_leverage * (0.5 if state[1] else 1.0)
                    desired = target_exposure * equity / (close[i] * args.multiplier)
                    target_lots = min(legal_max, int(math.floor(desired + 0.5)))
                elif mode == "survival":
                    per_lot_notional = close[i] * args.multiplier
                    shock_max = max(
                        0,
                        int(
                            math.floor(
                                equity
                                / (
                                    args.maintenance_margin
                                    + 0.10 * per_lot_notional
                                )
                            )
                        ),
                    )
                    full_desired = (
                        args.target_leverage * equity / per_lot_notional
                    )
                    full_target = min(
                        legal_max,
                        shock_max,
                        int(math.floor(full_desired + 0.5)),
                    )
                    target_lots = (
                        max(1, int(math.floor(full_target / 2)))
                        if state[1] and full_target > 0
                        else full_target
                    )
                elif mode == "margin":
                    target_lots = int(math.ceil(legal_max / 2)) if state[1] else legal_max
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            change = abs(target_lots - held_lots)
            equity -= change * args.cost_points * args.multiplier
            turnover += change
            held_lots = target_lots
            max_lots = max(max_lots, held_lots)
            required_maintenance = held_lots * args.maintenance_margin
            min_margin_buffer = min(min_margin_buffer, equity - required_maintenance)
            if held_lots and equity < required_maintenance:
                margin_call = True
                margin_call_time = str(pd.Timestamp(dt[i]))
                break
        previous_state = state

    return {
        "strategy": name,
        "ending_equity": equity,
        "return": equity / args.equity - 1.0,
        "maxdd": maxdd,
        "margin_call": margin_call,
        "margin_call_time": margin_call_time,
        "min_margin_buffer": min_margin_buffer,
        "max_lots": max_lots,
        "turnover_lots": turnover,
        "bars_survived": i + 1,
    }


def dynamic_stress_rows(close, dt, bar_days, signal, brake, args):
    years = pd.to_datetime(bar_days).year.to_numpy()
    periods: list[tuple[str, np.ndarray]] = [
        ("full", np.ones(len(close), dtype=bool)),
        ("oos_2019", bar_days >= np.datetime64("2019-01-01")),
        ("new_data_2024_05", bar_days >= np.datetime64("2024-05-17")),
    ]
    periods.extend((str(year), years == year) for year in sorted(np.unique(years)))
    rows = []
    for period, mask in periods:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        for name, mode in [
            ("Target 7.23x dynamic re-entry", "target"),
            ("Survival-capped dynamic re-entry", "survival"),
            ("Initial-margin max dynamic re-entry", "margin"),
        ]:
            row = {"period": period}
            row.update(
                simulate_dynamic_period(
                    name, mode, close[indices], dt[indices], signal[indices], brake[indices], args
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def current_shock_table(setup: dict, args: argparse.Namespace) -> pd.DataFrame:
    profiles = {
        "Target 7.23x full": setup["target_full_lots"],
        "Target 7.23x reduced": setup["target_reduced_lots"],
        "Survival-capped full": setup["survival_full_lots"],
        "Survival-capped reduced": setup["survival_reduced_lots"],
        "Initial-margin max full": setup["margin_full_lots"],
        "Initial-margin max reduced": setup["margin_reduced_lots"],
    }
    rows = []
    for name, lots in profiles.items():
        for decline in [0.01, 0.02, 0.0451, 0.08, 0.10]:
            loss = lots * setup["reference_close"] * args.multiplier * decline
            ending_equity = args.equity - loss
            maintenance_required = lots * args.maintenance_margin
            rows.append(
                {
                    "strategy_state": name,
                    "lots": lots,
                    "index_decline": decline,
                    "loss": loss,
                    "ending_equity": ending_equity,
                    "maintenance_required": maintenance_required,
                    "below_maintenance": ending_equity < maintenance_required,
                }
            )
    return pd.DataFrame(rows)


def survival_ladder(setup: dict, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    per_lot_notional = setup["reference_close"] * args.multiplier
    for lots in range(1, setup["margin_full_lots"] + 1):
        total_notional = lots * per_lot_notional
        row = {
            "lots": lots,
            "notional_leverage": total_notional / args.equity,
            "initial_margin_required": lots * args.initial_margin,
            "maintenance_required": lots * args.maintenance_margin,
            "decline_to_maintenance": (
                args.equity - lots * args.maintenance_margin
            ) / total_notional,
            "decline_to_zero": args.equity / total_notional,
        }
        for shock in [0.0451, 0.08, 0.10]:
            ending = args.equity - total_notional * shock
            row[f"equity_after_{shock:.4f}"] = ending
            row[f"above_maintenance_after_{shock:.4f}"] = (
                ending >= lots * args.maintenance_margin
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    setup: dict,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    stress: pd.DataFrame,
    dynamic: pd.DataFrame,
    shocks: pd.DataFrame,
    ladder: pd.DataFrame,
) -> None:
    def get(name):
        return summary.loc[summary["strategy"] == name].iloc[0]

    bh = get("Buy & Hold 1x")
    target = get("Target 7.23x theoretical")
    target_integer = get("Target 7.23x executable 3/2 lots")
    survival = get("Survival-capped executable 2/1 lots")
    margin = get("Initial-margin max executable 6/3 lots")
    stress_yearly = stress[stress["period"].str.fullmatch(r"\d{4}")]
    calls = stress_yearly.groupby("strategy")["margin_call"].agg(["sum", "count"])
    dynamic_yearly = dynamic[dynamic["period"].str.fullmatch(r"\d{4}")]
    dynamic_calls = dynamic_yearly.groupby("strategy")["margin_call"].agg(["sum", "count"])
    shock_451 = shocks[shocks["index_decline"] == 0.0451]
    max_10pct_lots = int(
        ladder.loc[ladder["above_maintenance_after_0.1000"], "lots"].max()
    )
    survival_yearly = yearly[yearly["strategy"] == "Survival-capped executable 2/1 lots"]
    return_wins = int(survival_yearly["beat_bh_return"].sum())
    sharpe_wins = int(survival_yearly["beat_bh_sharpe"].sum())
    year_count = len(survival_yearly)
    fixed_compare = stress[
        stress["period"].isin(["full", "oos_2019", "new_data_2024_05"])
        & stress["strategy"].isin(["Buy & Hold one micro", "Survival-capped executable"])
    ]
    dynamic_survival = dynamic[
        dynamic["period"].isin(["full", "oos_2019", "new_data_2024_05"])
        & (dynamic["strategy"] == "Survival-capped dynamic re-entry")
    ]
    bh_period_x = {
        "full": bh.full_x,
        "oos_2019": bh.oos_2019_x,
        "new_data_2024_05": bh.new_data_2024_05_x,
    }

    lines = [
        "# 7.23x 與原始保證金開滿壓力測試",
        "",
        f"本金 {setup['equity']:,.0f}；參考指數 {setup['reference_close']:,.0f}；TMF 原始/維持保證金 "
        f"{setup['initial_margin']:,.0f}/{setup['maintenance_margin']:,.0f} 元。",
        "",
        "|策略|目前口數（低/高波）|實際曝險（低/高波）|全期 Sharpe|盤中 MaxDD|",
        "|---|---:|---:|---:|---:|",
        f"|Buy & Hold|連續1x|1.00|{bh.full_sharpe:.3f}|{bh.full_intraday_maxdd:.1%}|",
        f"|7.23x 理論|小數|7.23/3.615|{target.full_sharpe:.3f}|{target.full_intraday_maxdd:.1%}|",
        f"|7.23x 可執行|{setup['target_full_lots']}/{setup['target_reduced_lots']}|"
        f"{setup['target_full_actual']:.2f}/{setup['target_reduced_actual']:.2f}|"
        f"{target_integer.full_sharpe:.3f}|{target_integer.full_intraday_maxdd:.1%}|",
        f"|10% 生存上限|{setup['survival_full_lots']}/{setup['survival_reduced_lots']}|"
        f"{setup['survival_full_actual']:.2f}/{setup['survival_reduced_actual']:.2f}|"
        f"{survival.full_sharpe:.3f}|{survival.full_intraday_maxdd:.1%}|",
        f"|原始保證金開滿|{setup['margin_full_lots']}/{setup['margin_reduced_lots']}|"
        f"{setup['margin_full_actual']:.2f}/{setup['margin_reduced_actual']:.2f}|"
        f"{margin.full_sharpe:.3f}|{margin.full_intraday_maxdd:.1%}|",
        "",
        "## 20萬固定口數現金路徑",
        "",
        "|期間|策略|期末權益|報酬|MaxDD|",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in fixed_compare.iterrows():
        lines.append(
            f"|{row['period']}|{row['strategy']}|{row['ending_equity']:,.0f}|"
            f"{row['return']:.1%}|{row['maxdd']:.1%}|"
        )
    lines.extend([
        "",
        "## 每次再進場依權益重算口數",
        "",
        "|期間|2/1生存公式期末權益|B&H 1x期末權益|策略MaxDD|",
        "|---|---:|---:|---:|",
    ])
    for _, row in dynamic_survival.iterrows():
        lines.append(
            f"|{row['period']}|{row['ending_equity']:,.0f}|"
            f"{setup['equity'] * bh_period_x[row['period']]:,.0f}|{row['maxdd']:.1%}|"
        )
    lines.extend([
        "",
        "## 每年以 20 萬重置的維持保證金測試",
        "",
    ])
    for strategy, row in calls.iterrows():
        lines.append(f"- 固定口數 {strategy}: {int(row['sum'])}/{int(row['count'])} 年觸發維持保證金不足。")
    for strategy, row in dynamic_calls.iterrows():
        lines.append(f"- 動態再進場 {strategy}: {int(row['sum'])}/{int(row['count'])} 年觸發維持保證金不足。")
    lines.extend(["", "## 當前 20 萬遇到單日 -4.51%"])
    for _, row in shock_451.iterrows():
        result = "跌破維持保證金" if row["below_maintenance"] else "仍高於維持保證金"
        lines.append(
            f"- {row['strategy_state']}：{int(row['lots'])} 口，損失 {row['loss']:,.0f}，"
            f"剩餘 {row['ending_equity']:,.0f}，{result}。"
        )
    lines.extend(
        [
            "",
            f"以單次 -10% 後仍高於維持保證金為硬條件，最大為 {max_10pct_lots} 口。",
            f"2/1口生存版逐年報酬勝 B&H：{return_wins}/{year_count}；逐年 Sharpe 勝：{sharpe_wins}/{year_count}。",
            "",
            "以上固定口數壓力測試在每個年度重新以 20 萬開始；觸發後停止，未假設補繳。",
            "歷史合併資料尚未正式換月還原，且以目前保證金套用歷史，不代表當年實際法定金額。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    all_bars = load_bars(args.csv)
    if all_bars.empty:
        raise RuntimeError("No bars loaded")
    reference_close = float(all_bars["Close"].iloc[-1])
    all_bars["trading_day"] = assign_trading_day(all_bars["dt"])
    last_trading_day = all_bars["trading_day"].iloc[-1]
    bars = all_bars[all_bars["trading_day"] < last_trading_day].reset_index(drop=True)

    close_s = bars["Close"].astype(float).reset_index(drop=True)
    close = close_s.to_numpy()
    dt = bars["dt"].to_numpy()
    bar_days = bars["trading_day"].to_numpy(dtype="datetime64[D]")
    days, starts = group_starts(bar_days)
    daily_close = daily_close_values(close, starts)
    price_returns = np.zeros_like(close)
    price_returns[1:] = close[1:] / close[:-1] - 1.0
    signal = ema_signal(close_s, 480, 2160, 5.0)
    brake = one_way_brake(daily_close, starts, len(close), 20, 0.15)

    notional = reference_close * args.multiplier
    legal_max_lots = int(math.floor(args.equity / args.initial_margin))
    desired_full = args.target_leverage * args.equity / notional
    desired_reduced = args.target_leverage * 0.5 * args.equity / notional
    target_full_lots = min(legal_max_lots, int(math.floor(desired_full + 0.5)))
    target_reduced_lots = min(legal_max_lots, int(math.floor(desired_reduced + 0.5)))
    margin_full_lots = legal_max_lots
    margin_reduced_lots = int(math.ceil(legal_max_lots / 2))

    target_full_actual = target_full_lots * notional / args.equity
    target_reduced_actual = target_reduced_lots * notional / args.equity
    margin_full_actual = margin_full_lots * notional / args.equity
    margin_reduced_actual = margin_reduced_lots * notional / args.equity
    survival_full_lots = int(
        math.floor(
            args.equity
            / (args.maintenance_margin + 0.10 * notional)
        )
    )
    survival_reduced_lots = max(1, int(math.floor(survival_full_lots / 2)))
    survival_full_actual = survival_full_lots * notional / args.equity
    survival_reduced_actual = survival_reduced_lots * notional / args.equity

    exposures = {
        "Buy & Hold 1x": np.ones_like(close),
        "Baseline brake 1.5x": 1.5 * signal * brake,
        "Target 7.23x theoretical": args.target_leverage * signal * brake,
        "Target 7.23x executable 3/2 lots": state_exposure(
            signal, brake, target_full_actual, target_reduced_actual
        ),
        "Survival-capped executable 2/1 lots": state_exposure(
            signal, brake, survival_full_actual, survival_reduced_actual
        ),
        "Initial-margin max executable 6/3 lots": state_exposure(
            signal, brake, margin_full_actual, margin_reduced_actual
        ),
    }
    rows = [
        summary_row(name, exposure, close, price_returns, starts, days, bar_days, args.cost_points)
        for name, exposure in exposures.items()
    ]
    summary = pd.DataFrame(rows)
    yearly = yearly_exposure_results(
        exposures, close, price_returns, starts, days, bar_days, args.cost_points
    )

    lots_profiles = {
        "Buy & Hold one micro": np.ones_like(signal, dtype=int),
        "Target 7.23x executable": state_lots(
            signal, brake, target_full_lots, target_reduced_lots
        ),
        "Initial-margin max executable": state_lots(
            signal, brake, margin_full_lots, margin_reduced_lots
        ),
        "Survival-capped executable": state_lots(
            signal, brake, survival_full_lots, survival_reduced_lots
        ),
    }
    stress = contract_stress_rows(close, dt, bar_days, lots_profiles, args)
    dynamic = dynamic_stress_rows(close, dt, bar_days, signal, brake, args)
    setup = {
        "equity": args.equity,
        "reference_close": reference_close,
        "initial_margin": args.initial_margin,
        "maintenance_margin": args.maintenance_margin,
        "target_full_lots": target_full_lots,
        "target_reduced_lots": target_reduced_lots,
        "margin_full_lots": margin_full_lots,
        "margin_reduced_lots": margin_reduced_lots,
        "target_full_actual": target_full_actual,
        "target_reduced_actual": target_reduced_actual,
        "margin_full_actual": margin_full_actual,
        "margin_reduced_actual": margin_reduced_actual,
        "survival_full_lots": survival_full_lots,
        "survival_reduced_lots": survival_reduced_lots,
        "survival_full_actual": survival_full_actual,
        "survival_reduced_actual": survival_reduced_actual,
        "margin_source": MARGIN_SOURCE,
    }
    shocks = current_shock_table(setup, args)
    ladder = survival_ladder(setup, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([setup]).to_csv(args.out_dir / "leverage_margin_setup.csv", index=False)
    summary.to_csv(args.out_dir / "leverage_margin_summary.csv", index=False)
    yearly.to_csv(args.out_dir / "leverage_margin_yearly.csv", index=False)
    stress.to_csv(args.out_dir / "leverage_margin_contract_stress.csv", index=False)
    dynamic.to_csv(args.out_dir / "leverage_margin_dynamic_stress.csv", index=False)
    shocks.to_csv(args.out_dir / "leverage_margin_current_shocks.csv", index=False)
    ladder.to_csv(args.out_dir / "leverage_margin_survival_ladder.csv", index=False)
    write_report(
        args.out_dir / "leverage_margin_report.md",
        setup,
        summary,
        yearly,
        stress,
        dynamic,
        shocks,
        ladder,
    )

    print(pd.DataFrame([setup]).to_string(index=False))
    print("\nContinuous-exposure comparison")
    print(
        summary[["strategy", "full_x", "full_sharpe", "full_intraday_maxdd", "oos_2019_sharpe", "new_data_2024_05_sharpe"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\n20萬逐年重置：維持保證金觸發")
    yearly = stress[stress["period"].str.fullmatch(r"\d{4}")]
    print(yearly.groupby("strategy")["margin_call"].agg(["sum", "count"]).to_string())
    print("\n20萬逐年重置且每次再進場重算口數：維持保證金觸發")
    dynamic_yearly = dynamic[dynamic["period"].str.fullmatch(r"\d{4}")]
    print(dynamic_yearly.groupby("strategy")["margin_call"].agg(["sum", "count"]).to_string())
    print("\n當前部位單日 -4.51% 壓力")
    print(shocks[shocks["index_decline"] == 0.0451].to_string(index=False))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()
