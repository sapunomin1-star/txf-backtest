#!/usr/bin/env python3
"""
永豐 Shioaji 模擬盤測試：EMA480/2160 + 20 日波動 >15% 減半。

安全界線
--------
- 金鑰只從環境變數讀：SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY（本腳本不硬寫、不列印、不存檔）
- 一律 simulation=True，程式沒有切換真實帳戶的參數
- 預設「paper 模式」：用 Shioaji 即時資料算訊號 + 本地追蹤模擬部位/損益，不送任何單
- 整數口數不得突破 --max-leverage；本金不足時寧可 0 口，不會偷偷放大槓桿
- `--place-sim-orders` 才會在 Shioaji 模擬帳戶調倉，送單後必須查到目標部位才算成功

用法
----
    export SHIOAJI_API_KEY="你的key"      # 只在你自己終端機；別貼到任何對話
    export SHIOAJI_SECRET_KEY="你的secret"
    # A：低於 10 萬保命驗證：正常多頭 1 口，高波動/空頭 0 口
    python3 shioaji_paper_test.py --equity 90000 --sizing-mode micro-pilot \
      --initial-margin 31800 --maintenance-margin 24400

    # B：低於 10 萬平衡驗證：EMA 多頭 1 口，EMA 空頭 0 口，高波動只警示
    python3 shioaji_paper_test.py --equity 90000 --sizing-mode micro-balanced \
      --initial-margin 31800 --maintenance-margin 24400

    # C：核心 G4 理論曝險 1.0x；權益不足一口時會維持 0 口
    python3 shioaji_paper_test.py --equity 500000 --sizing-mode target-leverage

    # D：資金依原始保證金開滿
    python3 shioaji_paper_test.py --equity 200000 --sizing-mode margin-max

    # 研究用 1.25x 固定倍率必須明確指定；不是新 alpha
    python3 shioaji_paper_test.py --equity 800000 --sizing-mode target-leverage \
      --target-leverage 1.25

完整執行 1.0x / 0.5x 的兩段核心策略，約需能配置 2 口 / 1 口微台的權益；
腳本會依當下指數自動顯示所需金額。
"""
from __future__ import annotations
import os, argparse, datetime as dt, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FAST, SLOW, BAND, VOL_TH = 480, 2160, 5.0, 0.15
STATE = Path(__file__).resolve().parent / "shioaji_data" / "paper_state.csv"
TMF_POINT = 10
DEFAULT_INITIAL_MARGIN = 31_800
DEFAULT_MAINTENANCE_MARGIN = 24_400


@dataclass(frozen=True)
class SizingDecision:
    target_contracts: int
    desired_contracts: float
    actual_exposure: float
    one_lot_leverage: float
    max_contracts: int
    min_equity_one_lot: float
    two_level_equity: float
    legal_max_contracts: int
    initial_margin_required: float
    sizing_mode: str
    shock_max_contracts: int
    shock_equity_after: float
    shock_margin_buffer: float
    status: str


def parse_args():
    p = argparse.ArgumentParser(description="Shioaji 模擬盤 paper 測試（EMA480/2160 + 高波減半）")
    p.add_argument("--equity", type=float, default=200_000, help="帳戶權益（算 sizing 用）")
    p.add_argument("--symbol", default="TMFR1", help="微台連續近月（大台=TXFR1，影響的只是點值不影響訊號）")
    p.add_argument("--history-days", type=int, default=60, help="抓多少天 1 分 K 暖身（EMA2160+20日波動需 ~30 天；Shioaji 單次上限 30 天，會自動分段）")
    p.add_argument("--sizing-mode", choices=["target-leverage", "margin-max", "micro-pilot", "micro-balanced"],
                   default="target-leverage", help="核心策略倍率、原始保證金開滿，或低於10萬的1/0口小資金驗證")
    p.add_argument("--target-leverage", type=float, default=1.0,
                   help="target-leverage 模式的滿額名目槓桿；預設 G4=1.0x，高波動時減半")
    p.add_argument("--max-exposure", type=float, default=None,
                   help="向後相容舊參數；提供時會覆蓋 --target-leverage")
    p.add_argument("--initial-margin", type=float, default=DEFAULT_INITIAL_MARGIN,
                   help="TMF 每口法定原始保證金，需依期交所公告更新")
    p.add_argument("--maintenance-margin", type=float, default=DEFAULT_MAINTENANCE_MARGIN,
                   help="TMF 每口維持保證金")
    p.add_argument("--survival-shock", type=float, default=0.10,
                   help="target-leverage 模式需承受的單次逆向跌幅，預設 10%%")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="target-leverage 整數口數硬上限；預設等於目標槓桿")
    p.add_argument("--place-sim-orders", action="store_true", help="實際在 Shioaji 模擬帳戶下單（預設關，需自行驗證）")
    p.add_argument("--sim-roundtrip-test", action="store_true",
                   help="模擬帳戶做 1 口買進再賣出測試，最後必須回到 0 口")
    p.add_argument("--ack-high-risk", action="store_true",
                   help="確認高槓桿風險；高槓桿模擬下單必須明確提供")
    args = p.parse_args()
    if args.equity <= 0:
        p.error("--equity 必須大於 0")
    if args.max_exposure is not None:
        args.target_leverage = args.max_exposure
    if args.target_leverage <= 0:
        p.error("--target-leverage 必須大於 0")
    if args.initial_margin <= 0 or args.maintenance_margin <= 0:
        p.error("保證金參數必須大於 0")
    if args.maintenance_margin >= args.initial_margin:
        p.error("維持保證金必須小於原始保證金")
    if not 0 < args.survival_shock < 0.5:
        p.error("--survival-shock 必須介於 0 與 0.5 之間")
    if args.max_leverage is None:
        args.max_leverage = args.target_leverage
    if args.max_leverage <= 0:
        p.error("--max-leverage 必須大於 0")
    if args.sim_roundtrip_test and not args.place_sim_orders:
        p.error("--sim-roundtrip-test 必須搭配 --place-sim-orders")
    high_risk = args.sizing_mode == "margin-max" or args.target_leverage > 1.25
    if args.place_sim_orders and high_risk and not args.ack_high_risk:
        p.error("高槓桿模擬下單必須加 --ack-high-risk")
    return args


def get_api():
    try:
        import shioaji as sj
    except ImportError:
        raise SystemExit("未安裝 shioaji：pip install shioaji")
    key, sec = os.environ.get("SHIOAJI_API_KEY"), os.environ.get("SHIOAJI_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("請先 export SHIOAJI_API_KEY 與 SHIOAJI_SECRET_KEY（腳本不會讀到明文以外的東西，也不會印出）")
    api = sj.Shioaji(simulation=True)                   # ← 硬性模擬（建構子寫死，全腳本無路徑可設 False）
    api.login(api_key=key, secret_key=sec)
    return api, sj


def get_contract(api, symbol):
    # 微台 TMF / 大台 TXF 連續近月。不同 shioaji 版本屬性可能不同，找不到時請手動指定。
    fut = api.Contracts.Futures
    cat = symbol[:3]                                    # TMF / TXF
    try:
        return getattr(getattr(fut, cat), symbol)
    except AttributeError:
        raise SystemExit(f"找不到合約 {symbol}；請用 api.Contracts.Futures.{cat} 列出可用代碼後改 --symbol")


def fetch_minutes(api, contract, days):
    # Shioaji kbars 單次上限 30 天 → 分段（每段 28 天）抓再接起來
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    frames, cur = [], start
    while cur <= end:
        ce = min(cur + dt.timedelta(days=27), end)     # 28 天 span，安全低於 30
        kb = api.kbars(contract, start=cur.isoformat(), end=ce.isoformat())
        f = pd.DataFrame({**kb})
        if len(f):
            frames.append(f)
        cur = ce + dt.timedelta(days=1)
    if not frames:
        raise SystemExit("抓不到任何 K 棒（可能非交易時段或合約無資料）")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return (df.drop_duplicates(subset="ts").sort_values("ts")
              .rename(columns={"ts": "datetime"}).reset_index(drop=True))


def assign_trading_day(ts):
    """把 15:00 後夜盤與凌晨盤歸到下一個台指期日盤交易日。"""
    calendar_day = ts.to_numpy(dtype="datetime64[D]")
    minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    day_session = (minute_of_day >= 8 * 60) & (minute_of_day < 15 * 60)
    session_days = np.unique(calendar_day[day_session])
    if not len(session_days):
        raise SystemExit("歷史資料沒有日盤 K 棒，無法建立交易日波動率")

    trading_day = calendar_day.copy()
    evening = minute_of_day >= 15 * 60
    early = (~day_session) & (~evening)
    evening_idx = np.searchsorted(session_days, calendar_day[evening], side="right")
    early_idx = np.searchsorted(session_days, calendar_day[early], side="left")
    evening_pos = np.flatnonzero(evening)
    early_pos = np.flatnonzero(early)
    valid_evening = evening_idx < len(session_days)
    valid_early = early_idx < len(session_days)
    trading_day[evening_pos[valid_evening]] = session_days[evening_idx[valid_evening]]
    trading_day[early_pos[valid_early]] = session_days[early_idx[valid_early]]
    trading_day[evening_pos[~valid_evening]] = calendar_day[evening_pos[~valid_evening]] + 1
    return trading_day


def compute_signal(df, target_leverage):
    c = df["Close"].astype(float)
    ef = c.ewm(span=FAST, adjust=False).mean()
    es = c.ewm(span=SLOW, adjust=False).mean()
    long_sig = bool(ef.iloc[-1] > es.iloc[-1] + BAND)

    work = df[["datetime", "Close"]].copy()
    work["trading_day"] = assign_trading_day(work["datetime"])
    current_trading_day = work["trading_day"].iloc[-1]
    daily = work.groupby("trading_day", sort=True)["Close"].last().astype(float)
    completed = daily[daily.index < current_trading_day]
    if len(completed) < 21:
        raise SystemExit(f"完整交易日只有 {len(completed)} 天，不足以計算 20 日波動率")
    rolling_vol = completed.pct_change().rolling(20).std(ddof=1) * np.sqrt(252)
    vol = float(rolling_vol.iloc[-1])
    if not np.isfinite(vol):
        raise SystemExit("20 日波動率無法計算，請增加 --history-days")
    risk_scale = 0.5 if vol > VOL_TH else 1.0
    exposure = target_leverage * risk_scale if long_sig else 0.0
    return dict(close=float(c.iloc[-1]), ema_fast=float(ef.iloc[-1]), ema_slow=float(es.iloc[-1]),
                vol=vol, vol_asof=pd.Timestamp(completed.index[-1]), long=long_sig,
                risk_scale=risk_scale, exposure=exposure, last=df["datetime"].iloc[-1],
                trading_day=pd.Timestamp(current_trading_day))


def size_micro_contracts(
    exposure,
    equity,
    index_price,
    max_leverage,
    full_exposure=None,
    initial_margin=DEFAULT_INITIAL_MARGIN,
    maintenance_margin=DEFAULT_MAINTENANCE_MARGIN,
    survival_shock=0.10,
    sizing_mode="target-leverage",
    risk_scale=1.0,
):
    """依目標槓桿或原始保證金開滿，轉成可成交整數口數。"""
    contract_notional = index_price * TMF_POINT
    desired = exposure * equity / contract_notional
    leverage_max_contracts = max(
        0, int(math.floor(max_leverage * equity / contract_notional + 1e-12))
    )
    legal_max_contracts = max(0, int(math.floor(equity / initial_margin)))
    shock_max_contracts = max(
        0,
        int(
            math.floor(
                equity / (maintenance_margin + survival_shock * contract_notional)
            )
        ),
    )
    if exposure == 0:
        target = 0
    elif sizing_mode == "target-leverage":
        full_exposure = full_exposure or max_leverage
        full_desired = full_exposure * equity / contract_notional
        full_nearest = max(0, int(math.floor(full_desired + 0.5)))
        full_target = min(
            full_nearest,
            leverage_max_contracts,
            legal_max_contracts,
            shock_max_contracts,
        )
        target = (
            max(1, int(math.floor(full_target / 2)))
            if risk_scale < 1.0 and full_target > 0
            else full_target
        )
    elif sizing_mode == "micro-pilot":
        if risk_scale < 1.0:
            target = 0
        else:
            target = min(1, leverage_max_contracts, legal_max_contracts, shock_max_contracts)
    elif sizing_mode == "micro-balanced":
        target = min(1, leverage_max_contracts, legal_max_contracts, shock_max_contracts)
    elif sizing_mode == "margin-max":
        target = (
            int(math.ceil(legal_max_contracts / 2))
            if risk_scale < 1.0
            else legal_max_contracts
        )
    else:
        raise ValueError(f"Unknown sizing mode: {sizing_mode}")
    actual_exposure = target * contract_notional / equity
    min_equity_one_lot = contract_notional / max_leverage
    # 兩口滿額、一口高波減半，才能忠實執行二段曝險。
    full_exposure = full_exposure or max_leverage
    two_level_equity = 2 * contract_notional / full_exposure
    shock_equity_after = equity - target * contract_notional * survival_shock
    shock_margin_buffer = shock_equity_after - target * maintenance_margin

    if exposure == 0:
        status = "訊號空手"
    elif legal_max_contracts == 0:
        status = "本金不足法定原始保證金"
    elif sizing_mode in {"target-leverage", "micro-pilot", "micro-balanced"} and leverage_max_contracts == 0:
        status = "本金不足：1 口會突破槓桿硬上限"
    elif sizing_mode == "micro-pilot" and risk_scale < 1.0:
        status = "小資金高波動風控：減碼改為 0 口"
    elif sizing_mode == "target-leverage" and shock_max_contracts < legal_max_contracts:
        status = f"生存上限啟用：承受 -{survival_shock:.0%} 後仍高於維持保證金"
    elif sizing_mode == "micro-pilot" and shock_max_contracts == 0:
        status = f"本金不足：1 口無法承受 -{survival_shock:.0%} 後仍高於維持保證金"
    elif sizing_mode == "micro-balanced" and shock_max_contracts == 0:
        status = f"本金不足：1 口無法承受 -{survival_shock:.0%} 後仍高於維持保證金"
    elif target == 0:
        status = "目標曝險小於半口，嚴格模式維持 0 口"
    elif sizing_mode == "micro-pilot":
        status = "小資金驗證：正常多頭 1 口，高波動/空頭 0 口"
    elif sizing_mode == "micro-balanced":
        status = "小資金平衡版：EMA 多頭 1 口；高波動只警示，不強制空手"
    elif sizing_mode == "margin-max":
        status = "原始保證金開滿：名目槓桿可能遠高於 7.23x"
    elif abs(actual_exposure - exposure) > 0.25:
        status = "可下單，但整數口數與理論曝險差距較大"
    else:
        status = "可執行"
    return SizingDecision(
        target_contracts=target,
        desired_contracts=desired,
        actual_exposure=actual_exposure,
        one_lot_leverage=contract_notional / equity,
        max_contracts=leverage_max_contracts,
        min_equity_one_lot=min_equity_one_lot,
        two_level_equity=two_level_equity,
        legal_max_contracts=legal_max_contracts,
        initial_margin_required=target * initial_margin,
        sizing_mode=sizing_mode,
        shock_max_contracts=shock_max_contracts,
        shock_equity_after=shock_equity_after,
        shock_margin_buffer=shock_margin_buffer,
        status=status,
    )


def update_paper(
    now,
    sig,
    sizing,
    starting_equity,
    initial_margin=DEFAULT_INITIAL_MARGIN,
    maintenance_margin=DEFAULT_MAINTENANCE_MARGIN,
):
    """同時追蹤理論小數曝險與可成交整數微台口數。"""
    cols = [
        "run_ts", "bar_ts", "trading_day", "close", "ema_fast", "ema_slow",
        "vol", "vol_asof", "exposure", "target_contracts", "actual_exposure",
        "sizing_mode", "initial_margin", "maintenance_margin", "initial_margin_required",
        "survival_shock", "shock_max_contracts", "shock_equity_after", "shock_margin_buffer",
        "paper_equity", "contract_equity", "sizing_status",
        "broker_mode", "broker_before", "broker_after", "broker_verified", "broker_status",
    ]
    if STATE.exists():
        log = pd.read_csv(STATE)
        prev = log.iloc[-1]
        ret = sig["close"] / float(prev["close"]) - 1.0
        eq = float(prev["paper_equity"]) * (1 + float(prev["exposure"]) * ret)
        previous_contracts = int(float(prev.get("target_contracts", 0) or 0))
        previous_contract_equity = float(prev.get("contract_equity", starting_equity))
        contract_eq = previous_contract_equity + previous_contracts * (sig["close"] - float(prev["close"])) * TMF_POINT
    else:
        log = pd.DataFrame(columns=cols)
        eq = 1.0
        contract_eq = starting_equity
    row = {
        "run_ts": now.isoformat(timespec="seconds"),
        "bar_ts": str(sig["last"]),
        "trading_day": str(sig["trading_day"].date()),
        "close": sig["close"],
        "ema_fast": sig["ema_fast"],
        "ema_slow": sig["ema_slow"],
        "vol": sig["vol"],
        "vol_asof": str(sig["vol_asof"].date()),
        "exposure": sig["exposure"],
        "target_contracts": sizing.target_contracts,
        "actual_exposure": sizing.actual_exposure,
        "sizing_mode": sizing.sizing_mode,
        "initial_margin": initial_margin,
        "maintenance_margin": maintenance_margin,
        "initial_margin_required": sizing.initial_margin_required,
        "survival_shock": np.nan,
        "shock_max_contracts": sizing.shock_max_contracts,
        "shock_equity_after": sizing.shock_equity_after,
        "shock_margin_buffer": sizing.shock_margin_buffer,
        "paper_equity": eq,
        "contract_equity": contract_eq,
        "sizing_status": sizing.status,
        "broker_mode": "local-paper",
        "broker_before": np.nan,
        "broker_after": np.nan,
        "broker_verified": False,
        "broker_status": "not_requested",
    }
    return log, row, eq


def save_paper_state(log, row):
    new = pd.DataFrame([row])
    out = new if log.empty else pd.concat([log, new], ignore_index=True)
    out = out.reindex(columns=list(row))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATE, index=False)


def main():
    a = parse_args()
    api, sj = get_api()
    try:
        contract = get_contract(api, a.symbol)
        df = fetch_minutes(api, contract, a.history_days)
        sig = compute_signal(df, a.target_leverage)

        idx = sig["close"]
        sizing = size_micro_contracts(
            sig["exposure"],
            a.equity,
            idx,
            a.max_leverage,
            full_exposure=a.target_leverage,
            initial_margin=a.initial_margin,
            maintenance_margin=a.maintenance_margin,
            survival_shock=a.survival_shock,
            sizing_mode=a.sizing_mode,
            risk_scale=sig["risk_scale"],
        )
        if a.sizing_mode in {"margin-max", "micro-pilot", "micro-balanced"}:
            sig["exposure"] = sizing.actual_exposure

        now = dt.datetime.now()
        log, row, eq = update_paper(
            now, sig, sizing, a.equity, a.initial_margin, a.maintenance_margin
        )
        row["survival_shock"] = a.survival_shock

        print("=" * 56)
        print(f"永豐 Shioaji 模擬盤 paper 測試  {now:%Y-%m-%d %H:%M}")
        code = getattr(contract, "code", a.symbol)
        delivery = getattr(contract, "delivery_month", "")
        print(f"行情 {a.symbol} | 下單合約 {code} {delivery} | 最後 K 棒 {sig['last']}  Close {idx:.0f}")
        print(f"EMA{FAST} {sig['ema_fast']:.1f}  {'>' if sig['long'] else '≤'}  EMA{SLOW} {sig['ema_slow']:.1f} (+{BAND:.0f})")
        print(f"20日年化波動 {sig['vol']*100:.1f}%（截至交易日 {sig['vol_asof'].date()}；"
              f"{'>' if sig['vol']>VOL_TH else '≤'}15% → {'減半' if sig['vol']>VOL_TH else '全額'}）")
        print(f"Sizing：{a.sizing_mode}；訊號：{'做多' if sig['long'] else '空手'}  →  目標曝險 {sig['exposure']:.2f}x")
        print("-" * 56)
        print(f"理論口數 {sizing.desired_contracts:.3f}；嚴格整數目標：{sizing.target_contracts} 口")
        print(f"實際曝險 {sizing.actual_exposure:.2f}x；{sizing.status}")
        print(f"法定原始保證金 {a.initial_margin:,.0f}/口；合法上限 {sizing.legal_max_contracts} 口；"
              f"本次占用 {sizing.initial_margin_required:,.0f}")
        print(f"維持保證金需求：{sizing.target_contracts * a.maintenance_margin:,.0f}")
        if a.sizing_mode == "target-leverage":
            print(f"-{a.survival_shock:.0%} 生存上限：最多 {sizing.shock_max_contracts} 口；"
                  f"壓力後權益 {sizing.shock_equity_after:,.0f}；"
                  f"維持保證金緩衝 {sizing.shock_margin_buffer:,.0f}")
        print(f"1 口微台 = {sizing.one_lot_leverage:.2f}x")
        if a.sizing_mode == "target-leverage":
            print(f"7.23x 目標目前採 {sizing.target_contracts} 口；高波動依整數最接近口數減碼")
        elif a.sizing_mode == "micro-pilot":
            print("micro-pilot：低於 10 萬只用 1/0 口驗證訊號；高波動不硬撐 1 口")
        elif a.sizing_mode == "micro-balanced":
            print("micro-balanced：低於 10 萬用 EMA 1/0 口；高波動只警示，不強制空手")
        else:
            print("margin-max 只受原始保證金限制，不套用 7.23x 名目槓桿上限")
        print(f"模擬權益（exposure 口徑，起始=1）：{eq:.4f}   累計 {len(log)+1} 筆於 {STATE.name}")
        print(f"整數口數 paper 權益：NT${row['contract_equity']:,.0f}")
        print("=" * 56)

        if a.place_sim_orders:
            try:
                roundtrip_status = ""
                if a.sim_roundtrip_test:
                    run_sim_roundtrip_test(api, sj, contract)
                    roundtrip_status = "roundtrip_ok; "
                broker = place_sim_orders(api, sj, contract, sizing.target_contracts)
                broker["broker_status"] = roundtrip_status + broker["broker_status"]
                row.update(broker)
            except Exception as exc:
                row.update({
                    "broker_mode": "shioaji-sim",
                    "broker_verified": False,
                    "broker_status": f"failed: {type(exc).__name__}: {exc}",
                })
                save_paper_state(log, row)
                raise
        else:
            print("（paper 模式：未送任何單。要在 Shioaji 模擬帳戶實際下單請加 --place-sim-orders 並先驗證 API）")
        save_paper_state(log, row)
    finally:
        try:
            api.logout()
        except Exception:
            pass


def place_sim_orders(api, sj, contract, target_contracts):
    """把 Shioaji 模擬帳戶的指定合約部位調到 target_contracts 並驗證。"""
    account = api.futopt_account
    contract_code = getattr(contract, "code", "")
    print(f"（Shioaji simulation=True；模擬帳戶 {getattr(account, 'account_id', '?')}）")
    held, positions = get_sim_position(api, sj, account, contract_code)
    print(f"送單前 {contract_code} 部位：{held} 口；模擬帳戶總部位筆數：{len(positions)}")
    delta = target_contracts - held
    if delta == 0:
        print("券商模擬部位已符合目標，查詢路徑驗證成功，無需送單。")
        return {
            "broker_mode": "shioaji-sim",
            "broker_before": held,
            "broker_after": held,
            "broker_verified": True,
            "broker_status": "position_already_matched",
        }

    action = sj.Action.Buy if delta > 0 else sj.Action.Sell
    order = api.Order(
        action=action, price=0, quantity=abs(delta),
        price_type=sj.FuturesPriceType.MKP,
        order_type=sj.OrderType.IOC,
        octype=sj.FuturesOCType.Auto,
        account=account,
    )
    trade = api.place_order(contract, order)
    print(f"已送模擬單：{action.value} {abs(delta)} 口")

    final_held = held
    for _ in range(10):
        time.sleep(0.5)
        api.update_status(account, trade=trade)
        final_held, _ = get_sim_position(api, sj, account, contract_code)
        if final_held == target_contracts:
            break

    status = getattr(getattr(trade, "status", None), "status", "unknown")
    status_code = getattr(getattr(trade, "status", None), "status_code", "")
    deal_quantity = getattr(getattr(trade, "status", None), "deal_quantity", 0)
    print(f"委託狀態：{status}；status_code={status_code}；成交 {deal_quantity} 口")
    print(f"送單後 {contract_code} 部位：{final_held} 口；目標：{target_contracts} 口")
    if final_held != target_contracts:
        raise RuntimeError("模擬委託後部位未達目標；請查看委託狀態，程式不會宣稱成功")
    print("券商模擬下單與部位驗證成功。")
    return {
        "broker_mode": "shioaji-sim",
        "broker_before": held,
        "broker_after": final_held,
        "broker_verified": True,
        "broker_status": f"{status}; code={status_code}; deals={deal_quantity}",
    }


def get_sim_position(api, sj, account, contract_code):
    api.update_status(account)
    positions = list(api.list_positions(account))
    held = 0
    for position in positions:
        if getattr(position, "code", "") != contract_code:
            continue
        direction = getattr(position, "direction", "")
        direction_value = getattr(direction, "value", str(direction))
        sign = 1 if direction in (sj.Action.Buy, "Buy") or direction_value == "Buy" else -1
        held += sign * int(getattr(position, "quantity", 0))
    return held, positions


def run_sim_roundtrip_test(api, sj, contract):
    """只在模擬帳戶測試一買一賣，並強制驗證最後回到空手。"""
    account = api.futopt_account
    contract_code = getattr(contract, "code", "")
    held, _ = get_sim_position(api, sj, account, contract_code)
    if held != 0:
        raise RuntimeError(f"roundtrip 測試要求起始 0 口，目前為 {held} 口")
    print("開始 Shioaji 模擬帳戶 roundtrip：買 1 口，再賣回 0 口。")
    try:
        place_sim_orders(api, sj, contract, 1)
    finally:
        cleanup_held, _ = get_sim_position(api, sj, account, contract_code)
        if cleanup_held != 0:
            place_sim_orders(api, sj, contract, 0)
    final_held, _ = get_sim_position(api, sj, account, contract_code)
    if final_held != 0:
        raise RuntimeError(f"roundtrip 結束後仍有 {final_held} 口，請立即檢查模擬帳戶")
    print("Shioaji 模擬 roundtrip 成功，最終部位 0 口。")


if __name__ == "__main__":
    main()
