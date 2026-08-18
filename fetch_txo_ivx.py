# -*- coding: utf-8 -*-
"""台指 VIX proxy (tw_ivx) 資料管線。stdlib-only (環境無 requests, 網路走 urllib)。
預註冊規格見 ~/trading-wiki/raw/reports/txf-vix-proxy-2026-07-06.md 第 2 節 (先釘死才寫的本檔)。

  fetch : FinMind TaiwanOptionDaily(TXO) 逐月抓 → 過濾月合約/±18% moneyness/T<=70d → txo_chain_monthly_atm.csv.gz (可續傳)
  build : 鏈檔 + taiex_daily.csv → 30 天 ATM 隱含波動 proxy → tw_ivx_daily.csv
  check : 覆蓋率 Q3 / 歷史錨點 Q2 / 對照官方近月 Q1 (official_vix_daily.csv)

用法: python3 fetch_txo_ivx.py fetch [--start 2006-12] [--end 2026-07] [--only 2015-06]
      python3 fetch_txo_ivx.py build
      python3 fetch_txo_ivx.py check
"""
import argparse, csv, datetime as dt, gzip, json, math, os, re, sys, time
import urllib.parse, urllib.request

D = "/Users/guichenxiang/txf_backtest/finmind_data"
CHAIN = os.path.join(D, "txo_chain_monthly_atm.csv.gz")
STATE = os.path.join(D, "txo_fetch_state.json")
IVX = os.path.join(D, "tw_ivx_daily.csv")
OFFICIAL = os.path.join(D, "official_vix_daily.csv")
SPOT = os.path.join(D, "taiex_daily.csv")
URL = "https://api.finmindtrade.com/api/v4/data"
BACKOFF = [30, 60, 120, 300, 600, 900]


def third_wed(y, m):
    first = dt.date(y, m, 1)
    return first + dt.timedelta(days=(2 - first.weekday()) % 7 + 14)


def load_spot():
    out = {}
    with open(SPOT, newline="") as f:
        for r in csv.DictReader(f):
            out[r["date"]] = float(r["close"])
    return out


def month_range(start, end):
    y, m = map(int, start.split("-")); ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13: y, m = y + 1, 1


def fetch_month(ym, spot):
    y, m = map(int, ym.split("-"))
    last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    q = urllib.parse.urlencode({
        "dataset": "TaiwanOptionDaily", "data_id": "TXO",
        "start_date": f"{ym}-01", "end_date": f"{ym}-{last:02d}"})
    for i, wait in enumerate(BACKOFF + [None]):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=180) as r:
                d = json.loads(r.read().decode())
            if d.get("msg") != "success":
                raise RuntimeError(f"api msg={d.get('msg')}")
            break
        except Exception as e:
            if wait is None:
                raise
            print(f"  {ym} 失敗({type(e).__name__}: {e}), {wait}s 後重試", flush=True)
            time.sleep(wait)
    kept, days = [], set()
    for row in d.get("data", []):
        c = str(row.get("contract_date", ""))
        if not (len(c) == 6 and c.isdigit()):
            continue  # 只留月合約 (預註冊 2a-1)
        if row.get("trading_session", "position") != "position":
            continue
        date = row["date"]; days.add(date)
        s = spot.get(date)
        settle = float(row.get("settlement_price") or 0)
        if s is None or settle <= 0:
            continue
        k = float(row["strike_price"])
        if abs(k - s) > 0.18 * s:
            continue  # fetch 留 ±18% 超集, build 再套正式 ±15%
        exp = third_wed(int(c[:4]), int(c[4:]))
        t_days = (exp - dt.date(*map(int, date.split("-")))).days
        if not (1 <= t_days <= 70):
            continue  # 只需近月+次月
        kept.append((date, c, f"{k:g}", row["call_put"][0], f"{settle:g}"))
    return kept, len(days)


def cmd_fetch(a):
    spot = load_spot()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"done": {}}
    months = [a.only] if a.only else [ym for ym in month_range(a.start, a.end)]
    todo = [ym for ym in months if ym not in state["done"]]
    print(f"待抓 {len(todo)}/{len(months)} 個月", flush=True)
    for ym in todo:
        t0 = time.time()
        kept, ndays = fetch_month(ym, spot)
        with gzip.open(CHAIN, "at", newline="") as f:
            w = csv.writer(f)
            for r in kept: w.writerow(r)
        state["done"][ym] = {"rows": len(kept), "days": ndays}
        json.dump(state, open(STATE, "w"))
        print(f"{ym}: {len(kept)} rows / {ndays} days ({time.time()-t0:.1f}s)", flush=True)
        time.sleep(a.sleep)
    print("fetch 完成", flush=True)


def load_chain():
    chain = {}
    with gzip.open(CHAIN, "rt", newline="") as f:
        for date, c, k, cp, settle in csv.reader(f):
            chain.setdefault(date, {})[(c, float(k), cp)] = float(settle)  # dedupe keep-last
    return chain


def term_sigma(quotes, s, t_days):
    """quotes: {(K,cp):settle} 單一到期月。回傳 sigma 或 None。預註冊 2a-4."""
    t = t_days / 365.0
    ks = sorted({k for k, cp in quotes})
    cand = [k for k in ks if abs(k - s) <= 0.15 * s
            and quotes.get((k, "c"), 0) > 0.1 and quotes.get((k, "p"), 0) > 0.1]
    if len(cand) < 4:
        return None
    khat = min(cand, key=lambda k: abs(quotes[(k, "c")] - quotes[(k, "p")]))
    fwd = khat + quotes[(khat, "c")] - quotes[(khat, "p")]
    kstar = min(cand, key=lambda k: abs(k - fwd))
    straddle = quotes[(kstar, "c")] + quotes[(kstar, "p")]
    if fwd <= 0 or straddle <= 0:
        return None
    # 跨式版 Brenner–Subrahmanyam: straddle ≈ 2·0.3989·F·σ√T → σ = √(π/2T)·straddle/F
    # (預註冊原寫 √(2π/T) 為單腿常數誤植, 2015-06 樣本月驗證抓到 2 倍偏差後勘誤, 見報告 2a)
    sig = math.sqrt(math.pi / (2 * t)) * straddle / fwd
    return sig if 0.01 < sig < 2.5 else None


def cmd_build(a):
    spot = load_spot(); chain = load_chain()
    out = []
    for date in sorted(chain):
        s = spot.get(date)
        if s is None: continue
        d0 = dt.date(*map(int, date.split("-")))
        terms = {}
        for (c, k, cp), settle in chain[date].items():
            terms.setdefault(c, {})[(k, cp)] = settle
        info = []
        for c, quotes in terms.items():
            t_days = (third_wed(int(c[:4]), int(c[4:])) - d0).days
            if t_days < 8: continue  # CBOE 8 天規則
            sig = term_sigma(quotes, s, t_days)
            if sig: info.append((t_days, sig))
        info.sort()
        if not info: continue
        if len(info) == 1:
            n1, s1 = info[0]; ivx = s1; n2 = ""
        else:
            (n1, s1), (n2, s2) = info[0], info[1]
            t1, t2 = n1 / 365.0, n2 / 365.0
            var30 = (t1 * s1 * s1 * (n2 - 30) + t2 * s2 * s2 * (30 - n1)) / (n2 - n1) * (365.0 / 30)
            ivx = math.sqrt(var30) if var30 > 0 else s1
        out.append((date, f"{ivx*100:.2f}", n1, n2))
    with open(IVX, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "ivx", "near_days", "next_days"])
        for r in out: w.writerow(r)
    print(f"build 完成: {len(out)} days -> {IVX}")
    if out:
        print("首末:", out[0], out[-1])


def pearson(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx * sy > 0 else float("nan")


def cmd_check(a):
    spot = load_spot()
    ivx = {}
    with open(IVX, newline="") as f:
        for r in csv.DictReader(f):
            ivx[r["date"]] = float(r["ivx"])
    days = sorted(d for d in spot if "2007-01-01" <= d)
    have = [d for d in days if d in ivx]
    print(f"Q3 覆蓋率: {len(have)}/{len(days)} = {len(have)/len(days)*100:.1f}% (門檻 97%)")
    miss_month = {}
    for d in days:
        if d not in ivx: miss_month[d[:7]] = miss_month.get(d[:7], 0) + 1
    worst = sorted(miss_month.items(), key=lambda kv: -kv[1])[:8]
    print("  缺最多的月份:", worst)
    vals = sorted(ivx.values()); p75 = vals[int(0.75 * len(vals))]
    print(f"Q2 錨點 (全樣本 p75 = {p75:.1f}):")
    anchors = ["2008-10", "2008-11", "2011-08", "2015-08", "2018-10", "2020-03",
               "2022-01", "2022-09", "2024-08", "2025-04"]
    for a_ in anchors:
        mv = [v for d, v in ivx.items() if d.startswith(a_)]
        avg = sum(mv) / len(mv) if mv else float("nan")
        print(f"  {a_}: 月均 {avg:.1f} {'✓' if mv and avg > p75 else '✗'} (n={len(mv)})")
    if os.path.exists(OFFICIAL):
        off = {}
        with open(OFFICIAL, newline="") as f:
            for r in csv.DictReader(f):
                off[r["date"]] = float(r["vix_close"])
        common = sorted(set(off) & set(ivx))
        xs = [off[d] for d in common]; ys = [ivx[d] for d in common]
        dx = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        dy = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        print(f"Q1 官方對照 ({len(common)} 天重疊 {common[0]}~{common[-1]}):")
        print(f"  水平 r = {pearson(xs, ys):.3f} (門檻 0.95) | 日變化 r = {pearson(dx, dy):.3f} (門檻 0.85)")
        print(f"  平均差 官方-proxy = {sum(x-y for x,y in zip(xs,ys))/len(xs):+.2f} vol pts")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch")
    pf.add_argument("--start", default="2006-12"); pf.add_argument("--end", default="2026-07")
    pf.add_argument("--only"); pf.add_argument("--sleep", type=float, default=1.0)
    sub.add_parser("build"); sub.add_parser("check")
    a = p.parse_args()
    {"fetch": cmd_fetch, "build": cmd_build, "check": cmd_check}[a.cmd](a)
