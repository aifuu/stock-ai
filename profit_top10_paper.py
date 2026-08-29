import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from daily_directional_top1 import (
    TICKERS,
    NAMES,
    download,
    make_nikkei,
    load_model,
    features,
    atr,
    directional_score,
)

TZ = ZoneInfo("Asia/Tokyo")
POLICY_FILE = "strategy_policy.json"
STATE_FILE = "profit_top10_paper_state.json"
HISTORY_FILE = "profit_top10_paper_history.csv"
MONTHLY_FILE = "profit_top10_monthly_performance.csv"
INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
TOP_N = 10
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))


def load_policy():
    if not os.path.exists(POLICY_FILE):
        raise RuntimeError("strategy_policy.json がありません。先にProfit Optimizerを実行してください。")
    with open(POLICY_FILE, encoding="utf-8") as f:
        p = json.load(f)
    required = ["status", "up_threshold", "min_score_for_buy", "nikkei_filter", "atr_tp_multiplier", "atr_sl_multiplier", "hold_days"]
    missing = [k for k in required if k not in p]
    if missing:
        raise RuntimeError("strategy_policy.json の不足項目: " + ", ".join(missing))
    if str(p.get("status", "")).upper() != "APPROVED":
        raise RuntimeError(f"strategy_policy.json がAPPROVEDではありません: status={p.get('status')}。安全のため新規取引を停止します。")
    p["up_threshold"] = float(p["up_threshold"])
    p["min_score_for_buy"] = float(p["min_score_for_buy"])
    p["nikkei_filter"] = str(p["nikkei_filter"]).lower() in ("true", "1", "yes", "on")
    p["atr_tp_multiplier"] = float(p["atr_tp_multiplier"])
    p["atr_sl_multiplier"] = float(p["atr_sl_multiplier"])
    p["hold_days"] = int(p["hold_days"])
    if not (0 < p["up_threshold"] <= 100):
        raise RuntimeError("policy up_threshold が不正です")
    if p["atr_tp_multiplier"] <= 0 or p["atr_sl_multiplier"] <= 0 or p["hold_days"] <= 0:
        raise RuntimeError("policy TP/SL/hold_days が不正です")
    return p


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                s = json.load(f)
            s.setdefault("capital", INITIAL_CAPITAL)
            s.setdefault("peak", s["capital"])
            s.setdefault("max_dd", 0.0)
            s.setdefault("positions", [])
            s.setdefault("last_entry_date", None)
            return s
        except Exception:
            pass
    return {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL, "max_dd": 0.0, "positions": [], "last_entry_date": None}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def append_history(row):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        df = pd.concat([pd.read_csv(HISTORY_FILE), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def update_monthly():
    if not os.path.exists(HISTORY_FILE):
        return None
    df = pd.read_csv(HISTORY_FILE)
    if df.empty:
        return None
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df = df.dropna(subset=["exit_date", "pnl"])
    if df.empty:
        return None
    m = (
        df.assign(month=df["exit_date"].dt.to_period("M"))
        .groupby("month")
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"), avg_return=("return_pct", "mean"), wins=("pnl", lambda s: int((s > 0).sum())))
        .reset_index()
    )
    m["win_rate_pct"] = m["wins"] / m["trades"] * 100
    m.to_csv(MONTHLY_FILE, index=False, encoding="utf-8-sig")
    return m.iloc[-1].to_dict()


def close_positions(state, policy):
    remaining = []
    messages = []
    for p in state["positions"]:
        df = download(p["ticker"], period="3mo")
        if df is None or df.empty:
            remaining.append(p)
            continue
        entry_date = pd.Timestamp(p["entry_date"])
        today = pd.Timestamp(datetime.now(TZ).date())
        bars = df[df.index.normalize() > entry_date.normalize()]
        bars = bars[bars.index.normalize() <= today]
        if bars.empty:
            remaining.append(p)
            continue
        exit_reason = None
        exit_price = None
        exit_date = None
        for idx, bar in bars.iterrows():
            high, low = float(bar["High"]), float(bar["Low"])
            if low <= p["sl"] and high >= p["tp"]:
                exit_reason, exit_price = "SL", p["sl"]
            elif high >= p["tp"]:
                exit_reason, exit_price = "TP", p["tp"]
            elif low <= p["sl"]:
                exit_reason, exit_price = "SL", p["sl"]
            if exit_reason:
                exit_date = idx
                break
        if exit_reason is None and len(bars) >= policy["hold_days"]:
            exit_date = bars.index[policy["hold_days"] - 1]
            exit_price = float(bars.iloc[policy["hold_days"] - 1]["Close"])
            exit_reason = "TIME"
        if exit_reason is None:
            remaining.append(p)
            continue
        entry = float(p["entry_price"])
        gross_ret = (float(exit_price) / entry - 1) * 100
        net_ret = gross_ret - FEE_RATE * 200
        allocation = float(p.get("allocation", 1.0 / TOP_N))
        pnl = state["capital"] * allocation * net_ret / 100
        state["capital"] += pnl
        append_history({
            "entry_date": p["entry_date"],
            "exit_date": str(pd.Timestamp(exit_date).date()),
            "ticker": p["ticker"],
            "company": p["company"],
            "entry_price": entry,
            "exit_price": float(exit_price),
            "tp": p["tp"],
            "sl": p["sl"],
            "score": p["score"],
            "up_probability": p["up_probability"],
            "return_pct": round(net_ret, 3),
            "pnl": round(pnl, 2),
            "result": exit_reason,
            "hold_days": len(pd.bdate_range(entry_date, pd.Timestamp(exit_date))),
            "allocation": allocation,
            "policy_updated_at": p.get("policy_updated_at"),
        })
        messages.append(f"決済 {p['ticker']} {exit_reason} {net_ret:+.2f}%")
    state["positions"] = remaining
    state["peak"] = max(float(state.get("peak", state["capital"])), float(state["capital"]))
    if state["peak"]:
        state["max_dd"] = max(state.get("max_dd", 0.0), (state["peak"] - state["capital"]) / state["peak"] * 100)
    return messages


def scan_candidates(policy):
    nikkei = make_nikkei()
    model = load_model()
    if nikkei is None or model is None:
        raise RuntimeError("日経データまたはAIモデルを取得できませんでした")
    candidates = []
    scanned = 0
    for ticker in TICKERS:
        df = download(ticker)
        if df is None or len(df) < 150:
            continue
        scanned += 1
        feature_cols = list(getattr(model, "feature_names_in_", []))
        x = features(df, nikkei).dropna(subset=feature_cols)
        if x.empty:
            continue
        try:
            last = x.iloc[-1]
            probs = model.predict_proba(x.iloc[-1:])[0]
            classes = list(model.classes_)
            if not all(c in classes for c in (0, 1, 2)):
                continue
            down = float(probs[classes.index(0)])
            up = float(probs[classes.index(2)])
            flat = float(probs[classes.index(1)])
            long_s, _ = directional_score(last, up, down)
            score = float(long_s)
            price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            if not np.isfinite(a) or a <= 0:
                continue
            if up * 100 < policy["up_threshold"] or up <= down or flat >= 50:
                continue
            if score < policy["min_score_for_buy"]:
                continue
            if policy["nikkei_filter"]:
                nlast = nikkei.reindex(x.index).ffill().iloc[-1]
                if not (float(nlast["kairi25"]) > 0 and float(nlast["ret5"]) > 0):
                    continue
            candidates.append({
                "ticker": ticker,
                "company": NAMES.get(ticker, ticker),
                "score": score,
                "up_probability": up * 100,
                "down_probability": down * 100,
                "flat_probability": flat * 100,
                "price": price,
                "tp": price + a * policy["atr_tp_multiplier"],
                "sl": price - a * policy["atr_sl_multiplier"],
                "data_date": str(x.index[-1].date()),
            })
        except Exception as e:
            print(ticker, "predict", e)
    candidates.sort(key=lambda z: (z["score"], z["up_probability"]), reverse=True)
    return candidates, scanned


def main():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    policy = load_policy()
    state = load_state()
    closed = close_positions(state, policy)

    existing = {p["ticker"] for p in state["positions"]}
    candidates, scanned = scan_candidates(policy)

    # 同一営業日にTOP10を何度も追加しない。5分cronは決済確認専用として動かす。
    if state.get("last_entry_date") != today:
        selected = [c for c in candidates if c["ticker"] not in existing][:TOP_N]
        allocation = 1.0 / TOP_N
        for c in selected:
            state["positions"].append({
                "entry_date": today,
                "ticker": c["ticker"],
                "company": c["company"],
                "entry_price": c["price"],
                "tp": c["tp"],
                "sl": c["sl"],
                "score": c["score"],
                "up_probability": c["up_probability"],
                "down_probability": c["down_probability"],
                "allocation": allocation,
                "policy_updated_at": policy.get("updated_at"),
            })
        state["last_entry_date"] = today

    save_state(state)
    monthly = update_monthly()

    print("=" * 72)
    print("🤖 PROFIT LOOP｜TOP10 PAPER TRADE")
    print(f"📅 {today} | scanned={scanned} | candidates={len(candidates)}")
    print("🔗 policy status:", policy["status"])
    print("🔗 policy updated:", policy.get("updated_at"))
    print("条件:", f"UP>={policy['up_threshold']}% / SCORE>={policy['min_score_for_buy']} / TP={policy['atr_tp_multiplier']}ATR / SL={policy['atr_sl_multiplier']}ATR / HOLD={policy['hold_days']}日 / 日経={policy['nikkei_filter']}")
    print("📌 positions:", len(state["positions"]))
    for i, p in enumerate(state["positions"], 1):
        print(f"{i:02d}. {p['ticker']} score={p['score']:.1f} UP={p['up_probability']:.1f}% entry={p['entry_price']:.0f}")
    print(f"💰 capital={state['capital']:,.0f}円 DD={state['max_dd']:.2f}%")
    if monthly:
        print(f"📅 month={monthly['month']} pnl={monthly['pnl']:+,.0f}円 win={monthly['win_rate_pct']:.1f}% trades={int(monthly['trades'])}")
    for m in closed:
        print(m)

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        import requests
        top_text = "\n".join(
            f"{i}. BUY {p['ticker']}｜score {p['score']:.1f}｜UP {p['up_probability']:.1f}%｜entry {p['entry_price']:,.0f}｜TP {p['tp']:,.0f}｜SL {p['sl']:,.0f}"
            for i, p in enumerate(state["positions"], 1)
        ) or "条件成立銘柄なし"
        month_text = "確定取引なし" if not monthly else f"今月損益 {monthly['pnl']:+,.0f}円｜取引 {int(monthly['trades'])}｜勝率 {monthly['win_rate_pct']:.1f}%"
        msg = (
            "🤖 PROFIT LOOP｜毎日TOP10ペーパートレード\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📅 {today}\n⚠️ 実注文なし\n\n"
            f"🔗 Policy: APPROVED｜更新 {policy.get('updated_at')}\n"
            f"条件: UP≥{policy['up_threshold']:.0f}% / SCORE≥{policy['min_score_for_buy']:.0f} / TP {policy['atr_tp_multiplier']:.2f}ATR / SL {policy['atr_sl_multiplier']:.2f}ATR / {policy['hold_days']}営業日\n"
            f"100銘柄対象｜データ取得 {scanned}銘柄｜候補 {len(candidates)}銘柄\n\n"
            f"🏆 保有TOP10\n{top_text}\n\n"
            f"💰 仮想資産 {state['capital']:,.0f}円｜最大DD {state['max_dd']:.2f}%\n"
            f"📅 {month_text}\n\n"
            "① Profit Optimizer → ② strategy_policy.json → ③ TOP10 → ④ 月間損益 → ⑤ 再検証・再学習 の完全ループ"
        )
        requests.post(webhook, json={"content": msg[:1950]}, timeout=30).raise_for_status()


if __name__ == "__main__":
    main()
