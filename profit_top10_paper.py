import json
import os
from collections import Counter
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from daily_directional_top1 import (
    TICKERS,
    NAMES,
    download,
    make_nikkei,
    load_model,
    features,
    atr,
    directional_score,
    append_history,
)

TZ = ZoneInfo("Asia/Tokyo")
POLICY_FILE = "strategy_policy.json"
STATE_FILE = "profit_top10_paper_state.json"
HISTORY_FILE = "profit_top10_paper_history.csv"
MONTHLY_FILE = "profit_top10_monthly_performance.csv"
INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
TOP_N = 10
# 同一銘柄は「決済後」であれば1日に最大10回まで再エントリー可能。
MAX_TRADES_PER_TICKER_PER_DAY = 10
# 暴走防止のため全銘柄合計にも上限を設ける。
MAX_TOTAL_TRADES_PER_DAY = 30
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
FORCED_EXIT = dtime(15, 25)


def load_policy():
    if not os.path.exists(POLICY_FILE):
        raise RuntimeError("strategy_policy.json がありません。先にProfit Optimizerを実行してください。")
    with open(POLICY_FILE, encoding="utf-8") as f:
        p = json.load(f)
    required = [
        "status", "up_threshold", "min_score_for_buy", "nikkei_filter",
        "atr_tp_multiplier", "atr_sl_multiplier", "hold_days",
    ]
    missing = [k for k in required if k not in p]
    if missing:
        raise RuntimeError("strategy_policy.json の不足項目: " + ", ".join(missing))
    if str(p.get("status", "")).upper() != "APPROVED":
        raise RuntimeError(
            f"strategy_policy.json がAPPROVEDではありません: status={p.get('status')}。安全のため新規取引を停止します。"
        )
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


def default_state():
    return {
        "capital": INITIAL_CAPITAL,
        "peak": INITIAL_CAPITAL,
        "max_dd": 0.0,
        "positions": [],
        "trade_count_date": None,
        "trades_today": 0,
        "trades_by_ticker_today": {},
        "daily_start_capital": INITIAL_CAPITAL,
    }


def load_state():
    base = default_state()
    if not os.path.exists(STATE_FILE):
        return base
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            base.update(saved)
        base.setdefault("positions", [])
        base.setdefault("trades_by_ticker_today", {})
        return base
    except Exception:
        return base


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def reset_daily_counter(state, today):
    if state.get("trade_count_date") != today:
        state["trade_count_date"] = today
        state["trades_today"] = 0
        state["trades_by_ticker_today"] = {}
        state["daily_start_capital"] = float(state.get("capital", INITIAL_CAPITAL))


def download_5m(ticker):
    try:
        df = yf.download(
            ticker, period="5d", interval="5m", auto_adjust=False,
            progress=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(TZ).tz_localize(None)
        else:
            idx = idx.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
        df = df.copy()
        df.index = idx
        return df.sort_index()
    except Exception as exc:
        print(f"5分足取得失敗 {ticker}: {exc}")
        return None


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
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            avg_return=("return_pct", "mean"),
            wins=("pnl", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    m["win_rate_pct"] = m["wins"] / m["trades"] * 100
    m.to_csv(MONTHLY_FILE, index=False, encoding="utf-8-sig")
    return m.iloc[-1].to_dict()


def close_positions(state, policy, now):
    remaining = []
    messages = []
    for p in state["positions"]:
        df = download_5m(p["ticker"])
        if df is None or df.empty:
            remaining.append(p)
            continue

        entry_date = str(p.get("entry_date", now.strftime("%Y-%m-%d")))
        entry_time = str(p.get("entry_time", "09:00"))
        entry_price = float(p["entry_price"])
        tp = float(p["tp"])
        sl = float(p["sl"])
        today = df[df.index.date == now.date()]
        if today.empty:
            remaining.append(p)
            continue
        now_naive = now.replace(tzinfo=None)
        if entry_date == now.strftime("%Y-%m-%d"):
            bars = today[(today.index >= pd.Timestamp(f"{entry_date} {entry_time}")) & (today.index <= now_naive)]
        else:
            bars = today[today.index <= now_naive]
        if bars.empty:
            remaining.append(p)
            continue

        exit_price = None
        reason = None
        exit_time = None
        for ts, bar in bars.iterrows():
            high, low = float(bar["High"]), float(bar["Low"])
            if low <= sl and high >= tp:
                exit_price, reason = sl, "SL_BOTH"
            elif high >= tp:
                exit_price, reason = tp, "TP"
            elif low <= sl:
                exit_price, reason = sl, "SL"
            if reason:
                exit_time = ts
                break
            if ts.time() >= FORCED_EXIT:
                exit_price, reason, exit_time = float(bar["Close"]), "EOD", ts
                break

        if exit_price is None and now.time() >= FORCED_EXIT:
            bar = bars.iloc[-1]
            exit_price, reason, exit_time = float(bar["Close"]), "EOD", bars.index[-1]
        if exit_price is None:
            remaining.append(p)
            continue

        gross_ret = (float(exit_price) / entry_price - 1.0) * 100.0
        net_ret = gross_ret - FEE_RATE * 200.0
        allocation = float(p.get("allocation", 1.0 / TOP_N))
        capital_before = float(state["capital"])
        pnl = capital_before * allocation * net_ret / 100.0
        state["capital"] = capital_before + pnl
        append_history({
            "entry_date": p.get("entry_date", entry_date),
            "entry_time": p.get("entry_time", entry_time),
            "exit_date": str(pd.Timestamp(exit_time).date()),
            "exit_time": pd.Timestamp(exit_time).strftime("%H:%M"),
            "ticker": p["ticker"],
            "company": p["company"],
            "direction": "BUY",
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "tp": tp,
            "sl": sl,
            "score": p["score"],
            "up_probability": p["up_probability"],
            "return_pct": round(net_ret, 3),
            "pnl": round(pnl, 2),
            "result": reason,
            "hold_days": 1,
            "allocation": allocation,
            "policy_updated_at": p.get("policy_updated_at"),
        })
        messages.append(f"決済 {p['ticker']} {reason} {net_ret:+.2f}%")

    state["positions"] = remaining
    state["peak"] = max(float(state.get("peak", state["capital"])), float(state["capital"]))
    if state["peak"]:
        state["max_dd"] = max(
            float(state.get("max_dd", 0.0)),
            (state["peak"] - state["capital"]) / state["peak"] * 100.0,
        )
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
            daily_price = float(df["Close"].iloc[-1])
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

            # 実行価格は可能なら最新5分足を使用。TP/SLの距離はOptimizerのATR条件を維持。
            intraday = download_5m(ticker)
            price = daily_price
            if intraday is not None and not intraday.empty:
                latest = intraday[intraday.index <= datetime.now(TZ).replace(tzinfo=None)]
                if not latest.empty:
                    price = float(latest["Close"].iloc[-1])
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


def open_positions(state, policy, candidates, today):
    active = {p["ticker"] for p in state["positions"]}
    selected = []
    for c in candidates:
        ticker = c["ticker"]
        # 保有中は追加買いしない。決済後の再エントリーだけ許可。
        if ticker in active:
            continue
        ticker_count = int(state.get("trades_by_ticker_today", {}).get(ticker, 0))
        if ticker_count >= MAX_TRADES_PER_TICKER_PER_DAY:
            continue
        if int(state.get("trades_today", 0)) >= MAX_TOTAL_TRADES_PER_DAY:
            break
        if len(state["positions"]) >= TOP_N:
            break

        allocation = 1.0 / TOP_N
        state["positions"].append({
            "entry_date": today,
            "entry_time": datetime.now(TZ).strftime("%H:%M"),
            "ticker": ticker,
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
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        counts = state.setdefault("trades_by_ticker_today", {})
        counts[ticker] = ticker_count + 1
        active.add(ticker)
        selected.append(c)
    return selected


def main():
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")
    policy = load_policy()
    state = load_state()
    reset_daily_counter(state, today)
    save_state(state)

    if now.weekday() >= 5 or now.time() < dtime(9, 0) or now.time() > dtime(15, 30):
        return

    start_capital = float(state.get("daily_start_capital", state.get("capital", INITIAL_CAPITAL)))
    daily_return = ((float(state.get("capital", 0.0)) / start_capital) - 1.0) * 100.0 if start_capital else 0.0
    if daily_return <= -1.5:
        close_positions(state, policy, now)
        save_state(state)
        print(f"🛑 日次損失上限 -1.5%｜新規停止｜日次損益 {daily_return:+.2f}%")
        return

    closed = close_positions(state, policy, now)

    # 決済で空いた枠へ、毎サイクル再スキャンして補充。
    candidates, scanned = scan_candidates(policy)
    opened = open_positions(state, policy, candidates, today)
    save_state(state)
    monthly = update_monthly()

    print("=" * 72)
    print("🤖 PROFIT LOOP｜TOP10 PAPER TRADE")
    print(f"📅 {today} {now:%H:%M} | scanned={scanned} | candidates={len(candidates)}")
    print("🔗 policy:", policy["status"], policy.get("updated_at"))
    print(
        "条件:",
        f"UP>={policy['up_threshold']}% / SCORE>={policy['min_score_for_buy']} / "
        f"TP={policy['atr_tp_multiplier']}ATR / SL={policy['atr_sl_multiplier']}ATR / "
        f"Optimizer HOLD={policy['hold_days']}日 / 日経={policy['nikkei_filter']}"
    )
    print(
        f"🔁 再エントリー: 同一銘柄 最大{MAX_TRADES_PER_TICKER_PER_DAY}回/日 "
        f"｜全銘柄 最大{MAX_TOTAL_TRADES_PER_DAY}回/日"
    )
    print(f"📌 新規累計: {state.get('trades_today', 0)}/{MAX_TOTAL_TRADES_PER_DAY}｜保有: {len(state['positions'])}/{TOP_N}")
    for i, p in enumerate(state["positions"], 1):
        print(f"{i:02d}. {p['ticker']} score={p['score']:.1f} UP={p['up_probability']:.1f}% entry={p['entry_price']:.1f}")
    print(f"💰 capital={state['capital']:,.0f}円 DD={state['max_dd']:.2f}%")
    if monthly:
        print(f"📅 month={monthly['month']} pnl={monthly['pnl']:+,.0f}円 win={monthly['win_rate_pct']:.1f}% trades={int(monthly['trades'])}")
    for m in closed:
        print(m)

    webhook = os.getenv("DISCORD_WEBHOOK")
    if webhook:
        import requests
        top_text = "\n".join(
            f"{i}. BUY {p['ticker']}｜score {p['score']:.1f}｜UP {p['up_probability']:.1f}%｜entry {p['entry_price']:,.1f}｜TP {p['tp']:,.1f}｜SL {p['sl']:,.1f}"
            for i, p in enumerate(state["positions"], 1)
        ) or "条件成立銘柄なし"
        counts = ", ".join(f"{k}:{v}回" for k, v in sorted(state.get("trades_by_ticker_today", {}).items())) or "なし"
        month_text = "確定取引なし" if not monthly else f"今月損益 {monthly['pnl']:+,.0f}円｜取引 {int(monthly['trades'])}｜勝率 {monthly['win_rate_pct']:.1f}%"
        msg = (
            "🤖 PROFIT LOOP｜TOP10 5分足ペーパートレード\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📅 {today} {now:%H:%M} JST\n⚠️ 実注文なし\n\n"
            f"🔗 Policy: APPROVED｜更新 {policy.get('updated_at')}\n"
            f"条件: UP≥{policy['up_threshold']:.0f}% / SCORE≥{policy['min_score_for_buy']:.0f}% / TP {policy['atr_tp_multiplier']:.2f}ATR / SL {policy['atr_sl_multiplier']:.2f}ATR\n"
            f"100銘柄対象｜取得成功 {scanned}｜候補 {len(candidates)}\n"
            f"🔁 同一銘柄 最大{MAX_TRADES_PER_TICKER_PER_DAY}回/日｜全体 最大{MAX_TOTAL_TRADES_PER_DAY}回/日\n"
            f"📊 本日銘柄別取引回数: {counts}\n\n"
            f"🏆 保有TOP10\n{top_text}\n\n"
            f"💰 仮想資産 {state['capital']:,.0f}円｜最大DD {state['max_dd']:.2f}%\n"
            f"📅 {month_text}\n\n"
            "① Profit Optimizer → ② strategy_policy.json → ③ TOP10 → ④ 決済/再エントリー → ⑤ 月間損益 → ⑥ 再検証・再学習"
        )
        requests.post(webhook, json={"content": msg[:1950]}, timeout=30).raise_for_status()


if __name__ == "__main__":
    main()
