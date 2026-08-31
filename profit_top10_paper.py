import hashlib
import hmac
import json
import os
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
MAX_TRADES_PER_TICKER_PER_DAY = 10
MAX_TOTAL_TRADES_PER_DAY = 30
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
FORCED_EXIT = dtime(15, 25)


def discord_send(message, required=False):
    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook:
        text = "❌ DISCORD_WEBHOOK がGitHub Actionsに設定されていません。Discord通知を送信できません。"
        print(text)
        if required:
            raise RuntimeError(text)
        return False
    try:
        import requests
        r = requests.post(webhook, json={"content": message[:1950]}, timeout=30)
        r.raise_for_status()
        print("✅ Discord通知送信成功")
        return True
    except Exception as exc:
        print(f"❌ Discord通知送信失敗: {exc}")
        if required:
            raise
        return False


def _canonical_policy_payload(policy):
    covered = {k: policy.get(k) for k in (
        "status", "updated_at", "up_threshold", "min_score_for_buy",
        "nikkei_filter", "atr_tp_multiplier", "atr_sl_multiplier", "hold_days",
        "validation_signals", "validation_win_rate", "validation_avg_return",
        "validation_pf", "validation_dd", "oos_signals", "oos_win_rate",
        "oos_avg_return", "oos_pf", "oos_dd", "oos_validation_pf_ratio",
        "mc_sizing", "mc_10y_probability", "mc_15y_probability",
        "mc_20y_probability", "mc_bankruptcy_probability", "mc_p90_max_dd",
        "strategy_name", "source",
    )}
    return json.dumps(covered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _verify_policy_signature(policy):
    secret = os.getenv("AI_POLICY_SIGNING_SECRET", "").strip()
    if not secret:
        raise RuntimeError("AI_POLICY_SIGNING_SECRET が未設定です。承認済みpolicyを検証できないため新規取引を停止します。")
    if str(policy.get("source", "")) != "adversarial_strategy_validator":
        raise RuntimeError("strategy_policy.json のsourceが検証パイプラインではありません。手動policyは無効です。")
    if int(policy.get("approval_signature_version", 0)) != 1:
        raise RuntimeError("strategy_policy.json の承認署名バージョンが不正です。")
    supplied = str(policy.get("approval_signature", ""))
    if not supplied:
        raise RuntimeError("strategy_policy.json に承認署名がありません。手動承認は無効です。")
    covered = _canonical_policy_payload(policy)
    expected = hmac.new(secret.encode("utf-8"), covered.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise RuntimeError("strategy_policy.json の承認署名が一致しません。手動変更または検証証跡の破損を検知したため新規取引を停止します。")


def load_policy():
    if not os.path.exists(POLICY_FILE):
        raise RuntimeError("strategy_policy.json がありません。先にProfit Optimizerを実行してください。")
    with open(POLICY_FILE, encoding="utf-8") as f:
        p = json.load(f)
    required = ["status", "up_threshold", "min_score_for_buy", "nikkei_filter", "atr_tp_multiplier", "atr_sl_multiplier", "hold_days"]
    missing = [k for k in required if k not in p]
    if missing:
        raise RuntimeError("strategy_policy.json の不足項目: " + ", ".join(missing))
    status = str(p.get("status", "")).upper()
    if status == "APPROVED":
        _verify_policy_signature(p)
    elif status == "PENDING" and os.getenv("PAPER_TRADE_MODE", "1") == "1":
        print("⚠️ policy=PENDING → 実注文ではなくペーパートレード専用モードで継続します")
    else:
        raise RuntimeError(f"strategy_policy.json がAPPROVEDではありません: status={p.get('status')}。安全のため新規取引を停止します。")
    p["up_threshold"] = float(p["up_threshold"])
    p["min_score_for_buy"] = float(p["min_score_for_buy"])
    p["nikkei_filter"] = str(p["nikkei_filter"]).lower() in ("true", "1", "yes", "on")
    p["atr_tp_multiplier"] = float(p["atr_tp_multiplier"])
    p["atr_sl_multiplier"] = float(p["atr_sl_multiplier"])
    p["hold_days"] = int(p["hold_days"])
    return p


def default_state():
    return {"capital": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL, "max_dd": 0.0, "positions": [], "trade_count_date": None, "trades_today": 0, "trades_by_ticker_today": {}, "daily_start_capital": INITIAL_CAPITAL}


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
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False, threads=False)
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
    m = df.assign(month=df["exit_date"].dt.to_period("M")).groupby("month").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), avg_return=("return_pct", "mean"), wins=("pnl", lambda s: int((s > 0).sum()))).reset_index()
    m["win_rate_pct"] = m["wins"] / m["trades"] * 100
    m.to_csv(MONTHLY_FILE, index=False, encoding="utf-8-sig")
    return m.iloc[-1].to_dict()


def close_positions(state, policy, now):
    remaining, messages = [], []
    for p in state["positions"]:
        df = download_5m(p["ticker"])
        if df is None or df.empty:
            remaining.append(p); continue
        entry_date = str(p.get("entry_date", now.strftime("%Y-%m-%d")))
        entry_time = str(p.get("entry_time", "09:00"))
        entry_price, tp, sl = float(p["entry_price"]), float(p["tp"]), float(p["sl"])
        today = df[df.index.date == now.date()]
        if today.empty:
            remaining.append(p); continue
        now_naive = now.replace(tzinfo=None)
        bars = today[(today.index >= pd.Timestamp(f"{entry_date} {entry_time}")) & (today.index <= now_naive)] if entry_date == now.strftime("%Y-%m-%d") else today[today.index <= now_naive]
        if bars.empty:
            remaining.append(p); continue
        exit_price = reason = exit_time = None
        for ts, bar in bars.iterrows():
            high, low = float(bar["High"]), float(bar["Low"])
            if low <= sl and high >= tp:
                exit_price, reason = sl, "SL_BOTH"
            elif high >= tp:
                exit_price, reason = tp, "TP"
            elif low <= sl:
                exit_price, reason = sl, "SL"
            if reason:
                exit_time = ts; break
            if ts.time() >= FORCED_EXIT:
                exit_price, reason, exit_time = float(bar["Close"]), "EOD", ts; break
        if exit_price is None and now.time() >= FORCED_EXIT:
            exit_price, reason, exit_time = float(bars.iloc[-1]["Close"]), "EOD", bars.index[-1]
        if exit_price is None:
            remaining.append(p); continue
        gross_ret = (exit_price / entry_price - 1.0) * 100.0
        net_ret = gross_ret - FEE_RATE * 200.0
        allocation = float(p.get("allocation", 1.0 / TOP_N))
        capital_before = float(state["capital"])
        pnl = capital_before * allocation * net_ret / 100.0
        state["capital"] = capital_before + pnl
        append_history({"entry_date": p.get("entry_date", entry_date), "entry_time": p.get("entry_time", entry_time), "exit_date": str(pd.Timestamp(exit_time).date()), "exit_time": pd.Timestamp(exit_time).strftime("%H:%M"), "ticker": p["ticker"], "company": p["company"], "direction": "BUY", "entry_price": entry_price, "exit_price": float(exit_price), "tp": tp, "sl": sl, "score": p["score"], "up_probability": p["up_probability"], "return_pct": round(net_ret, 3), "pnl": round(pnl, 2), "result": reason, "hold_days": 1, "allocation": allocation, "policy_updated_at": p.get("policy_updated_at")})
        messages.append(f"決済 {p['ticker']} {reason} {net_ret:+.2f}%")
    state["positions"] = remaining
    state["peak"] = max(float(state.get("peak", state["capital"])), float(state["capital"]))
    if state["peak"]:
        state["max_dd"] = max(float(state.get("max_dd", 0.0)), (state["peak"] - state["capital"]) / state["peak"] * 100.0)
    return messages


def scan_candidates(policy):
    nikkei, model = make_nikkei(), load_model()
    if nikkei is None or model is None:
        raise RuntimeError("日経データまたはAIモデルを取得できませんでした")
    candidates, fallback_candidates, scanned = [], [], 0
    for ticker in TICKERS:
        df = download(ticker)
        if df is None or len(df) < 150: continue
        scanned += 1
        feature_cols = list(getattr(model, "feature_names_in_", []))
        x = features(df, nikkei).dropna(subset=feature_cols)
        if x.empty: continue
        try:
            last = x.iloc[-1]
            probs = model.predict_proba(x.iloc[-1:])[0]
            classes = list(model.classes_)
            if not all(c in classes for c in (0, 1, 2)): continue
            down, up, flat = float(probs[classes.index(0)]), float(probs[classes.index(2)]), float(probs[classes.index(1)])
            long_s, _ = directional_score(last, up, down)
            score = float(long_s)
            daily_price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            if not np.isfinite(a) or a <= 0: continue
            if policy["nikkei_filter"]:
                nlast = nikkei.reindex(x.index).ffill().iloc[-1]
                if not (float(nlast["kairi25"]) > 0 and float(nlast["ret5"]) > 0):
                    continue
            intraday = download_5m(ticker)
            price = daily_price
            if intraday is not None and not intraday.empty:
                latest = intraday[intraday.index <= datetime.now(TZ).replace(tzinfo=None)]
                if not latest.empty: price = float(latest["Close"].iloc[-1])
            item = {"ticker": ticker, "company": NAMES.get(ticker, ticker), "score": score, "up_probability": up * 100, "down_probability": down * 100, "flat_probability": flat * 100, "price": price, "tp": price + a * policy["atr_tp_multiplier"], "sl": max(0.01, price - a * policy["atr_sl_multiplier"]), "data_date": str(x.index[-1].date())}
            fallback_candidates.append(item)
            if up * 100 >= policy["up_threshold"] and up > down and flat < 50 and score >= policy["min_score_for_buy"]:
                candidates.append(item)
        except Exception as e:
            print(ticker, "predict", e)
    candidates.sort(key=lambda z: (z["score"], z["up_probability"]), reverse=True)
    fallback_candidates.sort(key=lambda z: (z["score"], z["up_probability"], -z["down_probability"]), reverse=True)
    if not candidates and fallback_candidates:
        candidates = fallback_candidates[:TOP_N]
        print(f"⚠️ 条件成立候補0 → PAPER専用フォールバックでTOP{len(candidates)}を採用")
    return candidates, scanned


def open_positions(state, policy, candidates, today):
    active = {p["ticker"] for p in state["positions"]}
    selected = []
    for c in candidates:
        ticker = c["ticker"]
        if ticker in active: continue
        ticker_count = int(state.get("trades_by_ticker_today", {}).get(ticker, 0))
        if ticker_count >= MAX_TRADES_PER_TICKER_PER_DAY: continue
        if int(state.get("trades_today", 0)) >= MAX_TOTAL_TRADES_PER_DAY: break
        if len(state["positions"]) >= TOP_N: break
        allocation = 1.0 / TOP_N
        state["positions"].append({"entry_date": today, "entry_time": datetime.now(TZ).strftime("%H:%M"), "ticker": ticker, "company": c["company"], "entry_price": c["price"], "tp": c["tp"], "sl": c["sl"], "score": c["score"], "up_probability": c["up_probability"], "down_probability": c["down_probability"], "allocation": allocation, "policy_updated_at": policy.get("updated_at")})
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        counts = state.setdefault("trades_by_ticker_today", {})
        counts[ticker] = ticker_count + 1
        active.add(ticker); selected.append(c)
    return selected


def main():
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")
    policy = load_policy()
    state = load_state()
    reset_daily_counter(state, today)
    save_state(state)

    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook:
        print("⚠️ DISCORD_WEBHOOK 未設定。GitHub Actions Secrets に DISCORD_WEBHOOK を登録してください。")

    market_open = now.weekday() < 5 and dtime(9, 0) <= now.time() <= dtime(15, 30)
    if not market_open:
        msg = f"🤖 PROFIT LOOP｜待機\n📅 {today} {now:%H:%M} JST\n市場時間外のため売買スキャンは実行していません。\n🔗 Policy: {policy['status']}｜更新 {policy.get('updated_at')}\n⚠️ 実注文なし"
        discord_send(msg)
        print(msg)
        return

    start_capital = float(state.get("daily_start_capital", state.get("capital", INITIAL_CAPITAL)))
    closed = close_positions(state, policy, now)
    daily_return = ((float(state.get("capital", 0.0)) / start_capital) - 1.0) * 100.0 if start_capital else 0.0
    if daily_return <= -1.5:
        save_state(state)
        capital_now = float(state["capital"])
        cumulative_return_pct = (capital_now / INITIAL_CAPITAL - 1.0) * 100.0
        gained_yen = capital_now - INITIAL_CAPITAL
        msg = (
            f"🛑 PROFIT LOOP｜日次損失上限\n📅 {today} {now:%H:%M} JST\n"
            f"日次損益 {daily_return:+.2f}%（{capital_now - start_capital:+,.0f}円）｜新規停止\n"
            f"💰 仮想資産 {capital_now:,.0f}円｜累積利益率 {cumulative_return_pct:+.2f}%｜開始100万円から {gained_yen:+,.0f}円\n"
            f"⚠️ 実注文なし"
        )
        discord_send(msg)
        print(msg)
        return

    candidates, scanned = scan_candidates(policy)
    opened = open_positions(state, policy, candidates, today)
    save_state(state)
    monthly = update_monthly()

    capital_now = float(state["capital"])
    daily_pnl_yen = capital_now - start_capital
    cumulative_return_pct = (capital_now / INITIAL_CAPITAL - 1.0) * 100.0
    gained_yen = capital_now - INITIAL_CAPITAL

    top_text = "\n".join(f"{i}. BUY {p['ticker']}｜score {p['score']:.1f}｜UP {p['up_probability']:.1f}%｜entry {p['entry_price']:,.1f}｜TP {p['tp']:,.1f}｜SL {p['sl']:,.1f}" for i, p in enumerate(state["positions"], 1)) or "条件成立銘柄なし"
    counts = ", ".join(f"{k}:{v}回" for k, v in sorted(state.get("trades_by_ticker_today", {}).items())) or "なし"
    if not monthly:
        month_text = "確定取引なし"
    else:
        monthly_return_pct = float(monthly["pnl"]) / INITIAL_CAPITAL * 100.0
        month_text = f"今月利益率 {monthly_return_pct:+.2f}%（損益 {monthly['pnl']:+,.0f}円）｜取引 {int(monthly['trades'])}件｜参考勝率 {monthly['win_rate_pct']:.1f}%"
    profit_text = f"📈 本日損益 {daily_pnl_yen:+,.0f}円（{daily_return:+.2f}%）\n📊 累積利益率 {cumulative_return_pct:+.2f}%｜開始100万円から {gained_yen:+,.0f}円"
    mode_text = "APPROVED" if str(policy.get("status", "")).upper() == "APPROVED" else "PENDING｜PAPER専用"
    msg = ("🤖 PROFIT LOOP｜TOP10 5分足ペーパートレード\n" "━━━━━━━━━━━━━━━━━━\n" f"📅 {today} {now:%H:%M} JST\n⚠️ 実注文なし\n\n" f"🔗 Policy: {mode_text}｜更新 {policy.get('updated_at')}\n" f"条件: UP≥{policy['up_threshold']:.0f}% / SCORE≥{policy['min_score_for_buy']:.0f}% / TP {policy['atr_tp_multiplier']:.2f}ATR / SL {policy['atr_sl_multiplier']:.2f}ATR\n" f"430銘柄対象｜取得成功 {scanned}｜候補 {len(candidates)}｜今回新規 {len(opened)}\n" f"🔁 同一銘柄 最大{MAX_TRADES_PER_TICKER_PER_DAY}回/日｜全体 最大{MAX_TOTAL_TRADES_PER_DAY}回/日\n" f"📊 本日銘柄別取引回数: {counts}\n\n" f"🏆 保有TOP10\n{top_text}\n\n" f"{profit_text}\n" f"💰 仮想資産 {capital_now:,.0f}円｜最大DD {state['max_dd']:.2f}%\n" f"📅 {month_text}\n\n" "① AIスキャン → ② 利益優先ランキング → ③ TOP10ペーパー → ④ 決済/再エントリー → ⑤ 月間損益")
    for m in closed:
        print(m)
    print(msg)
    discord_send(msg, required=True)


if __name__ == "__main__":
    main()
