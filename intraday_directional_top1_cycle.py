"""Unified intraday directional TOP1 paper trader.

Every cycle:
1) monitor the existing directional TOP1 position with 5-minute bars;
2) when flat, run the 100-stock directional AI selector;
3) persist a valid paper state even when no trade can be selected;
4) monitor TP/SL/EOD and persist state on every cycle.

No real orders are sent.
"""
import json
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import daily_directional_top1 as trader

JST = ZoneInfo("Asia/Tokyo")
STATE_FILE = trader.STATE_FILE
HISTORY_FILE = trader.HISTORY_FILE
MAX_TRADES_PER_DAY = 10
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
FORCED_EXIT = dtime(15, 25)


def default_state():
    capital = float(trader.INITIAL_CAPITAL)
    return {
        "capital": capital,
        "position": None,
        "peak": capital,
        "max_dd": 0.0,
        "trades_today": 0,
        "trade_count_date": None,
        "daily_start_capital": capital,
    }


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state is not an object")
        base = default_state()
        base.update(state)
        return base
    except Exception:
        return default_state()


def save_state(state):
    # Atomic write: never leave a zero-byte state file after an interrupted run.
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def normalize_5m(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(JST).tz_localize(None)
    else:
        idx = idx.tz_localize("UTC").tz_convert(JST).tz_localize(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def download_5m(ticker):
    try:
        return normalize_5m(yf.download(
            ticker, period="5d", interval="5m", auto_adjust=False,
            progress=False, threads=False,
        ))
    except Exception as exc:
        print(f"5m取得失敗 {ticker}: {exc}")
        return None


def close_position_if_needed(state, now):
    pos = state.get("position")
    if not pos:
        return False

    ticker = str(pos["ticker"])
    df = download_5m(ticker)
    if df is None or df.empty:
        return False

    entry_date = str(pos.get("entry_date", now.strftime("%Y-%m-%d")))
    entry_time = str(pos.get("entry_time", "09:00"))
    direction = str(pos["direction"]).upper()
    entry_price = float(pos["entry_price"])
    tp = float(pos["tp"])
    sl = float(pos["sl"])

    today = df[df.index.date == now.date()]
    if today.empty:
        return False
    now_naive = now.replace(tzinfo=None)
    if entry_date == now.strftime("%Y-%m-%d"):
        start = pd.Timestamp(f"{entry_date} {entry_time}")
        bars = today[(today.index >= start) & (today.index <= now_naive)]
    else:
        bars = today[today.index <= now_naive]
    if bars.empty:
        return False

    exit_price = None
    reason = None
    exit_time = None
    for ts, bar in bars.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])
        if direction == "BUY":
            if low <= sl and high >= tp:
                exit_price, reason = sl, "SL_BOTH"
            elif high >= tp:
                exit_price, reason = tp, "TP"
            elif low <= sl:
                exit_price, reason = sl, "SL"
        else:
            if low <= tp and high >= sl:
                exit_price, reason = sl, "SL_BOTH"
            elif low <= tp:
                exit_price, reason = tp, "TP"
            elif high >= sl:
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
        return False

    gross = ((exit_price / entry_price) - 1.0) * 100.0 if direction == "BUY" else ((entry_price / exit_price) - 1.0) * 100.0
    net_return = gross - (FEE_RATE * 2.0 * 100.0)
    capital_before = float(state["capital"])
    pnl = capital_before * net_return / 100.0
    state["capital"] = capital_before + pnl
    state["position"] = None
    state["peak"] = max(float(state.get("peak", state["capital"])), state["capital"])
    dd = ((state["peak"] - state["capital"]) / state["peak"] * 100.0) if state["peak"] else 0.0
    state["max_dd"] = max(float(state.get("max_dd", 0.0)), dd)

    trader.append_history({
        "entry_date": pos.get("entry_date", entry_date),
        "entry_time": pos.get("entry_time", entry_time),
        "exit_date": str(exit_time.date()),
        "exit_time": exit_time.strftime("%H:%M"),
        "ticker": ticker,
        "company": pos.get("company", ticker),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp": tp,
        "sl": sl,
        "score": pos.get("score", np.nan),
        "up_probability": pos.get("up_probability", np.nan),
        "down_probability": pos.get("down_probability", np.nan),
        "return_pct": round(net_return, 3),
        "pnl": round(pnl, 2),
        "result": reason,
        "hold_days": 1,
        "capital_after": round(state["capital"], 2),
    })
    label = "買い" if direction == "BUY" else "空売り"
    why = {"TP": "🎯 利確", "SL": "🛑 損切り", "SL_BOTH": "🛑 損切り(同一5分足両到達)", "EOD": "⏰ 15:25強制決済"}.get(reason, reason)
    trader.send(
        f"✅ 5分足 自動決済\n銘柄: {ticker} {pos.get('company', '')}\n"
        f"方向: {label}\nエントリー: {entry_price:,.1f} → 決済: {exit_price:,.1f}\n"
        f"理由: {why}｜時刻: {exit_time:%H:%M}\n"
        f"損益率(手数料込み): {net_return:+.2f}%\n"
        f"損益: {pnl:+,.0f}円｜仮想資産: {state['capital']:,.0f}円\n"
        f"📌 次のサイクルで100銘柄を再スキャンします\n※ペーパートレード"
    )
    return True


def reset_daily_counter(state, today):
    if state.get("trade_count_date") != today:
        state["trade_count_date"] = today
        state["trades_today"] = 0
        state["daily_start_capital"] = float(state.get("capital", trader.INITIAL_CAPITAL))


def open_next_top1(state, today):
    if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
        trader.send(f"🛑 TOP1｜本日の新規売買上限 {MAX_TRADES_PER_DAY}回に到達。新規エントリー停止")
        save_state(state)
        return False

    # Existing directional engine performs the 100-stock AI selection.
    trader.main()
    selected = load_state()

    # Critical integration fix: a no-candidate/model/data cycle still creates a
    # valid state file, so the workflow can continue on the next 5-minute cycle.
    if not selected.get("position"):
        selected["trade_count_date"] = today
        selected["trades_today"] = int(state.get("trades_today", 0))
        selected["daily_start_capital"] = float(state.get("daily_start_capital", selected.get("capital", trader.INITIAL_CAPITAL)))
        save_state(selected)
        return False

    selected["trade_count_date"] = today
    selected["trades_today"] = int(state.get("trades_today", 0)) + 1
    selected["position"]["entry_time"] = datetime.now(JST).strftime("%H:%M")
    selected["position"]["execution_mode"] = "5M_INTRADAY"
    save_state(selected)
    trader.send(f"📈 本日新規売買: {selected['trades_today']}/{MAX_TRADES_PER_DAY}回｜5分足監視開始")
    return True


def main():
    now = datetime.now(JST)
    state = load_state()
    today = now.strftime("%Y-%m-%d")
    reset_daily_counter(state, today)

    # Always create/update a valid non-empty state before any market-hours guard.
    save_state(state)

    if now.weekday() >= 5 or now.time() < dtime(9, 0) or now.time() > dtime(15, 30):
        return

    start_cap = float(state.get("daily_start_capital", state.get("capital", trader.INITIAL_CAPITAL)))
    daily_return = ((float(state.get("capital", 0.0)) / start_cap) - 1.0) * 100.0 if start_cap else 0.0
    if daily_return <= -1.5:
        if state.get("position"):
            close_position_if_needed(state, now)
        save_state(state)
        trader.send(f"🛑 TOP1｜日次損失上限 -1.5%到達｜本日は新規停止\n日次損益: {daily_return:+.2f}%")
        return

    close_position_if_needed(state, now)
    save_state(state)

    if state.get("position"):
        return

    if int(state.get("trades_today", 0)) < MAX_TRADES_PER_DAY:
        open_next_top1(state, today)
    else:
        trader.send(f"🛑 TOP1｜本日の上限 {MAX_TRADES_PER_DAY}回。監視終了")
        save_state(state)


if __name__ == "__main__":
    main()
