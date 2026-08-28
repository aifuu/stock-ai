"""
intraday_exit_monitor.py

intraday_auto_entry_monitor.py がその日エントリーを検出した銘柄について、
TP / SL / 15:25強制決済のいずれかに達したかを確認し、
paper_intraday_history.csv の当日行を決済結果で埋める。

実注文は一切行わない。デイトレ実績は paper_intraday_history.csv に保存する。
何度呼び出しても安全な idempotent 設計。
"""

import json
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from common import is_tse_trading_day, send, COMPANY_NAMES

JST = ZoneInfo("Asia/Tokyo")
STATE_FILE = "intraday_auto_entry_state.json"
HISTORY_FILE = "paper_intraday_history.csv"
FORCED_EXIT = dtime(15, 25)
FEE_RATE = float(os.getenv("IT_FEE_RATE", "0.00055"))


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ state読み込み失敗: {e}")
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def download_5m(ticker):
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"{ticker}: 5m取得失敗: {exc}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return None

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(JST).tz_localize(None)
    else:
        idx = idx.tz_localize("UTC").tz_convert(JST).tz_localize(None)
    df.index = idx

    return df.sort_index()


def resolve_exit(df, trade_date, entry_time_str, tp, sl, now):
    """確定済み5分足だけでTP/SL/EODを判定する。"""
    today = df[df.index.date == trade_date]
    if today.empty:
        return None

    entry_dt = pd.Timestamp(f"{trade_date} {entry_time_str}")
    after_entry = today[today.index >= entry_dt]
    after_entry = after_entry[after_entry.index <= now.replace(tzinfo=None)]
    if after_entry.empty:
        return None

    for ts, bar in after_entry.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        # 同一足で両方到達した場合は保守的にSL優先。
        if low <= sl and high >= tp:
            return ts, sl, "SL_BOTH"
        if low <= sl:
            return ts, sl, "SL"
        if high >= tp:
            return ts, tp, "TP"
        if ts.time() >= FORCED_EXIT:
            return ts, float(bar["Close"]), "EOD"

    if now.time() >= FORCED_EXIT:
        last = after_entry.iloc[-1]
        return after_entry.index[-1], float(last["Close"]), "EOD"

    return None


def ensure_history_row(state):
    """入口側が履歴を書けなかった場合でも、stateから安全にOPEN行を作る。"""
    trade_date = state["date"]
    columns = [
        "date", "ticker", "rank", "strategy", "score", "probability",
        "entry_time", "entry_price", "tp", "sl", "gap_pct", "vol_ratio",
        "exit_time", "exit_price", "exit_reason", "return_pct", "status"
    ]

    if os.path.exists(HISTORY_FILE):
        try:
            history = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        except Exception:
            history = pd.DataFrame(columns=columns)
    else:
        history = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in history.columns:
            history[col] = np.nan

    mask = (
        history["date"].astype(str).eq(str(trade_date))
        & history["ticker"].astype(str).eq(str(state["ticker"]))
    )

    row = {
        "date": trade_date,
        "ticker": state["ticker"],
        "rank": state.get("rank", np.nan),
        "strategy": state.get("strategy", ""),
        "score": state.get("score", np.nan),
        "probability": state.get("probability", np.nan),
        "entry_time": state["entry_time"],
        "entry_price": state["entry_price"],
        "tp": state["tp"],
        "sl": state["sl"],
        "gap_pct": state.get("gap_pct", np.nan),
        "vol_ratio": state.get("vol_ratio", np.nan),
        "exit_time": "",
        "exit_price": np.nan,
        "exit_reason": "",
        "return_pct": np.nan,
        "status": "OPEN",
    }

    if mask.any():
        idx = history.index[mask][0]
        for k, v in row.items():
            history.at[idx, k] = v
    else:
        for col in history.columns:
            row.setdefault(col, np.nan)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    history.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    return history


def main():
    now = datetime.now(JST)
    trade_date = now.date()

    if not is_tse_trading_day(trade_date):
        print("東証休場日のため終了")
        return

    if now.time() < dtime(9, 15):
        print("決済監視開始前")
        return

    state = load_state()
    if state.get("date") != str(trade_date) or not state.get("signaled"):
        print("本日はエントリーなし")
        return

    if state.get("closed"):
        print("本日の決済は既に完了しています")
        return

    required_state = ["ticker", "entry_time", "entry_price", "tp", "sl"]
    missing = [k for k in required_state if k not in state]
    if missing:
        print(f"❌ stateに必要項目がありません: {missing}")
        return

    history = ensure_history_row(state)
    mask = (
        history["date"].astype(str).eq(str(trade_date))
        & history["ticker"].astype(str).eq(str(state["ticker"]))
    )

    if not mask.any():
        print("履歴行作成に失敗")
        return

    row = history[mask].iloc[0]
    if pd.notna(row.get("exit_price")) and str(row.get("exit_price")).strip() != "":
        state["closed"] = True
        save_state(state)
        print("既に決済結果が記録済みです")
        return

    ticker = str(state["ticker"])
    try:
        entry_price = float(state["entry_price"])
        tp = float(state["tp"])
        sl = float(state["sl"])
        entry_time_str = str(state["entry_time"])
    except Exception as e:
        print(f"❌ state数値項目不正: {e}")
        return

    df = download_5m(ticker)
    if df is None:
        print(f"{ticker}: 5分足取得失敗、次回再試行")
        return

    exited = resolve_exit(df, trade_date, entry_time_str, tp, sl, now)
    if exited is None:
        print(f"{ticker}: まだ決済条件未達 ({now:%H:%M})")
        return

    exit_time, exit_price, reason = exited
    gross = (exit_price / entry_price - 1.0) * 100.0
    net = gross - (FEE_RATE * 2.0 * 100.0)

    history.loc[mask, "exit_time"] = exit_time.strftime("%H:%M")
    history.loc[mask, "exit_price"] = exit_price
    history.loc[mask, "exit_reason"] = reason
    history.loc[mask, "return_pct"] = round(net, 3)
    history.loc[mask, "status"] = "CLOSED"
    history.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    state["closed"] = True
    state["exit_time"] = exit_time.strftime("%H:%M")
    state["exit_price"] = exit_price
    state["exit_reason"] = reason
    state["return_pct"] = round(net, 3)
    save_state(state)

    name = COMPANY_NAMES.get(ticker, "")
    reason_label = {
        "TP": "🎯 利確",
        "SL": "🛑 損切り",
        "SL_BOTH": "🛑 損切り(同一足でTP/SL両到達)",
        "EOD": "⏰ 15:25強制決済",
    }.get(reason, reason)

    msg = (
        "✅ 自動デイトレ決済結果\n"
        f"日付: {trade_date}\n"
        f"銘柄: {ticker} {name}\n"
        f"戦略: {state.get('strategy', 'N/A')}\n"
        f"買値: {entry_price:.1f} → 売値: {exit_price:.1f}\n"
        f"決済理由: {reason_label}\n"
        f"決済時刻: {exit_time.strftime('%H:%M')}\n"
        f"損益率(手数料込み): {net:+.2f}%\n"
        "※発注ではありません。ペーパートレード記録です。"
    )
    print(msg)
    send(msg)


if __name__ == "__main__":
    main()
