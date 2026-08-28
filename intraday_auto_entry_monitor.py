"""
intraday_auto_entry_monitor.py

毎営業日、日足AIのTOP3候補を使い、9:15〜10:00の5分足で
「寄り後チャート確認→その日最初に条件成立した1銘柄」の買いシグナルを自動判定する。

このスクリプトは証券会社へ発注せず、Discordへシグナルを通知するだけ。

3段階比較:
1. STANDARD: AI65 / 確率55 / 出来高1.0
2. RELAXED:  AI60 / 確率52 / 出来高0.9
3. LOOSE:    AI55 / 確率50 / 出来高0.8

共通条件: VWAP上、寄りGU上限5%、ATR TP/SL。
EMA5>EMA20は必須ではなく、参考情報として記録する。
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
HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
INTRADAY_HISTORY_FILE = "paper_intraday_history.csv"
ENTRY_START = dtime(9, 15)
ENTRY_END = dtime(10, 0)
FORCED_EXIT = dtime(15, 25)
TOP_N = 3
ATR_PERIOD = 14
ATR_TP = float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"))
ATR_SL = float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"))
MAX_GAP_PCT = float(os.getenv("AUTO_ENTRY_MAX_GAP", "5.0"))
STATE_FILE = "intraday_auto_entry_state.json"

STRATEGIES = {
    "STANDARD": {"min_score": 65.0, "min_up_prob": 55.0, "min_vol_ratio": 1.0},
    "RELAXED": {"min_score": 60.0, "min_up_prob": 52.0, "min_vol_ratio": 0.9},
    "LOOSE": {"min_score": 55.0, "min_up_prob": 50.0, "min_vol_ratio": 0.8},
}


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_top3(trade_date):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    if df.empty or "date" not in df.columns or "ticker" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    else:
        df["probability"] = np.nan
    day = df[df["date"].dt.date == trade_date].copy()
    day = day.dropna(subset=["ticker", "score"])
    if day.empty:
        return day
    day = day.sort_values("score", ascending=False).drop_duplicates("ticker")
    day["calc_rank"] = day["score"].rank(method="first", ascending=False)
    return day[day["calc_rank"] <= TOP_N].copy()


def download_5m(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False, threads=False)
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


def indicators(df):
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    cum_vol = x["Volume"].replace(0, np.nan).cumsum()
    x["vwap"] = (typical * x["Volume"]).cumsum() / cum_vol
    x["ema5"] = x["Close"].ewm(span=5, adjust=False).mean()
    x["ema20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["vol_ma20"] = x["Volume"].rolling(20, min_periods=5).mean()
    x["vol_ratio"] = x["Volume"] / x["vol_ma20"].replace(0, np.nan)
    prev = x["Close"].shift(1)
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev).abs(),
        (x["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=5).mean()
    return x


def previous_close(df, trade_date):
    hist = df[df.index.date < trade_date]
    if hist.empty:
        return np.nan
    return float(hist["Close"].iloc[-1])


def find_entry(df, row, trade_date, now, config):
    x = indicators(df)
    today = x[x.index.date == trade_date]
    if today.empty or now.time() < ENTRY_START:
        return None
    window = today[(today.index.time >= ENTRY_START) & (today.index.time <= ENTRY_END)]
    window = window[window.index <= now.replace(tzinfo=None)]
    if len(window) < 2:
        return None
    score = float(row.get("score", np.nan))
    prob = float(row.get("probability", np.nan)) if pd.notna(row.get("probability", np.nan)) else np.nan
    if not np.isfinite(score) or score < config["min_score"]:
        return None
    if np.isfinite(prob) and prob < config["min_up_prob"]:
        return None
    prev_close = previous_close(x, trade_date)
    first_open = float(today["Open"].iloc[0])
    gap = np.nan
    if prev_close > 0:
        gap = (first_open / prev_close - 1.0) * 100.0
        if gap > MAX_GAP_PCT:
            return None
    for i in range(len(window) - 1):
        bar = window.iloc[i]
        nxt = window.iloc[i + 1]
        cond_vwap = float(bar["Close"]) > float(bar["vwap"])
        cond_volume = pd.notna(bar["vol_ratio"]) and float(bar["vol_ratio"]) >= config["min_vol_ratio"]
        if not (cond_vwap and cond_volume):
            continue
        atr = float(bar["atr"]) if pd.notna(bar["atr"]) and float(bar["atr"]) > 0 else float(bar["Close"]) * 0.005
        entry_price = float(nxt["Open"])
        tp = entry_price + ATR_TP * atr
        sl = entry_price - ATR_SL * atr
        ema_bull = float(bar["ema5"]) > float(bar["ema20"])
        return {
            "entry_time": nxt.name.strftime("%H:%M"),
            "entry_price": entry_price,
            "tp": tp,
            "sl": sl,
            "score": score,
            "probability": prob,
            "gap_pct": gap,
            "vol_ratio": float(bar["vol_ratio"]),
            "ema_bull": ema_bull,
        }
    return None


def save_paper_entry(trade_date, ticker, rank, strategy_name, entry):
    row = {
        "date": str(trade_date),
        "ticker": ticker,
        "rank": int(rank),
        "strategy": strategy_name,
        "score": entry["score"],
        "probability": entry["probability"],
        "entry_time": entry["entry_time"],
        "entry_price": entry["entry_price"],
        "tp": entry["tp"],
        "sl": entry["sl"],
        "gap_pct": entry["gap_pct"],
        "vol_ratio": entry["vol_ratio"],
        "ema_bull": bool(entry["ema_bull"]),
        "exit_time": "",
        "exit_price": np.nan,
        "exit_reason": "",
        "return_pct": np.nan,
        "status": "OPEN",
    }
    columns = list(row.keys())
    if os.path.exists(INTRADAY_HISTORY_FILE):
        try:
            history = pd.read_csv(INTRADAY_HISTORY_FILE, encoding="utf-8-sig")
        except Exception:
            history = pd.DataFrame(columns=columns)
    else:
        history = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in history.columns:
            history[col] = np.nan
    for col in history.columns:
        if col not in row:
            row[col] = np.nan
    if not history.empty:
        same_day = (history["date"].astype(str) == str(trade_date)) if "date" in history.columns else pd.Series(False, index=history.index)
        if same_day.any():
            history = history[~same_day].copy()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(INTRADAY_HISTORY_FILE, index=False, encoding="utf-8-sig")


def main():
    now = datetime.now(JST)
    trade_date = now.date()
    if not is_tse_trading_day(trade_date):
        print("東証休場日のため終了")
        return
    if now.time() < ENTRY_START or now.time() > FORCED_EXIT:
        print(f"現在時刻 {now:%H:%M} は自動監視時間外")
        return
    top3 = load_top3(trade_date)
    if top3.empty:
        print("本日のTOP3履歴なし")
        return
    state = load_state()
    if state.get("date") == str(trade_date) and state.get("signaled"):
        print("本日は既にシグナル通知済み")
        return
    for strategy_name, config in STRATEGIES.items():
        for _, row in top3.sort_values("calc_rank").iterrows():
            ticker = str(row["ticker"])
            df = download_5m(ticker)
            if df is None:
                continue
            entry = find_entry(df, row, trade_date, now, config)
            if entry is None:
                continue
            state = {
                "date": str(trade_date),
                "signaled": True,
                "closed": False,
                "strategy": strategy_name,
                "ticker": ticker,
                "rank": int(row["calc_rank"]),
                "entry_time": entry["entry_time"],
                "entry_price": entry["entry_price"],
                "tp": entry["tp"],
                "sl": entry["sl"],
                "score": entry["score"],
                "probability": entry["probability"],
                "gap_pct": entry["gap_pct"],
                "vol_ratio": entry["vol_ratio"],
            }
            save_state(state)
            save_paper_entry(trade_date, ticker, row["calc_rank"], strategy_name, entry)
            name = COMPANY_NAMES.get(ticker, "")
            prob_text = f"{entry['probability']:.1f}%" if np.isfinite(entry["probability"]) else "N/A"
            ema_text = "EMA5>EMA20" if entry["ema_bull"] else "EMA5<=EMA20"
            msg = (
                "🚨 自動デイトレ買いシグナル\n"
                f"日付: {trade_date}\n"
                f"検証段階: {strategy_name}\n"
                f"TOP{int(row['calc_rank'])}: {ticker} {name}\n"
                f"AIスコア: {entry['score']:.1f}\n"
                f"上昇確率: {prob_text}\n"
                f"エントリー: {entry['entry_time']} 次の5分足始値\n"
                f"想定買値: {entry['entry_price']:.1f}\n"
                f"TP: {entry['tp']:.1f} / SL: {entry['sl']:.1f}\n"
                f"寄りギャップ: {entry['gap_pct']:+.2f}%\n"
                f"出来高倍率: {entry['vol_ratio']:.2f} / {ema_text}\n"
                "共通条件: VWAP上 / 寄りGU上限5%\n"
                "※この通知は発注ではありません。"
            )
            print(msg)
            send(msg)
            return
    print("3段階すべてで現時点のエントリー条件は未成立")


if __name__ == "__main__":
    main()
