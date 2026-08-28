"""
intraday_auto_entry_monitor.py

毎営業日、日足AIのTOP3候補を使い、9:15〜10:00の5分足で
「寄り後チャート確認→最初に条件成立した1銘柄」の買いシグナルを自動判定する。

注意:
- このスクリプトは証券会社へ注文を発注しない。
- Discordへ「買いシグナル / TP / SL」を通知するだけ。
- 実売買を行う場合は別途、証券会社APIとの接続と注文管理が必要。
"""

import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from common import is_tse_trading_day, send, TICKERS, COMPANY_NAMES

JST = ZoneInfo("Asia/Tokyo")
HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
ENTRY_START = dtime(9, 15)
ENTRY_END = dtime(10, 0)
FORCED_EXIT = dtime(15, 25)
TOP_N = 3
MIN_SCORE = float(os.getenv("AUTO_ENTRY_MIN_SCORE", "70"))
MIN_UP_PROB = float(os.getenv("AUTO_ENTRY_MIN_UP_PROB", "60"))
MIN_VOL_RATIO = float(os.getenv("AUTO_ENTRY_MIN_VOL_RATIO", "1.2"))
MAX_GAP_PCT = float(os.getenv("AUTO_ENTRY_MAX_GAP", "3.0"))
ATR_PERIOD = 14
ATR_TP = float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"))
ATR_SL = float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"))
STATE_FILE = "intraday_auto_entry_state.json"


def load_top3(trade_date):
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame()
    for c in ["score", "probability"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    day = df[df["date"].dt.date == trade_date].copy()
    if day.empty:
        return pd.DataFrame()
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
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(JST).tz_localize(None)
    else:
        idx = idx.tz_localize(JST).tz_localize(None)
    df.index = idx
    return df.sort_index()


def indicators(df):
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    x["vwap"] = (typical * x["Volume"]).cumsum() / x["Volume"].replace(0, np.nan).cumsum()
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


def find_entry(df, row, trade_date, now):
    x = indicators(df)
    today = x[x.index.date == trade_date]
    if today.empty:
        return None
    window = today[(today.index.time >= ENTRY_START) & (today.index.time <= ENTRY_END)]
    if now.time() < ENTRY_START:
        return None
    window = window[window.index <= now.replace(tzinfo=None)]
    if len(window) < 2:
        return None

    score = float(row.get("score", np.nan))
    prob = float(row.get("probability", np.nan)) if "probability" in row else np.nan
    if not np.isfinite(score) or score < MIN_SCORE:
        return None
    if np.isfinite(prob) and prob < MIN_UP_PROB:
        return None

    prev_close = previous_close(x, trade_date)
    first_open = float(today["Open"].iloc[0])
    if prev_close > 0:
        gap = (first_open / prev_close - 1) * 100
        if gap > MAX_GAP_PCT:
            return None
    else:
        gap = np.nan

    # 最新の確定足までを使い、次の5分足始値で入る。
    for i in range(len(window) - 1):
        bar = window.iloc[i]
        nxt = window.iloc[i + 1]
        ok = (
            float(bar["Close"]) > float(bar["vwap"])
            and float(bar["ema5"]) > float(bar["ema20"])
            and pd.notna(bar["vol_ratio"])
            and float(bar["vol_ratio"]) >= MIN_VOL_RATIO
        )
        if not ok:
            continue
        atr = float(bar["atr"]) if pd.notna(bar["atr"]) and float(bar["atr"]) > 0 else float(bar["Close"]) * 0.005
        entry_price = float(nxt["Open"])
        tp = entry_price + ATR_TP * atr
        sl = entry_price - ATR_SL * atr
        return {
            "entry_time": nxt.name.strftime("%H:%M"),
            "entry_price": entry_price,
            "tp": tp,
            "sl": sl,
            "score": score,
            "probability": prob,
            "gap_pct": gap,
            "vol_ratio": float(bar["vol_ratio"]),
        }
    return None


def main():
    now = datetime.now(JST)
    trade_date = now.date()
    if not is_tse_trading_day(trade_date):
        print("東証休場日のため終了")
        return
    if now.time() < ENTRY_START or now.time() > FORCED_EXIT:
        print(f"現在時刻 {now:%H:%M} は自動エントリー監視時間外")
        return

    top3 = load_top3(trade_date)
    if top3.empty:
        msg = f"⚠ TOP3履歴なし {now:%Y-%m-%d}"
        print(msg)
        send(msg)
        return

    for _, row in top3.sort_values("calc_rank").iterrows():
        ticker = str(row["ticker"])
        df = download_5m(ticker)
        if df is None:
            continue
        entry = find_entry(df, row, trade_date, now)
        if entry is None:
            continue

        state = {
            "date": str(trade_date),
            "ticker": ticker,
            "entry_time": entry["entry_time"],
        }
        if os.path.exists(STATE_FILE):
            try:
                old = pd.read_json(STATE_FILE, typ="series").to_dict()
                if old.get("date") == state["date"]:
                    print("本日は既にシグナル通知済み")
                    return
            except Exception:
                pass

        pd.Series(state).to_json(STATE_FILE, force_ascii=False)
        name = COMPANY_NAMES.get(ticker, "")
        prob_text = f"{entry['probability']:.1f}%" if np.isfinite(entry["probability"]) else "N/A"
        msg = (
            "🚨 自動デイトレ買いシグナル\n"
            f"日付: {trade_date}\n"
            f"TOP{int(row['calc_rank'])}: {ticker} {name}\n"
            f"AIスコア: {entry['score']:.1f}\n"
            f"上昇確率: {prob_text}\n"
            f"エントリー予定: {entry['entry_time']} 次の5分足始値\n"
            f"想定買値: {entry['entry_price']:.1f}\n"
            f"TP: {entry['tp']:.1f} / SL: {entry['sl']:.1f}\n"
            f"寄りギャップ: {entry['gap_pct']:+.2f}%\n"
            "条件: VWAP上 / EMA5>EMA20 / 出来高倍率>=1.2\n"
            "※この通知は発注ではありません。"
        )
        print(msg)
        send(msg)
        return

    print("現時点ではTOP3にエントリー条件成立なし")


if __name__ == "__main__":
    main()
