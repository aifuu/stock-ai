"""
intraday_auto_entry_monitor.py

日足AIの当日TOP3候補を使い、9:15〜10:00の5分足で
寄り後チャートを確認し、最初に条件成立した1銘柄だけを
ペーパートレード登録する。実注文は一切行わない。

STANDARD / RELAXED / LOOSE は intraday_strategy_backtest.py で比較し、
build_intraday_policy.py が最良段階を strategy_policy.json の
intraday_* キーへ保存する。本番監視では採用済み1段階だけを使用する。
"""

import json
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from common import COMPANY_NAMES, is_tse_trading_day, send

JST = ZoneInfo("Asia/Tokyo")
HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
INTRADAY_HISTORY_FILE = "paper_intraday_history.csv"
POLICY_FILE = "strategy_policy.json"
STATE_FILE = "intraday_auto_entry_state.json"
ENTRY_START = dtime(9, 15)
ENTRY_END = dtime(10, 0)
FORCED_EXIT = dtime(15, 25)
TOP_N = 3
ATR_PERIOD = 14
MAX_GAP_PCT = float(os.getenv("AUTO_ENTRY_MAX_GAP", "5.0"))
DEFAULT_ATR_TP = float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"))
DEFAULT_ATR_SL = float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"))

STRATEGIES = {
    "STANDARD": {"min_score": 65.0, "min_up_prob": 55.0, "min_vol_ratio": 1.0},
    "RELAXED": {"min_score": 60.0, "min_up_prob": 52.0, "min_vol_ratio": 0.9},
    "LOOSE": {"min_score": 55.0, "min_up_prob": 50.0, "min_vol_ratio": 0.8},
}


def load_policy():
    if not os.path.exists(POLICY_FILE):
        return {}
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
        return p if isinstance(p, dict) else {}
    except Exception as exc:
        print(f"⚠ {POLICY_FILE} 読み込み失敗: {exc}")
        return {}


def load_active_strategy():
    policy = load_policy()
    name = str(policy.get("intraday_strategy", "")).strip().upper()
    if name in STRATEGIES:
        base = STRATEGIES[name]
        cfg = {
            "min_score": float(policy.get("intraday_min_score", base["min_score"])),
            "min_up_prob": float(policy.get("intraday_min_up_prob", base["min_up_prob"])),
            "min_vol_ratio": float(policy.get("intraday_min_vol_ratio", base["min_vol_ratio"])),
        }
        atr_tp = float(policy.get("intraday_atr_tp", DEFAULT_ATR_TP))
        atr_sl = float(policy.get("intraday_atr_sl", DEFAULT_ATR_SL))
        return name, cfg, atr_tp, atr_sl
    return None, None, DEFAULT_ATR_TP, DEFAULT_ATR_SL


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
    try:
        df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    except Exception as exc:
        print(f"⚠ {HISTORY_FILE} 読み込み失敗: {exc}")
        return pd.DataFrame()
    required = {"date", "ticker", "score"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["probability"] = pd.to_numeric(df.get("probability", np.nan), errors="coerce")
    day = df[df["date"].dt.date == trade_date].dropna(subset=["ticker", "score"]).copy()
    if day.empty:
        return day
    day = day.sort_values("score", ascending=False).drop_duplicates("ticker")
    day["calc_rank"] = day["score"].rank(method="first", ascending=False)
    return day[day["calc_rank"] <= TOP_N].copy()


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
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev).abs(),
            (x["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=5).mean()
    return x


def previous_close(df, trade_date):
    hist = df[df.index.date < trade_date]
    if hist.empty:
        return np.nan
    return float(hist["Close"].iloc[-1])


def find_entry(df, row, trade_date, now, config, atr_tp, atr_sl):
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
    if np.isfinite(prev_close) and prev_close > 0:
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
        return {
            "entry_time": nxt.name.strftime("%H:%M"),
            "entry_price": entry_price,
            "tp": entry_price + atr_tp * atr,
            "sl": entry_price - atr_sl * atr,
            "score": score,
            "probability": prob,
            "gap_pct": gap,
            "vol_ratio": float(bar["vol_ratio"]),
            "ema_bull": bool(float(bar["ema5"]) > float(bar["ema20"])),
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
        "ema_bull": entry["ema_bull"],
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
    if "date" in history.columns:
        history = history[history["date"].astype(str) != str(trade_date)].copy()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(INTRADAY_HISTORY_FILE, index=False, encoding="utf-8-sig")


def main():
    now = datetime.now(JST)
    trade_date = now.date()
    if not is_tse_trading_day(trade_date):
        print("東証休場日のため終了")
        return
    if now.time() < ENTRY_START or now.time() > ENTRY_END:
        print(f"現在時刻 {now:%H:%M} はエントリー監視時間外")
        return

    top3 = load_top3(trade_date)
    if top3.empty:
        print("本日のTOP3履歴なし")
        return

    state = load_state()
    if state.get("date") == str(trade_date) and state.get("signaled"):
        print("本日は既にシグナル通知済み")
        return

    strategy_name, config, atr_tp, atr_sl = load_active_strategy()
    if strategy_name is None:
        print("⚠ デイトレ採用policyがありません。安全のため本日はエントリーしません")
        send("⚠️ デイトレ採用policyなし\nstrategy_policy.json の intraday_strategy が未設定のため、本日はペーパートレードを見送ります。")
        return

    print(f"採用デイトレ戦略: {strategy_name}")
    print(f"条件: score>={config['min_score']} / prob>={config['min_up_prob']} / vol>={config['min_vol_ratio']}")
    print(f"ATR: TP={atr_tp}x / SL={atr_sl}x")

    for _, row in top3.sort_values("calc_rank").iterrows():
        ticker = str(row["ticker"])
        df = download_5m(ticker)
        if df is None:
            continue
        entry = find_entry(df, row, trade_date, now, config, atr_tp, atr_sl)
        if entry is None:
            continue

        rank = int(row["calc_rank"])
        state = {
            "date": str(trade_date),
            "signaled": True,
            "closed": False,
            "strategy": strategy_name,
            "ticker": ticker,
            "rank": rank,
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
        save_paper_entry(trade_date, ticker, rank, strategy_name, entry)

        name = COMPANY_NAMES.get(ticker, "")
        prob_text = f"{entry['probability']:.1f}%" if np.isfinite(entry["probability"]) else "N/A"
        ema_text = "EMA5>EMA20" if entry["ema_bull"] else "EMA5<=EMA20"
        msg = (
            "🚨 自動デイトレ買いシグナル\n"
            f"日付: {trade_date}\n"
            f"採用戦略: {strategy_name}\n"
            f"TOP{rank}: {ticker} {name}\n"
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

    print(f"{strategy_name}: 現時点でTOP3のエントリー条件は未成立")


if __name__ == "__main__":
    main()
