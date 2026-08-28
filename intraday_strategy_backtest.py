"""
intraday_strategy_backtest.py

intraday_auto_entry_monitor.py と同じ STANDARD / RELAXED / LOOSE の3段階を
独立にバックテストし、intraday_strategy_comparison.csv を出力する。

未来情報リークを避けるため、条件成立した5分足の次足始値でエントリーする。
当日はTP / SL / 15:25強制決済で終了する。
"""

import os
import time
from datetime import time as dtime

import numpy as np
import pandas as pd
import yfinance as yf

HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
OUT_FILE = "intraday_strategy_comparison.csv"
START_DATE = pd.Timestamp(
    os.getenv(
        "IS_START_DATE",
        (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
    )
)
END_DATE = pd.Timestamp(
    os.getenv("IS_END_DATE", pd.Timestamp.today().strftime("%Y-%m-%d"))
)

TOP_N = 3
ATR_PERIOD = 14
ENTRY_START = dtime(9, 15)
ENTRY_END = dtime(10, 0)
FORCED_EXIT = dtime(15, 25)
MAX_GAP_PCT = float(os.getenv("AUTO_ENTRY_MAX_GAP", "5.0"))
ATR_TP = float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"))
ATR_SL = float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"))
FEE_RATE = float(os.getenv("IT_FEE_RATE", "0.00055"))
INITIAL_CAPITAL = float(os.getenv("IT_INITIAL_CAPITAL", "1_000_000"))
MIN_TRADES_FOR_RANKING = int(os.getenv("IS_MIN_TRADES", "10"))

STRATEGIES = {
    "STANDARD": {"min_score": 65.0, "min_up_prob": 55.0, "min_vol_ratio": 1.0},
    "RELAXED": {"min_score": 60.0, "min_up_prob": 52.0, "min_vol_ratio": 0.9},
    "LOOSE": {"min_score": 55.0, "min_up_prob": 50.0, "min_vol_ratio": 0.8},
}


def load_top3_history():
    if not os.path.exists(HISTORY_FILE):
        raise ValueError(f"{HISTORY_FILE} がありません")
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{HISTORY_FILE} が空です")
    required = {"date", "ticker", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{HISTORY_FILE} に必要列がありません: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    else:
        df["probability"] = np.nan

    df = df.dropna(subset=["date", "ticker", "score"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    if df.empty:
        return df

    df = (
        df.sort_values(["date", "score"], ascending=[True, False])
        .drop_duplicates(["date", "ticker"], keep="first")
    )
    df["calc_rank"] = df.groupby("date")["score"].rank(method="first", ascending=False)
    return df[df["calc_rank"] <= TOP_N].copy()


def download_intraday(ticker, trade_date, cache):
    key = (ticker, trade_date.date())
    if key in cache:
        return cache[key]

    start = pd.Timestamp(trade_date).tz_localize("Asia/Tokyo")
    end = start + pd.Timedelta(days=1)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"{ticker} {trade_date:%Y-%m-%d}: 5分足取得失敗: {exc}")
        cache[key] = None
        return None

    if df is None or df.empty:
        cache[key] = None
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        cache[key] = None
        return None
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    else:
        idx = idx.tz_localize("UTC").tz_convert("Asia/Tokyo").tz_localize(None)
    df.index = idx
    result = df.sort_index()
    cache[key] = result
    time.sleep(0.15)
    return result


def get_previous_close(ticker, trade_date, cache):
    key = ("prevclose", ticker, trade_date.date())
    if key in cache:
        return cache[key]
    start = trade_date - pd.Timedelta(days=10)
    try:
        d = yf.download(
            ticker,
            start=start,
            end=trade_date + pd.Timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if d is None or d.empty:
            cache[key] = np.nan
            return np.nan
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        idx = pd.to_datetime(d.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
        d.index = idx
        d = d[d.index < trade_date]
        value = float(d["Close"].iloc[-1]) if not d.empty else np.nan
    except Exception:
        value = np.nan
    cache[key] = value
    return value


def indicators(df):
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    cum_vol = x["Volume"].replace(0, np.nan).cumsum()
    x["vwap"] = (typical * x["Volume"]).cumsum() / cum_vol
    x["vol_ma20"] = x["Volume"].rolling(20, min_periods=5).mean()
    x["vol_ratio"] = x["Volume"] / x["vol_ma20"].replace(0, np.nan)
    x["ema5"] = x["Close"].ewm(span=5, adjust=False).mean()
    x["ema20"] = x["Close"].ewm(span=20, adjust=False).mean()

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


def find_entry(intra, prev_close, score, prob, config, trade_date):
    x = indicators(intra)
    today = x[x.index.date == trade_date.date()]
    if today.empty:
        return None

    window = today[
        (today.index.time >= ENTRY_START) &
        (today.index.time <= ENTRY_END)
    ]
    if len(window) < 2:
        return None

    if not np.isfinite(score) or score < config["min_score"]:
        return None
    if not np.isfinite(prob) or prob < config["min_up_prob"]:
        return None

    first_open = float(today["Open"].iloc[0])
    gap = np.nan
    if np.isfinite(prev_close) and prev_close > 0:
        gap = (first_open / prev_close - 1.0) * 100.0
        if gap > MAX_GAP_PCT:
            return None

    for i in range(len(window) - 1):
        bar = window.iloc[i]
        nxt = window.iloc[i + 1]
        if float(bar["Close"]) <= float(bar["vwap"]):
            continue
        if pd.isna(bar["vol_ratio"]) or float(bar["vol_ratio"]) < config["min_vol_ratio"]:
            continue

        atr = float(bar["atr"]) if pd.notna(bar["atr"]) and float(bar["atr"]) > 0 else float(bar["Close"]) * 0.005
        return {
            "entry_time": nxt.name,
            "entry_price": float(nxt["Open"]),
            "atr": atr,
            "gap_pct": gap,
            "vol_ratio": float(bar["vol_ratio"]),
            "ema_bull": bool(float(bar["ema5"]) > float(bar["ema20"])),
        }
    return None


def exit_trade(intra, entry):
    x = indicators(intra)
    future = x[x.index >= entry["entry_time"]]
    if future.empty:
        return None

    tp = entry["entry_price"] + ATR_TP * entry["atr"]
    sl = entry["entry_price"] - ATR_SL * entry["atr"]

    for ts, bar in future.iterrows():
        high, low = float(bar["High"]), float(bar["Low"])
        if low <= sl and high >= tp:
            return ts, sl, "SL_BOTH"
        if low <= sl:
            return ts, sl, "SL"
        if high >= tp:
            return ts, tp, "TP"
        if ts.time() >= FORCED_EXIT:
            return ts, float(bar["Close"]), "EOD"

    last = future.iloc[-1]
    return future.index[-1], float(last["Close"]), "EOD"


def backtest_strategy(strategy_name, config, top3, cache):
    rows = []
    for trade_date, day in top3.groupby("date"):
        day = day.sort_values("calc_rank").head(TOP_N)
        for _, row in day.iterrows():
            ticker = str(row["ticker"])
            intra = download_intraday(ticker, trade_date, cache)
            if intra is None or intra.empty:
                continue
            prev_close = get_previous_close(ticker, trade_date, cache)
            score = float(row["score"])
            prob = float(row["probability"]) if pd.notna(row["probability"]) else np.nan
            entry = find_entry(intra, prev_close, score, prob, config, trade_date)
            if entry is None:
                continue
            exited = exit_trade(intra, entry)
            if exited is None:
                continue

            exit_time, exit_price, reason = exited
            gross = (exit_price / entry["entry_price"] - 1.0) * 100.0
            net = gross - (FEE_RATE * 2.0 * 100.0)
            rows.append(
                {
                    "date": trade_date,
                    "rank": int(row["calc_rank"]),
                    "ticker": ticker,
                    "entry_time": entry["entry_time"],
                    "entry_price": entry["entry_price"],
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "reason": reason,
                    "return_pct": net,
                }
            )
            break

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {
            "strategy": strategy_name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "pf": 0.0,
            "total_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "final_capital": INITIAL_CAPITAL,
        }, trades

    returns = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
    win_rate = float((returns > 0).mean() * 100.0)
    avg_return = float(returns.mean())
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0.0
    for ret in trades.sort_values("date")["return_pct"]:
        capital *= 1.0 + ret / 100.0
        peak = max(peak, capital)
        dd = (capital / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)

    summary = {
        "strategy": strategy_name,
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "pf": pf,
        "total_return_pct": (capital / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_dd_pct": max_dd,
        "final_capital": capital,
    }
    return summary, trades


def main():
    top3 = load_top3_history()
    if top3.empty:
        raise ValueError("対象期間にTOP3履歴がありません")

    cache = {}
    summaries = []
    detail_frames = {}
    for name, config in STRATEGIES.items():
        summary, trades = backtest_strategy(name, config, top3, cache)
        summaries.append(summary)
        detail_frames[name] = trades

    out = pd.DataFrame(summaries)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    for name, trades in detail_frames.items():
        if not trades.empty:
            trades.to_csv(f"intraday_trades_{name}.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("📊 デイトレ3段階比較バックテスト")
    print("=" * 60)
    print(f"期間: {START_DATE.date()} ～ {END_DATE.date()}")
    print(out.to_string(index=False))
    print(f"\n結果CSV: {OUT_FILE}")

    qualified = out[out["trades"] >= MIN_TRADES_FOR_RANKING]
    if qualified.empty:
        print(f"\n⚠ 取引数{MIN_TRADES_FOR_RANKING}件以上の段階がないため、順位付けは参考値です")
    else:
        best = qualified.sort_values(
            ["pf", "avg_return_pct", "final_capital"],
            ascending=[False, False, False],
        ).iloc[0]
        print(
            f"\n✅ 現時点の最良候補: {best['strategy']} "
            f"(PF={best['pf']:.3f}, 件数={int(best['trades'])}, "
            f"勝率={best['win_rate']:.1f}%, 最終資産={best['final_capital']:,.0f}円)"
        )


if __name__ == "__main__":
    main()
