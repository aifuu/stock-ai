import os
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# TOP3・寄り後チャート確認・当日決済 バックテスト
# ============================================================
#
# 目的
#   prediction_history.csv の各営業日の TOP3 を候補にし、
#   寄り付き直後ではなく 5分足チャートを確認してからエントリー。
#   当日中に TP / SL / 大引け のいずれかで決済する。
#
# 重要
#   Yahoo Finance の intraday データには長期取得制限があるため、
#   このスクリプトは「利用可能な5分足期間」だけを検証する。
#   2021～2026 の長期検証を行う場合は、5分足の過去データCSVを
#   INTRADAY_DIR に置く方式へ切り替えられる。
#
#   prediction_history.csv 自体は現在の日足ベースの AI 推奨履歴。
#   したがって「その日の TOP3」を未来情報なしで選び、
#   その後の intraday データだけでエントリー/決済する構造にする。
# ============================================================

HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
INTRADAY_DIR = os.getenv("INTRADAY_DIR", "intraday_data")

START_DATE = pd.Timestamp(os.getenv("IT_START_DATE", "2026-01-01"))
END_DATE = pd.Timestamp(os.getenv("IT_END_DATE", "2026-08-22"))

TOP_N = 3
INTERVAL = "5m"

# 寄り後にチャート確認してエントリー
ENTRY_START = dtime(9, 15)
ENTRY_END = dtime(10, 0)

# 日本市場の大引け前に強制決済
FORCED_EXIT = dtime(15, 25)

# チャート条件
MIN_VOL_RATIO = float(os.getenv("IT_MIN_VOL_RATIO", "1.2"))
MIN_SCORE = float(os.getenv("IT_MIN_SCORE", "70"))
MIN_UP_PROB = float(os.getenv("IT_MIN_UP_PROB", "60"))
MAX_GAP_PCT = float(os.getenv("IT_MAX_GAP_PCT", "3.0"))

# リスク管理
ATR_PERIOD = 14
ATR_TP = float(os.getenv("IT_ATR_TP", "2.0"))
ATR_SL = float(os.getenv("IT_ATR_SL", "1.0"))

INITIAL_CAPITAL = float(os.getenv("IT_INITIAL_CAPITAL", "1_000_000"))
FEE_RATE = float(os.getenv("IT_FEE_RATE", "0.00055"))


def load_top3_history():
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("prediction_history.csv が空です")

    required = {"date", "ticker", "score", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction_history.csv に必要列がありません: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    else:
        df["probability"] = np.nan

    df = df.dropna(subset=["date", "ticker", "score"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

    # 同じ日・銘柄が重複していても1件へ正規化
    df = (
        df.sort_values(["date", "score"], ascending=[True, False])
        .drop_duplicates(["date", "ticker"], keep="first")
    )

    # rank が空/異常な履歴にも対応し、score順で TOP3 を作る
    df = df.sort_values(["date", "score"], ascending=[True, False])
    df["calc_rank"] = df.groupby("date")["score"].rank(method="first", ascending=False)
    df["rank"] = df["calc_rank"]
    return df[df["rank"] <= TOP_N].copy()


def read_intraday_csv(ticker, trade_date):
    """ローカルCSVを優先。長期5分足データを使う場合はこちら。"""
    safe = ticker.replace(".", "_")
    candidates = [
        os.path.join(INTRADAY_DIR, f"{safe}_{trade_date:%Y-%m-%d}_5m.csv"),
        os.path.join(INTRADAY_DIR, f"{safe}.csv"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue

            time_col = None
            for c in ["Datetime", "datetime", "Date", "date", "Timestamp", "timestamp"]:
                if c in df.columns:
                    time_col = c
                    break
            if time_col is None:
                continue

            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

            rename = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=rename)

            required = ["Open", "High", "Low", "Close", "Volume"]
            if any(c not in df.columns for c in required):
                continue

            return df
        except Exception:
            continue

    return None


def download_intraday(ticker, trade_date):
    """Yahoo Financeから直近5分足を取得。長期バックテストの主データ源にはしない。"""
    local = read_intraday_csv(ticker, trade_date)
    if local is not None:
        return local

    # yfinance は intraday の過去期間に制限があるため、直近範囲のみ取得
    start = pd.Timestamp(trade_date).tz_localize("Asia/Tokyo")
    end = start + pd.Timedelta(days=1)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"{ticker} {trade_date:%Y-%m-%d}: intraday取得失敗: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return None

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Asia/Tokyo").tz_localize(None)

    return df.sort_index()


def calc_intraday_indicators(df):
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # VWAP
    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    cum_vol = x["Volume"].replace(0, np.nan).cumsum()
    x["vwap"] = (typical * x["Volume"]).cumsum() / cum_vol

    # EMA
    x["ema5"] = x["Close"].ewm(span=5, adjust=False).mean()
    x["ema20"] = x["Close"].ewm(span=20, adjust=False).mean()

    # 当該日の平均出来高に対する倍率
    x["vol_ma20"] = x["Volume"].rolling(20, min_periods=5).mean()
    x["vol_ratio"] = x["Volume"] / x["vol_ma20"].replace(0, np.nan)

    # ATR
    prev_close = x["Close"].shift(1)
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev_close).abs(),
            (x["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=5).mean()

    return x


def get_previous_close(ticker, trade_date):
    """当日のギャップ判定用。日足1日分だけ取得する。"""
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
            return np.nan
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d = d[d.index < trade_date]
        if d.empty:
            return np.nan
        return float(d["Close"].iloc[-1])
    except Exception:
        return np.nan


def find_entry(intra, previous_close, ai_score, ai_prob):
    if intra is None or intra.empty:
        return None

    x = calc_intraday_indicators(intra)
    x = x[(x.index.time >= ENTRY_START) & (x.index.time <= ENTRY_END)]
    if x.empty:
        return None

    if ai_score < MIN_SCORE:
        return None
    if not np.isnan(ai_prob) and ai_prob < MIN_UP_PROB:
        return None

    first_open = float(x["Open"].iloc[0])
    if previous_close > 0:
        gap_pct = (first_open / previous_close - 1.0) * 100.0
        if gap_pct > MAX_GAP_PCT:
            return None

    # 買い判定は「確定した足」の次の足の始値で行う。
    # これにより同一足の未来情報を使わない。
    for i in range(len(x) - 1):
        bar = x.iloc[i]
        nxt = x.iloc[i + 1]

        cond_vwap = float(bar["Close"]) > float(bar["vwap"])
        cond_trend = float(bar["ema5"]) > float(bar["ema20"])
        cond_volume = bool(pd.notna(bar["vol_ratio"]) and bar["vol_ratio"] >= MIN_VOL_RATIO)

        if cond_vwap and cond_trend and cond_volume:
            atr = float(bar["atr"]) if pd.notna(bar["atr"]) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                atr = float(bar["Close"]) * 0.005

            entry_price = float(nxt["Open"])
            return {
                "entry_time": nxt.name,
                "entry_price": entry_price,
                "atr": atr,
                "gap_pct": (first_open / previous_close - 1.0) * 100.0 if previous_close > 0 else np.nan,
            }

    return None


def exit_trade(intra, entry):
    x = calc_intraday_indicators(intra)
    future = x[x.index >= entry["entry_time"]]
    if future.empty:
        return None

    entry_price = entry["entry_price"]
    tp = entry_price + ATR_TP * entry["atr"]
    sl = entry_price - ATR_SL * entry["atr"]

    for ts, bar in future.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        # 同一5分足で両方到達した場合は保守的にSL優先。
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


def backtest():
    top3 = load_top3_history()
    if top3.empty:
        raise ValueError("対象期間にTOP3履歴がありません")

    results = []
    cache = {}

    for trade_date, day in top3.groupby("date"):
        print(f"\n===== {trade_date:%Y-%m-%d} =====")
        day = day.sort_values("rank").head(TOP_N)

        for _, row in day.iterrows():
            ticker = str(row["ticker"])
            cache_key = (ticker, trade_date.date())

            if cache_key not in cache:
                cache[cache_key] = download_intraday(ticker, trade_date)
                time.sleep(0.2)

            intra = cache[cache_key]
            if intra is None or intra.empty:
                print(f"{ticker}: intradayデータなし → SKIP")
                continue

            previous_close = get_previous_close(ticker, trade_date)
            ai_score = float(row["score"])
            ai_prob = float(row["probability"]) if pd.notna(row["probability"]) else np.nan

            entry = find_entry(intra, previous_close, ai_score, ai_prob)
            if entry is None:
                results.append({
                    "date": trade_date.date(),
                    "ticker": ticker,
                    "rank": int(row["rank"]),
                    "score": ai_score,
                    "probability": ai_prob,
                    "status": "NO_ENTRY",
                    "entry_time": "",
                    "entry_price": np.nan,
                    "exit_time": "",
                    "exit_price": np.nan,
                    "return_pct": 0.0,
                    "reason": "chart_condition_not_met",
                })
                print(f"{ticker} TOP{int(row['rank'])}: NO ENTRY")
                continue

            exited = exit_trade(intra, entry)
            if exited is None:
                continue

            exit_time, exit_price, reason = exited
            gross = (exit_price / entry["entry_price"] - 1.0) * 100.0
            net = gross - (FEE_RATE * 2.0 * 100.0)

            results.append({
                "date": trade_date.date(),
                "ticker": ticker,
                "rank": int(row["rank"]),
                "score": ai_score,
                "probability": ai_prob,
                "status": "TRADE",
                "entry_time": entry["entry_time"],
                "entry_price": entry["entry_price"],
                "exit_time": exit_time,
                "exit_price": exit_price,
                "return_pct": net,
                "reason": reason,
                "gap_pct": entry["gap_pct"],
                "tp_distance_atr": ATR_TP,
                "sl_distance_atr": ATR_SL,
            })
            print(
                f"{ticker} TOP{int(row['rank'])}: BUY {entry['entry_price']:.2f} "
                f"→ {exit_price:.2f} {reason} = {net:+.2f}%"
            )

    out = pd.DataFrame(results)
    if out.empty:
        raise ValueError("バックテスト結果が0件です")

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "rank", "ticker"]).reset_index(drop=True)

    # 1営業日につき「最初に条件成立したTOP3の1銘柄」だけ実際に売買する。
    # これは資金をTOP3へ分割せず、毎日1トレードに寄せたい場合の仕様。
    trades = out[out["status"] == "TRADE"].copy()
    selected = (
        trades.sort_values(["date", "rank", "entry_time"])
        .groupby("date", as_index=False)
        .head(1)
        .copy()
    )

    # 日別複利
    capital = INITIAL_CAPITAL
    equity_rows = []
    trade_dates = pd.date_range(START_DATE, END_DATE, freq="B")
    selected_by_date = {r["date"]: r for _, r in selected.iterrows()}

    for d in trade_dates:
        d = pd.Timestamp(d)
        r = selected_by_date.get(d)
        ret = float(r["return_pct"]) / 100.0 if r is not None else 0.0
        capital *= (1.0 + ret)
        equity_rows.append({"date": d, "capital": capital, "daily_return_pct": ret * 100.0})

    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity["peak"] = equity["capital"].cummax()
        equity["drawdown_pct"] = (equity["capital"] / equity["peak"] - 1.0) * 100.0
        max_dd = float(equity["drawdown_pct"].min())
    else:
        max_dd = 0.0

    if not selected.empty:
        win_rate = float((selected["return_pct"] > 0).mean() * 100.0)
        avg_trade = float(selected["return_pct"].mean())
        gross_profit = float(selected.loc[selected["return_pct"] > 0, "return_pct"].sum())
        gross_loss = float(-selected.loc[selected["return_pct"] < 0, "return_pct"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    else:
        win_rate = 0.0
        avg_trade = 0.0
        pf = 0.0

    final_capital = float(equity["capital"].iloc[-1]) if not equity.empty else INITIAL_CAPITAL
    total_return = (final_capital / INITIAL_CAPITAL - 1.0) * 100.0

    out_file = "intraday_top3_backtest_results.csv"
    equity_file = "intraday_top3_equity.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("📊 TOP3・寄り後チャート確認・当日決済 バックテスト")
    print("=" * 60)
    print(f"期間              : {START_DATE.date()} ～ {END_DATE.date()}")
    print(f"TOP_N             : {TOP_N}")
    print(f"エントリー時間    : {ENTRY_START} ～ {ENTRY_END}")
    print(f"最低AIスコア      : {MIN_SCORE}")
    print(f"最低上昇確率      : {MIN_UP_PROB}")
    print(f"VWAP上             : YES")
    print(f"EMA5 > EMA20       : YES")
    print(f"出来高倍率         : >= {MIN_VOL_RATIO}x")
    print(f"TP / SL             : {ATR_TP}x ATR / {ATR_SL}x ATR")
    print(f"1日1トレード       : YES（TOP3の最初の成立銘柄）")
    print("-" * 60)
    print(f"トレード件数       : {len(selected)}")
    print(f"勝率               : {win_rate:.2f}%")
    print(f"平均1回リターン    : {avg_trade:+.3f}%")
    print(f"PF                 : {pf:.3f}")
    print(f"最大DD             : {max_dd:.2f}%")
    print(f"初期資金           : ¥{INITIAL_CAPITAL:,.0f}")
    print(f"最終資金           : ¥{final_capital:,.0f}")
    print(f"累積利益率         : {total_return:+.2f}%")
    print(f"結果CSV            : {out_file}")
    print(f"資産曲線CSV        : {equity_file}")


if __name__ == "__main__":
    backtest()
