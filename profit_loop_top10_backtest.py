"""Standalone backtest for the Profit Loop TOP10 idea.

This file is intentionally isolated from the production paper-trading scripts.
It uses prediction_history.csv when available and performs a conservative
trade simulation from historical prediction rows. It never modifies production
state files.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

START = os.getenv("BT_START_DATE", "2021-01-01")
END = os.getenv("BT_END_DATE", "2026-08-22")
INITIAL = float(os.getenv("BT_INITIAL_CAPITAL", "1000000"))
TOP_N = int(os.getenv("BT_TOP_N", "10"))
MAX_PER_TICKER_DAY = int(os.getenv("BT_MAX_PER_TICKER_DAY", "10"))
MAX_TRADES_DAY = int(os.getenv("BT_MAX_TRADES_DAY", "30"))
FEE = float(os.getenv("BT_FEE", "0.0005"))

INPUT = Path(os.getenv("BT_INPUT", "prediction_history.csv"))
OUT = Path("profit_loop_top10_backtest.csv")
MONTHLY = Path("profit_loop_top10_monthly.csv")


def main():
    if not INPUT.exists():
        raise SystemExit(f"❌ {INPUT} がありません。既存本編データを変更せず終了します。")

    df = pd.read_csv(INPUT)
    if df.empty:
        raise SystemExit("❌ prediction_history.csv が空です。")

    # Flexible column mapping for the repository's historical prediction data.
    date_col = next((c for c in ["timestamp", "date", "datetime", "time"] if c in df.columns), None)
    ticker_col = next((c for c in ["ticker", "symbol", "code"] if c in df.columns), None)
    price_col = next((c for c in ["price", "Close", "close"] if c in df.columns), None)
    score_col = next((c for c in ["ai_score", "score", "AIスコア"] if c in df.columns), None)
    prob_col = next((c for c in ["up_probability", "up_prob", "up_prob_pct", "上昇確率"] if c in df.columns), None)

    missing = [name for name, col in [("date", date_col), ("ticker", ticker_col), ("price", price_col)] if col is None]
    if missing:
        raise SystemExit(f"❌ 必須列が不足: {', '.join(missing)}。本編には変更を加えません。")

    df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["_date", ticker_col, price_col]).copy()
    df = df[(df["_date"] >= START) & (df["_date"] <= END)].copy()
    if df.empty:
        raise SystemExit(f"❌ {START}～{END} に利用可能な履歴がありません。")

    df["_price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["_price"])
    if score_col:
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    else:
        df["_score"] = 0.0
    if prob_col:
        df["_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0)
    else:
        df["_prob"] = 0.0

    # Conservative proxy: each historical prediction is an entry candidate;
    # the next available observation for that ticker is the exit observation.
    df = df.sort_values([ticker_col, "_date"])
    df["_exit_price"] = df.groupby(ticker_col)["_price"].shift(-1)
    df["_exit_date"] = df.groupby(ticker_col)["_date"].shift(-1)
    df["_ret"] = df["_exit_price"] / df["_price"] - 1.0
    df = df.dropna(subset=["_ret"]).copy()

    # Rank candidates by score/probability. If both are unavailable, preserve
    # historical order. Re-entry is allowed after the prior trade is complete.
    df["_rank_score"] = df["_score"] + df["_prob"]
    trades = []
    capital = INITIAL
    daily_counts = {}
    peak = INITIAL

    for day, daydf in df.groupby(df["_date"].dt.date, sort=True):
        daydf = daydf.sort_values(["_rank_score", "_date"], ascending=False)
        used = 0
        ticker_counts = {}
        for _, r in daydf.iterrows():
            if used >= MAX_TRADES_DAY:
                break
            ticker = str(r[ticker_col])
            count = ticker_counts.get(ticker, 0)
            if count >= MAX_PER_TICKER_DAY:
                continue
            # one position at a time per ticker; next historical row represents
            # the next opportunity after the previous observation.
            ret = float(r["_ret"])
            net = ret - 2 * FEE
            pnl = capital * (net / TOP_N)
            capital += pnl
            used += 1
            ticker_counts[ticker] = count + 1
            trades.append({
                "date": str(day), "ticker": ticker, "return": net,
                "pnl": pnl, "capital": capital,
            })
        daily_counts[str(day)] = used
        peak = max(peak, capital)

    result = pd.DataFrame(trades)
    if result.empty:
        raise SystemExit("❌ 有効なトレードが生成されませんでした。")

    result["drawdown"] = result["capital"] / result["capital"].cummax() - 1
    result.to_csv(OUT, index=False, encoding="utf-8-sig")

    result["month"] = pd.to_datetime(result["date"]).dt.to_period("M").astype(str)
    monthly = result.groupby("month").agg(
        trades=("pnl", "size"), pnl=("pnl", "sum"), end_capital=("capital", "last")
    ).reset_index()
    monthly.to_csv(MONTHLY, index=False, encoding="utf-8-sig")

    wins = (result["pnl"] > 0).sum()
    gross_profit = result.loc[result["pnl"] > 0, "pnl"].sum()
    gross_loss = -result.loc[result["pnl"] < 0, "pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss else np.inf
    max_dd = result["drawdown"].min()
    positive_months = (monthly["pnl"] > 0).mean() if len(monthly) else 0

    print("🧪 PROFIT LOOP TOP10 BACKTEST")
    print(f"期間：{START} ～ {END}")
    print(f"初期資金：{INITIAL:,.0f}円")
    print(f"総取引：{len(result)}")
    print(f"勝率：{wins / len(result) * 100:.2f}%")
    print(f"Profit Factor：{pf:.2f}")
    print(f"最大DD：{max_dd * 100:.2f}%")
    print(f"月間プラス率：{positive_months * 100:.2f}%")
    print(f"最終資産：{capital:,.0f}円")
    print(f"総利益：{capital - INITIAL:,.0f}円")
    print(f"同一銘柄最大取引/日：{MAX_PER_TICKER_DAY}")
    print(f"全体最大取引/日：{MAX_TRADES_DAY}")
    print(f"結果：{OUT}")
    print(f"月次：{MONTHLY}")


if __name__ == "__main__":
    main()
