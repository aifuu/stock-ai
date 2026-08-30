"""Enrich walk-forward candidates with causal regime/relative-strength fields.

Only information available on each candidate date is used. No future return,
TP/SL result, or later market data is used to build the regime or strength.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf

INPUT = "walk_forward_all_candidates.csv"
OUTPUT = "walk_forward_all_candidates.csv"


def download(tickers, start, end):
    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("価格データを取得できませんでした")
    return df


def close_series(batch, ticker):
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker not in batch.columns.get_level_values(0):
            return pd.Series(dtype=float)
        return batch[ticker]["Close"].dropna()
    if ticker == "SINGLE" and "Close" in batch.columns:
        return batch["Close"].dropna()
    return pd.Series(dtype=float)


def main():
    c = pd.read_csv(INPUT)
    if c.empty:
        raise RuntimeError(f"{INPUT} が空です")
    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    c = c.dropna(subset=["date", "ticker"]).copy()

    start = (c["date"].min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (c["date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    tickers = sorted(c["ticker"].astype(str).unique())

    prices = download(tickers, start, end)
    nikkei = yf.download("^N225", start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
    if nikkei is None or nikkei.empty:
        raise RuntimeError("日経平均データ取得失敗")
    if isinstance(nikkei.columns, pd.MultiIndex):
        nikkei.columns = nikkei.columns.get_level_values(0)
    nk_close = nikkei["Close"].squeeze().dropna()
    nk5 = nk_close.pct_change(5)
    nk25 = nk_close.rolling(25).mean()
    nk75 = nk_close.rolling(75).mean()

    stock_cache = {}
    for t in tickers:
        s = close_series(prices, t)
        if s.empty:
            continue
        stock_cache[t] = s.pct_change(5)

    rs = []
    regimes = []
    for _, row in c.iterrows():
        d = row["date"]
        t = str(row["ticker"])
        stock_ret5 = float(stock_cache.get(t, pd.Series(dtype=float)).get(d, np.nan))
        nk_ret5 = float(nk5.get(d, np.nan))
        rs.append(stock_ret5 - nk_ret5 if np.isfinite(stock_ret5) and np.isfinite(nk_ret5) else 0.0)
        kairi = float(((nk_close.get(d, np.nan) / nk25.get(d, np.nan)) - 1.0) * 100.0) if np.isfinite(nk_close.get(d, np.nan)) and np.isfinite(nk25.get(d, np.nan)) and nk25.get(d, np.nan) != 0 else 0.0
        if kairi > 0 and nk_ret5 > 0:
            regimes.append("RISK_ON")
        elif kairi < 0 and nk_ret5 < 0:
            regimes.append("RISK_OFF")
        else:
            regimes.append("NEUTRAL")

    c["relative_strength"] = rs
    c["market_regime"] = regimes
    c.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ 候補へcausal regime/relative_strength追加: {len(c)} rows")
    print(c["market_regime"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
