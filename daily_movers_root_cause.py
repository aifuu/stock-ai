#!/usr/bin/env python3
"""Daily movers observation layer.

This module is intentionally independent from the existing paper-trade selection path.
It observes the daily movers in the same universe, estimates measurable reasons for
up/down moves, stores the observations, and optionally reports the result to Discord.
It does NOT change strategy_policy.json, TOP1 selection, positions, or trade execution.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from common import TICKERS, COMPANY_NAMES
except Exception:
    from daily_directional_top1 import TICKERS, NAMES as COMPANY_NAMES

OUTPUT = Path("daily_movers_root_cause.csv")
HISTORY = Path("daily_movers_root_cause_history.csv")
TOP_N = int(os.getenv("DAILY_MOVERS_TOP_N", "10"))
MIN_PRICE = float(os.getenv("DAILY_MOVERS_MIN_PRICE", "100"))
MIN_AVG_VOLUME = int(os.getenv("DAILY_MOVERS_MIN_AVG_VOLUME", "300000"))


def _name(ticker: str) -> str:
    if isinstance(COMPANY_NAMES, dict):
        return str(COMPANY_NAMES.get(ticker, ticker))
    try:
        i = list(TICKERS).index(ticker)
        return str(COMPANY_NAMES[i]) if i < len(COMPANY_NAMES) else ticker
    except Exception:
        return ticker


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers, period="3mo", interval="1d", auto_adjust=False,
        group_by="ticker", threads=True, progress=False
    )
    out = {}
    if raw is None or raw.empty:
        return out
    if len(tickers) == 1:
        out[tickers[0]] = raw.dropna(how="all")
        return out
    for t in tickers:
        try:
            d = raw[t].copy().dropna(how="all")
            if not d.empty:
                out[t] = d
        except Exception:
            pass
    return out


def _market_return() -> float:
    try:
        d = yf.download("^N225", period="10d", interval="1d", auto_adjust=False, progress=False)
        close = d["Close"].squeeze().dropna()
        if len(close) >= 2:
            return float(close.iloc[-1] / close.iloc[-2] - 1.0)
    except Exception:
        pass
    return 0.0


def _cause(row: dict, market_ret: float) -> tuple[str, str]:
    reasons = []
    score = 0
    ret = row["return_pct"] / 100.0
    vol_ratio = row["volume_ratio"]
    gap = row["gap_pct"] / 100.0
    ma25 = row["ma25_gap_pct"] / 100.0
    rsi = row["rsi14"]
    range_pct = row["intraday_range_pct"] / 100.0

    if abs(market_ret) >= 0.008 and np.sign(ret) == np.sign(market_ret):
        reasons.append(f"日経連動({market_ret*100:+.2f}%)")
        score += 2
    if vol_ratio >= 2.0:
        reasons.append(f"出来高急増({vol_ratio:.1f}倍)")
        score += 2
    elif vol_ratio >= 1.4:
        reasons.append(f"出来高増({vol_ratio:.1f}倍)")
        score += 1
    if abs(gap) >= 0.02:
        reasons.append(f"寄り付きギャップ({gap*100:+.1f}%)")
        score += 2
    if abs(ma25) >= 0.03 and np.sign(ret) == np.sign(ma25):
        reasons.append(f"25日線乖離({ma25*100:+.1f}%)")
        score += 1
    if ret > 0 and rsi >= 70:
        reasons.append("過熱圏(RSI≥70)")
        score += 1
    elif ret < 0 and rsi <= 30:
        reasons.append("売られ過ぎ(RSI≤30)")
        score += 1
    if range_pct >= 0.06:
        reasons.append(f"日中値幅拡大({range_pct*100:.1f}%)")
        score += 1

    if not reasons:
        reasons.append("価格・出来高テクニカル要因のみでは主因を特定できず")
    confidence = "高" if score >= 5 else ("中" if score >= 3 else "低")
    return "＋".join(reasons[:4]), confidence


def main() -> int:
    tickers = list(dict.fromkeys(str(x) for x in TICKERS if str(x).endswith(".T")))
    frames = _download(tickers)
    market_ret = _market_return()
    rows = []
    for ticker, d in frames.items():
        try:
            close = d["Close"].astype(float).dropna()
            vol = d["Volume"].astype(float).dropna()
            high = d["High"].astype(float)
            low = d["Low"].astype(float)
            if len(close) < 30 or float(close.iloc[-1]) < MIN_PRICE:
                continue
            avg_vol = float(vol.tail(20).mean())
            if avg_vol < MIN_AVG_VOLUME:
                continue
            prev = float(close.iloc[-2])
            last = float(close.iloc[-1])
            day_ret = last / prev - 1.0
            volume_ratio = float(vol.iloc[-1] / max(1.0, vol.tail(20).mean()))
            gap = float(d["Open"].iloc[-1] / prev - 1.0)
            intraday_range = float(high.iloc[-1] / low.iloc[-1] - 1.0)
            ma25 = float(close.tail(25).mean())
            rsi = float(_rsi(close).iloc[-1]) if pd.notna(_rsi(close).iloc[-1]) else 50.0
            row = {
                "date": str(close.index[-1].date()),
                "ticker": ticker,
                "company": _name(ticker),
                "direction": "UP" if day_ret > 0 else "DOWN",
                "return_pct": round(day_ret * 100, 3),
                "volume_ratio": round(volume_ratio, 3),
                "gap_pct": round(gap * 100, 3),
                "intraday_range_pct": round(intraday_range * 100, 3),
                "ma25_gap_pct": round((last / ma25 - 1) * 100, 3),
                "rsi14": round(rsi, 2),
                "avg_volume20": int(avg_vol),
                "market_nikkei_return_pct": round(market_ret * 100, 3),
            }
            row["cause"], row["cause_confidence"] = _cause(row, market_ret)
            rows.append(row)
        except Exception:
            continue

    if not rows:
        print("⚠️ Daily movers: usable data not found")
        return 0

    df = pd.DataFrame(rows)
    df["abs_return"] = df["return_pct"].abs()
    # Keep both directions visible when possible: 5 strongest up + 5 strongest down.
    half = max(1, TOP_N // 2)
    ups = df[df.direction == "UP"].sort_values("abs_return", ascending=False).head(half)
    downs = df[df.direction == "DOWN"].sort_values("abs_return", ascending=False).head(TOP_N - len(ups))
    selected = pd.concat([ups, downs]).sort_values("abs_return", ascending=False).head(TOP_N).drop(columns=["abs_return"])

    selected.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    header = not HISTORY.exists()
    selected.to_csv(HISTORY, mode="a", header=header, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print(f"📊 毎日値動きTOP{TOP_N} 原因分析 | {selected.iloc[0]['date']} | 日経 {market_ret*100:+.2f}%")
    print("※観察・学習用。既存の売買選定/執行には介入しません。")
    print("=" * 90)
    for i, r in enumerate(selected.to_dict("records"), 1):
        print(f"{i:>2}. {r['ticker']} {r['company']} {r['direction']} {r['return_pct']:+.2f}% | 出来高{r['volume_ratio']:.1f}x | RSI {r['rsi14']:.0f} | {r['cause']} [{r['cause_confidence']}]")

    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if webhook:
        import requests
        lines = [f"📊 今日の値動きTOP{TOP_N}・原因分析 ({selected.iloc[0]['date']})", f"日経: {market_ret*100:+.2f}%", ""]
        for i, r in enumerate(selected.to_dict("records"), 1):
            lines.append(f"{i}. {r['ticker']} {r['company']} {r['direction']} {r['return_pct']:+.2f}%")
            lines.append(f"   原因: {r['cause']} / 信頼度:{r['cause_confidence']}")
        requests.post(webhook, json={"content": "\n".join(lines)[:1950]}, timeout=30).raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
