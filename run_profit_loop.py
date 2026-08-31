#!/usr/bin/env python3
"""Unified Profit Loop: rank 430 stocks -> TOP10 -> execute only TOP1.

The TOP10 is the daily candidate pool. At each market scan, the highest-ranked
eligible member of that TOP10 is the only paper position opened. Existing
positions are never duplicated. Daily total entries are capped at 10.
"""
from datetime import datetime, timedelta
import os

import profit_top10_paper as app

TOP10 = 10
MAX_DAILY_TRADES = 10
MAX_TICKER_TRADES = 10
SAME_TICKER_COOLDOWN_MINUTES = int(os.getenv("SAME_TICKER_COOLDOWN_MINUTES", "30"))


def profit_priority(candidates):
    ranked = []
    for c in candidates:
        price = float(c.get("price", 0) or 0)
        tp = float(c.get("tp", 0) or 0)
        sl = float(c.get("sl", 0) or 0)
        up = float(c.get("up_probability", 0) or 0) / 100.0
        if price <= 0:
            ev = -999.0
        else:
            reward = max(0.0, (tp / price - 1.0) * 100.0)
            risk = max(0.0, (1.0 - sl / price) * 100.0)
            ev = up * reward - (1.0 - up) * risk
        rank = 0.65 * float(c.get("score", 0)) + 0.35 * max(-10.0, min(10.0, ev)) * 10.0
        item = dict(c)
        item["profit_ev_pct"] = round(ev, 4)
        item["profit_priority"] = round(rank, 4)
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda x: (x["profit_priority"], x.get("score", 0), x.get("up_probability", 0)),
        reverse=True,
    )


_original_scan = app.scan_candidates


def scan_candidates_profit_first(policy):
    candidates, scanned = _original_scan(policy)
    ranked = profit_priority(candidates)
    top10 = ranked[:TOP10]
    print(f"💰 Profit-priority ranking: candidates={len(candidates)} scanned={scanned}")
    print(f"🏆 TOP10 candidate pool: {len(top10)}")
    for i, c in enumerate(top10, 1):
        print(
            f"  {i:02d}. {c['ticker']} score={c['score']:.1f} "
            f"UP={c['up_probability']:.1f}% EV={c['profit_ev_pct']:+.2f}% "
            f"rank={c['profit_priority']:.1f}"
        )
    return top10, scanned


_original_close = app.close_positions
_original_open = app.open_positions


def _as_aware_jst(value):
    ts = value if isinstance(value, datetime) else app.pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=app.TZ)
    return ts.astimezone(app.TZ)


def close_positions_with_cooldown(state, policy, now):
    before = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}
    messages = _original_close(state, policy, now)
    after = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}
    closed_tickers = before - after
    cooldowns = state.setdefault("last_exit_by_ticker", {})
    for ticker in sorted(closed_tickers):
        cooldowns[ticker] = app.pd.Timestamp(now).isoformat()
        print(f"⏳ 同一銘柄クールダウン開始: {ticker} {SAME_TICKER_COOLDOWN_MINUTES}分")
    return messages


def open_top1_only(state, policy, candidates, today):
    """TOP10 candidate poolから、現在最上位の1銘柄だけを新規採用する。"""
    cooldowns = state.setdefault("last_exit_by_ticker", {})
    now = datetime.now(app.TZ)
    active = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}
    eligible = []

    for candidate in candidates[:TOP10]:
        ticker = str(candidate.get("ticker", "")).strip()
        if not ticker or ticker in active:
            continue
        ticker_count = int(state.get("trades_by_ticker_today", {}).get(ticker, 0))
        if ticker_count >= MAX_TICKER_TRADES:
            continue
        if int(state.get("trades_today", 0)) >= MAX_DAILY_TRADES:
            break
        raw = cooldowns.get(ticker)
        if raw:
            try:
                remaining = (_as_aware_jst(raw) + timedelta(minutes=SAME_TICKER_COOLDOWN_MINUTES) - now).total_seconds()
            except Exception:
                remaining = 0
            if remaining > 0:
                continue
            cooldowns.pop(ticker, None)
        eligible.append(candidate)

    if not eligible:
        print("⏸ TOP10内に新規エントリー可能なTOP1なし")
        return []

    top1 = eligible[0]
    old_top_n = app.TOP_N
    old_max_total = app.MAX_TOTAL_TRADES_PER_DAY
    old_max_ticker = app.MAX_TRADES_PER_TICKER_PER_DAY
    try:
        app.TOP_N = 1
        app.MAX_TOTAL_TRADES_PER_DAY = MAX_DAILY_TRADES
        app.MAX_TRADES_PER_TICKER_PER_DAY = MAX_TICKER_TRADES
        opened = _original_open(state, policy, [top1], today)
    finally:
        app.TOP_N = old_top_n
        app.MAX_TOTAL_TRADES_PER_DAY = old_max_total
        app.MAX_TRADES_PER_TICKER_PER_DAY = old_max_ticker

    if opened:
        p = state["positions"][-1]
        p["allocation"] = 1.0
        p["selection_mode"] = "TOP10→TOP1"
        p["top10_rank"] = 1
        print(f"🏆 TOP10→TOP1 ENTRY: {top1['ticker']} score={top1['score']:.1f} UP={top1['up_probability']:.1f}%")
    return opened


app.scan_candidates = scan_candidates_profit_first
app.close_positions = close_positions_with_cooldown
app.open_positions = open_top1_only

if __name__ == "__main__":
    app.main()
