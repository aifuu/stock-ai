#!/usr/bin/env python3
"""Unified Profit Loop: progressive levels -> TOP10 -> TOP1 paper trade.

Research/OOS gates remain separate. This runtime path is for paper execution only.
It progressively relaxes entry thresholds until a usable candidate set exists,
then ranks the best candidates by profit priority and opens only TOP1.
"""
from datetime import datetime, timedelta
import os

import profit_top10_paper as app

TOP10 = 10
MAX_DAILY_TRADES = int(os.getenv("MAX_TRADES_PER_DAY", "30"))
MAX_TICKER_TRADES = int(os.getenv("MAX_TRADES_PER_TICKER_PER_DAY", "10"))
SAME_TICKER_COOLDOWN_MINUTES = int(os.getenv("SAME_TICKER_COOLDOWN_MINUTES", "30"))

# Paper-only progressive entry levels.
# Level 1 is strict; lower levels widen the pool. These values are runtime
# candidates, not research/OOS approval thresholds.
PAPER_ENTRY_LEVELS = [
    {"level": 1, "up_threshold": 60.0, "min_score": 70.0, "nikkei_filter": True},
    {"level": 2, "up_threshold": 55.0, "min_score": 65.0, "nikkei_filter": True},
    {"level": 3, "up_threshold": 50.0, "min_score": 60.0, "nikkei_filter": False},
    {"level": 4, "up_threshold": 45.0, "min_score": 55.0, "nikkei_filter": False},
    {"level": 5, "up_threshold": 40.0, "min_score": 50.0, "nikkei_filter": False},
]


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
        # Profit expectation has priority, score remains a tie-breaker.
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
_original_close = app.close_positions
_original_open = app.open_positions


def _policy_for_level(base_policy, spec):
    p = dict(base_policy)
    p["up_threshold"] = float(spec["up_threshold"])
    p["min_score_for_buy"] = float(spec["min_score"])
    p["nikkei_filter"] = bool(spec["nikkei_filter"])
    return p


def _passes_level(candidate, spec):
    up = float(candidate.get("up_probability", 0) or 0)
    down = float(candidate.get("down_probability", 0) or 0)
    flat = float(candidate.get("flat_probability", 0) or 0)
    score = float(candidate.get("score", 0) or 0)
    return (
        up >= float(spec["up_threshold"])
        and up > down
        and flat < 50.0
        and score >= float(spec["min_score"])
    )


def scan_candidates_progressive(policy):
    """Try paper-only levels from strict to relaxed until candidates exist.

    The underlying scanner may return its own fallback candidates when a policy
    produces no strict matches. Those fallback rows are filtered back out here,
    so a loose fallback cannot prematurely bypass the progressive levels.
    """
    last_scanned = 0
    last_pool = []

    for spec in PAPER_ENTRY_LEVELS:
        level_policy = _policy_for_level(policy, spec)
        raw, scanned = _original_scan(level_policy)
        last_scanned = scanned
        last_pool = raw or []
        qualified = [c for c in last_pool if _passes_level(c, spec)]
        ranked = profit_priority(qualified)
        top10 = ranked[:TOP10]

        print(
            f"🧭 PAPER LEVEL {spec['level']}: "
            f"UP≥{spec['up_threshold']:.0f}% SCORE≥{spec['min_score']:.0f} "
            f"NIKKEI={'ON' if spec['nikkei_filter'] else 'OFF'} "
            f"qualified={len(qualified)}"
        )

        if top10:
            for i, c in enumerate(top10, 1):
                print(
                    f"  {i:02d}. {c['ticker']} score={c['score']:.1f} "
                    f"UP={c['up_probability']:.1f}% EV={c.get('profit_ev_pct', 0):+.2f}% "
                    f"rank={c.get('profit_priority', 0):.1f}"
                )
            print(f"🏁 採用LEVEL={spec['level']} / TOP10={len(top10)}")
            for c in top10:
                c["selection_level"] = int(spec["level"])
                c["selection_mode"] = "normal" if spec["level"] == 1 else "progressive_level"
            return top10, scanned, int(spec["level"])

    # Final paper-only fallback: use the last available candidates, but tag it
    # separately. This is independent from OOS/research approval gates.
    if last_pool:
        ranked = profit_priority(last_pool)
        top10 = ranked[:TOP10]
        for c in top10:
            c["selection_level"] = len(PAPER_ENTRY_LEVELS) + 1
            c["selection_mode"] = "forced_min_trade"
        print("⚠️ 全通常LEVELで候補なし → 最終paper fallbackでTOP1候補を確保")
        return top10, last_scanned, len(PAPER_ENTRY_LEVELS) + 1

    return [], last_scanned, 0


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
    """Open only the highest profit-priority eligible candidate from TOP10."""
    cooldowns = state.setdefault("last_exit_by_ticker", {})
    now = datetime.now(app.TZ)
    active = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}
    eligible = []

    for candidate in profit_priority(candidates)[:TOP10]:
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
        p["selection_mode"] = top1.get("selection_mode", "normal")
        p["selection_level"] = int(top1.get("selection_level", 1))
        p["top10_rank"] = 1
        print(
            f"🏆 TOP10→TOP1 ENTRY: {top1['ticker']} "
            f"LEVEL={p['selection_level']} MODE={p['selection_mode']} "
            f"score={top1['score']:.1f} UP={top1['up_probability']:.1f}%"
        )
    return opened


app.scan_candidates = scan_candidates_progressive
app.close_positions = close_positions_with_cooldown
app.open_positions = open_top1_only

if __name__ == "__main__":
    app.main()
