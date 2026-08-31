#!/usr/bin/env python3
"""Run the existing signed TOP10 paper loop with profit-priority ranking.

Adds a conservative same-ticker cooldown: after a position is detected as
closed, that exact ticker is blocked for 30 minutes. Other tickers remain
eligible immediately. The cooldown is applied to normal and fallback paper
candidates alike and is persisted inside the existing paper state.
"""
from datetime import datetime, timedelta
import os

import profit_top10_paper as app

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
        # Score remains dominant; EV is the return-oriented tie breaker.
        rank = 0.65 * float(c.get("score", 0)) + 0.35 * max(-10.0, min(10.0, ev)) * 10.0
        item = dict(c)
        item["profit_ev_pct"] = round(ev, 4)
        item["profit_priority"] = round(rank, 4)
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda x: (
            x["profit_priority"],
            x.get("score", 0),
            x.get("up_probability", 0),
        ),
        reverse=True,
    )


_original_scan = app.scan_candidates


def scan_candidates_profit_first(policy):
    candidates, scanned = _original_scan(policy)
    ranked = profit_priority(candidates)
    print(f"💰 Profit-priority ranking: candidates={len(candidates)} scanned={scanned}")
    if ranked:
        print("TOP5:")
        for c in ranked[:5]:
            print(
                f"  {c['ticker']} score={c['score']:.1f} "
                f"UP={c['up_probability']:.1f}% EV={c['profit_ev_pct']:+.2f}% "
                f"rank={c['profit_priority']:.1f}"
            )
    return ranked, scanned


_original_close = app.close_positions


def _as_aware_jst(value):
    ts = value if isinstance(value, datetime) else app.pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=app.TZ)
    return ts.astimezone(app.TZ)


def close_positions_with_cooldown(state, policy, now):
    before = {
        str(p.get("ticker"))
        for p in state.get("positions", [])
        if p.get("ticker")
    }
    messages = _original_close(state, policy, now)
    after = {
        str(p.get("ticker"))
        for p in state.get("positions", [])
        if p.get("ticker")
    }
    closed_tickers = before - after

    cooldowns = state.setdefault("last_exit_by_ticker", {})
    for ticker in sorted(closed_tickers):
        cooldowns[ticker] = app.pd.Timestamp(now).isoformat()
        print(
            f"⏳ 同一銘柄クールダウン開始: {ticker} "
            f"{SAME_TICKER_COOLDOWN_MINUTES}分"
        )
    return messages


def open_positions_with_cooldown(state, policy, candidates, today):
    cooldowns = state.setdefault("last_exit_by_ticker", {})
    if state.get("cooldown_count_date") != today:
        state["cooldown_count_date"] = today
        state["cooldown_skip_count"] = 0

    now = datetime.now(app.TZ)
    filtered = []
    skipped = []
    expired = []

    for candidate in candidates:
        ticker = str(candidate.get("ticker", "")).strip()
        raw = cooldowns.get(ticker)
        if not raw:
            filtered.append(candidate)
            continue

        try:
            last_exit = _as_aware_jst(raw)
            until = last_exit + timedelta(minutes=SAME_TICKER_COOLDOWN_MINUTES)
            remaining = (until - now).total_seconds()
        except Exception:
            # Corrupt cooldown timestamps must not block a ticker forever.
            expired.append(ticker)
            filtered.append(candidate)
            continue

        if remaining > 0:
            skipped.append((ticker, remaining))
            continue

        expired.append(ticker)
        filtered.append(candidate)

    for ticker in expired:
        cooldowns.pop(ticker, None)

    state["cooldown_skip_count"] = int(state.get("cooldown_skip_count", 0)) + len(skipped)

    if skipped:
        for ticker, remaining in skipped[:10]:
            print(
                f"⏸ 同一銘柄クールダウン中: {ticker} "
                f"残り約{remaining / 60:.1f}分"
            )
        if len(skipped) > 10:
            print(f"⏸ クールダウン除外: {len(skipped)}銘柄")

    return _original_open(state, policy, filtered, today)


app.scan_candidates = scan_candidates_profit_first
app.close_positions = close_positions_with_cooldown
app.open_positions = open_positions_with_cooldown


if __name__ == "__main__":
    app.main()
