#!/usr/bin/env python3
"""Run the existing signed TOP10 paper loop with profit-expectation ranking.

The policy gate and entry thresholds are untouched. This runner only changes
candidate ordering so that expected net upside/downside is considered alongside
AI score. That keeps the objective focused on return, not raw win rate.
"""
import profit_top10_paper as app


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
    return sorted(ranked, key=lambda x: (x["profit_priority"], x.get("score", 0), x.get("up_probability", 0)), reverse=True)


_original_scan = app.scan_candidates


def scan_candidates_profit_first(policy):
    candidates, scanned = _original_scan(policy)
    ranked = profit_priority(candidates)
    print(f"💰 Profit-priority ranking: candidates={len(candidates)} scanned={scanned}")
    if ranked:
        print("TOP5:")
        for c in ranked[:5]:
            print(f"  {c['ticker']} score={c['score']:.1f} UP={c['up_probability']:.1f}% EV={c['profit_ev_pct']:+.2f}% rank={c['profit_priority']:.1f}")
    return ranked, scanned


app.scan_candidates = scan_candidates_profit_first

if __name__ == "__main__":
    app.main()
