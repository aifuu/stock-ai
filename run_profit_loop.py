#!/usr/bin/env python3
"""Run the same regime/strength/expected-return ranking used by OOS validation.

Only the execution surface is changed: TOP_N is fixed to 1. Entry gates are
not loosened here; the approved policy and its signature remain authoritative.
"""
import profit_top10_paper as app
from regime_profit_core import load_expectancy_json, rank_candidates

# Live execution target: one daily TOP1 position, not TOP10.
app.TOP_N = 1
app.MAX_TOTAL_TRADES_PER_DAY = 1

_original_scan = app.scan_candidates


def scan_candidates_exact(policy):
    candidates, scanned = _original_scan(policy)
    table = load_expectancy_json(policy.get("regime_expectancy_json", ""))
    ranked = rank_candidates(candidates, table, top_n=1)
    for c in ranked:
        print(
            f"TOP1 {c.get('ticker')} | regime={c.get('market_regime')} | "
            f"bucket={c.get('strength_bucket')} | EV={float(c.get('expected_return', 0.0)):+.4f}% | "
            f"strength={float(c.get('individual_strength', 0.0)):.2f} | "
            f"score={float(c.get('score', 0.0)):.2f}"
        )
    return ranked, scanned


app.scan_candidates = scan_candidates_exact

if __name__ == "__main__":
    app.main()
