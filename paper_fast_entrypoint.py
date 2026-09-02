#!/usr/bin/env python3
"""Lightweight Paper Trading entrypoint.

The existing Profit Loop keeps its progressive LEVEL1-6 semantics, but this
entrypoint makes the expensive universe scan happen only once per workflow run.
Each level then reuses the same scanned candidate pool instead of downloading
all tickers again. This preserves the 5-minute latest-TOP1 purpose while
avoiding repeated yfinance scans inside a single run.
"""
import run_profit_loop as loop

_original_scan = loop._original_scan
_cache = {"result": None}


def cached_scan(policy):
    if _cache["result"] is None:
        base_policy = dict(policy)
        base_policy["up_threshold"] = 0.0
        base_policy["min_score_for_buy"] = 0.0
        base_policy["nikkei_filter"] = False
        print("⚡ PAPER FAST SCAN: 全銘柄スキャンはこの実行で1回だけ。LEVEL1-6は同一候補プールを再利用します。")
        _cache["result"] = _original_scan(base_policy)
    else:
        print("♻️ PAPER FAST SCAN CACHE: 既取得候補プールを再利用")
    return _cache["result"]


loop._original_scan = cached_scan

if __name__ == "__main__":
    loop.app.scan_candidates = loop.scan_candidates_progressive
    loop.app.close_positions = loop.close_positions_with_cooldown
    loop.app.open_positions = loop.open_top1_only
    loop.app.main()
