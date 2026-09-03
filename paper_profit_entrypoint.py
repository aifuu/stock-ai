#!/usr/bin/env python3
"""Profit Loopの本番ペーパー実行入口。

run_profit_loop.py の選定ロジックはそのまま使い、
新規エントリー直前だけ共通 paper_risk_policy で停止判定する。
決済処理は止めないため、リスク停止後も既存ポジションは正常に決済できる。
"""
import run_profit_loop as loop
from paper_risk_policy import position_allowed

# run_profit_loop.py / profit_top10_paper.py の現行公開APIに統一する。
# scan() は (candidates, scanned) の2要素を返すため、そのまま利用する。
_original_scan = loop.app.scan


def compatible_scan(policy):
    result = _original_scan(policy)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("scan() returned an invalid result")
    candidates, scanned = result[0], result[1]
    return candidates, scanned


loop.app.scan = compatible_scan

_original_open = loop.app.open_positions


def guarded_open_positions(state, policy, candidates, today):
    if candidates:
        # TOP1を実際に開く前に共通リスクポリシーを一度だけ評価。
        top = candidates[0]
        ticker = str(top.get("ticker", "")).strip()
        ok, reason = position_allowed(state, ticker)
        if not ok:
            print(f"🛑 共通Paper Risk Gateで新規エントリー停止: {reason}")
            return []
    return _original_open(state, policy, candidates, today)


loop.app.open_positions = guarded_open_positions


if __name__ == "__main__":
    loop.app.main()
