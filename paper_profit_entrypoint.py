#!/usr/bin/env python3
"""Profit Loopの本番ペーパー実行入口。

run_profit_loop.py の選定ロジックはそのまま使い、
新規エントリー直前だけ共通 paper_risk_policy で停止判定する。
決済処理は止めないため、リスク停止後も既存ポジションは正常に決済できる。
"""
import run_profit_loop as loop
from paper_risk_policy import position_allowed

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
