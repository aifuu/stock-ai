"""ペーパートレード共通リスクポリシー。

実行系はこのモジュールの閾値を唯一の基準として利用する。
OOS/optimizerの判定ロジックとは分離し、ペーパー売買の停止条件だけを管理する。
"""
import os

INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
MAX_POSITIONS = int(os.getenv("AI_MAX_POSITIONS", "10"))
MAX_DAILY_TRADES = int(os.getenv("MAX_TRADES_PER_DAY", "30"))
MAX_TRADES_PER_TICKER = int(os.getenv("MAX_TRADES_PER_TICKER_PER_DAY", "10"))
DAILY_STOP_LOSS = float(os.getenv("AI_DAILY_STOP_LOSS", "0.015"))
MAX_DRAWDOWN = float(os.getenv("AI_MAX_DRAWDOWN", "0.30"))
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
SLIPPAGE_RATE = float(os.getenv("AI_SLIPPAGE_RATE", "0.0005"))


def evaluate(state):
    """profit_top10_paperのstateだけを入力にして、共通リスク判定を返す。"""
    capital = float(state.get("capital", INITIAL_CAPITAL))
    start = float(state.get("daily_start_capital", capital))
    peak = float(state.get("peak", capital))
    positions = state.get("positions", []) or []
    daily_return = (capital / start - 1.0) if start > 0 else 0.0
    drawdown = (capital / peak - 1.0) if peak > 0 else 0.0

    if daily_return <= -DAILY_STOP_LOSS:
        return False, f"日次損失上限 {daily_return * 100:.2f}% <= {-DAILY_STOP_LOSS * 100:.2f}%"
    if drawdown <= -MAX_DRAWDOWN:
        return False, f"最大DD {drawdown * 100:.2f}% <= {-MAX_DRAWDOWN * 100:.2f}%"
    if len(positions) >= MAX_POSITIONS:
        return False, f"同時保有数上限 {len(positions)}/{MAX_POSITIONS}"
    if int(state.get("trades_today", 0)) >= MAX_DAILY_TRADES:
        return False, f"日次取引上限 {state.get('trades_today', 0)}/{MAX_DAILY_TRADES}"
    return True, "OK"


def position_allowed(state, ticker):
    """個別銘柄の新規エントリー可否を共通ルールで判定する。"""
    ok, reason = evaluate(state)
    if not ok:
        return False, reason
    count = int(state.get("trades_by_ticker_today", {}).get(ticker, 0))
    if count >= MAX_TRADES_PER_TICKER:
        return False, f"同一銘柄日次上限 {ticker} {count}/{MAX_TRADES_PER_TICKER}"
    return True, "OK"
