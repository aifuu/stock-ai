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

# ★追加(2026-09): 実運用は1トレードにつき資金を全額集中投資する設計(ユーザー
# 承認済みのハイリスク運用)だが、これまでは-30%(MAX_DRAWDOWN)に達するまで
# 常にフルサイズで、そこで初めて新規エントリを完全停止する「オールオア
# ナッシング」だった。ドローダウンが深まるにつれ段階的にサイズを縮小し、
# -30%到達を避けやすくすることで、Fold3のような「どの戦略でも勝てない期間」
# でも資金を大きく減らさず市場回復を待てるようにする(ユーザー方針: 悪い期間は
# サイズを縮小して継続)。
DD_TIER1 = float(os.getenv("AI_DD_TIER1", "0.10"))
DD_TIER2 = float(os.getenv("AI_DD_TIER2", "0.20"))
DD_TIER2_SIZE = float(os.getenv("AI_DD_TIER2_SIZE", "0.50"))
DD_TIER3_SIZE = float(os.getenv("AI_DD_TIER3_SIZE", "0.25"))


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


def position_size_multiplier(state):
    """現在のピークからのドローダウンに応じて、新規エントリーの投資額を
    段階的に縮小する倍率を返す(1.0/0.5/0.25)。MAX_DRAWDOWN(既定30%)到達後は
    evaluate()側で新規エントリ自体が完全停止するため、ここでは考慮不要。
    """
    capital = float(state.get("capital", INITIAL_CAPITAL))
    peak = float(state.get("peak", capital))
    drawdown = (capital / peak - 1.0) if peak > 0 else 0.0
    dd_pct = -drawdown
    if dd_pct < DD_TIER1:
        return 1.0
    if dd_pct < DD_TIER2:
        return DD_TIER2_SIZE
    return DD_TIER3_SIZE
