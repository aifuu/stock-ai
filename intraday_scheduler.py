from __future__ import annotations

"""日中ペーパートレードの段階式スケジューラ。

08:30-09:00  前日までの日足で寄り付き候補を準備
09:00-09:10  寄り付き後の1分足/5分足・出来高・騰落トレンドで再評価
09:30        支持抵抗・MA・VWAP・出来高・モメンタムを総合してTOP10/TOP1確定
09:30以降    TOP1を中心に継続ペーパー売買

売買ルールは検証用OOSゲートとは独立した「ペーパー実行ルール」。
同一銘柄のみ決済後30分クールダウンし、他銘柄は通常どおり候補にできる。
1日最大30回、かつ通常候補がなくても14:45時点で0件ならforced_min_tradeを1回だけ発火する。
forced_min_trade は本体シグナル成績から分離する前提。
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("intraday_scheduler")

PREMARKET_START = dtime(8, 30)
PREMARKET_END = dtime(9, 0)
OPEN_RESCORE_START = dtime(9, 0)
OPEN_RESCORE_END = dtime(9, 10)
FINAL_DECISION_TIME = dtime(9, 30)
MARKET_CLOSE = dtime(15, 30)

COOLDOWN_MINUTES = 30
MAX_TRADES_PER_DAY = 30
MIN_TRADES_PER_DAY = 1
FORCE_MIN_TRADE_DEADLINE = dtime(14, 45)
TRADING_LOOP_INTERVAL_MINUTES = 5


@dataclass
class Candidate:
    symbol: str
    score: float
    selection_mode: str = "normal"


@dataclass
class TradeRecord:
    symbol: str
    timestamp: datetime
    selection_mode: str


@dataclass
class SessionState:
    trade_date: datetime
    premarket_candidates: list[Candidate] = field(default_factory=list)
    top10: list[Candidate] = field(default_factory=list)
    top1: Optional[Candidate] = None
    trades_today: list[TradeRecord] = field(default_factory=list)
    symbol_last_trade_time: dict[str, datetime] = field(default_factory=dict)
    forced_min_trade_done: bool = False

    def trade_count(self) -> int:
        return len(self.trades_today)

    def is_symbol_in_cooldown(self, symbol: str, now: datetime) -> bool:
        last = self.symbol_last_trade_time.get(symbol)
        if last is None:
            return False
        return (now - last) < timedelta(minutes=COOLDOWN_MINUTES)

    def record_trade(self, symbol: str, now: datetime, mode: str) -> None:
        self.trades_today.append(TradeRecord(symbol, now, mode))
        self.symbol_last_trade_time[symbol] = now


# ---------------------------------------------------------------------------
# 既存システム接続フック
# ---------------------------------------------------------------------------

def run_premarket_screening(state: SessionState) -> list[Candidate]:
    """08:30-09:00: 前日までのデータで寄り付き候補を作る。

    実運用では stock_scan.py / daily_directional_top1.py の既存スコアを
    ここへ接続する。接続前は空リストを返して安全に停止する。
    """
    log.info("[08:30-09:00] 前日データによる寄り付き候補分析")
    candidates: list[Candidate] = []
    state.premarket_candidates = candidates
    return candidates


def run_open_rescoring(state: SessionState) -> list[Candidate]:
    """09:00-09:10: 1分足/5分足、出来高、騰落、トレンドで再評価。"""
    log.info("[09:00-09:10] 寄り付き後の1分足/5分足・出来高・トレンド再評価")
    rescored: list[Candidate] = []
    # 実装接続後に state.premarket_candidates を再スコアする。
    state.top10 = sorted(rescored, key=lambda c: c.score, reverse=True)[:10]
    return state.top10


def run_final_decision(state: SessionState) -> Optional[Candidate]:
    """09:30: 支持/抵抗・MA・VWAP・出来高・モメンタム等でTOP10/TOP1確定。"""
    log.info("[09:30] 支持抵抗/MA/VWAP/出来高/モメンタム総合評価")
    final_top10: list[Candidate] = []
    state.top10 = sorted(final_top10, key=lambda c: c.score, reverse=True)[:10]
    state.top1 = state.top10[0] if state.top10 else None
    if state.top1:
        log.info("TOP1確定: %s score=%.3f", state.top1.symbol, state.top1.score)
    return state.top1


def select_next_trade_candidate(state: SessionState, now: datetime) -> Optional[Candidate]:
    """TOP10から、同一銘柄30分クールダウン外の最有力銘柄を選ぶ。"""
    if state.trade_count() >= MAX_TRADES_PER_DAY:
        return None
    eligible = [c for c in state.top10 if not state.is_symbol_in_cooldown(c.symbol, now)]
    if not eligible:
        log.info("TOP10全銘柄が同一銘柄クールダウン中")
        return None
    eligible.sort(key=lambda c: c.score, reverse=True)
    return eligible[0]


def execute_paper_trade(state: SessionState, candidate: Candidate, now: datetime) -> None:
    """ペーパートレードを記録する。"""
    log.info(
        "ペーパー発注: %s mode=%s score=%.3f trades_today=%d",
        candidate.symbol,
        candidate.selection_mode,
        candidate.score,
        state.trade_count() + 1,
    )
    state.record_trade(candidate.symbol, now, candidate.selection_mode)


def enforce_minimum_daily_trade(state: SessionState, now: datetime) -> None:
    """14:45時点で0件なら、通常ゲートとは別のforced_min_tradeを1回だけ行う。"""
    if state.forced_min_trade_done or state.trade_count() >= MIN_TRADES_PER_DAY:
        return
    if now.time() < FORCE_MIN_TRADE_DEADLINE:
        return

    candidate = state.top1 or (state.top10[0] if state.top10 else None)
    if candidate is None:
        log.warning("forced_min_trade: TOP10/TOP1が存在しないため発注不能")
        return
    if state.is_symbol_in_cooldown(candidate.symbol, now):
        alternatives = [c for c in state.top10 if not state.is_symbol_in_cooldown(c.symbol, now)]
        candidate = alternatives[0] if alternatives else None
    if candidate is None:
        log.warning("forced_min_trade: クールダウン外の銘柄がありません")
        return

    forced = Candidate(candidate.symbol, candidate.score, "forced_min_trade")
    execute_paper_trade(state, forced, now)
    state.forced_min_trade_done = True


def run_trading_loop(
    state: SessionState,
    now_provider=datetime.now,
    sleep_seconds: int = 60,
) -> None:
    log.info("[09:30以降] ペーパートレードループ開始")
    last_cycle_time: Optional[datetime] = None

    while True:
        now = now_provider()
        if now.time() >= MARKET_CLOSE:
            break

        enforce_minimum_daily_trade(state, now)

        if state.trade_count() < MAX_TRADES_PER_DAY:
            if last_cycle_time is None or (now - last_cycle_time) >= timedelta(minutes=TRADING_LOOP_INTERVAL_MINUTES):
                candidate = select_next_trade_candidate(state, now)
                if candidate is not None:
                    execute_paper_trade(state, candidate, now)
                last_cycle_time = now

        time.sleep(sleep_seconds)


def run_full_day(trade_date: Optional[datetime] = None) -> SessionState:
    trade_date = trade_date or datetime.now()
    state = SessionState(trade_date=trade_date)
    run_premarket_screening(state)
    run_open_rescoring(state)
    run_final_decision(state)
    run_trading_loop(state)
    normal = sum(t.selection_mode == "normal" for t in state.trades_today)
    forced = sum(t.selection_mode == "forced_min_trade" for t in state.trades_today)
    log.info("当日取引件数=%d (normal=%d forced_min_trade=%d)", state.trade_count(), normal, forced)
    return state


if __name__ == "__main__":
    run_full_day()
