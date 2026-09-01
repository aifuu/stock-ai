from __future__ import annotations

"""日中ペーパートレードの段階式スケジューラ。

08:30-09:00  前日までの日足データで寄り付き候補準備
09:00-09:10  寄り付き後の1分足/5分足・出来高・騰落トレンド再評価
09:10-09:30  継続再評価
09:30以降    既存Profit LoopによるTOP10→TOP1ペーパートレード

重要:
- OOS/Adversarialの採用ゲートとは独立したペーパー実行ルール。
- 同一銘柄は決済後30分だけ再取引禁止。別銘柄は選択可能。
- 1日最大30回。
- 1日最低1回は14:45時点で0件ならforced_min_tradeを発火するための状態フラグを保持。
- forced_min_trade は normal シグナル成績から分離する前提。

このモジュール自体は「時間フェーズと実行ルールの司令塔」であり、実際の
銘柄評価・売買執行は既存の run_profit_loop.py / profit_top10_paper.py に接続する。
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("intraday_scheduler")

JST = ZoneInfo("Asia/Tokyo")
STATE_FILE = Path("intraday_scheduler_state.json")
PREMARKET_START_MIN = 8 * 60 + 30
OPEN_RESCORE_START_MIN = 9 * 60
OPEN_RESCORE_END_MIN = 9 * 60 + 10
FINAL_DECISION_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 15 * 60 + 30
COOLDOWN_MINUTES = 30
MAX_TRADES_PER_DAY = 30
MIN_TRADES_PER_DAY = 1
FORCE_MIN_TRADE_DEADLINE_MIN = 14 * 60 + 45


def now_jst() -> datetime:
    return datetime.now(JST)


def minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def load_state() -> dict:
    today = now_jst().strftime("%Y-%m-%d")
    default = {
        "date": today,
        "phase": "",
        "premarket_done": False,
        "open_rescore_done": False,
        "final_decision_done": False,
        "forced_min_trade_done": False,
        "trades_today": 0,
        "last_exit_by_ticker": {},
    }
    if not STATE_FILE.exists():
        return default
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if state.get("date") != today:
            return default
        default.update(state)
        return default
    except Exception as exc:
        log.warning("scheduler state読込失敗: %s", exc)
        return default


def save_state(state: dict) -> None:
    state["updated_at"] = now_jst().isoformat()
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def determine_phase(now: datetime | None = None) -> str:
    now = now or now_jst()
    m = minute_of_day(now)
    if PREMARKET_START_MIN <= m < OPEN_RESCORE_START_MIN:
        return "premarket"
    if OPEN_RESCORE_START_MIN <= m < OPEN_RESCORE_END_MIN:
        return "open"
    if OPEN_RESCORE_END_MIN <= m < FINAL_DECISION_MIN:
        return "rescore"
    if FINAL_DECISION_MIN <= m < MARKET_CLOSE_MIN:
        return "trading"
    if m >= MARKET_CLOSE_MIN:
        return "closed"
    return "before_premarket"


def update_phase_state(phase: str) -> dict:
    state = load_state()
    state["phase"] = phase
    if phase == "premarket":
        state["premarket_done"] = True
        log.info("[08:30-09:00] 前日データ候補分析フェーズ")
    elif phase == "open":
        state["open_rescore_done"] = True
        log.info("[09:00-09:10] 寄り付き後1分足/5分足・出来高・騰落トレンド再評価フェーズ")
    elif phase == "rescore":
        log.info("[09:10-09:30] TOP10継続再評価フェーズ")
    elif phase == "trading":
        state["final_decision_done"] = True
        log.info("[09:30-15:30] TOP10→TOP1ペーパー取引フェーズ")
    elif phase == "closed":
        log.info("[15:30以降] 市場終了")
    else:
        log.info("現在フェーズ: %s", phase)
    save_state(state)
    return state


def apply_exit_cooldown(ticker: str, exit_time: datetime | None = None) -> dict:
    state = load_state()
    ts = exit_time or now_jst()
    state.setdefault("last_exit_by_ticker", {})[ticker] = ts.isoformat()
    save_state(state)
    log.info("⏳ 同一銘柄クールダウン開始: %s %d分", ticker, COOLDOWN_MINUTES)
    return state


def is_in_cooldown(ticker: str, now: datetime | None = None) -> bool:
    state = load_state()
    raw = state.get("last_exit_by_ticker", {}).get(ticker)
    if not raw:
        return False
    try:
        last = datetime.fromisoformat(raw)
        if last.tzinfo is None:
            last = last.replace(tzinfo=JST)
        current = now or now_jst()
        remaining = (last + timedelta(minutes=COOLDOWN_MINUTES) - current).total_seconds()
        if remaining > 0:
            log.info("⏸ 同一銘柄クールダウン中: %s 残り約%d分", ticker, int((remaining + 59) // 60))
            return True
        state["last_exit_by_ticker"].pop(ticker, None)
        save_state(state)
        return False
    except Exception:
        return False


def can_start_paper_trading(now: datetime | None = None) -> bool:
    now = now or now_jst()
    return minute_of_day(now) >= FINAL_DECISION_MIN and minute_of_day(now) < MARKET_CLOSE_MIN


def can_force_min_trade(now: datetime | None = None) -> bool:
    now = now or now_jst()
    state = load_state()
    if state.get("forced_min_trade_done"):
        return False
    return state.get("trades_today", 0) < MIN_TRADES_PER_DAY and minute_of_day(now) >= FORCE_MIN_TRADE_DEADLINE_MIN


def record_trade(ticker: str, mode: str = "normal") -> dict:
    state = load_state()
    trades = int(state.get("trades_today", 0))
    if trades >= MAX_TRADES_PER_DAY:
        raise RuntimeError(f"1日最大取引回数{MAX_TRADES_PER_DAY}回に到達")
    state["trades_today"] = trades + 1
    if mode == "forced_min_trade":
        state["forced_min_trade_done"] = True
    save_state(state)
    log.info("📄 paper trade recorded: %s mode=%s count=%d/%d", ticker, mode, state["trades_today"], MAX_TRADES_PER_DAY)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["auto", "premarket", "open", "rescore", "final", "trading", "closed"], default="auto")
    parser.add_argument("--record-trade", dest="record_trade_ticker")
    parser.add_argument("--record-mode", choices=["normal", "forced_min_trade"], default="normal")
    parser.add_argument("--record-exit", dest="record_exit_ticker")
    args = parser.parse_args()

    now = now_jst()
    phase = determine_phase(now) if args.phase == "auto" else args.phase
    if phase == "final":
        phase = "trading"

    state = update_phase_state(phase)

    if args.record_exit_ticker:
        apply_exit_cooldown(args.record_exit_ticker, now)

    if args.record_trade_ticker:
        if is_in_cooldown(args.record_trade_ticker, now):
            raise SystemExit(f"❌ {args.record_trade_ticker}: 同一銘柄30分クールダウン中")
        record_trade(args.record_trade_ticker, args.record_mode)

    print("========================================")
    print("INTRADAY SCHEDULER")
    print("========================================")
    print(f"JST: {now:%Y-%m-%d %H:%M:%S}")
    print(f"phase: {phase}")
    print(f"paper trading start: {'YES' if can_start_paper_trading(now) else 'NO'}")
    print(f"trades today: {state.get('trades_today', 0)}/{MAX_TRADES_PER_DAY}")
    print(f"minimum-trade force due: {'YES' if can_force_min_trade(now) else 'NO'}")


if __name__ == "__main__":
    main()
