from __future__ import annotations

"""日中ペーパートレードの段階式スケジューラ。

08:30-09:00  前日までの日足データで寄り付き候補準備
09:00-09:10  寄り付き後の1分足/5分足・出来高・騰落トレンド再評価
09:10-09:30  継続再評価
09:30以降    既存Profit LoopによるTOP10→TOP1ペーパートレード

重要:
- OOS/Adversarialの採用ゲートとは独立したペーパー実行ルール。
- 同一銘柄は決済後30分だけ再取引禁止。別銘柄は選択可能。
- 1日最大30回(実際の上限判定・カウントはprofit_top10_paper_state.json側で行う。
  詳細は下記2026-09追加コメント参照)。
- ★変更(2026-09): 以前あった「14:45時点で0件なら強制的に1回取引する
  (forced_min_trade)」という概念は、run_profit_loop.py側で無理やりエントリー
  する経路(緊い方の水準まで条件を緩める強制経路)を廃止した方針と矛盾するため
  廃止した。条件を満たす候補が無い日は無理に建てず見送るのが現行方針。
- ★修正(2026-09、追加): このモジュール自身が持っていたtrades_today/record_trade()
  は、実際の売買本体(run_profit_loop.py)からは一度も呼ばれておらず
  (--record-tradeはワークフロー上どこからも指定されない)、常に0のまま
  表示専用の別カウンターとして残っていた。実際の1日上限判定・カウントは
  profit_top10_paper_state.json側のtrades_todayで行われているため、
  二重管理による表示不一致を避けるためこちらのカウンターは廃止し、
  ステータス表示は実際の状態ファイルを直接参照する。

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
# 表示専用。実際の1日上限判定はrun_profit_loop.py側(MAX_TRADES_PER_DAY環境変数)で
# 行われるため、同じ環境変数名を参照して表示の食い違いを避ける。
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "30"))
PAPER_STATE_FILE = Path("profit_top10_paper_state.json")


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


def real_trades_today() -> int:
    """実際の売買本体(run_profit_loop.py/profit_top10_paper.py)が管理する
    profit_top10_paper_state.jsonから、本日の実取引数を読む(表示専用・
    ベストエフォート)。"""
    try:
        if not PAPER_STATE_FILE.exists():
            return 0
        state = json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))
        if state.get("trade_count_date") != now_jst().strftime("%Y-%m-%d"):
            return 0
        return int(state.get("trades_today", 0))
    except Exception as exc:
        log.warning("profit_top10_paper_state.json読込失敗: %s", exc)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["auto", "premarket", "open", "rescore", "final", "trading", "closed"], default="auto")
    parser.add_argument("--record-exit", dest="record_exit_ticker")
    args = parser.parse_args()

    now = now_jst()
    phase = determine_phase(now) if args.phase == "auto" else args.phase
    if phase == "final":
        phase = "trading"

    state = update_phase_state(phase)

    if args.record_exit_ticker:
        apply_exit_cooldown(args.record_exit_ticker, now)

    print("========================================")
    print("INTRADAY SCHEDULER")
    print("========================================")
    print(f"JST: {now:%Y-%m-%d %H:%M:%S}")
    print(f"phase: {phase}")
    print(f"paper trading start: {'YES' if can_start_paper_trading(now) else 'NO'}")
    print(f"trades today: {real_trades_today()}/{MAX_TRADES_PER_DAY}")


if __name__ == "__main__":
    main()
