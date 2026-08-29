import json
from datetime import datetime
from zoneinfo import ZoneInfo
import daily_directional_top1 as trader

TZ = ZoneInfo("Asia/Tokyo")
STATE_FILE = trader.STATE_FILE
MAX_TRADES_PER_DAY = 10


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"capital": trader.INITIAL_CAPITAL, "position": None, "peak": trader.INITIAL_CAPITAL, "max_dd": 0.0}


def main():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    state = load_state()
    if state.get("trade_count_date") != today:
        state["trade_count_date"] = today
        state["trades_today"] = 0
        trader.save_state(state)

    trades_today = int(state.get("trades_today", 0))
    if trades_today >= MAX_TRADES_PER_DAY:
        trader.send(f"🛑 DAILY TOP1｜本日の新規売買上限 {MAX_TRADES_PER_DAY}回に到達。以降は停止します。")
        return

    had_position = bool(state.get("position"))
    trader.main()

    after = load_state()
    has_position = bool(after.get("position"))
    # 新規ポジションが発生した回だけ当日売買回数を1加算。
    if not had_position and has_position:
        after["trade_count_date"] = today
        after["trades_today"] = trades_today + 1
        trader.save_state(after)
        trader.send(f"📊 本日新規売買: {after['trades_today']}/{MAX_TRADES_PER_DAY}回")


if __name__ == "__main__":
    main()
