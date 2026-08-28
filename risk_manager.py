"""AI株システム用リスク管理モジュール。

stock_scan.py から import される正式なモジュール名。
実注文は一切行わず、ペーパートレードの資金・ポジションだけを管理する。
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

POLICY_FILE = "strategy_policy.json"
RISK_STATE_FILE = "risk_state.json"
PREDICTION_HISTORY_FILE = "prediction_history.csv"

INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
RISK_PER_TRADE = float(os.getenv("AI_RISK_PER_TRADE", "0.005"))
MAX_POSITIONS = int(os.getenv("AI_MAX_POSITIONS", "20"))
DAILY_STOP_LOSS = float(os.getenv("AI_DAILY_STOP_LOSS", "0.02"))
MAX_DRAWDOWN = float(os.getenv("AI_MAX_DRAWDOWN", "0.30"))
MIN_LIVE_PROFIT_FACTOR = float(os.getenv("AI_MIN_LIVE_PF", "0.90"))
MIN_LIVE_TRADES_FOR_ALERT = int(os.getenv("AI_MIN_LIVE_TRADES", "30"))
RECENT_TRADE_WINDOWS = [30, 60]
MIN_RECENT_TRADES_FOR_ALERT = int(os.getenv("AI_MIN_RECENT_TRADES", "30"))
MIN_RECENT_PROFIT_FACTOR = float(os.getenv("AI_MIN_RECENT_PF", "0.90"))
TRADING_FEE_RATE = float(os.getenv("AI_TRADING_FEE_RATE", "0.001"))
SLIPPAGE_RATE = float(os.getenv("AI_SLIPPAGE_RATE", "0.0005"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("AI_MAX_CONSECUTIVE_LOSSES", "3"))
MONTHLY_TARGET = float(os.getenv("AI_MONTHLY_TARGET", "0.05"))
LOCK_RISK_AFTER_TARGET = os.getenv("AI_LOCK_RISK_AFTER_TARGET", "true").lower() == "true"
LOCKED_RISK_MULTIPLIER = float(os.getenv("AI_LOCKED_RISK_MULTIPLIER", "0.5"))
CONSECUTIVE_NEGATIVE_MONTH_STOP = int(os.getenv("AI_CONSECUTIVE_NEGATIVE_MONTH_STOP", "2"))

DEFAULT_POLICY = {
    "status": "DEFAULT",
    "up_threshold": 50,
    "min_score_for_buy": 60,
    "nikkei_filter": False,
    "atr_tp_multiplier": 3.0,
    "atr_sl_multiplier": 1.5,
    "hold_days": 5,
}


def now_jst():
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def load_policy():
    if not os.path.exists(POLICY_FILE):
        return DEFAULT_POLICY.copy()
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            policy = json.load(f)
        if not isinstance(policy, dict):
            return DEFAULT_POLICY.copy()
        result = DEFAULT_POLICY.copy()
        result.update(policy)
        return result
    except Exception as e:
        print(f"⚠ policy読み込み失敗: {e}")
        return DEFAULT_POLICY.copy()


def default_risk_state():
    current = now_jst()
    return {
        "date": current.strftime("%Y-%m-%d"),
        "month": current.strftime("%Y-%m"),
        "capital": INITIAL_CAPITAL,
        "peak_capital": INITIAL_CAPITAL,
        "day_start_capital": INITIAL_CAPITAL,
        "month_start_capital": INITIAL_CAPITAL,
        "daily_pnl": 0.0,
        "realized_pnl": 0.0,
        "open_positions": 0,
        "positions": {},
        "consecutive_losses": 0,
        "consecutive_negative_months": 0,
        "monthly_stop": False,
        "risk_locked": False,
        "trading_enabled": True,
        "stop_reason": "",
        "last_update": current.isoformat(),
    }


def save_risk_state(state):
    state["last_update"] = now_jst().isoformat()
    with open(RISK_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _completed_negative_month_streak():
    """確定済みの買い推奨取引だけから連続マイナス月を計算する。

    当月は途中経過なので自動停止判定には含めない。
    result が確定している行だけを対象にし、監視シグナルは対象外。
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return 0
    try:
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
        if df.empty or "date" not in df.columns or "result" not in df.columns or "return" not in df.columns:
            return 0
        if "category" in df.columns:
            df = df[df["category"].astype(str).str.lower().eq("buy")]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["return"] = pd.to_numeric(df["return"], errors="coerce")
        df["result"] = df["result"].astype(str).str.strip()
        df = df[df["date"].notna() & df["return"].notna() & df["result"].ne("")]
        if df.empty:
            return 0
        current_month = now_jst().strftime("%Y-%m")
        df["month"] = df["date"].dt.strftime("%Y-%m")
        df = df[df["month"] < current_month]
        if df.empty:
            return 0
        monthly = df.groupby("month")["return"].sum().sort_index()
        streak = 0
        for value in reversed(monthly.tolist()):
            if float(value) < 0:
                streak += 1
            else:
                break
        return streak
    except Exception as e:
        print(f"⚠ 月間連続マイナス判定失敗: {e}")
        return 0


def _sync_state(state):
    today = now_jst().strftime("%Y-%m-%d")
    month = now_jst().strftime("%Y-%m")
    if state.get("date") != today:
        state["date"] = today
        state["day_start_capital"] = float(state.get("capital", INITIAL_CAPITAL))
        state["daily_pnl"] = 0.0
        state["consecutive_losses"] = 0
        state["trading_enabled"] = not bool(state.get("monthly_stop", False))
        state["stop_reason"] = "月間連続マイナス停止" if state.get("monthly_stop", False) else ""
    if state.get("month") != month:
        state["month"] = month
        state["month_start_capital"] = float(state.get("capital", INITIAL_CAPITAL))
        state["risk_locked"] = False
    if not isinstance(state.get("positions"), dict):
        state["positions"] = {}
    state["open_positions"] = len(state["positions"])
    state["consecutive_losses"] = int(state.get("consecutive_losses", 0))
    state["consecutive_negative_months"] = int(state.get("consecutive_negative_months", 0))
    state["monthly_stop"] = bool(state.get("monthly_stop", False))
    return state


def load_risk_state():
    if not os.path.exists(RISK_STATE_FILE):
        state = default_risk_state()
        save_risk_state(state)
        return state
    try:
        with open(RISK_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("risk stateがdictではありません")
        defaults = default_risk_state()
        defaults.update(state)
        return _sync_state(defaults)
    except Exception as e:
        print(f"⚠ risk state読み込み失敗: {e}")
        return default_risk_state()


def get_available_cash(state=None):
    state = _sync_state(state or load_risk_state())
    invested = sum(float(p.get("value", 0.0)) for p in state["positions"].values())
    return max(0.0, float(state.get("capital", INITIAL_CAPITAL)) - invested)


def current_risk_per_trade(state=None):
    state = state or load_risk_state()
    if state.get("risk_locked", False):
        return RISK_PER_TRADE * LOCKED_RISK_MULTIPLIER
    return RISK_PER_TRADE


def max_loss_amount(capital, state=None):
    return float(capital) * current_risk_per_trade(state)


def current_drawdown(capital=None):
    state = _sync_state(load_risk_state())
    capital = float(state.get("capital", INITIAL_CAPITAL) if capital is None else capital)
    peak = float(state.get("peak_capital", capital))
    return 0.0 if peak <= 0 else capital / peak - 1.0


def calculate_position_size(capital, entry_price, stop_loss, open_value=0.0, state=None):
    capital = float(capital)
    entry_price = float(entry_price)
    stop_loss = float(stop_loss)
    if capital <= 0 or entry_price <= 0 or stop_loss >= entry_price:
        return 0
    per_share_risk = entry_price - stop_loss
    risk_budget = max_loss_amount(capital, state)
    shares_by_risk = int(risk_budget / per_share_risk)
    available_cash = max(0.0, capital - float(open_value))
    cost_per_share = entry_price * (1.0 + TRADING_FEE_RATE + SLIPPAGE_RATE)
    shares_by_cash = int(available_cash / cost_per_share) if cost_per_share > 0 else 0
    return max(0, min(shares_by_risk, shares_by_cash))


def build_position_plan(capital, ticker, entry_price, take_profit, stop_loss):
    state = _sync_state(load_risk_state())
    if len(state["positions"]) >= MAX_POSITIONS:
        return {
            "ticker": ticker,
            "shares": 0,
            "entry_price": float(entry_price),
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "position_value": 0.0,
            "max_loss": 0.0,
        }
    open_value = sum(float(p.get("value", 0.0)) for p in state["positions"].values())
    shares = calculate_position_size(capital, entry_price, stop_loss, open_value, state)
    position_value = shares * float(entry_price)
    max_loss = shares * max(0.0, float(entry_price) - float(stop_loss))
    return {
        "ticker": ticker,
        "shares": shares,
        "entry_price": float(entry_price),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "position_value": position_value,
        "max_loss": max_loss,
    }


def risk_check():
    """トレード可否を毎回フラットに再評価する。

    ★修正点(2026-08):
    以前は enabled の初期値を前回保存された trading_enabled から
    引き継いでいたため、register_position_open() が「同時保有数上限」で
    一度 trading_enabled=False を保存すると、ポジションが減っても
    同日中はずっと停止したままになるバグがあった。
    monthly_stop・daily_stop・DD・連敗・同時保有数はすべて現在の
    state から都度再計算できる条件なので、enabled は毎回 True から
    再評価すれば十分で、継続性が必要な monthly_stop は別フィールドで
    管理されている。
    """
    state = _sync_state(load_risk_state())
    capital = float(state.get("capital", INITIAL_CAPITAL))
    reason = ""
    enabled = True  # ← 前回値を引き継がず、毎回フラットに再評価する
    daily_start = float(state.get("day_start_capital", capital))
    daily_pnl = capital - daily_start
    state["daily_pnl"] = daily_pnl

    negative_streak = _completed_negative_month_streak()
    state["consecutive_negative_months"] = negative_streak
    if negative_streak >= CONSECUTIVE_NEGATIVE_MONTH_STOP:
        state["monthly_stop"] = True
        enabled = False
        reason = f"{negative_streak}ヶ月連続マイナス"

    if daily_start > 0 and daily_pnl / daily_start <= -DAILY_STOP_LOSS:
        enabled = False
        reason = "日次損失上限"
    peak = float(state.get("peak_capital", capital))
    if peak > 0 and capital / peak - 1.0 <= -MAX_DRAWDOWN:
        enabled = False
        reason = "最大DD"
    if int(state.get("consecutive_losses", 0)) >= MAX_CONSECUTIVE_LOSSES:
        enabled = False
        reason = "連敗ブレーカー"
    if len(state.get("positions", {})) >= MAX_POSITIONS:
        enabled = False
        reason = "同時保有数上限"
    month_start = float(state.get("month_start_capital", capital))
    if LOCK_RISK_AFTER_TARGET and month_start > 0 and capital / month_start - 1.0 >= MONTHLY_TARGET:
        state["risk_locked"] = True

    state["trading_enabled"] = enabled
    state["stop_reason"] = reason
    state["open_positions"] = len(state.get("positions", {}))
    state["consecutive_losses"] = int(state.get("consecutive_losses", 0))
    save_risk_state(state)
    return {
        "trading_enabled": enabled,
        "reason": reason,
        "capital": capital,
        "open_positions": state["open_positions"],
        "available_cash": get_available_cash(state),
        "drawdown": current_drawdown(capital),
        "risk_per_trade": current_risk_per_trade(state),
        "consecutive_losses": state["consecutive_losses"],
        "consecutive_negative_months": state["consecutive_negative_months"],
        "monthly_stop": state["monthly_stop"],
        "max_positions": MAX_POSITIONS,
    }


def register_position_open(ticker, shares, entry_price):
    state = _sync_state(load_risk_state())

    # ★追加(2026-08): monthly_stop はここでも直接確認する。
    # risk_check() を呼び忘れた将来のコード変更があっても、
    # 月間連続マイナス停止中は新規ポジションを登録させない
    # 最後の安全装置として機能させる。
    if state.get("monthly_stop", False):
        print(f"🛑 月間連続マイナス停止中のため新規ポジション登録不可: {ticker}")
        return state

    if not state.get("trading_enabled", True):
        print(f"🛑 取引停止中のため新規ポジション登録不可: {ticker}")
        return state

    positions = state["positions"]
    if ticker in positions:
        print(f"⚠ 既に保有中: {ticker}")
        return state

    if len(positions) >= MAX_POSITIONS:
        # ★修正(2026-08): ここで trading_enabled=False を保存しない。
        # 同時保有数上限は risk_check() が毎回 state から
        # 再評価する条件なので、この一件を見送るだけでよい。
        # 以前はここで trading_enabled=False を永続化していたため、
        # 決済でポジションが減っても同日中は新規エントリーが
        # 復活しないバグになっていた。
        print(f"⚠ 同時保有数上限のため見送り: {ticker}")
        return state

    shares = int(shares)
    entry_price = float(entry_price)
    if shares <= 0 or entry_price <= 0:
        return state
    value = shares * entry_price
    available = get_available_cash(state)
    required = value * (1.0 + TRADING_FEE_RATE + SLIPPAGE_RATE)
    if required > available:
        print(f"🛑 資金不足: {ticker}")
        return state
    positions[ticker] = {
        "shares": shares,
        "entry_price": entry_price,
        "value": value,
        "entry_fee": value * TRADING_FEE_RATE,
        "entry_slippage": value * SLIPPAGE_RATE,
        "entry_date": now_jst().strftime("%Y-%m-%d"),
    }
    state["open_positions"] = len(positions)
    save_risk_state(state)
    return state


def register_position_close(ticker, exit_price, new_capital=None):
    state = _sync_state(load_risk_state())
    position = state["positions"].pop(ticker, None)
    if position is None:
        save_risk_state(state)
        return state
    shares = int(position.get("shares", 0))
    entry = float(position.get("entry_price", 0.0))
    exit_price = float(exit_price)
    gross_pnl = shares * (exit_price - entry)
    total_cost = shares * (entry + exit_price) * TRADING_FEE_RATE
    total_cost += shares * (entry + exit_price) * SLIPPAGE_RATE
    pnl = gross_pnl - total_cost
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    state["capital"] = float(new_capital) if new_capital is not None else float(state.get("capital", INITIAL_CAPITAL)) + pnl
    state["peak_capital"] = max(float(state.get("peak_capital", state["capital"])), state["capital"])
    state["daily_pnl"] = state["capital"] - float(state.get("day_start_capital", state["capital"]))
    state["consecutive_losses"] = 0 if pnl > 0 else int(state.get("consecutive_losses", 0)) + 1
    state["open_positions"] = len(state["positions"])
    save_risk_state(state)
    return state


def risk_status_text():
    check = risk_check()
    status = "🟢 取引可能" if check["trading_enabled"] else "🛑 取引停止"
    return (
        f"🛡️ リスク管理: {status}\n"
        f"資金: {check['capital']:,.0f}円\n"
        f"利用可能資金: {check['available_cash']:,.0f}円\n"
        f"保有: {check['open_positions']}/{MAX_POSITIONS}\n"
        f"DD: {check['drawdown'] * 100:.2f}%\n"
        f"1回リスク: {check['risk_per_trade'] * 100:.2f}%\n"
        f"連敗: {check['consecutive_losses']}\n"
        f"連続マイナス月: {check['consecutive_negative_months']}ヶ月"
        + (f"\n停止理由: {check['reason']}" if check["reason"] else "")
    )
