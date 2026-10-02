#!/usr/bin/env python3
"""Day-trade separate paper-trading track: fixed net +5,000 / -8,000 JPY.

Independent of live (profit_top10_paper.py's TOP10/TOP1 track): same
candidate selection rule (reuses paper_fast_entrypoint.scan_progressive_
with_prefilter -> run_profit_loop.scan_candidates_fixed, which is itself
run_profit_loop.profit_priority()'s regime-aware ranking, so this reuses
the exact same disk scan cache written by paper_fast_entrypoint.py in the
same workflow tick -- no extra network calls for candidate selection), but
its own independent capital (fixed 1,000,000 JPY budget per trade, no
compounding, no shared pool with live), its own state/history files, and
its own exit rule: close each trade at an exact **net-of-fee** profit of
+5,000 JPY or loss of -8,000 JPY, instead of live's ATR-multiple TP/SL.

Trading rules (see run() / _try_entry() / _check_pending_fill() /
_evaluate_exit() for the exact mechanics):
  - one position at a time; a PENDING decision (see fill rule below)
    counts as the single slot too -- no other entry while pending; after
    an exit, the next tick's TOP1 is considered for re-entry (never
    re-enters within the same tick as an exit)
  - TOP1 = the live TOP1 rule: the same regime-filtered, profit_priority-
    ordered TOP10 that run_profit_loop.scan_candidates_fixed() returns,
    walked in order and skipping tickers under this track's own 30-minute
    same-ticker cooldown or whose price makes even a single 100-share lot
    exceed the budget
  - budget: up to 1,000,000 JPY, in 100-share lots (candidates needing
    more than one lot's worth of budget per share are skipped)
  - decisions only allowed 09:30-14:50 JST; forced exit at 15:20 JST
    regardless of price; no carry-over across days (a leftover position
    or pending decision found at the start of a new day -- should not
    happen given the 15:20 forced exit, but is a safety net for a
    skipped/failed tick -- is force-closed/cancelled immediately)
  - max 30 trades/day (a cancelled pending, see below, does not count)

TP/SL are solved exactly so the realized **net** P&L (after the live
round-trip fee rate, charged on both the entry and exit notional, exactly
as profit_top10_paper.mark_and_close computes it: pnl = gross -
(entry_price + exit_price) * shares * fee_rate) equals the target yen
amount for a given share count `sh`:
    BUY:   exit = (net/sh + entry_price*(1+fee)) / (1-fee)
    SHORT: exit = (entry_price*(1-fee) - net/sh) / (1+fee)
Because the target is a fixed yen amount rather than a fixed percentage,
the breakeven win rate is fixed (5,000 win / 8,000 loss -> breakeven win
rate = 8000/13000 = 61.5%) but the *percentage* TP/SL distance shrinks as
share count grows (i.e. varies with the stock's price and the resulting
lot count), and fixed 100-share lots structurally bias entries toward
lower-priced stocks (more lots fit in the 1,000,000 JPY budget).

=== Fill rule v2: decision_next_bar_open_v2 (current) ===
yfinance 5-minute data for Japan-listed tickers routinely arrives ~16
minutes (sometimes more) behind wall-clock. The original rule (v1, see
below) filled at wall-clock decision time using whatever bar happened to
be the latest *already available* one -- which, under that delay, is a
bar from *before* the decision, so TP/SL checks starting "after" it could
resolve at a bar timestamp earlier than the decision itself (observed in
production 2026-10-02: 5301.T decided/entered at 09:31 with
entry_price=09:15 close, closed TP at exit_time 09:20 -- before the
decision). That is unrealistic and flatters results, so it is now:

  1. A TOP1 decision at wall time T creates a PENDING entry (ticker,
     direction, rank/score/etc, decision_time=T, intended budget) instead
     of filling immediately. It holds the position slot; no other entry
     is considered while it is pending.
  2. Fill price = Open of the first 5-min bar whose START time is >= T
     (ceiling to the next 5-min boundary, or T itself if T already lands
     exactly on one) -- see _ceil_bar_time(). The fill is confirmed only
     once that bar actually appears in the downloaded data (on whatever
     later tick that happens to be, given the data delay). Shares are
     computed from the fill price within the 1,000,000 JPY budget in
     100-share lots (cancelled if even 100 shares exceeds budget), then
     TP/SL are solved from the fill price with the formulas above.
  3. Exit checks start WITH the fill bar itself (see _fill_bar_exit()):
     since the position enters exactly at that bar's open, only its
     High/Low *after* the open are checked (no gap-at-open rule on this
     bar) -- SL wins if both TP and SL are touched within it. From the
     next bar onward the existing gap-aware order below applies as-is.
  4. The pending entry is CANCELLED (not a trade -- logged to
     CANCELLED_LOG_FILE, never written to HISTORY_FILE) if: the fill bar
     would start at/after 15:20 JST, or the fill bar is still unavailable
     by wall-clock 15:20 JST, or a new trading day has started before it
     filled.
  5. Invariant, asserted in code and covered by tests: for every v2
     trade, exit_time >= fill_bar_time >= decision_time, and
     hold_minutes (>= 0) is computed from the fill time, not the
     decision time.
  6. Every v2 row carries fill_method='decision_next_bar_open_v2',
     decision_time, and fill_bar_time. MFE/MAE are measured from the
     fill price over bars from the fill bar onward.

Rows with no fill_method value (blank/NaN) are legacy v1 rows (see
below) written before this fix; past data is never rewritten.
_update_daily_summary() reports v1/v2 trade counts separately so legacy
rows stay visible rather than silently mixed into the v2 numbers.

=== Fill rule v1: stale_bar_close_v1 (legacy, pre-2026-10-02 fix) ===
Entry price was the latest available 5-minute bar's **close** at wall-
clock decision time (not the scan's own daily-bar-refined price, and not
wall-clock "now" itself); that bar's timestamp was recorded as
price_bar_time, with data_delay_minutes recording how far behind wall-
clock it was. Exit gap-fill checks started from the bar strictly AFTER
price_bar_time. This module keeps running this exact legacy path for any
position already open in state without a fill_method field at the time
of this fix (it will be closed under v1 rules; today's 6302.T is force-
closed at 15:20 regardless), but never creates new v1 positions.

Gap handling (5-minute bars, in this exact order -- SL wins when a single
bar's range touches both TP and SL):
    BUY:   open>=tp -> TP@open; elif open<=sl -> SL@open;
           elif low<=sl -> SL@sl; elif high>=tp -> TP@tp
    SHORT: open<=tp -> TP@open; elif open>=sl -> SL@open;
           elif high>=sl -> SL@sl; elif low<=tp -> TP@tp
For v1 positions this is checked from the bar AFTER price_bar_time
onward; for v2 positions the fill bar itself uses _fill_bar_exit()
instead (no gap-at-open rule on that one bar), then this same order from
the next bar on.

Policy choice: never calls futures_trend.detect_futures_trend() itself
(that would be a second, redundant network-bound trend judgement for the
same tick). Instead it reuses the live decision already made earlier in
this same tick by profit_top10_paper.select_policy_file() (which logs one
row per *day*, not per tick, to futures_trend_history.csv): if that file's
last row is dated today, its trend is reused directly; only if there is no
row for today yet does this module fall back to calling
profit_top10_paper.select_policy_file() itself. Which path was used is
recorded on every row (policy_source).

Every row is written with track='daytrade_tp5000_sl8000' and
validation_eligible=False: this track is a separate P&L experiment, not an
input to any OOS/walk-forward validation pipeline.

profit_top10_paper.py and run_profit_loop.py are imported read-only and
never modified; see the usual monkeypatch-revert note near the imports
below for why run_profit_loop's import-time side effects on
profit_top10_paper are undone immediately in this module too.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

import profit_top10_paper as live_p10
import run_profit_loop as live_loop

# run_profit_loop.py monkeypatches profit_top10_paper's scan/mark_and_close/
# open_positions/load_model as an import side effect (see run_profit_loop.py's
# own module docstring/bottom). This track must not change the live scan()'s
# behavior as a side effect of merely importing run_profit_loop for its
# profit_priority()/_passes_policy()/_as_aware_jst() helpers, so the
# monkeypatch is undone immediately (mirrors the same fix in
# all_candidates_paper.py). Order matters: this must happen before
# paper_fast_entrypoint is imported below, since paper_fast_entrypoint's own
# import-time code captures loop._original_scan/app.load_model as its bases.
live_p10.scan = live_loop._original_scan
live_p10.mark_and_close = live_loop._original_close
live_p10.open_positions = live_loop._original_open
live_p10.load_model = live_loop._original_load_model

import paper_fast_entrypoint as fast  # noqa: E402
import safe_state  # noqa: E402
from common import is_tse_trading_day  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")

TRACK = "daytrade_tp5000_sl8000"
STATE_FILE = "daytrade_tp5000_sl8000_state.json"
HISTORY_FILE = "daytrade_tp5000_sl8000_history.csv"
DAILY_FILE = "daytrade_tp5000_sl8000_daily.csv"
CANCELLED_LOG_FILE = "daytrade_tp5000_sl8000_cancelled.csv"

FILL_METHOD_V2 = "decision_next_bar_open_v2"
LEGACY_FILL_METHOD_LABEL = "stale_bar_close_v1"  # documentation only; never written

BUDGET_JPY = float(os.getenv("DAYTRADE_BUDGET_JPY", "1000000"))
LOT_SIZE = 100
TP_NET_JPY = 5000.0
SL_NET_JPY = -8000.0
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
COOLDOWN_MINUTES = 30
MAX_TRADES_PER_DAY = 30

ENTRY_WINDOW_START = dtime(9, 30)
ENTRY_WINDOW_END = dtime(14, 50)  # no NEW entries from this time onward
FORCED_EXIT_TIME = dtime(15, 20)
SESSION_START = dtime(9, 0)
SESSION_HARD_STOP = dtime(15, 35)

MODEL_FILE = "directional_model.pkl"
MODEL_META_FILE = "directional_model_meta.json"
LEGACY_MODEL_VERSION = "legacy-20260912"


def discord_send(message):
    """Never raises: a notify failure must never crash the trading tick."""
    label = "\U0001f4b4 【デイトレ別枠 +5000/−8000】"
    text = f"{label}\n{message}"
    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook:
        print(f"[daytrade] {text}")
        return False
    try:
        import requests
        r = requests.post(webhook, json={"content": text[:1950]}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"⚠️ daytrade Discord通知失敗(無視して続行): {exc}")
        return False


def default_state():
    return {"positions": [], "pending": None, "trade_date": None, "trades_today": 0,
            "last_exit_by_ticker": {}}


def load_state():
    s = default_state()
    loaded = safe_state.load_json_state(STATE_FILE, notify=discord_send, label=STATE_FILE, validate=lambda d: isinstance(d, dict))
    if loaded is not None:
        s.update(loaded)
    s.setdefault("positions", [])
    s.setdefault("pending", None)
    s.setdefault("last_exit_by_ticker", {})
    return s


def save_state(s):
    safe_state.atomic_write_json(STATE_FILE, s)


def reset_daily(s, today):
    if s.get("trade_date") != today:
        s.update({"trade_date": today, "trades_today": 0})


def append_history(row):
    row = dict(row)
    row["track"] = TRACK
    row["validation_eligible"] = False
    safe_state.safe_append_history(HISTORY_FILE, row, notify=discord_send, label=HISTORY_FILE)


def append_cancelled_log(row):
    """A cancelled PENDING decision is never a trade row -- it is logged
    here instead, separately from HISTORY_FILE."""
    safe_state.safe_append_history(CANCELLED_LOG_FILE, dict(row), notify=discord_send, label=CANCELLED_LOG_FILE)


def current_model_identity(work_dir="."):
    """model_id = sha256(directional_model.pkl bytes)[:16]; model_version =
    directional_model_meta.json's training_date if it exists and its
    model_id matches, else the legacy sentinel (mirrors
    all_candidates_paper.current_model_identity's convention, independently
    computed here since that module must not be imported from this one --
    two separate research tracks, no cross-dependency)."""
    path = os.path.join(work_dir, MODEL_FILE)
    if not os.path.exists(path):
        return None, LEGACY_MODEL_VERSION
    with open(path, "rb") as f:
        model_id = hashlib.sha256(f.read()).hexdigest()[:16]
    model_version = LEGACY_MODEL_VERSION
    meta_path = os.path.join(work_dir, MODEL_META_FILE)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and meta.get("model_id") == model_id and meta.get("training_date"):
                model_version = meta["training_date"]
        except Exception:
            pass
    return model_id, model_version


def choose_policy_file_reusing_live_tick(now=None, history_path="futures_trend_history.csv"):
    """Reuse the trend decision live's profit_top10_paper.select_policy_file()
    already logged for *today* (futures_trend_history.csv logs at most one
    row per calendar day, not per tick -- see futures_trend.log_daily_trend)
    instead of calling futures_trend.detect_futures_trend() a second time
    for the same tick. Falls back to calling select_policy_file() directly
    only when there is no row for today yet.

    Returns (policy_file, source) where source is 'reused_today_row' or
    'fallback_select_policy_file'.
    """
    now = now or datetime.now(TZ)
    today_str = now.strftime("%Y-%m-%d")
    try:
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            if not df.empty and str(df.iloc[-1].get("date")) == today_str:
                trend = str(df.iloc[-1].get("trend"))
                candidate = live_p10.POLICY_FILE_DOWN if trend == "down" else live_p10.POLICY_FILE_UP
                policy_file = candidate if os.path.exists(candidate) else live_p10.POLICY_FILE
                print(f"♻️ daytrade: 同一tickのlive判定を再利用: trend={trend} policy={policy_file}")
                return policy_file, "reused_today_row"
    except Exception as exc:
        print(f"⚠️ daytrade: futures_trend_history.csv読込失敗: {exc}")
    policy_file, _ = live_p10.select_policy_file()
    print(f"ℹ️ daytrade: 本日の行がないためselect_policy_file()を直接呼び出し: policy={policy_file}")
    return policy_file, "fallback_select_policy_file"


def _tp_sl_prices(entry_price, direction, shares):
    sh = max(1, int(shares))
    f = FEE_RATE
    if direction == "BUY":
        tp = (TP_NET_JPY / sh + entry_price * (1 + f)) / (1 - f)
        sl = (SL_NET_JPY / sh + entry_price * (1 + f)) / (1 - f)
    else:
        tp = (entry_price * (1 - f) - TP_NET_JPY / sh) / (1 + f)
        sl = (entry_price * (1 - f) - SL_NET_JPY / sh) / (1 + f)
    tp_pct = (tp - entry_price) / entry_price * 100.0
    sl_pct = (sl - entry_price) / entry_price * 100.0
    return tp, sl, tp_pct, sl_pct


def net_pnl(entry_price, exit_price, shares, direction):
    gross = (exit_price - entry_price) * shares if direction == "BUY" else (entry_price - exit_price) * shares
    return gross - (entry_price + exit_price) * shares * FEE_RATE


def _gap_fill_exit(direction, bar):
    """Returns (exit_price, reason) or (None, None). SL wins when a single
    bar's range touches both TP and SL (checked before TP in both branches
    below -- see module docstring for the exact rule)."""
    tp, sl = bar["_tp"], bar["_sl"]
    o, hi, lo = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
    if direction == "BUY":
        if o >= tp:
            return o, "TP"
        if o <= sl:
            return o, "SL"
        if lo <= sl:
            return sl, "SL"
        if hi >= tp:
            return tp, "TP"
    else:
        if o <= tp:
            return o, "TP"
        if o >= sl:
            return o, "SL"
        if hi >= sl:
            return sl, "SL"
        if lo <= tp:
            return tp, "TP"
    return None, None


def _fill_bar_exit(direction, bar):
    """Like _gap_fill_exit(), but for the v2 fill bar itself: the position
    enters exactly at this bar's Open, so there is no "gap at open" to
    check against it -- only the High/Low reached after that open matter.
    SL still wins when both TP and SL are touched within the bar."""
    tp, sl = bar["_tp"], bar["_sl"]
    hi, lo = float(bar["High"]), float(bar["Low"])
    if direction == "BUY":
        if lo <= sl:
            return sl, "SL"
        if hi >= tp:
            return tp, "TP"
    else:
        if hi >= sl:
            return sl, "SL"
        if lo <= tp:
            return tp, "TP"
    return None, None


def _ceil_bar_time(t):
    """The first 5-minute bar START time that is >= t: t itself if it
    already lands exactly on a 5-minute boundary, else the next one."""
    t = pd.Timestamp(t)
    floor = t.floor("5min")
    return floor if floor == t else floor + pd.Timedelta(minutes=5)


def _update_mfe_mae(pos, direction, hi, lo):
    ep, sh = float(pos["entry_price"]), int(pos["shares"])
    if direction == "BUY":
        fav, adv = (hi - ep) * sh, (lo - ep) * sh
    else:
        fav, adv = (ep - lo) * sh, (ep - hi) * sh
    pos["mfe_yen"] = max(float(pos.get("mfe_yen", 0.0)), fav)
    pos["mae_yen"] = min(float(pos.get("mae_yen", 0.0)), adv)


def _simple_atr_pct(bars, price, window=14):
    """ATR approximated from the already-fetched 5-minute bars (no extra
    daily download): mean high-low range of the last `window` 5-min bars,
    as a percentage of price. A coarse proxy, recorded for diagnostics only
    (not used by the TP/SL math)."""
    if bars is None or bars.empty or price <= 0:
        return None
    tail = bars.tail(window)
    tr = (tail["High"] - tail["Low"]).mean()
    return float(tr) / price * 100.0 if pd.notna(tr) else None


def _vwap_deviation_pct(bars, price, window=78):
    """Volume-weighted average price over the last `window` 5-min bars
    (~1 trading day), and price's deviation from it, as a percentage."""
    if bars is None or bars.empty or price <= 0:
        return None
    tail = bars.tail(window)
    vol = tail["Volume"].sum() if "Volume" in tail.columns else 0
    if not vol:
        return None
    vwap = float((tail["Close"] * tail["Volume"]).sum() / vol)
    if vwap <= 0:
        return None
    return (price - vwap) / vwap * 100.0


def _select_top1(top10, cooldowns, now, budget=BUDGET_JPY):
    """Walks the already regime-filtered, profit_priority-ordered TOP10
    (run_profit_loop.scan_candidates_fixed()'s output -- the exact same
    ranking run_profit_loop.open_top1_only() uses) and returns the first
    candidate that is both out of this track's own cooldown and affordable
    in at least one 100-share lot, or None.
    """
    for c in top10:
        ticker = str(c.get("ticker", "")).strip()
        if not ticker:
            continue
        raw = cooldowns.get(ticker)
        if raw:
            try:
                remaining = (live_loop._as_aware_jst(raw) + timedelta(minutes=COOLDOWN_MINUTES) - now).total_seconds()
            except Exception:
                remaining = 0.0
            if remaining > 0:
                continue
        price = float(c.get("price", 0) or 0)
        if price <= 0:
            continue
        if (int(budget // price) // LOT_SIZE) * LOT_SIZE <= 0:
            continue
        return c
    return None


def _try_entry(state, now, today):
    """Makes the TOP1 decision and records it as a PENDING entry (never
    fills immediately -- see the v2 fill rule in the module docstring).
    The pending entry holds the single position slot; _check_pending_fill()
    confirms the actual fill on whatever later tick the fill bar's data
    becomes available."""
    policy_file, policy_source = choose_policy_file_reusing_live_tick(now)
    policy = live_p10.load_policy(policy_file)
    top10, scanned = fast.scan_progressive_with_prefilter(policy)
    if not top10:
        print("⏸ daytrade: TOP10候補なし")
        return None

    chosen = _select_top1(top10, state.setdefault("last_exit_by_ticker", {}), now)
    if chosen is None:
        print("⏸ daytrade: cooldown/予算により新規エントリ可能な候補なし")
        return None

    ticker = str(chosen["ticker"]).strip()
    direction = str(chosen.get("direction", "BUY")).upper()
    model_id, model_version = current_model_identity()
    with open(policy_file, "rb") as f:
        policy_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    now_naive = now.replace(tzinfo=None)
    fill_bar_time = _ceil_bar_time(now_naive)

    pending = {
        "ticker": ticker, "direction": direction,
        "decision_date": today, "decision_time": now_naive.isoformat(),
        "fill_bar_time": fill_bar_time.isoformat(),
        "score": chosen.get("score"), "up_probability": chosen.get("up_probability"),
        "down_probability": chosen.get("down_probability"), "market_regime": chosen.get("market_regime"),
        "top10_rank": chosen.get("top10_rank"),
        "policy_file": policy_file, "policy_source": policy_source, "policy_hash": policy_hash,
        "model_id": model_id, "model_version": model_version,
        "budget": BUDGET_JPY,
    }
    state["pending"] = pending
    print(f"⏳ daytrade PENDING: {direction} {ticker} decision={now_naive} fill_bar_time>={fill_bar_time}")
    return (
        f"⏳ 判断(ペンディング)｜{ticker}｜{'買い' if direction == 'BUY' else '空売り'}\n"
        f"判断時刻 {now_naive.strftime('%H:%M:%S')}｜約定予定バー {fill_bar_time.strftime('%H:%M')}(Open約定)"
    )


def _cancel_pending(state, reason, now):
    """Cancels the current PENDING entry (never a trade row -- see
    append_cancelled_log()). No-op if there is no pending entry."""
    pending = state.get("pending")
    if pending is None:
        return None
    row = dict(pending)
    row["reason"] = reason
    row["cancelled_at"] = now.replace(tzinfo=None).isoformat()
    append_cancelled_log(row)
    state["pending"] = None
    print(f"\U0001f6ab daytrade CANCELLED_NO_FILL: {pending.get('ticker')} reason={reason}")
    return row


def _check_pending_fill(state, now, today):
    """Confirms the PENDING entry's fill once its fill bar (the first
    5-minute bar whose start is >= decision_time, per _ceil_bar_time())
    actually appears in the downloaded data; cancels it per the rules in
    the module docstring otherwise. Returns a list of zero or one
    notification message."""
    pending = state.get("pending")
    if pending is None:
        return []

    if pending.get("decision_date") != today:
        _cancel_pending(state, "CANCELLED_NO_FILL_NEW_DAY", now)
        return []

    decision_time = pd.Timestamp(pending["decision_time"])
    fill_bar_time = pd.Timestamp(pending["fill_bar_time"])
    if fill_bar_time.time() >= FORCED_EXIT_TIME:
        _cancel_pending(state, "CANCELLED_NO_FILL_TOO_LATE", now)
        return []

    ticker = pending["ticker"]
    bars = live_p10.download_5m(ticker)
    bar = None
    if bars is not None and not bars.empty and fill_bar_time in bars.index:
        bar = bars.loc[fill_bar_time]

    if bar is None:
        if now.time() >= FORCED_EXIT_TIME:
            _cancel_pending(state, "CANCELLED_NO_FILL_DATA_DELAY", now)
        return []

    fill_price = float(bar["Open"])
    budget = float(pending.get("budget", BUDGET_JPY))
    shares = (int(budget // fill_price) // LOT_SIZE) * LOT_SIZE
    if shares <= 0:
        _cancel_pending(state, "CANCELLED_NO_FILL_BUDGET", now)
        return []

    direction = pending["direction"]
    tp, sl, tp_pct, sl_pct = _tp_sl_prices(fill_price, direction, shares)
    now_naive = now.replace(tzinfo=None)
    data_delay_minutes = (now_naive - fill_bar_time).total_seconds() / 60.0

    position = {
        "ticker": ticker, "direction": direction, "entry_date": today,
        "entry_time": fill_bar_time.strftime("%H:%M"), "entry_datetime": fill_bar_time.isoformat(),
        "entry_price": fill_price, "shares": shares, "invested_amount": fill_price * shares,
        "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct,
        "score": pending.get("score"), "up_probability": pending.get("up_probability"),
        "down_probability": pending.get("down_probability"), "market_regime": pending.get("market_regime"),
        "top10_rank": pending.get("top10_rank"),
        "policy_file": pending.get("policy_file"), "policy_source": pending.get("policy_source"),
        "policy_hash": pending.get("policy_hash"),
        "model_id": pending.get("model_id"), "model_version": pending.get("model_version"),
        "price_bar_time": fill_bar_time.isoformat(), "data_delay_minutes": data_delay_minutes,
        "decision_time": decision_time.isoformat(), "fill_bar_time": fill_bar_time.isoformat(),
        "fill_method": FILL_METHOD_V2,
        "atr_pct": _simple_atr_pct(bars, fill_price), "vwap_deviation_pct": _vwap_deviation_pct(bars, fill_price),
        "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": fill_price,
    }
    assert pd.Timestamp(position["fill_bar_time"]) >= pd.Timestamp(position["decision_time"]), (
        "invariant violated: fill_bar_time < decision_time"
    )
    state["positions"] = [position]
    state["pending"] = None
    state["trades_today"] = int(state.get("trades_today", 0)) + 1
    print(f"\U0001f3c6 daytrade FILLED: {direction} {ticker} price={fill_price:.1f} shares={shares} "
          f"tp={tp:.1f} sl={sl:.1f} fill_bar_time={fill_bar_time} delay={data_delay_minutes:.1f}min")
    return [(
        f"\U0001f195 エントリー約定｜{ticker}｜{'買い' if direction == 'BUY' else '空売り'}\n"
        f"約定価格 {fill_price:,.1f}円｜{shares:,}株｜投資額 {fill_price*shares:,.0f}円\n"
        f"利確(+5000円) {tp:,.1f}｜損切(-8000円) {sl:,.1f}"
    )]


def _close_position(state, now, exit_price, reason, exit_ts):
    pos = state["positions"][0]
    ep, sh, direction = float(pos["entry_price"]), int(pos["shares"]), pos["direction"]
    pnl = net_pnl(ep, exit_price, sh, direction)
    invested = float(pos["invested_amount"])
    return_pct = pnl / invested * 100.0 if invested else 0.0
    entry_dt = pd.Timestamp(pos["entry_datetime"])
    exit_dt = pd.Timestamp(exit_ts) if exit_ts is not None else now.replace(tzinfo=None)
    hold_minutes = max(0.0, (exit_dt - entry_dt).total_seconds() / 60.0)
    mfe_yen, mae_yen = float(pos.get("mfe_yen", 0.0)), float(pos.get("mae_yen", 0.0))

    if pos.get("fill_method"):
        fill_bar_time = pd.Timestamp(pos["fill_bar_time"])
        decision_time = pd.Timestamp(pos["decision_time"])
        assert exit_dt >= fill_bar_time >= decision_time, (
            f"invariant violated: exit={exit_dt} fill_bar={fill_bar_time} decision={decision_time}"
        )

    row = {
        "entry_date": pos["entry_date"], "entry_time": pos["entry_time"], "entry_price": ep,
        "exit_date": exit_dt.strftime("%Y-%m-%d"), "exit_time": exit_dt.strftime("%H:%M"), "exit_price": exit_price,
        "ticker": pos["ticker"], "direction": direction, "shares": sh, "invested_amount": invested,
        "tp": pos["tp"], "sl": pos["sl"], "tp_pct": pos["tp_pct"], "sl_pct": pos["sl_pct"],
        "result": reason, "pnl": pnl, "return_pct": return_pct, "hold_minutes": hold_minutes,
        "mfe_yen": mfe_yen, "mae_yen": mae_yen,
        "mfe_pct": (mfe_yen / invested * 100.0) if invested else 0.0,
        "mae_pct": (mae_yen / invested * 100.0) if invested else 0.0,
        "atr_pct": pos.get("atr_pct"), "vwap_deviation_pct": pos.get("vwap_deviation_pct"),
        "score": pos.get("score"), "up_probability": pos.get("up_probability"),
        "down_probability": pos.get("down_probability"), "market_regime": pos.get("market_regime"),
        "top10_rank": pos.get("top10_rank"),
        "policy_file": pos.get("policy_file"), "policy_source": pos.get("policy_source"),
        "policy_hash": pos.get("policy_hash"),
        "model_id": pos.get("model_id"), "model_version": pos.get("model_version"),
        "price_bar_time": pos.get("price_bar_time"), "data_delay_minutes": pos.get("data_delay_minutes"),
        "fill_method": pos.get("fill_method"), "decision_time": pos.get("decision_time"),
        "fill_bar_time": pos.get("fill_bar_time"),
    }
    append_history(row)
    state.setdefault("last_exit_by_ticker", {})[pos["ticker"]] = pd.Timestamp(now).isoformat()
    state["positions"] = []
    emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
    return (
        f"{emoji} 決済｜{pos['ticker']}｜{reason}\n"
        f"決済価格 {exit_price:,.1f}円｜確定損益 {pnl:+,.0f}円｜ホールド {hold_minutes:.0f}分"
    )


def _evaluate_exit(state, now):
    """Dispatches to the v2 (fill_method set) or legacy v1 (no
    fill_method -- a position opened before this fix) exit path. See the
    module docstring for why the two differ."""
    pos = state["positions"][0]
    if pos.get("fill_method"):
        return _evaluate_exit_v2(state, now)
    return _evaluate_exit_legacy(state, now)


def _evaluate_exit_legacy(state, now):
    """v1 (stale_bar_close_v1): starts gap-fill checks from the bar AFTER
    price_bar_time (not from wall-clock now), per bar in order; forces an
    exit at the first bar whose timestamp is >= 15:20 JST if no TP/SL
    triggered first. If the 15:20 wall-clock deadline has already passed
    and no bar ever reached it (data delay), forces an exit immediately
    using the best available price rather than leaving the position open
    past the entry window. Kept only for a position already open in state
    without a fill_method field at the time of this fix.
    """
    pos = state["positions"][0]
    direction = pos["direction"]
    bars = live_p10.download_5m(pos["ticker"])
    exit_price = reason = exit_ts = None
    if bars is not None and not bars.empty:
        price_bar_time = pd.Timestamp(pos["price_bar_time"])
        subsequent = bars[bars.index > price_bar_time]
        for ts, b in subsequent.iterrows():
            _update_mfe_mae(pos, direction, float(b["High"]), float(b["Low"]))
            row = {"Open": b["Open"], "High": b["High"], "Low": b["Low"], "Close": b["Close"],
                   "_tp": pos["tp"], "_sl": pos["sl"]}
            x, r = _gap_fill_exit(direction, row)
            if x is not None:
                exit_price, reason, exit_ts = x, r, ts
                break
            if ts.time() >= FORCED_EXIT_TIME:
                exit_price, reason, exit_ts = float(b["Close"]), "FORCED_EXIT", ts
                break
        if exit_price is None and not subsequent.empty:
            pos["current_price"] = float(subsequent["Close"].iloc[-1])

    if exit_price is None and now.time() >= FORCED_EXIT_TIME:
        exit_price = float(pos.get("current_price", pos["entry_price"]))
        reason, exit_ts = "FORCED_EXIT", now.replace(tzinfo=None)

    if exit_price is None:
        return []
    return [_close_position(state, now, exit_price, reason, exit_ts)]


def _evaluate_exit_v2(state, now):
    """v2 (decision_next_bar_open_v2): checks start WITH the fill bar
    itself (index >= price_bar_time, which is set to fill_bar_time for a
    v2 position) using _fill_bar_exit() (no gap-at-open rule on that one
    bar, since entry is at its open); from the next bar on, the regular
    gap-aware _gap_fill_exit() applies. Forced-exit handling mirrors the
    legacy path."""
    pos = state["positions"][0]
    direction = pos["direction"]
    bars = live_p10.download_5m(pos["ticker"])
    exit_price = reason = exit_ts = None
    if bars is not None and not bars.empty:
        fill_bar_time = pd.Timestamp(pos["price_bar_time"])
        relevant = bars[bars.index >= fill_bar_time]
        for ts, b in relevant.iterrows():
            _update_mfe_mae(pos, direction, float(b["High"]), float(b["Low"]))
            row = {"Open": b["Open"], "High": b["High"], "Low": b["Low"], "Close": b["Close"],
                   "_tp": pos["tp"], "_sl": pos["sl"]}
            x, r = (_fill_bar_exit if ts == fill_bar_time else _gap_fill_exit)(direction, row)
            if x is not None:
                exit_price, reason, exit_ts = x, r, ts
                break
            if ts.time() >= FORCED_EXIT_TIME:
                exit_price, reason, exit_ts = float(b["Close"]), "FORCED_EXIT", ts
                break
        if exit_price is None and not relevant.empty:
            pos["current_price"] = float(relevant["Close"].iloc[-1])

    if exit_price is None and now.time() >= FORCED_EXIT_TIME:
        exit_price = float(pos.get("current_price", pos["entry_price"]))
        reason, exit_ts = "FORCED_EXIT", now.replace(tzinfo=None)

    if exit_price is None:
        return []
    return [_close_position(state, now, exit_price, reason, exit_ts)]


def _force_close_late(state, now):
    """Day-boundary safety net: a position whose entry_date isn't today
    (should not happen given the 15:20 forced exit, but a skipped/failed
    tick could leave one open overnight) is closed immediately using the
    best available price -- never carried forward another day."""
    pos = state["positions"][0]
    bars = live_p10.download_5m(pos["ticker"])
    price = float(pos.get("current_price", pos["entry_price"]))
    if bars is not None and not bars.empty:
        price = float(bars["Close"].iloc[-1])
    msg = _close_position(state, now, price, "FORCED_LATE", now.replace(tzinfo=None))
    discord_send(f"⚠️ 前日からの持ち越しポジションを強制決済しました\n{msg}")


def _update_daily_summary(today):
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        hist = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        print(f"⚠️ daytrade daily集計: history読込失敗: {exc}")
        return
    todays = hist[hist.get("exit_date") == today] if "exit_date" in hist.columns else hist.iloc[0:0]
    trades = len(todays)
    wins = int((pd.to_numeric(todays.get("pnl"), errors="coerce") > 0).sum()) if trades else 0
    net_pnl_total = float(pd.to_numeric(todays.get("pnl"), errors="coerce").sum()) if trades else 0.0
    if trades and "fill_method" in todays.columns:
        is_v2 = todays["fill_method"].astype(str) == FILL_METHOD_V2
        v2_trades = int(is_v2.sum())
        v1_legacy_trades = trades - v2_trades
    else:
        v2_trades, v1_legacy_trades = 0, trades
    row = {
        "date": today, "trades": trades, "wins": wins, "losses": trades - wins,
        "net_pnl": net_pnl_total, "win_rate_pct": (wins / trades * 100.0) if trades else None,
        "v2_trades": v2_trades, "v1_legacy_trades": v1_legacy_trades,
    }
    df = pd.DataFrame([row])
    if os.path.exists(DAILY_FILE):
        try:
            existing = pd.read_csv(DAILY_FILE)
            existing = existing[existing["date"] != today]
            df = pd.concat([existing, df], ignore_index=True).sort_values("date")
        except Exception as exc:
            print(f"⚠️ daytrade daily集計: 既存ファイル読込失敗(今日分のみで上書き): {exc}")
    df.to_csv(DAILY_FILE, index=False, encoding="utf-8-sig")


def run(now=None):
    now = now or datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")
    state = load_state()
    reset_daily(state, today)

    if state["positions"] and state["positions"][0].get("entry_date") != today:
        leftover_date = state["positions"][0]["entry_date"]
        _force_close_late(state, now)
        save_state(state)
        _update_daily_summary(leftover_date)

    if state.get("pending") and state["pending"].get("decision_date") != today:
        _cancel_pending(state, "CANCELLED_NO_FILL_NEW_DAY", now)
        save_state(state)

    if not (now.weekday() < 5 and is_tse_trading_day(now.date())):
        print("\U0001f4a4 daytrade: 休場日のためスキップ")
        return
    if not (SESSION_START <= now.time() <= SESSION_HARD_STOP):
        print("\U0001f4a4 daytrade: 市場時間外のためスキップ")
        return

    messages = []
    exited_this_tick = False

    if state.get("pending"):
        fill_msgs = _check_pending_fill(state, now, today)
        messages += fill_msgs
        save_state(state)

    if state["positions"]:
        exit_msgs = _evaluate_exit(state, now)
        if exit_msgs:
            exited_this_tick = True
        messages += exit_msgs
        save_state(state)
        if exit_msgs:
            _update_daily_summary(today)

    if (
        not exited_this_tick and not state["positions"] and not state.get("pending")
        and ENTRY_WINDOW_START <= now.time() < ENTRY_WINDOW_END
        and int(state.get("trades_today", 0)) < MAX_TRADES_PER_DAY
    ):
        opened_msg = _try_entry(state, now, today)
        if opened_msg:
            messages.append(opened_msg)
        save_state(state)

    for m in messages:
        discord_send(m)


def main():
    try:
        run()
    except Exception as exc:
        discord_send(f"\U0001f6a8 daytrade失敗: {exc}")
        raise


if __name__ == "__main__":
    main()
