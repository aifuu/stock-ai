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

Trading rules (see run() / _try_entry() / _evaluate_exit() for the exact
mechanics):
  - one position at a time; after an exit, the next tick's TOP1 is
    considered for re-entry (never re-enters within the same tick as an
    exit)
  - TOP1 = the live TOP1 rule: the same regime-filtered, profit_priority-
    ordered TOP10 that run_profit_loop.scan_candidates_fixed() returns,
    walked in order and skipping tickers under this track's own 30-minute
    same-ticker cooldown or whose price makes even a single 100-share lot
    exceed the budget
  - budget: up to 1,000,000 JPY, in 100-share lots (candidates needing
    more than one lot's worth of budget per share are skipped)
  - no new entries before 09:30 JST or from 14:50 JST; forced exit at
    15:20 JST regardless of price; no carry-over across days (a leftover
    position found open at the start of a new day -- should not happen
    given the 15:20 forced exit, but is a safety net for a skipped/failed
    tick -- is force-closed immediately as FORCED_LATE using the best
    available price)
  - max 30 trades/day

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

Gap handling (5-minute bars, checked from the bar AFTER the entry's
price_bar_time onward, in this exact order -- SL wins when a single bar's
range touches both TP and SL):
    BUY:   open>=tp -> TP@open; elif open<=sl -> SL@open;
           elif low<=sl -> SL@sl; elif high>=tp -> TP@tp
    SHORT: open<=tp -> TP@open; elif open>=sl -> SL@open;
           elif high>=sl -> SL@sl; elif low<=tp -> TP@tp

Entry price is the latest available 5-minute bar's **close** (not the
scan's own daily-bar-refined price, and not wall-clock "now"); that bar's
timestamp is recorded as price_bar_time, and data_delay_minutes records
how far behind wall-clock that bar was (yfinance 5-minute data for Japan-
listed tickers is commonly a few minutes behind real time, sometimes
more around session open/close or on thin names).

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
    return {"positions": [], "trade_date": None, "trades_today": 0, "last_exit_by_ticker": {}}


def load_state():
    s = default_state()
    loaded = safe_state.load_json_state(STATE_FILE, notify=discord_send, label=STATE_FILE, validate=lambda d: isinstance(d, dict))
    if loaded is not None:
        s.update(loaded)
    s.setdefault("positions", [])
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
    bars = live_p10.download_5m(ticker)
    if bars is None or bars.empty:
        print(f"⏸ daytrade: {ticker} の5分足が取得できないためこのtickは見送り")
        return None

    price_bar_time = pd.Timestamp(bars.index[-1])
    entry_price = float(bars["Close"].iloc[-1])
    shares = (int(BUDGET_JPY // entry_price) // LOT_SIZE) * LOT_SIZE
    if shares <= 0:
        print(f"⏸ daytrade: {ticker} entry_price={entry_price:.1f}で100株分の予算を確保できず見送り")
        return None

    tp, sl, tp_pct, sl_pct = _tp_sl_prices(entry_price, direction, shares)
    model_id, model_version = current_model_identity()
    with open(policy_file, "rb") as f:
        policy_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    now_naive = now.replace(tzinfo=None)
    data_delay_minutes = (now_naive - price_bar_time).total_seconds() / 60.0

    position = {
        "ticker": ticker, "direction": direction, "entry_date": today,
        "entry_time": now.strftime("%H:%M"), "entry_datetime": now_naive.isoformat(),
        "entry_price": entry_price, "shares": shares, "invested_amount": entry_price * shares,
        "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct,
        "score": chosen.get("score"), "up_probability": chosen.get("up_probability"),
        "down_probability": chosen.get("down_probability"), "market_regime": chosen.get("market_regime"),
        "top10_rank": chosen.get("top10_rank"),
        "policy_file": policy_file, "policy_source": policy_source, "policy_hash": policy_hash,
        "model_id": model_id, "model_version": model_version,
        "price_bar_time": price_bar_time.isoformat(), "data_delay_minutes": data_delay_minutes,
        "atr_pct": _simple_atr_pct(bars, entry_price), "vwap_deviation_pct": _vwap_deviation_pct(bars, entry_price),
        "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": entry_price,
    }
    state["positions"] = [position]
    state["trades_today"] = int(state.get("trades_today", 0)) + 1
    print(f"\U0001f3c6 daytrade ENTRY: {direction} {ticker} price={entry_price:.1f} shares={shares} tp={tp:.1f} sl={sl:.1f} bar_time={price_bar_time} delay={data_delay_minutes:.1f}min")
    return (
        f"\U0001f195 エントリー｜{ticker}｜{'買い' if direction == 'BUY' else '空売り'}\n"
        f"取得価格 {entry_price:,.1f}円｜{shares:,}株｜投資額 {entry_price*shares:,.0f}円\n"
        f"利確(+5000円) {tp:,.1f}｜損切(-8000円) {sl:,.1f}"
    )


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
    """Starts gap-fill checks from the bar AFTER price_bar_time (not from
    wall-clock now), per bar in order; forces an exit at the first bar whose
    timestamp is >= 15:20 JST if no TP/SL triggered first. If the 15:20
    wall-clock deadline has already passed and no bar ever reached it
    (data delay), forces an exit immediately using the best available
    price rather than leaving the position open past the entry window.
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
    row = {
        "date": today, "trades": trades, "wins": wins, "losses": trades - wins,
        "net_pnl": net_pnl_total, "win_rate_pct": (wins / trades * 100.0) if trades else None,
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

    if not (now.weekday() < 5 and is_tse_trading_day(now.date())):
        print("\U0001f4a4 daytrade: 休場日のためスキップ")
        return
    if not (SESSION_START <= now.time() <= SESSION_HARD_STOP):
        print("\U0001f4a4 daytrade: 市場時間外のためスキップ")
        return

    messages = []
    exited_this_tick = False
    if state["positions"]:
        exit_msgs = _evaluate_exit(state, now)
        if exit_msgs:
            exited_this_tick = True
        messages += exit_msgs
        save_state(state)
        if exit_msgs:
            _update_daily_summary(today)

    if (
        not exited_this_tick and not state["positions"]
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
