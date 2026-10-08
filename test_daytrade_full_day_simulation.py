"""Full simulated trading day for daytrade_tp5000_sl8000_paper.py, driven
through the same 5-minute tick sequence the ai-stock-scan.yml workflow
uses (a live tick, then the daytrade tick, every 5 minutes from 09:00 to
15:35 JST), with fixture data including a ~16-minute data delay (the real
cause of the 2026-10-02 bug this module's v2 fill rule fixes -- see
daytrade_tp5000_sl8000_paper.py's module docstring), a gap, and a
cooldown-driven candidate switch, proving:

  1. live's own state/history files are byte-identical whether or not the
     daytrade step runs (the "live tick" is stood in for by a small
     deterministic stub that writes to profit_top10_paper.STATE_FILE/
     HISTORY_FILE -- real content depends only on the tick number, never
     on daytrade, so any difference between the two simulated days would
     mean daytrade touched live's files)
  2. daytrade's own per-tick wall-clock cost stays far under the
     workflow's `timeout 90` budget (this is a mocked harness -- it
     proves there is no runaway loop or O(day) blow-up in the tick logic
     itself, not real yfinance network latency, which no offline test can
     simulate)
  3. the forced exit at 15:20 JST actually fires when a position rides
     unrealized the whole afternoon
  4. no trade in the simulated day violates the fill invariant
     (exit_time >= fill_bar_time >= decision_time)

This module makes real network calls if a test forgets to mock the right
entry point, so requests.post and yfinance.download are stubbed to raise
before anything is imported (same pattern as the other daytrade/
run_profit_loop test files).
"""
import json
import os
import tempfile
import time
import unittest
from datetime import timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402


def _blocked_post(*_a, **_k):
    raise AssertionError("network call attempted via requests.post in tests")


requests.post = _blocked_post

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("network call attempted via yfinance.download in tests")
)

import pandas as pd  # noqa: E402

import daytrade_tp5000_sl8000_paper as dt  # noqa: E402
import profit_top10_paper as live_p10  # noqa: E402
import run_profit_loop as live_loop  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")
DAY = "2026-09-25"
POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False,
          "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0, "hold_days": 1, "status": "PENDING"}

# The real-world delay this harness models: yfinance 5-minute bars for
# Japan-listed tickers routinely arrive this far behind wall-clock (see
# the 2026-10-02 production bug in the module docstring). fake_download_5m
# below only ever reveals bars whose start time is <= now - DATA_DELAY, so
# the fill bar for a decision at time T never appears before T + DATA_DELAY.
DATA_DELAY = timedelta(minutes=16)

CANDS = [
    {"ticker": "7203.T", "direction": "BUY", "price": 3005.0, "score": 80.0,
     "up_probability": 60.0, "down_probability": 10.0, "top10_rank": 1, "market_regime": "neutral"},
    {"ticker": "9984.T", "direction": "BUY", "price": 5000.0, "score": 70.0,
     "up_probability": 55.0, "down_probability": 15.0, "top10_rank": 2, "market_regime": "neutral"},
]


def _tick_range():
    idx = pd.date_range(f"{DAY} 09:00", f"{DAY} 15:35", freq="5min")
    return [ts.to_pydatetime().replace(tzinfo=TZ) for ts in idx]


def _build_day_bars():
    idx = pd.date_range(f"{DAY} 09:00", f"{DAY} 15:35", freq="5min")

    # 7203.T: flat near 3005 until an 11:00 gap-up opens straight through TP.
    close = [3005.0] * len(idx)
    bars_7203 = pd.DataFrame(
        {"Open": close, "High": [c + 2 for c in close], "Low": [c - 2 for c in close],
         "Close": close, "Volume": [1000] * len(idx)},
        index=idx,
    )
    gap_ts = pd.Timestamp(f"{DAY} 11:00")
    bars_7203.loc[gap_ts, ["Open", "High", "Low", "Close"]] = [3050.0, 3055.0, 3049.0, 3052.0]

    # 9984.T: flat at 5000 all day -- never touches its own TP/SL, rides to
    # the 15:20 forced exit.
    close9 = [5000.0] * len(idx)
    bars_9984 = pd.DataFrame(
        {"Open": close9, "High": [c + 1 for c in close9], "Low": [c - 1 for c in close9],
         "Close": close9, "Volume": [1000] * len(idx)},
        index=idx,
    )
    return {"7203.T": bars_7203, "9984.T": bars_9984}


def _simulate_live_tick(tick_no, now):
    """Stand-in for the real live tick (paper_fast_entrypoint.py calling
    into profit_top10_paper.py). Content depends only on tick_no, never on
    whether daytrade ran, so the two simulated days are directly
    comparable byte-for-byte.

    Also mirrors the real ai-stock-scan.yml loop's
    `rm -f scan_candidates_cache.json` (at the start of the iteration) then
    the live tick writing a fresh one -- Change A has daytrade read this
    file directly (instead of re-scanning) for this same iteration's TOP10,
    so the file must exist with this tick's timestamp by the time
    dt.run() is called below."""
    with open(live_p10.STATE_FILE, "w", encoding="utf-8") as f:
        f.write(f'{{"tick": {tick_no}, "capital": 1000000.0}}\n')
    with open(live_p10.HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{tick_no},live_row\n")
    if os.path.exists(dt.fast.SCAN_CACHE_FILE):
        os.remove(dt.fast.SCAN_CACHE_FILE)
    payload = {"timestamp": now.timestamp(), "raw": list(CANDS), "scanned": 225}
    with open(dt.fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _run_simulated_day(work_dir, with_daytrade):
    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

        full_bars = _build_day_bars()
        now_holder = {"now": None}

        def fake_download_5m(ticker):
            """Only ever reveals bars whose start is <= now - DATA_DELAY,
            modeling the real ~16-minute yfinance lag (see module
            docstring) rather than the unrealistic "data is always
            instantly available" the pre-fix harness assumed."""
            now = now_holder["now"]
            cutoff = pd.Timestamp(now.replace(tzinfo=None)) - DATA_DELAY
            df = full_bars.get(ticker)
            if df is None:
                return None
            sliced = df[df.index <= cutoff]
            return sliced if not sliced.empty else None

        runtimes = []
        with patch.object(live_p10, "download_5m", side_effect=fake_download_5m), \
             patch.object(live_loop, "_market_regime", return_value=("neutral", 0.0, 0.0)), \
             patch.object(live_loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}), \
             patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=POLICY):
            for i, tick in enumerate(_tick_range()):
                now_holder["now"] = tick
                _simulate_live_tick(i, tick)
                if with_daytrade:
                    t0 = time.monotonic()
                    dt.run(now=tick)
                    runtimes.append(time.monotonic() - t0)

        with open(live_p10.STATE_FILE, "rb") as f:
            live_state_bytes = f.read()
        with open(live_p10.HISTORY_FILE, "rb") as f:
            live_history_bytes = f.read()
        daytrade_history = (
            pd.read_csv(dt.HISTORY_FILE) if os.path.exists(dt.HISTORY_FILE) else pd.DataFrame()
        )
        cancelled_log = (
            pd.read_csv(dt.CANCELLED_LOG_FILE) if os.path.exists(dt.CANCELLED_LOG_FILE) else pd.DataFrame()
        )
        return live_state_bytes, live_history_bytes, daytrade_history, cancelled_log, runtimes
    finally:
        os.chdir(prev_cwd)


class FullDaySimulation(unittest.TestCase):
    def test_live_files_byte_identical_with_and_without_daytrade(self):
        with tempfile.TemporaryDirectory() as d_with, tempfile.TemporaryDirectory() as d_without:
            state_with, history_with, _, _, _ = _run_simulated_day(d_with, True)
            state_without, history_without, _, _, _ = _run_simulated_day(d_without, False)

        self.assertEqual(state_with, state_without)
        self.assertEqual(history_with, history_without)

        # sanity: the stub live tick actually ran all 80 ticks in both cases
        self.assertEqual(history_with.decode().count("live_row"), 80)

    def test_daytrade_tick_runtime_stays_far_under_90s_budget(self):
        with tempfile.TemporaryDirectory() as d:
            *_, runtimes = _run_simulated_day(d, True)
        self.assertEqual(len(runtimes), 80)
        self.assertLess(max(runtimes), 90.0)
        # this is a mocked harness: it rules out a runaway loop / O(day)
        # blow-up in the tick logic itself, not real yfinance network
        # latency (no offline test can simulate that).

    def test_full_day_trade_sequence_with_realistic_data_delay(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, daytrade_history, cancelled_log, _ = _run_simulated_day(d, True)

        self.assertEqual(len(daytrade_history), 2, daytrade_history.to_dict("records"))
        self.assertEqual(len(cancelled_log), 0)  # no cancellations in this scenario
        first, second = daytrade_history.iloc[0], daytrade_history.iloc[1]

        # trade 1: decision at 09:55 (the first eligible tick now that entry
        # decisions start at 09:55, not 09:30), fill bar is the 09:55 bar
        # itself (decision lands exactly on a 5-min boundary), filled only
        # once that bar clears the simulated 16-minute data delay (visible
        # from tick 10:15 onward: now - DATA_DELAY >= 09:55), then closed
        # via the 11:00 gap straight through TP (open>=tp).
        self.assertEqual(first["ticker"], "7203.T")
        self.assertEqual(first["fill_method"], dt.FILL_METHOD_V2)
        self.assertEqual(pd.Timestamp(first["decision_time"]), pd.Timestamp(f"{DAY} 09:55:00"))
        self.assertEqual(pd.Timestamp(first["fill_bar_time"]), pd.Timestamp(f"{DAY} 09:55:00"))
        self.assertEqual(first["entry_price"], 3005.0)  # Open of the 09:55 fill bar
        self.assertEqual(first["result"], "TP")
        self.assertEqual(first["exit_time"], "11:00")
        self.assertAlmostEqual(float(first["exit_price"]), 3050.0)
        self.assertEqual(first["entry_window_start"], "09:55")

        # trade 2: 7203.T's exit is only processed once its own data delay
        # clears (tick 11:20), so the next decision tick is 11:25; 7203.T is
        # still within its own 30-minute cooldown from that exit, so the
        # entry switches to 9984.T. It never touches its own TP/SL and
        # rides to the 15:20 forced exit.
        self.assertEqual(second["ticker"], "9984.T")
        self.assertEqual(second["fill_method"], dt.FILL_METHOD_V2)
        self.assertEqual(pd.Timestamp(second["decision_time"]), pd.Timestamp(f"{DAY} 11:25:00"))
        self.assertEqual(pd.Timestamp(second["fill_bar_time"]), pd.Timestamp(f"{DAY} 11:25:00"))
        self.assertEqual(second["entry_price"], 5000.0)
        self.assertEqual(second["result"], "FORCED_EXIT")
        self.assertEqual(second["exit_time"], "15:20")
        self.assertAlmostEqual(float(second["exit_price"]), 5000.0)
        self.assertEqual(second["entry_window_start"], "09:55")

        for row in (first, second):
            self.assertEqual(row["track"], dt.TRACK)
            self.assertEqual(bool(row["validation_eligible"]), False)

    def test_no_trade_violates_the_fill_invariant(self):
        """exit_time >= fill_bar_time >= decision_time for every v2 trade
        in the simulated day (also asserted live in _close_position(), but
        checked again here independently from the written CSV)."""
        with tempfile.TemporaryDirectory() as d:
            _, _, daytrade_history, _, _ = _run_simulated_day(d, True)
        self.assertGreater(len(daytrade_history), 0)
        for _, row in daytrade_history.iterrows():
            self.assertEqual(row["fill_method"], dt.FILL_METHOD_V2)
            decision_time = pd.Timestamp(row["decision_time"])
            fill_bar_time = pd.Timestamp(row["fill_bar_time"])
            exit_dt = pd.Timestamp(f"{row['exit_date']} {row['exit_time']}")
            self.assertGreaterEqual(fill_bar_time, decision_time, row.to_dict())
            self.assertGreaterEqual(exit_dt, fill_bar_time, row.to_dict())
            self.assertGreaterEqual(float(row["hold_minutes"]), 0.0)


# =====================================================================
# DOWN日(down用policy無し → strategy_policy.jsonへフォールバックして実売買)
# + 09:40の場中急落ブレーキ(新規買いのみ停止、空売り・決済は継続)を、実物の
# daily_decision.py(判断ファイル・policyハッシュ・ブレーキ)で1日通して回す。
# =====================================================================

import shutil  # noqa: E402

import daily_decision  # noqa: E402
import futures_trend  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
DOWN_CANDS = [
    {"ticker": "7203.T", "direction": "BUY", "price": 3005.0, "tp": 3065.0, "sl": 2975.0, "score": 95.0,
     "up_probability": 80.0, "down_probability": 5.0, "top10_rank": 1, "market_regime": "neutral"},
    {"ticker": "8035.T", "direction": "SHORT", "price": 4000.0, "tp": 3960.0, "sl": 4020.0, "score": 50.0,
     "up_probability": 10.0, "down_probability": 40.0, "top10_rank": 2, "market_regime": "neutral"},
]
BRAKE_FROM = "09:40"
RECOVER_FROM = "10:30"


def _down_futures_bars(*_a, **_k):
    idx = pd.bdate_range(end="2026-09-24", periods=40)
    return pd.DataFrame({"Close": np.linspace(70000, 60000, 40)}, index=idx)


def _build_down_day_bars():
    idx = pd.date_range(f"{DAY} 09:00", f"{DAY} 15:35", freq="5min")
    close = [3005.0] * len(idx)
    bars_7203 = pd.DataFrame({"Open": close, "High": [c + 2 for c in close], "Low": [c - 2 for c in close],
                              "Close": close, "Volume": [1000] * len(idx)}, index=idx)
    # 8035.T: 4000付近で横ばい → 11:00に3900へギャップダウン(空売りのTPを寄りで通過) → 以後横ばい
    close8 = [4000.0 if ts < pd.Timestamp(f"{DAY} 11:00") else 3900.0 for ts in idx]
    bars_8035 = pd.DataFrame({"Open": close8, "High": [c + 1 for c in close8], "Low": [c - 1 for c in close8],
                              "Close": close8, "Volume": [1000] * len(idx)}, index=idx)
    return {"7203.T": bars_7203, "8035.T": bars_8035}


def _run_down_day_with_brake(work_dir, with_daytrade):
    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        for f in ("strategy_policy.json", "strategy_policy_up.json"):
            shutil.copy(os.path.join(REPO, f), f)
        full_bars = _build_down_day_bars()
        now_holder = {"now": None}
        base = float(_down_futures_bars()["Close"].iloc[-1])

        def fake_download_5m(ticker):
            cutoff = pd.Timestamp(now_holder["now"].replace(tzinfo=None)) - DATA_DELAY
            df = full_bars.get(ticker)
            if df is None:
                return None
            sliced = df[df.index <= cutoff]
            return sliced if not sliced.empty else None

        def fake_intraday_price(*_a, **_k):
            hm = now_holder["now"].strftime("%H:%M")
            if hm < BRAKE_FROM:
                return base * 0.995
            if hm < RECOVER_FROM:
                return base * 0.98  # -2.0% → ブレーキ発動
            return base * 1.01  # 反発してもブレーキは当日中維持

        def live_tick(tick_no, now):
            # 本物のlive tick(profit_top10_paper._run)と同じく毎tick急落ブレーキを評価する
            # (判断ファイルはlive/daytrade共有。liveのstate/historyには書かない)。
            daily_decision.update_crash_brake(now)
            with open(live_p10.STATE_FILE, "w", encoding="utf-8") as f:
                f.write(f'{{"tick": {tick_no}, "capital": 1000000.0}}\n')
            with open(live_p10.HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(f"{tick_no},live_row\n")
            payload = {"timestamp": now.timestamp(), "raw": [dict(c) for c in DOWN_CANDS], "scanned": 225}
            with open(dt.fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        decisions_policy = []
        with patch.object(live_p10, "download_5m", side_effect=fake_download_5m), \
             patch.object(live_loop, "_market_regime", return_value=("neutral", 0.0, 0.0)), \
             patch.object(live_loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0}), \
             patch.object(futures_trend, "_download", side_effect=_down_futures_bars), \
             patch.object(futures_trend, "intraday_price", side_effect=fake_intraday_price), \
             patch.object(live_p10, "load_policy", return_value=POLICY):
            daily_decision.ensure_decision(_tick_range()[0].replace(hour=8, minute=30))
            for i, tick in enumerate(_tick_range()):
                now_holder["now"] = tick
                live_tick(i, tick)
                if with_daytrade:
                    dt.run(now=tick)
                    d = daily_decision.ensure_decision(tick)
                    decisions_policy.append((d["policy_file"], d["policy_hash"]))

        with open(live_p10.STATE_FILE, "rb") as f:
            live_state_bytes = f.read()
        with open(live_p10.HISTORY_FILE, "rb") as f:
            live_history_bytes = f.read()
        history = pd.read_csv(dt.HISTORY_FILE) if os.path.exists(dt.HISTORY_FILE) else pd.DataFrame()
        with open(daily_decision.DECISION_FILE, encoding="utf-8") as f:
            final_decision = json.load(f)
        return live_state_bytes, live_history_bytes, history, final_decision, decisions_policy
    finally:
        os.chdir(prev_cwd)


class FullDayDownDayWithCrashBrake(unittest.TestCase):
    def test_live_files_byte_identical_with_and_without_daytrade(self):
        with tempfile.TemporaryDirectory() as d_with, tempfile.TemporaryDirectory() as d_without:
            s_with, h_with, *_ = _run_down_day_with_brake(d_with, True)
            s_without, h_without, *_ = _run_down_day_with_brake(d_without, False)
        self.assertEqual(s_with, s_without)
        self.assertEqual(h_with, h_without)

    def test_down_day_trades_short_only_after_brake_and_keeps_exits(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, history, final_decision, decisions_policy = _run_down_day_with_brake(d, True)

        # 判断: DOWN日、汎用policyへフォールバックして実売買、1日中同じpolicy
        self.assertEqual(final_decision["trend"], "down")
        self.assertEqual(final_decision["policy_file"], "strategy_policy.json")
        self.assertTrue(final_decision["policy_fallback"])
        self.assertTrue(final_decision["entry_allowed"])
        self.assertEqual(len(set(decisions_policy)), 1)
        # ブレーキは09:40に発動し、10:30以降に反発しても当日中は解除されない
        self.assertTrue(final_decision["intraday_crash_brake"])
        self.assertTrue(final_decision["crash_brake_time"].startswith(f"{DAY}T09:40"))

        self.assertGreaterEqual(len(history), 1, history.to_dict("records"))
        # ブレーキ中は一度もBUYを建てない(TOP1のBUYはbrake_buyで飛ばされる)
        self.assertTrue((history["direction"] == "SHORT").all(), history.to_dict("records"))
        self.assertTrue((history["policy_file"] == "strategy_policy.json").all())
        first = history.iloc[0]
        self.assertEqual(first["ticker"], "8035.T")
        self.assertIn("7203.T:brake_buy", str(first["skipped"]))
        self.assertEqual(pd.Timestamp(first["decision_time"]), pd.Timestamp(f"{DAY} 09:55:00"))
        # 決済はブレーキ中も継続(11:00のギャップダウンで空売りTP)
        self.assertEqual(first["result"], "TP")
        self.assertEqual(first["exit_time"], "11:00")
        for _, row in history.iterrows():
            self.assertGreaterEqual(pd.Timestamp(row["fill_bar_time"]), pd.Timestamp(row["decision_time"]))


if __name__ == "__main__":
    unittest.main()
