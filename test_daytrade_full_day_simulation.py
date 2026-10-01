"""Full simulated trading day for daytrade_tp5000_sl8000_paper.py, driven
through the same 5-minute tick sequence the ai-stock-scan.yml workflow
uses (a live tick, then the daytrade tick, every 5 minutes from 09:00 to
15:35 JST), with fixture data including a gap and a cooldown-driven
candidate switch, proving:

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

This module makes real network calls if a test forgets to mock the right
entry point, so requests.post and yfinance.download are stubbed to raise
before anything is imported (same pattern as the other daytrade/
run_profit_loop test files).
"""
import os
import tempfile
import time
import unittest
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

TZ = ZoneInfo("Asia/Tokyo")
DAY = "2026-09-25"
POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False,
          "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0, "hold_days": 1, "status": "PENDING"}

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


def _simulate_live_tick(tick_no):
    """Stand-in for the real live tick (paper_fast_entrypoint.py calling
    into profit_top10_paper.py). Content depends only on tick_no, never on
    whether daytrade ran, so the two simulated days are directly
    comparable byte-for-byte."""
    with open(live_p10.STATE_FILE, "w", encoding="utf-8") as f:
        f.write(f'{{"tick": {tick_no}, "capital": 1000000.0}}\n')
    with open(live_p10.HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{tick_no},live_row\n")


def _run_simulated_day(work_dir, with_daytrade):
    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

        full_bars = _build_day_bars()
        now_holder = {"now": None}

        def fake_download_5m(ticker):
            now = now_holder["now"]
            df = full_bars.get(ticker)
            if df is None:
                return None
            sliced = df[df.index <= pd.Timestamp(now.replace(tzinfo=None))]
            return sliced if not sliced.empty else None

        def fake_scan(policy):
            return list(CANDS), 225

        runtimes = []
        with patch.object(live_p10, "download_5m", side_effect=fake_download_5m), \
             patch.object(dt.fast, "scan_progressive_with_prefilter", side_effect=fake_scan), \
             patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=POLICY):
            for i, tick in enumerate(_tick_range()):
                now_holder["now"] = tick
                _simulate_live_tick(i)
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
        return live_state_bytes, live_history_bytes, daytrade_history, runtimes
    finally:
        os.chdir(prev_cwd)


class FullDaySimulation(unittest.TestCase):
    def test_live_files_byte_identical_with_and_without_daytrade(self):
        with tempfile.TemporaryDirectory() as d_with, tempfile.TemporaryDirectory() as d_without:
            state_with, history_with, _, _ = _run_simulated_day(d_with, True)
            state_without, history_without, _, _ = _run_simulated_day(d_without, False)

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

    def test_full_day_trade_sequence_gap_tp_cooldown_switch_and_forced_exit(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, daytrade_history, _ = _run_simulated_day(d, True)

        self.assertEqual(len(daytrade_history), 2, daytrade_history.to_dict("records"))
        first, second = daytrade_history.iloc[0], daytrade_history.iloc[1]

        # trade 1: 7203.T entered at 09:30, closed via the 11:00 gap (TP@open)
        self.assertEqual(first["ticker"], "7203.T")
        self.assertEqual(first["entry_time"], "09:30")
        self.assertEqual(first["result"], "TP")
        self.assertEqual(first["exit_time"], "11:00")
        self.assertAlmostEqual(float(first["exit_price"]), 3050.0)

        # trade 2: 7203.T was in cooldown, so the next entry switched to
        # 9984.T; it never touches its own TP/SL and rides to the 15:20
        # forced exit
        self.assertEqual(second["ticker"], "9984.T")
        self.assertEqual(second["entry_time"], "11:05")
        self.assertEqual(second["result"], "FORCED_EXIT")
        self.assertEqual(second["exit_time"], "15:20")
        self.assertAlmostEqual(float(second["exit_price"]), 5000.0)

        for row in (first, second):
            self.assertEqual(row["track"], dt.TRACK)
            self.assertEqual(bool(row["validation_eligible"]), False)


if __name__ == "__main__":
    unittest.main()
