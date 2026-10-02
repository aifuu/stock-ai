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

        # trade 1: decision at 09:30 (the first eligible tick), fill bar is
        # the 09:30 bar itself (decision lands exactly on a 5-min boundary),
        # filled only once that bar clears the simulated 16-minute data
        # delay (visible from tick 09:50 onward: now - DATA_DELAY >= 09:30),
        # then closed via the 11:00 gap straight through TP (open>=tp).
        self.assertEqual(first["ticker"], "7203.T")
        self.assertEqual(first["fill_method"], dt.FILL_METHOD_V2)
        self.assertEqual(pd.Timestamp(first["decision_time"]), pd.Timestamp(f"{DAY} 09:30:00"))
        self.assertEqual(pd.Timestamp(first["fill_bar_time"]), pd.Timestamp(f"{DAY} 09:30:00"))
        self.assertEqual(first["entry_price"], 3005.0)  # Open of the 09:30 fill bar
        self.assertEqual(first["result"], "TP")
        self.assertEqual(first["exit_time"], "11:00")
        self.assertAlmostEqual(float(first["exit_price"]), 3050.0)

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


if __name__ == "__main__":
    unittest.main()
