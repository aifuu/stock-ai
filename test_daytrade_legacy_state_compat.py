"""Proves daytrade_tp5000_sl8000_paper.py's backward-compat guarantee
against the *real*, currently-live production state file on origin/main:
a position opened before the 2026-10-02 fill-rule fix (no fill_method
field) must keep loading and running under the legacy v1 exit path (see
the module docstring's "Fill rule v1" section) rather than crashing or
silently being reinterpreted as a v2 position.

Fetches daytrade_tp5000_sl8000_state.json fresh from origin/main via curl
(not a checked-in fixture -- the live ai-stock-scan workflow commits a new
version of this file roughly every 5 minutes while the market is open), so
this test is skipped rather than failed if the network/GitHub is
unreachable from this environment.

Network calls are otherwise blocked in every other daytrade test file;
this one is the deliberate, documented exception, and only ever GETs a
public raw file -- never posts, never touches yfinance.
"""
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import daytrade_tp5000_sl8000_paper as dt
import profit_top10_paper as live_p10

TZ = ZoneInfo("Asia/Tokyo")
RAW_URL = "https://raw.githubusercontent.com/aifuu/stock-ai/main/daytrade_tp5000_sl8000_state.json"


def _fetch_live_state_json():
    try:
        out = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", "20", RAW_URL],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        return None, str(exc)
    if out.returncode != 0 or not out.stdout.strip():
        return None, f"curl exit={out.returncode} stderr={out.stderr.strip()}"
    try:
        return json.loads(out.stdout), None
    except Exception as exc:
        return None, f"invalid JSON: {exc}"


class LegacyLiveStateLoadsAndRuns(unittest.TestCase):
    def setUp(self):
        data, err = _fetch_live_state_json()
        if data is None:
            self.skipTest(f"could not fetch live state from origin/main (network unavailable?): {err}")
        self.live_state = data
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()

    def test_live_state_shape_has_no_fill_method_on_open_position(self):
        # Sanity check on the fetched fixture itself: today's production
        # position(s) were opened under the pre-fix (v1) code path, so they
        # must have no fill_method key at all -- this is exactly the shape
        # _evaluate_exit() must route to the legacy path.
        positions = self.live_state.get("positions", [])
        for pos in positions:
            self.assertNotIn("fill_method", pos)

    def test_load_state_parses_the_real_production_file(self):
        with open(dt.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.live_state, f)
        loaded = dt.load_state()
        self.assertEqual(loaded["positions"], self.live_state.get("positions", []))
        self.assertIsNone(loaded.get("pending"))  # defaulted; field didn't exist pre-fix

    def test_run_does_not_crash_and_still_evaluates_the_legacy_position(self):
        with open(dt.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.live_state, f)
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

        positions = self.live_state.get("positions", [])
        if not positions:
            self.skipTest("live state currently has no open position to exercise the legacy exit path")
        pos = positions[0]
        ticker = pos["ticker"]
        entry_price = float(pos["entry_price"])
        today = self.live_state.get("trade_date") or datetime.now(TZ).strftime("%Y-%m-%d")

        # A flat/no-op 5m bar right after price_bar_time: proves the legacy
        # path runs end-to-end (download -> gap-check -> no exit) without
        # raising, using real field values straight from production.
        bar_time = pd.Timestamp(pos["price_bar_time"]) + pd.Timedelta(minutes=5)
        bars = pd.DataFrame(
            {"Open": [entry_price], "High": [entry_price + 1], "Low": [entry_price - 1],
             "Close": [entry_price], "Volume": [1000]},
            index=[bar_time],
        )
        now = pd.Timestamp(today).to_pydatetime().replace(
            hour=bar_time.hour, minute=bar_time.minute, tzinfo=TZ
        )
        with patch.object(live_p10, "download_5m", return_value=bars), \
             patch.object(dt.fast, "scan_progressive_with_prefilter", return_value=([], 225)):
            dt.run(now=now)  # must not raise

        reloaded = dt.load_state()
        # still exactly one position (not exited by this no-op bar), still
        # tagged with no fill_method -- the legacy shape survives untouched
        self.assertEqual(len(reloaded["positions"]), 1)
        self.assertEqual(reloaded["positions"][0]["ticker"], ticker)
        self.assertNotIn("fill_method", reloaded["positions"][0])


if __name__ == "__main__":
    unittest.main()
