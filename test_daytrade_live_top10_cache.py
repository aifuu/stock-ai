"""Tests for Change A (daytrade must pick from the SAME live-tick TOP10,
never its own re-scan) and Change B (Discord: cancellation notice, the
once-per-day summary, and the number-before-unit display format) in
daytrade_tp5000_sl8000_paper.py.

Same network-blocking pattern as the other daytrade test modules: this
module must never make a real yfinance/Discord call even if a test
forgets to mock the right entry point.
"""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
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
import paper_fast_entrypoint as fast  # noqa: E402
import profit_top10_paper as live_p10  # noqa: E402
import run_profit_loop as live_loop  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")


class TmpCwdMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()


class _RegimeAndFeedbackPatched:
    """Matches test_daytrade_tp5000_sl8000_paper.py's pattern: pins the
    regime-dependent ranking so tests are deterministic without hitting
    app.make_nikkei()'s own network call."""

    def __init__(self, regime="neutral"):
        self.regime = regime

    def __enter__(self):
        self._p1 = patch.object(live_loop, "_market_regime", return_value=(self.regime, 1.0, 1.0))
        self._p2 = patch.object(live_loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0})
        self._p1.start()
        self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop()
        self._p2.stop()


POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False}

# A realistic fixture mirroring 2026-10-02's real TOP10 shape: 4062.T and
# 6857.T both rank above 5301.T but neither fits one 100-share lot inside
# the 1,000,000 JPY budget -> 5301.T (budget-affordable) must be chosen,
# with both higher-ranked tickers recorded as budget-skipped.
REAL_SHAPED_POOL = [
    {"ticker": "4062.T", "company": "A", "direction": "BUY", "price": 12075.0, "score": 80.0,
     "up_probability": 60.0, "down_probability": 10.0, "flat_probability": 5.0},
    {"ticker": "6857.T", "company": "B", "direction": "BUY", "price": 38500.0, "score": 75.0,
     "up_probability": 55.0, "down_probability": 15.0, "flat_probability": 5.0},
    {"ticker": "5301.T", "company": "C", "direction": "BUY", "price": 2304.0, "score": 70.0,
     "up_probability": 50.0, "down_probability": 20.0, "flat_probability": 5.0},
]


def _write_cache(raw, scanned, now, path=None):
    path = path or fast.SCAN_CACHE_FILE
    payload = {"timestamp": now.timestamp(), "raw": raw, "scanned": scanned}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


# =====================================================================
# Change A: _load_live_tick_cache -- missing/corrupt/too old/other date
# =====================================================================

class LoadLiveTickCache(TmpCwdMixin, unittest.TestCase):
    NOW = datetime(2026, 10, 2, 9, 31, tzinfo=TZ)

    def test_missing_file_returns_none(self):
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))

    def test_fresh_cache_returns_raw_and_scanned(self):
        _write_cache(REAL_SHAPED_POOL, 225, self.NOW)
        result = dt._load_live_tick_cache(self.NOW)
        self.assertIsNotNone(result)
        raw, scanned = result
        self.assertEqual(raw, REAL_SHAPED_POOL)
        self.assertEqual(scanned, 225)

    def test_age_over_300s_but_under_15min_is_still_accepted(self):
        # the key behavior Change A fixes: fast.scan_progressive_with_prefilter's
        # own 300s TTL would have re-scanned here, but daytrade's own bound is 15min.
        written_at = self.NOW - timedelta(seconds=301)
        _write_cache(REAL_SHAPED_POOL, 225, written_at)
        result = dt._load_live_tick_cache(self.NOW)
        self.assertIsNotNone(result)

    def test_age_over_15min_returns_none(self):
        written_at = self.NOW - timedelta(minutes=15, seconds=1)
        _write_cache(REAL_SHAPED_POOL, 225, written_at)
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))

    def test_future_timestamp_returns_none(self):
        written_at = self.NOW + timedelta(seconds=5)
        _write_cache(REAL_SHAPED_POOL, 225, written_at)
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))

    def test_other_jst_date_returns_none(self):
        # same wall-clock age bound but a different JST calendar date --
        # e.g. a leftover file from just after midnight JST rollover.
        written_at = self.NOW - timedelta(hours=9, minutes=50)
        _write_cache(REAL_SHAPED_POOL, 225, written_at)
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))

    def test_corrupt_json_returns_none(self):
        with open(fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))

    def test_missing_raw_key_returns_none(self):
        with open(fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"timestamp": self.NOW.timestamp(), "scanned": 225}, f)
        self.assertIsNone(dt._load_live_tick_cache(self.NOW))


# =====================================================================
# Change A: _get_live_tick_top10 never scans/downloads on its own when
# a usable cache exists, and the ranking matches scan_candidates_fixed()
# on the exact same raw pool.
# =====================================================================

class GetLiveTickTop10NoScanNoDownload(TmpCwdMixin, unittest.TestCase):
    NOW = datetime(2026, 10, 2, 9, 31, tzinfo=TZ)

    def test_no_scan_or_download_function_called_when_cache_is_stale_but_within_bound(self):
        written_at = self.NOW - timedelta(seconds=301)  # past fast's own 300s TTL
        _write_cache(REAL_SHAPED_POOL, 225, written_at)
        state = dt.default_state()
        with _RegimeAndFeedbackPatched("neutral"), \
             patch.object(fast, "_load_disk_scan_cache") as disk_cache, \
             patch.object(fast, "_batch_download_all") as batch_dl, \
             patch.object(fast, "_base_scan") as base_scan:
            result = dt._get_live_tick_top10(POLICY, self.NOW, state)
        disk_cache.assert_not_called()
        batch_dl.assert_not_called()
        base_scan.assert_not_called()
        self.assertIsNotNone(result)
        top10, scanned = result
        self.assertEqual(scanned, 225)
        self.assertEqual([c["ticker"] for c in top10], ["4062.T", "6857.T", "5301.T"])

    def test_returns_none_and_never_scans_when_cache_missing(self):
        state = dt.default_state()
        with _RegimeAndFeedbackPatched("neutral"), \
             patch.object(fast, "_load_disk_scan_cache") as disk_cache, \
             patch.object(fast, "_batch_download_all") as batch_dl:
            result = dt._get_live_tick_top10(POLICY, self.NOW, state)
        disk_cache.assert_not_called()
        batch_dl.assert_not_called()
        self.assertIsNone(result)

    def test_ranking_equals_scan_candidates_fixed_on_the_same_raw_pool(self):
        _write_cache(REAL_SHAPED_POOL, 225, self.NOW)
        state = dt.default_state()
        with _RegimeAndFeedbackPatched("neutral"):
            top10_live, _ = dt._get_live_tick_top10(POLICY, self.NOW, state)

            orig = live_loop._original_scan
            live_loop._original_scan = lambda policy: (list(REAL_SHAPED_POOL), 225)
            try:
                top10_direct, _ = live_loop.scan_candidates_fixed(POLICY)
            finally:
                live_loop._original_scan = orig

        self.assertEqual([c["ticker"] for c in top10_live], [c["ticker"] for c in top10_direct])
        self.assertEqual([c["top10_rank"] for c in top10_live], [c["top10_rank"] for c in top10_direct])

    def test_fast_cache_result_restored_after_call(self):
        _write_cache(REAL_SHAPED_POOL, 225, self.NOW)
        state = dt.default_state()
        fast._cache["result"] = None
        with _RegimeAndFeedbackPatched("neutral"):
            dt._get_live_tick_top10(POLICY, self.NOW, state)
        self.assertIsNone(fast._cache["result"])  # restored, not left pointing at daytrade's pool


# =====================================================================
# Change A: priority walk skips unaffordable higher-ranked candidates and
# records them; the live-cache-miss condition skips entry and notifies
# Discord at most once per day.
# =====================================================================

class TryEntryUsesLiveCacheAndRecordsSkips(TmpCwdMixin, unittest.TestCase):
    NOW = datetime(2026, 10, 2, 9, 31, tzinfo=TZ)

    def setUp(self):
        super().setUp()
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

    def _patched(self):
        return (
            patch.object(dt, "choose_policy_file_reusing_live_tick",
                         return_value=("strategy_policy.json", "reused_today_row")),
            patch.object(live_p10, "load_policy", return_value=POLICY),
        )

    def test_budget_skips_are_recorded_and_5301_is_chosen(self):
        _write_cache(REAL_SHAPED_POOL, 225, self.NOW)
        state = dt.default_state()
        p1, p2 = self._patched()
        with _RegimeAndFeedbackPatched("neutral"), p1, p2:
            msg = dt._try_entry(state, self.NOW, "2026-10-02")
        self.assertIsNotNone(msg)
        pending = state["pending"]
        self.assertEqual(pending["ticker"], "5301.T")
        self.assertEqual(pending["top10_source"], dt.TOP10_SOURCE_LIVE_TICK_CACHE)
        self.assertEqual(pending["skipped"], "4062.T:budget,6857.T:budget")
        self.assertIn("TOP10(live):", msg)
        self.assertIn("4062.T", msg)
        self.assertIn("6857.T", msg)
        self.assertIn("5301.T", msg)

    def test_no_live_cache_skips_entry_without_scanning(self):
        state = dt.default_state()
        p1, p2 = self._patched()
        with p1, p2, patch.object(fast, "scan_progressive_with_prefilter") as scan_fn:
            msg = dt._try_entry(state, self.NOW, "2026-10-02")
        scan_fn.assert_not_called()
        self.assertIsNone(msg)
        self.assertIsNone(state.get("pending"))

    def test_live_cache_miss_notifies_discord_at_most_once_per_day(self):
        state = dt.default_state()
        p1, p2 = self._patched()
        sent = []
        with p1, p2, patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt._try_entry(state, self.NOW, "2026-10-02")
            dt._try_entry(state, self.NOW + timedelta(minutes=5), "2026-10-02")
            dt._try_entry(state, self.NOW + timedelta(minutes=10), "2026-10-02")
        self.assertEqual(sent, ["⏸ daytrade: live TOP10未取得のため見送り"])  # only once
        self.assertEqual(state["live_cache_miss_notice_date"], "2026-10-02")

    def test_live_cache_miss_notifies_again_on_a_new_day(self):
        state = dt.default_state()
        p1, p2 = self._patched()
        sent = []
        with p1, p2, patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt._try_entry(state, self.NOW, "2026-10-02")
            dt._try_entry(state, self.NOW + timedelta(days=1), "2026-10-03")
        self.assertEqual(len(sent), 2)


# =====================================================================
# Change B: cancellation notice to Discord
# =====================================================================

class CancellationNotifiesDiscord(TmpCwdMixin, unittest.TestCase):
    def test_cancel_sends_discord_message_with_ticker_direction_reason_time(self):
        state = dt.default_state()
        state["pending"] = {
            "ticker": "7203.T", "direction": "SHORT",
            "decision_time": "2026-10-02T14:49:00", "decision_date": "2026-10-02",
            "fill_bar_time": "2026-10-02T14:50:00",
        }
        sent = []
        with patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt._cancel_pending(state, "CANCELLED_NO_FILL_TOO_LATE", datetime(2026, 10, 2, 14, 49, tzinfo=TZ))
        self.assertEqual(len(sent), 1)
        self.assertIn("7203.T", sent[0])
        self.assertIn("空売り", sent[0])
        self.assertIn("CANCELLED_NO_FILL_TOO_LATE", sent[0])
        self.assertIn("14:49:00", sent[0])

    def test_no_op_and_no_discord_call_when_nothing_pending(self):
        state = dt.default_state()
        with patch.object(dt, "discord_send") as send:
            result = dt._cancel_pending(state, "CANCELLED_NO_FILL_NEW_DAY", datetime(2026, 10, 2, 9, 0, tzinfo=TZ))
        send.assert_not_called()
        self.assertIsNone(result)


# =====================================================================
# Change B: once-per-day Discord summary -- timing, format, split, safety
# =====================================================================

class DailySummaryTiming(TmpCwdMixin, unittest.TestCase):
    def test_not_sent_before_1520(self):
        state = dt.default_state()
        with patch.object(dt, "discord_send") as send:
            dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 19, tzinfo=TZ), "2026-10-02")
        send.assert_not_called()
        self.assertIsNone(state.get("daily_summary_sent_date"))

    def test_not_sent_while_position_or_pending_open(self):
        state = dt.default_state()
        state["positions"] = [{"ticker": "7203.T"}]
        with patch.object(dt, "discord_send") as send:
            dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 25, tzinfo=TZ), "2026-10-02")
        send.assert_not_called()

        state2 = dt.default_state()
        state2["pending"] = {"ticker": "7203.T"}
        with patch.object(dt, "discord_send") as send2:
            dt._maybe_send_daily_summary(state2, datetime(2026, 10, 2, 15, 25, tzinfo=TZ), "2026-10-02")
        send2.assert_not_called()

    def test_sent_once_at_1520_and_flag_prevents_resend(self):
        state = dt.default_state()
        with patch.object(dt, "discord_send", return_value=True) as send:
            dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 20, tzinfo=TZ), "2026-10-02")
        self.assertEqual(send.call_count, 1)
        self.assertEqual(state["daily_summary_sent_date"], "2026-10-02")
        self.assertEqual(dt.load_state()["daily_summary_sent_date"], "2026-10-02")

        with patch.object(dt, "discord_send") as send2:
            dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 30, tzinfo=TZ), "2026-10-02")
        send2.assert_not_called()  # already sent today -- never twice

    def test_sent_on_zero_trade_day(self):
        state = dt.default_state()
        sent = []
        with patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 20, tzinfo=TZ), "2026-10-02")
        self.assertTrue(any("本日取引なし" in m for m in sent))

    def test_flag_set_even_when_webhook_empty(self):
        # discord_send() itself prints "[daytrade] ..." and returns False
        # when DISCORD_WEBHOOK is unset -- the flag must still be set so
        # this never resends on a later tick in the same day.
        state = dt.default_state()
        dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 20, tzinfo=TZ), "2026-10-02")
        self.assertEqual(state["daily_summary_sent_date"], "2026-10-02")

    def test_discord_exception_does_not_crash_the_tick(self):
        state = dt.default_state()

        def _raise(_m):
            raise RuntimeError("boom")

        # discord_send() itself always swallows exceptions (see its own
        # docstring); simulate a caller that bypassed that contract to
        # prove _maybe_send_daily_summary still can't be crashed by it.
        with patch.object(dt, "discord_send", side_effect=_raise):
            with self.assertRaises(RuntimeError):
                dt._maybe_send_daily_summary(state, datetime(2026, 10, 2, 15, 20, tzinfo=TZ), "2026-10-02")
        # the real discord_send (unpatched) never raises in the first place:
        os.environ.pop("DISCORD_WEBHOOK", None)
        state2 = dt.default_state()
        dt._maybe_send_daily_summary(state2, datetime(2026, 10, 2, 15, 20, tzinfo=TZ), "2026-10-02")
        self.assertEqual(state2["daily_summary_sent_date"], "2026-10-02")

    def test_fires_right_after_a_same_tick_forced_exit(self):
        pos = {
            "ticker": "7203.T", "direction": "BUY", "entry_date": "2026-10-02", "entry_time": "09:35",
            "entry_datetime": "2026-10-02T09:35:00", "entry_price": 3000.0, "shares": 300,
            "invested_amount": 900000.0, "tp": 4000.0, "sl": 2000.0, "tp_pct": 33.0, "sl_pct": -33.0,
            "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": 3000.0, "price_bar_time": "2026-10-02T09:35:00",
            "fill_bar_time": "2026-10-02T09:35:00", "decision_time": "2026-10-02T09:31:00",
            "fill_method": dt.FILL_METHOD_V2,
        }
        state = dt.default_state()
        state["positions"] = [pos]
        state["trade_date"] = "2026-10-02"
        dt.save_state(state)
        idx = pd.date_range("2026-10-02 09:35", "2026-10-02 15:25", freq="5min")
        closes = [3000.0] * len(idx)
        bars = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                              "Volume": [1000] * len(idx)}, index=idx)
        sent = []
        with patch.object(live_p10, "download_5m", return_value=bars), \
             patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt.run(now=datetime(2026, 10, 2, 15, 20, tzinfo=TZ))
        self.assertTrue(any("本日サマリー" in m for m in sent))
        self.assertEqual(dt.load_state()["daily_summary_sent_date"], "2026-10-02")


class DailySummaryContent(TmpCwdMixin, unittest.TestCase):
    TODAY = "2026-10-02"

    def _write_history(self, rows):
        pd.DataFrame(rows).to_csv(dt.HISTORY_FILE, index=False, encoding="utf-8-sig")

    def test_format_matches_number_before_unit_pattern(self):
        self._write_history([
            {"ticker": "7203.T", "direction": "BUY", "entry_time": "09:35", "exit_time": "10:00",
             "exit_date": self.TODAY, "result": "TP", "pnl": 5000.0, "fill_method": dt.FILL_METHOD_V2},
            {"ticker": "9984.T", "direction": "BUY", "entry_time": "11:00", "exit_time": "11:30",
             "exit_date": self.TODAY, "result": "SL", "pnl": -8000.0, "fill_method": dt.FILL_METHOD_V2},
        ])
        messages = dt._build_daily_summary_messages(self.TODAY)
        text = "\n".join(messages)
        self.assertRegex(text, r"\d+件 \d+勝\d+敗")
        self.assertIn("勝率50.0%", text)
        self.assertIn("利確1回 損切1回 強制決済0回", text)
        self.assertIn("合計 -3,000円", text)
        self.assertNotRegex(text, r"件数\d+")  # never label-then-number

    def test_v1_v2_split_reported(self):
        self._write_history([
            {"ticker": "7203.T", "direction": "BUY", "entry_time": "09:35", "exit_time": "10:00",
             "exit_date": self.TODAY, "result": "TP", "pnl": 5000.0, "fill_method": dt.FILL_METHOD_V2},
            {"ticker": "6302.T", "direction": "BUY", "entry_time": "09:00", "exit_time": "15:20",
             "exit_date": self.TODAY, "result": "FORCED_EXIT", "pnl": -500.0, "fill_method": ""},
        ])
        text = "\n".join(dt._build_daily_summary_messages(self.TODAY))
        self.assertIn("v2(新方式)1件", text)
        self.assertIn("v1(旧方式・参考値)1件", text)
        self.assertIn(dt.LEGACY_FILL_METHOD_LABEL, text)  # blank fill_method labeled as legacy v1

    def test_cancellations_counted(self):
        pd.DataFrame([
            {"ticker": "7203.T", "decision_date": self.TODAY, "reason": "CANCELLED_NO_FILL_BUDGET"},
            {"ticker": "9984.T", "decision_date": "2026-10-01", "reason": "CANCELLED_NO_FILL_TOO_LATE"},
        ]).to_csv(dt.CANCELLED_LOG_FILE, index=False, encoding="utf-8-sig")
        text = "\n".join(dt._build_daily_summary_messages(self.TODAY))
        self.assertIn("本日キャンセル 1件", text)  # only today's cancellation counted

    def test_zero_trade_day_message(self):
        text = "\n".join(dt._build_daily_summary_messages(self.TODAY))
        self.assertIn("本日取引なし", text)
        self.assertIn("本日キャンセル 0件", text)

    def test_cumulative_v2_vs_breakeven(self):
        self._write_history([
            {"ticker": "7203.T", "direction": "BUY", "entry_time": "09:35", "exit_time": "10:00",
             "exit_date": "2026-09-30", "result": "TP", "pnl": 5000.0, "fill_method": dt.FILL_METHOD_V2},
            {"ticker": "9984.T", "direction": "BUY", "entry_time": "11:00", "exit_time": "11:30",
             "exit_date": self.TODAY, "result": "SL", "pnl": -8000.0, "fill_method": dt.FILL_METHOD_V2},
        ])
        text = "\n".join(dt._build_daily_summary_messages(self.TODAY))
        self.assertIn("累計v2 2件", text)
        self.assertIn("損益分岐61.5%", text)
        self.assertIn("合計-3,000円", text.replace(" ", ""))

    def test_long_summary_splits_under_1950_chars_each(self):
        rows = [
            {"ticker": f"{1000 + i}.T", "direction": "BUY", "entry_time": "09:35", "exit_time": "10:00",
             "exit_date": self.TODAY, "result": "TP", "pnl": 5000.0, "fill_method": dt.FILL_METHOD_V2}
            for i in range(80)
        ]
        self._write_history(rows)
        messages = dt._build_daily_summary_messages(self.TODAY)
        self.assertGreater(len(messages), 1)
        for m in messages:
            self.assertLessEqual(len(m), 1950)

    def test_corrupt_history_does_not_raise(self):
        with open(dt.HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write("not,a,valid\nheader\nrow,too,few,cols,here\n")
        messages = dt._build_daily_summary_messages(self.TODAY)
        self.assertIsInstance(messages, list)


# =====================================================================
# History/cancelled-log CSV: new columns appended at the end, existing
# rows keep their values (empty for the new columns).
# =====================================================================

class HistoryColumnExtensionIsSafe(TmpCwdMixin, unittest.TestCase):
    def test_old_rows_survive_new_columns_being_appended(self):
        # a realistic pre-Change-A row: already has track/validation_eligible
        # (append_history has always added those) but not top10_source/skipped.
        old_row = {"ticker": "7203.T", "direction": "BUY", "pnl": 1234.5, "result": "TP",
                   "track": dt.TRACK, "validation_eligible": False}
        pd.DataFrame([old_row]).to_csv(dt.HISTORY_FILE, index=False, encoding="utf-8-sig")

        new_row = dict(old_row)
        new_row.update({"ticker": "9984.T", "pnl": -500.0, "top10_source": "live_tick_cache",
                         "skipped": "4062.T:budget"})
        dt.append_history(new_row)

        hist = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist.iloc[0]["ticker"], "7203.T")
        self.assertEqual(float(hist.iloc[0]["pnl"]), 1234.5)
        self.assertTrue(pd.isna(hist.iloc[0]["top10_source"]))  # old row: empty for new column
        self.assertTrue(pd.isna(hist.iloc[0]["skipped"]))
        self.assertEqual(hist.iloc[1]["ticker"], "9984.T")
        self.assertEqual(hist.iloc[1]["top10_source"], "live_tick_cache")
        self.assertEqual(hist.iloc[1]["skipped"], "4062.T:budget")
        # new columns land after all pre-existing columns
        self.assertEqual(hist.columns.tolist()[-2:], ["top10_source", "skipped"])


if __name__ == "__main__":
    unittest.main()
