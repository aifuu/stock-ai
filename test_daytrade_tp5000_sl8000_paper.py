"""Tests for daytrade_tp5000_sl8000_paper.py (Part A: day-trade separate
+5,000/-8,000 JPY track).

This module makes real network calls if a test forgets to mock the right
entry point (yfinance, Discord), so requests.post and yfinance.download are
stubbed to raise before anything is imported (same pattern as
test_run_profit_loop_cooldown.py), and DISCORD_WEBHOOK is forced unset.
"""
import os
import tempfile
import unittest
from datetime import datetime
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
import run_profit_loop as loop  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")


class TmpCwdMixin:
    """daytrade's state/history files are written relative to the cwd; run
    each test in an isolated tmpdir so tests never touch the repo's real
    daytrade_tp5000_sl8000_*.json/csv files."""

    def setUp(self):
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()


def _bars(closes, start, freq="5min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame(
        {"Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
         "Close": closes, "Volume": [1000] * len(closes)},
        index=idx,
    )


# =====================================================================
# TOP1はliveのopen_top1_only()と一致する(regime-aware profit_priority順)
# =====================================================================

class _RegimeAndFeedbackPatched:
    def __init__(self, regime="neutral"):
        self.regime = regime

    def __enter__(self):
        self._p1 = patch.object(loop, "_market_regime", return_value=(self.regime, 1.0, 1.0))
        self._p2 = patch.object(loop, "_load_feedback_weights", return_value={"BUY": 1.0, "SHORT": 1.0})
        self._p1.start()
        self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop()
        self._p2.stop()


class Top1MatchesLiveOpenTop1Only(unittest.TestCase):
    POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False}

    def _raw_pool(self):
        return [
            {"ticker": "7203.T", "company": "A", "direction": "BUY", "price": 3000.0, "tp": 3100.0,
             "sl": 2950.0, "score": 60.0, "up_probability": 55.0, "down_probability": 10.0,
             "flat_probability": 10.0, "data_date": "2026-09-25"},
            {"ticker": "9984.T", "company": "B", "direction": "BUY", "price": 5000.0, "tp": 5200.0,
             "sl": 4900.0, "score": 90.0, "up_probability": 70.0, "down_probability": 5.0,
             "flat_probability": 5.0, "data_date": "2026-09-25"},
            {"ticker": "8035.T", "company": "C", "direction": "SHORT", "price": 4000.0, "tp": 3800.0,
             "sl": 4100.0, "score": 70.0, "up_probability": 5.0, "down_probability": 65.0,
             "flat_probability": 5.0, "data_date": "2026-09-25"},
        ]

    def _top10_via_scan_candidates_fixed(self):
        orig = loop._original_scan
        loop._original_scan = lambda policy: (self._raw_pool(), 225)
        try:
            return loop.scan_candidates_fixed(self.POLICY)
        finally:
            loop._original_scan = orig

    def _live_choice(self, top10):
        def fake_open(state, policy, cands, today):
            state["positions"].append(dict(cands[0]))
            return list(cands)

        orig_open = loop._original_open
        loop._original_open = fake_open
        try:
            state = {"positions": [], "trades_today": 0, "trades_by_ticker_today": {},
                     "last_exit_by_ticker": {}}
            opened = loop.open_top1_only(state, self.POLICY, top10, "2026-09-25")
        finally:
            loop._original_open = orig_open
        return opened[0]["ticker"], opened[0]["direction"]

    def test_daytrade_select_top1_matches_live_neutral_regime(self):
        with _RegimeAndFeedbackPatched("neutral"):
            top10, _ = self._top10_via_scan_candidates_fixed()
            live_ticker, live_direction = self._live_choice(top10)
            chosen = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertIsNotNone(chosen)
        self.assertEqual((chosen["ticker"], chosen["direction"]), (live_ticker, live_direction))

    def test_daytrade_select_top1_matches_live_bearish_regime(self):
        with _RegimeAndFeedbackPatched("bearish"):
            top10, _ = self._top10_via_scan_candidates_fixed()
            live_ticker, live_direction = self._live_choice(top10)
            chosen = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertEqual(live_direction, "SHORT")  # bearish regime must pick a SHORT
        self.assertEqual((chosen["ticker"], chosen["direction"]), (live_ticker, live_direction))

    def test_daytrade_select_top1_skips_cooldown_ticker_like_live_would_skip_active(self):
        with _RegimeAndFeedbackPatched("neutral"):
            top10, _ = self._top10_via_scan_candidates_fixed()
            now = datetime(2026, 9, 25, 9, 35, tzinfo=TZ)
            # the top-ranked ticker is still in this track's own cooldown
            top_ticker = top10[0]["ticker"]
            cooldowns = {top_ticker: (now - loop.timedelta(minutes=5)).isoformat()}
            chosen = dt._select_top1(top10, cooldowns, now)
        self.assertIsNotNone(chosen)
        self.assertNotEqual(chosen["ticker"], top_ticker)

    def test_empty_top10_returns_none(self):
        self.assertIsNone(dt._select_top1([], {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ)))

    def test_unaffordable_candidate_is_skipped(self):
        with _RegimeAndFeedbackPatched("neutral"):
            top10, _ = self._top10_via_scan_candidates_fixed()
        # make every candidate's price exceed the budget for even one lot
        for c in top10:
            c["price"] = 20_000_000.0
        chosen = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertIsNone(chosen)


# =====================================================================
# ギャップ処理: SLがTP/SL同時タッチで優先される
# =====================================================================

class GapFillBuy(unittest.TestCase):
    def test_tp_at_open_when_gap_above_tp(self):
        bar = {"Open": 110.0, "High": 112.0, "Low": 109.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._gap_fill_exit("BUY", bar), (110.0, "TP"))

    def test_sl_at_open_when_gap_below_sl(self):
        bar = {"Open": 90.0, "High": 92.0, "Low": 88.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._gap_fill_exit("BUY", bar), (90.0, "SL"))

    def test_sl_priority_when_both_touched_intrabar(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 94.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._gap_fill_exit("BUY", bar), (95.0, "SL"))

    def test_tp_when_only_tp_touched_intrabar(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 97.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._gap_fill_exit("BUY", bar), (105.0, "TP"))

    def test_no_exit_when_neither_touched(self):
        bar = {"Open": 100.0, "High": 103.0, "Low": 98.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._gap_fill_exit("BUY", bar), (None, None))


class GapFillShort(unittest.TestCase):
    def test_tp_at_open_when_gap_below_tp(self):
        bar = {"Open": 90.0, "High": 92.0, "Low": 88.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._gap_fill_exit("SHORT", bar), (90.0, "TP"))

    def test_sl_at_open_when_gap_above_sl(self):
        bar = {"Open": 110.0, "High": 112.0, "Low": 108.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._gap_fill_exit("SHORT", bar), (110.0, "SL"))

    def test_sl_priority_when_both_touched_intrabar(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 94.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._gap_fill_exit("SHORT", bar), (105.0, "SL"))

    def test_tp_when_only_tp_touched_intrabar(self):
        bar = {"Open": 100.0, "High": 103.0, "Low": 94.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._gap_fill_exit("SHORT", bar), (95.0, "TP"))

    def test_no_exit_when_neither_touched(self):
        bar = {"Open": 100.0, "High": 102.0, "Low": 97.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._gap_fill_exit("SHORT", bar), (None, None))


# =====================================================================
# TP/SL価格は厳密にnet±5000/-8000円になるように解かれている
# =====================================================================

class TpSlPricesSolveExactNetYen(unittest.TestCase):
    def test_buy_tp_and_sl_hit_exact_net_yen(self):
        tp, sl, _, _ = dt._tp_sl_prices(3000.0, "BUY", 300)
        self.assertAlmostEqual(dt.net_pnl(3000.0, tp, 300, "BUY"), dt.TP_NET_JPY, places=6)
        self.assertAlmostEqual(dt.net_pnl(3000.0, sl, 300, "BUY"), dt.SL_NET_JPY, places=6)

    def test_short_tp_and_sl_hit_exact_net_yen(self):
        tp, sl, _, _ = dt._tp_sl_prices(3000.0, "SHORT", 300)
        self.assertAlmostEqual(dt.net_pnl(3000.0, tp, 300, "SHORT"), dt.TP_NET_JPY, places=6)
        self.assertAlmostEqual(dt.net_pnl(3000.0, sl, 300, "SHORT"), dt.SL_NET_JPY, places=6)

    def test_breakeven_win_rate_matches_8000_over_13000(self):
        self.assertAlmostEqual(-dt.SL_NET_JPY / (dt.TP_NET_JPY - dt.SL_NET_JPY), 8000.0 / 13000.0)


# =====================================================================
# ポリシー選択: 同一tickのlive判定(futures_trend_history.csv)を再利用する
# =====================================================================

class PolicyChoiceReusesLiveTick(unittest.TestCase):
    def test_reuses_todays_row_without_calling_select_policy_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "futures_trend_history.csv")
            today = datetime.now(TZ).strftime("%Y-%m-%d")
            pd.DataFrame([{"date": today, "trend": "down"}]).to_csv(path, index=False)
            with patch.object(live_p10, "select_policy_file") as sp:
                policy_file, source = dt.choose_policy_file_reusing_live_tick(
                    now=datetime.now(TZ), history_path=path
                )
            sp.assert_not_called()
        self.assertEqual(source, "reused_today_row")
        # strategy_policy_down.json is not a committed live file -> falls back to POLICY_FILE
        self.assertEqual(policy_file, live_p10.POLICY_FILE)

    def test_reused_up_trend_maps_to_policy_file_up(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "futures_trend_history.csv")
            today = datetime.now(TZ).strftime("%Y-%m-%d")
            pd.DataFrame([{"date": today, "trend": "up"}]).to_csv(path, index=False)
            with patch.object(live_p10, "select_policy_file") as sp:
                policy_file, source = dt.choose_policy_file_reusing_live_tick(
                    now=datetime.now(TZ), history_path=path
                )
            sp.assert_not_called()
        self.assertEqual(source, "reused_today_row")
        self.assertEqual(policy_file, live_p10.POLICY_FILE_UP)

    def test_falls_back_when_no_row_for_today(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "futures_trend_history.csv")
            pd.DataFrame([{"date": "2000-01-01", "trend": "up"}]).to_csv(path, index=False)
            with patch.object(live_p10, "select_policy_file",
                               return_value=("strategy_policy_up.json", {"trend": "up"})) as sp:
                policy_file, source = dt.choose_policy_file_reusing_live_tick(
                    now=datetime.now(TZ), history_path=path
                )
            sp.assert_called_once()
        self.assertEqual(source, "fallback_select_policy_file")
        self.assertEqual(policy_file, "strategy_policy_up.json")

    def test_missing_file_falls_back(self):
        with patch.object(live_p10, "select_policy_file",
                           return_value=("strategy_policy.json", {"trend": "up"})) as sp:
            policy_file, source = dt.choose_policy_file_reusing_live_tick(
                now=datetime.now(TZ), history_path="/nonexistent/futures_trend_history.csv"
            )
        sp.assert_called_once()
        self.assertEqual(source, "fallback_select_policy_file")
        self.assertEqual(policy_file, "strategy_policy.json")


# =====================================================================
# エントリー基準: 直近5分足の終値 / price_bar_time / data_delay_minutes
# =====================================================================

class EntryUsesLatest5mCloseNotScanPrice(TmpCwdMixin, unittest.TestCase):
    POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False,
              "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0, "hold_days": 1, "status": "PENDING"}

    def setUp(self):
        super().setUp()
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

    def test_entry_price_is_latest_5m_close_and_records_bar_time_and_delay(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "score": 80.0,
                        "up_probability": 60.0, "down_probability": 10.0, "top10_rank": 1,
                        "market_regime": "neutral"}]
        bars = _bars([3010.0, 3012.0, 3015.0], start="2026-09-25 10:00")
        state = dt.default_state()
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=self.POLICY), \
             patch.object(dt.fast, "scan_progressive_with_prefilter", return_value=(candidates, 225)), \
             patch.object(live_p10, "download_5m", return_value=bars):
            now = datetime(2026, 9, 25, 10, 16, tzinfo=TZ)
            msg = dt._try_entry(state, now, "2026-09-25")

        self.assertIsNotNone(msg)
        self.assertEqual(len(state["positions"]), 1)
        pos = state["positions"][0]
        self.assertEqual(pos["entry_price"], 3015.0)  # latest 5m close, NOT scan's price=3000.0
        self.assertEqual(pd.Timestamp(pos["price_bar_time"]), bars.index[-1])
        self.assertAlmostEqual(pos["data_delay_minutes"], 6.0)  # 10:16 - 10:10
        self.assertEqual(pos["shares"], 300)  # floor(1_000_000/3015/100)*100
        self.assertFalse(hasattr(pos, "validation_eligible"))  # not set on the position itself

    def test_delayed_data_records_larger_delay(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "score": 80.0,
                        "up_probability": 60.0, "down_probability": 10.0, "top10_rank": 1,
                        "market_regime": "neutral"}]
        bars = _bars([3010.0], start="2026-09-25 10:00")  # stale: only one bar at 10:00
        state = dt.default_state()
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=self.POLICY), \
             patch.object(dt.fast, "scan_progressive_with_prefilter", return_value=(candidates, 225)), \
             patch.object(live_p10, "download_5m", return_value=bars):
            now = datetime(2026, 9, 25, 10, 32, tzinfo=TZ)  # 32 minutes after the only bar
            dt._try_entry(state, now, "2026-09-25")
        self.assertAlmostEqual(state["positions"][0]["data_delay_minutes"], 32.0)


# =====================================================================
# 退出判定: price_bar_time**より後**のバーから開始、15:20強制決済、
# ヒストリー行のtrack/validation_eligible
# =====================================================================

class EvaluateExitStartsAfterPriceBarTime(TmpCwdMixin, unittest.TestCase):
    def _open_position(self, entry_price=3000.0, direction="BUY", price_bar_time="2026-09-25 09:30:00"):
        return {
            "ticker": "7203.T", "direction": direction, "entry_date": "2026-09-25", "entry_time": "09:35",
            "entry_datetime": "2026-09-25T09:35:00", "entry_price": entry_price, "shares": 300,
            "invested_amount": entry_price * 300, "tp": entry_price * 1.05, "sl": entry_price * 0.95,
            "tp_pct": 5.0, "sl_pct": -5.0, "mfe_yen": 0.0, "mae_yen": 0.0,
            "current_price": entry_price, "price_bar_time": price_bar_time,
        }

    def test_does_not_trigger_on_the_entry_bar_itself(self):
        # The price_bar_time bar's own High/Low would trip the TP if it were
        # (wrongly) included; it must be excluded (exit checks start strictly
        # AFTER price_bar_time).
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 3010.0, 2900.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3020.0], start="2026-09-25 09:30")  # the SAME bar as price_bar_time, would hit TP
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 36, tzinfo=TZ))
        self.assertEqual(msgs, [])
        self.assertEqual(len(state["positions"]), 1)  # still open

    def test_triggers_on_the_first_bar_after_price_bar_time(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 3010.0, 2900.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3020.0], start="2026-09-25 09:30")  # second bar (09:35) hits TP
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 40, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(state["positions"], [])

    def test_forced_exit_at_1520_when_no_tp_sl_hit(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 4000.0, 2000.0  # unreachable TP/SL
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        idx = pd.date_range("2026-09-25 09:35", "2026-09-25 15:25", freq="5min")
        closes = [3000.0] * len(idx)
        bars = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                              "Volume": [1000] * len(closes)}, index=idx)
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 15, 30, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        self.assertIn("FORCED_EXIT", msgs[0])
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["result"], "FORCED_EXIT")
        self.assertEqual(str(history.iloc[-1]["exit_time"]), "15:20")

    def test_forced_exit_falls_back_to_wall_clock_when_data_never_reaches_1520(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 4000.0, 2000.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3001.0], start="2026-09-25 09:35")  # data stops well before 15:20
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 15, 21, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        self.assertIn("FORCED_EXIT", msgs[0])

    def test_history_row_has_track_and_validation_eligible_false(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 3010.0, 2900.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 40, tzinfo=TZ))
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["track"], dt.TRACK)
        self.assertEqual(bool(history.iloc[-1]["validation_eligible"]), False)

    def test_same_ticker_cooldown_recorded_on_exit(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 3010.0, 2900.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 40, tzinfo=TZ))
        self.assertIn("7203.T", state["last_exit_by_ticker"])


# =====================================================================
# No carry-over: 前日からの持ち越しポジションはFORCED_LATEで即時決済
# =====================================================================

class NoCarryOverAcrossDays(TmpCwdMixin, unittest.TestCase):
    def test_leftover_position_from_previous_day_force_closed(self):
        state = dt.default_state()
        state["positions"] = [{
            "ticker": "7203.T", "direction": "BUY", "entry_date": "2026-09-24", "entry_time": "14:55",
            "entry_datetime": "2026-09-24T14:55:00", "entry_price": 3000.0, "shares": 300,
            "invested_amount": 900000.0, "tp": 3100.0, "sl": 2900.0, "tp_pct": 3.0, "sl_pct": -3.0,
            "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": 3000.0, "price_bar_time": "2026-09-24T14:55:00",
        }]
        dt.save_state(state)
        bars = _bars([3005.0], start="2026-09-25 09:00")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt.run(now=datetime(2026, 9, 25, 9, 1, tzinfo=TZ))
        reloaded = dt.load_state()
        self.assertEqual(reloaded["positions"], [])
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["result"], "FORCED_LATE")
        self.assertEqual(history.iloc[-1]["entry_date"], "2026-09-24")


# =====================================================================
# エントリー時間帯/1日上限ガード(run()レベル)
# =====================================================================

class EntryWindowAndDailyLimitGuards(TmpCwdMixin, unittest.TestCase):
    def test_no_entry_attempt_before_0930(self):
        with patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 9, 20, tzinfo=TZ))
        try_entry.assert_not_called()

    def test_no_entry_attempt_from_1450(self):
        with patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 14, 50, tzinfo=TZ))
        try_entry.assert_not_called()

    def test_entry_attempted_within_window(self):
        with patch.object(dt, "_try_entry", return_value=None) as try_entry:
            dt.run(now=datetime(2026, 9, 25, 10, 0, tzinfo=TZ))
        try_entry.assert_called_once()

    def test_no_entry_attempt_once_daily_limit_reached(self):
        state = dt.default_state()
        state["trade_date"] = "2026-09-25"
        state["trades_today"] = dt.MAX_TRADES_PER_DAY
        dt.save_state(state)
        with patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 10, 0, tzinfo=TZ))
        try_entry.assert_not_called()

    def test_no_reentry_on_the_same_tick_as_an_exit(self):
        pos = {
            "ticker": "7203.T", "direction": "BUY", "entry_date": "2026-09-25", "entry_time": "09:35",
            "entry_datetime": "2026-09-25T09:35:00", "entry_price": 3000.0, "shares": 300,
            "invested_amount": 900000.0, "tp": 3010.0, "sl": 2900.0, "tp_pct": 0.3, "sl_pct": -3.0,
            "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": 3000.0, "price_bar_time": "2026-09-25T09:30:00",
        }
        state = dt.default_state()
        state["positions"] = [pos]
        state["trade_date"] = "2026-09-25"
        dt.save_state(state)
        bars = _bars([3000.0, 3020.0], start="2026-09-25 09:30")  # triggers TP on the 09:35 bar
        with patch.object(live_p10, "download_5m", return_value=bars), \
             patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 10, 0, tzinfo=TZ))
        try_entry.assert_not_called()
        self.assertEqual(dt.load_state()["positions"], [])


if __name__ == "__main__":
    unittest.main()
