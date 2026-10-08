"""Tests for daytrade_tp5000_sl8000_paper.py (Part A: day-trade separate
+5,000/-8,000 JPY track).

This module makes real network calls if a test forgets to mock the right
entry point (yfinance, Discord), so requests.post and yfinance.download are
stubbed to raise before anything is imported (same pattern as
test_run_profit_loop_cooldown.py), and DISCORD_WEBHOOK is forced unset.
"""
import json
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
            chosen, skipped = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertIsNotNone(chosen)
        self.assertEqual((chosen["ticker"], chosen["direction"]), (live_ticker, live_direction))
        self.assertEqual(skipped, [])  # top-ranked candidate chosen directly, nothing skipped

    def test_daytrade_select_top1_matches_live_bearish_regime(self):
        with _RegimeAndFeedbackPatched("bearish"):
            top10, _ = self._top10_via_scan_candidates_fixed()
            live_ticker, live_direction = self._live_choice(top10)
            chosen, _ = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertEqual(live_direction, "SHORT")  # bearish regime must pick a SHORT
        self.assertEqual((chosen["ticker"], chosen["direction"]), (live_ticker, live_direction))

    def test_daytrade_select_top1_skips_cooldown_ticker_like_live_would_skip_active(self):
        with _RegimeAndFeedbackPatched("neutral"):
            top10, _ = self._top10_via_scan_candidates_fixed()
            now = datetime(2026, 9, 25, 9, 35, tzinfo=TZ)
            # the top-ranked ticker is still in this track's own cooldown
            top_ticker = top10[0]["ticker"]
            cooldowns = {top_ticker: (now - loop.timedelta(minutes=5)).isoformat()}
            chosen, skipped = dt._select_top1(top10, cooldowns, now)
        self.assertIsNotNone(chosen)
        self.assertNotEqual(chosen["ticker"], top_ticker)
        self.assertEqual(skipped, [(top_ticker, "cooldown")])

    def test_empty_top10_returns_none(self):
        chosen, skipped = dt._select_top1([], {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertIsNone(chosen)
        self.assertEqual(skipped, [])

    def test_brake_skips_buy_candidates_as_brake_buy_and_picks_short(self):
        now = datetime(2026, 9, 25, 10, 0, tzinfo=TZ)
        top10 = [
            {"ticker": "7203.T", "direction": "BUY", "price": 3000.0},
            {"ticker": "9984.T", "direction": "BUY", "price": 5000.0},
            {"ticker": "8035.T", "direction": "SHORT", "price": 4000.0},
        ]
        chosen, skipped = dt._select_top1(top10, {}, now, block_buy=True)
        self.assertEqual(chosen["ticker"], "8035.T")
        self.assertEqual(skipped, [("7203.T", "brake_buy"), ("9984.T", "brake_buy")])
        # ブレーキ無しなら従来どおり先頭のBUY
        chosen, skipped = dt._select_top1(top10, {}, now)
        self.assertEqual(chosen["ticker"], "7203.T")
        self.assertEqual(skipped, [])

    def test_brake_with_only_buy_candidates_returns_none(self):
        now = datetime(2026, 9, 25, 10, 0, tzinfo=TZ)
        top10 = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0}]
        chosen, skipped = dt._select_top1(top10, {}, now, block_buy=True)
        self.assertIsNone(chosen)
        self.assertEqual(skipped, [("7203.T", "brake_buy")])

    def test_unaffordable_candidate_is_skipped(self):
        with _RegimeAndFeedbackPatched("neutral"):
            top10, _ = self._top10_via_scan_candidates_fixed()
        # make every candidate's price exceed the budget for even one lot
        for c in top10:
            c["price"] = 20_000_000.0
        chosen, skipped = dt._select_top1(top10, {}, datetime(2026, 9, 25, 9, 35, tzinfo=TZ))
        self.assertIsNone(chosen)
        self.assertEqual([t for t, _ in skipped], [c["ticker"] for c in top10])
        self.assertTrue(all(r == "budget" for _, r in skipped))


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
# 約定バー自身の判定(_fill_bar_exit): openに対するギャップ判定はなし、
# High/Lowのみ。TP/SL同時タッチはSL優先。
# =====================================================================

class FillBarExit(unittest.TestCase):
    def test_buy_sl_priority_when_both_touched_in_fill_bar(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 94.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._fill_bar_exit("BUY", bar), (95.0, "SL"))

    def test_buy_tp_only(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 97.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._fill_bar_exit("BUY", bar), (105.0, "TP"))

    def test_buy_no_exit(self):
        bar = {"Open": 100.0, "High": 103.0, "Low": 98.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._fill_bar_exit("BUY", bar), (None, None))

    def test_buy_open_beyond_tp_is_not_treated_as_a_gap(self):
        # Unlike _gap_fill_exit, the fill bar's own Open is the entry price
        # itself -- it must never be compared to tp/sl as a "gap".
        bar = {"Open": 110.0, "High": 112.0, "Low": 109.0, "_tp": 105.0, "_sl": 95.0}
        self.assertEqual(dt._fill_bar_exit("BUY", bar), (105.0, "TP"))

    def test_short_sl_priority_when_both_touched_in_fill_bar(self):
        bar = {"Open": 100.0, "High": 106.0, "Low": 94.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._fill_bar_exit("SHORT", bar), (105.0, "SL"))

    def test_short_tp_only(self):
        bar = {"Open": 100.0, "High": 103.0, "Low": 94.0, "_tp": 95.0, "_sl": 105.0}
        self.assertEqual(dt._fill_bar_exit("SHORT", bar), (95.0, "TP"))


# =====================================================================
# _ceil_bar_time: 判断時刻Tの直後(start>=T)の最初の5分バー境界
# =====================================================================

class CeilBarTime(unittest.TestCase):
    def test_non_boundary_rounds_up_to_next_5min(self):
        self.assertEqual(dt._ceil_bar_time("2026-09-25 09:31:00"), pd.Timestamp("2026-09-25 09:35:00"))

    def test_mid_bar_seconds_rounds_up_to_next_5min(self):
        self.assertEqual(dt._ceil_bar_time("2026-09-25 09:35:20"), pd.Timestamp("2026-09-25 09:40:00"))

    def test_exact_boundary_maps_to_itself(self):
        self.assertEqual(dt._ceil_bar_time("2026-09-25 09:35:00"), pd.Timestamp("2026-09-25 09:35:00"))


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
# ポリシー選択(C案): daily_decision.jsonの当日判断だけを使う
# =====================================================================

def _decision(**kw):
    d = {"date": datetime.now(TZ).strftime("%Y-%m-%d"), "trend": "up",
         "trend_data_as_of": "2026-10-07", "policy_file": "strategy_policy_up.json",
         "policy_hash": "x", "entry_allowed": True, "entry_block_reason": None,
         "shadow_policy_file": None, "intraday_crash_brake": False}
    d.update(kw)
    return d


class PolicyChoiceFromDailyDecision(unittest.TestCase):
    def test_uses_decision_policy_and_records_data_as_of(self):
        with patch.object(dt.daily_decision, "update_crash_brake", return_value=_decision()), \
             patch.object(dt.daily_decision, "entry_status", return_value=(True, None)), \
             patch.object(live_p10, "select_policy_file") as sp:
            policy_file, source = dt.choose_policy_file_reusing_live_tick(now=datetime.now(TZ))
        sp.assert_not_called()
        self.assertEqual(policy_file, "strategy_policy_up.json")
        self.assertEqual(source, "daily_decision:2026-10-07")

    def test_blocked_day_returns_none_never_falls_back_to_generic_policy(self):
        d = _decision(trend="down", policy_file=None, entry_allowed=False,
                      entry_block_reason="no_approved_policy_for_down",
                      shadow_policy_file="strategy_policy.json")
        with patch.object(dt.daily_decision, "update_crash_brake", return_value=d):
            policy_file, source = dt.choose_policy_file_reusing_live_tick(now=datetime.now(TZ))
        self.assertIsNone(policy_file)
        self.assertEqual(source, "blocked:no_approved_policy_for_down")

    def test_crash_brake_does_not_block_policy_choice(self):
        # 急落ブレーキは新規の買いだけを止める(_select_top1でbrake_buyスキップ)。
        # policy選択自体は止めないので、空売りは同じpolicyで継続できる。
        with patch.object(dt.daily_decision, "update_crash_brake",
                          return_value=_decision(intraday_crash_brake=True, policy_hash=None)), \
             patch.object(dt.daily_decision, "entry_status", return_value=(True, None)):
            policy_file, source = dt.choose_policy_file_reusing_live_tick(now=datetime.now(TZ))
        self.assertEqual(policy_file, "strategy_policy_up.json")
        self.assertEqual(source, "daily_decision:2026-10-07")

    def test_down_day_fallback_policy_is_used_for_real_entries(self):
        d = _decision(trend="down", policy_file="strategy_policy.json", policy_fallback=True)
        with patch.object(dt.daily_decision, "update_crash_brake", return_value=d), \
             patch.object(dt.daily_decision, "entry_status", return_value=(True, None)):
            policy_file, source = dt.choose_policy_file_reusing_live_tick(now=datetime.now(TZ))
        self.assertEqual(policy_file, "strategy_policy.json")
        self.assertEqual(source, "daily_decision:2026-10-07")

    def test_try_entry_stops_before_loading_any_policy_when_blocked(self):
        state = dt.default_state() if hasattr(dt, "default_state") else {"positions": [], "pending": None}
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                          return_value=(None, "blocked:policy_changed_since_decision")), \
             patch.object(live_p10, "load_policy") as lp, \
             patch.object(dt, "_record_daytrade_shadow") as shadow:
            out = dt._try_entry(state, datetime.now(TZ), datetime.now(TZ).strftime("%Y-%m-%d"))
        self.assertIsNone(out)
        lp.assert_not_called()
        shadow.assert_not_called()  # policy差替え検知による停止はシャドー対象外

    def test_try_entry_records_shadow_on_no_policy_day(self):
        state = {"positions": [], "pending": None}
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                          return_value=(None, "blocked:no_approved_policy_for_down")), \
             patch.object(live_p10, "load_policy") as lp, \
             patch.object(dt, "_record_daytrade_shadow") as shadow:
            out = dt._try_entry(state, datetime.now(TZ), datetime.now(TZ).strftime("%Y-%m-%d"))
        self.assertIsNone(out)
        lp.assert_not_called()
        shadow.assert_called_once()
        self.assertIsNone(state["pending"])


# =====================================================================
# エントリー判断: 即時約定しない。PENDINGを作成し、decision_time/
# fill_bar_timeを記録する(_try_entry)
# =====================================================================

class EntryCreatesPendingNotImmediateFill(TmpCwdMixin, unittest.TestCase):
    POLICY = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": False,
              "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0, "hold_days": 1, "status": "PENDING"}

    def setUp(self):
        super().setUp()
        with open("strategy_policy.json", "w", encoding="utf-8") as f:
            f.write('{"status": "PENDING"}')

    def _write_live_cache(self, now, raw=None):
        """Stands in for the live tick having just written
        scan_candidates_cache.json this same iteration (Change A reads it
        directly instead of scanning on its own)."""
        raw = raw if raw is not None else [{"ticker": "placeholder"}]
        payload = {"timestamp": now.timestamp(), "raw": raw, "scanned": 225}
        with open(dt.fast.SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _try_entry(self, now, candidates=None):
        candidates = candidates or [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "score": 80.0,
                                      "up_probability": 60.0, "down_probability": 10.0, "top10_rank": 1,
                                      "market_regime": "neutral"}]
        self._write_live_cache(now)
        state = dt.default_state()
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=self.POLICY), \
             patch.object(dt.fast, "scan_progressive_with_prefilter", return_value=(candidates, 225)):
            msg = dt._try_entry(state, now, now.strftime("%Y-%m-%d"))
        return state, msg

    def test_decision_creates_pending_not_a_position(self):
        now = datetime(2026, 9, 25, 9, 31, tzinfo=TZ)
        state, msg = self._try_entry(now)
        self.assertIsNotNone(msg)
        self.assertEqual(state["positions"], [])  # no immediate fill
        pending = state["pending"]
        self.assertIsNotNone(pending)
        self.assertEqual(pending["ticker"], "7203.T")
        self.assertEqual(pending["direction"], "BUY")
        self.assertEqual(pd.Timestamp(pending["decision_time"]), pd.Timestamp("2026-09-25 09:31:00"))
        # ceil(09:31) -> 09:35 bar, per _ceil_bar_time()
        self.assertEqual(pd.Timestamp(pending["fill_bar_time"]), pd.Timestamp("2026-09-25 09:35:00"))
        self.assertEqual(pending["decision_date"], "2026-09-25")
        self.assertEqual(pending["top10_source"], dt.TOP10_SOURCE_LIVE_TICK_CACHE)
        self.assertEqual(pending["skipped"], "")
        self.assertEqual(pending["entry_window_start"], "09:55")

    def test_decision_exactly_on_5min_boundary_targets_same_bar(self):
        now = datetime(2026, 9, 25, 9, 30, tzinfo=TZ)
        self._write_live_cache(now)
        state = dt.default_state()
        with patch.object(dt, "choose_policy_file_reusing_live_tick",
                           return_value=("strategy_policy.json", "reused_today_row")), \
             patch.object(live_p10, "load_policy", return_value=self.POLICY), \
             patch.object(dt.fast, "scan_progressive_with_prefilter",
                           return_value=([{"ticker": "7203.T", "direction": "BUY", "price": 3000.0,
                                            "score": 80.0, "up_probability": 60.0, "down_probability": 10.0,
                                            "top10_rank": 1, "market_regime": "neutral"}], 225)):
            dt._try_entry(state, now, "2026-09-25")
        self.assertEqual(pd.Timestamp(state["pending"]["fill_bar_time"]), pd.Timestamp("2026-09-25 09:30:00"))


# =====================================================================
# PENDING約定確認(_check_pending_fill): fill_bar_timeのバーがデータに
# 現れた「後のtick」で約定し、約定価格(Open)から株数/TP/SLを再計算する
# =====================================================================

class PendingFillConfirmation(TmpCwdMixin, unittest.TestCase):
    def _pending(self, decision_time="2026-09-25 09:31:00", direction="BUY", ticker="7203.T", budget=1_000_000.0):
        decision_time = pd.Timestamp(decision_time)
        return {
            "ticker": ticker, "direction": direction,
            "decision_date": "2026-09-25", "decision_time": decision_time.isoformat(),
            "fill_bar_time": dt._ceil_bar_time(decision_time).isoformat(),
            "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "market_regime": "neutral",
            "top10_rank": 1, "policy_file": "strategy_policy.json", "policy_source": "reused_today_row",
            "policy_hash": "abc123", "model_id": None, "model_version": "legacy-20260912",
            "budget": budget, "entry_window_start": "09:55",
        }

    def test_fill_bar_not_yet_available_stays_pending(self):
        state = dt.default_state()
        state["pending"] = self._pending()
        bars = _bars([3010.0], start="2026-09-25 09:30")  # only the 09:30 bar so far
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 36, tzinfo=TZ), "2026-09-25")
        self.assertEqual(msgs, [])
        self.assertIsNotNone(state["pending"])  # still pending
        self.assertEqual(state["positions"], [])

    def test_fills_on_a_later_tick_once_fill_bar_appears(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 09:31:00")  # fill_bar_time = 09:35
        bars = _bars([3010.0, 3020.0], start="2026-09-25 09:30")  # 09:30, 09:35 bars now present
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ), "2026-09-25")
        self.assertEqual(len(msgs), 1)
        self.assertIsNone(state["pending"])
        self.assertEqual(len(state["positions"]), 1)
        pos = state["positions"][0]
        self.assertEqual(pos["entry_price"], 3020.0)  # Open of the 09:35 bar
        self.assertEqual(pd.Timestamp(pos["price_bar_time"]), pd.Timestamp("2026-09-25 09:35:00"))
        self.assertEqual(pd.Timestamp(pos["fill_bar_time"]), pd.Timestamp("2026-09-25 09:35:00"))
        self.assertEqual(pd.Timestamp(pos["decision_time"]), pd.Timestamp("2026-09-25 09:31:00"))
        self.assertEqual(pos["fill_method"], dt.FILL_METHOD_V2)
        self.assertEqual(pos["entry_datetime"], pd.Timestamp("2026-09-25 09:35:00").isoformat())
        self.assertEqual(state["trades_today"], 1)
        self.assertEqual(pos["entry_window_start"], "09:55")  # carried from the pending row

    def test_shares_are_recomputed_from_fill_price_not_scan_price(self):
        # fill price (3020, bar Open) differs from whatever the original
        # scan/decision price was -- shares must come from the fill price.
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 09:31:00", budget=1_000_000.0)
        bars = _bars([3010.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ), "2026-09-25")
        pos = state["positions"][0]
        self.assertEqual(pos["shares"], 300)  # floor(1_000_000 / 3020 / 100) * 100

    def test_tp_sl_solved_from_fill_price(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 09:31:00")
        bars = _bars([3010.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ), "2026-09-25")
        pos = state["positions"][0]
        self.assertAlmostEqual(dt.net_pnl(pos["entry_price"], pos["tp"], pos["shares"], "BUY"), dt.TP_NET_JPY, places=4)
        self.assertAlmostEqual(dt.net_pnl(pos["entry_price"], pos["sl"], pos["shares"], "BUY"), dt.SL_NET_JPY, places=4)

    def test_cancelled_when_100_shares_exceed_budget(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 09:31:00", budget=1_000_000.0)
        # Open of the fill bar is far above what 1,000,000 JPY can buy even one lot of
        bars = _bars([3010.0, 20_000_000.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ), "2026-09-25")
        self.assertEqual(msgs, [])
        self.assertIsNone(state["pending"])
        self.assertEqual(state["positions"], [])
        log = pd.read_csv(dt.CANCELLED_LOG_FILE)
        self.assertEqual(log.iloc[-1]["reason"], "CANCELLED_NO_FILL_BUDGET")
        self.assertFalse(os.path.exists(dt.HISTORY_FILE))  # never a trade row

    def test_cancelled_when_fill_bar_would_start_at_or_after_1520(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 15:17:00")  # ceil -> 15:20 bar
        with patch.object(live_p10, "download_5m") as dl:
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 15, 18, tzinfo=TZ), "2026-09-25")
        dl.assert_not_called()  # cancelled before even trying to fetch data
        self.assertEqual(msgs, [])
        self.assertIsNone(state["pending"])
        log = pd.read_csv(dt.CANCELLED_LOG_FILE)
        self.assertEqual(log.iloc[-1]["reason"], "CANCELLED_NO_FILL_TOO_LATE")

    def test_cancelled_when_data_never_arrives_by_1520(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 14:49:00")  # fill_bar_time=14:50
        bars = _bars([3010.0], start="2026-09-25 09:30")  # data stalled, 14:50 bar never shows up
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 15, 20, tzinfo=TZ), "2026-09-25")
        self.assertEqual(msgs, [])
        self.assertIsNone(state["pending"])
        log = pd.read_csv(dt.CANCELLED_LOG_FILE)
        self.assertEqual(log.iloc[-1]["reason"], "CANCELLED_NO_FILL_DATA_DELAY")

    def test_still_waiting_before_1520_even_without_data_is_not_cancelled(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 14:49:00")
        bars = _bars([3010.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 25, 15, 19, tzinfo=TZ), "2026-09-25")
        self.assertEqual(msgs, [])
        self.assertIsNotNone(state["pending"])  # not yet 15:20 -> still waiting

    def test_cancelled_on_new_trading_day(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 14:49:00")
        with patch.object(live_p10, "download_5m") as dl:
            msgs = dt._check_pending_fill(state, datetime(2026, 9, 26, 9, 5, tzinfo=TZ), "2026-09-26")
        dl.assert_not_called()
        self.assertEqual(msgs, [])
        self.assertIsNone(state["pending"])
        log = pd.read_csv(dt.CANCELLED_LOG_FILE)
        self.assertEqual(log.iloc[-1]["reason"], "CANCELLED_NO_FILL_NEW_DAY")

    def test_fill_bar_invariant_holds(self):
        state = dt.default_state()
        state["pending"] = self._pending(decision_time="2026-09-25 09:31:00")
        bars = _bars([3010.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._check_pending_fill(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ), "2026-09-25")
        pos = state["positions"][0]
        self.assertGreaterEqual(pd.Timestamp(pos["fill_bar_time"]), pd.Timestamp(pos["decision_time"]))


# =====================================================================
# 退出判定(レガシーv1, fill_methodなし): price_bar_time**より後**のバー
# から開始、15:20強制決済、ヒストリー行のtrack/validation_eligible。
# これは _evaluate_exit() が fill_method のないポジション(=このfix以前
# に開いた実ポジション)を旧ロジックのまま動かし続けることの検証でもある
# (後方互換パス)。
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

    def test_position_has_no_fill_method_and_uses_legacy_path(self):
        pos = self._open_position()
        self.assertIsNone(pos.get("fill_method"))  # legacy shape: no fill_method key at all

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
        self.assertTrue(pd.isna(history.iloc[-1]["fill_method"]))  # legacy row: no fill_method

    def test_same_ticker_cooldown_recorded_on_exit(self):
        pos = self._open_position(entry_price=3000.0, price_bar_time="2026-09-25 09:30:00")
        pos["tp"], pos["sl"] = 3010.0, 2900.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3020.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 40, tzinfo=TZ))
        self.assertIn("7203.T", state["last_exit_by_ticker"])


# =====================================================================
# 退出判定(v2, fill_method=decision_next_bar_open_v2): 約定バー自身
# (price_bar_time==fill_bar_time)はopenへのギャップ判定なしで
# High/Lowのみ判定(SL優先)、次のバーからは通常のギャップ判定。
# =====================================================================

class EvaluateExitV2(TmpCwdMixin, unittest.TestCase):
    def _v2_position(self, entry_price=3000.0, direction="BUY", fill_bar_time="2026-09-25 09:35:00",
                      decision_time="2026-09-25 09:31:00", shares=300):
        fill_bar_time = pd.Timestamp(fill_bar_time)
        return {
            "ticker": "7203.T", "direction": direction, "entry_date": "2026-09-25",
            "entry_time": fill_bar_time.strftime("%H:%M"), "entry_datetime": fill_bar_time.isoformat(),
            "entry_price": entry_price, "shares": shares, "invested_amount": entry_price * shares,
            "tp": entry_price * 1.05, "sl": entry_price * 0.95, "tp_pct": 5.0, "sl_pct": -5.0,
            "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": entry_price,
            "price_bar_time": fill_bar_time.isoformat(), "fill_bar_time": fill_bar_time.isoformat(),
            "decision_time": pd.Timestamp(decision_time).isoformat(), "fill_method": dt.FILL_METHOD_V2,
        }

    def test_fill_bar_sl_priority_when_both_touched(self):
        pos = self._v2_position(entry_price=3000.0, fill_bar_time="2026-09-25 09:35:00")
        pos["tp"], pos["sl"] = 3050.0, 2950.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        # the fill bar itself touches both tp and sl -> SL must win, and the
        # bar's Open (3000, far from either) must never be checked as a gap
        bars = _bars([3000.0, 3000.0], start="2026-09-25 09:30")
        bars.loc[pd.Timestamp("2026-09-25 09:35:00"), ["Open", "High", "Low"]] = [3000.0, 3060.0, 2940.0]
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["result"], "SL")
        self.assertEqual(history.iloc[-1]["exit_time"], "09:35")  # fires on the fill bar, not later
        self.assertEqual(history.iloc[-1]["fill_method"], dt.FILL_METHOD_V2)

    def test_no_exit_on_fill_bar_moves_to_gap_rule_on_next_bar(self):
        pos = self._v2_position(entry_price=3000.0, fill_bar_time="2026-09-25 09:35:00")
        pos["tp"], pos["sl"] = 3050.0, 2950.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3000.0, 3060.0], start="2026-09-25 09:30")  # 09:40 bar gaps up through TP
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["result"], "TP")
        self.assertEqual(history.iloc[-1]["exit_time"], "09:40")

    def test_forced_exit_at_1520(self):
        pos = self._v2_position(entry_price=3000.0, fill_bar_time="2026-09-25 09:35:00")
        pos["tp"], pos["sl"] = 4000.0, 2000.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        idx = pd.date_range("2026-09-25 09:35", "2026-09-25 15:25", freq="5min")
        closes = [3000.0] * len(idx)
        bars = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                              "Volume": [1000] * len(idx)}, index=idx)
        with patch.object(live_p10, "download_5m", return_value=bars):
            msgs = dt._evaluate_exit(state, datetime(2026, 9, 25, 15, 30, tzinfo=TZ))
        self.assertEqual(len(msgs), 1)
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["result"], "FORCED_EXIT")
        self.assertEqual(history.iloc[-1]["exit_time"], "15:20")

    def test_hold_minutes_computed_from_fill_time_not_decision_time(self):
        pos = self._v2_position(entry_price=3000.0, fill_bar_time="2026-09-25 09:35:00",
                                 decision_time="2026-09-25 09:31:00")
        pos["tp"], pos["sl"] = 3050.0, 2950.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        bars = _bars([3000.0, 3000.0, 3060.0], start="2026-09-25 09:30")
        with patch.object(live_p10, "download_5m", return_value=bars):
            dt._evaluate_exit(state, datetime(2026, 9, 25, 9, 51, tzinfo=TZ))
        history = pd.read_csv(dt.HISTORY_FILE)
        # exit at 09:40, fill (entry_datetime) at 09:35 -> 5 minutes, NOT from decision_time 09:31
        self.assertAlmostEqual(float(history.iloc[-1]["hold_minutes"]), 5.0)

    def test_invariant_violation_raises_assertion_error(self):
        # decision_time AFTER fill_bar_time must never happen; _close_position
        # asserts the invariant rather than silently writing a bad row.
        pos = self._v2_position(entry_price=3000.0, fill_bar_time="2026-09-25 09:35:00",
                                 decision_time="2026-09-25 09:40:00")  # decision after fill -- invalid
        pos["tp"], pos["sl"] = 3050.0, 2950.0
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        with self.assertRaises(AssertionError):
            dt._close_position(state, datetime(2026, 9, 25, 9, 45, tzinfo=TZ), 3010.0, "TP",
                                pd.Timestamp("2026-09-25 09:45:00"))


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
    def test_no_entry_attempt_before_0955(self):
        for hour, minute, second in [(9, 20, 0), (9, 30, 0), (9, 50, 0), (9, 54, 59)]:
            with self.subTest(time=(hour, minute, second)), \
                 patch.object(dt, "_try_entry") as try_entry:
                dt.run(now=datetime(2026, 9, 25, hour, minute, second, tzinfo=TZ))
            try_entry.assert_not_called()

    def test_entry_attempted_exactly_at_0955(self):
        with patch.object(dt, "_try_entry", return_value=None) as try_entry:
            dt.run(now=datetime(2026, 9, 25, 9, 55, 0, tzinfo=TZ))
        try_entry.assert_called_once()

    def test_no_entry_attempt_from_1450(self):
        with patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 14, 50, tzinfo=TZ))
        try_entry.assert_not_called()

    def test_entry_attempted_at_1449(self):
        with patch.object(dt, "_try_entry", return_value=None) as try_entry:
            dt.run(now=datetime(2026, 9, 25, 14, 49, tzinfo=TZ))
        try_entry.assert_called_once()

    def test_entry_attempted_within_window(self):
        with patch.object(dt, "_try_entry", return_value=None) as try_entry:
            dt.run(now=datetime(2026, 9, 25, 10, 0, tzinfo=TZ))
        try_entry.assert_called_once()

    def test_carried_pending_still_evaluated_before_0955(self):
        # the entry gate blocks NEW decisions before 09:55, but a pending
        # decision already created earlier (or a cancellation/day-reset)
        # must still be evaluated on every tick regardless of the window.
        state = dt.default_state()
        state["pending"] = {
            "ticker": "7203.T", "direction": "BUY",
            "decision_date": "2026-09-24", "decision_time": "2026-09-24T14:49:00",
            "fill_bar_time": "2026-09-24T14:50:00", "score": 80.0, "up_probability": 60.0,
            "down_probability": 10.0, "market_regime": "neutral", "top10_rank": 1,
            "top10_source": dt.TOP10_SOURCE_LIVE_TICK_CACHE, "skipped": "",
            "policy_file": "strategy_policy.json", "policy_source": "reused_today_row",
            "policy_hash": "abc123", "model_id": None, "model_version": "legacy-20260912",
            "budget": 1_000_000.0, "entry_window_start": "09:30",
        }
        state["trade_date"] = "2026-09-25"
        dt.save_state(state)
        with patch.object(live_p10, "download_5m") as dl, \
             patch.object(dt, "_try_entry") as try_entry:
            dt.run(now=datetime(2026, 9, 25, 9, 40, tzinfo=TZ))
        dl.assert_not_called()  # cancelled on day-mismatch before even trying to fetch data
        try_entry.assert_not_called()  # still before 09:55 -- no new decision
        self.assertIsNone(dt.load_state()["pending"])
        log = pd.read_csv(dt.CANCELLED_LOG_FILE)
        self.assertEqual(log.iloc[-1]["reason"], "CANCELLED_NO_FILL_NEW_DAY")

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


# =====================================================================
# entry_window_start: 09:55の由来を記録する新しい末尾カラム。
# 既存の過去行は書き換えず、空値のまま(docstringで09:30と定義)。
# =====================================================================

class EntryWindowStartProvenance(TmpCwdMixin, unittest.TestCase):
    def test_history_row_carries_entry_window_start(self):
        pos = {
            "ticker": "7203.T", "direction": "BUY", "entry_date": "2026-09-25",
            "entry_time": "09:55", "entry_datetime": "2026-09-25T09:55:00",
            "entry_price": 3000.0, "shares": 300, "invested_amount": 900000.0,
            "tp": 3010.0, "sl": 2900.0, "tp_pct": 0.3, "sl_pct": -3.0,
            "mfe_yen": 0.0, "mae_yen": 0.0, "current_price": 3000.0,
            "price_bar_time": "2026-09-25T09:55:00", "fill_bar_time": "2026-09-25T09:55:00",
            "decision_time": "2026-09-25T09:55:00", "fill_method": dt.FILL_METHOD_V2,
            "entry_window_start": dt.ENTRY_WINDOW_START_STR,
        }
        state = {"positions": [pos], "last_exit_by_ticker": {}}
        dt._close_position(state, datetime(2026, 9, 25, 10, 0, tzinfo=TZ), 3010.0, "TP",
                            pd.Timestamp("2026-09-25 10:00:00"))
        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(history.iloc[-1]["entry_window_start"], "09:55")

    def test_old_history_rows_unchanged_after_entry_window_start_column_appended(self):
        # simulate a pre-existing history file written before this field
        # existed (no entry_window_start column at all).
        old_row = {"ticker": "9984.T", "direction": "BUY", "entry_price": 5000.0,
                   "exit_price": 5050.0, "result": "TP", "pnl": 1234.0,
                   "track": dt.TRACK, "validation_eligible": False}
        pd.DataFrame([old_row]).to_csv(dt.HISTORY_FILE, index=False, encoding="utf-8-sig")

        new_row = dict(old_row)
        new_row["ticker"] = "7203.T"
        new_row["entry_window_start"] = "09:55"
        dt.append_history(new_row)

        history = pd.read_csv(dt.HISTORY_FILE)
        self.assertEqual(len(history), 2)
        # old row's original columns/values are untouched...
        for key, value in old_row.items():
            self.assertEqual(history.iloc[0][key], value)
        # ...and its new entry_window_start cell is empty (defined as '09:30').
        self.assertTrue(pd.isna(history.iloc[0]["entry_window_start"]))
        self.assertEqual(history.iloc[1]["entry_window_start"], "09:55")
        self.assertEqual(list(history.columns)[-1], "entry_window_start")


if __name__ == "__main__":
    unittest.main()
