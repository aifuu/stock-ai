"""daily_decision.py(C案)のテスト。ネットワークは一切使わない。

2026-10-08の事象の再現を含む:
  朝の確定判定は「上昇」だったが、場中に先物が前日比-1.5%を割ったtickで
  旧実装は当日の未確定足込みで再判定→「下落」→down用policy無し→
  汎用strategy_policy.jsonへフォールバックして未検証条件で売買していた。
"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402

requests.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: requests.post"))

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network: yfinance.download"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import daily_decision as dd  # noqa: E402
import futures_trend as ft  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")
REPO = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime(2026, 10, 8, 8, 30, tzinfo=TZ)


def _bars(last_confirmed_day="2026-10-07", n=40, today_change=None, trend="up"):
    """確定足n本(最終日=last_confirmed_day)+任意で当日の未確定足。"""
    idx = pd.bdate_range(end=last_confirmed_day, periods=n)
    base = np.linspace(60000, 70000, n) if trend == "up" else np.linspace(70000, 60000, n)
    d = pd.DataFrame({"Close": base}, index=idx)
    if today_change is not None:
        d.loc[pd.Timestamp("2026-10-08")] = base[-1] * (1 + today_change)
    return d


class _TmpRepo(unittest.TestCase):
    """policyファイルを置いた一時ディレクトリで実行(リポジトリを汚さない)。"""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = tempfile.mkdtemp()
        for f in ("strategy_policy.json", "strategy_policy_up.json"):
            shutil.copy(os.path.join(REPO, f), os.path.join(self.dir, f))
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.dir, ignore_errors=True)

    def read_decision(self):
        with open(dd.DECISION_FILE, encoding="utf-8") as f:
            return json.load(f)


class ConfirmedCloseOnly(unittest.TestCase):
    def test_todays_partial_bar_is_ignored(self):
        # 当日足が-2%(旧実装なら急落判定でdown)でも、確定足だけなら上昇のまま。
        r = ft.confirmed_trend(TODAY.date(), downloader=lambda *a, **k: _bars(today_change=-0.02))
        self.assertEqual(r["trend"], ft.UP)
        self.assertEqual(r["trend_data_as_of"], "2026-10-07")

    def test_old_detector_would_have_flipped_on_same_data(self):
        # 回帰の記録: 旧detect_futures_trend()は同じデータで下落に反転していた。
        with patch.object(ft, "_download", return_value=_bars(today_change=-0.02)):
            self.assertEqual(ft.detect_futures_trend()["trend"], ft.DOWN)

    def test_live_and_backtest_share_one_formula(self):
        # 本番(当日D)の判定 == 検証系列のD-1の値(候補日D-1→翌営業日Dに建てる対応付け)。
        bars = _bars(n=60)
        live = ft.confirmed_trend(TODAY.date(), downloader=lambda *a, **k: bars)
        series = ft.classify_close_series(bars["Close"])
        self.assertEqual(live["trend"], series["trend"].iloc[-1])

    def test_confirmed_crash_day_still_forces_down(self):
        bars = _bars()
        bars.iloc[-1, 0] = bars.iloc[-2, 0] * 0.98  # 10/7確定終値が-2%
        r = ft.confirmed_trend(TODAY.date(), downloader=lambda *a, **k: bars)
        self.assertEqual(r["trend"], ft.DOWN)

    def test_no_data_is_fail_closed(self):
        r = ft.confirmed_trend(TODAY.date(), downloader=lambda *a, **k: None)
        self.assertEqual(r["source"], "unavailable")


class DecisionFileContract(_TmpRepo):
    def test_audit_fields_and_values_for_up_day(self):
        with patch.object(ft, "_download", return_value=_bars()):
            d = dd.ensure_decision(TODAY)
        for k in ("date", "decision_time", "trend", "policy_file", "policy_hash",
                  "trend_data_as_of", "entry_allowed", "intraday_crash_brake"):
            self.assertIn(k, d)
        self.assertEqual(d["date"], "2026-10-08")
        self.assertEqual(d["decision_time"], "2026-10-08T08:30:00+09:00")
        self.assertEqual(d["trend_data_as_of"], "2026-10-07")
        self.assertEqual(d["policy_file"], "strategy_policy_up.json")
        self.assertEqual(d["policy_hash"], dd.policy_hash("strategy_policy_up.json"))
        self.assertTrue(d["entry_allowed"])
        self.assertFalse(d["intraday_crash_brake"])
        self.assertEqual(self.read_decision(), d)

    def test_built_once_per_day_never_rebuilt(self):
        with patch.object(ft, "_download", return_value=_bars(trend="up")):
            first = dd.ensure_decision(TODAY)
        later = TODAY.replace(hour=11, minute=0)
        with patch.object(ft, "_download", return_value=_bars(trend="down")) as dl:
            again = dd.ensure_decision(later)
        dl.assert_not_called()
        self.assertEqual(again, first)

    def test_yesterdays_file_is_replaced_by_new_day(self):
        with patch.object(ft, "_download", return_value=_bars(last_confirmed_day="2026-10-06")):
            dd.ensure_decision(TODAY.replace(day=7))
        with patch.object(ft, "_download", return_value=_bars()):
            d = dd.ensure_decision(TODAY)
        self.assertEqual(d["date"], "2026-10-08")
        self.assertEqual(d["trend_data_as_of"], "2026-10-07")

    def test_down_day_without_down_policy_blocks_and_sets_shadow(self):
        with patch.object(ft, "_download", return_value=_bars(trend="down")):
            d = dd.ensure_decision(TODAY)
        self.assertEqual(d["trend"], ft.DOWN)
        self.assertIsNone(d["policy_file"])
        self.assertFalse(d["entry_allowed"])
        self.assertEqual(d["entry_block_reason"], "no_approved_policy_for_down")
        self.assertEqual(d["shadow_policy_file"], "strategy_policy.json")
        self.assertEqual(dd.entry_status(d), (False, "no_approved_policy_for_down"))

    def test_down_day_uses_down_policy_once_it_exists(self):
        shutil.copy("strategy_policy_up.json", "strategy_policy_down.json")
        with patch.object(ft, "_download", return_value=_bars(trend="down")):
            d = dd.ensure_decision(TODAY)
        self.assertEqual(d["policy_file"], "strategy_policy_down.json")
        self.assertTrue(d["entry_allowed"])

    def test_policy_changed_after_decision_blocks(self):
        with patch.object(ft, "_download", return_value=_bars()):
            d = dd.ensure_decision(TODAY)
        with open("strategy_policy_up.json", "a", encoding="utf-8") as f:
            f.write(" ")
        self.assertEqual(dd.entry_status(d), (False, "policy_changed_since_decision"))


class CrashBrake(_TmpRepo):
    def setUp(self):
        super().setUp()
        with patch.object(ft, "_download", return_value=_bars()):
            self.d = dd.ensure_decision(TODAY)
        self.base = self.d["last_confirmed_close"]

    def test_small_dip_does_not_brake(self):
        d = dd.update_crash_brake(TODAY.replace(hour=10), price_fetcher=lambda *a: self.base * 0.991)
        self.assertFalse(d["intraday_crash_brake"])
        self.assertEqual(dd.entry_status(d), (True, None))

    def test_crash_brakes_without_changing_policy_and_is_sticky(self):
        d = dd.update_crash_brake(TODAY.replace(hour=9, minute=51), price_fetcher=lambda *a: self.base * 0.984)
        self.assertTrue(d["intraday_crash_brake"])
        self.assertEqual(d["policy_file"], "strategy_policy_up.json")  # policyは変えない
        self.assertEqual(d["trend"], ft.UP)
        self.assertEqual(dd.entry_status(d), (False, "intraday_crash_brake"))
        # 反発しても当日中は解除しない
        d2 = dd.update_crash_brake(TODAY.replace(hour=13), price_fetcher=lambda *a: self.base * 1.01)
        self.assertTrue(d2["intraday_crash_brake"])
        self.assertTrue(self.read_decision()["intraday_crash_brake"])
        self.assertIn("crash_brake_time", d2)

    def test_price_fetch_failure_keeps_state(self):
        def boom(*a):
            raise RuntimeError("net down")
        d = dd.update_crash_brake(TODAY.replace(hour=10), price_fetcher=boom)
        self.assertFalse(d["intraday_crash_brake"])


class Shadow(_TmpRepo):
    def test_shadow_is_deduped_per_ticker_per_day_and_track(self):
        with patch.object(ft, "_download", return_value=_bars(trend="down")):
            d = dd.ensure_decision(TODAY)
        c = {"ticker": "6472.T", "price": 380.5, "score": 42.5}
        now = TODAY.replace(hour=9, minute=51)
        self.assertTrue(dd.record_shadow(d, c, now, track="profit_top10"))
        self.assertFalse(dd.record_shadow(d, c, now, track="profit_top10"))
        self.assertTrue(dd.record_shadow(d, c, now, track="daytrade_tp5000_sl8000"))
        rows = pd.read_csv(dd.SHADOW_FILE, encoding="utf-8-sig")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows.iloc[0]["trend_data_as_of"], "2026-10-07")
        self.assertEqual(rows.iloc[0]["entry_block_reason"], "no_approved_policy_for_down")


class Incident20261008Replay(_TmpRepo):
    """両トラックが同じ判断を読み、場中急落で未検証policyへ飛ばないことを通しで確認。"""

    def test_both_tracks_agree_and_crash_never_switches_policy(self):
        import profit_top10_paper as live
        import daytrade_tp5000_sl8000_paper as dt

        with patch.object(ft, "_download", return_value=_bars()):
            dd.ensure_decision(TODAY)

        # 09:51: 場中で先物-1.6%(旧実装ではここでstrategy_policy.jsonに飛んだ)
        crash_now = TODAY.replace(hour=9, minute=51)
        intraday = _bars(today_change=-0.016)
        with patch.object(ft, "_download", return_value=intraday), \
             patch("daily_decision.datetime") as mdt:
            mdt.now.return_value = crash_now
            live_file, live_result = live.select_policy_file()
            dt_file, dt_source = dt.choose_policy_file_reusing_live_tick(now=crash_now)
            d = dd.update_crash_brake(crash_now)

        # 判断は朝の「上昇」のまま固定、両トラック同じpolicy
        self.assertEqual(live_result["trend"], ft.UP)
        self.assertEqual(live_file, "strategy_policy_up.json")
        self.assertNotEqual(live_file, "strategy_policy.json")
        # 急落はブレーキとして効き、どちらのトラックも新規しない
        self.assertTrue(d["intraday_crash_brake"])
        self.assertIsNone(dt_file)
        self.assertEqual(dt_source, "blocked:intraday_crash_brake")
        self.assertEqual(dd.entry_status(d), (False, "intraday_crash_brake"))


class MainTrackRunGate(unittest.TestCase):
    """profit_top10_paper._run(): 判断に応じて新規を止め、決済は続ける。"""

    def _run(self, decision, brake_decision=None):
        import profit_top10_paper as live
        now = TODAY.replace(hour=10)
        state = {"capital": 1e6, "peak": 1e6, "max_dd": 0.0, "positions": [],
                 "trades_today": 0, "trades_by_ticker_today": {}, "daily_start_capital": 1e6}
        cand = {"ticker": "6472.T", "score": 42.5, "direction": "BUY", "price": 380.5}
        policy = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "atr_tp_multiplier": 3.0,
                  "atr_sl_multiplier": 1.0, "nikkei_filter": True, "hold_days": 3}
        with patch.object(live, "datetime") as mdt, \
             patch.object(live, "is_tse_trading_day", return_value=True), \
             patch.object(live.daily_decision, "ensure_decision", return_value=decision), \
             patch.object(live.daily_decision, "update_crash_brake", return_value=brake_decision or decision), \
             patch.object(live.daily_decision, "entry_status",
                          side_effect=lambda d: (bool(d.get("entry_allowed")) and not d.get("intraday_crash_brake"),
                                                 "x")), \
             patch.object(live.daily_decision, "record_shadow") as shadow, \
             patch.object(live, "load_policy", return_value=policy), \
             patch.object(live, "load_state", return_value=state), \
             patch.object(live, "save_state"), \
             patch.object(live, "mark_and_close", return_value=[]) as close, \
             patch.object(live, "scan", return_value=([cand], 225)), \
             patch.object(live, "open_positions", return_value=[]) as opn, \
             patch.object(live, "discord_send"), \
             patch.object(live, "discord_progress"):
            mdt.now.return_value = now
            live._run()
        return close, opn, shadow

    def test_normal_day_opens(self):
        d = {"trend": "up", "policy_file": "strategy_policy_up.json", "entry_allowed": True}
        close, opn, shadow = self._run(d)
        close.assert_called_once()
        opn.assert_called_once()
        shadow.assert_not_called()

    def test_no_policy_day_records_shadow_only(self):
        d = {"trend": "down", "policy_file": None, "entry_allowed": False,
             "entry_block_reason": "no_approved_policy_for_down", "shadow_policy_file": "strategy_policy.json"}
        close, opn, shadow = self._run(d)
        close.assert_called_once()  # 決済は継続
        opn.assert_not_called()
        shadow.assert_called_once()

    def test_crash_brake_blocks_without_shadow(self):
        d = {"trend": "up", "policy_file": "strategy_policy_up.json", "entry_allowed": True}
        braked = dict(d, intraday_crash_brake=True)
        close, opn, shadow = self._run(d, braked)
        close.assert_called_once()
        opn.assert_not_called()
        shadow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
