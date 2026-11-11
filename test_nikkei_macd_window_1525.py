"""nikkei_macd_dip_paper.pyの場中判定ウィンドウ終了を15:20→15:25に延長した
修正(2026-11-11、オーナー承認「案2」)の回帰テスト。

背景: 15:15狙いの最終チェックポイントのdispatchが遅れ、11/9(15:20:51)・
11/10(15:20:03)・11/11(15:21:23)に発火し、スクリプト実行時点で15:20を
過ぎていたため3日連続で判定スキップになった。11/11はmid(4911.T 資生堂、
SL 3,991)の価格が15時台に3,971〜3,986でSLを割っていたのに決済されず
翌日に持ち越された。

ネットワークI/Oと時刻はモックし、実際のyfinance呼び出し・実時刻には依存しない。
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from datetime import time as dtime

import pandas as pd

import nikkei_macd_dip_paper as app


class _FixedNow:
    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


# 2026-11-11(水)は東証営業日。
DAY = (2026, 11, 11)

MID_POSITION = {
    "ticker": "4911.T",
    "company": "資生堂",
    "entry_date": "2026-11-11",
    "entry_price": 4107.0,
    "shares": 200,
    "invested_amount": 821400.0,
    "tp": 4309.9943319943895,
    "sl": 3991.003238860349,
    "atr": 162.39546559551135,
    "score": 72.68558430749002,
    "up_probability": 35.01603397588706,
    "down_probability": 26.039094966691597,
    "nikkei_macd_at_entry": -1688.9748816792926,
    "nikkei_macd_percentile_at_entry": 0.3968253968253968,
}


class TradingWindowBoundaryTests(unittest.TestCase):
    def test_market_close_constant_is_1525(self):
        self.assertEqual(app.MARKET_CLOSE, dtime(15, 25))

    def test_actual_late_dispatch_times_are_now_inside_window(self):
        # 実際に判定スキップになった3日分の最終チェックポイント
        # (dispatch時刻+依存インストール約20秒)。
        for h, m, s, day in [(15, 21, 50, 11), (15, 20, 25, 10), (15, 21, 12, 9)]:
            now = datetime(2026, 11, day, h, m, s, tzinfo=app.TZ)
            with self.subTest(now=now):
                self.assertTrue(app.is_trading_window(now))

    def test_1525_boundary_inclusive_and_after_is_rejected(self):
        self.assertTrue(app.is_trading_window(datetime(*DAY, 15, 25, 0, tzinfo=app.TZ)))
        self.assertFalse(app.is_trading_window(datetime(*DAY, 15, 25, 1, tzinfo=app.TZ)))
        self.assertFalse(app.is_trading_window(datetime(*DAY, 15, 30, 0, tzinfo=app.TZ)))
        self.assertFalse(app.is_trading_window(datetime(*DAY, 15, 35, 0, tzinfo=app.TZ)))

    def test_open_boundary_unchanged(self):
        self.assertFalse(app.is_trading_window(datetime(*DAY, 8, 59, 59, tzinfo=app.TZ)))
        self.assertTrue(app.is_trading_window(datetime(*DAY, 9, 0, 0, tzinfo=app.TZ)))


class LateFinalCheckpointClosesSlTests(unittest.TestCase):
    """11/11 15:21:50の実行を再現: midの資生堂がSL(3,991)を割った価格(3,979)
    で、修正後は判定がスキップされずSL決済されること。"""

    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="nikkei_macd_window_1525_test_")
        os.chdir(self.tmp)
        self._orig = (app.datetime, app.make_nikkei, app.nikkei_macd_signal,
                      app.find_best_candidate, app.get_current_price, app.send)
        app.make_nikkei = lambda: pd.DataFrame(
            {"macd": [0.0] * 10}, index=pd.bdate_range(end="2026-11-10", periods=10))
        app.nikkei_macd_signal = lambda nikkei: {
            "today_macd": -1700.0, "today_thr": -900.0, "pct_rank": 0.4,
            "signal": True, "nikkei_date": "2026-11-10"}
        app.find_best_candidate = lambda nikkei: (None, 225, "テスト: 候補なし")
        app.get_current_price = lambda ticker, fallback_close=None: 3979.0
        app.send = lambda msg: False
        for cfg in app.CONFIGS:
            s = app.default_state(cfg)
            if cfg["name"] == "mid":
                s["position"] = dict(MID_POSITION)
                s["last_exit_date"] = "2026-11-10"
            app.save_state(cfg, s)

    def tearDown(self):
        (app.datetime, app.make_nikkei, app.nikkei_macd_signal,
         app.find_best_candidate, app.get_current_price, app.send) = self._orig
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mid_state(self):
        return app.load_state(next(c for c in app.CONFIGS if c["name"] == "mid"))

    def test_1521_run_now_closes_mid_at_sl(self):
        app.datetime = _FixedNow(datetime(*DAY, 15, 21, 50, tzinfo=app.TZ))
        app._run()
        s = self._mid_state()
        self.assertIsNone(s["position"], "15:21の実行でSL割れが決済されていない")
        self.assertEqual(s["last_exit_date"], "2026-11-11")
        # trades_totalはエントリー時に加算される(決済では変わらない)。
        self.assertEqual(s["trades_total"], 0)
        self.assertLess(float(s["capital"]), app.BASE_INITIAL_CAPITAL)
        hist = pd.read_csv("nikkei_macd_dip_paper_history_mid.csv")
        self.assertEqual(hist.iloc[-1]["result"], "SL")
        self.assertAlmostEqual(float(hist.iloc[-1]["exit_price"]), 3979.0)

    def test_after_1525_still_skipped(self):
        app.datetime = _FixedNow(datetime(*DAY, 15, 25, 30, tzinfo=app.TZ))
        app._run()
        s = self._mid_state()
        self.assertIsNotNone(s["position"], "15:25以降は従来どおり判定スキップのはず")
        self.assertEqual(s["last_exit_date"], "2026-11-10")


if __name__ == "__main__":
    unittest.main()
