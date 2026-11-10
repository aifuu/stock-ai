"""2026-11-10 オーナー承認の2修正のテスト(ネットワーク不使用)。

A. 急騰補助条件: 前日確定比 >= +1.5% なら本番の日次判定(confirmed_trend)を上昇扱い。
   11/6急落後、先物が+4.8%反発してもMA5<MA20で「下落」=空売りのみだった事象への対応。
   週次検証用のclassify_close_seriesには入れない(policy検証・研究凍結の系列を変えない)。
B. 始値ギャップ決済: 始値の時点で既にSL/TPを越えていたら始値で決済する
   (10/9 6472.T: SL371.1に対し始値369.3で寄ったのにSL価格で記録されていた)。
   寄りの判定は日足の公式始値を優先(11/10 8750.T: 日足Open1,789.0 / 5分足Open1,799.5)。
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import all_candidates_paper as acp
import daily_decision as dd
import futures_trend as ft
import profit_top10_paper as app

TZ = ZoneInfo("Asia/Tokyo")
REPO = os.path.dirname(os.path.abspath(__file__))


def _down_bars(last_change, last_day="2026-11-09", n=40):
    """下落基調(MA5<MA20)の確定足で、最終日の前日比だけをlast_changeにする。"""
    idx = pd.bdate_range(end=last_day, periods=n)
    close = np.linspace(66000, 58000, n)
    close[-1] = close[-2] * (1 + last_change)
    return pd.DataFrame({"Close": close}, index=idx)


class SurgeAssist(unittest.TestCase):
    TODAY = datetime(2026, 11, 10, 8, 12, tzinfo=TZ).date()

    def test_surge_forces_up_in_live_decision(self):
        r = ft.confirmed_trend(self.TODAY, downloader=lambda *a, **k: _down_bars(0.048))
        self.assertLess(r["ma_short"], r["ma_long"])  # MAだけなら下落
        self.assertEqual(r["trend"], ft.UP)
        self.assertIn("急騰補助条件", r["reason"])

    def test_below_threshold_stays_down(self):
        r = ft.confirmed_trend(self.TODAY, downloader=lambda *a, **k: _down_bars(0.014))
        self.assertEqual(r["trend"], ft.DOWN)
        self.assertNotIn("急騰", r["reason"])

    def test_exact_threshold_is_up(self):
        bars = _down_bars(0.0)
        bars.iloc[-1, 0] = bars.iloc[-2, 0] * 1.016
        r = ft.confirmed_trend(self.TODAY, downloader=lambda *a, **k: bars)
        self.assertEqual(r["trend"], ft.UP)

    def test_crash_still_forces_down(self):
        r = ft.confirmed_trend(self.TODAY, downloader=lambda *a, **k: _down_bars(-0.02))
        self.assertEqual(r["trend"], ft.DOWN)
        self.assertIn("急落補助条件", r["reason"])

    def test_validation_series_is_unchanged(self):
        # 週次検証の系列には急騰補助を入れない(意図的な差分)。
        s = ft.classify_close_series(_down_bars(0.048)["Close"])
        self.assertEqual(s["trend"].iloc[-1], ft.DOWN)

    def test_decision_uses_up_policy_on_surge_day(self):
        cwd, d = os.getcwd(), tempfile.mkdtemp()
        try:
            for f in ("strategy_policy.json", "strategy_policy_up.json"):
                shutil.copy(os.path.join(REPO, f), os.path.join(d, f))
            os.chdir(d)
            with patch.object(ft, "_download", return_value=_down_bars(0.048)):
                dec = dd.compute_decision(datetime(2026, 11, 10, 8, 12, tzinfo=TZ))
            self.assertEqual(dec["trend"], ft.UP)
            self.assertEqual(dec["policy_file"], "strategy_policy_up.json")
            self.assertTrue(dec["entry_allowed"])
            self.assertIn("急騰補助条件", dec["trend_reason"])
        finally:
            os.chdir(cwd)
            shutil.rmtree(d, ignore_errors=True)


def _pos(ticker, direction, entry_date, ep, tp, sl, shares):
    return {"ticker": ticker, "company": ticker, "direction": direction, "entry_date": entry_date,
            "entry_time": "09:05", "entry_price": ep, "shares": shares, "invested_amount": ep * shares,
            "tp": tp, "sl": sl, "score": 70.0, "up_probability": 40.0, "down_probability": 40.0,
            "current_price": ep}


def _state(p):
    return {"capital": 1_000_000.0, "peak": 1_000_000.0, "max_dd": 0.0, "positions": [p],
            "daily_start_capital": 1_000_000.0, "trade_count_date": "x", "trades_today": 0,
            "trades_by_ticker_today": {}}


def _5m(day, rows):
    idx = pd.to_datetime([f"{day} {t}" for t, *_ in rows])
    o, h, l, c = zip(*[r[1:] for r in rows])
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


def _daily(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    o, h, l, c = zip(*[r[1:] for r in rows])
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


class Top1GapExit(unittest.TestCase):
    def _run(self, p, now, daily_df, m5_df):
        s, rows = _state(p), []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", return_value=m5_df), \
             patch.object(app, "append_history", side_effect=rows.append):
            app.mark_and_close(s, now, {"hold_days": 3})
        return s, rows

    def test_6472_daily_rollback_gap_down_fills_at_open(self):
        # 10/9 6472.T BUY: SL371.1、10/9始値369.3(既に下)。次営業日10/13(10/12は祝日)に日足で遡って検知。
        p = _pos("6472.T", "BUY", "2026-10-08", 375.0, 390.0, 371.1, 1100)
        daily_df = _daily([("2026-10-09", 369.3, 376.3, 367.4, 370.0)])
        _, rows = self._run(p, datetime(2026, 10, 13, 10, 0, tzinfo=TZ), daily_df, None)
        self.assertEqual(rows[0]["result"], "SL")
        self.assertAlmostEqual(rows[0]["exit_price"], 369.3)

    def test_8750_official_open_inside_sl_keeps_sl_price(self):
        # 11/10 8750.T SHORT SL1,789.93: 5分足の最初のOpenは1,799.5だが公式始値は1,789.0(SL未満)。
        p = _pos("8750.T", "SHORT", "2026-11-09", 1738.5, 1532.79, 1789.926522108639, 700)
        m5 = _5m("2026-11-10", [("09:00", 1799.5, 1814.0, 1792.5, 1805.0), ("09:05", 1805.0, 1810.0, 1800.0, 1802.0)])
        daily_df = _daily([("2026-11-10", 1789.0, 1825.5, 1785.0, 1804.5)])
        _, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), daily_df, m5)
        self.assertEqual(rows[0]["result"], "SL")
        self.assertAlmostEqual(rows[0]["exit_price"], 1789.926522108639)

    def test_short_gap_up_through_sl_fills_at_official_open(self):
        p = _pos("8750.T", "SHORT", "2026-11-09", 1738.5, 1532.79, 1789.93, 700)
        m5 = _5m("2026-11-10", [("09:00", 1801.0, 1814.0, 1798.0, 1805.0)])
        daily_df = _daily([("2026-11-10", 1800.0, 1825.5, 1795.0, 1804.5)])
        s, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), daily_df, m5)
        self.assertEqual(rows[0]["result"], "SL")
        self.assertAlmostEqual(rows[0]["exit_price"], 1800.0)
        self.assertAlmostEqual(s["capital"], 1_000_000.0 + rows[0]["pnl"])

    def test_gap_fallback_to_5m_open_when_daily_missing(self):
        p = _pos("8750.T", "SHORT", "2026-11-09", 1738.5, 1532.79, 1789.93, 700)
        m5 = _5m("2026-11-10", [("09:00", 1801.0, 1814.0, 1798.0, 1805.0)])
        _, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), None, m5)
        self.assertAlmostEqual(rows[0]["exit_price"], 1801.0)

    def test_buy_gap_up_through_tp_fills_at_open(self):
        p = _pos("7203.T", "BUY", "2026-11-09", 3000.0, 3100.0, 2900.0, 100)
        m5 = _5m("2026-11-10", [("09:00", 3150.0, 3160.0, 3140.0, 3155.0)])
        daily_df = _daily([("2026-11-10", 3150.0, 3160.0, 3140.0, 3155.0)])
        _, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), daily_df, m5)
        self.assertEqual(rows[0]["result"], "TP")
        self.assertAlmostEqual(rows[0]["exit_price"], 3150.0)

    def test_no_gap_touch_inside_bar_keeps_line_price(self):
        p = _pos("7203.T", "BUY", "2026-11-09", 3000.0, 3100.0, 2900.0, 100)
        m5 = _5m("2026-11-10", [("09:00", 2950.0, 2960.0, 2890.0, 2900.0)])
        daily_df = _daily([("2026-11-10", 2950.0, 2960.0, 2890.0, 2900.0)])
        _, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), daily_df, m5)
        self.assertEqual(rows[0]["result"], "SL")
        self.assertAlmostEqual(rows[0]["exit_price"], 2900.0)

    def test_afternoon_gap_after_lunch_fills_at_bar_open(self):
        p = _pos("7203.T", "BUY", "2026-11-09", 3000.0, 3100.0, 2900.0, 100)
        m5 = _5m("2026-11-10", [("09:00", 2990.0, 3000.0, 2980.0, 2990.0),
                                ("11:25", 2950.0, 2960.0, 2940.0, 2950.0),
                                ("12:30", 2880.0, 2890.0, 2870.0, 2875.0)])
        daily_df = _daily([("2026-11-10", 2990.0, 3000.0, 2870.0, 2875.0)])
        _, rows = self._run(p, datetime(2026, 11, 10, 13, 0, tzinfo=TZ), daily_df, m5)
        self.assertAlmostEqual(rows[0]["exit_price"], 2880.0)

    def test_fill_bar_itself_is_not_a_gap(self):
        # 当日建て: 約定足(09:05、始値で建てた足)の始値はギャップ扱いしない。
        p = _pos("7203.T", "SHORT", "2026-11-10", 3000.0, 2900.0, 3010.0, 100)
        p["fill_bar_time"] = "2026-11-10T09:05:00"
        m5 = _5m("2026-11-10", [("09:00", 3050.0, 3060.0, 3040.0, 3050.0),
                                ("09:05", 3000.0, 3005.0, 2995.0, 3000.0)])
        s, rows = self._run(p, datetime(2026, 11, 10, 9, 20, tzinfo=TZ), None, m5)
        self.assertEqual(rows, [])
        self.assertEqual(len(s["positions"]), 1)


class AllCandidatesGapExit(unittest.TestCase):
    def test_daily_rollback_gap_fills_at_open(self):
        p = _pos("6472.T", "BUY", "2026-10-08", 375.0, 390.0, 371.1, 1100)
        daily_df = _daily([("2026-10-09", 369.3, 376.3, 367.4, 370.0)])
        remaining, closed = acp.evaluate_exits([p], datetime(2026, 10, 13, 16, 5, tzinfo=TZ),
                                               download_fn=lambda *a, **k: daily_df,
                                               download_5m_fn=lambda *a, **k: None)
        self.assertEqual(remaining, [])
        self.assertAlmostEqual(closed[0]["exit_price"], 369.3)

    def test_intraday_first_bar_uses_official_open(self):
        p = _pos("8750.T", "SHORT", "2026-11-09", 1738.5, 1532.79, 1789.93, 700)
        m5 = _5m("2026-11-10", [("09:00", 1799.5, 1814.0, 1792.5, 1805.0)])
        daily_df = _daily([("2026-11-10", 1789.0, 1825.5, 1785.0, 1804.5)])
        _, closed = acp.evaluate_exits([p], datetime(2026, 11, 10, 16, 5, tzinfo=TZ),
                                       download_fn=lambda *a, **k: daily_df,
                                       download_5m_fn=lambda *a, **k: m5)
        self.assertAlmostEqual(closed[0]["exit_price"], 1789.93)
        daily_gap = _daily([("2026-11-10", 1800.0, 1825.5, 1795.0, 1804.5)])
        _, closed = acp.evaluate_exits([p], datetime(2026, 11, 10, 16, 5, tzinfo=TZ),
                                       download_fn=lambda *a, **k: daily_gap,
                                       download_5m_fn=lambda *a, **k: m5)
        self.assertAlmostEqual(closed[0]["exit_price"], 1800.0)


if __name__ == "__main__":
    unittest.main()
