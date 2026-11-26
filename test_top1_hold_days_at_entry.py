"""TOP1の保有上限をエントリー時のpolicyで判定する修正(2026-11-26、オーナー承認)の回帰テスト。

バグ: mark_and_close()は決済時点(当日)のpolicyのhold_daysで保有上限を判定していた。
11/19に下落日policy(hold_days=1)で建てた7532.Tは11/24朝にHOLD_LIMIT決済される
はずだったが、11/24は急騰補助で上昇日policy(hold_days=3)に切り替わったため
決済されず、TOP1枠を塞いだまま11/25朝に11/20終値832.8でHOLD_LIMIT記録された。

修正: open_positions()がポジションにhold_daysを記録し、mark_and_close()はそれを使う。
hold_daysを持たない旧ポジションは従来どおり当日policyの値を使う。

ネットワークは一切使わない(download/download_5m/append_historyはモック)。
"""
import os
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.pop("DISCORD_WEBHOOK", None)

import pandas as pd  # noqa: E402

import profit_top10_paper as app  # noqa: E402
import run_profit_loop as loop  # noqa: E402

TZ = ZoneInfo("Asia/Tokyo")


def _daily(rows):
    """rows: [(date_str, open, high, low, close), ...]"""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
                         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]}, index=idx)


def _bars(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
                         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]}, index=idx)


# 実データ(yfinance日足)に合わせた7532.T 11/20・11/24。
DAILY_7532 = _daily([
    ("2026-11-20", 837.8, 847.5, 829.0, 832.8),
    ("2026-11-24", 840.0, 845.5, 812.8, 826.8),
])


def _pos_7532(**extra):
    p = {
        "ticker": "7532.T", "company": "パン・パシフィック・インターナショナルホールディングス",
        "direction": "SHORT", "entry_date": "2026-11-19", "entry_time": "10:30",
        "entry_price": 832.7000122070312, "shares": 1600, "invested_amount": 1332320.01953125,
        "tp": 740.8271178316028, "sl": 855.6682358008884, "score": 41.39, "up_probability": 28.29,
        "down_probability": 31.48, "expected_value_pct": 2.70, "buy_reason": "test",
        "current_price": 821.3, "fill_method": "decision_next_bar_open_top1_v1",
        "decision_time": "2026-11-19T10:25:48.032223", "fill_bar_time": "2026-11-19T10:30:00",
        "decision_price": 831.2999877929688,
    }
    p.update(extra)
    return p


def _state(p):
    return {"capital": 1_366_150.52, "peak": 1_366_150.52, "max_dd": 0.0, "positions": [p],
            "daily_start_capital": 1_366_150.52, "trade_count_date": "x", "trades_today": 0,
            "trades_by_ticker_today": {}}


class HoldLimitUsesEntryPolicy(unittest.TestCase):
    def _close(self, s, now, policy, daily_df, m5=None):
        rows = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", return_value=m5), \
             patch.object(app, "append_history", side_effect=rows.append):
            app.mark_and_close(s, now, policy)
        return rows

    def test_7532_entered_hold1_closes_on_1124_even_under_up_policy(self):
        s = _state(_pos_7532(hold_days=1))
        rows = self._close(s, datetime(2026, 11, 24, 9, 0, 30, tzinfo=TZ), {"hold_days": 3}, DAILY_7532)
        self.assertEqual(s["positions"], [], "11/24朝に決済されてTOP1枠が空くこと")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "HOLD_LIMIT")
        self.assertEqual(rows[0]["exit_date"], "2026-11-20")
        self.assertEqual(rows[0]["exit_time"], "15:30")
        self.assertAlmostEqual(rows[0]["exit_price"], 832.8)

    def test_legacy_position_without_hold_days_keeps_old_behaviour(self):
        # hold_daysを持たない旧ポジション(本番の11/24状態そのもの)は当日policyの値を使う。
        s = _state(_pos_7532())
        m5 = _bars([("2026-11-24 09:00", 840.0, 841.0, 835.0, 836.0)])
        rows = self._close(s, datetime(2026, 11, 24, 9, 30, tzinfo=TZ), {"hold_days": 3}, DAILY_7532, m5)
        self.assertEqual(rows, [])
        self.assertEqual(len(s["positions"]), 1)

    def test_entered_hold3_is_not_cut_short_by_down_policy(self):
        # 逆方向: 上昇日policy(hold_days=3)で建てたポジションが、翌日の下落日policy
        # (hold_days=1)で前日終値に巻き戻し決済されないこと。
        p = _pos_7532(entry_date="2026-11-24", entry_time="10:00", fill_bar_time="2026-11-24T10:00:00",
                      hold_days=3, tp=700.0, sl=900.0)
        s = _state(p)
        daily_df = _daily([("2026-11-25", 841.8, 870.7, 838.2, 861.3)])
        m5 = _bars([("2026-11-26 09:00", 858.5, 860.9, 850.0, 855.0)])
        rows = self._close(s, datetime(2026, 11, 26, 9, 30, tzinfo=TZ), {"hold_days": 1}, daily_df, m5)
        self.assertEqual(rows, [], "保有1日目(11/25)で上限3日には達していない")
        self.assertEqual(len(s["positions"]), 1)

    def test_entered_hold3_closes_on_its_own_limit(self):
        p = _pos_7532(entry_date="2026-11-24", entry_time="10:00", fill_bar_time="2026-11-24T10:00:00",
                      hold_days=3, tp=700.0, sl=900.0)
        s = _state(p)
        daily_df = _daily([("2026-11-25", 841.8, 870.7, 838.2, 861.3),
                           ("2026-11-26", 858.5, 860.9, 838.5, 845.9),
                           ("2026-11-27", 846.0, 850.0, 840.0, 842.0)])
        rows = self._close(s, datetime(2026, 11, 30, 9, 0, 30, tzinfo=TZ), {"hold_days": 1}, daily_df)
        self.assertEqual(rows[0]["result"], "HOLD_LIMIT")
        self.assertEqual(rows[0]["exit_date"], "2026-11-27")
        self.assertAlmostEqual(rows[0]["exit_price"], 842.0)

    def test_invalid_hold_days_falls_back_to_policy(self):
        for bad in (None, "", "x", 0, -2):
            with self.subTest(bad=bad):
                self.assertEqual(app._position_hold_limit({"hold_days": bad}, 3), 3)
        self.assertEqual(app._position_hold_limit({}, 1), 1)
        self.assertEqual(app._position_hold_limit({"hold_days": "2"}, 1), 2)


class EntryRecordsHoldDays(unittest.TestCase):
    # run_profit_loopのimportでapp.open_positionsはopen_top1_onlyに差し替わるため、
    # 元のprofit_top10_paper.open_positionsはloop._original_openで呼ぶ。
    def _cand(self):
        return {"ticker": "9532.T", "company": "大阪ガス", "direction": "SHORT", "price": 5810.0,
                "tp": 5270.0, "sl": 5945.0, "score": 40.6, "up_probability": 31.4,
                "down_probability": 33.5, "buy_reason": "test", "expected_value_pct": 2.4}

    def _state(self):
        return {"capital": 1_364_524.92, "peak": 1_366_150.52, "max_dd": 0.0, "positions": [],
                "trades_today": 0, "trades_by_ticker_today": {}, "daily_start_capital": 1_364_524.92}

    def test_open_positions_stamps_policy_hold_days(self):
        for hd in (1, 3):
            with self.subTest(hold_days=hd):
                s = self._state()
                out = loop._original_open(s, {"hold_days": hd}, [self._cand()], "2026-11-25")
                self.assertEqual(len(out), 1)
                self.assertEqual(s["positions"][0]["hold_days"], hd)

    def test_candidate_cannot_override_hold_days(self):
        s = self._state()
        c = dict(self._cand(), hold_days=99)
        loop._original_open(s, {"hold_days": 1}, [c], "2026-11-25")
        self.assertEqual(s["positions"][0]["hold_days"], 1)

    def test_top1_pending_fill_path_records_hold_days(self):
        # 本番の約定経路(run_profit_loop.fill_pending_entry -> _original_open)でも記録されること。
        s = self._state()
        s.update(last_exit_by_ticker={}, cancelled_entries=[])
        s["pending_entry"] = dict(self._cand(), decision_time="2026-11-25T10:21:05",
                                  decision_date="2026-11-25", fill_bar_time="2026-11-25T10:25:00",
                                  decision_price=5830.0)
        bars = _bars([("2026-11-25 10:25", 5810.0, 5815.0, 5800.0, 5805.0)])
        with patch.object(app, "download_5m", return_value=bars):
            opened = loop.fill_pending_entry(s, {"hold_days": 1}, datetime(2026, 11, 25, 10, 41, tzinfo=TZ),
                                             True, False)
        self.assertEqual(len(opened), 1)
        self.assertEqual(s["positions"][0]["hold_days"], 1)
        self.assertEqual(s["positions"][0]["entry_price"], 5810.0)


if __name__ == "__main__":
    unittest.main()
