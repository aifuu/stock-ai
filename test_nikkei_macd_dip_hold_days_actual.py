"""F4回帰テスト: nikkei_macd_dip_paper.close_position()が記録する
hold_days_actualが、実際の決済判定(check_exit_intraday()のHOLD_LIMIT判定=
エントリー翌営業日からの経過営業日数)と一致することを確認する。

修正前はエントリー当日を含めて数えていたため、HOLD_DAYS=3で決済しても
履歴CSVには4と記録されていた(判定基準とのズレ)。このカラムは
nikkei_macd_dip_daily_report.py 含むどの消費者からも値の意味に依存した
読み方をされていない(表示・記録専用)ため、判定側の定義に合わせて補正する。
"""
import csv
import os
import shutil
import tempfile
import unittest

import pandas as pd

import nikkei_macd_dip_paper as app


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="nikkei_macd_dip_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _cfg():
    return {
        "name": "test",
        "label": "test",
        "history_file": "history_test.csv",
        "initial_capital": 1_000_000.0,
    }


def _state():
    return {"capital": 1_000_000.0, "peak": 1_000_000.0, "max_dd": 0.0, "position": None}


def _position(entry_date, entry_price, tp, sl):
    return {
        "ticker": "TEST.T",
        "company": "テスト",
        "entry_date": entry_date,
        "entry_price": entry_price,
        "shares": 100,
        "invested_amount": entry_price * 100,
        "tp": tp,
        "sl": sl,
        "score": 80.0,
        "up_probability": 60.0,
        "down_probability": 20.0,
    }


class HoldDaysActualMatchesDecisionTests(TmpDirMixin, unittest.TestCase):
    def test_hold_limit_exit_records_hold_days_matching_decision(self):
        # entry=2026-09-01(火)、HOLD_DAYS=3、祝日なしの連続営業日
        # 09-02,09-03,09-04(木金)経過後の09-04にHOLD_LIMIT到達(held=3)。
        entry_date = "2026-09-01"
        today = "2026-09-04"
        state = _state()
        state["position"] = _position(entry_date, entry_price=1000.0, tp=2000.0, sl=100.0)

        current_price = 1200.0  # tp/sl未到達 -> HOLD_LIMITのはず
        held_bdays = app.count_tse_trading_days(
            pd.Timestamp(entry_date) + pd.Timedelta(days=1), pd.Timestamp(today)
        )
        self.assertEqual(held_bdays, app.HOLD_DAYS)

        result = app.check_exit_intraday(state["position"], current_price, today)
        self.assertIsNotNone(result)
        exit_price, reason = result
        self.assertEqual(reason, "HOLD_LIMIT")

        msg = app.close_position(_cfg(), state, current_price, today)
        self.assertIsNotNone(msg)

        with open("history_test.csv", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "HOLD_LIMIT")
        self.assertEqual(int(rows[0]["hold_days"]), app.HOLD_DAYS)
        self.assertEqual(int(rows[0]["hold_days"]), held_bdays)
        self.assertIn(f"保有: {app.HOLD_DAYS}営業日", msg)

    def test_tp_exit_before_hold_limit_records_consistent_hold_days(self):
        # 1営業日目(09-02)でTP到達。判定側のheld(entry翌営業日〜today)は1。
        entry_date = "2026-09-01"
        today = "2026-09-02"
        state = _state()
        state["position"] = _position(entry_date, entry_price=1000.0, tp=1100.0, sl=900.0)

        held_bdays = app.count_tse_trading_days(
            pd.Timestamp(entry_date) + pd.Timedelta(days=1), pd.Timestamp(today)
        )
        self.assertEqual(held_bdays, 1)

        msg = app.close_position(_cfg(), state, 1150.0, today)
        self.assertIsNotNone(msg)

        with open("history_test.csv", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["result"], "TP")
        self.assertEqual(int(rows[0]["hold_days"]), 1)
        self.assertIn("保有: 1営業日", msg)


if __name__ == "__main__":
    unittest.main()
