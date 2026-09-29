"""daily_model_retrain.pyのライブセッション中再学習禁止ガード回帰テスト。

背景(2026-09-29のインシデント):
daily-model-retrain.ymlはクラウド側ルーチンがJST 07:15にworkflow_dispatchする
運用に一本化されたが、ワークフロー自身のschedule('30 22 * * 0-4' = JST 07:30
予定)もGitHub Actionsの実運用遅延(実測1.5〜2.5時間程度)によりJST 09:20〜
10:00頃に発火することがあり、ai-stock-scan.ymlのAMセッション(09:20〜12:35
JST、5分ループでpullしながら売買判定)と重なってdirectional_model.pklを
差し替えると、ライブのペーパートレードセッション中にモデルが入れ替わる
事故になる。retrain_window_open()(JST時刻ウィンドウガード)と
already_retrained_today()(当日実施済みガード)の2つを組み合わせて防ぐ。

このファイルはdaily_model_retrain.py・pandas・共通ライブラリのみを使う。
adversarial_strategy_validator.py, multi_oos_profit_validator.py,
build_strategy_policy.py, stock_scan.py, walk_forward*.py は一切importしない。
"""
import csv
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import daily_model_retrain as m

JST = ZoneInfo("Asia/Tokyo")


def _jst(hour, minute, day=29):
    return datetime(2026, 9, day, hour, minute, tzinfo=JST)


class RetrainWindowOpenBoundaryTests(unittest.TestCase):
    """JST 08:40〜15:34(ライブセッション想定時間帯)は閉、それ以外は開。"""

    def test_08_39_is_open(self):
        self.assertTrue(m.retrain_window_open(_jst(8, 39)))

    def test_08_40_is_closed(self):
        self.assertFalse(m.retrain_window_open(_jst(8, 40)))

    def test_15_34_is_closed(self):
        self.assertFalse(m.retrain_window_open(_jst(15, 34)))

    def test_15_35_is_open(self):
        self.assertTrue(m.retrain_window_open(_jst(15, 35)))

    def test_deep_in_session_is_closed(self):
        # ai-stock-scan.ymlのAMセッション(09:20〜)の最中に相当。
        self.assertFalse(m.retrain_window_open(_jst(9, 30)))
        self.assertFalse(m.retrain_window_open(_jst(12, 0)))

    def test_early_morning_dispatch_is_open(self):
        # クラウド側ルーチンの07:15 workflow_dispatchに相当。
        self.assertTrue(m.retrain_window_open(_jst(7, 15)))

    def test_late_night_manual_run_is_open(self):
        self.assertTrue(m.retrain_window_open(_jst(23, 0)))
        self.assertTrue(m.retrain_window_open(_jst(0, 30)))

    def test_midnight_boundary_is_open(self):
        self.assertTrue(m.retrain_window_open(_jst(0, 0)))


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="daily_model_retrain_gate_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _write_report(rows, path="daily_retrain_report.csv"):
    fieldnames = [
        "date", "train_rows", "oos_days", "oos_trades", "oos_pf",
        "oos_win_rate", "oos_max_dd_pct", "deployed", "reason",
        "live_recent_trades", "live_recent_win_rate", "live_recent_pnl",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _row(date, deployed=False):
    return {
        "date": date, "train_rows": 1000, "oos_days": 60, "oos_trades": 20,
        "oos_pf": 1.1, "oos_win_rate": 55.0, "oos_max_dd_pct": -10.0,
        "deployed": deployed, "reason": "test",
        "live_recent_trades": 0, "live_recent_win_rate": "", "live_recent_pnl": 0.0,
    }


class AlreadyRetrainedTodayTests(TmpDirMixin, unittest.TestCase):
    def test_no_report_file_is_not_done(self):
        self.assertFalse(m.already_retrained_today("2026-09-29"))

    def test_report_without_todays_row_is_not_done(self):
        # ★失敗/中断した実行は当日行を残さないため、有効な後続起動をブロック
        # してはならない(2026-09-28分の完走行だけがある状態を模擬)。
        _write_report([_row("2026-09-28")])
        self.assertFalse(m.already_retrained_today("2026-09-29"))

    def test_report_with_todays_row_is_done(self):
        _write_report([_row("2026-09-28"), _row("2026-09-29", deployed=True)])
        self.assertTrue(m.already_retrained_today("2026-09-29"))

    def test_empty_report_file_is_not_done(self):
        with open("daily_retrain_report.csv", "w", encoding="utf-8-sig") as f:
            f.write("")
        self.assertFalse(m.already_retrained_today("2026-09-29"))

    def test_report_missing_date_column_is_not_done(self):
        with open("daily_retrain_report.csv", "w", encoding="utf-8-sig") as f:
            f.write("some_other_column\nvalue\n")
        self.assertFalse(m.already_retrained_today("2026-09-29"))

    def test_custom_report_file_path_is_honored(self):
        _write_report([_row("2026-09-29")], path="custom_report.csv")
        self.assertFalse(m.already_retrained_today("2026-09-29"))
        self.assertTrue(m.already_retrained_today("2026-09-29", report_file="custom_report.csv"))


class _FixedNow:
    """main()内のdatetime.now(JST)を固定値に差し替えるための最小スタブ。"""

    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


class MainSkipsWithoutSideEffectsTests(TmpDirMixin, unittest.TestCase):
    """ゲートで弾かれた場合、main()は重い処理(build_universe以降)を一切
    呼ばず、ファイルも一切書き込まない(コミットなし)ことを確認する。"""

    def setUp(self):
        super().setUp()
        self._orig_datetime = m.datetime
        self._orig_build_universe = m.build_universe

        def _boom(*args, **kwargs):
            raise AssertionError("build_universe() must not be called when the gate skips main()")

        m.build_universe = _boom

    def tearDown(self):
        m.datetime = self._orig_datetime
        m.build_universe = self._orig_build_universe
        super().tearDown()

    def test_skips_during_live_session_window(self):
        m.datetime = _FixedNow(_jst(10, 0))
        m.main()  # would raise via the build_universe stub if the gate failed to skip
        self.assertFalse(os.path.exists("daily_retrain_report.csv"))
        self.assertFalse(os.path.exists("train_data.csv"))

    def test_skips_when_already_retrained_today(self):
        _write_report([_row("2026-09-29")])
        with open("daily_retrain_report.csv", encoding="utf-8-sig") as f:
            before = f.read()
        m.datetime = _FixedNow(_jst(23, 0))
        m.main()
        with open("daily_retrain_report.csv", encoding="utf-8-sig") as f:
            after = f.read()
        self.assertEqual(before, after)
        self.assertFalse(os.path.exists("train_data.csv"))


if __name__ == "__main__":
    unittest.main()
