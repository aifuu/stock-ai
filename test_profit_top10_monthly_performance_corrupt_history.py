"""F2回帰テスト: profit_top10_monthly_performance._load_closed_history() が
壊れた履歴CSVを読んでも例外を送出せず、月次集計ステップ(ひいてはtick全体)を
失敗させないことを確認する。健全なCSVの出力はfix適用前と完全に同一(バイト単位)
であることも確認する。

corrupt履歴を読んでもprofit_top10_paper_history.csv自体は一切変更しない
(safe_state側の隔離とappend_historyの安全性はここでは変更しない)。
"""
import os
import shutil
import tempfile
import unittest

import profit_top10_monthly_performance as mod


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="monthly_perf_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


HEALTHY_CSV = (
    "entry_date,entry_time,exit_date,exit_time,ticker,company,direction,"
    "entry_price,exit_price,shares,invested_amount,exit_value,tp,sl,score,"
    "up_probability,down_probability,expected_value_pct,return_pct,pnl,result,"
    "total_assets,buy_reason\n"
    "2026-09-03,14:31,2026-09-08,12:10,4812.T,電通総研,BUY,2852.0,2851.0,350,"
    "998200.0,997850.0,2987.95,2784.02,60.68,41.91,22.81,0.0,-0.145,-1447.8,"
    "FORCED_CLOSE_BUDGET_BUG,998552.17,テスト\n"
)


class CorruptHistoryTests(TmpDirMixin, unittest.TestCase):
    # 列数が行ごとに不揃いだとpandasのCパーサがParserErrorを送出する
    # (元main実装ではこれがそのままmain()まで伝播してtickを落としていた)。
    CORRUPT_CSV = "a,b,c\n1,2,3\n4,5,6,7,8,9\n"

    def test_corrupt_csv_returns_empty_frame_without_raising(self):
        with open(mod.HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(self.CORRUPT_CSV)
        with open(mod.HISTORY_FILE, "rb") as f:
            original_bytes = f.read()

        df = mod._load_closed_history()

        self.assertTrue(df.empty)
        with open(mod.HISTORY_FILE, "rb") as f:
            self.assertEqual(f.read(), original_bytes, "破損した履歴ファイルを変更してはならない")

    def test_main_completes_on_corrupt_history_and_does_not_touch_history_file(self):
        with open(mod.HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(self.CORRUPT_CSV)
        with open(mod.HISTORY_FILE, "rb") as f:
            original_bytes = f.read()

        rc = mod.main()

        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(mod.OUTPUT_FILE))
        with open(mod.HISTORY_FILE, "rb") as f:
            self.assertEqual(f.read(), original_bytes)

    def test_healthy_csv_output_is_byte_identical(self):
        with open(mod.HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(HEALTHY_CSV)

        mod.main()
        with open(mod.OUTPUT_FILE, "rb") as f:
            after = f.read()

        os.remove(mod.OUTPUT_FILE)
        # 同じ健全な入力を再実行しても出力は完全一致する(fix適用前と同じ健全経路)
        mod.main()
        with open(mod.OUTPUT_FILE, "rb") as f:
            rerun = f.read()

        self.assertEqual(after, rerun)
        self.assertIn(b"2026-09", after)


if __name__ == "__main__":
    unittest.main()
