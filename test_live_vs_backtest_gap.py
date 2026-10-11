#!/usr/bin/env python3
"""live_vs_backtest_gap.py のユニットテスト。

このテストはネットワークを一切使わない。価格データ取得フック
(live_vs_backtest_gap.fetch_daily_bars)をモックの合成OHLCに差し替えて実行する。
adversarial_strategy_validator.py は一切importしない(importは重い実処理を
伴うため)。
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import live_vs_backtest_gap as gap  # noqa: E402


def bar(date, o, h, l, c):
    return {"date": date, "open": o, "high": h, "low": l, "close": c}


class BacktestConventionExit(unittest.TestCase):
    def test_tp_hit_on_first_future_bar(self):
        bars = [bar("2026-09-01", 100, 100, 100, 100), bar("2026-09-02", 100, 112, 99, 105)]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "TP")
        self.assertAlmostEqual(exit_price, 110.0)
        self.assertEqual(exit_date, "2026-09-02")

    def test_sl_hit(self):
        bars = [bar("2026-09-01", 100, 100, 100, 100), bar("2026-09-02", 100, 103, 88, 90)]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "SL")
        self.assertAlmostEqual(exit_price, 90.0)

    def test_same_bar_both_hit_sl_priority(self):
        bars = [bar("2026-09-01", 100, 100, 100, 100), bar("2026-09-02", 100, 115, 85, 95)]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "SL_BOTH_SL_PRIORITY")
        self.assertAlmostEqual(exit_price, 90.0)

    def test_timeout_hold_when_no_tp_sl_within_hold_days(self):
        bars = [
            bar("2026-09-01", 100, 100, 100, 100),
            bar("2026-09-02", 100, 103, 99, 102),
            bar("2026-09-03", 102, 104, 101, 103),
        ]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=2.0, sl_mult=2.0,
            hold_days=2, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "HOLD")
        self.assertAlmostEqual(exit_price, 103.0)  # second future bar (09-03) is the hold_days=2'th bar

    def test_timeout_loss_when_close_below_entry(self):
        bars = [
            bar("2026-09-01", 100, 100, 100, 100),
            bar("2026-09-02", 100, 103, 99, 95),
        ]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=5.0, sl_mult=5.0,
            hold_days=1, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "TIMEOUT_LOSS")

    def test_short_direction_tp(self):
        bars = [bar("2026-09-01", 100, 100, 100, 100), bar("2026-09-02", 100, 101, 88, 90)]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "SHORT", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        reason, ret, exit_date, exit_price, days = result
        self.assertEqual(reason, "TP")
        self.assertAlmostEqual(exit_price, 90.0)

    def test_no_future_bars_returns_none(self):
        bars = [bar("2026-09-01", 100, 100, 100, 100)]
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        self.assertIsNone(result)

    def test_future_bars_exclude_entry_date_itself(self):
        bars = [bar("2026-09-01", 100, 200, 50, 180)]  # huge move but ON entry date -> ignored
        result = gap.backtest_convention_exit(
            "2026-09-01", 100.0, "BUY", atr_ratio_pct=10.0, tp_mult=1.0, sl_mult=1.0,
            hold_days=3, bars=bars,
        )
        self.assertIsNone(result)


class AtrRatioDerivation(unittest.TestCase):
    def test_derive_atr_ratio_basic(self):
        # entry=100, tp=110, mult=2.0 => atr_ratio = (0.10)/2.0*100 = 5.0
        self.assertAlmostEqual(gap.derive_atr_ratio_pct(100.0, 110.0, 2.0), 5.0)

    def test_derive_atr_ratio_zero_mult_safe(self):
        self.assertEqual(gap.derive_atr_ratio_pct(100.0, 110.0, 0.0), 0.0)

    def test_derive_atr_ratio_zero_entry_safe(self):
        self.assertEqual(gap.derive_atr_ratio_pct(0.0, 110.0, 2.0), 0.0)


class PolicyResolution(unittest.TestCase):
    def test_uses_policy_file_column_when_present(self):
        row = {"policy_file": "strategy_policy_up.json"}
        policy = gap.resolve_policy_for_row(row, "daytrade")
        self.assertEqual(policy["policy_file"], "strategy_policy_up.json")
        self.assertIsNone(policy["assumption"])

    def test_falls_back_and_flags_assumption_when_missing(self):
        row = {}
        policy = gap.resolve_policy_for_row(row, "top1")
        self.assertEqual(policy["policy_file"], "strategy_policy.json")
        self.assertIsNotNone(policy["assumption"])


class ProcessTrackWithMockedBars(unittest.TestCase):
    def setUp(self):
        self._orig_fetch = gap.fetch_daily_bars

    def tearDown(self):
        gap.fetch_daily_bars = self._orig_fetch

    def test_process_track_top1_produces_rows_with_mocked_bars(self):
        def fake_fetch(ticker, start, end):
            return [
                bar("2026-09-03", 2852.0, 2852.0, 2852.0, 2852.0),
                bar("2026-09-04", 2860.0, 2900.0, 2840.0, 2855.0),
                bar("2026-09-05", 2855.0, 2870.0, 2820.0, 2830.0),
                bar("2026-09-08", 2830.0, 2850.0, 2700.0, 2851.0),
            ]
        gap.fetch_daily_bars = fake_fetch
        rows = gap.process_track("top1", "2026-09-01")
        self.assertGreater(len(rows), 0)
        row = rows[0]
        self.assertEqual(row["track"], "top1")
        self.assertIn("gap_pct", row)
        self.assertIn("candidate_causes", row)
        self.assertIn("FORCED_CLOSE_BUDGET_BUG", row["candidate_causes"])
        self.assertIn("pre-2026-10-06", row["candidate_causes"])

    def test_process_track_daytrade_flags_exit_rule_mismatch(self):
        def fake_fetch(ticker, start, end):
            return [
                bar("2026-10-02", 2304.0, 2304.0, 2304.0, 2304.0),
                bar("2026-10-03", 2310.0, 2340.0, 2290.0, 2320.0),
                bar("2026-10-04", 2320.0, 2350.0, 2300.0, 2330.0),
            ]
        gap.fetch_daily_bars = fake_fetch
        rows = gap.process_track("daytrade", "2026-09-01")
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("exit_rule_mismatch", row["candidate_causes"])

    def test_process_track_skips_rows_before_since_date(self):
        def fake_fetch(ticker, start, end):
            return [bar("2099-01-01", 1, 1, 1, 1)]
        gap.fetch_daily_bars = fake_fetch
        rows = gap.process_track("top1", "2099-01-01")
        self.assertEqual(rows, [])

    def test_no_price_data_flags_cause(self):
        def fake_fetch(ticker, start, end):
            return []
        gap.fetch_daily_bars = fake_fetch
        rows = gap.process_track("daytrade", "2026-09-01")
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("no_price_data", row["candidate_causes"])
            self.assertIsNone(row["backtest_return_pct"])


class ReportGeneration(unittest.TestCase):
    def test_generate_report_handles_empty_rows(self):
        report = gap.generate_report([])
        self.assertIn("対象トレードがありません", report)

    def test_generate_report_includes_sample_size_caveat(self):
        report = gap.generate_report([{
            "track": "top1", "ticker": "1234.T", "entry_date": "2026-09-01", "exit_date": "2026-09-02",
            "direction": "BUY", "policy_file": "strategy_policy.json", "live_entry_price": 100.0,
            "live_exit_price": 101.0, "live_result": "TP", "live_return_pct": 1.0, "live_pnl": 100.0,
            "backtest_atr_ratio_pct": 1.0, "backtest_exit_reason": "TP", "backtest_exit_date": "2026-09-02",
            "backtest_exit_price": 101.0, "backtest_return_pct": 0.9, "gap_pct": 0.1, "candidate_causes": "",
        }])
        self.assertIn("サンプルサイズの注意", report)


class EndToEndWithRealLiveCsvAndMockedBars(unittest.TestCase):
    """実際のリポジトリのprofit_top10_paper_history.csv / daytrade_...history.csvを
    読み込みつつ、価格データだけをモックして end-to-end にスクリプトが
    例外なく動くことを確認する(read-only、ネットワーク不使用)。"""

    def setUp(self):
        self._orig_fetch = gap.fetch_daily_bars

        def fake_fetch(ticker, start, end):
            # 十分な本数のダミーOHLCを返す(TP/SLには当たらない水平線)。
            from datetime import datetime, timedelta
            d = datetime.strptime(start, "%Y-%m-%d")
            end_d = datetime.strptime(end, "%Y-%m-%d")
            out = []
            while d <= end_d:
                out.append(bar(d.strftime("%Y-%m-%d"), 1000.0, 1001.0, 999.0, 1000.0))
                d += timedelta(days=1)
            return out

        gap.fetch_daily_bars = fake_fetch

    def tearDown(self):
        gap.fetch_daily_bars = self._orig_fetch

    def test_main_runs_without_network_and_produces_outputs(self):
        rows = []
        for track in ("top1", "daytrade"):
            rows.extend(gap.process_track(track, gap.SINCE_DATE))
        # 実データが存在する前提(このリポジトリには存在する)でチェック。
        if not gap.LIVE_FILES["top1"].exists() and not gap.LIVE_FILES["daytrade"].exists():
            self.skipTest("live history files not present in this checkout")
        report = gap.generate_report(rows)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)


if __name__ == "__main__":
    unittest.main()
