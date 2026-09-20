"""common.tse_trading_days_between / count_tse_trading_days の差分テスト。

要件: 東証祝日を含まない期間ではpd.bdate_range(start, end)と完全に同じ
結果になり、祝日を含む期間でのみ意図した差分(祝日を営業日として数えない)
が出ることを、2025-2026年の実カレンダーで検証する。
"""
import datetime
import unittest

import jpholiday
import pandas as pd

from common import count_tse_trading_days, is_tse_trading_day, tse_trading_days_between


def _all_days(year_start=2025, year_end=2026):
    d = datetime.date(year_start, 1, 1)
    end = datetime.date(year_end, 12, 31)
    out = []
    while d <= end:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _has_tse_holiday_between(a, b):
    """[a, b]の間(両端含む)にTSE非営業日(祝日/年末年始)で、かつ土日ではない
    日が1日でもあるか。"""
    d = a
    while d <= b:
        if d.weekday() < 5 and not is_tse_trading_day(d):
            return True
        d += datetime.timedelta(days=1)
    return False


ALL_DAYS_2025_2026 = _all_days()


class NoHolidayPairsMatchBdateRange(unittest.TestCase):
    """祝日を含まない日付ペアでは、新ヘルパーはpd.bdate_rangeと完全一致する。"""

    def test_sampled_no_holiday_pairs_match_exactly(self):
        # 全ペア(365*2 ^ 2)は多すぎるので、各startにつき短い経過日数(0〜10日)
        # だけを総当りする(保有日数チェックは常に短い経過日数で使われるため、
        # これで実運用の全パターンをカバーしつつ現実的な実行時間に収める)。
        checked_no_holiday = 0
        checked_with_holiday = 0
        for start in ALL_DAYS_2025_2026:
            for delta in range(0, 11):
                end = start + datetime.timedelta(days=delta)
                if end.year > 2026:
                    continue
                if _has_tse_holiday_between(start, end):
                    checked_with_holiday += 1
                    continue
                checked_no_holiday += 1
                expected = len(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end)))
                got = count_tse_trading_days(start, end)
                self.assertEqual(
                    got, expected,
                    f"mismatch on holiday-free pair {start}..{end}: got={got} expected={expected}",
                )
                expected_days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
                got_days = tse_trading_days_between(start, end)
                self.assertTrue(expected_days.equals(got_days), f"date list mismatch {start}..{end}")
        # sanity: this test actually exercised both branches over 2 years of data.
        self.assertGreater(checked_no_holiday, 1000)
        self.assertGreater(checked_with_holiday, 100)


class HolidayPairsDifferIntentionally(unittest.TestCase):
    """祝日を挟む日付ペアでは、新ヘルパーはpd.bdate_rangeより少なく数え、
    実際の東証営業日数と一致する(意図した差分)。"""

    def test_known_2026_autumn_holiday_block(self):
        # 2026-09-21 敬老の日, 09-22 国民の休日, 09-23 秋分の日 (月火水、3連休)
        for d in (datetime.date(2026, 9, 21), datetime.date(2026, 9, 22), datetime.date(2026, 9, 23)):
            self.assertTrue(jpholiday.is_holiday(d))
            self.assertFalse(is_tse_trading_day(d))

        start = datetime.date(2026, 9, 17)  # Thu (trading day)
        end = datetime.date(2026, 9, 24)  # Thu (trading day, after the block)

        old_count = len(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end)))
        new_count = count_tse_trading_days(start, end)

        # old (bdate_range, weekends-only) counts 17,18,21,22,23,24 = 6
        self.assertEqual(old_count, 6)
        # new (TSE trading days) counts 17,18,24 = 3 (holidays correctly excluded)
        self.assertEqual(new_count, 3)
        self.assertLess(new_count, old_count)

        got_days = [d.date() for d in tse_trading_days_between(start, end)]
        self.assertEqual(got_days, [
            datetime.date(2026, 9, 17),
            datetime.date(2026, 9, 18),
            datetime.date(2026, 9, 24),
        ])

    def test_live_position_8601_mid_bottom_hold3_exit_date_unchanged(self):
        """本番ライブポジション(nikkei_macd_dip_paper mid/bottom, 8601.T,
        entry_date=2026-09-16, HOLD_DAYS=3)は、旧ロジック・新ロジックとも
        場中tickが実際に走る最初の東証営業日(2026-09-24)でHOLD_LIMITに
        到達する(値は6→3に補正されるが、この銘柄の決済日そのものは
        変わらないことを確認する=健全系での挙動不変の裏付け)。"""
        entry_date = pd.Timestamp("2026-09-16")
        hold_days = 3

        def next_trading_days(start, n):
            d = pd.Timestamp(start)
            out = []
            while len(out) < n:
                d = d + pd.Timedelta(days=1)
                if is_tse_trading_day(d.date()):
                    out.append(d)
            return out

        exec_days = next_trading_days(entry_date, 10)
        old_exit = new_exit = None
        for d in exec_days:
            old_held = len(pd.bdate_range(entry_date + pd.Timedelta(days=1), d))
            new_held = count_tse_trading_days(entry_date + pd.Timedelta(days=1), d)
            if old_exit is None and old_held >= hold_days:
                old_exit = (d.date(), old_held)
            if new_exit is None and new_held >= hold_days:
                new_exit = (d.date(), new_held)
            if old_exit and new_exit:
                break

        self.assertEqual(old_exit, (datetime.date(2026, 9, 24), 6))
        self.assertEqual(new_exit, (datetime.date(2026, 9, 24), 3))

    def test_h3_scenario_friday_entry_premature_exit_fixed(self):
        """H3シナリオ: 2026-09-18(金)エントリー、HOLD_DAYS=3。
        旧ロジックは祝日ブロックの直後(2026-09-24、実質1営業日しか経過して
        いない)にHOLD_LIMITへ到達してしまう(早すぎる決済)。
        新ロジックは実際に3営業日経過した2026-09-28まで正しく待つ。"""
        entry_date = pd.Timestamp("2026-09-18")
        hold_days = 3

        def next_trading_days(start, n):
            d = pd.Timestamp(start)
            out = []
            while len(out) < n:
                d = d + pd.Timedelta(days=1)
                if is_tse_trading_day(d.date()):
                    out.append(d)
            return out

        exec_days = next_trading_days(entry_date, 10)
        old_exit = new_exit = None
        for d in exec_days:
            old_held = len(pd.bdate_range(entry_date + pd.Timedelta(days=1), d))
            new_held = count_tse_trading_days(entry_date + pd.Timedelta(days=1), d)
            if old_exit is None and old_held >= hold_days:
                old_exit = d.date()
            if new_exit is None and new_held >= hold_days:
                new_exit = d.date()
            if old_exit and new_exit:
                break

        self.assertEqual(old_exit, datetime.date(2026, 9, 24))  # premature (bug)
        self.assertEqual(new_exit, datetime.date(2026, 9, 28))  # correct (fixed)
        self.assertGreater((new_exit - old_exit).days, 0)

    def test_h1_scenario_friday_entry_hold1(self):
        """H1シナリオ: 同じ2026-09-18エントリーでHOLD_DAYS=1。旧ロジックは
        held数自体は水増しされる(4)が、たまたま最初に実行される営業日
        (2026-09-24)がHOLD_DAYS=1を満たす最短日でもあるため、決済日自体は
        新ロジックと一致する(内部カウントの誤りは常に決済日のズレへ直結する
        わけではないことの確認)。"""
        entry_date = pd.Timestamp("2026-09-18")
        hold_days = 1
        d = pd.Timestamp("2026-09-24")
        old_held = len(pd.bdate_range(entry_date + pd.Timedelta(days=1), d))
        new_held = count_tse_trading_days(entry_date + pd.Timedelta(days=1), d)
        self.assertEqual(old_held, 4)
        self.assertEqual(new_held, 1)
        self.assertGreaterEqual(old_held, hold_days)
        self.assertGreaterEqual(new_held, hold_days)


class EdgeCases(unittest.TestCase):
    def test_start_after_end_returns_empty(self):
        self.assertEqual(count_tse_trading_days(datetime.date(2026, 1, 5), datetime.date(2026, 1, 1)), 0)
        self.assertEqual(len(tse_trading_days_between(datetime.date(2026, 1, 5), datetime.date(2026, 1, 1))), 0)

    def test_same_day_trading_day_counts_one(self):
        self.assertEqual(count_tse_trading_days(datetime.date(2026, 9, 17), datetime.date(2026, 9, 17)), 1)

    def test_same_day_holiday_counts_zero(self):
        self.assertEqual(count_tse_trading_days(datetime.date(2026, 9, 21), datetime.date(2026, 9, 21)), 0)


if __name__ == "__main__":
    unittest.main()
