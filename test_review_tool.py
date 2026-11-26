import csv
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

import review_tool as rt

H = {
    "top1": ["entry_date", "entry_time", "exit_date", "exit_time", "ticker", "company", "direction",
             "entry_price", "exit_price", "shares", "pnl", "result"],
    "daytrade": ["entry_date", "entry_time", "entry_price", "exit_date", "exit_time", "exit_price",
                 "ticker", "direction", "shares", "result", "pnl"],
}
MACD = ["entry_date", "exit_date", "ticker", "company", "entry_price", "exit_price", "shares", "pnl", "result"]


def t(d, hm):
    return datetime.fromisoformat(f"{d}T{hm}:00+09:00")


class Base(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp()
        self.logs = tempfile.mkdtemp()
        self._old = rt.LOG_DIR
        rt.LOG_DIR = self.logs
        for tr, fn in rt.HISTORIES.items():
            self.write_csv(fn, H.get(tr, MACD), [])
        self.write_json("daily_decision.json", {"date": "2026-11-27", "trend": "down", "trend_reason": "MA"})
        self.write_json("profit_top10_paper_state.json", {"capital": 99, "peak": 100, "positions": []})
        self.write_json("daytrade_tp5000_sl8000_state.json", {"positions": [], "pending": None})
        for fn in rt.MACD_STATES.values():
            self.write_json(fn, {"capital": 1, "peak": 1, "position": None})

    def tearDown(self):
        rt.LOG_DIR = self._old
        shutil.rmtree(self.repo)
        shutil.rmtree(self.logs)

    def write_json(self, fn, obj):
        with open(os.path.join(self.repo, fn), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def write_csv(self, fn, cols, rows):
        with open(os.path.join(self.repo, fn), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    def save_log(self, date, snap):
        with open(os.path.join(self.logs, f"{date}.json"), "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)


class CadenceTest(unittest.TestCase):
    def test_0907_gap_is_flagged(self):
        c = rt.cadence_from_times([t("2026-09-07", "10:24"), t("2026-09-07", "11:23")], "2026-09-07")
        self.assertEqual(c["status"], "gap")
        self.assertIn({"from": "10:24", "to": "11:23", "minutes": 59}, c["gaps_over_threshold"])

    def test_normal_5min_day_is_ok(self):
        times = []
        cur = t("2026-11-26", "08:13")
        while cur <= t("2026-11-26", "15:30"):
            times.append(cur)
            cur = cur.replace(minute=cur.minute) + (t("2026-11-26", "08:18") - t("2026-11-26", "08:13"))
        c = rt.cadence_from_times(times, "2026-11-26")
        self.assertEqual(c["status"], "ok")
        self.assertLessEqual(c["max_gap_min"], 5)

    def test_api_failure_is_unverified_not_ok(self):
        self.assertEqual(rt.cadence_from_times(None, "2026-11-26")["status"], "unverified")

    def test_no_commits_on_trading_day_is_gap(self):
        c = rt.cadence_from_times([], "2026-11-26", trading_day=True)
        self.assertEqual(c["status"], "gap")

    def test_holiday_is_not_a_gap(self):
        # 2026-11-23 勤労感謝の日: 記録0件でも欠損扱いしない
        self.assertEqual(rt.cadence_from_times([], "2026-11-23", trading_day=False)["status"], "holiday")
        self.assertEqual(rt.cadence_from_times([], "2026-12-31")["status"], "holiday")

    def test_unknown_calendar_with_no_commits_asks_to_check(self):
        self.assertEqual(rt.cadence_from_times([], "2026-11-23", trading_day=None)["status"], "no_session")

    def test_jpholiday_calendar_if_available(self):
        import importlib.util
        if importlib.util.find_spec("jpholiday") is None:
            self.skipTest("jpholiday 無し")
        self.assertFalse(rt.is_trading_day("2026-11-23"))
        self.assertTrue(rt.is_trading_day("2026-11-24"))


class SnapshotAndGradeTest(Base):
    def test_backdated_top1_hold_limit_is_caught_by_row_diff(self):
        # 11/27 に記録される HOLD_LIMIT は exit_date=11/26(期限日)。exit_date 一致では漏れる。
        self.save_log("2026-11-26", {"date": "2026-11-26", "history_counts": {"top1": 0}, "review": {
            "outlook": {"next_trend": "down", "expected_closes": [{"track": "top1", "ticker": "9532.T"}]}}})
        self.write_csv("profit_top10_paper_history.csv", H["top1"], [dict(
            entry_date="2026-11-25", entry_time="10:25", exit_date="2026-11-26", exit_time="15:30",
            ticker="9532.T", company="大阪ガス", direction="SHORT", entry_price=5810, exit_price=5724,
            shares=200, pnl=17200, result="HOLD_LIMIT")])
        s = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])
        self.assertEqual([r["ticker"] for r in s["closed_today"]], ["9532.T"])
        self.assertEqual(s["pnl_by_track"], {"top1": 17200.0})
        g = s["previous_outlook_grade"]
        self.assertEqual(g["status"], "graded")
        self.assertTrue(g["trend"]["hit"])
        self.assertEqual((g["closes_hit"], g["closes_expected"]), (1, 1))
        self.assertEqual(g["unexpected_closes"], [])

    def test_missed_and_unexpected_closes_and_trend_miss(self):
        self.save_log("2026-11-26", {"date": "2026-11-26", "history_counts": {"daytrade": 0, "top1": 0},
                                     "review": {"outlook": {"next_trend": "up", "expected_closes": [
                                         {"track": "top1", "ticker": "9532.T"}]}}})
        self.write_json("profit_top10_paper_state.json", {"capital": 1, "peak": 1, "positions": [
            {"ticker": "9532.T", "direction": "SHORT", "entry_price": 5810}]})
        self.write_csv("daytrade_tp5000_sl8000_history.csv", H["daytrade"], [dict(
            entry_date="2026-11-27", entry_time="10:25", entry_price=100, exit_date="2026-11-27",
            exit_time="15:20", exit_price=101, ticker="1111.T", direction="SHORT", shares=100,
            result="FORCED_EXIT", pnl=-100)])
        s = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])
        g = s["previous_outlook_grade"]
        self.assertFalse(g["trend"]["hit"])
        self.assertEqual(g["expected_closes"][0], {"track": "top1", "ticker": "9532.T",
                                                    "closed": False, "still_open": True})
        self.assertEqual(g["unexpected_closes"], [{"track": "daytrade", "ticker": "1111.T"}])
        self.assertEqual(len(s["losses"]), 1)

    def test_stale_decision_does_not_count_as_actual_trend(self):
        self.save_log("2026-11-26", {"date": "2026-11-26", "review": {"outlook": {
            "next_trend": "down", "expected_closes": []}}})
        self.write_json("daily_decision.json", {"date": "2026-11-26", "trend": "down"})
        g = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])["previous_outlook_grade"]
        self.assertIsNone(g["trend"]["hit"])

    def test_bom_csv_keeps_entry_date(self):
        path = os.path.join(self.repo, "daytrade_tp5000_sl8000_history.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=H["daytrade"])
            w.writeheader()
            w.writerow(dict(entry_date="2026-11-27", entry_time="10:25", entry_price=1, exit_date="2026-11-27",
                            exit_time="15:20", exit_price=1, ticker="1.T", direction="SHORT", shares=1,
                            result="TP", pnl=1))
        s = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])
        self.assertEqual(s["closed_today"][0]["entry_date"], "2026-11-27")

    def test_first_day_without_previous_log(self):
        s = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])
        self.assertEqual(s["previous_outlook_grade"]["status"], "no_previous_log")

    def test_positions_from_all_tracks(self):
        self.write_json("nikkei_macd_dip_paper_state_mid.json", {"capital": 1, "peak": 1, "position": {
            "ticker": "7951.T", "entry_price": 1399}})
        self.write_json("daytrade_tp5000_sl8000_state.json", {"positions": [], "pending": {"ticker": "2222.T"}})
        ps = rt.build_snapshot(self.repo, "2026-11-27", cadence_times=[])["positions"]
        self.assertIn({"track": "macd_mid", "ticker": "7951.T", "entry_price": 1399, "direction": "LONG"}, ps)
        self.assertTrue(any(p.get("pending") and p["ticker"] == "2222.T" for p in ps))


class ReviewValidationTest(Base):
    def good(self):
        return {"kaiko": "a", "tenbou": "b", "kensho": "c", "price_verified": "unverified",
                "price_unverified_reason": "yfinance 403",
                "outlook": {"next_trend": "down", "expected_closes": [{"track": "top1", "ticker": "9532.T"}]}}

    def test_good_review_passes(self):
        self.assertEqual(rt.validate_review(self.good(), {"losses": []}), [])

    def test_bad_reviews_rejected(self):
        for mut in (lambda r: r.pop("kaiko"), lambda r: r.update(price_verified="maybe"),
                    lambda r: r.update(price_unverified_reason=""),
                    lambda r: r["outlook"].update(next_trend="flat"),
                    lambda r: r["outlook"].update(expected_closes=[{"track": "TOP1", "ticker": "x"}])):
            r = self.good()
            mut(r)
            self.assertTrue(rt.validate_review(r, {"losses": []}), r)

    def test_loss_day_cannot_skip_price_check(self):
        r = self.good()
        r["price_verified"] = "not_needed"
        self.assertTrue(rt.validate_review(r, {"losses": [{"pnl": -1}]}))

    def test_holiday_snapshot_writes_nothing(self):
        self.assertEqual(rt.main(["snapshot", "--repo", self.repo, "--date", "2026-12-31", "--no-fetch"]), 0)
        self.assertFalse(os.path.exists(os.path.join(self.logs, "2026-12-31.json")))

    def test_snapshot_rerun_keeps_review_and_summary(self):
        rt.main(["snapshot", "--repo", self.repo, "--date", "2026-11-27", "--no-fetch"])
        p = os.path.join(self.repo, "rv.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.good(), f)
        self.assertEqual(rt.main(["add-review", "--date", "2026-11-27", "--review", p]), 0)
        rt.main(["snapshot", "--repo", self.repo, "--date", "2026-11-27", "--no-fetch"])
        with open(os.path.join(self.logs, "2026-11-27.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["review"]["kaiko"], "a")
        old = rt.SUMMARY
        rt.SUMMARY = os.path.join(self.logs, "s.csv")
        try:
            rt.main(["summary"])
            with open(rt.SUMMARY, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["date"], "2026-11-27")
            self.assertEqual(rows[0]["has_review"], "1")
            self.assertEqual(rows[0]["trend"], "down")
        finally:
            rt.SUMMARY = old


if __name__ == "__main__":
    unittest.main()
