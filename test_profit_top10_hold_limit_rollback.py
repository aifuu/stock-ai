"""profit_top10_paper.mark_and_close() の HOLD_LIMIT ロールバック修正の回帰テスト。

バグ: yfinanceのTSE 5分足データには15:25のバーが存在しない(最終バーは15:20)ため、
同日ループのHOLD_LIMIT判定(`held>=hold_limit and ts.time()>=FORCED_EXIT`)は
実質発火せず、保有上限(hold_days)を超えたポジションが無期限に保有され続けていた。

修正: 翌営業日以降の実行時、前営業日を日足で遡るロールバックループ内で、
TP/SLが未成立の日の時点で保有営業日数が上限に達していれば、その日の日足Closeで
HOLD_LIMIT決済する(all_candidates_paper.evaluate_exitsの既存rollback分岐と同じ
意図・同じexit_dt=その日15:30)。

このテストはネットワークを一切使わない(download/download_5m/discord_sendは
すべてモック)。
"""
import copy
import json
import os
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import profit_top10_paper as app
import all_candidates_paper as acp

TZ = ZoneInfo("Asia/Tokyo")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _daily(rows):
    """rows: [(date_str, high, low, close), ...] -> DataFrame indexed by date."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"High": [r[1] for r in rows], "Low": [r[2] for r in rows], "Close": [r[3] for r in rows]},
        index=idx,
    )


def _position(ticker, direction, entry_date, entry_price, tp, sl, shares=100, **extra):
    base = {
        "ticker": ticker,
        "company": ticker,
        "direction": direction,
        "entry_date": entry_date,
        "entry_time": "09:15",
        "entry_price": entry_price,
        "tp": tp,
        "sl": sl,
        "shares": shares,
        "invested_amount": entry_price * shares,
        "score": 70.0,
        "up_probability": 35.0,
        "down_probability": 29.0,
        "expected_value_pct": 2.5,
        "buy_reason": "test",
        "current_price": entry_price,
    }
    base.update(extra)
    return base


def _state(positions, capital=1_000_000.0, peak=1_000_000.0, max_dd=0.0):
    return {"capital": capital, "peak": peak, "max_dd": max_dd, "positions": positions}


def _no_intraday(*a, **k):
    return None


def _unexpected_5m_call(*a, **k):
    raise AssertionError("ロールバックでTP/SL/HOLD_LIMITが確定しているはずなので5分足取得は不要")


class RealTicker7267HoldLimitRollback(unittest.TestCase):
    """(a) 実案件: 7267.T BUY 2026-09-30 09:15 @1669.0 x500, hold_days=3 ->
    limit day 2026-10-05。10/06にTP上抜けがあっても使われず、10/05のCloseで
    HOLD_LIMIT決済されること。"""

    def test_a_hold_limit_fires_on_limit_day_not_tp_on_later_day(self):
        position = _position(
            "7267.T", "BUY", "2026-09-30", 1669.0, tp=1820.4339878966116, sl=1631.141503025847,
            shares=500, company="本田技研工業",
        )
        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)
        daily_df = _daily([
            ("2026-10-01", 1700.0, 1660.0, 1690.0),   # held=1, no TP/SL
            ("2026-10-02", 1710.0, 1665.0, 1695.0),   # held=2, no TP/SL
            ("2026-10-05", 1700.0, 1650.0, 1670.5),   # held=3=limit, no TP/SL -> HOLD_LIMIT at Close
            ("2026-10-06", 1850.0, 1660.0, 1800.0),   # High > TP, but must NEVER be reached/used
        ])
        s = _state([position])
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            msgs = app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(s["positions"], [], "ポジションは決済されて除去される")
        self.assertEqual(len(captured), 1, "履歴行は1件だけ")
        row = captured[0]
        self.assertEqual(row["result"], "HOLD_LIMIT")
        self.assertEqual(row["exit_price"], 1670.5)
        self.assertEqual(row["exit_date"], "2026-10-05")
        self.assertEqual(row["exit_time"], "15:30")
        self.assertNotEqual(row["result"], "TP")

        gross = (1670.5 - 1669.0) * 500
        fee = (1669.0 + 1670.5) * 500 * app.FEE_RATE
        expected_pnl = gross - fee
        self.assertAlmostEqual(s["capital"], 1_000_000.0 + expected_pnl, places=6)
        self.assertAlmostEqual(row["pnl"], expected_pnl, places=6)

        self.assertEqual(len(msgs), 1)
        self.assertIn("7267.T", msgs[0])
        self.assertIn("決済", msgs[0])


class TpBeatsHoldLimitOnLimitDay(unittest.TestCase):
    """(b) 保有上限到達日にTPも成立していればTPが優先される(HOLD_LIMITにならない)。"""

    def test_b_tp_on_limit_day_wins_over_hold_limit(self):
        position = _position("TEST1.T", "BUY", "2026-09-30", 2000.0, tp=2100.0, sl=1900.0, shares=100)
        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)
        daily_df = _daily([
            ("2026-10-01", 2050.0, 1950.0, 2020.0),
            ("2026-10-02", 2080.0, 1960.0, 2050.0),
            ("2026-10-05", 2105.0, 1950.0, 2098.0),  # limit day: High>=TP(2100) -> TP wins
            ("2026-10-06", 1000.0, 999.0, 999.5),    # must never be reached
        ])
        s = _state([position])
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["result"], "TP")
        self.assertEqual(captured[0]["exit_price"], 2100.0)
        self.assertEqual(captured[0]["exit_date"], "2026-10-05")


class SlOnDayTwoBeforeLimit(unittest.TestCase):
    """(c) 上限到達前(day2)にSLが成立していればSLで決済され、ロールバックの
    HOLD_LIMIT判定まで到達しない。"""

    def test_c_sl_on_day_two_before_limit_day(self):
        position = _position("TEST2.T", "BUY", "2026-09-30", 2000.0, tp=2200.0, sl=1900.0, shares=100)
        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)
        daily_df = _daily([
            ("2026-10-01", 2050.0, 1950.0, 2020.0),  # held=1, no hit
            ("2026-10-02", 2080.0, 1880.0, 1900.0),  # held=2, SL hit (Low<=1900)
            ("2026-10-05", 1.0, 1.0, 1.0),            # must never be reached (held=3=limit day)
        ])
        s = _state([position])
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["result"], "SL")
        self.assertEqual(captured[0]["exit_price"], 1900.0)
        self.assertEqual(captured[0]["exit_date"], "2026-10-02")


class WithinHoldLimitStaysOpen(unittest.TestCase):
    """(d) 保有営業日数がまだ上限未到達(held=2 < hold_limit=3)ならポジションは
    開いたままであること(実在する2026年9月の祝日3連休を跨ぐケース)。"""

    def test_d_held_two_of_three_stays_open(self):
        position = _position("7203.T", "BUY", "2026-09-17", 3000.0, tp=999999.0, sl=-999999.0, shares=100)
        now = datetime(2026, 9, 24, 16, 5, tzinfo=TZ)  # 9/18(1),[9/19-23 holiday],9/24(2) -> held=2
        daily_df = _daily([("2026-09-18", 3010.0, 2995.0, 3005.0)])
        intraday_df = pd.DataFrame(
            {"High": [3010.0], "Low": [2995.0], "Close": [3005.0]},
            index=pd.to_datetime(["2026-09-24 15:30"]),
        )
        s = _state([position])
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", return_value=intraday_df), \
             patch.object(app, "append_history") as mocked_history:
            msgs = app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(len(s["positions"]), 1, "held=2 < hold_limit=3 なのでまだ開いたまま")
        mocked_history.assert_not_called()
        self.assertEqual(msgs, [])


class ShortMirrorHoldLimitRollback(unittest.TestCase):
    """(e) SHORTポジションでもHOLD_LIMITロールバックが同じ営業日数で発火すること。"""

    def test_e_short_hold_limit_rollback(self):
        position = _position("TEST3.T", "SHORT", "2026-09-30", 2000.0, tp=1800.0, sl=2200.0, shares=100)
        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)
        daily_df = _daily([
            ("2026-10-01", 2050.0, 1950.0, 2020.0),  # held=1, no hit
            ("2026-10-02", 2080.0, 1960.0, 2040.0),  # held=2, no hit
            ("2026-10-05", 2090.0, 1850.0, 1950.0),  # held=3=limit, no hit -> HOLD_LIMIT at Close
            ("2026-10-06", 1.0, 1.0, 1.0),             # must never be reached
        ])
        s = _state([position])
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["result"], "HOLD_LIMIT")
        self.assertEqual(captured[0]["exit_price"], 1950.0)
        self.assertEqual(captured[0]["exit_date"], "2026-10-05")
        self.assertEqual(captured[0]["exit_time"], "15:30")

        gross = (2000.0 - 1950.0) * 100
        fee = (2000.0 + 1950.0) * 100 * app.FEE_RATE
        expected_pnl = gross - fee
        self.assertAlmostEqual(s["capital"], 1_000_000.0 + expected_pnl, places=6)


class HoldDaysOneMatchesResearchTrack(unittest.TestCase):
    """(f) hold_days=1: 当日実行が飛ばされた翌営業日実行で、liveの
    mark_and_close()とall_candidates_paper.evaluate_exits()が同一入力で
    完全に同じ結果(exit_price/reason/exit_date)になること。"""

    def test_f_identical_day_counting_vs_research_track(self):
        entry_date = "2026-09-24"  # limit day = 2026-09-25 (hold_days=1)
        now = datetime(2026, 9, 28, 16, 5, tzinfo=TZ)  # 9/26,27は週末。rollback実行
        daily_df = _daily([("2026-09-25", 3010.0, 2995.0, 3011.0)])

        live_position = _position(
            "7203.T", "BUY", entry_date, 3000.0, tp=999999.0, sl=-999999.0, shares=100,
            hold_days=1,  # 無視される(liveはpolicy['hold_days']を使う)が、research側と揃える
            up_probability=60.0, down_probability=10.0,
        )
        research_position = {
            "ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0,
            "tp": 999999.0, "sl": -999999.0, "entry_date": entry_date, "hold_days": 1,
            "score": 70.0, "up_probability": 60.0, "down_probability": 10.0,
        }

        s = _state([live_position])
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 1})

        remaining, closed = acp.evaluate_exits(
            [research_position], now,
            download_fn=lambda t, period=None: daily_df,
            download_5m_fn=_unexpected_5m_call,
        )

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(closed), 1)
        self.assertEqual(captured[0]["result"], closed[0]["exit_reason"])
        self.assertEqual(captured[0]["exit_price"], closed[0]["exit_price"])
        self.assertEqual(captured[0]["exit_date"], closed[0]["exit_date"])
        self.assertEqual(captured[0]["result"], "HOLD_LIMIT")
        self.assertEqual(captured[0]["exit_price"], 3011.0)
        self.assertEqual(captured[0]["exit_date"], "2026-09-25")
        self.assertEqual(remaining, [])


class NoDataDoesNotCrash(unittest.TestCase):
    """(g) 日足データがNone/空でもクラッシュせず、ポジションは開いたままになる。"""

    def test_g_daily_none_stays_open_without_crash(self):
        position = _position("7203.T", "BUY", "2026-09-30", 3000.0, tp=999999.0, sl=-999999.0, shares=100)
        now = datetime(2026, 10, 2, 9, 30, tzinfo=TZ)
        s = _state([position])
        with patch("profit_top10_paper.download", return_value=None), \
             patch("profit_top10_paper.download_5m", return_value=None), \
             patch.object(app, "append_history") as mocked_history:
            msgs = app.mark_and_close(s, now, {"hold_days": 3})
        self.assertEqual(len(s["positions"]), 1)
        mocked_history.assert_not_called()
        self.assertEqual(msgs, [])

    def test_g_daily_empty_dataframe_stays_open_without_crash(self):
        position = _position("7203.T", "BUY", "2026-09-30", 3000.0, tp=999999.0, sl=-999999.0, shares=100)
        now = datetime(2026, 10, 2, 9, 30, tzinfo=TZ)
        s = _state([position])
        with patch("profit_top10_paper.download", return_value=pd.DataFrame()), \
             patch("profit_top10_paper.download_5m", return_value=pd.DataFrame()), \
             patch.object(app, "append_history") as mocked_history:
            msgs = app.mark_and_close(s, now, {"hold_days": 3})
        self.assertEqual(len(s["positions"]), 1)
        mocked_history.assert_not_called()
        self.assertEqual(msgs, [])


class MaxDrawdownTracking(unittest.TestCase):
    """(h) state['max_dd']が更新され、一度上がった最大DDは後で含み益/資本回復が
    あっても減らないこと。実在state(peak=1,000,000 / capital=961,815.25)で
    profit_top10_monthly_performance.csvのmax_drawdown_pct(3.818)と一致すること。"""

    def test_h_max_dd_updates_to_real_state_value(self):
        s = _state([], capital=961_815.2513077424, peak=1_000_000.0, max_dd=0.0)
        app.mark_and_close(s, datetime(2026, 10, 7, 9, 30, tzinfo=TZ), {"hold_days": 3})
        self.assertAlmostEqual(s["max_dd"], 3.818, places=2)
        self.assertEqual(s["peak"], 1_000_000.0, "capitalがpeak未満の間はpeakを動かさない")

    def test_h_max_dd_never_decreases(self):
        s = _state([], capital=900_000.0, peak=1_000_000.0, max_dd=0.0)
        app.mark_and_close(s, datetime(2026, 10, 7, 9, 30, tzinfo=TZ), {"hold_days": 3})
        dd1 = s["max_dd"]
        self.assertAlmostEqual(dd1, 10.0, places=6)

        s["capital"] = 950_000.0  # 資本は回復したが、まだpeak未満
        app.mark_and_close(s, datetime(2026, 10, 8, 9, 30, tzinfo=TZ), {"hold_days": 3})
        self.assertEqual(s["max_dd"], dd1, "最大DDは資本回復後も減らない")

    def test_h_new_peak_resets_drawdown_to_zero_for_that_tick(self):
        s = _state([], capital=1_100_000.0, peak=1_000_000.0, max_dd=5.0)
        app.mark_and_close(s, datetime(2026, 10, 7, 9, 30, tzinfo=TZ), {"hold_days": 3})
        self.assertEqual(s["peak"], 1_100_000.0)
        self.assertEqual(s["max_dd"], 5.0, "新高値更新時はdd=0なので既存の最大DDは維持される")


class MultiplePositionsMixed(unittest.TestCase):
    """(i) 複数ポジションが同時に存在する場合、HOLD_LIMIT対象だけが決済され、
    上限未到達のポジションは影響を受けずに残ること。"""

    def test_i_mixed_positions_only_hold_limit_one_closes(self):
        closing = _position("CLOSE.T", "BUY", "2026-09-30", 1000.0, tp=999999.0, sl=-999999.0, shares=100)
        staying = _position("STAY.T", "BUY", "2026-10-05", 2000.0, tp=999999.0, sl=-999999.0, shares=50)
        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)

        def fake_download(ticker, period=None):
            if ticker == "CLOSE.T":
                return _daily([
                    ("2026-10-01", 1050.0, 950.0, 1010.0),
                    ("2026-10-02", 1050.0, 950.0, 1015.0),
                    ("2026-10-05", 1050.0, 950.0, 1012.0),  # held=3=limit -> HOLD_LIMIT
                ])
            # STAY.T entered 2026-10-05, so as of 2026-10-07 held=1 (< hold_limit=3)
            return _daily([("2026-10-06", 2050.0, 1950.0, 2010.0)])

        captured = []
        s = _state([closing, staying], capital=1_000_000.0)
        with patch("profit_top10_paper.download", side_effect=fake_download), \
             patch("profit_top10_paper.download_5m", return_value=None), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 3})

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["ticker"], "CLOSE.T")
        self.assertEqual(captured[0]["result"], "HOLD_LIMIT")
        self.assertEqual([p["ticker"] for p in s["positions"]], ["STAY.T"])


class RealRepoStateReplay(unittest.TestCase):
    """実リポジトリのprofit_top10_paper_state.json/historyを読み込み、
    1回のmark_and_close()実行(10/05終値1670.5をモック)で7267.Tだけが決済され、
    既存の履歴行や他のポジションは一切変更されないことを確認する。"""

    def test_replay_real_state_closes_only_7267(self):
        with open(os.path.join(REPO_ROOT, "profit_top10_paper_state.json"), encoding="utf-8") as f:
            real_state = json.load(f)
        history_path = os.path.join(REPO_ROOT, "profit_top10_paper_history.csv")
        with open(history_path, "rb") as f:
            history_before = f.read()

        s = copy.deepcopy(real_state)
        tickers = [p["ticker"] for p in s["positions"]]
        self.assertIn("7267.T", tickers)

        now = datetime(2026, 10, 7, 9, 30, tzinfo=TZ)

        def fake_download(ticker, period=None):
            if ticker == "7267.T":
                return _daily([
                    ("2026-10-01", 1700.0, 1660.0, 1690.0),
                    ("2026-10-02", 1710.0, 1665.0, 1695.0),
                    ("2026-10-05", 1700.0, 1650.0, 1670.5),
                ])
            return None

        captured = []
        with patch("profit_top10_paper.download", side_effect=fake_download), \
             patch("profit_top10_paper.download_5m", return_value=None), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 3})

        remaining_tickers = [p["ticker"] for p in s["positions"]]
        self.assertNotIn("7267.T", remaining_tickers)
        self.assertEqual(len(captured), 1)
        row = captured[0]
        self.assertEqual(row["ticker"], "7267.T")
        self.assertEqual(row["result"], "HOLD_LIMIT")
        self.assertEqual(row["exit_price"], 1670.5)
        self.assertEqual(row["exit_date"], "2026-10-05")
        self.assertEqual(row["exit_time"], "15:30")
        print("REAL STATE REPLAY history row:", json.dumps(row, ensure_ascii=False, default=str))
        print("REAL STATE REPLAY new capital:", s["capital"])

        with open(history_path, "rb") as f:
            history_after = f.read()
        self.assertEqual(history_before, history_after, "テストは実履歴ファイルに一切書き込んでいない")


if __name__ == "__main__":
    unittest.main()
