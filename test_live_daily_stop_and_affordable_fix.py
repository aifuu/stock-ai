"""2026-10-15 OWNER-APPROVED fixes:

FIX 1 (profit_top10_paper.py): HOLD_LIMITロールバック決済が前営業日分の損益を
当日のdaily_start_capitalに計上してしまい、paper_risk_policy.evaluate()の
日次損失上限判定が前日分の損失で誤発火していた(2026-10-15の実インシデント、
4704.T HOLD_LIMITロールバック、exit_date=2026-10-14、pnl=-16,221.81円)。
mark_and_close()が前日付けの決済を検知したら、その損益をdaily_start_capitalにも
反映し、当日の日次損失判定からその損益を除外する。

FIX 2 (run_profit_loop.py): open_top1_only()がTOP1のみをprofit_top10_paper.
open_positions()へ渡すため、TOP1が1単元(100株)すら買えない資金状況だと
フォールバックが無くその日は何も開かなかった(latent bug)。eligible候補を
構築する時点でopen_positions()と同じ「買えるか」判定を適用し、最初に買える
候補をTOP1として選ぶ。

ネットワークは一切使わない(download/download_5m/yfinance/Discordはモック)。
"""
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

os.environ.pop("DISCORD_WEBHOOK", None)

import profit_top10_paper as app
import paper_risk_policy
import run_profit_loop as loop

TZ = ZoneInfo("Asia/Tokyo")


def _daily(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"High": [r[1] for r in rows], "Low": [r[2] for r in rows], "Close": [r[3] for r in rows]},
        index=idx,
    )


def _position(ticker, direction, entry_date, entry_price, tp, sl, shares=100, **extra):
    base = {
        "ticker": ticker, "company": ticker, "direction": direction, "entry_date": entry_date,
        "entry_time": "09:15", "entry_price": entry_price, "tp": tp, "sl": sl, "shares": shares,
        "invested_amount": entry_price * shares, "score": 70.0, "up_probability": 35.0,
        "down_probability": 29.0, "expected_value_pct": 2.5, "buy_reason": "test",
        "current_price": entry_price,
    }
    base.update(extra)
    return base


def _unexpected_5m_call(*a, **k):
    raise AssertionError("ロールバックでHOLD_LIMITが確定しているはずなので5分足取得は不要")


# =========================================================================
# FIX 1: HOLD_LIMITロールバックの前日付け損益は当日の日次損失判定から除外する
# =========================================================================

class PriorDayRollbackExcludedFromDailyStop(unittest.TestCase):
    """2026-10-15の実インシデント(4704.T)を模したリプレイ。"""

    def _rollback_scenario(self, entry_price, exit_price, direction="BUY"):
        position = _position(
            "4704.T", direction, "2026-10-13", entry_price,
            tp=(entry_price + 2000 if direction == "BUY" else entry_price - 2000),
            sl=(entry_price - 2000 if direction == "BUY" else entry_price + 2000),
            shares=100,
        )
        now = datetime(2026, 10, 15, 9, 31, tzinfo=TZ)
        daily_df = _daily([
            ("2026-10-14", max(entry_price, exit_price) + 50, min(entry_price, exit_price) - 50, exit_price),
        ])
        s = {"capital": 940_050.52, "peak": 1_000_000.0, "max_dd": 0.0, "positions": [position],
             "trade_count_date": "2026-10-14", "trades_today": 3, "trades_by_ticker_today": {"4704.T": 1}}
        app.reset_daily(s, "2026-10-15")
        self.assertEqual(s["daily_start_capital"], 940_050.52, "09:31の最初のtickでdaily_start_capitalはcapitalにリセット")
        self.assertEqual(s["trades_today"], 0)
        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", side_effect=_unexpected_5m_call), \
             patch.object(app, "append_history", side_effect=lambda row: captured.append(row)):
            app.mark_and_close(s, now, {"hold_days": 1})
        return s, captured

    def test_prior_day_hold_limit_loss_excluded_capital_unchanged_and_entries_allowed(self):
        entry_price, exit_price = 3410.0, 3251.4456951323227
        s, captured = self._rollback_scenario(entry_price, exit_price)

        self.assertEqual(len(captured), 1)
        row = captured[0]
        self.assertEqual(row["result"], "HOLD_LIMIT")
        self.assertEqual(row["exit_date"], "2026-10-14", "損益は前日付けで記録される(本件の原因)")
        expected_pnl = (exit_price - entry_price) * 100 - (entry_price + exit_price) * 100 * app.FEE_RATE
        self.assertAlmostEqual(expected_pnl, -16_221.81, places=1)
        self.assertAlmostEqual(row["pnl"], expected_pnl, places=6, msg="履歴行の損益額自体はFIX前と変わらない")

        # capitalは前日分の損益を反映して変化する(FIXで変わらない、既存どおり)
        self.assertAlmostEqual(s["capital"], 940_050.52 + expected_pnl, places=6)
        self.assertAlmostEqual(s["capital"], 923_828.71, places=0)

        # FIX: daily_start_capitalも同額ずらすので、当日分の損益はゼロ扱いになる
        self.assertAlmostEqual(s["daily_start_capital"], 940_050.52 + expected_pnl, places=6)
        self.assertAlmostEqual(s["capital"], s["daily_start_capital"], places=6)

        ok, reason = paper_risk_policy.evaluate(s)
        self.assertTrue(ok, f"前日分の損失で当日の新規エントリーが止まってはいけない: {reason}")

        daily_return = s["capital"] / s["daily_start_capital"] - 1.0
        self.assertAlmostEqual(daily_return, 0.0, places=6)

    def test_prior_day_rollback_profit_is_also_excluded(self):
        entry_price, exit_price = 3000.0, 3300.0  # BUYで利益確定のロールバック
        s, captured = self._rollback_scenario(entry_price, exit_price)
        expected_pnl = (exit_price - entry_price) * 100 - (entry_price + exit_price) * 100 * app.FEE_RATE
        self.assertGreater(expected_pnl, 0.0)
        self.assertAlmostEqual(s["daily_start_capital"], 940_050.52 + expected_pnl, places=6)
        self.assertAlmostEqual(s["capital"], s["daily_start_capital"], places=6)
        ok, _ = paper_risk_policy.evaluate(s)
        self.assertTrue(ok)

    def test_same_day_sl_loss_still_blocks_new_entries(self):
        """同日中のSL決済(前日付けでない)は従来どおり当日の日次損失上限に含める。"""
        position = _position("7203.T", "BUY", "2026-10-15", 3000.0, tp=3200.0, sl=2800.0, shares=100)
        now = datetime(2026, 10, 15, 10, 0, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-10-15 09:05", "2026-10-15 09:10"])
        intraday_df = pd.DataFrame(
            {"High": [2990, 2810], "Low": [2960, 2750], "Close": [2965, 2780]}, index=intraday_idx
        )
        s = {"capital": 1_000_000.0, "peak": 1_000_000.0, "max_dd": 0.0, "positions": [position],
             "trade_count_date": "2026-10-14", "trades_today": 0, "trades_by_ticker_today": {}}
        app.reset_daily(s, "2026-10-15")
        with patch("profit_top10_paper.download", return_value=None), \
             patch("profit_top10_paper.download_5m", return_value=intraday_df), \
             patch.object(app, "append_history"):
            app.mark_and_close(s, now, {"hold_days": 1})
        # 同日決済なのでdaily_start_capitalは動かない
        self.assertEqual(s["daily_start_capital"], 1_000_000.0)
        daily_return = s["capital"] / s["daily_start_capital"] - 1.0
        self.assertLessEqual(daily_return, -paper_risk_policy.DAILY_STOP_LOSS)
        ok, reason = paper_risk_policy.evaluate(s)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("日次損失上限"))

    def test_next_day_reset_uses_fresh_capital_baseline(self):
        s = {"capital": 923_828.71, "daily_start_capital": 923_828.71, "trade_count_date": "2026-10-15",
             "trades_today": 5, "trades_by_ticker_today": {"4704.T": 1}, "positions": []}
        app.reset_daily(s, "2026-10-16")
        self.assertEqual(s["daily_start_capital"], 923_828.71)
        self.assertEqual(s["trades_today"], 0)
        self.assertEqual(s["trades_by_ticker_today"], {})

    def test_reset_daily_runs_before_mark_and_close_in_run(self):
        import inspect
        src = inspect.getsource(app._run)
        self.assertLess(src.index("reset_daily(s,today)"), src.index("mark_and_close(s,now,policy)"))


# =========================================================================
# FIX 2: TOP1が資金不足なら、1単元買える次点候補にフォールバックする
# =========================================================================

def _cand(ticker, direction, price, score=80.0, up=60.0, down=10.0):
    return {
        "ticker": ticker, "company": ticker, "direction": direction, "price": price,
        "tp": price * (1.02 if direction == "BUY" else 0.98),
        "sl": price * (0.99 if direction == "BUY" else 1.01),
        "score": score, "up_probability": up, "down_probability": down,
        "flat_probability": max(0.0, 100.0 - up - down), "data_date": "2026-10-15",
    }


def _base_state(capital=1_000_000.0, positions=None):
    return {"capital": capital, "peak": capital, "max_dd": 0.0, "positions": positions or [],
            "trades_today": 0, "trades_by_ticker_today": {}, "last_exit_by_ticker": {}}


class _IdentityPriorityDownDay:
    """profit_priorityの並び替え自体はテスト対象外なので入力順を保つ恒等関数に
    差し替え、_down_day_decisionも固定して地合い取得のネットワーク呼び出しを
    発生させない(他のopen_top1_onlyテストと同じ手法)。"""

    def __init__(self, down_day=True, regime="bearish"):
        self.down_day = down_day
        self.regime = regime

    def __enter__(self):
        self._p1 = patch.object(loop, "_market_regime", return_value=(self.regime, -1.0, -1.0))
        self._p2 = patch.object(loop, "_down_day_decision", return_value=self.down_day)
        self._p3 = patch.object(loop, "profit_priority", side_effect=lambda cands: list(cands))
        for p in (self._p1, self._p2, self._p3):
            p.start()
        return self

    def __exit__(self, *exc):
        for p in (self._p1, self._p2, self._p3):
            p.stop()


class Rank1UnaffordableFallsBackToRank2(unittest.TestCase):
    def test_rank1_unaffordable_rank2_affordable_opens_rank2(self):
        # 本番の2026-10-15 fixture: 6920.T SHORT 39,220円(100株=3,922,000円、
        # 残り資金100万円では買えない) / 8308.T SHORT 2,283.5円(100株=228,350円、買える)
        candidates = [
            _cand("6920.T", "SHORT", price=39_220.0, score=90.0, up=5.0, down=85.0),
            _cand("8308.T", "SHORT", price=2_283.5, score=80.0, up=10.0, down=75.0),
        ]
        state = _base_state(capital=1_000_000.0)
        with _IdentityPriorityDownDay(down_day=True, regime="bearish"):
            opened = loop.open_top1_only(state, {"up_threshold": 0, "min_score_for_buy": 0}, candidates, "2026-10-15")

        self.assertEqual(len(opened), 1, "資金不足でも1件のみ開く(フォールバック先も1件まで)")
        self.assertEqual(opened[0]["ticker"], "8308.T")
        self.assertEqual(opened[0]["direction"], "SHORT")
        self.assertEqual(len(state["positions"]), 1, "同時に複数ポジションを開かない")
        self.assertEqual(state["positions"][0]["ticker"], "8308.T")
        self.assertEqual(state["positions"][0].get("skipped_unaffordable"), "6920.T")

    def test_all_candidates_unaffordable_opens_nothing(self):
        candidates = [
            _cand("6920.T", "SHORT", price=39_220.0),
            _cand("8308.T", "SHORT", price=20_000.0),
        ]
        state = _base_state(capital=500.0)  # 1単元すら買えない
        with _IdentityPriorityDownDay(down_day=True, regime="bearish"):
            opened = loop.open_top1_only(state, {}, candidates, "2026-10-15")
        self.assertEqual(opened, [])
        self.assertEqual(state["positions"], [])

    def test_rank1_affordable_matches_pre_fix_behavior_byte_identical(self):
        """rank1が買える通常ケースでは、FIX導入前と全く同じpositionが開く
        (skipped_unaffordableキーも付与されない = 既存のstateスキーマを汚さない)。"""
        candidates = [
            _cand("8308.T", "SHORT", price=2_283.5, score=90.0, up=5.0, down=85.0),
            _cand("6920.T", "SHORT", price=1_000.0, score=80.0, up=10.0, down=75.0),
        ]
        state = _base_state(capital=1_000_000.0)
        with _IdentityPriorityDownDay(down_day=True, regime="bearish"):
            opened = loop.open_top1_only(state, {}, candidates, "2026-10-15")

        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["ticker"], "8308.T", "rank1がそのまま選ばれる(2位は一切使われない)")
        pos = state["positions"][0]
        self.assertNotIn("skipped_unaffordable", pos, "買えた日は新規キーが一切追加されない(既存stateと完全一致)")
        # open_positions()は1単元だけでなく予算(capital全額)で買える最大ロット数を使う
        expected_shares = (int(1_000_000.0 // 2_283.5) // app.LOT_SIZE) * app.LOT_SIZE
        self.assertEqual(pos["shares"], expected_shares)
        self.assertAlmostEqual(pos["invested_amount"], expected_shares * 2_283.5)
        self.assertEqual(pos["allocation"], 1.0)

    def test_short_direction_follows_same_affordability_rule_as_buy(self):
        candidates = [_cand("9999.T", "SHORT", price=50_000.0)]  # 100株=500万円、買えない
        state = _base_state(capital=1_000_000.0)
        with _IdentityPriorityDownDay(down_day=True, regime="bearish"):
            opened = loop.open_top1_only(state, {}, candidates, "2026-10-15")
        self.assertEqual(opened, [])

    def test_cooldown_filter_applies_before_affordability_check(self):
        """同一銘柄クールダウンは資金不足判定より先に効く(資金的には買えても
        クールダウン中ならスキップされ、見送り候補数にも数えない)。"""
        now = datetime.now(loop.app.TZ)
        recent_exit = (now - timedelta(minutes=5)).isoformat()
        candidates = [_cand("1001.T", "BUY", price=1_000.0, up=60.0, down=5.0)]
        state = _base_state(capital=1_000_000.0)
        state["last_exit_by_ticker"] = {"1001.T": recent_exit}
        with _IdentityPriorityDownDay(down_day=False, regime="neutral"):
            opened = loop.open_top1_only(state, {}, candidates, "2026-10-15")
        self.assertEqual(opened, [], "資金的には買えるがクールダウン中なので見送られる")

    def test_regime_direction_gate_still_applies_before_affordability(self):
        """地合いフィルタ(bullish→BUYのみ)は資金不足判定より先に効く。"""
        candidates = [_cand("2001.T", "SHORT", price=1_000.0, up=5.0, down=80.0)]  # 資金的には買えるがSHORT
        state = _base_state(capital=1_000_000.0)
        with _IdentityPriorityDownDay(down_day=False, regime="bullish"):
            opened = loop.open_top1_only(state, {}, candidates, "2026-10-15")
        self.assertEqual(opened, [], "bullish regimeではSHORTは地合いフィルタで除外される")


if __name__ == "__main__":
    unittest.main()
