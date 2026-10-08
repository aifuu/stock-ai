"""all_candidates_paper.py (独立6ヶ月データ収集トラック) のテスト。

profit_top10_paper.pyの動作は一切変更しない前提で、以下を検証する:
  - 凍結policyファイルはコミット時点のlive policyとバイト単位で一致する
  - トレンド→凍結ファイルのマッピングがlive select_policy_file()の判定と一致
  - scan()にはFROZEN policyが渡される(liveのstrategy_policy*.jsonではない)
  - BUY/SHORTは別トレードとしてdedupされる
  - 決済ロジック(TP/SL/HOLD_LIMIT)がliveのmark_and_close()と同じ判定になる
  - state往復・破損復旧(404は正常/非404はハード失敗)・リトライ時の非重複
  - 月次ローテーション、ALL/TOP1/TOP3/TOP5が同一scan結果由来であること
"""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

import all_candidates_paper as acp
import profit_top10_paper as live

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TZ = ZoneInfo("Asia/Tokyo")


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="acp_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


# =====================================================================
# 凍結policyファイル
# =====================================================================

class FrozenPolicyByteIdentical(unittest.TestCase):
    def test_frozen_normal_matches_live_bytes(self):
        with open(os.path.join(REPO_ROOT, "strategy_policy.json"), "rb") as f:
            live_bytes = f.read()
        with open(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), "rb") as f:
            frozen_bytes = f.read()
        self.assertEqual(live_bytes, frozen_bytes)

    def test_frozen_up_matches_live_bytes(self):
        with open(os.path.join(REPO_ROOT, "strategy_policy_up.json"), "rb") as f:
            live_bytes = f.read()
        with open(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), "rb") as f:
            frozen_bytes = f.read()
        self.assertEqual(live_bytes, frozen_bytes)

    def test_known_sha256_prefixes_unchanged(self):
        normal_hash = hashlib.sha256(open(os.path.join(REPO_ROOT, "strategy_policy.json"), "rb").read()).hexdigest()
        up_hash = hashlib.sha256(open(os.path.join(REPO_ROOT, "strategy_policy_up.json"), "rb").read()).hexdigest()
        self.assertTrue(normal_hash.startswith("cf106bdc4c56"))
        self.assertTrue(up_hash.startswith("cfe6da4cc960"))

    def test_frozen_files_hash_matches_sha256_file_helper(self):
        expected = hashlib.sha256(open(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), "rb").read()).hexdigest()
        self.assertEqual(acp.sha256_file(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json")), expected)


# =====================================================================
# トレンド→凍結ファイルのマッピング
# =====================================================================

class TrendMappingMatchesLiveDecision(unittest.TestCase):
    def test_up_file_maps_to_frozen_up(self):
        self.assertEqual(acp.choose_frozen_policy_file("strategy_policy_up.json"), acp.FROZEN_POLICY_FILE_UP)

    def test_normal_fallback_file_maps_to_frozen_normal(self):
        self.assertEqual(acp.choose_frozen_policy_file("strategy_policy.json"), acp.FROZEN_POLICY_FILE)

    def test_down_file_name_maps_to_frozen_normal_not_a_frozen_down(self):
        # strategy_policy_down.json はliveに存在しない前提だが、将来出来ても
        # このトラックは常にnormalへフォールバックする(down専用の凍結ファイルは作らない)。
        self.assertEqual(acp.choose_frozen_policy_file("strategy_policy_down.json"), acp.FROZEN_POLICY_FILE)

    def test_matches_live_select_policy_file_up_case(self):
        fake_decision = {"trend": "up", "trend_reason": "test", "trend_source": "futures",
                         "policy_file": live.POLICY_FILE_UP, "entry_allowed": True}
        with patch("profit_top10_paper.daily_decision.ensure_decision", return_value=fake_decision):
            live_file, live_result = live.select_policy_file()
        self.assertEqual(live_file, live.POLICY_FILE_UP)
        self.assertEqual(acp.choose_frozen_policy_file(live_file), acp.FROZEN_POLICY_FILE_UP)
        self.assertEqual(live_result["trend"], "up")

    def test_matches_live_select_policy_file_down_case_falls_back_to_normal(self):
        # strategy_policy_down.json は存在しない -> 実売買なし(entry_allowed=False)。
        # 互換のためファイル名はPOLICY_FILEを返し、ALLトラックは凍結normalへ対応付く。
        fake_decision = {"trend": "down", "trend_reason": "test", "trend_source": "futures",
                         "policy_file": None, "entry_allowed": False,
                         "entry_block_reason": "no_approved_policy_for_down"}
        with patch("profit_top10_paper.daily_decision.ensure_decision", return_value=fake_decision):
            live_file, live_result = live.select_policy_file()
        self.assertFalse(live_result["entry_allowed"])
        self.assertEqual(live_file, live.POLICY_FILE)
        self.assertEqual(acp.choose_frozen_policy_file(live_file), acp.FROZEN_POLICY_FILE)
        self.assertTrue(acp.choose_frozen_policy_file(live_file) != acp.FROZEN_POLICY_FILE_UP)


# =====================================================================
# scan()にFROZEN policyが渡ること(spy)
# =====================================================================

class ScanCalledWithFrozenPolicy(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), "all_candidates_frozen_policy.json")
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), "all_candidates_frozen_policy_up.json")

    def test_run_passes_frozen_policy_dict_not_live_strategy_policy(self):
        marker_normal = {"up_threshold": 1.0, "min_score_for_buy": 1.0, "nikkei_filter": False,
                          "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0, "hold_days": 1,
                          "_frozen_marker": "NORMAL"}
        marker_up = dict(marker_normal, _frozen_marker="UP")

        def fake_load_frozen_policy(path):
            return marker_up if path.endswith(acp.FROZEN_POLICY_FILE_UP) else marker_normal

        fake_scan = MagicMock(return_value=([], 0))
        with patch.object(acp, "load_frozen_policy", side_effect=fake_load_frozen_policy), \
             patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", fake_scan), \
             patch.object(acp, "fetch_state", return_value=(acp.default_state(), "initialized_empty")), \
             patch.object(acp, "promote_and_upload_state"), \
             patch.object(acp, "list_release_assets", return_value=[]), \
             patch.object(acp, "append_trade_rows", return_value={}), \
             patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
            acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=".")

        self.assertEqual(fake_scan.call_count, 1)
        called_policy = fake_scan.call_args[0][0]
        self.assertEqual(called_policy["_frozen_marker"], "NORMAL")  # live returned normal (not up) file
        self.assertEqual(fake_scan.call_args[1].get("limit"), None)


# =====================================================================
# BUY/SHORT 別トレードとしてdedup
# =====================================================================

class BuyShortSeparateTradeIds(unittest.TestCase):
    def _policy(self):
        return {"nikkei_filter": False, "hold_days": 1}

    def test_same_ticker_buy_and_short_same_day_get_two_distinct_trade_ids(self):
        candidates = [
            {"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
             "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"},
            {"ticker": "7203.T", "direction": "SHORT", "price": 3000.0, "tp": 2900.0, "sl": 3050.0,
             "score": 75.0, "up_probability": 10.0, "down_probability": 60.0, "data_date": "2026-09-25"},
        ]
        new_positions = acp.build_new_positions(set(), candidates, "2026-09-25", self._policy(), "all_candidates_frozen_policy.json", "hash123", False)
        self.assertEqual(len(new_positions), 2)
        ids = {p["trade_id"] for p in new_positions}
        self.assertEqual(len(ids), 2)
        directions = {p["direction"] for p in new_positions}
        self.assertEqual(directions, {"BUY", "SHORT"})
        # rank must reflect scan()'s sort order (1-indexed, in input order here)
        self.assertEqual(new_positions[0]["rank"], 1)
        self.assertEqual(new_positions[1]["rank"], 2)

    def test_dedup_is_per_direction_not_per_ticker(self):
        candidates = [
            {"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
             "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"},
            {"ticker": "7203.T", "direction": "SHORT", "price": 3000.0, "tp": 2900.0, "sl": 3050.0,
             "score": 75.0, "up_probability": 10.0, "down_probability": 60.0, "data_date": "2026-09-25"},
        ]
        buy_id = acp.build_trade_id("2026-09-25", "7203.T", "BUY", "2026-09-25", "hash123")
        known = {buy_id}
        new_positions = acp.build_new_positions(known, candidates, "2026-09-25", self._policy(), "f.json", "hash123", False)
        self.assertEqual(len(new_positions), 1)
        self.assertEqual(new_positions[0]["direction"], "SHORT")


# =====================================================================
# 決済ロジック: liveのmark_and_close()と同じ判定になることを確認
# =====================================================================

def _make_daily_df(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"High": [r[1] for r in rows], "Low": [r[2] for r in rows], "Close": [r[3] for r in rows]}, index=idx)


def _live_extra_fields():
    """live.mark_and_close's append_history() dereferences these keys
    directly (no .get default), so the shared position fixture needs them
    even though acp.evaluate_exits itself never reads them."""
    return {"entry_time": "09:05", "company": "トヨタ自動車", "invested_amount": 300000.0, "shares": 100}


class ExitLogicCrossCheckedAgainstLive(unittest.TestCase):
    def test_tp_hit_via_daily_rollback_matches_live(self):
        position = {"ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0, "tp": 3100.0,
                    "sl": 2900.0, "entry_date": "2026-09-14", "hold_days": 3, "score": 80.0,
                    "up_probability": 60.0, "down_probability": 10.0, **_live_extra_fields()}
        now = datetime(2026, 9, 17, 15, 25, tzinfo=TZ)  # several (non-holiday) trading days later
        daily_df = _make_daily_df([
            ("2026-09-15", 3050, 3020, 3040),
            ("2026-09-16", 3105, 3040, 3100),  # TP hit here (High>=3100)
        ])

        captured = []
        with patch("profit_top10_paper.download", return_value=daily_df), \
             patch("profit_top10_paper.download_5m", return_value=None), \
             patch.object(live, "append_history", side_effect=lambda row: captured.append(row)):
            s = {"positions": [dict(position)], "capital": 1_000_000.0, "peak": 1_000_000.0}
            live.mark_and_close(s, now, {"hold_days": 3})

        remaining, closed = acp.evaluate_exits([dict(position)], now,
                                                download_fn=lambda t, period=None: daily_df,
                                                download_5m_fn=lambda t: None)

        self.assertEqual(len(closed), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(closed[0]["exit_reason"], captured[0]["result"])
        self.assertEqual(closed[0]["exit_price"], captured[0]["exit_price"])
        self.assertEqual(closed[0]["exit_reason"], "TP")
        self.assertEqual(closed[0]["exit_price"], 3100.0)
        self.assertEqual(remaining, [])

    def test_sl_hit_intraday_matches_live(self):
        position = {"ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0, "tp": 3200.0,
                    "sl": 2950.0, "entry_date": "2026-09-24", "hold_days": 1, "score": 80.0,
                    "up_probability": 60.0, "down_probability": 10.0, **_live_extra_fields()}
        now = datetime(2026, 9, 25, 10, 0, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-09-25 09:05", "2026-09-25 09:10"])
        intraday_df = pd.DataFrame({"High": [2990, 2985], "Low": [2960, 2940], "Close": [2965, 2945]}, index=intraday_idx)

        captured = []
        with patch("profit_top10_paper.download", return_value=None), \
             patch("profit_top10_paper.download_5m", return_value=intraday_df), \
             patch.object(live, "append_history", side_effect=lambda row: captured.append(row)):
            s = {"positions": [dict(position)], "capital": 1_000_000.0, "peak": 1_000_000.0}
            live.mark_and_close(s, now, {"hold_days": 1})

        remaining, closed = acp.evaluate_exits([dict(position)], now,
                                                download_fn=lambda t, period=None: None,
                                                download_5m_fn=lambda t: intraday_df)

        self.assertEqual(closed[0]["exit_reason"], captured[0]["result"])
        self.assertEqual(closed[0]["exit_price"], captured[0]["exit_price"])
        self.assertEqual(closed[0]["exit_reason"], "SL")
        self.assertEqual(closed[0]["exit_price"], 2950.0)

    def test_hold_limit_forced_exit_matches_live(self):
        position = {"ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0, "tp": 3500.0,
                    "sl": 2000.0, "entry_date": "2026-09-24", "hold_days": 1, "score": 80.0,
                    "up_probability": 60.0, "down_probability": 10.0, **_live_extra_fields()}
        now = datetime(2026, 9, 25, 15, 26, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-09-25 09:05", "2026-09-25 15:25"])
        intraday_df = pd.DataFrame({"High": [3010, 3020], "Low": [2995, 3000], "Close": [3005, 3015]}, index=intraday_idx)

        captured = []
        with patch("profit_top10_paper.download", return_value=None), \
             patch("profit_top10_paper.download_5m", return_value=intraday_df), \
             patch.object(live, "append_history", side_effect=lambda row: captured.append(row)):
            s = {"positions": [dict(position)], "capital": 1_000_000.0, "peak": 1_000_000.0}
            live.mark_and_close(s, now, {"hold_days": 1})

        remaining, closed = acp.evaluate_exits([dict(position)], now,
                                                download_fn=lambda t, period=None: None,
                                                download_5m_fn=lambda t: intraday_df)

        self.assertEqual(closed[0]["exit_reason"], captured[0]["result"])
        self.assertEqual(closed[0]["exit_price"], captured[0]["exit_price"])
        self.assertEqual(closed[0]["exit_reason"], "HOLD_LIMIT")

    def test_no_exit_condition_position_remains_open_like_live(self):
        position = {"ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0, "tp": 3500.0,
                    "sl": 2000.0, "entry_date": "2026-09-24", "hold_days": 3, "score": 80.0,
                    "up_probability": 60.0, "down_probability": 10.0}
        now = datetime(2026, 9, 25, 10, 0, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-09-25 09:05"])
        intraday_df = pd.DataFrame({"High": [3010], "Low": [2995], "Close": [3005]}, index=intraday_idx)

        with patch("profit_top10_paper.download", return_value=None), \
             patch("profit_top10_paper.download_5m", return_value=intraday_df), \
             patch.object(live, "append_history"):
            s = {"positions": [dict(position)], "capital": 1_000_000.0, "peak": 1_000_000.0}
            live.mark_and_close(s, now, {"hold_days": 3})
            live_still_open = len(s["positions"]) == 1

        remaining, closed = acp.evaluate_exits([dict(position)], now,
                                                download_fn=lambda t, period=None: None,
                                                download_5m_fn=lambda t: intraday_df)
        self.assertTrue(live_still_open)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(len(closed), 0)


# =====================================================================
# HOLD_LIMIT修正: 15:20実行だと当日ループでは検知できず、rollback側も
# TP/SLしか見ていなかったため無期限に持ち越されていたバグの回帰テスト。
# ここでは実際のカレンダー(2026年9月、敬老の日9/21・国民の休日9/22・
# 秋分の日9/23という実在の3連休を含む)を使う。
# =====================================================================

def _hold_limit_position(entry_date, hold_days, tp=999999.0, sl=-999999.0):
    """TP/SLに絶対到達しない(数値的にあり得ない)水準のBUYポジション。
    HOLD_LIMIT以外の決済理由が紛れ込まないようにするためのフィクスチャ。
    """
    return {
        "ticker": "7203.T", "direction": "BUY", "entry_price": 3000.0,
        "tp": tp, "sl": sl, "entry_date": entry_date, "hold_days": hold_days,
        "score": 80.0, "up_probability": 60.0, "down_probability": 10.0,
    }


class HoldLimitRollbackFix(unittest.TestCase):
    # (i) 16:05実行(修正後cron)なら、当日ループの15:25以降バーでHOLD_LIMITが
    # 検知できる。
    def test_i_run_at_1605_with_1525_bar_closes_hold_limit_today(self):
        position = _hold_limit_position("2026-09-24", hold_days=1)  # limit day = 2026-09-25
        now = datetime(2026, 9, 25, 16, 5, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-09-25 09:05", "2026-09-25 15:25"])
        intraday_df = pd.DataFrame({"High": [3010, 3010], "Low": [2995, 2995], "Close": [3005, 3012]}, index=intraday_idx)

        remaining, closed = acp.evaluate_exits(
            [dict(position)], now,
            download_fn=lambda t, period=None: None,
            download_5m_fn=lambda t: intraday_df,
        )
        self.assertEqual(remaining, [])
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "HOLD_LIMIT")
        self.assertEqual(closed[0]["exit_price"], 3012.0)
        self.assertEqual(closed[0]["exit_date"], "2026-09-25")
        self.assertEqual(closed[0]["exit_time"], "15:25")

    # (ii) 旧cron相当(15:20、バーは15:15までしか無い)だとその日は検知できず
    # (=旧バグの再現)、翌営業日の実行でrollback側がHOLD_LIMITとして
    # limit dayのCloseで決済する。
    def test_ii_missed_today_bar_then_next_day_rollback_closes_hold_limit(self):
        position = _hold_limit_position("2026-09-24", hold_days=1)  # limit day = 2026-09-25

        old_cron_now = datetime(2026, 9, 25, 15, 20, tzinfo=TZ)
        intraday_idx = pd.to_datetime(["2026-09-25 09:05", "2026-09-25 15:15"])
        intraday_df = pd.DataFrame({"High": [3010, 3010], "Low": [2995, 2995], "Close": [3005, 3007]}, index=intraday_idx)
        remaining, closed = acp.evaluate_exits(
            [dict(position)], old_cron_now,
            download_fn=lambda t, period=None: None,
            download_5m_fn=lambda t: intraday_df,
        )
        self.assertEqual(len(closed), 0, "旧バグの再現: 15:25バーが無いので当日は検知できない")
        self.assertEqual(len(remaining), 1)

        next_run_now = datetime(2026, 9, 28, 16, 5, tzinfo=TZ)  # 9/26,27は週末
        daily_df = _make_daily_df([("2026-09-25", 3010.0, 2995.0, 3008.0)])
        remaining2, closed2 = acp.evaluate_exits(
            remaining, next_run_now,
            download_fn=lambda t, period=None: daily_df,
            download_5m_fn=lambda t: (_ for _ in ()).throw(AssertionError("rollbackで決済済みのはずなのでintraday取得は不要")),
        )
        self.assertEqual(remaining2, [])
        self.assertEqual(len(closed2), 1)
        self.assertEqual(closed2[0]["exit_reason"], "HOLD_LIMIT")
        self.assertEqual(closed2[0]["exit_price"], 3008.0)
        self.assertEqual(closed2[0]["exit_date"], "2026-09-25", "exit_dateはlimit dayそのもの")
        self.assertEqual(closed2[0]["exit_time"], "15:30")

    # (iii) rollback対象日にTP/SLとHOLD_LIMITが両方成立する場合はTP/SLが勝つ
    # (liveと同じ優先順位)。
    def test_iii_tp_beats_hold_limit_on_same_rollback_day(self):
        position = _hold_limit_position("2026-09-24", hold_days=1, tp=3100.0, sl=2000.0)
        now = datetime(2026, 9, 28, 16, 5, tzinfo=TZ)
        # 2026-09-25はHOLD_LIMIT到達日でもあり、TPも踏んでいる
        daily_df = _make_daily_df([("2026-09-25", 3105.0, 2995.0, 3008.0)])
        remaining, closed = acp.evaluate_exits(
            [dict(position)], now,
            download_fn=lambda t, period=None: daily_df,
            download_5m_fn=lambda t: (_ for _ in ()).throw(AssertionError("TP/SLで決済済みのはずなのでintraday取得は不要")),
        )
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "TP")
        self.assertEqual(closed[0]["exit_price"], 3100.0)

    # (iv) hold_days=1とhold_days=3で、上限に達するまでは開いたままであること。
    def test_iv_hold_days_1_and_3_only_fire_at_their_own_limit(self):
        with self.subTest(hold_days=1):
            position = _hold_limit_position("2026-09-24", hold_days=1)
            now = datetime(2026, 9, 24, 16, 5, tzinfo=TZ)  # entry当日 = held 0
            intraday_df = pd.DataFrame({"High": [3010.0], "Low": [2995.0], "Close": [3005.0]},
                                        index=pd.to_datetime(["2026-09-24 15:30"]))
            remaining, closed = acp.evaluate_exits(
                [dict(position)], now,
                download_fn=lambda t, period=None: None,
                download_5m_fn=lambda t: intraday_df,
            )
            self.assertEqual(len(closed), 0)
            self.assertEqual(len(remaining), 1)

        with self.subTest(hold_days=3):
            position = _hold_limit_position("2026-09-17", hold_days=3)
            # 実トレーディング日: 9/18(1), [9/19-23休場], 9/24(2), 9/25(3=上限)
            now = datetime(2026, 9, 24, 16, 5, tzinfo=TZ)  # held=2、まだ上限に未到達
            daily_df = _make_daily_df([("2026-09-18", 3010.0, 2995.0, 3005.0)])
            intraday_df = pd.DataFrame({"High": [3010.0], "Low": [2995.0], "Close": [3005.0]},
                                        index=pd.to_datetime(["2026-09-24 15:30"]))
            remaining, closed = acp.evaluate_exits(
                [dict(position)], now,
                download_fn=lambda t, period=None: daily_df,
                download_5m_fn=lambda t: intraday_df,
            )
            self.assertEqual(len(closed), 0, "held=2 < hold_limit=3 なのでまだ開いたまま")
            self.assertEqual(len(remaining), 1)

            now_at_limit = datetime(2026, 9, 25, 16, 5, tzinfo=TZ)  # held=3=上限
            daily_df2 = _make_daily_df([("2026-09-18", 3010.0, 2995.0, 3005.0), ("2026-09-24", 3010.0, 2995.0, 3005.0)])
            intraday_df2 = pd.DataFrame({"High": [3010.0], "Low": [2995.0], "Close": [3009.0]},
                                         index=pd.to_datetime(["2026-09-25 15:30"]))
            remaining2, closed2 = acp.evaluate_exits(
                remaining, now_at_limit,
                download_fn=lambda t, period=None: daily_df2,
                download_5m_fn=lambda t: intraday_df2,
            )
            self.assertEqual(len(closed2), 1)
            self.assertEqual(closed2[0]["exit_reason"], "HOLD_LIMIT")

    # (v) ワークフローがまるまる1日飛ばされたケース(D実行・D+1未実行・D+2実行)
    # -> H1ポジションはD+1(限界日)付でrollback決済される。
    def test_v_skipped_day_then_rollback_closes_at_day_after_entry(self):
        position = _hold_limit_position("2026-09-24", hold_days=1)  # D=9/24(entry), D+1=9/25(limit day, run skipped)
        # D+2実行 = 2026-09-28(9/26,27は週末)
        d_plus_2 = datetime(2026, 9, 28, 16, 5, tzinfo=TZ)
        daily_df = _make_daily_df([("2026-09-25", 3010.0, 2995.0, 3011.0)])
        remaining, closed = acp.evaluate_exits(
            [dict(position)], d_plus_2,
            download_fn=lambda t, period=None: daily_df,
            download_5m_fn=lambda t: (_ for _ in ()).throw(AssertionError("rollbackで決済済みのはずなのでintraday取得は不要")),
        )
        self.assertEqual(remaining, [])
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "HOLD_LIMIT")
        self.assertEqual(closed[0]["exit_date"], "2026-09-25", "exit_date = D+1(限界日)")
        self.assertEqual(closed[0]["exit_price"], 3011.0)

    # (vi) 実在の東証休日(2026年9月の敬老の日・国民の休日・秋分の日の3連休)を
    # 保有期間が跨いでも、営業日ベースで正しくカウントされること。
    # 素朴に暦日/土日のみ除外でカウントすると9/18,19,20,21,22の5日で
    # H3が誤って早期発火してしまうが、祝日を正しく除外すれば実際の営業日は
    # 9/18・9/24・9/25の3日でH3に到達する。
    def test_vi_holiday_cluster_counted_correctly_via_rollback(self):
        position = _hold_limit_position("2026-09-17", hold_days=3)
        now = datetime(2026, 9, 28, 16, 5, tzinfo=TZ)  # 9/18,9/24,9/25の3営業日分をrollbackで遡る
        daily_df = _make_daily_df([
            ("2026-09-18", 3010.0, 2995.0, 3001.0),  # held=1
            ("2026-09-24", 3010.0, 2995.0, 3002.0),  # held=2 (9/19-23はholidayとしてスキップされる)
            ("2026-09-25", 3010.0, 2995.0, 3003.0),  # held=3=上限
        ])
        remaining, closed = acp.evaluate_exits(
            [dict(position)], now,
            download_fn=lambda t, period=None: daily_df,
            download_5m_fn=lambda t: (_ for _ in ()).throw(AssertionError("rollbackで決済済みのはずなのでintraday取得は不要")),
        )
        self.assertEqual(remaining, [])
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "HOLD_LIMIT")
        self.assertEqual(closed[0]["exit_date"], "2026-09-25", "9/18や9/24ではなく、正しく3営業日目の9/25で発火する")
        self.assertEqual(closed[0]["exit_price"], 3003.0)


# =====================================================================
# Release I/O モック基盤
# =====================================================================

class FakeReleaseServer:
    """subprocess.runをモックして、curl(ダウンロード)とgh(アップロード/一覧)を
    メモリ上のフェイクReleaseストレージに対して動かす。実ネットワークは
    一切使わない。
    """

    def __init__(self):
        # tag -> {asset_name: (status:int|'TIMEOUT', content:bytes|None)}
        self.tags = {}
        self.calls = []

    def set_asset(self, tag, asset_name, status, content=None):
        self.tags.setdefault(tag, {})[asset_name] = (status, content)

    def assets_for(self, tag):
        return list(self.tags.get(tag, {}).keys())

    def __call__(self, cmd, capture_output=True, text=True, check=False, **kwargs):
        self.calls.append(list(cmd))
        if cmd[0] == "curl":
            dest = cmd[cmd.index("-o") + 1]
            url = cmd[-1]
            tag, asset_name = url.rsplit("/", 2)[-2:]
            entry = self.tags.get(tag, {}).get(asset_name)
            if entry is None:
                return subprocess.CompletedProcess(cmd, returncode=0, stdout="404", stderr="")
            status, content = entry
            if status == "TIMEOUT":
                raise subprocess.TimeoutExpired(cmd, 30)
            if content is not None:
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                with open(dest, mode) as f:
                    f.write(content)
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=str(status), stderr="")
        if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "upload":
            tag, local_path = cmd[3], cmd[4]
            asset_name = os.path.basename(local_path)
            with open(local_path, "rb") as f:
                content = f.read()
            self.set_asset(tag, asset_name, 200, content)
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "create":
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "view":
            tag = cmd[3]
            names = "\n".join(self.assets_for(tag))
            rc = 0 if tag in self.tags else 1
            return subprocess.CompletedProcess(cmd, returncode=rc, stdout=names, stderr="")
        raise AssertionError(f"unexpected command in FakeReleaseServer: {cmd}")


# =====================================================================
# State往復・破損復旧
# =====================================================================

class StateRoundTrip(TmpDirMixin, unittest.TestCase):
    def test_write_serialize_reparse_produces_identical_positions(self):
        server = FakeReleaseServer()
        state = {"positions": [
            {"trade_id": "2026-09-25|7203.T|BUY|2026-09-25|hash", "ticker": "7203.T", "entry_price": 3000.0},
        ]}
        with patch("all_candidates_paper.subprocess.run", side_effect=server):
            acp.promote_and_upload_state(state, work_dir=".", upload=True)

        with open(acp.STATE_ASSET_NAME, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk, state)

        uploaded = server.tags[acp.RELEASE_TAG_STATE][acp.STATE_ASSET_NAME][1]
        self.assertEqual(json.loads(uploaded), state)


class CorruptStateRecoversFromBak(TmpDirMixin, unittest.TestCase):
    def test_corrupt_primary_valid_bak_recovers_and_warns(self):
        server = FakeReleaseServer()
        good_state = {"positions": [{"trade_id": "abc", "ticker": "7203.T"}]}
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 200, "{not valid json")
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_BAK_ASSET_NAME, 200, json.dumps(good_state))

        warnings = []
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            state, source = acp.fetch_state(work_dir=".", notify=lambda m: warnings.append(m))

        self.assertEqual(state, good_state)
        self.assertEqual(source, "bak_recovered")
        self.assertEqual(len(warnings), 1)
        self.assertIn("破損", warnings[0])


class CorruptStateAndCorruptBakHardFails(TmpDirMixin, unittest.TestCase):
    def test_both_corrupt_raises_and_does_not_reset_to_empty(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 200, "{not valid json")
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_BAK_ASSET_NAME, 200, "also not valid json")

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.StateCorruptError):
                acp.fetch_state(work_dir=".")

    def test_primary_corrupt_bak_missing_raises(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 200, "{not valid json")
        # bak asset simply not registered -> curl returns 404 for it

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.StateCorruptError):
                acp.fetch_state(work_dir=".")


class Http404IsLegitimateEmptyState(TmpDirMixin, unittest.TestCase):
    def test_404_on_state_initializes_empty_not_an_error(self):
        server = FakeReleaseServer()  # nothing registered -> 404 for every asset

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            state, source = acp.fetch_state(work_dir=".")

        self.assertEqual(state, acp.default_state())
        self.assertEqual(source, "initialized_empty")


class NonNotFoundFailureIsHardFailure(TmpDirMixin, unittest.TestCase):
    def test_500_after_retries_raises_and_does_not_fall_back_to_empty(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 500, "internal error")

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.ReleaseFetchError):
                acp.fetch_state(work_dir=".")

    def test_timeout_after_retries_raises(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, "TIMEOUT")

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.ReleaseFetchError):
                acp.fetch_state(work_dir=".")

    def test_retries_exactly_three_times_before_raising(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 503, "unavailable")
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.ReleaseFetchError):
                acp.download_release_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, acp.STATE_ASSET_NAME, retries=3, retry_wait=0)
        curl_calls = [c for c in server.calls if c[0] == "curl"]
        self.assertEqual(len(curl_calls), 3)


# =====================================================================
# BUG2回帰テスト: curlが既存アセットの302リダイレクトを追跡すること
# (-Lが無いと、既存アセットのdownloadは常にstatus=302を返し、200でも
# 404でもないため非404失敗としてリトライ→ハード失敗になっていた)
# =====================================================================

class CurlFollowsRedirectsForExistingAssets(TmpDirMixin, unittest.TestCase):
    def test_curl_invocation_includes_dash_L(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="200", stderr="")

        with patch("all_candidates_paper.subprocess.run", side_effect=fake_run):
            status, err = acp._curl_download_with_status("https://example.invalid/x", "acp_curl_L_test")

        self.assertIsNone(err)
        self.assertEqual(status, 200)
        self.assertIn("-L", captured["cmd"])
        # -L must come before the URL (last arg) to actually apply to this request
        self.assertLess(captured["cmd"].index("-L"), len(captured["cmd"]) - 1)

    def test_a_302_that_is_never_followed_would_be_treated_as_a_retryable_failure(self):
        # このテストは「-Lを外すとどうなるか」を明示するための回帰ガード:
        # curlがリダイレクトを追跡せずstatus=302を返した場合、
        # download_release_asset()は200でも404でもないため必ず非404失敗
        # (=ReleaseFetchError)として扱われることを確認する。
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 302, "")
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.ReleaseFetchError):
                acp.download_release_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, acp.STATE_ASSET_NAME, retries=3, retry_wait=0)


# =====================================================================
# BUG1回帰テスト: 集計は「今回アップロードした月」の再リストに依存しない
# (アップロード直後にgh release view --json assetsで再リストすると
# eventual-consistencyラグでその月が一覧に載らず、集計が空になっていた)
# =====================================================================

class StaleListFakeReleaseServer(FakeReleaseServer):
    """gh release view(一覧取得)は常に「何もアップロードされていない」
    かのように空を返す(returncode=1)。curlでの個別ダウンロードや
    gh release uploadは通常どおり動く。「直前にアップロードした資産が
    一覧にまだ反映されない」eventual-consistencyラグを再現するフェイク。
    """

    def __call__(self, cmd, capture_output=True, text=True, check=False, **kwargs):
        if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "view":
            self.calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")
        return super().__call__(cmd, capture_output=capture_output, text=text, check=check, **kwargs)


class SummaryUsesInMemoryTodayDataNotStaleRelist(TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), "all_candidates_frozen_policy.json")
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), "all_candidates_frozen_policy_up.json")

    def test_todays_just_uploaded_month_appears_in_summary_despite_stale_asset_list(self):
        server = StaleListFakeReleaseServer()
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                       "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"}]

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"), \
             patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", return_value=(candidates, 100)):
            result = acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=".")

        # gh release viewは呼ばれたが、常に空を返す(=一覧に依存していたら
        # 集計は0件になっていたはず)ことを確認したうえで、実際の集計結果を検証する
        view_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "view"]
        self.assertTrue(len(view_calls) >= 1)

        daily = result["daily_summary"]
        today_all = daily[(daily["date"] == "2026-09-25") & (daily["bucket"] == "ALL")]
        self.assertEqual(len(today_all), 1)
        self.assertEqual(int(today_all.iloc[0]["candidate_count"]), 1)

        monthly = result["monthly_summary"]
        month_all = monthly[(monthly["month"] == "2026-09") & (monthly["bucket"] == "ALL")]
        self.assertEqual(len(month_all), 1)
        self.assertEqual(int(month_all.iloc[0]["candidate_count"]), 1)

    def test_prior_months_untouched_this_run_still_come_from_download_all_months(self):
        # 8月分は今回アップロードしていない(=既に決済済みでopenポジションが
        # 無い)月として、run開始時に取得したprior_monthsのリストに基づいて
        # download_all_months()で正しく取り込まれることを確認する。
        server = StaleListFakeReleaseServer()
        aug_row = {
            "trade_id": "2026-08-20|9984.T|BUY|2026-08-20|priorhash", "date": "2026-08-20",
            "entry_date": "2026-08-20", "ticker": "9984.T", "direction": "BUY", "rank": 1,
            "score": 70.0, "up_probability": 55.0, "down_probability": 20.0, "nikkei_filter": False,
            "policy_file": "all_candidates_frozen_policy.json", "policy_hash": "priorhash",
            "trend_down_flag": False, "entry_price": 3000.0, "tp": 3100.0, "sl": 2950.0, "hold_days": 1,
            "entry_time": "09:05", "exit_price": 3100.0, "exit_time": "10:00", "exit_date": "2026-08-21",
            "exit_reason": "TP", "return_pct": 3.3,
        }
        # 8月分アセットを直接サーバへ登録(=今回のrunより前から存在する
        # プライベート済みの過去月データ、というシナリオ)。
        aug_df = acp.rows_to_dataframe([aug_row])
        import io
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(aug_df.to_csv(index=False).encode("utf-8"))
        server.set_asset(acp.RELEASE_TAG_DATA, acp.month_asset_name("2026-08-01"), 200, buf.getvalue())

        # このシナリオではlist_release_assets自体は正常(8月分は既に
        # プロパゲーション済み)なので、list_release_assetsだけ直接差し替える。
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"), \
             patch.object(acp, "list_release_assets", return_value=[acp.month_asset_name("2026-08-01")]), \
             patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", return_value=([], 0)):
            result = acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=".")

        daily = result["daily_summary"]
        aug_all = daily[(daily["date"] == "2026-08-20") & (daily["bucket"] == "ALL")]
        self.assertEqual(len(aug_all), 1)
        self.assertEqual(int(aug_all.iloc[0]["candidate_count"]), 1)
        self.assertEqual(int(aug_all.iloc[0]["trades_closed"]), 1)


# =====================================================================
# リトライ/再実行時の非重複
# =====================================================================

class RerunDoesNotDuplicateTrades(unittest.TestCase):
    def test_reopen_path_dedup_on_rerun(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                       "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"}]
        policy = {"nikkei_filter": False, "hold_days": 1}
        first_run = acp.build_new_positions(set(), candidates, "2026-09-25", policy, "f.json", "hash", False)
        known_ids = {p["trade_id"] for p in first_run}
        # simulate a same-day workflow retry re-scanning the identical candidate
        second_run = acp.build_new_positions(known_ids, candidates, "2026-09-25", policy, "f.json", "hash", False)
        self.assertEqual(len(first_run), 1)
        self.assertEqual(len(second_run), 0)

    def test_close_path_dedup_via_monthly_append(self):
        server = FakeReleaseServer()
        row = {
            "trade_id": "2026-09-25|7203.T|BUY|2026-09-25|hash", "date": "2026-09-25", "entry_date": "2026-09-25",
            "ticker": "7203.T", "direction": "BUY", "rank": 1, "score": 80.0, "up_probability": 60.0,
            "down_probability": 10.0, "nikkei_filter": False, "policy_file": "f.json", "policy_hash": "hash",
            "trend_down_flag": False, "entry_price": 3000.0, "tp": 3100.0, "sl": 2950.0, "hold_days": 1,
            "entry_time": "09:05", "exit_price": 3100.0, "exit_time": "10:00", "exit_date": "2026-09-25",
            "exit_reason": "TP", "return_pct": 3.3,
        }
        with tempfile.TemporaryDirectory() as work_dir, patch("all_candidates_paper.subprocess.run", side_effect=server):
            df = acp.rows_to_dataframe([row])
            acp.append_month_rows(df, "2026-09", work_dir=work_dir)
            # simulate retry: append the exact same closed row again
            acp.append_month_rows(df, "2026-09", work_dir=work_dir)
            local_path = os.path.join(work_dir, acp.month_asset_name("2026-09-01"))
            with gzip.open(local_path, "rt", encoding="utf-8") as f:
                final = pd.read_csv(f)
        self.assertEqual(len(final), 1)
        self.assertEqual(final.iloc[0]["trade_id"], row["trade_id"])


# =====================================================================
# 月次ローテーション
# =====================================================================

class MonthlyRolloverCreatesNewAssetWithoutTouchingPrevious(unittest.TestCase):
    def test_new_month_gets_distinct_asset_previous_month_untouched(self):
        server = FakeReleaseServer()
        aug_row = {"trade_id": "2026-08-20|7203.T|BUY|2026-08-20|hash", "date": "2026-08-20"}
        sep_row = {"trade_id": "2026-09-03|7203.T|BUY|2026-09-03|hash", "date": "2026-09-03"}

        with tempfile.TemporaryDirectory() as work_dir, patch("all_candidates_paper.subprocess.run", side_effect=server):
            acp.append_month_rows(acp.rows_to_dataframe([aug_row]), "2026-08", work_dir=work_dir)
            aug_asset = acp.month_asset_name("2026-08-01")
            sep_asset = acp.month_asset_name("2026-09-01")
            self.assertNotEqual(aug_asset, sep_asset)
            aug_uploaded_before = server.tags[acp.RELEASE_TAG_DATA][aug_asset][1]

            acp.append_month_rows(acp.rows_to_dataframe([sep_row]), "2026-09", work_dir=work_dir)

            self.assertIn(sep_asset, server.tags[acp.RELEASE_TAG_DATA])
            aug_uploaded_after = server.tags[acp.RELEASE_TAG_DATA][aug_asset][1]
            self.assertEqual(aug_uploaded_before, aug_uploaded_after)  # untouched by September's append

            with gzip.open(os.path.join(work_dir, aug_asset), "rt", encoding="utf-8") as f:
                aug_final = pd.read_csv(f)
            with gzip.open(os.path.join(work_dir, sep_asset), "rt", encoding="utf-8") as f:
                sep_final = pd.read_csv(f)
        self.assertEqual(list(aug_final["trade_id"]), [aug_row["trade_id"]])
        self.assertEqual(list(sep_final["trade_id"]), [sep_row["trade_id"]])


# =====================================================================
# ALL/TOP1/TOP3/TOP5は同一scan結果由来
# =====================================================================

class BucketsShareSingleScanResult(unittest.TestCase):
    def test_buckets_are_prefixes_of_the_same_rank_ordering(self):
        rows = []
        for rank in range(1, 8):
            rows.append({
                "trade_id": f"t{rank}", "date": "2026-09-25", "ticker": f"T{rank}", "direction": "BUY",
                "rank": rank, "exit_price": 3100.0, "return_pct": 1.0 * rank,
            })
        df = acp.rows_to_dataframe(rows)
        summary = acp.compute_daily_summary(df)
        counts = {row["bucket"]: row["candidate_count"] for _, row in summary.iterrows()}
        self.assertEqual(counts["ALL"], 7)
        self.assertEqual(counts["TOP1"], 1)
        self.assertEqual(counts["TOP3"], 3)
        self.assertEqual(counts["TOP5"], 5)

        top3_tickers = set(df[df["rank"] <= 3]["ticker"])
        top5_tickers = set(df[df["rank"] <= 5]["ticker"])
        self.assertTrue(top3_tickers.issubset(top5_tickers))
        self.assertTrue(top5_tickers.issubset(set(df["ticker"])))

    def test_run_calls_scan_exactly_once_regardless_of_bucket_count(self):
        with tempfile.TemporaryDirectory() as work_dir:
            shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), os.path.join(work_dir, "all_candidates_frozen_policy.json"))
            shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), os.path.join(work_dir, "all_candidates_frozen_policy_up.json"))
            fake_scan = MagicMock(return_value=([], 0))
            with patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
                 patch.object(acp, "scan", fake_scan), \
                 patch.object(acp, "fetch_state", return_value=(acp.default_state(), "initialized_empty")), \
                 patch.object(acp, "promote_and_upload_state"), \
                 patch.object(acp, "list_release_assets", return_value=[]), \
                 patch.object(acp, "append_trade_rows", return_value={}), \
                 patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
                acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=work_dir)
        self.assertEqual(fake_scan.call_count, 1)


# =====================================================================
# 集計ロジック(equal-weight simulation)の妥当性
# =====================================================================

class EqualWeightSimulationSanity(unittest.TestCase):
    def test_equal_weight_pnl_uses_candidate_count_as_denominator(self):
        rows = [
            {"trade_id": "a", "date": "2026-09-25", "ticker": "A", "direction": "BUY", "rank": 1,
             "exit_price": 3100.0, "return_pct": 10.0},
            {"trade_id": "b", "date": "2026-09-25", "ticker": "B", "direction": "BUY", "rank": 2,
             "exit_price": None, "return_pct": None},  # still open
        ]
        df = acp.rows_to_dataframe(rows)
        summary = acp.compute_daily_summary(df)
        all_row = summary[summary["bucket"] == "ALL"].iloc[0]
        self.assertEqual(all_row["candidate_count"], 2)
        self.assertEqual(all_row["trades_closed"], 1)
        expected_pnl = (acp.EQUAL_WEIGHT_CAPITAL / 2) * (10.0 / 100)
        self.assertAlmostEqual(all_row["equal_weight_pnl_jpy"], expected_pnl)


# =====================================================================
# 同日冪等化ガード(GitHub Actionsのcron遅延により同日中に2回runされうる
# ケースの回帰防止。理由はall_candidates_paper.py run()内のコメント参照)
# =====================================================================

class SameDayIdempotencyGuard(unittest.TestCase):
    def _patched_run(self, work_dir, state_holder, scan_mock, select_policy_return=("strategy_policy.json", {"trend": "up"})):
        """fetch_state/promote_and_upload_stateをstate_holder(dict, key 'state')に
        対する読み書きとして振る舞わせ、scanはscan_mockに差し替えたrun()呼び出しの
        コンテキストマネージャを返す。append_trade_rows/download_all_monthsは
        空データとして扱う(このテスト群の関心はstate/scan呼び出し回数のみ)。
        """
        def fake_fetch_state(wd):
            return dict(state_holder["state"]), "primary"

        def fake_promote(new_state, work_dir=None, upload=True):
            state_holder["state"] = new_state
            state_holder["uploaded_count"] = state_holder.get("uploaded_count", 0) + 1

        return (
            patch.object(acp, "select_policy_file", return_value=select_policy_return),
            patch.object(acp, "scan", scan_mock),
            patch.object(acp, "fetch_state", side_effect=fake_fetch_state),
            patch.object(acp, "promote_and_upload_state", side_effect=fake_promote),
            patch.object(acp, "list_release_assets", return_value=[]),
            patch.object(acp, "append_trade_rows", return_value={}),
            patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])),
        )

    def setUp(self):
        self.work_dir_ctx = tempfile.TemporaryDirectory()
        self.work_dir = self.work_dir_ctx.name
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), os.path.join(self.work_dir, "all_candidates_frozen_policy.json"))
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), os.path.join(self.work_dir, "all_candidates_frozen_policy_up.json"))

    def tearDown(self):
        self.work_dir_ctx.cleanup()

    def test_second_run_same_day_makes_zero_scan_calls_and_skips(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                       "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"}]
        scan_mock = MagicMock(return_value=(candidates, 100))
        state_holder = {"state": acp.default_state()}
        patches = self._patched_run(self.work_dir, state_holder, scan_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            first = acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=self.work_dir)
            self.assertNotIn("skipped", first)
            self.assertEqual(scan_mock.call_count, 1)
            self.assertEqual(state_holder["state"].get("last_completed_run_date"), "2026-09-25")

            daily_path = os.path.join(self.work_dir, acp.DAILY_SUMMARY_FILE)
            monthly_path = os.path.join(self.work_dir, acp.MONTHLY_SUMMARY_FILE)
            with open(daily_path, encoding="utf-8-sig") as f:
                daily_before = f.read()
            with open(monthly_path, encoding="utf-8-sig") as f:
                monthly_before = f.read()
            uploads_before = state_holder["uploaded_count"]

            second = acp.run(now=datetime(2026, 9, 25, 21, 30, tzinfo=TZ), work_dir=self.work_dir)

            self.assertEqual(second, {"today": "2026-09-25", "skipped": "already_completed_today"})
            self.assertEqual(scan_mock.call_count, 1)  # still just the first run's call
            self.assertEqual(state_holder["uploaded_count"], uploads_before)  # no second upload

            with open(daily_path, encoding="utf-8-sig") as f:
                daily_after = f.read()
            with open(monthly_path, encoding="utf-8-sig") as f:
                monthly_after = f.read()
            self.assertEqual(daily_before, daily_after)
            self.assertEqual(monthly_before, monthly_after)

    def test_midway_failure_does_not_mark_day_done_second_run_proceeds(self):
        state_holder = {"state": acp.default_state()}
        scan_mock = MagicMock(side_effect=RuntimeError("boom"))
        patches = self._patched_run(self.work_dir, state_holder, scan_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            with self.assertRaises(RuntimeError):
                acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=self.work_dir)
            self.assertNotIn("last_completed_run_date", state_holder["state"])
            self.assertEqual(state_holder.get("uploaded_count", 0), 0)

            scan_mock.side_effect = None
            scan_mock.return_value = ([], 0)
            second = acp.run(now=datetime(2026, 9, 25, 21, 30, tzinfo=TZ), work_dir=self.work_dir)
            self.assertNotIn("skipped", second)
            self.assertEqual(scan_mock.call_count, 2)
            self.assertEqual(state_holder["state"].get("last_completed_run_date"), "2026-09-25")

    def test_next_trading_day_proceeds_normally(self):
        state_holder = {"state": {"positions": [], "last_completed_run_date": "2026-09-25"}}
        scan_mock = MagicMock(return_value=([], 0))
        patches = self._patched_run(self.work_dir, state_holder, scan_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            result = acp.run(now=datetime(2026, 9, 28, 16, 10, tzinfo=TZ), work_dir=self.work_dir)
        self.assertNotIn("skipped", result)
        self.assertEqual(scan_mock.call_count, 1)
        self.assertEqual(state_holder["state"].get("last_completed_run_date"), "2026-09-28")

    def test_state_without_field_backward_compat_proceeds(self):
        # 実運用中のstate資産(2026-09-25時点)と同じ形: positionsキーのみで
        # last_completed_run_dateフィールドが存在しない。
        real_shape_state = {
            "positions": [
                {
                    "trade_id": "2026-09-25|1721.T|BUY|2026-09-25|cfe6da4cc960baeda7f3d6e581a4937b059c96347255b2cd8a04c6438e1c07de",
                    "date": "2026-09-25", "ticker": "1721.T", "direction": "BUY", "rank": 1,
                    "score": 70.43253657525702, "up_probability": 34.95507933122712,
                    "down_probability": 26.67643794681185, "nikkei_filter": False,
                    "policy_file": "all_candidates_frozen_policy_up.json",
                    "policy_hash": "cfe6da4cc960baeda7f3d6e581a4937b059c96347255b2cd8a04c6438e1c07de",
                    "trend_down_flag": False, "entry_price": 5606.0, "tp": 6007.475772361155,
                    "sl": 5472.174742546282, "hold_days": 3, "entry_date": "2026-09-25", "entry_time": "20:29",
                },
            ],
        }
        self.assertNotIn("last_completed_run_date", real_shape_state)
        state_holder = {"state": real_shape_state}
        scan_mock = MagicMock(return_value=([], 0))

        def fake_fetch_state(wd):
            return dict(state_holder["state"]), "primary"

        def fake_promote(new_state, work_dir=None, upload=True):
            state_holder["state"] = new_state

        with patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", scan_mock), \
             patch.object(acp, "fetch_state", side_effect=fake_fetch_state), \
             patch.object(acp, "promote_and_upload_state", side_effect=fake_promote), \
             patch.object(acp, "evaluate_exits", return_value=([], [])), \
             patch.object(acp, "list_release_assets", return_value=[]), \
             patch.object(acp, "append_trade_rows", return_value={}), \
             patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
            result = acp.run(now=datetime(2026, 9, 28, 16, 10, tzinfo=TZ), work_dir=self.work_dir)

        self.assertNotIn("skipped", result)
        self.assertEqual(scan_mock.call_count, 1)
        self.assertEqual(state_holder["state"].get("last_completed_run_date"), "2026-09-28")

    def test_holiday_gate_still_skips_before_state_check(self):
        scan_mock = MagicMock(return_value=([], 0))
        with patch.object(acp, "scan", scan_mock), \
             patch.object(acp, "fetch_state") as fetch_mock:
            result = acp.run(now=datetime(2026, 9, 26, 16, 10, tzinfo=TZ), work_dir=self.work_dir)  # Saturday
        self.assertEqual(result, {"today": "2026-09-26", "skipped": "not_a_trading_day"})
        fetch_mock.assert_not_called()
        scan_mock.assert_not_called()


# =====================================================================
# 実行時刻ウィンドウガード(2026-09-29のインシデント回帰防止: schedule:の
# 大幅遅延で日付をまたいで発火した場合に、翌営業日の寄り前を誤って
# 「今日」として処理してしまうバグの再発防止)
# =====================================================================

class RunTimeWindowGuard(unittest.TestCase):
    def setUp(self):
        self.work_dir_ctx = tempfile.TemporaryDirectory()
        self.work_dir = self.work_dir_ctx.name
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"), os.path.join(self.work_dir, "all_candidates_frozen_policy.json"))
        shutil.copyfile(os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"), os.path.join(self.work_dir, "all_candidates_frozen_policy_up.json"))

    def tearDown(self):
        self.work_dir_ctx.cleanup()

    def _run_with_guards(self, now, scan_mock=None):
        scan_mock = scan_mock or MagicMock(return_value=([], 0))
        with patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", scan_mock), \
             patch.object(acp, "fetch_state", return_value=(acp.default_state(), "initialized_empty")) as fetch_mock, \
             patch.object(acp, "promote_and_upload_state") as promote_mock, \
             patch.object(acp, "list_release_assets", return_value=[]), \
             patch.object(acp, "append_trade_rows", return_value={}), \
             patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
            result = acp.run(now=now, work_dir=self.work_dir)
        return result, scan_mock, fetch_mock, promote_mock

    def test_post_midnight_run_is_skipped(self):
        # 2026-09-29 00:18 JST -- 実際に発生したインシデントの発火時刻。
        result, scan_mock, fetch_mock, promote_mock = self._run_with_guards(
            datetime(2026, 9, 29, 0, 18, tzinfo=TZ)
        )
        self.assertEqual(result, {"today": "2026-09-29", "skipped": "outside_run_window"})
        fetch_mock.assert_not_called()
        scan_mock.assert_not_called()
        promote_mock.assert_not_called()

    def test_15_34_is_skipped_one_minute_before_window(self):
        result, scan_mock, fetch_mock, promote_mock = self._run_with_guards(
            datetime(2026, 9, 25, 15, 34, tzinfo=TZ)
        )
        self.assertEqual(result, {"today": "2026-09-25", "skipped": "outside_run_window"})
        fetch_mock.assert_not_called()
        scan_mock.assert_not_called()
        promote_mock.assert_not_called()

    def test_15_35_lower_boundary_proceeds(self):
        result, scan_mock, fetch_mock, promote_mock = self._run_with_guards(
            datetime(2026, 9, 25, 15, 35, tzinfo=TZ)
        )
        self.assertNotIn("skipped", result)
        scan_mock.assert_called_once()

    def test_23_59_upper_boundary_proceeds(self):
        result, scan_mock, fetch_mock, promote_mock = self._run_with_guards(
            datetime(2026, 9, 25, 23, 59, tzinfo=TZ)
        )
        self.assertNotIn("skipped", result)
        scan_mock.assert_called_once()

    def test_same_day_guard_still_applies_inside_window(self):
        state_holder = {"state": acp.default_state()}

        def fake_fetch_state(wd):
            return dict(state_holder["state"]), "primary"

        def fake_promote(new_state, work_dir=None, upload=True):
            state_holder["state"] = new_state

        scan_mock = MagicMock(return_value=([], 0))
        with patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})), \
             patch.object(acp, "scan", scan_mock), \
             patch.object(acp, "fetch_state", side_effect=fake_fetch_state), \
             patch.object(acp, "promote_and_upload_state", side_effect=fake_promote), \
             patch.object(acp, "list_release_assets", return_value=[]), \
             patch.object(acp, "append_trade_rows", return_value={}), \
             patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
            first = acp.run(now=datetime(2026, 9, 25, 15, 35, tzinfo=TZ), work_dir=self.work_dir)
            self.assertNotIn("skipped", first)
            second = acp.run(now=datetime(2026, 9, 25, 23, 59, tzinfo=TZ), work_dir=self.work_dir)
        self.assertEqual(second, {"today": "2026-09-25", "skipped": "already_completed_today"})
        self.assertEqual(scan_mock.call_count, 1)

    def test_holiday_gate_runs_before_window_gate(self):
        # 2026-09-26は土曜日かつウィンドウ外(00:18)。holidayガードが先に
        # 判定される(=skipped理由がoutside_run_windowではなくnot_a_trading_day)
        # ことを確認する。
        result, scan_mock, fetch_mock, promote_mock = self._run_with_guards(
            datetime(2026, 9, 26, 0, 18, tzinfo=TZ)
        )
        self.assertEqual(result, {"today": "2026-09-26", "skipped": "not_a_trading_day"})
        fetch_mock.assert_not_called()
        scan_mock.assert_not_called()


# =====================================================================
# profit_top10_paper.py はゼロ差分であること
# =====================================================================

class LiveFileZeroDiff(unittest.TestCase):
    def test_profit_top10_paper_has_zero_diff_vs_head(self):
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", "profit_top10_paper.py"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(), "", f"profit_top10_paper.py has uncommitted diff: {result.stdout}")


if __name__ == "__main__":
    unittest.main()
