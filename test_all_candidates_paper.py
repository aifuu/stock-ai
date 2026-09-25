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
        fake_result = {"trend": "up", "reason": "test", "source": "futures"}
        with patch("profit_top10_paper.futures_trend.detect_futures_trend", return_value=fake_result), \
             patch("profit_top10_paper.futures_trend.log_daily_trend"), \
             patch("profit_top10_paper.os.path.exists", return_value=True):
            live_file, live_result = live.select_policy_file()
        self.assertEqual(live_file, live.POLICY_FILE_UP)
        self.assertEqual(acp.choose_frozen_policy_file(live_file), acp.FROZEN_POLICY_FILE_UP)
        self.assertEqual(live_result["trend"], "up")

    def test_matches_live_select_policy_file_down_case_falls_back_to_normal(self):
        fake_result = {"trend": "down", "reason": "test", "source": "futures"}
        # strategy_policy_down.json は存在しない -> live自体もPOLICY_FILEへフォールバックする
        with patch("profit_top10_paper.futures_trend.detect_futures_trend", return_value=fake_result), \
             patch("profit_top10_paper.futures_trend.log_daily_trend"), \
             patch("profit_top10_paper.os.path.exists", return_value=False):
            live_file, live_result = live.select_policy_file()
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
             patch.object(acp, "append_trade_rows", return_value={}), \
             patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
            acp.run(now=datetime(2026, 9, 25, 15, 20, tzinfo=TZ), work_dir=".")

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
                 patch.object(acp, "append_trade_rows", return_value={}), \
                 patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])):
                acp.run(now=datetime(2026, 9, 25, 15, 20, tzinfo=TZ), work_dir=work_dir)
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
