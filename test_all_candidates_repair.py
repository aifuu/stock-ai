"""all_candidates_repair.py (all_candidates_paper.py緊急データ修復ツール)のテスト。

すべてI/O(subprocess.run)をFakeReleaseServer(test_all_candidates_paper.py
と共有)でモックする。実ネットワーク呼び出しは一切行わない。

検証する不変条件:
  - 指定したtrade_idの行だけがstate/月次データから消え、他は一切変わらない
  - 未知のtrade_idを1件でも含めるとハード失敗し、何もアップロードされない
  - 6件以上指定すると拒否され、何もダウンロード/アップロードされない
  - dry_run=True(既定)ではアップロード・ローカルファイル書き込みが一切ない
  - バックアップは「他の変更(state/月次データの再アップロード)より前に」
    アップロードされ、再ダウンロードしてsha256照合される
  - 集計CSV(daily/monthly)が削除後のデータと整合する
"""
import gzip
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import all_candidates_paper as acp
import all_candidates_repair as acr
from test_all_candidates_paper import FakeReleaseServer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="acr_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _position(trade_id, ticker="7203.T", direction="BUY", entry_date="2026-09-25", **extra):
    p = {
        "trade_id": trade_id, "date": entry_date, "ticker": ticker, "direction": direction,
        "rank": 1, "score": 80.0, "up_probability": 60.0, "down_probability": 10.0,
        "nikkei_filter": False, "policy_file": "all_candidates_frozen_policy.json",
        "policy_hash": "hash123", "trend_down_flag": False, "entry_price": 3000.0,
        "tp": 3100.0, "sl": 2950.0, "hold_days": 1, "entry_date": entry_date, "entry_time": "00:20",
    }
    p.update(extra)
    return p


def _trade_row(trade_id, ticker="7203.T", direction="BUY", date="2026-09-25", closed=False, **extra):
    row = _position(trade_id, ticker=ticker, direction=direction, entry_date=date)
    if closed:
        row.update(exit_price=3100.0, exit_time="10:00", exit_date=date, exit_reason="TP", return_pct=3.3)
    else:
        row.update(exit_price=None, exit_time=None, exit_date=None, exit_reason=None, return_pct=None)
    row.update(extra)
    return row


def _gzip_csv_bytes(df):
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    return buf.getvalue()


def _seed_server(server, positions, month_rows_by_month):
    """serverへstate.json + 月次csv.gzを直接登録する(populated fixture)。"""
    state = {"positions": positions, "last_completed_run_date": "2026-09-29"}
    server.set_asset(acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME, 200, json.dumps(state))
    for month_key, rows in month_rows_by_month.items():
        df = acp.rows_to_dataframe(rows)
        server.set_asset(acp.RELEASE_TAG_DATA, acp.month_asset_name(month_key + "-01"), 200, _gzip_csv_bytes(df))
    return state


class RemovesExactlyTargetRowsAndNothingElse(TmpDirMixin, unittest.TestCase):
    def test_removes_target_position_keeps_five_others_byte_equal(self):
        server = FakeReleaseServer()
        others = [_position(f"good-{i}", ticker=f"T{i}", entry_date="2026-09-25") for i in range(5)]
        bogus = _position("bogus-543A", ticker="543A.T", direction="SHORT", entry_date="2026-09-29",
                           entry_time="00:20", rank=1, hold_days=1)
        positions = others + [bogus]
        rows = [_trade_row(p["trade_id"], ticker=p["ticker"], direction=p["direction"], date=p["entry_date"])
                for p in positions]
        _seed_server(server, positions, {"2026-09": rows})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            result = acr.run_repair(["bogus-543A"], reason="test", dry_run=False, work_dir=".")

        self.assertEqual(len(result["removed_positions"]), 1)
        self.assertEqual(result["removed_positions"][0]["trade_id"], "bogus-543A")

        uploaded_state = json.loads(server.tags[acp.RELEASE_TAG_STATE][acp.STATE_ASSET_NAME][1])
        remaining_ids = {p["trade_id"] for p in uploaded_state["positions"]}
        self.assertEqual(remaining_ids, {p["trade_id"] for p in others})
        self.assertEqual(len(uploaded_state["positions"]), 5)
        for p in uploaded_state["positions"]:
            original = next(o for o in others if o["trade_id"] == p["trade_id"])
            self.assertEqual(p, original, "他の5ポジションはbyte-equal(内容が完全一致)であること")

        # last_completed_run_date is untouched
        self.assertEqual(uploaded_state["last_completed_run_date"], "2026-09-29")

        month_bytes = server.tags[acp.RELEASE_TAG_DATA][acp.month_asset_name("2026-09-01")][1]
        with gzip.open(__import__("io").BytesIO(month_bytes), "rt", encoding="utf-8") as f:
            final_df = pd.read_csv(f)
        self.assertNotIn("bogus-543A", set(final_df["trade_id"]))
        self.assertEqual(len(final_df), 5)


class UnknownTradeIdHardFailsWithoutUpload(TmpDirMixin, unittest.TestCase):
    def test_unknown_trade_id_raises_and_uploads_nothing(self):
        server = FakeReleaseServer()
        positions = [_position("known-1")]
        _seed_server(server, positions, {"2026-09": [_trade_row("known-1")]})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acr.TradeIdNotFoundError):
                acr.run_repair(["does-not-exist"], reason="test", dry_run=False, work_dir=".")

        upload_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "upload"]
        self.assertEqual(upload_calls, [])

    def test_partially_known_trade_ids_still_hard_fails_nothing_uploaded(self):
        """一部が既知でも、1件でも未知が混じっていれば全体を中断する(部分適用しない)。"""
        server = FakeReleaseServer()
        positions = [_position("known-1")]
        _seed_server(server, positions, {"2026-09": [_trade_row("known-1")]})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acr.TradeIdNotFoundError):
                acr.run_repair(["known-1", "does-not-exist"], reason="test", dry_run=False, work_dir=".")

        upload_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "upload"]
        self.assertEqual(upload_calls, [])


class MoreThanFiveIdsRefused(unittest.TestCase):
    def test_six_ids_refused_before_any_io(self):
        with patch("all_candidates_paper.subprocess.run", side_effect=AssertionError("should never be called")):
            with self.assertRaises(acr.RepairRefusedError):
                acr.run_repair([f"id-{i}" for i in range(6)], reason="test", dry_run=True, work_dir=".")

    def test_exactly_five_ids_is_allowed_to_proceed_to_lookup(self):
        server = FakeReleaseServer()  # empty -> everything 404s -> not-found error, but proves no refusal
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises((acr.TradeIdNotFoundError, RuntimeError)) as ctx:
                acr.run_repair([f"id-{i}" for i in range(5)], reason="test", dry_run=True, work_dir=".")
        self.assertNotIsInstance(ctx.exception, acr.RepairRefusedError)

    def test_empty_list_refused(self):
        with self.assertRaises(acr.RepairRefusedError):
            acr.run_repair([], reason="test", dry_run=True, work_dir=".")

    def test_duplicate_ids_refused(self):
        with self.assertRaises(acr.RepairRefusedError):
            acr.run_repair(["a", "a"], reason="test", dry_run=True, work_dir=".")


class DryRunUploadsAndWritesNothing(TmpDirMixin, unittest.TestCase):
    def test_dry_run_true_zero_uploads_and_no_local_file_mutation(self):
        server = FakeReleaseServer()
        positions = [_position("bogus-1", ticker="543A.T", direction="SHORT", entry_date="2026-09-29")]
        _seed_server(server, positions, {"2026-09": [_trade_row("bogus-1", ticker="543A.T", direction="SHORT", date="2026-09-29")]})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            result = acr.run_repair(["bogus-1"], reason="test", dry_run=True, work_dir=".")

        self.assertEqual(len(result["removed_positions"]), 1)
        upload_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "upload"]
        create_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "create"]
        self.assertEqual(upload_calls, [], "dry_runではアップロードが一切発生しないこと")
        self.assertEqual(create_calls, [], "dry_runではensure_release_existsも呼ばれないこと")
        self.assertFalse(os.path.exists(acp.DAILY_SUMMARY_FILE), "dry_runでは集計CSVも書き込まないこと")
        self.assertFalse(os.path.exists(acp.MONTHLY_SUMMARY_FILE))
        # no pre-repair backup files were ever created locally either
        self.assertFalse(any("pre-repair" in f for f in os.listdir(".")))


class BackupsUploadedAndVerifiedBeforeModification(TmpDirMixin, unittest.TestCase):
    def test_backup_upload_precedes_final_repaired_asset_upload(self):
        server = FakeReleaseServer()
        positions = [_position("bogus-1", ticker="543A.T", direction="SHORT", entry_date="2026-09-29")]
        _seed_server(server, positions, {"2026-09": [_trade_row("bogus-1", ticker="543A.T", direction="SHORT", date="2026-09-29")]})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"), \
             patch("all_candidates_repair.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260929151500"
            acr.run_repair(["bogus-1"], reason="test", dry_run=False, work_dir=".")

        upload_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "upload"]
        uploaded_names = [os.path.basename(c[4]) for c in upload_calls]

        state_backup = f"{acp.STATE_ASSET_NAME}.pre-repair-20260929151500"
        month_backup = f"{acp.month_asset_name('2026-09-01')}.pre-repair-20260929151500"

        self.assertIn(state_backup, uploaded_names)
        self.assertIn(month_backup, uploaded_names)
        self.assertLess(
            uploaded_names.index(state_backup), uploaded_names.index(acp.STATE_ASSET_NAME),
            "state backup must upload before the repaired state itself",
        )
        self.assertLess(
            uploaded_names.index(month_backup), uploaded_names.index(acp.month_asset_name("2026-09-01")),
            "month backup must upload before the repaired month asset itself",
        )

        # verified by re-download: a curl for the backup asset name happened
        curl_calls = [c for c in server.calls if c[0] == "curl"]
        curl_urls = [c[-1] for c in curl_calls]
        self.assertTrue(any(state_backup in u for u in curl_urls), "backup must be re-downloaded to verify sha256")

    def test_backup_sha256_mismatch_aborts_before_touching_originals(self):
        server = FakeReleaseServer()
        positions = [_position("bogus-1", ticker="543A.T", direction="SHORT", entry_date="2026-09-29")]
        _seed_server(server, positions, {"2026-09": [_trade_row("bogus-1", ticker="543A.T", direction="SHORT", date="2026-09-29")]})

        real_call = server.__call__

        def corrupting_call(cmd, **kwargs):
            result = real_call(cmd, **kwargs)
            if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "upload" and "pre-repair" in cmd[4]:
                tag = cmd[3]
                asset_name = os.path.basename(cmd[4])
                status, _content = server.tags[tag][asset_name]
                server.tags[tag][asset_name] = (status, b"corrupted-backup-content")
            return result

        with patch("all_candidates_paper.subprocess.run", side_effect=corrupting_call), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(RuntimeError):
                acr.run_repair(["bogus-1"], reason="test", dry_run=False, work_dir=".")

        # the real (non-backup) state asset must never have been overwritten
        state_asset = server.tags[acp.RELEASE_TAG_STATE][acp.STATE_ASSET_NAME][1]
        state = json.loads(state_asset)
        self.assertEqual({p["trade_id"] for p in state["positions"]}, {"bogus-1"}, "backup検証失敗時は本体を一切変更しない")


class SummariesRecomputedConsistently(TmpDirMixin, unittest.TestCase):
    def test_daily_and_monthly_summary_reflect_removal(self):
        server = FakeReleaseServer()
        others = [_position(f"good-{i}", ticker=f"T{i}", entry_date="2026-09-25") for i in range(5)]
        bogus = _position("bogus-1", ticker="543A.T", direction="SHORT", entry_date="2026-09-29")
        positions = others + [bogus]
        rows = [_trade_row(p["trade_id"], ticker=p["ticker"], direction=p["direction"], date=p["entry_date"], closed=True)
                for p in others]
        rows.append(_trade_row("bogus-1", ticker="543A.T", direction="SHORT", date="2026-09-29", closed=False))
        _seed_server(server, positions, {"2026-09": rows})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            result = acr.run_repair(["bogus-1"], reason="test", dry_run=False, work_dir=".")

        daily = result["daily_summary"]
        self.assertTrue((daily[daily["date"] == "2026-09-29"]).empty, "2026-09-29のコホートは空になったので消えること")
        sep25_all = daily[(daily["date"] == "2026-09-25") & (daily["bucket"] == "ALL")]
        self.assertEqual(int(sep25_all.iloc[0]["candidate_count"]), 5)

        monthly = result["monthly_summary"]
        sep_all = monthly[(monthly["month"] == "2026-09") & (monthly["bucket"] == "ALL")]
        self.assertEqual(int(sep_all.iloc[0]["candidate_count"]), 5)

        # the on-disk CSVs were actually written (workflow later commits these)
        self.assertTrue(os.path.exists(acp.DAILY_SUMMARY_FILE))
        self.assertTrue(os.path.exists(acp.MONTHLY_SUMMARY_FILE))


class StateBakHoldsThePreRepairGenerationNotThePostRepairOne(TmpDirMixin, unittest.TestCase):
    """回帰テスト: promote_and_upload_state()の2世代保存(ディスク上の既存
    state.jsonを.bakへ昇格 -> 新state.jsonを書き込み)を機能させるには、
    それを呼ぶ前にstate.jsonへ新内容を書き込んではならない。先に書き込むと
    「昇格される内容」自体が新state.jsonになってしまい、.bakが修復前の
    スナップショットではなく修復後と同じものになる(2026-09-29の実運用で
    実際に発生したバグ)。
    """

    def test_bak_after_repair_still_has_the_original_six_position_state(self):
        server = FakeReleaseServer()
        others = [_position(f"good-{i}", ticker=f"T{i}", entry_date="2026-09-25") for i in range(5)]
        bogus = _position("bogus-543A", ticker="543A.T", direction="SHORT", entry_date="2026-09-29")
        positions = others + [bogus]
        rows = [_trade_row(p["trade_id"], ticker=p["ticker"], direction=p["direction"], date=p["entry_date"])
                for p in positions]
        _seed_server(server, positions, {"2026-09": rows})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            acr.run_repair(["bogus-543A"], reason="test", dry_run=False, work_dir=".")

        uploaded_bak = json.loads(server.tags[acp.RELEASE_TAG_STATE][acp.STATE_BAK_ASSET_NAME][1])
        bak_ids = {p["trade_id"] for p in uploaded_bak["positions"]}
        self.assertEqual(
            bak_ids, {p["trade_id"] for p in positions},
            "state.bak.jsonは修復前(bogusを含む6件)のスナップショットであること(修復後の5件になっていてはならない)",
        )
        self.assertIn("bogus-543A", bak_ids)


class OnlyFoundInMonthlyDataNotState(TmpDirMixin, unittest.TestCase):
    def test_trade_id_only_in_closed_monthly_row_removes_without_touching_state(self):
        """既にstateから消えた(決済済みでpositionsに残っていない)トレードを
        月次データだけから消す場合、state.jsonへは一切アップロードしない
        (state_changed=Falseならstateの再アップロードをスキップする)。
        """
        server = FakeReleaseServer()
        positions = [_position("still-open-1")]
        rows = [_trade_row("still-open-1"), _trade_row("already-closed-elsewhere", closed=True)]
        _seed_server(server, positions, {"2026-09": rows})

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            result = acr.run_repair(["already-closed-elsewhere"], reason="test", dry_run=False, work_dir=".")

        self.assertEqual(result["removed_positions"], [])
        upload_calls = [c for c in server.calls if c[0] == "gh" and c[1] == "release" and c[2] == "upload"]
        uploaded_state_names = [os.path.basename(c[4]) for c in upload_calls if c[3] == acp.RELEASE_TAG_STATE]
        self.assertEqual(uploaded_state_names, [], "state.jsonは変更されていないのでアップロードしないこと")


if __name__ == "__main__":
    unittest.main()
