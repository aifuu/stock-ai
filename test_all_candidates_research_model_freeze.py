"""④研究モデル凍結(承認ベース)・⑤モデル識別(model_id/model_version)のテスト。

④: research_model_freeze_approval.json(git管理・承認ベース)が無い/条件を
満たさない限りall_candidates_paper.pyのrun()は現行liveモデルでデータ収集を
続ける。4条件(承認ファイル・model_id一致・参考採用でない投入かつ最低評価
条件・未凍結)をすべて満たした時だけ、その時点のdirectional_model.pklを
GitHub Release(all-candidates-research-model)へ凍結保存し(アップロード後に
再ダウンロード・sha256照合してから成功とする)、以後は自動で二度と変更しない。
凍結後は毎回その凍結モデルをダウンロード・sha256検証して使い、失敗時は
liveへフォールバックせず必ずハード失敗する。

⑤: 各trade行(open/closed)にはその候補選定に実際に使われたモデルの
model_id/model_versionを記録する。directional_model_meta.jsonが無い/
model_idが現行pklと不一致(2026-09-12投入のレガシーモデル)の場合は
model_version="legacy-20260912"と定義する。

このファイルはall_candidates_paper.subprocess.run(FakeReleaseServer経由)を
モックし、実ネットワーク呼び出しは一切行わない。real yfinance/AIモデル
呼び出しを避けるため、run()の統合テストではacp.scan/acp.evaluate_exits等の
重い依存もモックする。
"""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import all_candidates_paper as acp
import profit_top10_paper as live_p10

from test_all_candidates_paper import FakeReleaseServer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TZ = ZoneInfo("Asia/Tokyo")


class TmpDirMixin:
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="acp_freeze_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_meta(path, model_id, deploy_reason="OOSゲート通過(PF=1.50>=1.0, 最大DD=-8.0%)",
                 validation_trades=20, training_date="2026-09-30"):
    _write_json(path, {
        "model_id": model_id,
        "training_date": training_date,
        "deploy_reason": deploy_reason,
        "validation_trades": validation_trades,
        "validation_pf": 1.5,
    })


def _model_id_of_bytes(payload):
    return hashlib.sha256(payload).hexdigest()[:16]


def _dump_pkl(path, payload=b"model-bytes-v1"):
    with open(path, "wb") as f:
        f.write(payload)
    return _model_id_of_bytes(payload)


def _dump_joblib_pkl(path, payload_obj):
    """joblib.load()で実際に読み戻せる本物のpklを書く(凍結→再ロードまで
    通しで検証する統合テスト用)。戻り値はそのファイルのmodel_id(sha256先頭16桁)。
    """
    import joblib
    joblib.dump(payload_obj, path)
    with open(path, "rb") as f:
        content = f.read()
    return _model_id_of_bytes(content)


# =====================================================================
# evaluate_freeze_eligibility(): 4条件それぞれ
# =====================================================================

class EvaluateFreezeEligibilityTests(TmpDirMixin, unittest.TestCase):
    def test_no_approval_file_is_not_eligible(self):
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)
        self.assertIsNone(meta)
        self.assertIn("承認ファイル", reason)

    def test_approval_not_approved_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": False, "expected_model_id": "abc123"})
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)
        self.assertIn("承認ファイル", reason)

    def test_approval_missing_expected_model_id_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True})
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)

    def test_corrupt_approval_file_is_not_eligible(self):
        with open("research_model_freeze_approval.json", "w", encoding="utf-8") as f:
            f.write("{not valid json")
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)

    def test_no_live_meta_is_not_eligible_legacy_model(self):
        # directional_model_meta.jsonが無い=2026-09-12投入のレガシーモデル相当。
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)
        self.assertIsNone(meta)
        self.assertIn("レガシー", reason)

    def test_model_id_mismatch_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "expectedaaaa1111"})
        _write_meta("directional_model_meta.json", model_id="differentbbbb2222")
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)
        self.assertIn("不一致", reason)

    def test_reference_adoption_deploy_reason_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        _write_meta(
            "directional_model_meta.json", model_id="abc123",
            deploy_reason="OOS取引数不足(5件<15件)のため判定不能だが、既存モデル無し(初回投入前)のため参考採用として投入",
            validation_trades=5,
        )
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)
        self.assertIn("参考採用", reason)

    def test_insufficient_validation_trades_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        _write_meta("directional_model_meta.json", model_id="abc123", validation_trades=10)
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".", min_validation_trades=15)
        self.assertFalse(eligible)
        self.assertIn("最低評価基準", reason)

    def test_missing_validation_trades_field_is_not_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        _write_json("directional_model_meta.json", {
            "model_id": "abc123", "deploy_reason": "OOSゲート通過", "training_date": "2026-09-30",
        })
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertFalse(eligible)

    def test_already_frozen_state_is_not_eligible_even_with_valid_approval(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        _write_meta("directional_model_meta.json", model_id="abc123", validation_trades=20)
        eligible, reason, meta = acp.evaluate_freeze_eligibility(
            {"frozen_model_id": "already-frozen-id"}, work_dir=".",
        )
        self.assertFalse(eligible)
        self.assertIn("既に", reason)

    def test_all_conditions_met_is_eligible(self):
        _write_json("research_model_freeze_approval.json", {"approved": True, "expected_model_id": "abc123"})
        _write_meta("directional_model_meta.json", model_id="abc123", validation_trades=20)
        eligible, reason, meta = acp.evaluate_freeze_eligibility({}, work_dir=".")
        self.assertTrue(eligible)
        self.assertEqual(meta["model_id"], "abc123")


# =====================================================================
# freeze_research_model(): アップロード + 再ダウンロード検証(validate-then-promote)
# =====================================================================

class FreezeResearchModelTests(TmpDirMixin, unittest.TestCase):
    def test_uploads_pkl_and_meta_verified_by_redownload(self):
        model_id = _dump_pkl("directional_model.pkl", b"frozen-payload-v1")
        meta = {"model_id": model_id, "training_date": "2026-09-30", "deploy_reason": "OOSゲート通過", "validation_trades": 20}
        server = FakeReleaseServer()

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            fields = acp.freeze_research_model(meta, work_dir=".", upload=True, now=datetime(2026, 9, 30, 16, 10, tzinfo=TZ))

        expected_asset = f"research_model_20260930_{model_id}.joblib"
        self.assertEqual(fields["frozen_model_id"], model_id)
        self.assertEqual(fields["frozen_model_asset"], expected_asset)
        self.assertEqual(fields["frozen_model_training_date"], "2026-09-30")
        self.assertEqual(fields["frozen_model_sha256"], hashlib.sha256(b"frozen-payload-v1").hexdigest())

        uploaded_names = acp.list_release_assets(acp.RELEASE_TAG_RESEARCH_MODEL) if False else server.assets_for(acp.RELEASE_TAG_RESEARCH_MODEL)
        self.assertIn(expected_asset, uploaded_names)
        self.assertIn(f"research_model_20260930_{model_id}.meta.json", uploaded_names)

        # verified by re-download: curl was issued for the just-uploaded asset
        curl_urls = [c[-1] for c in server.calls if c[0] == "curl"]
        self.assertTrue(any(expected_asset in u for u in curl_urls), "凍結モデルは再ダウンロードでsha256検証される")

    def test_sha256_mismatch_after_upload_raises_and_state_fields_never_used(self):
        model_id = _dump_pkl("directional_model.pkl", b"frozen-payload-v2")
        meta = {"model_id": model_id, "training_date": "2026-09-30", "deploy_reason": "OOSゲート通過", "validation_trades": 20}
        server = FakeReleaseServer()
        real_call = server.__call__

        def corrupting_call(cmd, **kwargs):
            result = real_call(cmd, **kwargs)
            if cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "upload" and cmd[3] == acp.RELEASE_TAG_RESEARCH_MODEL:
                asset_name = os.path.basename(cmd[4])
                if asset_name.endswith(".joblib"):
                    status, _content = server.tags[cmd[3]][asset_name]
                    server.tags[cmd[3]][asset_name] = (status, b"corrupted-on-server")
            return result

        with patch("all_candidates_paper.subprocess.run", side_effect=corrupting_call), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(RuntimeError):
                acp.freeze_research_model(meta, work_dir=".", upload=True, now=datetime(2026, 9, 30, 16, 10, tzinfo=TZ))

    def test_upload_false_skips_network_entirely(self):
        model_id = _dump_pkl("directional_model.pkl", b"frozen-payload-v3")
        meta = {"model_id": model_id, "training_date": "2026-09-30", "deploy_reason": "OOSゲート通過", "validation_trades": 20}
        with patch("all_candidates_paper.subprocess.run") as run_mock:
            fields = acp.freeze_research_model(meta, work_dir=".", upload=False, now=datetime(2026, 9, 30, 16, 10, tzinfo=TZ))
        run_mock.assert_not_called()
        self.assertEqual(fields["frozen_model_id"], model_id)


# =====================================================================
# load_frozen_model(): ダウンロード失敗・sha256不一致は必ずハード失敗
# (liveへのフォールバックは絶対に行わない)
# =====================================================================

class LoadFrozenModelTests(TmpDirMixin, unittest.TestCase):
    def _state_for(self, asset_name, sha):
        return {"frozen_model_id": "abc123", "frozen_model_asset": asset_name, "frozen_model_sha256": sha}

    def test_successful_download_and_verify_loads_model(self):
        import joblib
        payload_obj = {"kind": "fake-frozen-model", "value": 42}
        joblib.dump(payload_obj, "source.joblib")
        with open("source.joblib", "rb") as f:
            content = f.read()
        sha = hashlib.sha256(content).hexdigest()

        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_RESEARCH_MODEL, "research_model_20260930_abc123.joblib", 200, content)
        state = self._state_for("research_model_20260930_abc123.joblib", sha)

        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            loaded = acp.load_frozen_model(state, work_dir=".")
        self.assertEqual(loaded, payload_obj)

    def test_download_404_is_hard_failure_not_fallback(self):
        server = FakeReleaseServer()  # asset not registered -> 404
        state = self._state_for("research_model_20260930_abc123.joblib", "deadbeef")
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(RuntimeError):
                acp.load_frozen_model(state, work_dir=".")

    def test_download_5xx_after_retries_is_hard_failure(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_RESEARCH_MODEL, "research_model_20260930_abc123.joblib", 500, b"")
        state = self._state_for("research_model_20260930_abc123.joblib", "deadbeef")
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(acp.ReleaseFetchError):
                acp.load_frozen_model(state, work_dir=".")

    def test_sha256_mismatch_is_hard_failure(self):
        server = FakeReleaseServer()
        server.set_asset(acp.RELEASE_TAG_RESEARCH_MODEL, "research_model_20260930_abc123.joblib", 200, b"tampered-content")
        state = self._state_for("research_model_20260930_abc123.joblib", hashlib.sha256(b"original-content").hexdigest())
        with patch("all_candidates_paper.subprocess.run", side_effect=server), \
             patch("all_candidates_paper.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                acp.load_frozen_model(state, work_dir=".")
        self.assertIn("sha256不一致", str(ctx.exception))


# =====================================================================
# _scan_with_model(): profit_top10_paper.load_modelのプロセス内パッチ+復元
# (frozen-load parity: このプロセス内だけ差し替わり、他へは波及しないこと)
# =====================================================================

class ScanWithModelPatchesLoadModelTests(unittest.TestCase):
    def test_patches_load_model_during_scan_and_restores_after(self):
        sentinel = object()
        original = live_p10.load_model
        observed = {}

        def fake_scan(policy, limit=None):
            observed["model_seen_during_scan"] = live_p10.load_model()
            return ([], 0)

        with patch.object(acp, "scan", fake_scan):
            acp._scan_with_model({}, sentinel)

        self.assertIs(observed["model_seen_during_scan"], sentinel)
        self.assertIs(live_p10.load_model, original)

    def test_restores_load_model_even_if_scan_raises(self):
        original = live_p10.load_model

        def fake_scan(policy, limit=None):
            raise RuntimeError("boom")

        with patch.object(acp, "scan", fake_scan):
            with self.assertRaises(RuntimeError):
                acp._scan_with_model({}, object())

        self.assertIs(live_p10.load_model, original)


# =====================================================================
# current_model_identity(): ⑤ model_id/model_version決定ロジック
# =====================================================================

class CurrentModelIdentityTests(TmpDirMixin, unittest.TestCase):
    def test_frozen_state_wins_regardless_of_live_files(self):
        _dump_pkl("directional_model.pkl", b"live-bytes")
        _write_meta("directional_model_meta.json", model_id=_model_id_of_bytes(b"live-bytes"))
        state = {"frozen_model_id": "frozen-id-999", "frozen_model_training_date": "2026-09-15"}
        model_id, model_version = acp.current_model_identity(state, work_dir=".")
        self.assertEqual(model_id, "frozen-id-999")
        self.assertEqual(model_version, "2026-09-15")

    def test_frozen_state_without_training_date_falls_back_to_legacy_label(self):
        state = {"frozen_model_id": "frozen-id-999"}
        model_id, model_version = acp.current_model_identity(state, work_dir=".")
        self.assertEqual(model_id, "frozen-id-999")
        self.assertEqual(model_version, acp.LEGACY_MODEL_VERSION)

    def test_no_live_pkl_returns_none_id_and_legacy_version(self):
        model_id, model_version = acp.current_model_identity({}, work_dir=".")
        self.assertIsNone(model_id)
        self.assertEqual(model_version, acp.LEGACY_MODEL_VERSION)

    def test_live_pkl_without_meta_is_legacy(self):
        # 2026-09-12投入モデル相当: directional_model_meta.jsonが無い。
        model_id = _dump_pkl("directional_model.pkl", b"legacy-live-bytes")
        got_id, got_version = acp.current_model_identity({}, work_dir=".")
        self.assertEqual(got_id, model_id)
        self.assertEqual(got_version, acp.LEGACY_MODEL_VERSION)

    def test_live_pkl_with_matching_meta_uses_training_date(self):
        model_id = _dump_pkl("directional_model.pkl", b"live-bytes-2")
        _write_meta("directional_model_meta.json", model_id=model_id, training_date="2026-09-28")
        got_id, got_version = acp.current_model_identity({}, work_dir=".")
        self.assertEqual(got_id, model_id)
        self.assertEqual(got_version, "2026-09-28")

    def test_live_pkl_with_stale_mismatched_meta_is_legacy(self):
        # metaが古い(pklは差し替わったがmeta更新前 等)場合、model_idが一致しないので安全側でlegacy扱い。
        model_id = _dump_pkl("directional_model.pkl", b"new-live-bytes")
        _write_meta("directional_model_meta.json", model_id="stale-id-from-before", training_date="2026-09-20")
        got_id, got_version = acp.current_model_identity({}, work_dir=".")
        self.assertEqual(got_id, model_id)
        self.assertEqual(got_version, acp.LEGACY_MODEL_VERSION)


# =====================================================================
# build_new_positions(): model_id/model_versionがdedupキーに影響しないこと
# =====================================================================

class BuildNewPositionsModelTaggingTests(unittest.TestCase):
    def test_model_fields_recorded_but_not_part_of_trade_id(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                       "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"}]
        policy = {"nikkei_filter": False, "hold_days": 3}
        positions = acp.build_new_positions(
            set(), candidates, "2026-09-25", policy, "all_candidates_frozen_policy.json",
            "somehash", False, model_id="model-id-abc", model_version="2026-09-20",
        )
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["model_id"], "model-id-abc")
        self.assertEqual(positions[0]["model_version"], "2026-09-20")
        self.assertNotIn("model-id-abc", positions[0]["trade_id"])

    def test_missing_model_fields_default_to_none(self):
        candidates = [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                       "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": "2026-09-25"}]
        policy = {"nikkei_filter": False, "hold_days": 3}
        positions = acp.build_new_positions(
            set(), candidates, "2026-09-25", policy, "all_candidates_frozen_policy.json", "somehash", False,
        )
        self.assertIsNone(positions[0]["model_id"])
        self.assertIsNone(positions[0]["model_version"])

    def test_legacy_row_without_model_columns_becomes_na_in_dataframe(self):
        # 過去(この機能導入前)に書かれた行は、model_id/model_version列が無い
        # ままdictとして存在しうる。rows_to_dataframe()はこれをNaN(未定義=
        # レガシーモデル扱い)として埋め、既存データを書き換えない。
        legacy_row = {
            "trade_id": "x", "date": "2026-09-01", "entry_date": "2026-09-01", "ticker": "7203.T",
            "direction": "BUY", "rank": 1, "score": 1.0,
        }
        df = acp.rows_to_dataframe([legacy_row])
        self.assertTrue(df.loc[0, "model_id"] is None or str(df.loc[0, "model_id"]) == "nan" or df["model_id"].isna().iloc[0])
        self.assertTrue(df["model_version"].isna().iloc[0])


# =====================================================================
# 複数日シミュレーション(統合): day1無承認→day2 model_id不一致→
# day3正しい承認で凍結→day4live変化でも凍結モデル継続
# =====================================================================

class MultiDayFreezeSimulationTests(unittest.TestCase):
    """run()を4営業日分順に呼び、④/⑤の一連の振る舞いを通しで検証する。

    実ネットワークは一切使わない: fetch_state/promote_and_upload_stateは
    state_holderに対するin-memory読み書きに差し替え、evaluate_exits/scan/
    list_release_assets/append_trade_rows/download_all_monthsは軽量mockに
    差し替える。freeze_research_model()/load_frozen_model()だけは実際の
    コードパス(subprocess.run経由)を通し、FakeReleaseServerで検証する。
    """

    def setUp(self):
        self.work_dir_ctx = tempfile.TemporaryDirectory()
        self.work_dir = self.work_dir_ctx.name
        shutil.copyfile(
            os.path.join(REPO_ROOT, "all_candidates_frozen_policy.json"),
            os.path.join(self.work_dir, "all_candidates_frozen_policy.json"),
        )
        shutil.copyfile(
            os.path.join(REPO_ROOT, "all_candidates_frozen_policy_up.json"),
            os.path.join(self.work_dir, "all_candidates_frozen_policy_up.json"),
        )
        self.addCleanup(self.work_dir_ctx.cleanup)

    def _candidate(self, today):
        return [{"ticker": "7203.T", "direction": "BUY", "price": 3000.0, "tp": 3100.0, "sl": 2950.0,
                 "score": 80.0, "up_probability": 60.0, "down_probability": 10.0, "data_date": today}]

    def test_four_day_freeze_lifecycle(self):
        server = FakeReleaseServer()
        state_holder = {"state": acp.default_state()}

        def fake_fetch_state(wd):
            return dict(state_holder["state"]), "primary"

        def fake_promote(new_state, work_dir=None, upload=True):
            state_holder["state"] = new_state

        common_patches = [
            patch.object(acp, "select_policy_file", return_value=("strategy_policy.json", {"trend": "up"})),
            patch.object(acp, "evaluate_exits", return_value=([], [])),
            patch.object(acp, "fetch_state", side_effect=fake_fetch_state),
            patch.object(acp, "promote_and_upload_state", side_effect=fake_promote),
            patch.object(acp, "list_release_assets", return_value=[]),
            patch.object(acp, "append_trade_rows", return_value={}),
            patch.object(acp, "download_all_months", return_value=acp.rows_to_dataframe([])),
            patch("all_candidates_paper.subprocess.run", side_effect=server),
            patch("all_candidates_paper.time.sleep"),
        ]

        # ---- Day1 (2026-09-25): 承認ファイル無し。ライブモデル(レガシー、メタ無し)で収集 ----
        day1_model_id = _dump_pkl(os.path.join(self.work_dir, "directional_model.pkl"), b"live-payload-day1")
        scan_mock = patch.object(acp, "scan", return_value=(self._candidate("2026-09-25"), 100))
        with scan_mock, common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
             common_patches[4], common_patches[5], common_patches[6], common_patches[7], common_patches[8]:
            result1 = acp.run(now=datetime(2026, 9, 25, 16, 10, tzinfo=TZ), work_dir=self.work_dir)

        self.assertNotIn("skipped", result1)
        self.assertNotIn("frozen_model_id", state_holder["state"])
        day1_positions = state_holder["state"]["positions"]
        self.assertEqual(len(day1_positions), 1)
        self.assertEqual(day1_positions[0]["model_id"], day1_model_id)
        self.assertEqual(day1_positions[0]["model_version"], acp.LEGACY_MODEL_VERSION)

        # ---- Day2 (2026-09-28): 承認ファイルはあるがexpected_model_idが現行と不一致 → 凍結しない ----
        # (day3で実際に凍結・day4で再ロードされるモデルなので、本物のjoblib pklにする)
        day2_model_id = _dump_joblib_pkl(
            os.path.join(self.work_dir, "directional_model.pkl"), {"kind": "fake-model", "day": 2},
        )
        _write_meta(
            os.path.join(self.work_dir, "directional_model_meta.json"),
            model_id=day2_model_id, training_date="2026-09-27", validation_trades=20,
        )
        _write_json(
            os.path.join(self.work_dir, "research_model_freeze_approval.json"),
            {"approved": True, "expected_model_id": "some-other-unrelated-id-0000"},
        )
        scan_mock = patch.object(acp, "scan", return_value=(self._candidate("2026-09-28"), 100))
        with scan_mock, common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
             common_patches[4], common_patches[5], common_patches[6], common_patches[7], common_patches[8]:
            result2 = acp.run(now=datetime(2026, 9, 28, 16, 10, tzinfo=TZ), work_dir=self.work_dir)

        self.assertNotIn("skipped", result2)
        self.assertNotIn("frozen_model_id", state_holder["state"])
        day2_positions = state_holder["state"]["positions"]
        self.assertEqual(len(day2_positions), 1)
        self.assertEqual(day2_positions[0]["model_id"], day2_model_id)
        self.assertEqual(day2_positions[0]["model_version"], "2026-09-27")
        research_model_uploads_after_day2 = [
            c for c in server.calls
            if c[0] == "gh" and c[1] == "release" and c[2] == "upload" and c[3] == acp.RELEASE_TAG_RESEARCH_MODEL
        ]
        self.assertEqual(research_model_uploads_after_day2, [], "model_id不一致では凍結アップロードが一切発生してはならない")

        # ---- Day3 (2026-09-29): 承認ファイルのexpected_model_idが現行と一致 → 凍結する ----
        _write_json(
            os.path.join(self.work_dir, "research_model_freeze_approval.json"),
            {"approved": True, "expected_model_id": day2_model_id},
        )
        scan_mock = patch.object(acp, "scan", return_value=(self._candidate("2026-09-29"), 100))
        with scan_mock, common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
             common_patches[4], common_patches[5], common_patches[6], common_patches[7], common_patches[8]:
            result3 = acp.run(now=datetime(2026, 9, 29, 16, 10, tzinfo=TZ), work_dir=self.work_dir)

        self.assertNotIn("skipped", result3)
        self.assertEqual(state_holder["state"].get("frozen_model_id"), day2_model_id)
        frozen_asset = state_holder["state"]["frozen_model_asset"]
        day3_positions = state_holder["state"]["positions"]
        self.assertEqual(len(day3_positions), 1)
        self.assertEqual(day3_positions[0]["model_id"], day2_model_id)
        self.assertEqual(day3_positions[0]["model_version"], "2026-09-27")

        # ---- Day4 (2026-09-30): ライブモデルが変わっても、凍結モデルを使い続ける ----
        day4_model_id = _dump_pkl(os.path.join(self.work_dir, "directional_model.pkl"), b"live-payload-day4-DIFFERENT")
        _write_meta(
            os.path.join(self.work_dir, "directional_model_meta.json"),
            model_id=day4_model_id, training_date="2026-09-30", validation_trades=30,
        )
        scan_mock = patch.object(acp, "scan", return_value=(self._candidate("2026-09-30"), 100))
        with scan_mock, common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
             common_patches[4], common_patches[5], common_patches[6], common_patches[7], common_patches[8]:
            result4 = acp.run(now=datetime(2026, 9, 30, 16, 10, tzinfo=TZ), work_dir=self.work_dir)

        self.assertNotIn("skipped", result4)
        # 凍結モデルIDはday3のまま変化しない(day4のliveモデルIDとは別物)。
        self.assertEqual(state_holder["state"].get("frozen_model_id"), day2_model_id)
        self.assertNotEqual(state_holder["state"]["frozen_model_id"], day4_model_id)
        day4_positions = state_holder["state"]["positions"]
        self.assertEqual(len(day4_positions), 1)
        self.assertEqual(day4_positions[0]["model_id"], day2_model_id, "day4もフリーズ済みモデルのmodel_idを使い続けること")
        self.assertNotEqual(day4_positions[0]["model_id"], day4_model_id)

        # 凍結アップロードはday3の1回のみ(re-freeze禁止)。
        research_model_uploads_total = [
            c for c in server.calls
            if c[0] == "gh" and c[1] == "release" and c[2] == "upload" and c[3] == acp.RELEASE_TAG_RESEARCH_MODEL
        ]
        uploaded_joblib_names = {os.path.basename(c[4]) for c in research_model_uploads_total if c[4].endswith(".joblib")}
        self.assertEqual(uploaded_joblib_names, {frozen_asset})


if __name__ == "__main__":
    unittest.main()
