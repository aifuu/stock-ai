"""daily_model_retrain.pyのチャンピオン/チャレンジャー方式デプロイ判定の回帰テスト。

背景(2026-09-30の調査): directional_model.pkl(2026-09-12投入、OOS取引5件・
PF0.218・勝率20%・最大DD-26%で「参考採用」)が、以後17回以上の再学習
(OOS取引17〜21件・PF0.49〜0.78)でも一度も差し替えられていなかった。原因は
デプロイ判定がチャレンジャーを固定の絶対ゲート(PF>=1.0)にのみ通す設計で、
現行モデル(チャンピオン)がどれほど悪くても比較対象にしていなかったため。

このファイルはdecide_deploy()・load_model_meta()・save_model_meta()・
incumbent_window_status()という、実データ取得(yfinance)やモデル学習を伴わない
純粋なロジック部分だけを対象にする(daily_model_retrain.py・safe_state.py・
標準ライブラリのみを使う)。evaluate_incumbent()はjoblib.load/simulate_oos_top1
への薄いラッパーのため、それらをモンキーパッチして配線だけを確認する。
"""
import json
import os
import shutil
import tempfile
import unittest

import daily_model_retrain as m


def _metrics(trades, pf, win_rate=50.0, max_dd_pct=-10.0):
    return {"trades": trades, "pf": pf, "win_rate": win_rate, "max_dd_pct": max_dd_pct}


class DecideDeployTooFewTradesTests(unittest.TestCase):
    """チャレンジャーの取引数不足時は、従来通り「現行モデル有無」だけで決まる
    (チャンピオン/チャレンジャー比較には入らない)。"""

    def test_too_few_trades_with_incumbent_keeps_incumbent(self):
        challenger = _metrics(trades=5, pf=5.0)  # PFが良くても取引数不足なら見送り
        deploy, reason = m.decide_deploy(challenger, None, "unknown", incumbent_exists=True)
        self.assertFalse(deploy)
        self.assertIn("取引数不足", reason)
        self.assertIn("継続使用", reason)

    def test_too_few_trades_without_incumbent_deploys_as_reference(self):
        challenger = _metrics(trades=5, pf=0.218, win_rate=20.0, max_dd_pct=-26.0)
        deploy, reason = m.decide_deploy(challenger, None, "no_incumbent", incumbent_exists=False)
        self.assertTrue(deploy)
        self.assertIn("参考採用", reason)


class DecideDeployAbsoluteGateTests(unittest.TestCase):
    """現行モデルが無い/比較不能な場合は、従来通り絶対ゲートのみで判定する。"""

    def test_no_incumbent_passes_absolute_gate(self):
        challenger = _metrics(trades=20, pf=1.5, max_dd_pct=-10.0)
        deploy, reason = m.decide_deploy(challenger, None, "no_incumbent", incumbent_exists=False)
        self.assertTrue(deploy)
        self.assertIn("OOSゲート通過", reason)

    def test_no_incumbent_fails_absolute_gate(self):
        challenger = _metrics(trades=20, pf=0.7, max_dd_pct=-10.0)
        deploy, reason = m.decide_deploy(challenger, None, "no_incumbent", incumbent_exists=False)
        self.assertFalse(deploy)
        self.assertIn("OOSゲート未通過", reason)

    def test_leaky_incumbent_falls_back_to_absolute_gate_only(self):
        # 現行モデルはあるが学習カットオフがOOS区間に重なる(リーク) → 比較せず絶対ゲートのみ。
        challenger = _metrics(trades=20, pf=0.7, max_dd_pct=-10.0)  # 絶対ゲート未通過
        deploy, reason = m.decide_deploy(challenger, None, "leaky", incumbent_exists=True)
        self.assertFalse(deploy)
        self.assertIn("OOSゲート未通過", reason)
        self.assertIn("leaky", reason)

    def test_unknown_incumbent_metadata_falls_back_to_absolute_gate_only(self):
        # ちょうど今の directional_model.pkl(2026-09-12投入、メタ情報無し)の状況。
        challenger = _metrics(trades=20, pf=1.5, max_dd_pct=-10.0)  # 絶対ゲート通過
        deploy, reason = m.decide_deploy(challenger, None, "unknown", incumbent_exists=True)
        self.assertTrue(deploy)
        self.assertIn("OOSゲート通過", reason)
        self.assertIn("unknown", reason)


class DecideDeployRelativeGateTests(unittest.TestCase):
    """現行モデルと同一区間で比較できる場合(window_status=='safe')は、
    絶対ゲート未通過でもチャンピオンを明確に上回れば差し替える。"""

    def test_challenger_clearly_beats_incumbent_deploys_even_below_absolute_gate(self):
        # 現行(チャンピオン)がPF0.218の粗悪モデル、チャレンジャーはPF0.63で
        # 絶対ゲート(1.0)には届かないが、チャンピオンを明確に上回っている。
        incumbent = _metrics(trades=5, pf=0.218, win_rate=20.0, max_dd_pct=-26.0)
        challenger = _metrics(trades=17, pf=0.631, win_rate=35.3, max_dd_pct=-14.99)
        deploy, reason = m.decide_deploy(challenger, incumbent, "safe", incumbent_exists=True)
        self.assertTrue(deploy)
        self.assertIn("相対ゲート", reason)

    def test_challenger_within_margin_of_incumbent_keeps_incumbent(self):
        # マージン(デフォルト0.10)未満の差は「有意に上回った」とみなさない。
        incumbent = _metrics(trades=20, pf=0.60, max_dd_pct=-10.0)
        challenger = _metrics(trades=20, pf=0.65, max_dd_pct=-10.0)
        deploy, reason = m.decide_deploy(challenger, incumbent, "safe", incumbent_exists=True)
        self.assertFalse(deploy)
        self.assertIn("有意に上回れず", reason)

    def test_challenger_beats_pf_but_dd_much_worse_keeps_incumbent(self):
        incumbent = _metrics(trades=20, pf=0.60, max_dd_pct=-10.0)
        challenger = _metrics(trades=20, pf=0.90, max_dd_pct=-40.0)  # DDが30ポイント悪化
        deploy, reason = m.decide_deploy(challenger, incumbent, "safe", incumbent_exists=True)
        self.assertFalse(deploy)

    def test_challenger_worse_than_incumbent_keeps_incumbent(self):
        incumbent = _metrics(trades=20, pf=1.2, max_dd_pct=-8.0)
        challenger = _metrics(trades=20, pf=0.7, max_dd_pct=-15.0)
        deploy, reason = m.decide_deploy(challenger, incumbent, "safe", incumbent_exists=True)
        self.assertFalse(deploy)

    def test_absolute_gate_pass_still_deploys_even_if_incumbent_is_great(self):
        # 絶対ゲート通過ならチャンピオンの成績に関わらず投入する(従来挙動維持)。
        incumbent = _metrics(trades=20, pf=5.0, max_dd_pct=-5.0)
        challenger = _metrics(trades=20, pf=1.5, max_dd_pct=-10.0)
        deploy, reason = m.decide_deploy(challenger, incumbent, "safe", incumbent_exists=True)
        self.assertTrue(deploy)
        self.assertIn("絶対ゲート", reason)


class IncumbentWindowStatusTests(unittest.TestCase):
    def test_no_metadata_is_unknown(self):
        self.assertEqual(m.incumbent_window_status(None, "2026-07-01"), "unknown")

    def test_train_cutoff_before_oos_cutoff_is_safe(self):
        meta = {"train_cutoff": "2026-06-01"}
        self.assertEqual(m.incumbent_window_status(meta, "2026-07-01"), "safe")

    def test_train_cutoff_after_oos_cutoff_is_leaky(self):
        # ちょうど今のdirectional_model.pkl(train_cutoff~2026-09-12)相当:
        # 学習カットオフがOOS区間開始より後 → 完全にリーク。
        meta = {"train_cutoff": "2026-09-12"}
        self.assertEqual(m.incumbent_window_status(meta, "2026-07-02"), "leaky")

    def test_train_cutoff_equal_to_oos_cutoff_is_leaky(self):
        meta = {"train_cutoff": "2026-07-01"}
        self.assertEqual(m.incumbent_window_status(meta, "2026-07-01"), "leaky")

    def test_malformed_metadata_is_unknown(self):
        self.assertEqual(m.incumbent_window_status({"train_cutoff": "not-a-date"}, "2026-07-01"), "unknown")


class ModelMetaRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="daily_model_retrain_meta_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_returns_none(self):
        self.assertIsNone(m.load_model_meta())

    def test_save_then_load_round_trips(self):
        challenger = _metrics(trades=18, pf=1.234, win_rate=44.4, max_dd_pct=-12.34)
        m.save_model_meta("2026-09-30", "2026-09-30", challenger)
        meta = m.load_model_meta()
        self.assertEqual(meta["train_cutoff"], "2026-09-30")
        self.assertEqual(meta["deployed_date"], "2026-09-30")
        self.assertEqual(meta["oos_pf"], 1.234)
        self.assertEqual(meta["oos_trades"], 18)

    def test_corrupt_file_returns_none(self):
        with open(m.MODEL_META_FILE, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(m.load_model_meta())

    def test_file_without_train_cutoff_key_returns_none(self):
        with open(m.MODEL_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"some_other_key": 1}, f)
        self.assertIsNone(m.load_model_meta())


class EvaluateIncumbentWiringTests(unittest.TestCase):
    """evaluate_incumbent()はjoblib.load()+simulate_oos_top1()+compute_pf_metrics()
    の薄いラッパー。実モデル/実データ無しで配線だけを確認する。"""

    def setUp(self):
        self._orig_load = m.joblib.load
        self._orig_simulate = m.simulate_oos_top1

    def tearDown(self):
        m.joblib.load = self._orig_load
        m.simulate_oos_top1 = self._orig_simulate

    def test_success_path_returns_metrics(self):
        sentinel_model = object()
        m.joblib.load = lambda path: sentinel_model
        m.simulate_oos_top1 = lambda model, frames, dates, nikkei: (
            [{"entry_date": "2026-09-01", "return_pct": 5.0}] if model is sentinel_model else []
        )
        result = m.evaluate_incumbent({}, [], None)
        self.assertEqual(result["trades"], 1)

    def test_load_failure_returns_none(self):
        def _boom(path):
            raise OSError("no such file")
        m.joblib.load = _boom
        self.assertIsNone(m.evaluate_incumbent({}, [], None))


class AppendRetrainReportBackwardCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._prev_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="daily_model_retrain_report_test_")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_columns_appended_without_breaking_old_rows(self):
        old_row = {
            "date": "2026-09-12", "train_rows": 107094, "oos_days": 63,
            "oos_trades": 5, "oos_pf": 0.218, "oos_win_rate": 20.0,
            "oos_max_dd_pct": -26.08, "deployed": True, "reason": "old-style row",
            "live_recent_trades": 3, "live_recent_win_rate": 0.0, "live_recent_pnl": -6673.11,
        }
        m.append_retrain_report(old_row)

        new_row = dict(old_row)
        new_row.update({
            "date": "2026-09-30",
            "incumbent_window_status": "unknown",
            "incumbent_oos_trades": 5,
            "incumbent_oos_pf": 0.218,
            "incumbent_oos_win_rate": 20.0,
            "incumbent_oos_max_dd_pct": -26.08,
        })
        m.append_retrain_report(new_row)

        import pandas as pd
        df = pd.read_csv(m.RETRAIN_REPORT_FILE)
        self.assertEqual(len(df), 2)
        # 旧形式の行(1行目)は新列がNaN(未記録)のままで、日付など既存列は保持される。
        self.assertEqual(df.loc[0, "date"], "2026-09-12")
        self.assertTrue(pd.isna(df.loc[0, "incumbent_window_status"]))
        # 新形式の行(2行目)には新列がそのまま入っている。
        self.assertEqual(df.loc[1, "incumbent_window_status"], "unknown")
        self.assertEqual(df.loc[1, "incumbent_oos_pf"], 0.218)


if __name__ == "__main__":
    unittest.main()
