"""ai-stock-scan.ymlの「AM/PMセッション開始通知」JSTウィンドウガードの回帰テスト。

背景(2026-09-29のインシデント): stock-ai-paper-tradingのconcurrencyグループで
scheduleの発火がクラウド側ルーチンのdispatch実行の後ろに並ぶと、15:35 JST
(大引け後)を過ぎてからAMジョブが起動することがあり、そのままだと
「寄り前(AM)セッション開始」という誤解を招くDiscord通知が大引け後に飛んで
しまっていた。取引ロジック・ループ・concurrency・schedule等は一切変更せず、
Discord通知ステップだけをジョブ起動時刻(JST)が08:00〜15:30の範囲外なら
スキップするよう変更した。このテストはyaml.safe_loadとテキストスキャンのみで
検証し、ワークフロー自身のスクリプトは一切import/実行しない。
"""
import re
import unittest
from datetime import time as dtime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ai-stock-scan.yml"

# ワークフローのnotify windowステップと同一の判定式(08:00〜15:30 JST)。
# ワークフロー側の定数を変更したのにこのテストの期待値を更新し忘れる、という
# ズレを検知できるよう、下のテストでrun:テキスト中の "dtime(8, 0)" /
# "dtime(15, 30)" の存在も別途アサートする。
WINDOW_START = dtime(8, 0)
WINDOW_END = dtime(15, 30)


def _in_notify_window(t):
    return WINDOW_START <= t <= WINDOW_END


def _load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _steps(doc, job_name):
    return doc["jobs"][job_name]["steps"]


def _step_by_name(steps, name):
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step not found: {name}")


class NotifyWindowBoundaryFormulaTests(unittest.TestCase):
    """判定式そのもの(このテストが仕様として持つコピー)の境界値確認。"""

    def test_07_59_is_outside(self):
        self.assertFalse(_in_notify_window(dtime(7, 59)))

    def test_08_00_is_inside(self):
        self.assertTrue(_in_notify_window(dtime(8, 0)))

    def test_15_30_is_inside(self):
        self.assertTrue(_in_notify_window(dtime(15, 30)))

    def test_15_31_is_outside(self):
        self.assertFalse(_in_notify_window(dtime(15, 31)))

    def test_normal_dispatch_time_08_30_is_inside(self):
        self.assertTrue(_in_notify_window(dtime(8, 30)))

    def test_delayed_post_close_run_15_35_is_outside(self):
        # インシデントの実例(大引け後15:35頃に遅延発火したケース)。
        self.assertFalse(_in_notify_window(dtime(15, 35)))


class WorkflowYamlParsesAndGatesNotify(unittest.TestCase):
    def setUp(self):
        self.doc = _load_workflow()

    def test_yaml_safe_loads(self):
        self.assertIsInstance(self.doc, dict)

    def test_window_constants_present_in_both_steps(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        occurrences_start = len(re.findall(r"dtime\(8,\s*0\)", text))
        occurrences_end = len(re.findall(r"dtime\(15,\s*30\)", text))
        self.assertEqual(occurrences_start, 2, "expected AM and PM notify-window steps both using dtime(8, 0)")
        self.assertEqual(occurrences_end, 2, "expected AM and PM notify-window steps both using dtime(15, 30)")

    def _assert_gate(self, job_name, gate_step_name, notify_step_name):
        steps = _steps(self.doc, job_name)
        gate = _step_by_name(steps, gate_step_name)
        notify = _step_by_name(steps, notify_step_name)

        self.assertTrue(gate.get("id"), f"{gate_step_name}: missing step id")
        self.assertTrue(
            gate.get("continue-on-error"),
            f"{gate_step_name}: must continue-on-error so a broken time check never breaks the job",
        )

        expected_if = f"steps.{gate['id']}.outputs.in_window == 'true'"
        self.assertEqual(
            notify.get("if"), expected_if,
            msg=f"{notify_step_name}: expected if: {expected_if!r}, got {notify.get('if')!r}",
        )

        # ゲートステップは通知ステップより前になければならない。
        self.assertLess(steps.index(gate), steps.index(notify))

    def test_am_notify_is_gated(self):
        self._assert_gate("session_am", "Compute JST notify window (AM)", "Notify AM session start (Discord)")

    def test_pm_notify_is_gated(self):
        self._assert_gate("session_pm", "Compute JST notify window (PM)", "Notify PM session start (Discord)")

    def test_trading_logic_untouched_markers_still_present(self):
        """通知ゲート以外(取引ロジック・ループ・concurrency・schedule)は
        変更していないことのテキストレベルでの簡易確認。"""
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        for marker in (
            "- cron: '30 23 * * 0-4'",
            "group: stock-ai-paper-trading",
            "cancel-in-progress: false",
            "SESSION_END_MIN=755",
            "SESSION_END_MIN=935",
            "HARD_WALL_EPOCH=$((START_EPOCH + 320*60))",
            "trap 'echo \"🛑 中断シグナル受信、次のチェックポイントでループを終了します\"; STOP=1' INT TERM",
            "git rebase --autostash origin/main && git push origin HEAD:main",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_notify_start_calls_unchanged(self):
        steps_am = _steps(self.doc, "session_am")
        steps_pm = _steps(self.doc, "session_pm")
        am = _step_by_name(steps_am, "Notify AM session start (Discord)")
        pm = _step_by_name(steps_pm, "Notify PM session start (Discord)")
        self.assertEqual(am["run"], "python -c \"import discord_progress; discord_progress.notify_start('AM')\"")
        self.assertEqual(pm["run"], "python -c \"import discord_progress; discord_progress.notify_start('PM')\"")


if __name__ == "__main__":
    unittest.main()
