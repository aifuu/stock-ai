"""デイトレ売買通知(判断・約定・決済)がDiscordで赤文字(```ansi + ESC[31m)になり、
1行目が大見出し(# 🔴)でも出ることの検証。"""
import unittest
from datetime import datetime
from unittest.mock import patch

import daytrade_tp5000_sl8000_paper as dt

RED = "\u001b[31m"
HEAD = "# \U0001f534 "


def _split(out):
    head, block = out.split("\n", 1)
    return head, block


class RedTextTest(unittest.TestCase):
    def test_first_line_is_big_heading_and_every_line_is_red(self):
        out = dt._red_text("\U0001f195 エントリー約定｜1332.T｜空売り\n約定価格 1,021.0円")
        head, block = _split(out)
        self.assertEqual(head, HEAD + "\U0001f195 エントリー約定｜1332.T｜空売り")
        self.assertTrue(block.startswith("```ansi\n"))
        self.assertTrue(block.endswith("\n```"))
        inner = block[len("```ansi\n"):-len("\n```")].split("\n")
        self.assertEqual(len(inner), 2)
        self.assertTrue(all(line.startswith(RED) for line in inner))
        self.assertIn("1332.T", inner[0])
        self.assertIn("1,021.0", inner[1])

    def test_long_message_keeps_closing_fence_after_discord_cut(self):
        label = "\U0001f4b4 【デイトレ別枠 ATR TP1.0×/SL0.5×】"
        for msg in ["あ" * 5000, "見出し\n" + "あ" * 5000]:
            with self.subTest(n=len(msg)):
                out = dt._red_text(msg)
                text = f"{label}\n{out}"
                self.assertLessEqual(len(text), 1950)
                self.assertTrue(text[:1950].endswith("```"))
                self.assertLessEqual(len(_split(out)[0]), len(HEAD) + 80)

    def test_backticks_in_message_cannot_break_block(self):
        out = dt._red_text("a```b")
        self.assertEqual(out.count("```"), 2)
        self.assertEqual(_split(out)[0], HEAD + "a'''b")

    def test_blank_first_line_gets_no_empty_heading(self):
        out = dt._red_text("\n本文")
        self.assertTrue(out.startswith("```ansi\n"))


class RunSendsRedTest(unittest.TestCase):
    def test_trade_messages_from_run_are_big_and_red(self):
        now = datetime(2026, 11, 9, 10, 0, tzinfo=dt.TZ)
        state = dt.default_state()
        state["trade_date"] = "2026-11-09"
        sent = []
        with patch.object(dt, "load_state", return_value=state), \
             patch.object(dt, "save_state"), \
             patch.object(dt, "is_tse_trading_day", return_value=True), \
             patch.object(dt, "_try_entry", return_value="⏳ 判断(ペンディング)｜1332.T｜空売り"), \
             patch.object(dt, "_maybe_send_daily_summary"), \
             patch.object(dt, "discord_send", side_effect=lambda m: sent.append(m) or True):
            dt.run(now=now)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].startswith(HEAD + "⏳ 判断(ペンディング)｜1332.T｜空売り\n```ansi\n" + RED))


if __name__ == "__main__":
    unittest.main()
