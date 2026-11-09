"""デイトレ売買通知(判断・約定・決済)がDiscordで赤文字(```ansi + ESC[31m)になることの検証。"""
import unittest
from datetime import datetime
from unittest.mock import patch

import daytrade_tp5000_sl8000_paper as dt

RED = "\u001b[31m"


class RedTextTest(unittest.TestCase):
    def test_every_line_is_red_inside_ansi_block(self):
        out = dt._red_text("\U0001f195 エントリー約定｜1332.T｜空売り\n約定価格 1,021.0円")
        self.assertTrue(out.startswith("```ansi\n"))
        self.assertTrue(out.endswith("\n```"))
        inner = out[len("```ansi\n"):-len("\n```")].split("\n")
        self.assertEqual(len(inner), 2)
        self.assertTrue(all(line.startswith(RED) for line in inner))
        self.assertIn("1332.T", out)
        self.assertIn("空売り", out)

    def test_long_message_keeps_closing_fence_after_discord_cut(self):
        out = dt._red_text("あ" * 5000)
        text = f"label\n{out}"
        self.assertLessEqual(len(text), 1950)
        self.assertTrue(text[:1950].endswith("```"))

    def test_backticks_in_message_cannot_break_block(self):
        out = dt._red_text("a```b")
        self.assertEqual(out.count("```"), 2)


class RunSendsRedTest(unittest.TestCase):
    def test_trade_messages_from_run_are_red(self):
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
        self.assertTrue(sent[0].startswith("```ansi\n" + RED))
        self.assertIn("1332.T", sent[0])


if __name__ == "__main__":
    unittest.main()
