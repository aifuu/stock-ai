"""Unit tests for run_profit_loop.py's same-ticker cooldown self-heal.

Item A (audit UNIT 4): a malformed stored cooldown timestamp used to be
silently treated as "no cooldown" (`except Exception: remaining=0`), which
fail-opens the 30-minute same-ticker re-entry block. The fix replaces the
malformed value with a fresh cooldown starting now (self-heals) instead of
letting the ticker re-enter immediately.

This module makes real network calls if imported carelessly (yfinance,
Discord), so requests.post and yfinance.download are stubbed to raise
before anything is imported, and DISCORD_WEBHOOK is forced unset so
discord_send() never even reaches requests.post.
"""
import os
import unittest
from datetime import datetime, timedelta

os.environ.pop("DISCORD_WEBHOOK", None)

import requests  # noqa: E402


def _blocked_post(*_args, **_kwargs):
    raise AssertionError("network call attempted via requests.post in tests")


requests.post = _blocked_post

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("network call attempted via yfinance.download in tests")
)

import run_profit_loop as loop  # noqa: E402


def _candidate(ticker="TEST"):
    return {
        # price: FIX(資金不足見送り)がopen_top1_only内で100株=price*100円が買えるか
        # チェックするようになったため、この値が無いと常に資金不足扱いになり
        # クールダウン自己修復ロジック自体のテストが無関係な理由で落ちる。
        "ticker": ticker, "direction": "BUY", "price": 3000.0, "score": 10, "up_probability": 60,
        "down_probability": 10, "profit_priority": 1.0, "profit_ev_pct": 1.0,
        "feedback_weight": 1.0, "selection_mode": "normal", "selection_level": 1,
        "top10_rank": 1, "market_regime": "neutral", "regime_preferred": False,
    }


class CooldownSelfHealTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("DISCORD_WEBHOOK", None)
        self._orig_regime = loop._market_regime
        self._orig_down_day = loop._down_day_decision
        self._orig_priority = loop.profit_priority
        self._orig_open = loop._original_open
        loop._market_regime = lambda: ("neutral", None, None)
        # cwdをchdirしないため実repoのdaily_decision.json(当日分)を読んでしまい得る。
        # このテストはcooldown自己修復ロジックだけが関心事なので、下落日ゲートは
        # 常に無効(既存のneutral regime挙動)に固定する。
        loop._down_day_decision = lambda: False
        loop.profit_priority = lambda cands: cands
        loop._original_open = lambda state, policy, cands, today: []

    def tearDown(self):
        loop._market_regime = self._orig_regime
        loop._down_day_decision = self._orig_down_day
        loop.profit_priority = self._orig_priority
        loop._original_open = self._orig_open

    def _state(self, raw):
        entries = {"TEST": raw} if raw is not None else {}
        return {
            "positions": [], "trades_today": 0, "trades_by_ticker_today": {},
            "last_exit_by_ticker": entries,
        }

    def _run(self, raw):
        state = self._state(raw)
        result = loop.open_top1_only(state, {}, [_candidate()], "2026-09-20")
        self._last_state = state
        return result, state["last_exit_by_ticker"]

    def test_healthy_active_cooldown_blocks_and_is_byte_identical(self):
        now = datetime.now(loop.app.TZ)
        fresh = (now - timedelta(minutes=5)).isoformat()
        result, cooldowns = self._run(fresh)
        self.assertEqual(result, [])
        self.assertEqual(cooldowns["TEST"], fresh)

    def test_healthy_expired_cooldown_pops_entry_and_allows_entry(self):
        # 2026-11 TOP1建値バグ修正: open_top1_onlyはこの時点では実際の発注をせず
        # 実stateには触れない。ただし2026-11連鎖キャンセルバグ修正により、実際に
        # 開けるかをstateのコピー(probe)上でドライラン確認するため、_original_open
        # は(実stateではなくコピーを渡して)1回だけ呼ばれるようになった。選んだ
        # 候補はstate['pending_entry']に保存され、実際の約定はfill_pending_entry()
        # が後続tickで行う(test_top1_next_bar_fill.py参照)。
        now = datetime.now(loop.app.TZ)
        expired = (now - timedelta(minutes=loop.SAME_TICKER_COOLDOWN_MINUTES + 5)).isoformat()
        state = self._state(expired)
        opened = []

        def _stub_open(s, policy, cands, today):
            opened.append((s is state, cands))
            return [dict(cands[0])]

        loop._original_open = _stub_open
        result = loop.open_top1_only(state, {}, [_candidate()], "2026-09-20")
        cooldowns = state["last_exit_by_ticker"]
        self.assertNotIn("TEST", cooldowns)
        self.assertEqual(result, [])
        self.assertEqual(len(opened), 1, "実際に開けるかのドライラン確認で_original_openが1回呼ばれる")
        self.assertFalse(opened[0][0], "ドライランはstateのコピー上で行われ、実stateは渡されない")
        self.assertEqual(state["positions"], [], "ドライランは実stateのpositionsを変化させない")
        pending = state["pending_entry"]
        self.assertEqual(pending["ticker"], "TEST")

    def test_no_cooldown_entry_allows_entry(self):
        # 2026-11 TOP1建値バグ修正/2026-11連鎖キャンセルバグ修正: 同上、発注では
        # なくpending_entryの作成を確認する。_original_openはドライラン確認の
        # ため1回だけ、実stateではなくコピーを渡して呼ばれる。
        state = self._state(None)
        opened = []

        def _stub_open(s, policy, cands, today):
            opened.append((s is state, cands))
            return [dict(cands[0])]

        loop._original_open = _stub_open
        result = loop.open_top1_only(state, {}, [_candidate()], "2026-09-20")
        cooldowns = state["last_exit_by_ticker"]
        self.assertNotIn("TEST", cooldowns)
        self.assertEqual(result, [])
        self.assertEqual(len(opened), 1, "実際に開けるかのドライラン確認で_original_openが1回呼ばれる")
        self.assertFalse(opened[0][0], "ドライランはstateのコピー上で行われ、実stateは渡されない")
        self.assertEqual(state["pending_entry"]["ticker"], "TEST")

    def test_naive_timestamp_treated_as_jst_wall_clock(self):
        now = datetime.now(loop.app.TZ)
        naive_fresh = (now - timedelta(minutes=5)).replace(tzinfo=None).isoformat()
        result, cooldowns = self._run(naive_fresh)
        self.assertEqual(result, [])
        self.assertEqual(cooldowns["TEST"], naive_fresh)

    def _assert_self_heals(self, raw):
        now = datetime.now(loop.app.TZ)
        result, cooldowns = self._run(raw)
        self.assertEqual(result, [], f"malformed value must not fail-open: {raw!r}")
        self.assertIn("TEST", cooldowns, f"self-healed entry must remain: {raw!r}")
        healed = cooldowns["TEST"]
        self.assertNotEqual(healed, raw)
        parsed = loop._as_aware_jst(healed)
        self.assertLess(abs((now - parsed).total_seconds()), 5)

    def test_malformed_string_self_heals_instead_of_failing_open(self):
        self._assert_self_heals("not-a-timestamp")

    def test_invalid_calendar_date_self_heals_instead_of_failing_open(self):
        self._assert_self_heals("1970-13-99T99:99:99")

    def test_garbage_token_self_heals_instead_of_failing_open(self):
        self._assert_self_heals("abc123")

    def test_self_heal_does_not_raise(self):
        # No exception should ever escape open_top1_only for a malformed value.
        try:
            self._run("!!not parseable at all!!")
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"open_top1_only raised on malformed cooldown value: {exc!r}")


if __name__ == "__main__":
    unittest.main()
