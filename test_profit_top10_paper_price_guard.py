"""F1回帰テスト: profit_top10_paper.open_positions() が非有限(NaN/inf)や
0以下の価格を持つ候補をクラッシュせずスキップし、有効な候補は従来通り
開けることを確認する。

crash reproduction (fix適用前): NaN price bypasses the ``price<=0`` guard
(NaN<=0 is False), then ``int(slot_budget//price)`` raises
``ValueError: cannot convert float NaN to integer`` and the whole tick dies.
"""
import math
import unittest

import profit_top10_paper as app


def _state():
    return {
        "capital": 1_000_000.0,
        "positions": [],
        "peak": 1_000_000.0,
        "daily_start_capital": 1_000_000.0,
        "trades_today": 0,
        "trades_by_ticker_today": {},
    }


def _policy():
    return {"updated_at": "2026-09-01"}


def _cand(ticker, price, **extra):
    base = {
        "ticker": ticker,
        "company": ticker,
        "price": price,
        "direction": "BUY",
        "score": 80.0,
        "tp": (price + 10) if isinstance(price, (int, float)) and math.isfinite(price) else 0,
        "sl": (price - 10) if isinstance(price, (int, float)) and math.isfinite(price) else 0,
        "up_probability": 70.0,
        "down_probability": 10.0,
        "expected_value_pct": 1.0,
        "buy_reason": "test",
    }
    base.update(extra)
    return base


class TestNonFinitePriceGuard(unittest.TestCase):
    def test_nan_price_is_skipped_without_crash(self):
        s = _state()
        cands = [_cand("NAN1", float("nan"))]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(opened, [])
        self.assertEqual(s["positions"], [])

    def test_inf_price_is_skipped_without_crash(self):
        s = _state()
        cands = [_cand("INF1", float("inf"))]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(opened, [])

    def test_neg_inf_price_is_skipped_without_crash(self):
        s = _state()
        cands = [_cand("NINF1", float("-inf"))]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(opened, [])

    def test_zero_price_is_skipped(self):
        s = _state()
        cands = [_cand("ZERO1", 0.0)]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(opened, [])

    def test_negative_price_is_skipped(self):
        s = _state()
        cands = [_cand("NEG1", -500.0)]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(opened, [])

    def test_valid_price_still_opens_position(self):
        s = _state()
        cands = [_cand("OK1", 1000.0)]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["ticker"], "OK1")
        self.assertEqual(opened[0]["entry_price"], 1000.0)
        self.assertEqual(len(s["positions"]), 1)

    def test_invalid_candidate_does_not_block_later_valid_candidate(self):
        s = _state()
        cands = [_cand("NAN2", float("nan")), _cand("OK2", 1500.0)]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["ticker"], "OK2")

    def test_duplicate_ticker_handling_unchanged(self):
        s = _state()
        s["positions"].append({"ticker": "DUP1", "invested_amount": 0})
        cands = [_cand("DUP1", 1000.0), _cand("OK3", 1200.0)]
        opened = app.open_positions(s, _policy(), cands, "2026-09-21")
        self.assertEqual([p["ticker"] for p in opened], ["OK3"])


if __name__ == "__main__":
    unittest.main()
