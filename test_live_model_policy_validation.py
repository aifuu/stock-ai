"""Unit tests for live_model_policy_validation.py (research-only, no live impact).

Network is blocked before import (same pattern as
test_run_profit_loop_cooldown.py) because live_model_policy_validation.py
imports daily_directional_top1 / daily_model_retrain / profit_top10_paper /
futures_trend / paper_risk_policy, none of which may make a network call
merely by being imported.

run_profit_loop.py is never imported here (see the big module docstring in
live_model_policy_validation.py for why: importing it monkey-patches
profit_top10_paper's module-level scan/open_positions/mark_and_close/
load_model attributes as a side effect of import, which would leak into any
other test module sharing this process). Instead, the two pure decision
functions this repo actually uses live (`_passes_policy` and
`profit_priority`) are extracted from run_profit_loop.py's source via `ast`
(same technique as test_adversarial_strategy_validator_boundary_purge.py)
and executed in an isolated namespace, so the comparison below is against
the *real* live source text, not a hand-copied duplicate of it.
"""
import ast
import os
import types
import unittest
from pathlib import Path

os.environ.pop("DISCORD_WEBHOOK", None)
os.environ.setdefault("PAPER_TRADE_MODE", "1")

import requests  # noqa: E402


def _blocked_post(*_args, **_kwargs):
    raise AssertionError("network call attempted via requests.post in tests")


requests.post = _blocked_post

import yfinance as yf  # noqa: E402

yf.download = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("network call attempted via yfinance.download in tests")
)

import pandas as pd  # noqa: E402

import live_model_policy_validation as lmv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
RUN_PROFIT_LOOP_PATH = REPO_ROOT / "run_profit_loop.py"


def _load_live_decision_functions():
    """Extract profit_priority()/_passes_policy() from run_profit_loop.py's
    own source (never imports the module, so its import-time monkeypatches
    of profit_top10_paper never run)."""
    source = RUN_PROFIT_LOOP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUN_PROFIT_LOOP_PATH))
    wanted_names = {"profit_priority", "_passes_policy"}
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted_names]
    found = {n.name for n in wanted}
    missing = wanted_names - found
    assert not missing, f"run_profit_loop.py: functions not found: {missing}"

    module_ast = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module_ast)

    regime_holder = {"value": ("neutral", None, None)}
    namespace = {
        "_market_regime": lambda: regime_holder["value"],
        "_load_feedback_weights": lambda: {"BUY": 1.0, "SHORT": 1.0},
        "app": types.SimpleNamespace(FEE_RATE=0.00055),
    }
    exec(compile(module_ast, filename=str(RUN_PROFIT_LOOP_PATH), mode="exec"), namespace)
    return namespace["profit_priority"], namespace["_passes_policy"], regime_holder


LIVE_PROFIT_PRIORITY, LIVE_PASSES_POLICY, _REGIME_HOLDER = _load_live_decision_functions()


def _candidate(ticker="7203.T", direction="BUY", price=1000.0, tp=1100.0, sl=950.0,
                up=60.0, down=20.0, flat=20.0, score=70.0):
    return {
        "ticker": ticker, "direction": direction, "price": price, "tp": tp, "sl": sl,
        "up_probability": up, "down_probability": down, "flat_probability": flat, "score": score,
    }


class PassesPolicyMatchesLive(unittest.TestCase):
    """live_model_policy_validation.passes_policy() must be value-identical
    to run_profit_loop._passes_policy() (byte-extracted from its own source).

    Note: neither the real function nor our replica checks nikkei_filter --
    this is the documented live behavior (see the module docstring finding
    #3), not an omission on our side.
    """

    def _cases(self):
        policy = {"up_threshold": 20.0, "min_score_for_buy": 40.0, "nikkei_filter": True}
        cases = [
            _candidate(direction="BUY", up=60, down=10, flat=30, score=50),
            _candidate(direction="BUY", up=10, down=60, flat=30, score=50),  # up<down -> fail
            _candidate(direction="BUY", up=60, down=10, flat=30, score=10),  # score<min -> fail
            _candidate(direction="BUY", up=60, down=10, flat=55, score=50),  # flat>=50 -> fail
            _candidate(direction="SHORT", up=10, down=60, flat=30, score=50),
            _candidate(direction="SHORT", up=60, down=10, flat=30, score=50),  # down<up -> fail
            _candidate(direction="BUY", up=20.0, down=5, flat=10, score=40.0),  # exact threshold
        ]
        return policy, cases

    def test_matches_live_function_on_all_cases(self):
        policy, cases = self._cases()
        for c in cases:
            with self.subTest(c=c):
                live = LIVE_PASSES_POLICY(c, policy)
                mine = lmv.passes_policy(c["direction"], c["up_probability"], c["down_probability"],
                                          c["flat_probability"], c["score"], policy)
                self.assertEqual(bool(live), bool(mine))


class ProfitPriorityMatchesLive(unittest.TestCase):
    """live_model_policy_validation's (profit_priority_rank + regime_allows)
    combination must reproduce run_profit_loop.profit_priority()'s rank
    value and regime hard-gate for each candidate, and the resulting sort
    order, on constructed candidate pools."""

    def _run_live(self, candidates, regime):
        _REGIME_HOLDER["value"] = (regime, 1.0 if regime == "bullish" else (-1.0 if regime == "bearish" else 0.0),
                                    1.0 if regime == "bullish" else (-1.0 if regime == "bearish" else 0.0))
        return LIVE_PROFIT_PRIORITY(candidates)

    def _run_mine(self, candidates, regime):
        out = []
        for c in candidates:
            if not lmv.regime_allows(regime, c["direction"]):
                continue
            rank, ev, bonus = lmv.profit_priority_rank(
                c["direction"], c["score"], c["price"], c["tp"], c["sl"],
                c["up_probability"] / 100.0, c["down_probability"] / 100.0, c["flat_probability"] / 100.0,
                regime, lmv.FEE_RATE_PCT,
            )
            item = dict(c)
            item["profit_priority"] = round(rank, 4)
            out.append(item)
        out.sort(key=lambda x: (x["profit_priority"], x.get("score", 0),
                                 max(x.get("up_probability", 0), x.get("down_probability", 0))), reverse=True)
        return out

    def test_neutral_regime_mixed_directions(self):
        candidates = [
            _candidate(ticker="A", direction="BUY", up=70, down=10, flat=20, score=80),
            _candidate(ticker="B", direction="SHORT", up=10, down=65, flat=25, score=60),
            _candidate(ticker="C", direction="BUY", up=55, down=20, flat=25, score=40),
        ]
        live = self._run_live(candidates, "neutral")
        mine = self._run_mine(candidates, "neutral")
        self.assertEqual([c["ticker"] for c in live], [c["ticker"] for c in mine])
        for lc, mc in zip(live, mine):
            self.assertAlmostEqual(lc["profit_priority"], mc["profit_priority"], places=6)

    def test_bullish_regime_filters_short(self):
        candidates = [
            _candidate(ticker="A", direction="BUY", up=70, down=10, flat=20, score=80),
            _candidate(ticker="B", direction="SHORT", up=10, down=65, flat=25, score=95),
        ]
        live = self._run_live(candidates, "bullish")
        mine = self._run_mine(candidates, "bullish")
        self.assertEqual([c["ticker"] for c in live], ["A"])
        self.assertEqual([c["ticker"] for c in mine], ["A"])

    def test_bearish_regime_filters_buy(self):
        candidates = [
            _candidate(ticker="A", direction="BUY", up=70, down=10, flat=20, score=80),
            _candidate(ticker="B", direction="SHORT", up=10, down=65, flat=25, score=50),
        ]
        live = self._run_live(candidates, "bearish")
        mine = self._run_mine(candidates, "bearish")
        self.assertEqual([c["ticker"] for c in live], ["B"])
        self.assertEqual([c["ticker"] for c in mine], ["B"])


class NaturalDirectionTests(unittest.TestCase):
    def test_buy_when_up_greater(self):
        self.assertEqual(lmv.natural_direction(60, 20, 20), "BUY")

    def test_short_when_down_greater_and_enabled(self):
        self.assertEqual(lmv.natural_direction(20, 60, 20, short_enabled=True), "SHORT")

    def test_short_disabled_returns_none(self):
        self.assertIsNone(lmv.natural_direction(20, 60, 20, short_enabled=False))

    def test_flat_dominant_returns_none(self):
        self.assertIsNone(lmv.natural_direction(30, 20, 50))


class RegimeFromNikkeiTests(unittest.TestCase):
    def test_bullish(self):
        self.assertEqual(lmv.market_regime_from_nikkei(1.0, 1.0), "bullish")

    def test_bearish(self):
        self.assertEqual(lmv.market_regime_from_nikkei(-1.0, -1.0), "bearish")

    def test_neutral_mixed_signs(self):
        self.assertEqual(lmv.market_regime_from_nikkei(1.0, -1.0), "neutral")

    def test_neutral_on_missing_data(self):
        self.assertEqual(lmv.market_regime_from_nikkei(None, None), "neutral")
        self.assertEqual(lmv.market_regime_from_nikkei(float("nan"), 1.0), "neutral")


class SelectPolicyForDateTests(unittest.TestCase):
    def test_up_trend_uses_up_policy(self):
        normal, up = {"name": "normal"}, {"name": "up"}
        self.assertIs(lmv.select_policy_for_date(lmv.futures_trend.UP, normal, up), up)

    def test_down_trend_falls_back_to_normal_policy(self):
        # strategy_policy_down.json does not exist in this repo, so live
        # profit_top10_paper.select_policy_file() falls back to
        # strategy_policy.json on DOWN days -- mirrored here.
        normal, up = {"name": "normal"}, {"name": "up"}
        self.assertIs(lmv.select_policy_for_date(lmv.futures_trend.DOWN, normal, up), normal)


class CheckExitTests(unittest.TestCase):
    def test_buy_take_profit(self):
        price, reason = lmv.check_exit("BUY", high=110, low=95, close=108, tp=110, sl=90, days_held=1, hold_limit=5)
        self.assertEqual((price, reason), (110, "TP"))

    def test_buy_stop_loss(self):
        price, reason = lmv.check_exit("BUY", high=100, low=85, close=90, tp=110, sl=90, days_held=1, hold_limit=5)
        self.assertEqual((price, reason), (90, "SL"))

    def test_buy_both_touched_prefers_stop_loss(self):
        price, reason = lmv.check_exit("BUY", high=120, low=80, close=100, tp=110, sl=90, days_held=1, hold_limit=5)
        self.assertEqual((price, reason), (90, "SL_BOTH"))

    def test_short_take_profit(self):
        price, reason = lmv.check_exit("SHORT", high=101, low=89, close=95, tp=90, sl=110, days_held=1, hold_limit=5)
        self.assertEqual((price, reason), (90, "TP"))

    def test_short_stop_loss(self):
        price, reason = lmv.check_exit("SHORT", high=115, low=100, close=112, tp=90, sl=110, days_held=1, hold_limit=5)
        self.assertEqual((price, reason), (110, "SL"))

    def test_hold_limit_forces_time_exit_at_close(self):
        price, reason = lmv.check_exit("BUY", high=105, low=99, close=103, tp=200, sl=50, days_held=5, hold_limit=5)
        self.assertEqual((price, reason), (103, "TIME"))

    def test_no_exit_before_hold_limit_and_no_touch(self):
        price, reason = lmv.check_exit("BUY", high=105, low=99, close=103, tp=200, sl=50, days_held=2, hold_limit=5)
        self.assertEqual((price, reason), (None, None))


class TradeReturnFeeTests(unittest.TestCase):
    def test_buy_return_includes_roundtrip_fee(self):
        fee_pct = 0.00055 * 2 * 100.0
        ret = lmv.trade_return_pct("BUY", entry=100.0, exit_price=110.0, fee_rate_pct=fee_pct)
        self.assertAlmostEqual(ret, 10.0 - fee_pct, places=9)

    def test_short_return_includes_roundtrip_fee(self):
        fee_pct = 0.00055 * 2 * 100.0
        ret = lmv.trade_return_pct("SHORT", entry=100.0, exit_price=90.0, fee_rate_pct=fee_pct)
        self.assertAlmostEqual(ret, (100.0 / 90.0 - 1.0) * 100.0 - fee_pct, places=9)

    def test_module_fee_rate_pct_matches_live_fee_rate_definition(self):
        # profit_top10_paper.FEE_RATE is the one-way fee; live round-trip
        # cost (as used by run_profit_loop.profit_priority's flat_cost and
        # daily_model_retrain.FEE_RATE_PCT) is FEE_RATE * 2 * 100.
        import profit_top10_paper as papertrader
        self.assertAlmostEqual(lmv.FEE_RATE_PCT, papertrader.FEE_RATE * 2 * 100.0, places=9)


class BuildFoldsTests(unittest.TestCase):
    def _dates(self, n):
        return list(pd.bdate_range("2022-01-03", periods=n))

    def test_recent_fold_is_last_n_days(self):
        dates = self._dates(600)
        folds = lmv.build_folds(dates, hold_days_for_purge=1)
        recent = [f for f in folds if f.name == "recent60"][0]
        self.assertEqual(len(recent.oos_dates), lmv.RECENT_FOLD_DAYS)
        self.assertEqual(list(recent.oos_dates), dates[-lmv.RECENT_FOLD_DAYS:])

    def test_main_folds_do_not_overlap_recent_fold(self):
        dates = self._dates(600)
        folds = lmv.build_folds(dates, hold_days_for_purge=1)
        recent = [f for f in folds if f.name == "recent60"][0]
        recent_set = set(recent.oos_dates)
        for f in folds:
            if f.name == "recent60":
                continue
            self.assertTrue(recent_set.isdisjoint(set(f.oos_dates)))

    def test_folds_are_chronological_and_non_overlapping(self):
        dates = self._dates(600)
        folds = lmv.build_folds(dates, hold_days_for_purge=1)
        main_folds = [f for f in folds if f.name != "recent60"]
        self.assertEqual(len(main_folds), lmv.N_FOLDS)
        for a, b in zip(main_folds, main_folds[1:]):
            self.assertLess(max(a.oos_dates), min(b.oos_dates))

    def test_train_cutoff_is_strictly_before_fold_start(self):
        dates = self._dates(600)
        folds = lmv.build_folds(dates, hold_days_for_purge=3)
        for f in folds:
            self.assertLess(f.train_cutoff, pd.Timestamp(f.oos_dates[0]))

    def test_too_short_history_raises(self):
        dates = self._dates(50)
        with self.assertRaises(RuntimeError):
            lmv.build_folds(dates, hold_days_for_purge=1)


class MetricsTests(unittest.TestCase):
    def test_compute_flat_metrics_empty(self):
        m = lmv.compute_flat_metrics([])
        self.assertEqual(m["trades"], 0)
        self.assertEqual(m["pf"], 0.0)

    def test_compute_flat_metrics_pf_and_win_rate(self):
        trades = [{"return_pct": 10.0}, {"return_pct": -5.0}, {"return_pct": 5.0}]
        m = lmv.compute_flat_metrics(trades)
        self.assertEqual(m["trades"], 3)
        self.assertAlmostEqual(m["pf"], 15.0 / 5.0)
        self.assertAlmostEqual(m["win_rate"], 200.0 / 3.0, places=6)

    def test_compute_flat_metrics_all_wins_is_infinite_pf(self):
        m = lmv.compute_flat_metrics([{"return_pct": 3.0}])
        self.assertEqual(m["pf"], float("inf"))

    def test_equity_curve_metrics_max_dd(self):
        idx = pd.date_range("2022-01-01", periods=4, freq="D")
        s = pd.Series([10.0, -10.0, -10.0, 5.0], index=idx)
        out = lmv.compute_equity_curve_metrics(s)
        self.assertLess(out["max_dd_pct"], 0.0)

    def test_aggregate_top1_uses_equity_points(self):
        trades = [{"return_pct": 5.0}]
        equity_points = [{"date": pd.Timestamp("2022-01-05"), "capital": 1_050_000.0}]
        out = lmv.aggregate_top1(trades, equity_points)
        self.assertEqual(out["trades"], 1)
        self.assertIn("avg_month_return", out)
        self.assertIn("max_dd_pct", out)

    def test_aggregate_equal_weight_groups_by_entry_date(self):
        trades = [
            {"entry_date": pd.Timestamp("2022-01-05"), "return_pct": 10.0},
            {"entry_date": pd.Timestamp("2022-01-05"), "return_pct": -10.0},
        ]
        out = lmv.aggregate_equal_weight(trades)
        self.assertEqual(out["trades"], 2)


class DiagnosticVsWalkForwardTests(unittest.TestCase):
    def test_empty_inputs_return_none_metrics(self):
        out = lmv.diagnostic_vs_walk_forward([], None)
        self.assertEqual(out["overlap_days"], 0)
        self.assertIsNone(out["spearman_up_prob"])

    def test_perfect_agreement_gives_full_overlap_and_correlation_one(self):
        live_rows = [
            {"date": "2024-01-02", "ticker": "A", "up_probability": 90.0, "score": 80.0},
            {"date": "2024-01-02", "ticker": "B", "up_probability": 40.0, "score": 30.0},
            {"date": "2024-01-03", "ticker": "A", "up_probability": 70.0, "score": 60.0},
            {"date": "2024-01-03", "ticker": "B", "up_probability": 20.0, "score": 10.0},
        ]
        wf_df = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "A", "up_prob": 90.0, "score": 80.0},
            {"date": "2024-01-02", "ticker": "B", "up_prob": 40.0, "score": 30.0},
            {"date": "2024-01-03", "ticker": "A", "up_prob": 70.0, "score": 60.0},
            {"date": "2024-01-03", "ticker": "B", "up_prob": 20.0, "score": 10.0},
        ])
        out = lmv.diagnostic_vs_walk_forward(live_rows, wf_df)
        self.assertEqual(out["overlap_days"], 2)
        self.assertAlmostEqual(out["spearman_up_prob"], 1.0)
        self.assertAlmostEqual(out["spearman_score"], 1.0)
        self.assertAlmostEqual(out["top1_overlap_rate"], 100.0)


class DownloadPeriodMonkeypatchTests(unittest.TestCase):
    """_extend_download_period must only change the period argument it
    forwards, and must be restored by the caller (build_universe does this
    in a try/finally)."""

    def test_wrapped_forwards_period_and_restores_original(self):
        calls = []

        def fake_download(ticker, period="3y"):
            calls.append((ticker, period))
            return "DATA"

        original = lmv.trader.download
        lmv.trader.download = fake_download
        try:
            restored = lmv._extend_download_period("5y")
            self.assertIs(restored, fake_download)
            result = lmv.trader.download("7203.T")
            self.assertEqual(result, "DATA")
            self.assertEqual(calls, [("7203.T", "5y")])
        finally:
            lmv.trader.download = original


class SimulateTop1EndToEndTests(unittest.TestCase):
    """End-to-end test of simulate_top1's orchestration: entry convention
    (that day's Close), TP/SL/hold_days exit via check_exit, round-trip fee,
    and drawdown-based position sizing -- with build_day_candidates stubbed
    out so the test does not need to fabricate all ~40 FEATURES columns
    (those pure per-ticker functions are exercised directly by the tests
    above and by ProfitPriorityMatchesLive / PassesPolicyMatchesLive)."""

    def _frame(self, dates, high, low, close):
        return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=pd.DatetimeIndex(dates))

    def test_single_trade_tp_exit_and_fee_and_compounding(self):
        dates = list(pd.bdate_range("2024-01-02", periods=5))
        # Ticker A: day0 entry close=100; day1 high touches TP=110.
        ticker_frames = {
            "A": self._frame(dates, high=[100, 111, 100, 100, 100], low=[99, 105, 95, 95, 95],
                              close=[100, 108, 100, 100, 100]),
        }
        policy = {"hold_days": 5, "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0,
                  "up_threshold": 0, "min_score_for_buy": 0}

        call_count = {"n": 0}

        def fake_build_day_candidates(date, tf, model, features, policy_, regime, fee_rate_pct):
            call_count["n"] += 1
            if call_count["n"] > 1:
                return []
            price = float(tf["A"].loc[date, "Close"])
            return [{"ticker": "A", "direction": "BUY", "score": 90.0,
                     "up_probability": 80.0, "down_probability": 5.0,
                     "price": price, "tp": price + 10.0, "sl": price - 10.0,
                     "rank": 50.0, "ev": 1.0}]

        original = lmv.build_day_candidates
        lmv.build_day_candidates = fake_build_day_candidates
        try:
            nikkei_ff = pd.DataFrame({"kairi25": [0.0] * 5, "ret5": [0.0] * 5}, index=pd.DatetimeIndex(dates))
            trend_by_date = {d: lmv.futures_trend.DOWN for d in dates}
            trades, equity_points = lmv.simulate_top1(
                dates, ticker_frames, model=None, features=[], nikkei_ff=nikkei_ff,
                trend_by_date=trend_by_date, policy_normal=policy, policy_up=policy,
                fee_rate_pct=lmv.FEE_RATE_PCT,
            )
        finally:
            lmv.build_day_candidates = original

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["entry_date"], dates[0])
        self.assertEqual(trade["exit_date"], dates[1])
        self.assertEqual(trade["reason"], "TP")
        # entry = day0 Close (documented entry-price convention) = 100, tp = 110
        expected_ret = (110.0 / 100.0 - 1.0) * 100.0 - lmv.FEE_RATE_PCT
        self.assertAlmostEqual(trade["return_pct"], expected_ret, places=9)
        expected_capital = lmv.paper_risk_policy.INITIAL_CAPITAL * (1 + expected_ret / 100.0)
        self.assertAlmostEqual(trade["capital_after"], expected_capital, places=3)
        self.assertEqual(len(equity_points), 1)

    def test_hold_days_time_exit(self):
        dates = list(pd.bdate_range("2024-01-02", periods=4))
        ticker_frames = {
            "A": self._frame(dates, high=[100, 101, 101, 101], low=[99, 99, 99, 99],
                              close=[100, 101, 101, 102]),
        }
        policy = {"hold_days": 2, "atr_tp_multiplier": 5.0, "atr_sl_multiplier": 5.0,
                  "up_threshold": 0, "min_score_for_buy": 0}
        call_count = {"n": 0}

        def fake_build_day_candidates(date, tf, model, features, policy_, regime, fee_rate_pct):
            call_count["n"] += 1
            if call_count["n"] > 1:
                return []
            price = float(tf["A"].loc[date, "Close"])
            return [{"ticker": "A", "direction": "BUY", "score": 90.0,
                     "up_probability": 80.0, "down_probability": 5.0,
                     "price": price, "tp": price + 1000.0, "sl": price - 1000.0,
                     "rank": 50.0, "ev": 1.0}]

        original = lmv.build_day_candidates
        lmv.build_day_candidates = fake_build_day_candidates
        try:
            nikkei_ff = pd.DataFrame({"kairi25": [0.0] * 4, "ret5": [0.0] * 4}, index=pd.DatetimeIndex(dates))
            trend_by_date = {d: lmv.futures_trend.DOWN for d in dates}
            trades, _ = lmv.simulate_top1(
                dates, ticker_frames, model=None, features=[], nikkei_ff=nikkei_ff,
                trend_by_date=trend_by_date, policy_normal=policy, policy_up=policy,
                fee_rate_pct=lmv.FEE_RATE_PCT,
            )
        finally:
            lmv.build_day_candidates = original

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "TIME")
        # hold_days=2: exit on the 2nd bar after entry (dates[2]) at its Close.
        self.assertEqual(trades[0]["exit_date"], dates[2])


class SimulateTopNEqualWeightTests(unittest.TestCase):
    def test_top_n_limits_picks_per_day(self):
        # 4 trading days with hold_days=1: each day's pick needs one future
        # bar to resolve its exit (mirrors adversarial_strategy_validator's
        # evaluate_trade(), which drops a signal with no future bar left),
        # so only the first 3 of the 4 days actually produce a trade.
        dates = list(pd.bdate_range("2024-01-02", periods=4))
        frames = {}
        for t in ("A", "B", "C"):
            frames[t] = pd.DataFrame(
                {"High": [105] * 4, "Low": [95] * 4, "Close": [100] * 4},
                index=pd.DatetimeIndex(dates),
            )
        policy = {"hold_days": 1, "atr_tp_multiplier": 1.0, "atr_sl_multiplier": 1.0,
                  "up_threshold": 0, "min_score_for_buy": 0}

        def fake_build_day_candidates(date, tf, model, features, policy_, regime, fee_rate_pct):
            out = []
            for i, t in enumerate(("A", "B", "C")):
                price = float(tf[t].loc[date, "Close"])
                out.append({"ticker": t, "direction": "BUY", "score": 90.0 - i,
                            "up_probability": 80.0, "down_probability": 5.0,
                            "price": price, "tp": price + 10, "sl": price - 10,
                            "rank": 90.0 - i, "ev": 1.0})
            return out

        original = lmv.build_day_candidates
        lmv.build_day_candidates = fake_build_day_candidates
        try:
            nikkei_ff = pd.DataFrame({"kairi25": [0.0] * 4, "ret5": [0.0] * 4}, index=pd.DatetimeIndex(dates))
            trend_by_date = {d: lmv.futures_trend.DOWN for d in dates}
            trades_top1 = lmv.simulate_topn(dates, frames, None, [], nikkei_ff, trend_by_date,
                                             policy, policy, lmv.FEE_RATE_PCT, top_n=1)
            trades_all = lmv.simulate_topn(dates, frames, None, [], nikkei_ff, trend_by_date,
                                            policy, policy, lmv.FEE_RATE_PCT, top_n=None)
        finally:
            lmv.build_day_candidates = original

        resolvable_days = len(dates) - 1
        self.assertEqual(len(trades_top1), resolvable_days * 1)
        self.assertEqual(len(trades_all), resolvable_days * 3)
        self.assertTrue(all(t["ticker"] == "A" for t in trades_top1))


class ScriptHygieneTests(unittest.TestCase):
    def test_py_compiles(self):
        import py_compile
        py_compile.compile(str(REPO_ROOT / "live_model_policy_validation.py"), doraise=True)
        py_compile.compile(str(Path(__file__)), doraise=True)


if __name__ == "__main__":
    unittest.main()
