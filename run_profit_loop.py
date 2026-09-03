#!/usr/bin/env python3
"""Unified Profit Loop: progressive levels -> TOP10 -> TOP1 paper trade.

Research/OOS gates remain separate. This runtime path is for paper execution only.
Both BUY and SHORT are evaluated. Market regime determines the direction:
Nikkei bullish -> BUY only, bearish -> SHORT only, neutral -> compare both.
"""
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd

import profit_top10_paper as app

TOP10 = 10
MAX_DAILY_TRADES = int(os.getenv("MAX_TRADES_PER_DAY", "30"))
MAX_TICKER_TRADES = int(os.getenv("MAX_TRADES_PER_TICKER_PER_DAY", "10"))
SAME_TICKER_COOLDOWN_MINUTES = int(os.getenv("SAME_TICKER_COOLDOWN_MINUTES", "30"))

PAPER_ENTRY_LEVELS = [
    {"level": 1, "up_threshold": 60.0, "min_score": 70.0, "nikkei_filter": True},
    {"level": 2, "up_threshold": 55.0, "min_score": 65.0, "nikkei_filter": True},
    {"level": 3, "up_threshold": 50.0, "min_score": 60.0, "nikkei_filter": False},
    {"level": 4, "up_threshold": 45.0, "min_score": 55.0, "nikkei_filter": False},
    {"level": 5, "up_threshold": 40.0, "min_score": 50.0, "nikkei_filter": False},
    {"level": 6, "up_threshold": 35.0, "min_score": 45.0, "nikkei_filter": False},
]

def _market_regime():
    try:
        nikkei = app.make_nikkei()
        if nikkei is None or nikkei.empty:
            return "neutral", None, None
        last = nikkei.ffill().iloc[-1]
        kairi = float(last["kairi25"])
        ret5 = float(last["ret5"])
        if kairi > 0 and ret5 > 0:
            return "bullish", kairi, ret5
        if kairi < 0 and ret5 < 0:
            return "bearish", kairi, ret5
        return "neutral", kairi, ret5
    except Exception as exc:
        print(f"⚠️ 日経レジーム判定失敗 → neutral: {exc}")
        return "neutral", None, None

def profit_priority(candidates):
    """Regime gate: bearish means SHORT candidates only; bullish means BUY only.
    Neutral compares BUY/SHORT by expected value and score."""
    regime, kairi25, ret5 = _market_regime()
    print(f"🌐 日経レジーム: {regime.upper()}" + (f"｜25MA乖離 {kairi25:+.2f}%｜5日騰落 {ret5:+.2f}%" if kairi25 is not None else ""))
    ranked = []
    for c in candidates:
        direction = str(c.get("direction", "BUY")).upper()
        if regime == "bullish" and direction != "BUY":
            continue
        if regime == "bearish" and direction != "SHORT":
            continue
        price = float(c.get("price", 0) or 0)
        tp = float(c.get("tp", 0) or 0)
        sl = float(c.get("sl", 0) or 0)
        up = float(c.get("up_probability", 0) or 0) / 100.0
        down = float(c.get("down_probability", 0) or 0) / 100.0
        if price <= 0:
            ev = -999.0
        elif direction == "SHORT":
            reward = max(0.0, (1.0 - tp / price) * 100.0)
            risk = max(0.0, (sl / price - 1.0) * 100.0)
            ev = down * reward - (1.0 - down) * risk
        else:
            reward = max(0.0, (tp / price - 1.0) * 100.0)
            risk = max(0.0, (1.0 - sl / price) * 100.0)
            ev = up * reward - (1.0 - up) * risk
        preferred = (regime == "bullish" and direction == "BUY") or (regime == "bearish" and direction == "SHORT")
        regime_bonus = 10.0 if preferred else 0.0
        rank = 0.65 * float(c.get("score", 0)) + 0.35 * max(-10.0, min(10.0, ev)) * 10.0 + regime_bonus
        item = dict(c)
        item["market_regime"] = regime
        item["regime_preferred"] = bool(preferred)
        item["regime_bonus"] = regime_bonus
        item["profit_ev_pct"] = round(ev, 4)
        item["profit_priority"] = round(rank, 4)
        ranked.append(item)
    return sorted(ranked, key=lambda x: (x["profit_priority"], x.get("score", 0), max(x.get("up_probability", 0), x.get("down_probability", 0))), reverse=True)

_original_scan = app.scan_candidates
_original_close = app.close_positions
_original_open = app.open_positions
_original_load_model = app.load_model

class _FeatureSafeModel:
    """Adapter that guarantees sklearn receives exactly the feature columns used at fit time."""
    def __init__(self, model):
        self._model = model
        self.classes_ = getattr(model, "classes_", np.array([0, 1, 2]))
        self.feature_names_in_ = getattr(model, "feature_names_in_", np.array([]))
    def predict_proba(self, X):
        cols = list(getattr(self._model, "feature_names_in_", []))
        if cols:
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError("trained feature missing: " + ", ".join(missing))
            X = X.loc[:, cols]
        return self._model.predict_proba(X)
    def __getattr__(self, name):
        return getattr(self._model, name)

def _load_model_feature_safe():
    model = _original_load_model()
    if model is None: return None
    return _FeatureSafeModel(model)

app.load_model = _load_model_feature_safe

def _policy_for_level(base_policy, spec):
    p = dict(base_policy); p["up_threshold"] = float(spec["up_threshold"]); p["min_score_for_buy"] = float(spec["min_score"]); p["nikkei_filter"] = False; return p

def _passes_level(candidate, spec):
    direction = str(candidate.get("direction", "BUY")).upper(); up = float(candidate.get("up_probability", 0) or 0); down = float(candidate.get("down_probability", 0) or 0); flat = float(candidate.get("flat_probability", 0) or 0); score = float(candidate.get("score", 0) or 0); threshold = float(spec["up_threshold"]); score_min = float(spec["min_score"])
    if flat >= 50.0 or score < score_min: return False
    if direction == "SHORT": return down >= threshold and down > up
    return up >= threshold and up > down

def _print_gate_diagnostics(emergency_pool, scanned):
    pool = emergency_pool or []; print("\n" + "=" * 86); print("🔎 PAPER候補ゲート診断（ペーパー専用。OOS/Adversarialとは独立）"); print("=" * 86); print(f"スキャン成功: {int(scanned or 0)}"); print("固定条件通過: BUY=UP>DOWN / SHORT=DOWN>UP かつ Flat<50%"); print(f"固定条件通過件数: {len(pool)}\n"); print("LEVEL | 方向 | 確率条件 | SCORE条件 | 両方通過"); print("------+-------+----------+------------+----------")
    for spec in PAPER_ENTRY_LEVELS:
        long_both = [c for c in pool if str(c.get("direction", "BUY")).upper() != "SHORT" and _passes_level(c, spec)]; short_both = [c for c in pool if str(c.get("direction", "BUY")).upper() == "SHORT" and _passes_level(c, spec)]
        print(f" {spec['level']:>2}   | BUY   | UP≥{spec['up_threshold']:>3.0f}%   | SCORE≥{spec['min_score']:>3.0f}     | {len(long_both):>8}"); print(f" {spec['level']:>2}   | SHORT | DOWN≥{spec['up_threshold']:>3.0f}% | SCORE≥{spec['min_score']:>3.0f}     | {len(short_both):>8}")
    print("注: 日経レジーム判定後、弱気=SHORTのみ、強気=BUYのみ、neutral=両方向比較でTOP1を決定します。\n" + "=" * 86)

def scan_candidates_progressive(policy):
    last_scanned = 0
    for spec in PAPER_ENTRY_LEVELS:
        raw, scanned = _original_scan(_policy_for_level(policy, spec)); last_scanned = max(last_scanned, int(scanned or 0)); pool = raw or []; qualified = [c for c in pool if _passes_level(c, spec)]; ranked = profit_priority(qualified); top10 = ranked[:TOP10]
        print(f"🧭 PAPER LEVEL {spec['level']}: BUY UP≥{spec['up_threshold']:.0f}% / SHORT DOWN≥{spec['up_threshold']:.0f}% SCORE≥{spec['min_score']:.0f} REGIME-AWARE qualified={len(qualified)} / TOP10={len(top10)}")
        if top10:
            print(f"🏁 実行LEVEL={spec['level']} / 候補={len(top10)} → TOP1へ")
            for rank, c in enumerate(top10, 1):
                c["selection_level"] = int(spec["level"]); c["selection_mode"] = "normal" if spec["level"] == 1 else "progressive_level"; c["top10_rank"] = rank
            return top10, last_scanned
        print("  ↳ 候補0 → 次のLEVELへ条件緩和")
    emergency_policy = dict(policy); emergency_policy["up_threshold"] = 0.0; emergency_policy["min_score_for_buy"] = 0.0; emergency_policy["nikkei_filter"] = False
    try: emergency_pool, emergency_scanned = _original_scan(emergency_policy)
    except Exception as exc: emergency_pool, emergency_scanned = [], 0; print(f"⚠️ PAPER強制経路の再スキャン失敗: {exc}")
    last_scanned = max(last_scanned, int(emergency_scanned or 0)); _print_gate_diagnostics(emergency_pool, last_scanned)
    if emergency_pool:
        ranked = profit_priority(emergency_pool); top10 = ranked[:TOP10]
        for rank, c in enumerate(top10, 1): c["selection_level"] = len(PAPER_ENTRY_LEVELS) + 1; c["selection_mode"] = "forced_min_trade"; c["top10_rank"] = rank
        print(f"🟠 最終PAPER強制経路: スキャン済み候補={len(emergency_pool)} → TOP10={len(top10)} → TOP1を選出"); return top10, last_scanned
    print("❌ 銘柄データ自体を取得できないため、paper trade候補を生成できません"); return [], last_scanned

def _as_aware_jst(value):
    ts = value if isinstance(value, datetime) else app.pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None: return ts.replace(tzinfo=app.TZ)
    return ts.astimezone(app.TZ)

def close_positions_with_cooldown(state, policy, now):
    before = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}; messages = _original_close(state, policy, now); after = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}; cooldowns = state.setdefault("last_exit_by_ticker", {})
    for ticker in sorted(before - after): cooldowns[ticker] = app.pd.Timestamp(now).isoformat(); print(f"⏳ 同一銘柄クールダウン開始: {ticker} {SAME_TICKER_COOLDOWN_MINUTES}分")
    return messages

def open_top1_only(state, policy, candidates, today):
    cooldowns = state.setdefault("last_exit_by_ticker", {}); now = datetime.now(app.TZ); active = {str(p.get("ticker")) for p in state.get("positions", []) if p.get("ticker")}; eligible = []
    regime, _, _ = _market_regime()
    for candidate in profit_priority(candidates)[:TOP10]:
        ticker = str(candidate.get("ticker", "")).strip(); direction = str(candidate.get("direction", "BUY")).upper()
        if regime == "bearish" and direction != "SHORT": continue
        if regime == "bullish" and direction != "BUY": continue
        if not ticker or ticker in active: continue
        if int(state.get("trades_by_ticker_today", {}).get(ticker, 0)) >= MAX_TICKER_TRADES: continue
        if int(state.get("trades_today", 0)) >= MAX_DAILY_TRADES: break
        raw = cooldowns.get(ticker)
        if raw:
            try: remaining = (_as_aware_jst(raw) + timedelta(minutes=SAME_TICKER_COOLDOWN_MINUTES) - now).total_seconds()
            except Exception: remaining = 0
            if remaining > 0: print(f"⏸ 同一銘柄クールダウン中: {ticker} 残り約{int(remaining // 60) + 1}分"); continue
            cooldowns.pop(ticker, None)
        eligible.append(candidate)
    if not eligible: print("⏸ 候補内に新規エントリー可能なTOP1なし"); return []
    top1 = eligible[0]; old_top_n = app.TOP_N; old_max_total = app.MAX_TOTAL_TRADES_PER_DAY; old_max_ticker = app.MAX_TRADES_PER_TICKER_PER_DAY
    try:
        app.TOP_N = 1; app.MAX_TOTAL_TRADES_PER_DAY = MAX_DAILY_TRADES; app.MAX_TRADES_PER_TICKER_PER_DAY = MAX_TICKER_TRADES; opened = _original_open(state, policy, [top1], today)
    finally:
        app.TOP_N = old_top_n; app.MAX_TOTAL_TRADES_PER_DAY = old_max_total; app.MAX_TRADES_PER_TICKER_PER_DAY = old_max_ticker
    if opened:
        p = state["positions"][-1]; p["allocation"] = 1.0; p["selection_mode"] = top1.get("selection_mode", "normal"); p["selection_level"] = int(top1.get("selection_level", 1)); p["top10_rank"] = int(top1.get("top10_rank", 1)); p["market_regime"] = top1.get("market_regime", regime); p["regime_preferred"] = bool(top1.get("regime_preferred", False)); p["profit_ev_pct"] = float(top1.get("profit_ev_pct", 0.0)); p["profit_priority"] = float(top1.get("profit_priority", 0.0)); print(f"🏆 TOP→TOP1 ENTRY: {top1.get('direction', 'BUY')} {top1['ticker']} LEVEL={p['selection_level']} MODE={p['selection_mode']} REGIME={p['market_regime']} PREFERRED={p['regime_preferred']} score={top1['score']:.1f} UP={top1['up_probability']:.1f}% DOWN={top1.get('down_probability', 0):.1f}%")
    return opened

app.scan_candidates = scan_candidates_progressive
app.close_positions = close_positions_with_cooldown
app.open_positions = open_top1_only

if __name__ == "__main__": app.main()
