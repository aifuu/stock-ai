#!/usr/bin/env python3
"""Unified Profit Loop: fixed approved policy -> TOP10 -> TOP1 paper trade.

Research/OOS gates remain separate. This runtime path is for paper execution only.
Both BUY and SHORT are evaluated. Market regime determines the direction:
Nikkei bullish -> BUY only, bearish -> SHORT only, neutral -> compare both.
"""
from datetime import datetime, timedelta
import json
import os
import numpy as np
import pandas as pd

import profit_top10_paper as app

TOP10 = 10
MAX_DAILY_TRADES = int(os.getenv("MAX_TRADES_PER_DAY", "30"))
MAX_TICKER_TRADES = int(os.getenv("MAX_TRADES_PER_TICKER_PER_DAY", "10"))
SAME_TICKER_COOLDOWN_MINUTES = int(os.getenv("SAME_TICKER_COOLDOWN_MINUTES", "30"))

# ★追加(2026-09): trade_feedback_engine.py(profit_top10_paper_history.csvの実績を
# 集計)が出力するdirection_weightsを、TOP1優先順位付けに反映する。これまでこの
# ファイルはどこからも読まれておらず、「取引しながら成長するAI」の輪が閉じて
# いなかった(フィードバック分析自体もファイル参照ミスで常に空だった、別途修正済み)。
FEEDBACK_POLICY_FILE = "trade_feedback_policy.json"
_FEEDBACK_WEIGHTS_CACHE = None


def _load_feedback_weights():
    global _FEEDBACK_WEIGHTS_CACHE
    if _FEEDBACK_WEIGHTS_CACHE is not None:
        return _FEEDBACK_WEIGHTS_CACHE
    weights = {"BUY": 1.0, "SHORT": 1.0}
    try:
        with open(FEEDBACK_POLICY_FILE, encoding="utf-8") as f:
            policy = json.load(f)
        raw = policy.get("direction_weights", {})
        for key in ("BUY", "SHORT"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and 0.5 <= value <= 2.0:
                weights[key] = float(value)
    except Exception:
        pass
    _FEEDBACK_WEIGHTS_CACHE = weights
    return weights

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
    feedback_weights = _load_feedback_weights()
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
        flat = float(c.get("flat_probability", 0) or 0) / 100.0
        flat_cost = -(app.FEE_RATE * 2 * 100.0)
        if price <= 0:
            ev = -999.0
        elif direction == "SHORT":
            reward = max(0.0, (1.0 - tp / price) * 100.0)
            risk = max(0.0, (sl / price - 1.0) * 100.0)
            ev = down * reward - up * risk + flat * flat_cost
        else:
            reward = max(0.0, (tp / price - 1.0) * 100.0)
            risk = max(0.0, (1.0 - sl / price) * 100.0)
            ev = up * reward - down * risk + flat * flat_cost
        preferred = (regime == "bullish" and direction == "BUY") or (regime == "bearish" and direction == "SHORT")
        regime_bonus = 10.0 if preferred else 0.0
        feedback_weight = feedback_weights.get(direction, 1.0)
        rank = (0.65 * float(c.get("score", 0)) + 0.35 * max(-10.0, min(10.0, ev)) * 10.0 + regime_bonus) * feedback_weight
        item = dict(c)
        item["market_regime"] = regime
        item["regime_preferred"] = bool(preferred)
        item["regime_bonus"] = regime_bonus
        item["profit_ev_pct"] = round(ev, 4)
        item["feedback_weight"] = feedback_weight
        item["profit_priority"] = round(rank, 4)
        ranked.append(item)
    return sorted(ranked, key=lambda x: (x["profit_priority"], x.get("score", 0), max(x.get("up_probability", 0), x.get("down_probability", 0))), reverse=True)

_original_scan = app.scan
_original_close = app.mark_and_close
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

def _passes_policy(candidate, policy):
    direction = str(candidate.get("direction", "BUY")).upper()
    up = float(candidate.get("up_probability", 0) or 0)
    down = float(candidate.get("down_probability", 0) or 0)
    flat = float(candidate.get("flat_probability", 0) or 0)
    score = float(candidate.get("score", 0) or 0)
    threshold = float(policy["up_threshold"])
    score_min = float(policy["min_score_for_buy"])
    if flat >= 50.0 or score < score_min: return False
    if direction == "SHORT": return down >= threshold and down > up
    return up >= threshold and up > down

def scan_candidates_fixed(policy):
    """承認済みpolicy(strategy_policy.json)の固定条件で1回だけスキャンする。
    ★変更(2026-09): 従来のLEVEL段階的緩和(PAPER_ENTRY_LEVELS)と、それでも
    候補0件だった場合の無条件強制エントリ(emergency_policy, up_threshold=0)を
    廃止した。LEVELの閾値は承認済みpolicy(例: UP45/SCORE60)と一致しない
    ハードコード値で、実運用では毎回のように緊い方の水準(LEVEL6や強制経路)まで
    条件が緩和されてから約定しており、検証済みのOOS実績(PF/月次収益率)を
    大きく下回る弱い/無条件のシグナルで取引してしまっていた。「毎月プラスを
    優先」する方針では、条件を満たす候補が無い日は無理に建てず見送る方が良い。
    """
    raw, scanned = _original_scan(policy)
    pool = raw or []
    qualified = [c for c in pool if _passes_policy(c, policy)]
    top10 = profit_priority(qualified)[:TOP10]
    print(f"🧭 PAPER FIXED POLICY: BUY UP≥{policy['up_threshold']:.0f}% / SHORT DOWN≥{policy['up_threshold']:.0f}% SCORE≥{policy['min_score_for_buy']:.0f} REGIME-AWARE qualified={len(qualified)} / TOP10={len(top10)}")
    if not top10:
        print("⏸ 承認済み条件を満たす候補なし → 本日はエントリなし(強制経路は廃止済み)")
        return [], scanned
    for rank, c in enumerate(top10, 1):
        c["selection_level"] = 1
        c["selection_mode"] = "normal"
        c["top10_rank"] = rank
    return top10, scanned

def _as_aware_jst(value):
    ts=value if isinstance(value,datetime) else app.pd.Timestamp(value).to_pydatetime()
    return ts.replace(tzinfo=app.TZ) if ts.tzinfo is None else ts.astimezone(app.TZ)

def close_positions_with_cooldown(state,now,policy):
    before={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};messages=_original_close(state,now,policy);after={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};cooldowns=state.setdefault("last_exit_by_ticker",{})
    for ticker in sorted(before-after):cooldowns[ticker]=app.pd.Timestamp(now).isoformat();print(f"⏳ 同一銘柄クールダウン開始: {ticker} {SAME_TICKER_COOLDOWN_MINUTES}分")
    return messages

def open_top1_only(state,policy,candidates,today):
    cooldowns=state.setdefault("last_exit_by_ticker",{});now=datetime.now(app.TZ);active={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};eligible=[];regime,_,_=_market_regime()
    for candidate in profit_priority(candidates)[:TOP10]:
        ticker=str(candidate.get("ticker","")).strip();direction=str(candidate.get("direction","BUY")).upper()
        if regime=="bearish" and direction!="SHORT":continue
        if regime=="bullish" and direction!="BUY":continue
        if not ticker or ticker in active:continue
        if int(state.get("trades_by_ticker_today",{}).get(ticker,0))>=MAX_TICKER_TRADES:continue
        if int(state.get("trades_today",0))>=MAX_DAILY_TRADES:break
        raw=cooldowns.get(ticker)
        if raw:
            try:remaining=(_as_aware_jst(raw)+timedelta(minutes=SAME_TICKER_COOLDOWN_MINUTES)-now).total_seconds()
            except Exception:remaining=0
            if remaining>0:print(f"⏸ 同一銘柄クールダウン中: {ticker} 残り約{int(remaining//60)+1}分");continue
            cooldowns.pop(ticker,None)
        eligible.append(candidate)
    if not eligible:print("⏸ 候補内に新規エントリ可能なTOP1なし");return []
    top1=eligible[0];old_max_total=app.MAX_TOTAL_TRADES_PER_DAY;old_max_ticker=app.MAX_TRADES_PER_TICKER_PER_DAY
    # 注意: app.TOP_N は上書きしない。budget=capital/TOP_N の計算がTOP_N=1だと
    # 毎回「資金全額」を予算にしてしまい、複数ポジション同時保有時に資金オーバーする
    # バグがあったため、常にapp.TOP_N(=同時保有上限と一致した分割数)を使う。
    try:
        app.MAX_TOTAL_TRADES_PER_DAY=MAX_DAILY_TRADES;app.MAX_TRADES_PER_TICKER_PER_DAY=MAX_TICKER_TRADES;opened=_original_open(state,policy,[top1],today)
    finally:
        app.MAX_TOTAL_TRADES_PER_DAY=old_max_total;app.MAX_TRADES_PER_TICKER_PER_DAY=old_max_ticker
    if opened:
        p=state["positions"][-1];p["allocation"]=1.0;p["selection_mode"]=top1.get("selection_mode","normal");p["selection_level"]=int(top1.get("selection_level",1));p["top10_rank"]=int(top1.get("top10_rank",1));p["market_regime"]=top1.get("market_regime",regime);p["regime_preferred"]=bool(top1.get("regime_preferred",False));p["profit_ev_pct"]=float(top1.get("profit_ev_pct",0.0));p["profit_priority"]=float(top1.get("profit_priority",0.0));p["feedback_weight"]=float(top1.get("feedback_weight",1.0));print(f"🏆 TOP→TOP1 ENTRY: {top1.get('direction','BUY')} {top1['ticker']} LEVEL={p['selection_level']} MODE={p['selection_mode']} REGIME={p['market_regime']} score={top1['score']:.1f} UP={top1['up_probability']:.1f}% DOWN={top1.get('down_probability',0):.1f}% FEEDBACK_W={p['feedback_weight']:.2f}")
    return opened

app.scan=scan_candidates_fixed
app.mark_and_close=close_positions_with_cooldown
app.open_positions=open_top1_only

if __name__=="__main__":app.main()
