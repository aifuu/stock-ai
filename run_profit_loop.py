#!/usr/bin/env python3
"""Unified Profit Loop: fixed approved policy -> TOP10 -> TOP1 paper trade.

Research/OOS gates remain separate. This runtime path is for paper execution only.
Both BUY and SHORT are evaluated. Market regime determines the direction:
Nikkei bullish -> BUY only, bearish -> SHORT only, neutral -> compare both.
"""
from datetime import datetime, timedelta, time as dtime
import json
import math
import os
import numpy as np
import pandas as pd

import profit_top10_paper as app
import daily_decision
import futures_trend

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

# ★追加(2026-10, A案): 2026-10-09の事象(daily_decisionはDOWNだったが日経レジームが
# 強気だったため、両トラックとも通常のレジーム判定(強気→買いのみ)に従って買い、
# 下落日に買って損失を出した)を受けたオーナー承認の方向転換。
# daily_decision.jsonの当日trendが'down'の日は、日経レジームに関わらず新規エントリーを
# 空売りのみに固定する(地合い判定より優先)。判断ファイルが当日分無し/壊れている場合は
# ネットワークに依存せず既存の地合いロジックへフォールバックする(クラッシュしない)。
def _down_day_decision():
    try:
        trend = daily_decision.todays_trend()
    except Exception as exc:
        print(f"⚠️ daily_decision読み込み失敗 → 既存の地合いロジックを使用: {exc}")
        return False
    if trend is None:
        print("⚠️ daily_decision未取得(当日分なし) → 既存の地合いロジックを使用")
        return False
    return trend == futures_trend.DOWN

def _direction_gate_label(down_day, regime):
    return "short_only_down_day" if down_day else f"regime_{regime}"

def profit_priority(candidates):
    """Regime gate: bearish means SHORT candidates only; bullish means BUY only.
    Neutral compares BUY/SHORT by expected value and score.

    ★変更(2026-10, A案): daily_decisionの当日trendが'down'の日は、この地合い判定の
    結果に関わらず新規エントリーを空売りのみに固定する(effective direction gate)。
    UP日の挙動(フィルター・スコアリングとも)は変更しない。
    """
    regime, kairi25, ret5 = _market_regime()
    down_day = _down_day_decision()
    direction_gate = _direction_gate_label(down_day, regime)
    feedback_weights = _load_feedback_weights()
    print(f"🌐 日経レジーム: {regime.upper()}" + (f"｜25MA乖離 {kairi25:+.2f}%｜5日騰落 {ret5:+.2f}%" if kairi25 is not None else ""))
    if down_day:
        print(f"📉 下落日: 新規は空売りのみ(地合い判定 {regime} より優先)")
    ranked = []
    for c in candidates:
        direction = str(c.get("direction", "BUY")).upper()
        if down_day:
            if direction != "SHORT":
                continue
        else:
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
        # ★追加(2026-10, A案監査用、末尾に追加): market_regime_raw=日経レジームの生値
        # (market_regimeと同じ値だが、down_day時にフィルターが地合いを上書きしたことが
        # わかるようdirection_gateと併せて残す)。direction_gate=実際に使われた方向制限。
        item["market_regime_raw"] = regime
        item["direction_gate"] = direction_gate
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

# ★追加(2026-11, TOP1建値バグ修正): yfinanceの5分足は実時刻から15-20分程度遅延
# するため、09:0x台のtickでdownload_5m()の最終バーが前営業日最終バー(例:
# 11/04 15:20 close)になっている場合があり、それをそのまま約定価格として記録
# すると実際には存在しない値段で約定したことになってしまう(2026-11-05の285A.T:
# 記録上10,640円で約定したことになっていたが、実際の初値は09:05 Open 9,820円、
# 当日の値動き全体が9,223-10,270円で10,640円は一度も付いていない)。
# daytrade_tp5000_sl8000_paper.pyのFill rule v2(decision_next_bar_open_v2)と
# 同じ原則で、TOP1も「判断(decision)」と「約定(fill)」を分離する:
#   1. open_top1_only() はTOP1を選ぶが実際には発注しない。選んだ候補を
#      state['pending_entry']として保存し、次に5分足の直後バーが実際に
#      出現したtickでfill_pending_entry()が約定させる(このモジュールのみの
#      変更。選定ロジック・スコアリング・方向ゲート・クールダウン・資金不足
#      スキップはすべて従来どおり、約定タイミングだけを変える)。
#   2. fill_pending_entry()は決定時刻をceilした次の5分足バーのOpenで約定する
#      (そのバーが欠けていても、当日のそれ以降最初のバーを使う。285A.Tの
#      11/05は09:00バー自体が存在せず最初のバーが09:05)。
FILL_METHOD_TOP1_V1 = "decision_next_bar_open_top1_v1"

def _ceil_bar_time(t):
    """The first 5-minute bar START time that is >= t: t itself if it
    already lands exactly on a 5-minute boundary, else the next one."""
    t = pd.Timestamp(t)
    floor = t.floor("5min")
    return floor if floor == t else floor + pd.Timedelta(minutes=5)

def close_positions_with_cooldown(state,now,policy):
    before={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};messages=_original_close(state,now,policy);after={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};cooldowns=state.setdefault("last_exit_by_ticker",{})
    for ticker in sorted(before-after):cooldowns[ticker]=app.pd.Timestamp(now).isoformat();print(f"⏳ 同一銘柄クールダウン開始: {ticker} {SAME_TICKER_COOLDOWN_MINUTES}分")
    return messages

def open_top1_only(state,policy,candidates,today):
    # ★変更(2026-11): ペンディング中の約定枠が1つあれば、それが埋まるかキャンセル
    # されるまで新しい判断はしない(1枠を保持したまま重複判断しない)。
    if state.get("pending_entry") is not None:
        print("⏳ ペンディング中のTOP1約定待ちのため新規判断をスキップ");return []
    cooldowns=state.setdefault("last_exit_by_ticker",{});now=datetime.now(app.TZ);active={str(p.get("ticker")) for p in state.get("positions",[]) if p.get("ticker")};eligible=[];regime,_,_=_market_regime();down_day=_down_day_decision()
    # ★追加: rank1がRANK1でも1単元(100株)すら買えない場合、open_positions側は
    # その1件を渡されてcontinueするだけでその日は何も開かない(フォールバック無し)。
    # ここでeligible構築時にopen_positionsと同じ「買えるか」判定を適用し、最初に
    # 買える候補をTOP1として選ぶ。capital/残り現金はこのループ中は変化しない
    # (このループでは実際にポジションを開かない)ため1回だけ計算する。
    capital=float(state.get("capital",app.INITIAL_CAPITAL));cash_remaining=capital-sum(float(p.get("invested_amount",0)) for p in state.get("positions",[]))
    skipped_unaffordable=[]
    for candidate in profit_priority(candidates)[:TOP10]:
        ticker=str(candidate.get("ticker","")).strip();direction=str(candidate.get("direction","BUY")).upper()
        if down_day:
            if direction!="SHORT":continue
        else:
            if regime=="bearish" and direction!="SHORT":continue
            if regime=="bullish" and direction!="BUY":continue
        if not ticker or ticker in active:continue
        if int(state.get("trades_by_ticker_today",{}).get(ticker,0))>=MAX_TICKER_TRADES:continue
        if int(state.get("trades_today",0))>=MAX_DAILY_TRADES:break
        raw=cooldowns.get(ticker)
        if raw:
            try:
                remaining=(_as_aware_jst(raw)+timedelta(minutes=SAME_TICKER_COOLDOWN_MINUTES)-now).total_seconds()
            except Exception as e:
                print(f"⚠️ 不正なクールダウン値を検出: {ticker} raw={raw!r} err={e} → 自己修復(新規クールダウン開始)")
                try:app.discord_send(f"⚠️ クールダウン値が不正のため自己修復しました: {ticker} raw={raw!r} err={e}")
                except Exception:pass
                cooldowns[ticker]=app.pd.Timestamp(now).isoformat()
                remaining=SAME_TICKER_COOLDOWN_MINUTES*60
            if remaining>0:print(f"⏸ 同一銘柄クールダウン中: {ticker} 残り約{int(remaining//60)+1}分");continue
            cooldowns.pop(ticker,None)
        price=float(candidate.get("price",0) or 0)
        if not(math.isfinite(price) and price>0 and app.LOT_SIZE*price<=cash_remaining):
            skipped_unaffordable.append(ticker)
            print(f"⏭ 資金不足のため見送り: {ticker} 100株={price*app.LOT_SIZE:,.0f}円 > 残り{cash_remaining:,.0f}円")
            continue
        eligible.append(candidate)
    if not eligible:print("⏸ 候補内に新規エントリ可能なTOP1なし");return []
    top1=eligible[0]
    # ★変更(2026-11): ここでは発注しない(decision)。約定(fill)はfill_pending_entry()が
    # 次に出現する5分足の直後バーのOpenで行う(モジュールdocstring参照)。
    now_naive=now.replace(tzinfo=None);fill_bar_time=_ceil_bar_time(now_naive);decision_price=float(top1.get("price",0) or 0)
    pending=dict(top1)
    pending.update(decision_time=now_naive.isoformat(),decision_date=today,fill_bar_time=fill_bar_time.isoformat(),decision_price=decision_price)
    if skipped_unaffordable:pending["skipped_unaffordable"]=",".join(skipped_unaffordable)
    state["pending_entry"]=pending
    print(f"⏳ TOP1 PENDING: {top1.get('direction','BUY')} {top1['ticker']} 判断={now_naive} 約定予定バー>={fill_bar_time}")
    try:app.discord_send(f"⏳ 判断(ペンディング)｜{top1['ticker']}｜{'買い' if top1.get('direction','BUY')=='BUY' else '空売り'}\n判断時刻 {now_naive.strftime('%H:%M:%S')}｜約定予定バー {fill_bar_time.strftime('%H:%M')}(Open約定)")
    except Exception as e:print(f"⚠️ discord通知失敗(ペンディング): {e}")
    return []

def fill_pending_entry(state,policy,now,entry_ok,buy_braked):
    """state['pending_entry']の約定を確認する。約定予定バー(fill_bar_time)以降で
    最初に実在する当日バーのOpenで約定させる(そのバー自体が欠けていてもよい。
    285A.T 2026-11-05は09:00バーが無く最初のバーが09:05)。前営業日のバーは
    絶対に使わない(本バグ修正の核心)。"""
    pending=state.get("pending_entry")
    if pending is None:return []
    today=now.strftime("%Y-%m-%d")
    ticker=str(pending.get("ticker","")).strip();direction=str(pending.get("direction","BUY")).upper()

    def _cancel(reason):
        state["pending_entry"]=None
        row={"ticker":ticker,"direction":direction,"decision_time":pending.get("decision_time"),
             "fill_bar_time":pending.get("fill_bar_time"),"reason":reason,
             "cancelled_at":now.replace(tzinfo=None).isoformat()}
        cancelled=state.setdefault("cancelled_entries",[]);cancelled.append(row)
        del cancelled[:-50]
        print(f"🚫 PENDING CANCELLED: {ticker} reason={reason}")
        try:app.discord_send(f"🚫 ペンディング取消｜{ticker}｜{'買い' if direction=='BUY' else '空売り'}\n理由 {reason}")
        except Exception as e:print(f"⚠️ discord通知失敗(取消): {e}")
        return []

    try:
        if pending.get("decision_date")!=today:
            return _cancel("CANCELLED_NO_FILL_NEW_DAY")
        if not entry_ok:
            return _cancel("CANCELLED_NO_FILL_ENTRY_BLOCKED")
        if direction=="BUY" and buy_braked:
            return _cancel("CANCELLED_NO_FILL_CRASH_BRAKE")

        fill_bar_time=pd.Timestamp(pending["fill_bar_time"]);decision_price=float(pending.get("decision_price",0) or 0)
        bars=app.download_5m(ticker);bar_ts=None
        if bars is not None and not bars.empty:
            today_bars=bars[bars.index.date==pd.Timestamp(today).date()]
            later=today_bars[today_bars.index>=fill_bar_time]
            if not later.empty:bar_ts=later.index[0]

        if bar_ts is None:
            if now.time()>=dtime(15,30):
                return _cancel("CANCELLED_NO_FILL_DATA_DELAY")
            return []

        fill_price=float(bars.loc[bar_ts]["Open"])
        if not(math.isfinite(fill_price) and fill_price>0):
            return []  # 価格がまだ不正/未確定 → 次tickで再確認(ペンディング維持)

        shift=fill_price-decision_price;candidate=dict(pending)
        candidate["price"]=fill_price;candidate["tp"]=float(pending["tp"])+shift;candidate["sl"]=float(pending["sl"])+shift

        old_max_total=app.MAX_TOTAL_TRADES_PER_DAY;old_max_ticker=app.MAX_TRADES_PER_TICKER_PER_DAY
        try:
            app.MAX_TOTAL_TRADES_PER_DAY=MAX_DAILY_TRADES;app.MAX_TRADES_PER_TICKER_PER_DAY=MAX_TICKER_TRADES
            opened=_original_open(state,policy,[candidate],today)
        finally:
            app.MAX_TOTAL_TRADES_PER_DAY=old_max_total;app.MAX_TRADES_PER_TICKER_PER_DAY=old_max_ticker

        if not opened:
            return _cancel("CANCELLED_NO_FILL_NOT_OPENED")

        p=state["positions"][-1]
        p["allocation"]=1.0;p["entry_time"]=bar_ts.strftime("%H:%M")
        p["fill_method"]=FILL_METHOD_TOP1_V1;p["decision_time"]=pending.get("decision_time")
        p["fill_bar_time"]=bar_ts.isoformat();p["decision_price"]=decision_price
        for key in ("selection_mode","selection_level","top10_rank","market_regime","regime_preferred",
                    "profit_ev_pct","profit_priority","feedback_weight","market_regime_raw","direction_gate"):
            if key in pending:p[key]=pending[key]
        if pending.get("skipped_unaffordable"):p["skipped_unaffordable"]=pending["skipped_unaffordable"]
        state["pending_entry"]=None
        print(f"🏆 TOP1 FILLED: {direction} {ticker} fill={fill_price:.1f} bar={bar_ts} decision={decision_price:.1f} delay_bars_ok")
        return opened
    except Exception as e:
        print(f"⚠️ fill_pending_entry失敗(ペンディング維持、取引停止せず継続): {e}")
        return []

app.scan=scan_candidates_fixed
app.mark_and_close=close_positions_with_cooldown
app.open_positions=open_top1_only
app.fill_pending_entry=fill_pending_entry

if __name__=="__main__":app.main()
