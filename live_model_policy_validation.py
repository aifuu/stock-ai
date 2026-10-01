"""live_model_policy_validation.py (研究専用・新規ファイル)

背景(CONTEXT): 現在承認済みのstrategy_policy.json/strategy_policy_up.json の
OOS PF 10〜30 は walk_forward.py のフォールドごとモデルで検証されたものであり、
本番(daily_model_retrain.py が学習する directional_model.pkl を
profit_top10_paper.scan() が使う)のライブモデル・パイプラインでは一度も
検証されていない。ライブモデル(2026-09-12「reference adoption」投入、
5トレードでOOS PF 0.218)は今も使われ続けており、本番retrainが新チャレンジャー
を採用しなかった(DD -30.3% > 上限30%)ため居座っている。

本モジュールは「ライブが実際に取引している組み合わせ(ライブ学習パイプライン
×scan()の選定ロジック×承認済みpolicy)に、そもそもエッジがあるのか」を
Walk-Forward・リークなしで検証する研究専用ツール。日足AI本番システム
(daily_directional_top1.py・daily_model_retrain.py・profit_top10_paper.py・
all_candidates_paper.py・futures_trend.py・strategy_policy*.json・
directional_model.pkl)は一切変更しない。state/history/policy/モデルへの
書き込みは行わない。research_model_freeze_approval.jsonも作成しない。

★再利用(import、変更しない):
  daily_directional_top1(trader): features/atr/directional_score/make_nikkei/
    TICKERS/FEATURES
  daily_model_retrain(dmr): build_ticker_frame/build_universe/
    flatten_training_rows/fit_rf/compute_pf_metrics/FEE_RATE_PCT
  profit_top10_paper(live_p10): _tp_sl/_reward_risk/SHORT_ENABLED/
    POLICY_FILE/POLICY_FILE_UP/POLICY_FILE_DOWN
  all_candidates_paper(acp): load_frozen_policy/choose_frozen_policy_file/
    FROZEN_POLICY_FILE/FROZEN_POLICY_FILE_UP
  futures_trend: historical_trend_series/UP/DOWN
  common: count_tse_trading_days

★モデル化しているもの/していないもの(重要、レポートにも明記):
  - モデル化: ライブのscan()と同一のok条件・(expected_value_pct,score)降順
    ランキング・'pool = cand or fallback'(その日ok条件通過が0件なら
    フィルター無視の参考順位にフォールバックする、というscan()自身の実挙動)、
    futures_trendによる日次のpolicy切替(UP→up policy、DOWN/その他→normal
    policy、どちらもliveの承認済みpolicyの内容を凍結コピーしたall_candidates_
    frozen_policy*.jsonで代用=署名検証を経由しない)、ATR×TP/SL、
    ポジション保有中のhold_days上限はエントリー日ではなく「その日の
    アクティブpolicy」を毎日参照する(profit_top10_paper.mark_and_closeと
    同じ、やや意外な実挙動)、往復手数料(dmr.FEE_RATE_PCT)。
  - モデル化していない(簡略化、TOP1バケットの数値を読む際に考慮すること):
    1日あたり銘柄別/全体の取引数上限(MAX_TRADES_PER_TICKER_PER_DAY/
    MAX_TOTAL_TRADES_PER_DAY)、paper_risk_policy のドローダウン連動
    ポジションサイズ縮小・日次損失上限・最大DD全停止。TOP1バケットは
    単純に「毎回その時点の全資産を1トレードに賭ける」複利モデル。
  - 価格精緻化(download_5mによる5分足リファイン)は行わない。過去日付の
    5分足はcloud/Actionsどちらからも取得不能なため、常に日足ベース。
    代わりに2つのエントリー規約(next_open/same_close)を両方計算して
    提示する。

entry conventions:
  same_close: 判定日Dのその日の終値でエントリー(研究トラック
    all_candidates_paper.pyと同じ規約)。
  next_open: D+1(翌営業日)の始値でエントリー(ライブの約09:00エントリーに
    最も近い規約)。TP/SLは判定日DのATRのまま、実際の約定価格(D+1始値)で
    再計算する(scan()が価格精緻化後にtp/slを再計算するのと同じ考え方)。
"""
import bisect
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import all_candidates_paper as acp
import daily_directional_top1 as trader
import daily_model_retrain as dmr
import futures_trend
import profit_top10_paper as live_p10
from common import count_tse_trading_days

TZ = ZoneInfo("Asia/Tokyo")
FEATURES = trader.FEATURES

ENTRY_CONVENTIONS = ("next_open", "same_close")
BUCKETS = ("TOP1", "ALL", "TOP3", "TOP5")
EQUAL_WEIGHT_CAPITAL = 1_000_000.0
TOP1_INITIAL_CAPITAL = 1_000_000.0

# ★HOLDOUT/embargo: dmr.main()と同じ「直近retrainの地平線パージ」方式
# (train_cutoff = oos_cutoff - HOLD_DAYS*2日)を、本検証でも使う policy が
# 2種類(normal hold_days=1 / up hold_days=3)あるため、より安全側の
# max(両方のhold_days)*2 を共通マージンとして使う。
_POLICY_CACHE = {}


def _load_policy_pair():
    """normal/upの凍結policyを1回だけ読み込み、キャッシュする。"""
    if not _POLICY_CACHE:
        _POLICY_CACHE["normal"] = acp.load_frozen_policy(acp.FROZEN_POLICY_FILE)
        _POLICY_CACHE["up"] = acp.load_frozen_policy(acp.FROZEN_POLICY_FILE_UP)
    return _POLICY_CACHE["normal"], _POLICY_CACHE["up"]


def purge_margin_days():
    normal, up = _load_policy_pair()
    return max(int(normal["hold_days"]), int(up["hold_days"])) * 2


def live_policy_file_for_trend(trend):
    """profit_top10_paper.select_policy_file()と同じ選択式(futures_trend判定
    結果→どのliveファイル名を選ぶか)。ネットワーク呼び出し(detect_futures_trend)
    は行わず、既に計算済みのtrend文字列を受け取る。"""
    candidate = live_p10.POLICY_FILE_DOWN if trend == futures_trend.DOWN else live_p10.POLICY_FILE_UP
    if os.path.exists(candidate):
        return candidate
    return live_p10.POLICY_FILE


def frozen_policy_for_trend(trend):
    """trend('up'/'down')から、その日使う凍結policy(dict)とファイル名を返す。"""
    normal, up = _load_policy_pair()
    live_name = live_policy_file_for_trend(trend)
    frozen_file = acp.choose_frozen_policy_file(live_name)
    policy = up if frozen_file == acp.FROZEN_POLICY_FILE_UP else normal
    return frozen_file, policy


def build_trend_series(start, end):
    """指定期間の日次futures_trend系列(up/down)。futures_trend.historical_trend_series
    をそのまま再利用する(このファイルでは一切トレンド判定ロジックを書き直さない)。"""
    return futures_trend.historical_trend_series(start, end)


def trend_for_date(trend_series, date, default=futures_trend.UP):
    if trend_series is None or trend_series.empty:
        return default
    ts = pd.Timestamp(date).normalize()
    if ts in trend_series.index:
        return trend_series.loc[ts, "trend"]
    prior = trend_series.index[trend_series.index <= ts]
    if len(prior) == 0:
        return default
    return trend_series.loc[prior[-1], "trend"]


# =====================================================================
# scan()のok条件・ランキングの再現(過去日付版、価格精緻化なし)
# =====================================================================

def build_candidate_pool_for_date(date, ticker_frames, nikkei_ff, model, policy, short_enabled=None):
    """profit_top10_paper.scan()のok条件・(expected_value_pct,score)降順
    ランキング・'pool = cand or fallback' フォールバックを、既に構築済みの
    ticker_frames(dmr.build_ticker_frame出力)だけを使って過去日付dateに
    ついて再現する。download_5mによる価格精緻化は行わない(モジュール
    docstring参照)。

    戻り値: (pool, cand, fallback) の3リスト。各候補dictは
    ticker/direction/score/up_probability/down_probability/flat_probability/
    price/tp/sl/expected_value_pct/data_dateを持つ。
    """
    if short_enabled is None:
        short_enabled = live_p10.SHORT_ENABLED
    cols = list(getattr(model, "feature_names_in_", []))
    rows, tickers_order = [], []
    for ticker, xf in ticker_frames.items():
        if date not in xf.index:
            continue
        row = xf.loc[date]
        if pd.isna(row[cols]).any():
            continue
        atr_abs = float(row.get("atr_abs", np.nan))
        if not np.isfinite(atr_abs) or atr_abs <= 0:
            continue
        rows.append(row)
        tickers_order.append(ticker)
    if not rows:
        return [], [], []

    batch = pd.DataFrame(rows, index=tickers_order)
    probs = model.predict_proba(batch[cols])
    classes = list(model.classes_)
    down_col, up_col, flat_col = classes.index(0), classes.index(2), classes.index(1)

    nikkei_up = True
    if nikkei_ff is not None and date in nikkei_ff.index:
        nrow = nikkei_ff.loc[date]
        nikkei_up = bool(nrow.get("nikkei_uptrend", True)) if not pd.isna(nrow.get("nikkei_uptrend")) else True
    nikkei_filter_on = bool(policy.get("nikkei_filter"))

    cand, fallback = [], []
    data_date_str = str(pd.Timestamp(date).date())
    for i, ticker in enumerate(tickers_order):
        row = batch.iloc[i]
        down = float(probs[i, down_col]) * 100.0
        up = float(probs[i, up_col]) * 100.0
        flat = float(probs[i, flat_col]) * 100.0
        long_s, short_s = trader.directional_score(row, up / 100.0, down / 100.0)
        price = float(row["Close"])
        atr_abs = float(row["atr_abs"])
        base = {
            "ticker": ticker, "up_probability": up, "down_probability": down,
            "flat_probability": flat, "data_date": data_date_str,
        }
        for direction, score in (("BUY", float(long_s)), ("SHORT", float(short_s))):
            tp, sl = live_p10._tp_sl(price, direction, atr_abs, policy)
            item = dict(base)
            item.update(direction=direction, score=score, price=price, tp=tp, sl=sl, atr_abs=atr_abs)
            reward_pct, risk_pct = live_p10._reward_risk(price, tp, sl, direction)
            win_prob, loss_prob = (up / 100.0, down / 100.0) if direction == "BUY" else (down / 100.0, up / 100.0)
            item["expected_value_pct"] = win_prob * reward_pct - loss_prob * max(risk_pct, 0.0)
            fallback.append(item)
            if direction == "BUY":
                ok = up >= policy["up_threshold"] and up > down and flat < 50 and score >= policy["min_score_for_buy"]
            else:
                ok = (short_enabled and down >= policy["up_threshold"] and down > up
                      and flat < 50 and score >= policy["min_score_for_buy"])
            if ok and nikkei_filter_on and direction == "BUY" and not nikkei_up:
                ok = False
            if ok and nikkei_filter_on and direction == "SHORT" and nikkei_up:
                ok = False
            if ok:
                cand.append(item)
    cand.sort(key=lambda z: (z["expected_value_pct"], z["score"]), reverse=True)
    fallback.sort(key=lambda z: (z["expected_value_pct"], z["score"]), reverse=True)
    pool = cand if cand else fallback
    return pool, cand, fallback


# =====================================================================
# エントリー規約の適用 + 決済シミュレーション(日足ベース)
# =====================================================================

def _next_trading_date(all_dates_sorted, date):
    idx = np.searchsorted(all_dates_sorted, pd.Timestamp(date), side="right")
    if idx >= len(all_dates_sorted):
        return None
    return all_dates_sorted[idx]


def resolve_entry(candidate, ticker_frames, decision_date, all_dates_sorted, convention, policy):
    """convention('next_open'/'same_close')に応じて、実際のエントリー日・
    エントリー価格・(必要ならTP/SL再計算)を決める。next_openでエントリー
    できる翌営業日バーが存在しない場合はNoneを返す(その日はエントリー不可)。
    """
    xf = ticker_frames.get(candidate["ticker"])
    if xf is None:
        return None
    direction = candidate["direction"]
    atr_abs = float(candidate["atr_abs"])

    if convention == "same_close":
        if decision_date not in xf.index:
            return None
        entry_date = decision_date
        entry_price = float(xf.loc[decision_date, "Close"])
        tp, sl = candidate["tp"], candidate["sl"]
        include_entry_bar = False
    else:  # next_open
        nxt = _next_trading_date(all_dates_sorted, decision_date)
        if nxt is None or nxt not in xf.index:
            return None
        entry_date = nxt
        entry_price = float(xf.loc[nxt, "Open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None
        tp, sl = live_p10._tp_sl(entry_price, direction, atr_abs, policy)
        include_entry_bar = True

    return {
        "entry_date": entry_date, "entry_price": entry_price, "tp": tp, "sl": sl,
        "include_entry_bar": include_entry_bar, "direction": direction,
    }


def simulate_trade_exit(xf, entry_date, entry_price, tp, sl, direction, hold_days,
                         fee_rate_pct, include_entry_bar):
    """日足High/Low/Closeだけで決済判定する(daily_model_retrain.simulate_oos_top1
    と同じSL_BOTH/TP/SL/TIME優先順位・日足ロールバック方式)。xfの末尾まで
    決済条件に到達しなければFORCED_EOS(最終日の終値で強制決済)として返す。
    """
    idx = xf.index
    pos = idx.get_indexer([entry_date])[0]
    if pos < 0:
        return None
    start = pos if include_entry_bar else pos + 1
    days_held = 0
    for i in range(start, len(idx)):
        bar = xf.iloc[i]
        high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        days_held += 1
        exit_price = exit_reason = None
        if direction == "BUY":
            if low <= sl and high >= tp:
                exit_price, exit_reason = sl, "SL_BOTH"
            elif high >= tp:
                exit_price, exit_reason = tp, "TP"
            elif low <= sl:
                exit_price, exit_reason = sl, "SL"
        else:
            if high >= sl and low <= tp:
                exit_price, exit_reason = sl, "SL_BOTH"
            elif low <= tp:
                exit_price, exit_reason = tp, "TP"
            elif high >= sl:
                exit_price, exit_reason = sl, "SL"
        if exit_price is None and days_held >= hold_days:
            exit_price, exit_reason = close, "TIME"
        if exit_price is not None:
            ret = ((exit_price / entry_price - 1.0) * 100.0 if direction == "BUY"
                   else (entry_price / exit_price - 1.0) * 100.0)
            ret -= fee_rate_pct
            return {"exit_date": idx[i], "exit_price": exit_price, "reason": exit_reason,
                    "return_pct": ret, "days_held": days_held}
    if len(idx) > start:
        close = float(xf.iloc[-1]["Close"])
        ret = ((close / entry_price - 1.0) * 100.0 if direction == "BUY"
               else (entry_price / close - 1.0) * 100.0)
        ret -= fee_rate_pct
        return {"exit_date": idx[-1], "exit_price": close, "reason": "FORCED_EOS",
                "return_pct": ret, "days_held": days_held}
    return None


# =====================================================================
# TOP1: 複利・同時1ポジションのみ(ライブの実際の挙動に最も近い)
# =====================================================================

def simulate_top1_track(oos_dates, ticker_frames, nikkei_ff, model, trend_series, convention,
                         all_dates_sorted, fee_rate_pct=None):
    if fee_rate_pct is None:
        fee_rate_pct = dmr.FEE_RATE_PCT
    trades = []
    i = 0
    dates = list(oos_dates)
    while i < len(dates):
        date = dates[i]
        trend = trend_for_date(trend_series, date)
        frozen_file, policy = frozen_policy_for_trend(trend)
        pool, cand, _ = build_candidate_pool_for_date(date, ticker_frames, nikkei_ff, model, policy)
        if not pool:
            i += 1
            continue
        top = pool[0]
        from_fallback = len(cand) == 0
        entry = resolve_entry(top, ticker_frames, date, all_dates_sorted, convention, policy)
        if entry is None:
            i += 1
            continue
        xf = ticker_frames[top["ticker"]]
        # ★ポジション保有中のhold_days上限は、エントリー日ではなく
        # 「エントリー時点でのpolicy」を使う(profit_top10_paper.mark_and_closeは
        # 毎回その日のpolicyを再読込するが、本検証ではトレード単位の
        # 決済シミュレーションを一括で行うため、エントリー日のpolicyで固定する
        # 近似を取る。normal=1営業日/up=3営業日と短いため差は小さい)。
        hold_days = int(policy["hold_days"])
        result = simulate_trade_exit(
            xf, entry["entry_date"], entry["entry_price"], entry["tp"], entry["sl"],
            entry["direction"], hold_days, fee_rate_pct, entry["include_entry_bar"],
        )
        if result is None:
            i += 1
            continue
        held = count_tse_trading_days(entry["entry_date"], result["exit_date"])
        trades.append({
            "decision_date": date, "entry_date": entry["entry_date"], "exit_date": result["exit_date"],
            "ticker": top["ticker"], "direction": top["direction"], "return_pct": result["return_pct"],
            "reason": result["reason"], "held_trading_days": held, "policy_file": frozen_file,
            "up_probability": top["up_probability"], "down_probability": top["down_probability"],
            "score": top["score"], "expected_value_pct": top["expected_value_pct"],
            "from_fallback": from_fallback,
        })
        nxt = _next_trading_date(all_dates_sorted, result["exit_date"])
        if nxt is None:
            break
        i = bisect.bisect_left(dates, nxt)
    return trades


def equity_curve_from_trades(trades, initial_capital=TOP1_INITIAL_CAPITAL):
    capital = initial_capital
    peak = initial_capital
    max_dd = 0.0
    curve = []
    for t in sorted(trades, key=lambda z: z["exit_date"]):
        capital *= (1.0 + t["return_pct"] / 100.0)
        peak = max(peak, capital)
        dd = (capital / peak - 1.0) * 100.0 if peak else 0.0
        max_dd = min(max_dd, dd)
        curve.append({"exit_date": t["exit_date"], "capital": capital, "dd_pct": dd})
    return curve, capital, max_dd


# =====================================================================
# ALL/TOP3/TOP5: 日次コホート等金額(複利なし、all_candidates_paper.pyの
# 集計方式に合わせる)
# =====================================================================

def simulate_equal_weight_buckets(oos_dates, ticker_frames, nikkei_ff, model, trend_series, convention,
                                   all_dates_sorted, fee_rate_pct=None):
    """各日、その日のpool(scan()と同じ'cand or fallback')からTOP1/TOP3/TOP5/ALLを
    取り、それぞれ独立した当日コホート予算(EQUAL_WEIGHT_CAPITAL / 候補数)で
    均等配分した場合のトレード一覧を返す(all_candidates_paper.compute_daily_summary
    と同じ思想。複利なし=日ごとに独立した仮想資金)。
    """
    if fee_rate_pct is None:
        fee_rate_pct = dmr.FEE_RATE_PCT
    per_bucket_trades = {b: [] for b in BUCKETS}
    for date in oos_dates:
        trend = trend_for_date(trend_series, date)
        _, policy = frozen_policy_for_trend(trend)
        pool, _, _ = build_candidate_pool_for_date(date, ticker_frames, nikkei_ff, model, policy)
        if not pool:
            continue
        hold_days = int(policy["hold_days"])
        resolved = []
        for rank, c in enumerate(pool, start=1):
            entry = resolve_entry(c, ticker_frames, date, all_dates_sorted, convention, policy)
            if entry is None:
                continue
            xf = ticker_frames[c["ticker"]]
            result = simulate_trade_exit(
                xf, entry["entry_date"], entry["entry_price"], entry["tp"], entry["sl"],
                c["direction"], hold_days, fee_rate_pct, entry["include_entry_bar"],
            )
            if result is None:
                continue
            resolved.append({
                "decision_date": date, "rank": rank, "ticker": c["ticker"], "direction": c["direction"],
                "entry_date": entry["entry_date"], "exit_date": result["exit_date"],
                "return_pct": result["return_pct"], "reason": result["reason"],
            })
        if not resolved:
            continue
        for bucket in BUCKETS:
            n = 1 if bucket == "TOP1" else (int(bucket[3:]) if bucket.startswith("TOP") else None)
            members = resolved if n is None else [r for r in resolved if r["rank"] <= n]
            if not members:
                continue
            weight = EQUAL_WEIGHT_CAPITAL / len(members)
            for m in members:
                row = dict(m)
                row["weight_jpy"] = weight
                row["pnl_jpy"] = weight * m["return_pct"] / 100.0
                per_bucket_trades[bucket].append(row)
    return per_bucket_trades


# =====================================================================
# 指標計算
# =====================================================================

def compute_full_metrics(trades, return_key="return_pct", date_key="exit_date"):
    """dmr.compute_pf_metrics(trades,pf,win_rate,max_dd_pct)を拡張し、
    avg_return/月次平均リターン/プラス月比率/最長連敗も計算する。
    trades: [{return_key: float, date_key: date-like}, ...]
    """
    if not trades:
        return {
            "trades": 0, "pf": 0.0, "win_rate": 0.0, "avg_return_pct": 0.0,
            "max_dd_pct": 0.0, "avg_month_return_pct": 0.0, "pct_months_positive": 0.0,
            "longest_losing_streak": 0,
        }
    df = pd.DataFrame(trades).sort_values(date_key)
    df[date_key] = pd.to_datetime(df[date_key])
    gross_profit = float(df.loc[df[return_key] > 0, return_key].sum())
    gross_loss = float(-df.loc[df[return_key] < 0, return_key].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = float((df[return_key] > 0).mean() * 100.0)
    avg_return = float(df[return_key].mean())

    capital, peak, max_dd = 1.0, 1.0, 0.0
    for r in df[return_key]:
        capital *= (1.0 + r / 100.0)
        peak = max(peak, capital)
        dd = (capital / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)

    longest_streak = cur_streak = 0
    for r in df[return_key]:
        if r < 0:
            cur_streak += 1
            longest_streak = max(longest_streak, cur_streak)
        else:
            cur_streak = 0

    monthly = df.assign(month=df[date_key].dt.to_period("M")).groupby("month")[return_key].sum()
    avg_month_return = float(monthly.mean()) if len(monthly) else 0.0
    pct_months_positive = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0

    return {
        "trades": int(len(df)), "pf": pf, "win_rate": win_rate, "avg_return_pct": avg_return,
        "max_dd_pct": max_dd, "avg_month_return_pct": avg_month_return,
        "pct_months_positive": pct_months_positive, "longest_losing_streak": longest_streak,
    }


def compute_equal_weight_metrics(bucket_trades):
    """simulate_equal_weight_buckets()が返すpnl_jpy付きトレードから、
    avg_return_pct(=各トレードのreturn_pct平均、均等配分なので単純平均でよい)
    を含む指標を作る。pfはpnl_jpyベース(総利益/総損失)。"""
    if not bucket_trades:
        return {
            "trades": 0, "pf": 0.0, "win_rate": 0.0, "avg_return_pct": 0.0,
            "total_pnl_jpy": 0.0, "avg_month_pnl_jpy": 0.0, "pct_months_positive": 0.0,
        }
    df = pd.DataFrame(bucket_trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    gross_profit = float(df.loc[df["pnl_jpy"] > 0, "pnl_jpy"].sum())
    gross_loss = float(-df.loc[df["pnl_jpy"] < 0, "pnl_jpy"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = float((df["return_pct"] > 0).mean() * 100.0)
    avg_return = float(df["return_pct"].mean())
    total_pnl = float(df["pnl_jpy"].sum())
    monthly = df.assign(month=df["exit_date"].dt.to_period("M")).groupby("month")["pnl_jpy"].sum()
    avg_month_pnl = float(monthly.mean()) if len(monthly) else 0.0
    pct_months_positive = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0
    return {
        "trades": int(len(df)), "pf": pf, "win_rate": win_rate, "avg_return_pct": avg_return,
        "total_pnl_jpy": total_pnl, "avg_month_pnl_jpy": avg_month_pnl,
        "pct_months_positive": pct_months_positive,
    }


# =====================================================================
# フォールド構築
# =====================================================================

def usable_dates(ticker_frames, min_coverage_tickers=50):
    """全銘柄のdateの和集合のうち、FEATURES全列が非NaNな銘柄数が
    min_coverage_tickers以上ある日だけを「学習・検証に使える日」とする
    (252日ローリング特徴量のウォームアップ明け以降に相当)。"""
    all_dates = sorted(set().union(*[set(x.index) for x in ticker_frames.values()]))
    coverage = pd.Series(0, index=pd.DatetimeIndex(all_dates))
    for x in ticker_frames.values():
        valid = x[FEATURES].notna().all(axis=1)
        coverage = coverage.add(valid.astype(int).reindex(coverage.index, fill_value=0), fill_value=0)
    good = coverage[coverage >= min_coverage_tickers].index
    return pd.DatetimeIndex(sorted(good)), pd.DatetimeIndex(all_dates)


def build_folds(dates, n_folds=4, recent_days=60):
    """dates(使用可能日、昇順)をn_folds個のほぼ等分割OOS区間に分け、
    さらに「直近recent_days営業日」フォールドを追加する(CONTEXT要求:
    >=4フォールド + 直近60営業日)。各フォールドのtrain_cutoffは
    (フォールド開始日 - purge_margin_days()日)。"""
    dates = pd.DatetimeIndex(sorted(dates))
    n = len(dates)
    if n < n_folds * 10:
        raise RuntimeError(f"使用可能日数が少なすぎます({n}日)。フォールド分割不能")
    margin = pd.Timedelta(days=purge_margin_days())
    chunk = n // n_folds
    folds = []
    for k in range(n_folds):
        start_i = k * chunk
        end_i = (k + 1) * chunk - 1 if k < n_folds - 1 else n - 1
        oos_start, oos_end = dates[start_i], dates[end_i]
        folds.append({
            "name": f"fold{k + 1}", "oos_start": oos_start, "oos_end": oos_end,
            "train_cutoff": oos_start - margin,
        })
    recent_start = dates[-recent_days] if n >= recent_days else dates[0]
    folds.append({
        "name": "recent60", "oos_start": recent_start, "oos_end": dates[-1],
        "train_cutoff": recent_start - margin,
    })
    return folds


def fold_oos_dates(dates, fold):
    return [d for d in dates if fold["oos_start"] <= d <= fold["oos_end"]]


# =====================================================================
# Fidelity checks (BUILD 2): 現行directional_model.pklを使った再現性確認
# =====================================================================

def fidelity_check_live_top1(ticker_frames, nikkei_ff, trend_series, target_date, model=None,
                              expected_ticker="5019.T", expected_direction="BUY"):
    """現行directional_model.pkl(ticker_framesに既に含まれる実データ)で、
    target_date時点のscan()出口のTOP1を再現できるか確認する。
    既定の期待値: 2026-09-24時点で 5019.T BUY (CONTEXT記載)。
    """
    model = model or trader.load_model()
    target = pd.Timestamp(target_date).normalize()
    result = {"target_date": str(target.date()), "model_available": model is not None}
    if model is None:
        result["status"] = "SKIP: directional_model.pkl が読み込めません"
        return result
    trend = trend_for_date(trend_series, target)
    frozen_file, policy = frozen_policy_for_trend(trend)
    pool, cand, fallback = build_candidate_pool_for_date(target, ticker_frames, nikkei_ff, model, policy)
    top = pool[0] if pool else None
    result.update({
        "trend": trend, "frozen_policy_file": frozen_file,
        "pool_size": len(pool), "cand_size": len(cand), "fallback_only": len(cand) == 0,
        "top1_ticker": top["ticker"] if top else None,
        "top1_direction": top["direction"] if top else None,
        "top1_score": top["score"] if top else None,
    })
    matched = bool(top) and top["ticker"] == expected_ticker and top["direction"] == expected_direction
    result["expected"] = {"ticker": expected_ticker, "direction": expected_direction}
    result["matched"] = matched
    if not matched and top:
        rank_of_expected = next(
            (i + 1 for i, c in enumerate(pool) if c["ticker"] == expected_ticker and c["direction"] == expected_direction),
            None,
        )
        result["expected_rank_in_pool"] = rank_of_expected
    return result


def fidelity_check_research_track(ticker_frames, nikkei_ff, trend_series, target_date, recorded_df, model=None):
    """all_candidates_paper.pyの research track が target_date に記録した候補
    (recorded_df、all_candidates_YYYY-MM.csv.gzの該当日分)と、本検証の
    build_candidate_pool_for_date()が同じ日・同じ凍結policyで作る候補集合の
    一致率を見る。recorded_dfは少なくとも'ticker','direction'列を持つこと。
    """
    model = model or trader.load_model()
    target = pd.Timestamp(target_date).normalize()
    result = {"target_date": str(target.date()), "model_available": model is not None}
    if model is None:
        result["status"] = "SKIP: directional_model.pkl が読み込めません"
        return result
    trend = trend_for_date(trend_series, target)
    frozen_file, policy = frozen_policy_for_trend(trend)
    pool, cand, fallback = build_candidate_pool_for_date(target, ticker_frames, nikkei_ff, model, policy)
    generated = {(c["ticker"], c["direction"]) for c in (cand if cand else pool)}
    recorded = set(zip(recorded_df["ticker"].astype(str), recorded_df["direction"].astype(str)))
    intersect = generated & recorded
    match_rate_recorded = (len(intersect) / len(recorded)) if recorded else None
    match_rate_generated = (len(intersect) / len(generated)) if generated else None
    result.update({
        "trend": trend, "frozen_policy_file": frozen_file,
        "recorded_count": len(recorded), "generated_count": len(generated),
        "intersection": len(intersect),
        "match_rate_vs_recorded": match_rate_recorded,
        "match_rate_vs_generated": match_rate_generated,
        "recorded_only": sorted(recorded - generated)[:20],
        "generated_only": sorted(generated - recorded)[:20],
    })
    return result


def diagnostic_vs_walk_forward(ticker_frames, nikkei_ff, trend_series, model, wf_df, max_dates=None):
    """walk_forward_all_candidates.csv.gz(walk_forward.pyのフォールドモデル産出)
    と、本検証のライブ学習パイプライン・モデルの予測を、重なる日付で比較する。
    wf_dfの列名は実行時に判明するため、up確率らしき列・score列・ticker/date列を
    緩く推定する(見つからない指標はNoneのまま返す)。
    """
    result = {"overlapping_dates": 0, "rows_compared": 0, "columns_found": {}}
    cols = {c.lower(): c for c in wf_df.columns}
    date_col = next((cols[c] for c in cols if c in ("date", "data_date")), None)
    ticker_col = next((cols[c] for c in cols if c in ("ticker", "code", "symbol")), None)
    up_col = next((cols[c] for c in cols if "up" in c and "prob" in c), None)
    score_col = next((cols[c] for c in cols if c == "score" or c.endswith("_score")), None)
    result["columns_found"] = {"date": date_col, "ticker": ticker_col, "up_probability": up_col, "score": score_col}
    if date_col is None or ticker_col is None:
        result["status"] = "SKIP: date/ticker列が見つかりません"
        return result

    wf = wf_df.copy()
    wf[date_col] = pd.to_datetime(wf[date_col], errors="coerce").dt.normalize()
    wf = wf.dropna(subset=[date_col])
    wf_dates = sorted(wf[date_col].unique())
    if max_dates:
        wf_dates = wf_dates[-max_dates:]
    result["overlapping_dates"] = 0

    live_up, live_score, wf_up, wf_score = [], [], [], []
    top1_agree, top1_total = 0, 0
    for d in wf_dates:
        d = pd.Timestamp(d)
        pool, cand, fallback = build_candidate_pool_for_date(d, ticker_frames, nikkei_ff, model,
                                                               frozen_policy_for_trend(trend_for_date(trend_series, d))[1])
        if not pool:
            continue
        result["overlapping_dates"] += 1
        live_by_ticker = {c["ticker"]: c for c in pool if c["direction"] == "BUY"}
        day_wf = wf[wf[date_col] == d]
        if day_wf.empty:
            continue
        for _, r in day_wf.iterrows():
            t = str(r[ticker_col])
            if t not in live_by_ticker:
                continue
            if up_col is not None and pd.notna(r.get(up_col)):
                live_up.append(live_by_ticker[t]["up_probability"])
                wf_up.append(float(r[up_col]))
            if score_col is not None and pd.notna(r.get(score_col)):
                live_score.append(live_by_ticker[t]["score"])
                wf_score.append(float(r[score_col]))
        if score_col is not None and not day_wf.empty:
            wf_top1 = day_wf.sort_values(score_col, ascending=False).iloc[0][ticker_col]
            live_top1 = pool[0]["ticker"]
            top1_total += 1
            if str(wf_top1) == str(live_top1):
                top1_agree += 1

    result["rows_compared"] = len(live_up) or len(live_score)
    if len(live_up) >= 5:
        result["spearman_up_probability"] = float(pd.Series(live_up).corr(pd.Series(wf_up), method="spearman"))
    if len(live_score) >= 5:
        result["spearman_score"] = float(pd.Series(live_score).corr(pd.Series(wf_score), method="spearman"))
    result["top1_overlap_rate"] = (top1_agree / top1_total) if top1_total else None
    result["top1_days_compared"] = top1_total
    return result


def download_release_asset(repo, tag, asset_name, dest_path):
    """gh CLI経由でGitHub Releaseアセットを取得する(GitHub Actionsランナー内
    でのみ想定。失敗時は(False, stderr)を返し、呼び出し側はfidelity/diagnostic
    チェックをSKIP扱いにする)。
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["gh", "release", "download", tag, "--repo", repo, "-p", asset_name,
             "-D", os.path.dirname(dest_path) or ".", "--clobber"],
            capture_output=True, text=True, check=False, timeout=120,
        )
        if proc.returncode != 0:
            return False, proc.stderr.strip()
        return True, None
    except Exception as e:
        return False, str(e)


# =====================================================================
# main orchestration(GitHub Actions内でのみ実行、Yahoo Financeアクセス必須)
# =====================================================================

def run_validation(tickers=None, n_folds=4, recent_days=60, conventions=ENTRY_CONVENTIONS,
                    min_coverage_tickers=50, progress=print):
    tickers = tickers or trader.TICKERS
    progress(f"=== live_model_policy_validation: universe={len(tickers)}銘柄 ===")
    ticker_frames, nikkei = dmr.build_universe(tickers)
    all_dates_u = sorted(set().union(*[set(x.index) for x in ticker_frames.values()]))
    nikkei_ff = nikkei.reindex(all_dates_u).ffill()
    all_dates_sorted = pd.DatetimeIndex(all_dates_u)

    good_dates, _ = usable_dates(ticker_frames, min_coverage_tickers=min_coverage_tickers)
    if len(good_dates) == 0:
        raise RuntimeError(
            "使用可能日(FEATURES全列が揃った銘柄がmin_coverage_tickers件以上ある日)が"
            "1日もありません。取得期間が短すぎる(252日ローリング特徴量のウォームアップに"
            "満たない)か、min_coverage_tickersが銘柄数に対して高すぎる可能性があります。"
        )
    progress(f"使用可能日(特徴量ウォームアップ明け): {len(good_dates)}日"
             f"（{good_dates[0].date()}〜{good_dates[-1].date()}）")

    folds = build_folds(good_dates, n_folds=n_folds, recent_days=recent_days)
    trend_series = build_trend_series(good_dates[0] - pd.Timedelta(days=30), good_dates[-1] + pd.Timedelta(days=5))

    results = {"folds": [], "overall": {}, "meta": {
        "universe_size": len(tickers), "usable_dates": len(good_dates),
        "usable_date_range": [str(good_dates[0].date()), str(good_dates[-1].date())],
        "purge_margin_days": purge_margin_days(),
    }}

    overall_trades = {c: {"TOP1": []} for c in conventions}
    overall_bucket_trades = {c: {b: [] for b in BUCKETS} for c in conventions}

    for fold in folds:
        progress(f"--- {fold['name']}: OOS {fold['oos_start'].date()}〜{fold['oos_end'].date()}"
                 f" (train_cutoff={fold['train_cutoff'].date()}) ---")
        fit_rows = dmr.flatten_training_rows(ticker_frames, before_date=fold["train_cutoff"])
        progress(f"学習行数: {len(fit_rows):,}")
        if len(fit_rows) < 500:
            progress(f"⚠ {fold['name']}: 学習データ不足のためスキップ")
            continue
        model = dmr.fit_rf(fit_rows)
        oos_dates = fold_oos_dates(good_dates, fold)

        fold_result = {"name": fold["name"], "oos_start": str(fold["oos_start"].date()),
                        "oos_end": str(fold["oos_end"].date()), "oos_days": len(oos_dates),
                        "train_rows": len(fit_rows), "conventions": {}}

        for conv in conventions:
            top1_trades = simulate_top1_track(
                oos_dates, ticker_frames, nikkei_ff, model, trend_series, conv, all_dates_sorted,
            )
            bucket_trades = simulate_equal_weight_buckets(
                oos_dates, ticker_frames, nikkei_ff, model, trend_series, conv, all_dates_sorted,
            )
            overall_trades[conv]["TOP1"].extend(top1_trades)
            for b in BUCKETS:
                overall_bucket_trades[conv][b].extend(bucket_trades[b])

            # ★注意: BUCKETS=("TOP1","ALL","TOP3","TOP5")はsimulate_equal_weight_buckets()
            # (all_candidates_paper.pyの等金額・複利なし日次コホート方式)のバケット名で、
            # 「TOP1」という名前を複利・同時1ポジションのcompute_full_metrics(top1_trades)
            # (CONTEXT要求のTOP1=本当のliveに近い方)とそのまま同じキーに入れると後者を
            # 上書きしてしまう。等金額側だけEW_プレフィックスを付けて区別する。
            conv_result = {"TOP1": compute_full_metrics(top1_trades)}
            for b in BUCKETS:
                conv_result[f"EW_{b}"] = compute_equal_weight_metrics(bucket_trades[b])
            fold_result["conventions"][conv] = conv_result
            progress(f"  [{fold['name']}/{conv}] TOP1: trades={conv_result['TOP1']['trades']} "
                     f"pf={conv_result['TOP1']['pf']:.2f} win={conv_result['TOP1']['win_rate']:.1f}% "
                     f"maxDD={conv_result['TOP1']['max_dd_pct']:.1f}%")

        results["folds"].append(fold_result)

    for conv in conventions:
        overall_conv = {"TOP1": compute_full_metrics(overall_trades[conv]["TOP1"])}
        for b in BUCKETS:
            overall_conv[f"EW_{b}"] = compute_equal_weight_metrics(overall_bucket_trades[conv][b])
        results["overall"][conv] = overall_conv

    return results, ticker_frames, nikkei_ff, trend_series, all_dates_sorted


# =====================================================================
# レポート表の整形(ログ・$GITHUB_STEP_SUMMARY共通)
# =====================================================================

def _fmt_pf(pf):
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def format_metrics_table(title, per_convention):
    lines = [f"### {title}", "", "| convention | bucket | trades | PF | win% | avg_ret% | avg_month_ret | %months+ | maxDD% |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for conv, buckets in per_convention.items():
        for bucket, m in buckets.items():
            avg_month = m.get("avg_month_return_pct", m.get("avg_month_pnl_jpy"))
            lines.append(
                f"| {conv} | {bucket} | {m['trades']} | {_fmt_pf(m['pf'])} | {m['win_rate']:.1f} | "
                f"{m['avg_return_pct']:.2f} | {avg_month if avg_month is None else round(avg_month, 2)} | "
                f"{m.get('pct_months_positive', 0.0):.0f} | {m.get('max_dd_pct', 0.0):.1f} |"
            )
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import gzip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--tickers-limit", type=int, default=None,
                         help="テスト用に銘柄数を絞る(本番は未指定=全225銘柄)")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)

    tickers = trader.TICKERS[: args.tickers_limit] if args.tickers_limit else trader.TICKERS
    results, ticker_frames, nikkei_ff, trend_series, all_dates_sorted = run_validation(
        tickers=tickers, n_folds=args.n_folds, recent_days=args.recent_days,
    )

    live_model = trader.load_model()

    print("\n=== Fidelity check (a): live TOP1 2026-09-24 ===")
    fidelity_a = fidelity_check_live_top1(ticker_frames, nikkei_ff, trend_series, "2026-09-24", model=live_model)
    print(json.dumps(fidelity_a, indent=2, default=str, ensure_ascii=False))

    print("\n=== Fidelity check (b): research track 2026-09-30 vs all_candidates_2026-09.csv.gz ===")
    fidelity_b = {"status": "SKIP"}
    repo = "aifuu/stock-ai"
    gz_path = os.path.join(args.output_dir, "all_candidates_2026-09.csv.gz")
    ok, err = download_release_asset(repo, "all-candidates-paper-data", "all_candidates_2026-09.csv.gz", gz_path)
    if ok and os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                recorded = pd.read_csv(f)
            day = recorded[recorded["date"].astype(str) == "2026-09-30"]
            fidelity_b = fidelity_check_research_track(
                ticker_frames, nikkei_ff, trend_series, "2026-09-30", day, model=live_model,
            )
        except Exception as e:
            fidelity_b = {"status": f"SKIP: 読み込み失敗 {e}"}
    else:
        fidelity_b = {"status": f"SKIP: release資産取得失敗 {err}"}
    print(json.dumps(fidelity_b, indent=2, default=str, ensure_ascii=False))

    print("\n=== Diagnostic (3): vs walk_forward_all_candidates.csv.gz ===")
    diagnostic = {"status": "SKIP"}
    wf_gz_path = os.path.join(args.output_dir, "walk_forward_all_candidates.csv.gz")
    ok, err = download_release_asset(repo, "walk-forward-candidates-latest",
                                      "walk_forward_all_candidates.csv.gz", wf_gz_path)
    if ok and os.path.exists(wf_gz_path):
        try:
            with gzip.open(wf_gz_path, "rt", encoding="utf-8") as f:
                wf_df = pd.read_csv(f)
            diagnostic = diagnostic_vs_walk_forward(
                ticker_frames, nikkei_ff, trend_series, live_model, wf_df, max_dates=60,
            )
        except Exception as e:
            diagnostic = {"status": f"SKIP: 読み込み失敗 {e}"}
    else:
        diagnostic = {"status": f"SKIP: release資産取得失敗 {err}"}
    print(json.dumps(diagnostic, indent=2, default=str, ensure_ascii=False))

    full_output = {
        "results": results, "fidelity_live_top1_20260924": fidelity_a,
        "fidelity_research_20260930": fidelity_b, "diagnostic_vs_walk_forward": diagnostic,
    }
    out_json = os.path.join(args.output_dir, "live_model_policy_validation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n✅ 結果JSON: {out_json}")

    for fold in results["folds"]:
        print(f"\n{format_metrics_table(fold['name'], fold['conventions'])}")
    print(f"\n{format_metrics_table('OVERALL', results['overall'])}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("# live_model_policy_validation\n\n")
            f.write(f"usable dates: {results['meta']['usable_date_range']} "
                    f"({results['meta']['usable_dates']}日) / universe={results['meta']['universe_size']}銘柄\n\n")
            f.write("## Fidelity (a) live TOP1 2026-09-24\n\n```json\n")
            f.write(json.dumps(fidelity_a, indent=2, default=str, ensure_ascii=False))
            f.write("\n```\n\n## Fidelity (b) research track 2026-09-30\n\n```json\n")
            f.write(json.dumps(fidelity_b, indent=2, default=str, ensure_ascii=False))
            f.write("\n```\n\n## Diagnostic (c) vs walk_forward_all_candidates\n\n```json\n")
            f.write(json.dumps(diagnostic, indent=2, default=str, ensure_ascii=False))
            f.write("\n```\n\n")
            for fold in results["folds"]:
                f.write(format_metrics_table(fold["name"], fold["conventions"]) + "\n\n")
            f.write(format_metrics_table("OVERALL", results["overall"]) + "\n")

    return full_output


if __name__ == "__main__":
    main()
