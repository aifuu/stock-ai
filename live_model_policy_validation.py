#!/usr/bin/env python3
"""live_model_policy_validation.py

研究専用・本番へ一切影響しないウォークフォワード検証スクリプト。

背景(なぜこれが要るか):
  本番のペーパートレードは daily_directional_top1.py(特徴量・ATR・
  directional_score・make_nikkei・TICKERS) + daily_model_retrain.py が
  fit()する RandomForest(directional_model.pkl) + profit_top10_paper.py
  (候補フィルタ・TP/SL/保有日数・SHORT)で構成される。承認済みpolicyファイル
  (strategy_policy.json=通常/DOWNレジーム, strategy_policy_up.json=UPレジーム)
  に記録されている oos_pf / oos_avg_month_return 等は、実は
  walk_forward.py(fold毎に再学習する別系統のモデル)で検証された数値であり、
  本番で実際に使われている daily_model_retrain.py 系統のモデルでは一度も
  検証されていない。本スクリプトは「本番と同じモデル学習ロジック・本番と
  同じ日次選定/決済ロジック」で、リーク無しウォークフォワードにより
  実際のOOS成績を測定し、policyファイルの記録値と比較する。

★重要な発見(ユーザー確認事項、README代わりにここへ明記):
  1) 本番の実際のエントリー経路は
     ai-stock-scan.yml → paper_fast_entrypoint.py → run_profit_loop.py
       → profit_top10_paper.py(app)
     であり、run_profit_loop.py が profit_top10_paper.scan/open_positions を
     モンキーパッチする。最終的なTOP1選定は profit_top10_paper.scan() 内部の
     (expected_value_pct, score) ソートではなく、run_profit_loop.profit_priority()
     の rank = 0.65*score + 0.35*clip(ev,-10,10)*10 + regime_bonus (実質regime_bonus
     は同一日内で一律に効くだけで相対順位には影響しない) で決まる。
     本スクリプトはこの実際の経路を再現する(_passes_policy/profit_priorityの
     複製、run_profit_loop.pyをそのままimportはしない、理由は下記)。
  2) run_profit_loop.py はimportされた瞬間に(if __name__=="__main__"の外側で)
     profit_top10_paper.scan/open_positions/mark_and_close/load_model を
     モジュールレベルでモンキーパッチする副作用を持つ。これは本番プロセス内では
     意図された設計だが、このプロセス内でimportすると
     profit_top10_paper モジュールのグローバル状態を汚染し、同一プロセス内の
     他のテスト/スクリプトに影響しうる。そのため本ファイルは run_profit_loop.py を
     importせず、run_profit_loop._passes_policy() / profit_priority() の純粋な
     計算ロジックだけを複製する。複製が本物とbyte/value一致することは
     test_live_model_policy_validation.py が AST 抽出で live 関数を直接呼び出し
     照合することで担保する。
  3) run_profit_loop.scan_candidates_fixed() の実際の呼び出しでは
     cached_scan() が nikkei_filter=False の base_policy で
     profit_top10_paper.scan() を呼び、その後の _passes_policy() は
     nikkei_filter を一切チェックしない。つまり承認済みpolicyの
     "nikkei_filter": true (strategy_policy.json) は実運用では効いていない
     (ライブコードのバグ/仕様)。本スクリプトはこの実挙動をそのまま再現する
     (nikkei_filterを適用しない)。ユーザー報告の判断材料として明記する。
  4) エントリー価格規約: profit_top10_paper.scan() は日足終値
     (price=float(d['Close'].iloc[-1])、d=download(t)) を基準にし、
     条件を満たした候補にだけ5分足の直近終値で価格を精緻化する
     (download_5m()の refined=float(intr['Close'].iloc[-1])、
     profit_top10_paper.py:155-162)。実際のエントリー価格は
     「スキャン時刻時点の直近5分足終値(取得できない場合は当日日足終値)」。
     本スクリプトは2022年〜の長期日足ヒストリカルデータしか使えず、過去の
     分足は入手できないため、エントリー価格は「その日の日足終値」で近似する
     (既知の乖離、このモジュールの出力にも明記する)。
  5) daily_model_retrain.py はTP/SL/hold_days/up_threshold/min_score_for_buy/
     nikkei_filterを常にstrategy_policy.json(通常/DOWN用)固定で読み、
     futures_trend による UP/DOWN policy切替を一切行わない(自身のOOSゲート
     専用の簡略シミュレーションのため)。本スクリプトはこれを踏襲せず、
     profit_top10_paper.select_policy_file() と同じ futures_trend ベースの
     policy切替(UP日はstrategy_policy_up.json、それ以外はstrategy_policy.json
     にフォールバック=strategy_policy_down.jsonが存在しないため)を再現する。
  6) 学習データのラベル構成・fit()は daily_model_retrain.py の
     build_ticker_frame/flatten_training_rows/fit_rf をそのままimportして使う
     (直接re-implementしない)。ラベルの horizon purge (=2×hold_days暦日、
     daily_model_retrain.pyのoos_cutoff - HOLD_DAYS*2日と同じ考え方)も
     そのまま踏襲する。

Fold設計:
  - 全期間: LMV_START_DATE 〜 LMV_END_DATE (既定 2022-01-01 〜 実行日)。
  - 直近 LMV_RECENT_FOLD_DAYS 営業日(既定60)を専用の "recent" foldとして
    予約する。
  - 残り期間を LMV_N_FOLDS 個(既定4)の連続するOOS区間に均等分割する
    (fold_1が最古、fold_{N}が直近寄り、その直後にrecent foldが続く)。
  - 各foldの学習データは「fold開始日より前の全データ」(拡大窓、
    daily_model_retrain.pyの設計を踏襲)から、
      a) ラベルのhorizon purge: daily_model_retrain.pyと同じ
         HOLD_DAYS*2暦日ぶんをfold開始日直前から除外
         (HOLD_DAYSはstrategy_policy.jsonのhold_days、学習ラベルは
         常にこの値を使う。daily_model_retrain.pyの設計をそのまま踏襲)。
      b) 追加のfold間ギャップ: adversarial_strategy_validator.pyの
         PURGE_DAYS/EMBARGO_DAYS(既定7営業日)と同じ考え方で、
         fold開始日直前の7営業日ぶんの学習行も追加で除外する
         (2つの目的が異なるため両方適用する: (a)はラベルの未来参照
         そのものを防ぐため、(b)はfold境界をまたぐ近接データによる
         楽観的な相関を避けるため)。

ライブ実行時のyfinance呼び出しはこのスクリプト自身が行う
(walk_forward.py / daily_model_retrain.py と同じ方式)。このサンドボックス
環境ではyfinanceに到達できないため、ネットワークを要する経路は
test_live_model_policy_validation.py の合成データ/モックで検証する。
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import daily_directional_top1 as trader
import daily_model_retrain as retrain
import futures_trend
import paper_risk_policy
import profit_top10_paper as papertrader

# =====================================================================
# 環境変数(このリポジトリの WF_* 命名規則に合わせ LMV_ プレフィックスを使う)
# =====================================================================
START_DATE = pd.Timestamp(os.getenv("LMV_START_DATE", "2022-01-01"))
END_DATE = pd.Timestamp(os.getenv("LMV_END_DATE") or pd.Timestamp.today().normalize())
RECENT_FOLD_DAYS = int(os.getenv("LMV_RECENT_FOLD_DAYS", "60"))
N_FOLDS = int(os.getenv("LMV_N_FOLDS", "4"))
FOLD_PURGE_DAYS = int(os.getenv("LMV_FOLD_PURGE_DAYS", "7"))  # adversarial_strategy_validator.py既定を踏襲
MIN_TRAIN_ROWS = int(os.getenv("LMV_MIN_TRAIN_ROWS", str(retrain.MIN_TRAIN_ROWS)))
DOWNLOAD_PERIOD = os.getenv("LMV_DOWNLOAD_PERIOD", "5y")
OUTPUT_DIR = os.getenv("LMV_OUTPUT_DIR", ".")
RESULTS_JSON = os.getenv("LMV_RESULTS_JSON", "live_model_policy_validation_results.json")
TRADES_CSV_PREFIX = os.getenv("LMV_TRADES_CSV_PREFIX", "live_model_policy_validation_trades")
TICKERS_OVERRIDE = os.getenv("LMV_TICKERS")  # カンマ区切り、テスト/スモーク用
POLICY_NORMAL_FILE = os.getenv("LMV_POLICY_NORMAL_FILE", "strategy_policy.json")
POLICY_UP_FILE = os.getenv("LMV_POLICY_UP_FILE", "strategy_policy_up.json")

CANDIDATES_CSV_LOCAL = os.getenv("LMV_CANDIDATES_CSV")  # ローカルオーバーライド(テスト用)
CANDIDATES_RELEASE_TAG = os.getenv("LMV_CANDIDATES_RELEASE_TAG", "walk-forward-candidates-latest")
CANDIDATES_ASSET_NAME = os.getenv("LMV_CANDIDATES_ASSET", "walk_forward_all_candidates.csv.gz")
CANDIDATES_REPO = os.getenv("LMV_CANDIDATES_REPO", "aifuu/stock-ai")
SKIP_DIAGNOSTIC = os.getenv("LMV_SKIP_DIAGNOSTIC", "0").strip().lower() in ("1", "true", "yes", "on")

FEE_RATE_PCT = float(os.getenv("INTRADAY_FEE_RATE", str(papertrader.FEE_RATE))) * 2 * 100.0
SHORT_ENABLED = os.getenv("ENABLE_SHORT_PAPER", "1" if papertrader.SHORT_ENABLED else "0").lower() in (
    "1", "true", "yes", "on",
)


# =====================================================================
# ライブロジックの複製(run_profit_loop.pyは副作用があるためimportしない。
# ここでの複製が本物と一致することは test_live_model_policy_validation.py の
# AST抽出テストで担保する)。
# =====================================================================

def market_regime_from_nikkei(kairi25, ret5):
    """run_profit_loop._market_regime() / daily_model_retrain._oos_regime() /
    adversarial_strategy_validator.pyの market_regime 列と同じ3値判定式。"""
    if kairi25 is None or ret5 is None or pd.isna(kairi25) or pd.isna(ret5):
        return "neutral"
    kairi25, ret5 = float(kairi25), float(ret5)
    if kairi25 > 0 and ret5 > 0:
        return "bullish"
    if kairi25 < 0 and ret5 < 0:
        return "bearish"
    return "neutral"


def natural_direction(up_pct, down_pct, flat_pct, short_enabled=True):
    """profit_top10_paper.scan()の候補プール構築ゲートを、
    cached_scan()が使う閾値ゼロのbase_policy(up_threshold=0, min_score=0,
    nikkei_filter=False)で評価した場合と等価な"自然な方向"を返す。
    scan()内部の`ok`条件: up>=0 and up>down and flat<50 (BUYの場合)、
    down>=0 and down>up and flat<50 (SHORTの場合)。up_probability/
    down_probabilityは常に0以上なのでup>=0/down>=0は恒等的に真であり、
    scoreも常に0以上(directional_scoreの値域)なのでscore>=0も恒等的に真。
    """
    if flat_pct >= 50.0:
        return None
    if up_pct > down_pct:
        return "BUY"
    if short_enabled and down_pct > up_pct:
        return "SHORT"
    return None


def passes_policy(direction, up_pct, down_pct, flat_pct, score, policy):
    """run_profit_loop._passes_policy()と同一の判定
    (nikkei_filterは一切見ない=実運用のバグ/仕様をそのまま再現する)。"""
    threshold = float(policy["up_threshold"])
    score_min = float(policy["min_score_for_buy"])
    if flat_pct >= 50.0 or score < score_min:
        return False
    if direction == "SHORT":
        return down_pct >= threshold and down_pct > up_pct
    return up_pct >= threshold and up_pct > down_pct


def profit_priority_rank(direction, score, price, tp, sl, up, down, flat, regime, fee_rate_pct):
    """run_profit_loop.profit_priority()の1候補ぶんの計算を複製する。
    up/down/flatは0..1の確率(パーセントではない)。feedback_weightは
    daily_model_retrain.pyと同じ理由(将来実績への依存=リーク回避)で
    常に1.0に固定する(本番のtrade_feedback_policy.jsonは使わない)。
    戻り値: (rank, ev, regime_bonus)
    """
    flat_cost = -(fee_rate_pct)
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
    feedback_weight = 1.0
    rank = (0.65 * float(score) + 0.35 * max(-10.0, min(10.0, ev)) * 10.0 + regime_bonus) * feedback_weight
    return rank, ev, regime_bonus


def regime_allows(regime, direction):
    """run_profit_loop.profit_priority()冒頭のレジームハードゲート
    (bullish→BUYのみ、bearish→SHORTのみ、neutral→両方可)。"""
    if regime == "bullish" and direction != "BUY":
        return False
    if regime == "bearish" and direction != "SHORT":
        return False
    return True


def check_exit(direction, high, low, close, tp, sl, days_held, hold_limit):
    """profit_top10_paper.mark_and_close() / update_open_position() と同じ
    優先順位(同一バーでTP/SLどちらにも触れた場合はSL優先)。
    hold_limitに達した日はClose価格でTIME決済する
    (daily_model_retrain.simulate_oos_top1と同じロジック)。
    戻り値: (exit_price, reason) または (None, None)。
    """
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
    if exit_price is None and days_held >= hold_limit:
        exit_price, exit_reason = close, "TIME"
    return exit_price, exit_reason


def trade_return_pct(direction, entry, exit_price, fee_rate_pct):
    ret = (exit_price / entry - 1.0) * 100.0 if direction == "BUY" else (entry / exit_price - 1.0) * 100.0
    return ret - fee_rate_pct


# =====================================================================
# データ準備(本番の関数をそのまま流用する。period だけ本スクリプト用に
# 延長するため trader.download を一時的にラップする)。
# =====================================================================

def _extend_download_period(period):
    """trader.download() の既定期間(3y)を、長期ウォークフォワードに必要な
    期間へ拡張するためのプロセス内モンキーパッチ。このプロセス(本研究
    スクリプト自身のワークフロー実行)以外には一切影響しない。元関数への
    委譲以外の挙動変更はしない(単体テストで検証)。"""
    original = trader.download

    def _wrapped(ticker, period_arg=period):
        return original(ticker, period=period_arg)

    trader.download = _wrapped
    return original


def build_universe(tickers):
    """daily_model_retrain.build_universe をそのまま呼ぶ(学習ラベル・
    特徴量ロジックの完全な再利用)。period拡張のモンキーパッチは
    呼び出し前後で必ず元に戻す。"""
    original_download = _extend_download_period(DOWNLOAD_PERIOD)
    try:
        ticker_frames, nikkei = retrain.build_universe(tickers)
    finally:
        trader.download = original_download
    return ticker_frames, nikkei


# =====================================================================
# Fold境界の計算
# =====================================================================

@dataclass
class Fold:
    name: str
    oos_dates: list
    train_cutoff: pd.Timestamp


def build_folds(all_dates, hold_days_for_purge):
    """all_datesは昇順ソート済みの営業日リスト。
    直近RECENT_FOLD_DAYS日をrecent foldとして予約し、残りをN_FOLDS個の
    連続する区間に均等分割する。各foldの学習カットオフは
    「fold開始日 - max(FOLD_PURGE_DAYS営業日, HOLD_DAYS*2暦日相当)」。
    """
    all_dates = list(all_dates)
    if len(all_dates) <= RECENT_FOLD_DAYS + FOLD_PURGE_DAYS + N_FOLDS:
        raise RuntimeError("期間が短すぎてfoldを構成できません")

    recent_dates = all_dates[-RECENT_FOLD_DAYS:]
    remaining = all_dates[: -(RECENT_FOLD_DAYS + FOLD_PURGE_DAYS)]
    if len(remaining) < N_FOLDS * 20:
        raise RuntimeError("recent fold控除後の期間が短すぎます")

    fold_size = len(remaining) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < N_FOLDS - 1 else len(remaining)
        oos_dates = remaining[start_idx:end_idx]
        if not oos_dates:
            continue
        fold_start = pd.Timestamp(oos_dates[0])
        # (a) daily_model_retrain.py と同じラベルhorizon purge(2×hold_days暦日)
        label_purge_cutoff = fold_start - pd.Timedelta(days=hold_days_for_purge * 2)
        # (b) adversarial_strategy_validator.py と同じfold間ギャップ(営業日ベース)
        prior_trading_dates = [d for d in all_dates if d < fold_start]
        trading_day_cutoff = (
            pd.Timestamp(prior_trading_dates[-FOLD_PURGE_DAYS])
            if len(prior_trading_dates) > FOLD_PURGE_DAYS
            else (pd.Timestamp(prior_trading_dates[0]) if prior_trading_dates else fold_start)
        )
        train_cutoff = min(label_purge_cutoff, trading_day_cutoff)
        folds.append(Fold(name=f"fold_{i + 1}", oos_dates=oos_dates, train_cutoff=train_cutoff))

    recent_start = pd.Timestamp(recent_dates[0])
    recent_label_cutoff = recent_start - pd.Timedelta(days=hold_days_for_purge * 2)
    prior_trading_dates = [d for d in all_dates if d < recent_start]
    recent_trading_cutoff = (
        pd.Timestamp(prior_trading_dates[-FOLD_PURGE_DAYS])
        if len(prior_trading_dates) > FOLD_PURGE_DAYS
        else (pd.Timestamp(prior_trading_dates[0]) if prior_trading_dates else recent_start)
    )
    folds.append(Fold(
        name="recent60",
        oos_dates=recent_dates,
        train_cutoff=min(recent_label_cutoff, recent_trading_cutoff),
    ))
    return folds


# =====================================================================
# 1日ぶんの候補構築
# =====================================================================

def build_day_candidates(date, ticker_frames, model, features, policy, regime, fee_rate_pct):
    """指定日の全銘柄について、実運用の
    scan()(閾値ゼロプール) → _passes_policy(承認済み閾値) →
    regimeハードゲート → profit_priority(rank)
    の連鎖を再現し、rank降順でソートした候補リストを返す。
    各要素: dict(ticker, direction, score, up, down, flat, price, tp, sl, rank, ev)
    """
    out = []
    for ticker, xf in ticker_frames.items():
        if date not in xf.index:
            continue
        row = xf.loc[date]
        if pd.isna(row[features]).any():
            continue
        atr_abs = float(row["atr_abs"])
        if not np.isfinite(atr_abs) or atr_abs <= 0:
            continue
        try:
            probs = model.predict_proba(row[features].to_frame().T)[0]
            classes = list(model.classes_)
            down = float(probs[classes.index(0)])
            up = float(probs[classes.index(2)])
            flat = float(probs[classes.index(1)])
        except Exception:
            continue

        long_s, short_s = trader.directional_score(row, up, down)
        price = float(row["Close"])
        up_pct, down_pct, flat_pct = up * 100.0, down * 100.0, flat * 100.0

        direction = natural_direction(up_pct, down_pct, flat_pct, SHORT_ENABLED)
        if direction is None:
            continue
        score = float(long_s if direction == "BUY" else short_s)
        if not passes_policy(direction, up_pct, down_pct, flat_pct, score, policy):
            continue
        if not regime_allows(regime, direction):
            continue

        tp, sl = papertrader._tp_sl(price, direction, atr_abs, policy)
        rank, ev, _bonus = profit_priority_rank(direction, score, price, tp, sl, up, down, flat, regime, fee_rate_pct)
        out.append({
            "ticker": ticker, "direction": direction, "score": score,
            "up_probability": up_pct, "down_probability": down_pct,
            "price": price, "tp": tp, "sl": sl, "rank": rank, "ev": ev,
        })
    out.sort(key=lambda c: (c["rank"], c["score"], max(c["up_probability"], c["down_probability"])), reverse=True)
    return out


def select_policy_for_date(trend, policy_normal, policy_up):
    """profit_top10_paper.select_policy_file()と同じフォールバック
    (strategy_policy_down.jsonが存在しないためDOWN日もstrategy_policy.jsonを使う)。"""
    return policy_up if trend == futures_trend.UP else policy_normal


# =====================================================================
# Fold単位のシミュレーション
# =====================================================================

@dataclass
class OpenPosition:
    ticker: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    entry_date: pd.Timestamp
    hold_limit: int
    dd_mult: float = 1.0
    days_held: int = 0


def simulate_top1(fold_oos_dates, ticker_frames, model, features, nikkei_ff, trend_by_date,
                   policy_normal, policy_up, fee_rate_pct):
    """TOP1(本番と同じ、資金集中・複利)のシミュレーション。
    paper_risk_policy.position_size_multiplier()によるドローダウン段階縮小も
    再現する(本番のopen_positions()と同じ挙動)。
    """
    capital = paper_risk_policy.INITIAL_CAPITAL
    peak = capital
    position: Optional[OpenPosition] = None
    trades = []
    equity_points = []

    for date in fold_oos_dates:
        if position is not None:
            xf = ticker_frames.get(position.ticker)
            if xf is None or date not in xf.index:
                continue
            bar = xf.loc[date]
            high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            position.days_held += 1
            exit_price, reason = check_exit(
                position.direction, high, low, close, position.tp, position.sl,
                position.days_held, position.hold_limit,
            )
            if exit_price is not None:
                ret = trade_return_pct(position.direction, position.entry_price, exit_price, fee_rate_pct)
                # position_size_multiplier(profit_top10_paper.open_positions()の
                # slot_budget=capital*dd_mult)を近似: エントリー時点のdd_multぶんの
                # 予算だけが損益に反映される(本番のLOT_SIZE丸めは無視し連続値で近似)。
                pnl = capital * position.dd_mult * ret / 100.0
                capital += pnl
                peak = max(peak, capital)
                trades.append({
                    "entry_date": position.entry_date, "exit_date": date,
                    "ticker": position.ticker, "direction": position.direction,
                    "return_pct": ret, "reason": reason, "capital_after": capital,
                })
                equity_points.append({"date": date, "capital": capital})
                position = None
            continue

        state = {"capital": capital, "peak": peak}
        dd_mult = paper_risk_policy.position_size_multiplier(state)
        trend = trend_by_date.get(date, futures_trend.DOWN)
        policy = select_policy_for_date(trend, policy_normal, policy_up)
        regime = market_regime_from_nikkei(
            nikkei_ff.loc[date, "kairi25"] if date in nikkei_ff.index else None,
            nikkei_ff.loc[date, "ret5"] if date in nikkei_ff.index else None,
        )
        candidates = build_day_candidates(date, ticker_frames, model, features, policy, regime, fee_rate_pct)
        if not candidates:
            continue
        top = candidates[0]
        # dd_mult(投資額縮小)は損益率(return_pct)そのものには影響しないが、
        # 損益額の資産への反映割合として本番のslot_budget=capital*dd_multを
        # 近似する。株数丸め(LOT_SIZE)は無視し、連続値として扱う
        # (日足のみのヒストリカル検証のための単純化、明記する)。
        entry_price = top["price"]
        position = OpenPosition(
            ticker=top["ticker"], direction=top["direction"], entry_price=entry_price,
            tp=top["tp"], sl=top["sl"], entry_date=date, hold_limit=int(policy["hold_days"]),
            dd_mult=dd_mult,
        )

    return trades, equity_points


def simulate_topn(fold_oos_dates, ticker_frames, model, features, nikkei_ff, trend_by_date,
                   policy_normal, policy_up, fee_rate_pct, top_n=None):
    """ALL(top_n=None)/TOP3/TOP5の等金額(equal-weight)・非複利シミュレーション。
    各日の(承認済み条件通過・regimeゲート後の)候補のうちtop_n件を独立した
    等金額トレードとして扱う(既存の adversarial_strategy_validator.py の
    legacy_overlap会計=stats()のgroupby("date")["return"].mean()と同じ考え方:
    エントリー日ごとに平均リターンを取り、その系列を複利換算して資産曲線/
    最大DDを算出する)。ポジションは重複保有を許す(全額集中投資ではなく、
    各エントリーが「予算のN分の1」という前提のため)。
    """
    trades = []
    for date in fold_oos_dates:
        trend = trend_by_date.get(date, futures_trend.DOWN)
        policy = select_policy_for_date(trend, policy_normal, policy_up)
        regime = market_regime_from_nikkei(
            nikkei_ff.loc[date, "kairi25"] if date in nikkei_ff.index else None,
            nikkei_ff.loc[date, "ret5"] if date in nikkei_ff.index else None,
        )
        candidates = build_day_candidates(date, ticker_frames, model, features, policy, regime, fee_rate_pct)
        if not candidates:
            continue
        picks = candidates if top_n is None else candidates[:top_n]
        hold_limit = int(policy["hold_days"])
        for c in picks:
            xf = ticker_frames.get(c["ticker"])
            if xf is None:
                continue
            future_idx = xf.index[xf.index > date]
            future_bars = xf.loc[future_idx].head(hold_limit)
            exit_price = exit_reason = exit_date = None
            for day_no, (idx, bar) in enumerate(future_bars.iterrows(), 1):
                high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
                exit_price, exit_reason = check_exit(c["direction"], high, low, close, c["tp"], c["sl"], day_no, hold_limit)
                if exit_price is not None:
                    exit_date = idx
                    break
            if exit_price is None:
                continue
            ret = trade_return_pct(c["direction"], c["price"], exit_price, fee_rate_pct)
            trades.append({
                "entry_date": date, "exit_date": exit_date, "ticker": c["ticker"],
                "direction": c["direction"], "return_pct": ret, "reason": exit_reason,
            })
    return trades


# =====================================================================
# 集計
# =====================================================================

def compute_flat_metrics(trades):
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "avg_return": 0.0}
    df = pd.DataFrame(trades)
    ret = pd.to_numeric(df["return_pct"], errors="coerce").dropna()
    gains = float(ret[ret > 0].sum())
    losses = float(-ret[ret < 0].sum())
    pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    return {
        "trades": int(len(df)),
        "win_rate": float((ret > 0).mean() * 100.0),
        "pf": pf,
        "avg_return": float(ret.mean()),
    }


def compute_equity_curve_metrics(daily_return_pct_series):
    """暦月次リターン・最大DDを算出する(adversarial_strategy_validator.stats()と
    同じ、日次(エントリー日ベース)平均リターン系列からの複利換算方式)。"""
    if daily_return_pct_series is None or len(daily_return_pct_series) == 0:
        return {"avg_month_return": 0.0, "max_dd_pct": 0.0}
    s = daily_return_pct_series.sort_index()
    equity = (1 + s / 100.0).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min() * 100.0)
    monthly = s.groupby(pd.PeriodIndex(s.index, freq="M")).apply(lambda x: float(((1 + x / 100.0).prod() - 1) * 100.0))
    avg_month = float(monthly.mean()) if len(monthly) else 0.0
    return {"avg_month_return": avg_month, "max_dd_pct": dd}


def aggregate_top1(trades, equity_points):
    flat = compute_flat_metrics(trades)
    if equity_points:
        eq = pd.DataFrame(equity_points).set_index("date")["capital"]
        rets = eq.pct_change().fillna(eq.iloc[0] / paper_risk_policy.INITIAL_CAPITAL - 1.0) * 100.0
        curve = compute_equity_curve_metrics(rets)
    else:
        curve = {"avg_month_return": 0.0, "max_dd_pct": 0.0}
    flat.update(curve)
    return flat


def aggregate_equal_weight(trades):
    flat = compute_flat_metrics(trades)
    if trades:
        df = pd.DataFrame(trades)
        daily = df.groupby("entry_date")["return_pct"].mean()
        curve = compute_equity_curve_metrics(daily)
    else:
        curve = {"avg_month_return": 0.0, "max_dd_pct": 0.0}
    flat.update(curve)
    return flat


# =====================================================================
# walk_forward候補CSVとの診断比較
# =====================================================================

def _download_candidates_csv(dest_path):
    url = f"https://github.com/{CANDIDATES_REPO}/releases/download/{CANDIDATES_RELEASE_TAG}/{CANDIDATES_ASSET_NAME}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"⚠ candidates CSV ダウンロード失敗 (試行{attempt + 1}/3): {exc}")
    return False


def load_walk_forward_candidates():
    if CANDIDATES_CSV_LOCAL:
        if not os.path.exists(CANDIDATES_CSV_LOCAL):
            print(f"⚠ LMV_CANDIDATES_CSV={CANDIDATES_CSV_LOCAL} が見つかりません。診断をスキップします")
            return None
        return pd.read_csv(CANDIDATES_CSV_LOCAL)

    gz_path = CANDIDATES_ASSET_NAME
    csv_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path
    if not os.path.exists(csv_path):
        if not os.path.exists(gz_path):
            ok = _download_candidates_csv(gz_path)
            if not ok:
                print("⚠ walk_forward_all_candidates.csv.gz を取得できませんでした。診断をスキップします")
                return None
        try:
            with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        except OSError as exc:
            print(f"⚠ candidates CSV 展開失敗: {exc}")
            return None
    return pd.read_csv(csv_path)


def diagnostic_vs_walk_forward(live_daily_rows, wf_df):
    """live_daily_rows: list of dict(date, ticker, up_probability, score)
    (その日候補プールの各銘柄、TOP1選定前の全候補)。wf_dfはwalk_forwardの
    候補CSV(date, ticker, score, up_prob, ...)。日付+銘柄で内部結合し、
    Spearman順位相関とTOP1銘柄一致率を計算する。"""
    if not live_daily_rows or wf_df is None or wf_df.empty:
        return {"overlap_days": 0, "spearman_up_prob": None, "spearman_score": None, "top1_overlap_rate": None}

    live_df = pd.DataFrame(live_daily_rows)
    live_df["date"] = pd.to_datetime(live_df["date"]).dt.normalize()
    wf = wf_df.copy()
    wf["date"] = pd.to_datetime(wf["date"], errors="coerce").dt.normalize()
    for col in ("score", "up_prob"):
        wf[col] = pd.to_numeric(wf[col], errors="coerce")
    wf = wf.dropna(subset=["date", "ticker", "score", "up_prob"])

    merged = live_df.merge(wf, on=["date", "ticker"], suffixes=("_live", "_wf"))
    overlap_days = merged["date"].nunique()
    if merged.empty:
        return {"overlap_days": 0, "spearman_up_prob": None, "spearman_score": None, "top1_overlap_rate": None}

    spearman_up = float(merged["up_probability"].corr(merged["up_prob"], method="spearman"))
    spearman_score = float(merged["score_live"].corr(merged["score_wf"], method="spearman"))

    live_top1 = live_df.sort_values(["date", "up_probability"], ascending=[True, False]).groupby("date").head(1)
    wf_top1 = wf.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(1)
    joined = live_top1.merge(wf_top1, on="date", suffixes=("_live", "_wf"))
    overlap_rate = float((joined["ticker_live"] == joined["ticker_wf"]).mean() * 100.0) if len(joined) else None

    return {
        "overlap_days": int(overlap_days),
        "spearman_up_prob": spearman_up,
        "spearman_score": spearman_score,
        "top1_overlap_rate": overlap_rate,
    }


# =====================================================================
# メイン
# =====================================================================

def _read_policy_recorded_metrics(path):
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return {"oos_pf": p.get("oos_pf"), "oos_avg_month_return": p.get("oos_avg_month_return"),
                "oos_dd": p.get("oos_dd"), "oos_signals": p.get("oos_signals")}
    except Exception as exc:
        return {"error": str(exc)}


def main():
    print("=" * 90)
    print("🔬 LIVE MODEL POLICY VALIDATION (research-only, no live side effects)")
    print(f"期間: {START_DATE.date()} 〜 {END_DATE.date()}｜Fold数: {N_FOLDS}+recent{RECENT_FOLD_DAYS}")
    print("=" * 90)

    tickers = trader.TICKERS if not TICKERS_OVERRIDE else [t.strip() for t in TICKERS_OVERRIDE.split(",") if t.strip()]

    policy_normal = papertrader.load_policy(POLICY_NORMAL_FILE)
    policy_up = papertrader.load_policy(POLICY_UP_FILE)
    hold_days_for_purge = int(policy_normal["hold_days"])

    ticker_frames, nikkei = build_universe(tickers)
    all_dates = sorted(set().union(*[set(x.index) for x in ticker_frames.values()]))
    all_dates = [d for d in all_dates if START_DATE <= pd.Timestamp(d) <= END_DATE]
    nikkei_ff = nikkei.reindex(all_dates).ffill()

    trend_series = futures_trend.historical_trend_series(START_DATE, END_DATE)
    trend_by_date = trend_series["trend"].to_dict() if not trend_series.empty else {}

    folds = build_folds(all_dates, hold_days_for_purge)

    wf_candidates = None if SKIP_DIAGNOSTIC else load_walk_forward_candidates()

    fold_results = []
    all_trades_top1, all_equity_top1 = [], []
    all_trades = {"ALL": [], "TOP3": [], "TOP5": []}
    all_live_daily_rows = []

    for fold in folds:
        print(f"\n--- {fold.name}: OOS {pd.Timestamp(fold.oos_dates[0]).date()} 〜 "
              f"{pd.Timestamp(fold.oos_dates[-1]).date()} ({len(fold.oos_dates)}営業日) "
              f"｜学習カットオフ {fold.train_cutoff.date()} ---")
        fit_rows = retrain.flatten_training_rows(ticker_frames, before_date=fold.train_cutoff)
        if len(fit_rows) < MIN_TRAIN_ROWS:
            print(f"⚠ {fold.name}: 学習データ不足({len(fit_rows)}行<{MIN_TRAIN_ROWS}) スキップ")
            continue
        model = retrain.fit_rf(fit_rows)

        trades_top1, equity_top1 = simulate_top1(
            fold.oos_dates, ticker_frames, model, trader.FEATURES, nikkei_ff, trend_by_date,
            policy_normal, policy_up, FEE_RATE_PCT,
        )
        trades_all = simulate_topn(
            fold.oos_dates, ticker_frames, model, trader.FEATURES, nikkei_ff, trend_by_date,
            policy_normal, policy_up, FEE_RATE_PCT, top_n=None,
        )
        trades_top3 = simulate_topn(
            fold.oos_dates, ticker_frames, model, trader.FEATURES, nikkei_ff, trend_by_date,
            policy_normal, policy_up, FEE_RATE_PCT, top_n=3,
        )
        trades_top5 = simulate_topn(
            fold.oos_dates, ticker_frames, model, trader.FEATURES, nikkei_ff, trend_by_date,
            policy_normal, policy_up, FEE_RATE_PCT, top_n=5,
        )

        fold_metrics = {
            "fold": fold.name,
            "oos_start": str(pd.Timestamp(fold.oos_dates[0]).date()),
            "oos_end": str(pd.Timestamp(fold.oos_dates[-1]).date()),
            "train_cutoff": str(fold.train_cutoff.date()),
            "train_rows": int(len(fit_rows)),
            "TOP1": aggregate_top1(trades_top1, equity_top1),
            "ALL": aggregate_equal_weight(trades_all),
            "TOP3": aggregate_equal_weight(trades_top3),
            "TOP5": aggregate_equal_weight(trades_top5),
        }
        fold_results.append(fold_metrics)
        print(f"  TOP1: trades={fold_metrics['TOP1']['trades']} pf={fold_metrics['TOP1']['pf']:.3f} "
              f"avg_month_return={fold_metrics['TOP1']['avg_month_return']:.2f}%")

        all_trades_top1.extend(trades_top1)
        all_equity_top1.extend(equity_top1)
        all_trades["ALL"].extend(trades_all)
        all_trades["TOP3"].extend(trades_top3)
        all_trades["TOP5"].extend(trades_top5)

        if wf_candidates is not None:
            for date in fold.oos_dates:
                trend = trend_by_date.get(date, futures_trend.DOWN)
                policy = select_policy_for_date(trend, policy_normal, policy_up)
                regime = market_regime_from_nikkei(
                    nikkei_ff.loc[date, "kairi25"] if date in nikkei_ff.index else None,
                    nikkei_ff.loc[date, "ret5"] if date in nikkei_ff.index else None,
                )
                for c in build_day_candidates(date, ticker_frames, model, trader.FEATURES, policy, regime, FEE_RATE_PCT):
                    all_live_daily_rows.append({
                        "date": date, "ticker": c["ticker"],
                        "up_probability": c["up_probability"], "score": c["score"],
                    })

    overall = {
        "TOP1": aggregate_top1(all_trades_top1, all_equity_top1),
        "ALL": aggregate_equal_weight(all_trades["ALL"]),
        "TOP3": aggregate_equal_weight(all_trades["TOP3"]),
        "TOP5": aggregate_equal_weight(all_trades["TOP5"]),
    }

    diagnostic = diagnostic_vs_walk_forward(all_live_daily_rows, wf_candidates)

    policy_recorded = {
        "normal": _read_policy_recorded_metrics(POLICY_NORMAL_FILE),
        "up": _read_policy_recorded_metrics(POLICY_UP_FILE),
    }

    results = {
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "period": {"start": str(START_DATE.date()), "end": str(END_DATE.date())},
        "entry_price_convention": (
            "日足終値近似(本番は直近5分足終値、取得不可時は当日日足終値。"
            "profit_top10_paper.py:138,155-162参照。過去の分足データが無いため"
            "長期ヒストリカル検証では日足終値で近似している)"
        ),
        "nikkei_filter_note": (
            "本番の実行経路(run_profit_loop.scan_candidates_fixed→_passes_policy)は"
            "policyのnikkei_filterを一切適用しない(実装上の既知の乖離)。本スクリプトは"
            "この実挙動をそのまま再現している。"
        ),
        "fee_rate_pct_roundtrip": FEE_RATE_PCT,
        "short_enabled": SHORT_ENABLED,
        "folds": fold_results,
        "overall": overall,
        "policy_recorded_metrics": policy_recorded,
        "diagnostic_vs_walk_forward": diagnostic,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, RESULTS_JSON)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    for label, trades in (("top1", all_trades_top1), ("all", all_trades["ALL"]),
                           ("top3", all_trades["TOP3"]), ("top5", all_trades["TOP5"])):
        path = os.path.join(OUTPUT_DIR, f"{TRADES_CSV_PREFIX}_{label}.csv")
        pd.DataFrame(trades).to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("📊 SUMMARY (markdown)")
    print("=" * 90)
    print(render_markdown_summary(results))

    return results


def render_markdown_summary(results):
    lines = ["# Live Model Policy Validation", ""]
    lines.append(f"Period: {results['period']['start']} — {results['period']['end']}")
    lines.append("")
    lines.append(f"Entry price convention: {results['entry_price_convention']}")
    lines.append("")
    lines.append(f"⚠ {results['nikkei_filter_note']}")
    lines.append("")
    lines.append("## Overall (pooled across all folds)")
    lines.append("")
    lines.append("| bucket | trades | win_rate% | PF | avg_return% | avg_month_return% | max_dd% |")
    lines.append("|---|---|---|---|---|---|---|")
    for bucket in ("TOP1", "ALL", "TOP3", "TOP5"):
        m = results["overall"][bucket]
        pf_disp = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
        lines.append(
            f"| {bucket} | {m['trades']} | {m['win_rate']:.1f} | {pf_disp} | "
            f"{m['avg_return']:.2f} | {m['avg_month_return']:.2f} | {m['max_dd_pct']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-fold TOP1")
    lines.append("")
    lines.append("| fold | OOS range | trades | PF | avg_month_return% | max_dd% |")
    lines.append("|---|---|---|---|---|---|")
    for fold in results["folds"]:
        m = fold["TOP1"]
        pf_disp = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
        lines.append(
            f"| {fold['fold']} | {fold['oos_start']}〜{fold['oos_end']} | {m['trades']} | "
            f"{pf_disp} | {m['avg_month_return']:.2f} | {m['max_dd_pct']:.2f} |"
        )
    lines.append("")
    lines.append("## Comparison vs recorded policy OOS metrics (walk_forward.py-validated, different model family)")
    lines.append("")
    for label in ("normal", "up"):
        rec = results["policy_recorded_metrics"][label]
        lines.append(f"- policy={label}: recorded oos_pf={rec.get('oos_pf')}, "
                      f"oos_avg_month_return={rec.get('oos_avg_month_return')}, "
                      f"oos_dd={rec.get('oos_dd')}, oos_signals={rec.get('oos_signals')}")
    lines.append("")
    lines.append("## Diagnostic vs walk_forward_all_candidates.csv.gz (different model family)")
    lines.append("")
    diag = results["diagnostic_vs_walk_forward"]
    lines.append(f"- overlap_days: {diag['overlap_days']}")
    lines.append(f"- spearman(up_probability): {diag['spearman_up_prob']}")
    lines.append(f"- spearman(score): {diag['spearman_score']}")
    lines.append(f"- TOP1 ticker overlap rate: {diag['top1_overlap_rate']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
