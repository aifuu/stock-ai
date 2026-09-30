"""
daily_model_retrain.py

毎営業日、取引開始前(JST 07:30目安)に1回だけ実行し、
daily_directional_top1.py が使う model.pkl を再学習する。

★このスクリプトのゲート思想(2026-08、walk_forward.pyの検証思想を移植):
walk_forward.py 本体(複数パラメータ候補を多重検定補正しながら探す、
5日型モデル用の重厚な検証パイプライン)は、TOP1方向性モデル(買い/空売り
両対応・単一銘柄選択)には構造がそのまま合わないため流用できない。
そのため「本番投入前に、未来データを一切使わないOOS区間で実際に
シミュレーション売買し、PF(プロフィットファクター)・勝率・最大DD・
取引数が一定基準を満たさない限り本番差し替えしない」という
walk_forward.py と同じ検証思想だけを、このTOP1システム用に作り直した。

具体的なゲート:
・直近 HOLDOUT_DAYS 日を OOS(Out-Of-Sample)区間として学習から完全に除外
  (walk_forward.pyの OOS-SANCTUARY と同じ考え方＝探索・学習に絶対使わない)
・そのOOS区間で、daily_directional_top1.py と全く同じロジック
  (TOP1選択・ATR×TP/SL・最大HOLD_DAYS営業日保有)を新モデルで
  日次シミュレーションし、実際に取引した場合のPF/勝率/最大DD/取引数を計算
・ゲート条件(OOS取引数が十分な場合): PF>=1.0 かつ 最大DD<=30%
  を満たした時だけ model.pkl を本番差し替え。満たさなければ見送り、
  今までのmodel.pklを維持する
・OOS取引数が少なすぎて判定不能な場合は、参考採用として差し替えるが
  レポートにその旨を明記する(判断材料が無いのに機械的に止め続けない
  ようにするため)

directional_paper_history.csv(実際のペーパートレード結果)は学習
データそのものには混ぜず、"直近の実績"として daily_retrain_report.csv
に記録するだけに留める(自分の予測ミスを学習に混ぜて悪循環になるのを避けるため)。
"""

import hashlib
import json
import os
import time
from datetime import datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import daily_directional_top1 as trader
import safe_state

JST = ZoneInfo("Asia/Tokyo")

MODEL_FILE = trader.MODEL_FILE
TRAIN_FILE = trader.TRAIN_FILE
# ★変更(2026-09): trader.HISTORY_FILE("directional_paper_history.csv")は
# 実際には誰も書き込まない別系統の履歴ファイル名で、本番のTOP10ペーパー
# トレードの実績は"profit_top10_paper_history.csv"(profit_top10_paper.py側)
# に記録される。recent_live_performance()がこの取り違えのせいで常に
# 「実績なし」になっていたため、実際に書き込まれるファイルを直接指す。
HISTORY_FILE = "profit_top10_paper_history.csv"
FEATURES = trader.FEATURES

# ★修正(2026-09): 以前はTP_MULT/SL_MULT/HOLD_DAYSをdaily_directional_top1.py側の
# ハードコード値(3.0/1.5/5)からそのまま使っていたため、OOSゲートのシミュレーションが
# 実運用(profit_top10_paper.py、承認済みすtrategy_policy.jsonのatr_tp_multiplier=4.0/
# atr_sl_multiplier=1.75)と異なるTP/SL幅で「合格/不合格」を判定していた。TP/SL幅が
# 違えばTP到達率・SL到達率・TIME決済率・PF・最大DDは全て変わりうるため、「OOSで合格
# =実運用条件でも合格」と言えない状態だった。strategy_policy.jsonから直接読み、実運用と
# 同じ条件でOOSシミュレーションするようにする(署名検証はここでは行わない。実発注の
# 安全性はprofit_top10_paper.load_policy()側の厳格な検証が別途担保しているため、ここは
# OOSシミュレーション用の参考値取得に留める)。
#
# ★修正(2026-09、追加): TP/SL同期だけでは不十分だった。simulate_oos_top1()の候補選定が
# 「全銘柄から単純にスコア最大の1件」を無条件採用する別戦略になっており、実運用
# (run_profit_loop.py)が使う (1)承認済みpolicyのUP/SCORE閾値・日経フィルター、
# (2)日経レジーム(強気=BUYのみ/弱気=SHORTのみ)、(3)売買手数料 を一切反映していなかった。
# これらもpolicyから読み込み、simulate_oos_top1()側で再現する。
# (trade_feedback_policy.jsonのフィードバック重みだけは、フィードバック自体が
# 「その時点までの実績」に依存し将来分を含めるとリークになり得るため、意図的に含めない)
POLICY_FILE = "strategy_policy.json"


def _load_policy_rules():
    try:
        with open(POLICY_FILE, encoding="utf-8") as f:
            p = json.load(f)
        return {
            "tp_mult": float(p["atr_tp_multiplier"]),
            "sl_mult": float(p["atr_sl_multiplier"]),
            "hold_days": int(p["hold_days"]),
            "up_threshold": float(p["up_threshold"]),
            "min_score": float(p["min_score_for_buy"]),
            "nikkei_filter": str(p["nikkei_filter"]).lower() in ("true", "1", "yes", "on"),
        }
    except Exception as e:
        print(f"⚠ {POLICY_FILE}読込失敗、フィルター無し(旧仕様相当)にフォールバック: {e}")
        return {
            "tp_mult": trader.TP_MULT, "sl_mult": trader.SL_MULT, "hold_days": trader.HOLD_DAYS,
            "up_threshold": 0.0, "min_score": 0.0, "nikkei_filter": False,
        }


_POLICY_RULES = _load_policy_rules()
TP_MULT = _POLICY_RULES["tp_mult"]
SL_MULT = _POLICY_RULES["sl_mult"]
HOLD_DAYS = _POLICY_RULES["hold_days"]
UP_THRESHOLD = _POLICY_RULES["up_threshold"]
MIN_SCORE_FOR_BUY = _POLICY_RULES["min_score"]
NIKKEI_FILTER_ON = _POLICY_RULES["nikkei_filter"]
# profit_top10_paper.pyのFEE_RATE(片道)と同じデフォルト・同じ環境変数名。
# 実運用は1トレードにつき往復(エントリー+決済の2回)分の手数料を資金から
# 差し引くため、%換算では概ね FEE_RATE*2*100 に相当する(profit_priority()の
# flat_cost計算と同じ近似)。
FEE_RATE_PCT = float(os.getenv("INTRADAY_FEE_RATE", "0.00055")) * 2 * 100.0
ATR_TARGET_MULTIPLIER = 1.0

PREV_MODEL_FILE = "model_prev.pkl"
RETRAIN_REPORT_FILE = "daily_retrain_report.csv"
# ★追加(2026-09-30、チャンピオン/チャレンジャー方式の導入): 現行model.pklが
# 「いつ時点までのデータで学習されたか(train_cutoff)」を記録するサイドカー。
# これが無いと、次回以降の再学習が現行モデルを正しく評価できているか
# (＝OOS区間が本当に現行モデル未学習の区間か)を判定できない。
# 既存のdirectional_model.pkl(2026-09-12投入)にはこのファイルが無いため、
# 「不明(leaky扱い)」として安全側にフォールバックする。
MODEL_META_FILE = "directional_model_meta.json"

HOLDOUT_DAYS = int(os.getenv("RETRAIN_HOLDOUT_DAYS", "90"))
MIN_OOS_TRADES = int(os.getenv("RETRAIN_MIN_OOS_TRADES", "15"))
MIN_OOS_PF = float(os.getenv("RETRAIN_MIN_OOS_PF", "1.0"))
MAX_OOS_DD_PCT = float(os.getenv("RETRAIN_MAX_OOS_DD_PCT", "30.0"))
MIN_TRAIN_ROWS = int(os.getenv("RETRAIN_MIN_ROWS", "3000"))
DOWNLOAD_SLEEP = float(os.getenv("RETRAIN_DOWNLOAD_SLEEP", "0.15"))
# ★追加(2026-09-30): チャレンジャーが絶対ゲート(PF>=MIN_OOS_PF)に届かなくても、
# 現行チャンピオンを明確に上回れば差し替える相対ゲートのマージン。
# 「絶対ゲートは通らないが、今のモデルよりは明確にマシ」を拾うためのもので、
# 僅差の入れ替え(ノイズによる無駄な差し替え)を避けるため一定のマージンを要求する。
RETRAIN_CHAMPION_MARGIN_PF = float(os.getenv("RETRAIN_CHAMPION_MARGIN_PF", "0.10"))
RETRAIN_CHAMPION_MAX_DD_WORSE_PCT = float(os.getenv("RETRAIN_CHAMPION_MAX_DD_WORSE_PCT", "5.0"))

# ★決定(ライブセッション中の再学習・モデル差し替え禁止、2026-09-29の
# インシデントを受けて): daily-model-retrain.ymlはクラウド側ルーチンが
# JST 07:15にworkflow_dispatchする運用に一本化したが、このスクリプト自身の
# schedule('30 22 * * 0-4' = JST 07:30予定)もGitHub Actionsの実運用遅延
# (実測で1.5〜2.5時間程度)によりJST 09:20〜10:00頃に発火してしまい、
# ai-stock-scan.ymlのAMセッション(09:20〜12:35 JST、5分ループでpullしながら
# 売買判定)と重なってdirectional_model.pklを差し替えてしまうと、ライブの
# ペーパートレードセッション中にモデルが入れ替わるという事故になる。
# scheduleそのものは撤廃せずフォールバックとして残す方針のため、ここで
# 二重の安全策を設ける:
#   (i) 時刻ウィンドウガード: JST 08:40〜15:34(ライブセッション想定時間帯、
#       AMセッション開始08:30の少し後〜PMセッション終了15:35の直前)は
#       いかなるトリガーでも再学習・モデル差し替えを行わない。
#   (ii) 当日実施済みガード: daily_retrain_report.csv(このrunがコミット
#       する唯一の「完走した」証跡)に本日日付の行が既にあれば、2回目以降の
#       起動は何もせずスキップする。この行はmain()がOOSシミュレーション・
#       学習・デプロイ判定まで完走した直後にのみ追記されるため、失敗/中断
#       した実行は当日行を残さず、後続の有効な起動をブロックしない。
RETRAIN_WINDOW_BLOCK_START = dtime(8, 40)
RETRAIN_WINDOW_BLOCK_END = dtime(15, 35)


def retrain_window_open(now):
    """JST時刻がライブセッション想定時間帯(08:40〜15:34)の外ならTrue。
    境界は「08:40以降15:35未満は禁止」= 08:39は許可・08:40は禁止、
    15:34は禁止・15:35は許可。"""
    t = now.time()
    return t < RETRAIN_WINDOW_BLOCK_START or t >= RETRAIN_WINDOW_BLOCK_END


def already_retrained_today(today_str, report_file=None):
    """daily_retrain_report.csvに本日(today_str, 'YYYY-MM-DD')の行が
    既にあればTrue。ファイルが無い/壊れている/date列が無い場合は「未実施」
    として扱う(安全側=再学習を止めない側)。この行はmain()がOOSシミュレー
    ションからデプロイ判定まで完走した最後にのみ追記されるため、失敗/中断
    した実行はここでTrueにならず、後続の有効な起動をブロックしない。"""
    path = report_file or RETRAIN_REPORT_FILE
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "date" not in df.columns:
        return False
    return bool((df["date"].astype(str) == today_str).any())


def notify(msg):
    """再学習結果のDiscord通知。
    Discord送信の失敗は再学習/OOSゲートの成否とは無関係なので、
    通知例外でジョブ全体を失敗扱いにしない。
    """
    try:
        trader.send(msg)
    except Exception as e:
        print(f"⚠ Discord通知に失敗しましたが、再学習処理自体は正常終了として扱います: {e}")


def make_futures_features():
    """日経225先物(NIY=F)の特徴量を、1日ラグを入れて計算する。"""
    f = trader.download("NIY=F")
    if f is None or f.empty:
        print("⚠ 日経225先物データ取得失敗。future_*特徴量はプレースホルダのまま")
        return None
    c = f["Close"].squeeze()
    out = pd.DataFrame(index=f.index)
    out["future_return"] = c.pct_change()
    out["future_ma5"] = c.rolling(5).mean()
    out["future_rsi"] = trader.rsi(c)
    out["future_gap"] = (c - c.shift(1)) / c.shift(1)
    for col in out.columns:
        out[col] = out[col].shift(1)
    return out


def build_ticker_frame(ticker, nikkei, futures_df):
    """1銘柄分の特徴量+target+シミュレーション用の生値を持つDataFrameを返す。"""
    df = trader.download(ticker)
    if df is None or len(df) < 150:
        return None

    x = trader.features(df, nikkei)

    if futures_df is not None:
        aligned = futures_df.reindex(x.index).ffill()
        for col in ["future_return", "future_ma5", "future_rsi", "future_gap"]:
            x[col] = aligned[col]

    x["atr_abs"] = trader.atr(x)

    future_price = x["Close"].shift(-HOLD_DAYS)
    future_return = (future_price / x["Close"] - 1.0) * 100.0
    atr_threshold = x["atr_ratio"] * ATR_TARGET_MULTIPLIER * np.sqrt(HOLD_DAYS)
    x["target"] = np.select(
        [future_return <= -atr_threshold, future_return >= atr_threshold],
        [0, 2],
        default=1,
    )
    x["target_valid"] = future_price.notna()
    return x


def build_universe(tickers):
    """全銘柄の特徴量フレームを構築する。"""
    nikkei = trader.make_nikkei()
    if nikkei is None:
        raise RuntimeError("日経平均データが取得できず、学習データを作れません")
    futures_df = make_futures_features()

    ticker_frames = {}
    for i, ticker in enumerate(tickers, 1):
        x = build_ticker_frame(ticker, nikkei, futures_df)
        if x is None:
            print(f"[{i}/{len(tickers)}] {ticker}: スキップ(データ不足)")
        else:
            ticker_frames[ticker] = x
            print(f"[{i}/{len(tickers)}] {ticker}: {len(x):,}行")
        time.sleep(DOWNLOAD_SLEEP)

    if not ticker_frames:
        raise RuntimeError("有効な学習データが1件も作れませんでした")
    return ticker_frames, nikkei


def flatten_training_rows(ticker_frames, before_date=None):
    frames = []
    for ticker, x in ticker_frames.items():
        part = x[x["target_valid"]]
        if before_date is not None:
            part = part[part.index < before_date]
        part = part.dropna(subset=FEATURES)
        if part.empty:
            continue
        flat = part[["target"] + FEATURES].copy()
        flat["ticker"] = ticker
        flat["date"] = part.index
        frames.append(flat)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "target"] + FEATURES)
    return pd.concat(frames, ignore_index=True)


def _oos_regime(nikkei_ff, date):
    """指定日時点での日経レジーム(run_profit_loop._market_regime()と同じ判定式)。
    データが無い/NaNの場合はneutral・日経上昇扱い(安全側=フィルター実質無効)にする。"""
    if date not in nikkei_ff.index:
        return "neutral", True
    row = nikkei_ff.loc[date]
    kairi, ret5 = row.get("kairi25"), row.get("ret5")
    if pd.isna(kairi) or pd.isna(ret5):
        return "neutral", True
    kairi, ret5 = float(kairi), float(ret5)
    if kairi > 0 and ret5 > 0:
        regime = "bullish"
    elif kairi < 0 and ret5 < 0:
        regime = "bearish"
    else:
        regime = "neutral"
    nikkei_up = bool(row.get("nikkei_uptrend", True))
    return regime, nikkei_up


def simulate_oos_top1(model, ticker_frames, oos_dates, nikkei_ff):
    """OOS区間だけでTOP1を日次シミュレーションする。
    ★修正(2026-09): 承認済みpolicyのUP/SCORE閾値・日経フィルター・日経レジーム
    (強気=BUYのみ/弱気=SHORTのみ)・売買手数料を反映し、実運用(run_profit_loop.py+
    profit_top10_paper.py)の選定ロジックに揃える(feedback_weightのみ意図的に除外、
    理由は本ファイル冒頭コメント参照)。"""
    position = None
    trades = []

    for date in oos_dates:
        if position is not None:
            ticker = position["ticker"]
            xf = ticker_frames.get(ticker)
            if xf is None or date not in xf.index:
                continue
            bar = xf.loc[date]
            high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            exit_price = exit_reason = None

            if position["direction"] == "BUY":
                if low <= position["sl"] and high >= position["tp"]:
                    exit_price, exit_reason = position["sl"], "SL_BOTH"
                elif high >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
                elif low <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
            else:
                if high >= position["sl"] and low <= position["tp"]:
                    exit_price, exit_reason = position["sl"], "SL_BOTH"
                elif low <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
                elif high >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"

            position["days_held"] += 1
            if exit_price is None and position["days_held"] >= HOLD_DAYS:
                exit_price, exit_reason = close, "TIME"

            if exit_price is not None:
                entry = position["entry_price"]
                ret = (
                    (exit_price / entry - 1.0) * 100.0
                    if position["direction"] == "BUY"
                    else (entry / exit_price - 1.0) * 100.0
                )
                ret -= FEE_RATE_PCT
                trades.append({
                    "entry_date": position["entry_date"], "exit_date": date,
                    "ticker": ticker, "direction": position["direction"],
                    "return_pct": ret, "reason": exit_reason,
                })
                position = None
            continue

        regime, nikkei_up = _oos_regime(nikkei_ff, date)
        candidates = []
        for ticker, xf in ticker_frames.items():
            if date not in xf.index:
                continue
            row = xf.loc[date]
            if pd.isna(row[FEATURES]).any():
                continue
            atr_abs = float(row["atr_abs"])
            if not np.isfinite(atr_abs) or atr_abs <= 0:
                continue
            try:
                probs = model.predict_proba(row[FEATURES].to_frame().T)[0]
                classes = list(model.classes_)
                down = float(probs[classes.index(0)])
                up = float(probs[classes.index(2)])
                flat = float(probs[classes.index(1)])
            except Exception:
                continue

            long_s, short_s = trader.directional_score(row, up, down)
            price = float(row["Close"])
            up_pct, down_pct, flat_pct = up * 100.0, down * 100.0, flat * 100.0

            for direction, score in (("BUY", long_s), ("SHORT", short_s)):
                if direction == "BUY":
                    ok = up_pct >= UP_THRESHOLD and up_pct > down_pct and flat_pct < 50.0 and score >= MIN_SCORE_FOR_BUY
                else:
                    ok = down_pct >= UP_THRESHOLD and down_pct > up_pct and flat_pct < 50.0 and score >= MIN_SCORE_FOR_BUY
                if not ok:
                    continue
                if NIKKEI_FILTER_ON and direction == "BUY" and not nikkei_up:
                    continue
                if NIKKEI_FILTER_ON and direction == "SHORT" and nikkei_up:
                    continue
                if regime == "bullish" and direction != "BUY":
                    continue
                if regime == "bearish" and direction != "SHORT":
                    continue

                if direction == "BUY":
                    tp, sl = price + atr_abs * TP_MULT, price - atr_abs * SL_MULT
                    reward_pct = max(0.0, (tp / price - 1.0) * 100.0) if price > 0 else 0.0
                    risk_pct = max(0.0, (1.0 - sl / price) * 100.0) if price > 0 else 0.0
                    ev = up * reward_pct - down * risk_pct
                else:
                    tp, sl = price - atr_abs * TP_MULT, price + atr_abs * SL_MULT
                    reward_pct = max(0.0, (1.0 - tp / price) * 100.0) if price > 0 else 0.0
                    risk_pct = max(0.0, (sl / price - 1.0) * 100.0) if price > 0 else 0.0
                    ev = down * reward_pct - up * risk_pct
                ev -= flat * FEE_RATE_PCT

                preferred = (regime == "bullish" and direction == "BUY") or (regime == "bearish" and direction == "SHORT")
                regime_bonus = 10.0 if preferred else 0.0
                rank = 0.65 * score + 0.35 * max(-10.0, min(10.0, ev)) * 10.0 + regime_bonus

                candidates.append({
                    "ticker": ticker, "direction": direction, "score": score, "rank": rank,
                    "price": price, "tp": tp, "sl": sl,
                })

        if not candidates:
            continue
        candidates.sort(key=lambda z: z["rank"], reverse=True)
        top = candidates[0]
        position = {
            "ticker": top["ticker"], "direction": top["direction"],
            "entry_price": top["price"], "tp": top["tp"], "sl": top["sl"],
            "entry_date": date, "days_held": 0,
        }

    if position is not None:
        xf = ticker_frames.get(position["ticker"])
        if xf is not None and not xf.empty:
            last_close = float(xf["Close"].iloc[-1])
            entry = position["entry_price"]
            ret = (
                (last_close / entry - 1.0) * 100.0
                if position["direction"] == "BUY"
                else (entry / last_close - 1.0) * 100.0
            )
            ret -= FEE_RATE_PCT
            trades.append({
                "entry_date": position["entry_date"], "exit_date": xf.index[-1],
                "ticker": position["ticker"], "direction": position["direction"],
                "return_pct": ret, "reason": "FORCED_EOS",
            })

    return trades


def compute_pf_metrics(trades):
    if not trades:
        return {"trades": 0, "pf": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0}

    df = pd.DataFrame(trades).sort_values("entry_date")
    gross_profit = float(df.loc[df["return_pct"] > 0, "return_pct"].sum())
    gross_loss = float(-df.loc[df["return_pct"] < 0, "return_pct"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = float((df["return_pct"] > 0).mean() * 100.0)

    capital, peak, max_dd = 1.0, 1.0, 0.0
    for r in df["return_pct"]:
        capital *= (1.0 + r / 100.0)
        peak = max(peak, capital)
        dd = (capital / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)

    return {"trades": int(len(df)), "pf": pf, "win_rate": win_rate, "max_dd_pct": max_dd}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def compute_model_id(path=None):
    """model_id = デプロイされたdirectional_model.pklのバイト列のsha256先頭16桁。"""
    return sha256_file(path or MODEL_FILE)[:16]


def compute_policy_hash():
    """strategy_policy.json / strategy_policy_up.jsonのsha256先頭12桁
    (cf106bdc4c56 / cfe6da4cc960 のように既存箇所で使われている桁数と揃える)。
    存在しないファイルはキーごと省略する。
    """
    hashes = {}
    for name in (POLICY_FILE, "strategy_policy_up.json"):
        if os.path.exists(name):
            hashes[name] = sha256_file(name)[:12]
    return hashes


def load_model_meta(path=None):
    """MODEL_META_FILEを読む。無い/壊れている場合はNone(=不明扱い、安全側)。"""
    p = path or MODEL_META_FILE
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or "train_cutoff" not in meta:
            return None
        return meta
    except Exception:
        return None


def save_model_meta(
    training_data_end_date,
    today,
    validation_metrics,
    oos_cutoff=None,
    deploy_reason=None,
    previous_meta=None,
    model_file=None,
    path=None,
):
    """新たに本番投入したモデルのメタ情報(サイドカー、gitコミット対象)を保存する。

    ★重要(validation_* と 実際にデプロイされる重みの違い):
    validation_trades/validation_pf/validation_win_rate/validation_drawdownは、
    OOS区間(training_data_end_dateより前で打ち切ったholdout-excluded fit=
    oos_model)をtraining_data_end_date直後から評価した「未来を全く見ていない」
    成績。しかし実際にmodel.pklとしてデプロイされる重みは、このOOS評価の後に
    training_data_end_dateまでの全データ(OOS区間も含む)で再学習し直した
    別モデル(final_model)であり、validation_*が示す精度とは厳密には別物
    (walk_forward.pyと同じ「検証はholdout、本番投入は全データ再学習」という
    設計。ファイル冒頭のモジュールdocstring参照)。

    次回以降の再学習は、train_cutoff(=training_data_end_date)を見て、
    このモデルに対する今回のOOS区間が未学習区間かどうか(incumbent_window_status)
    を判定する。
    """
    model_id = compute_model_id(model_file)
    meta = {
        "model_id": model_id,
        "training_date": today,
        "training_data_end_date": str(pd.Timestamp(training_data_end_date).date()),
        # ★後方互換: incumbent_window_status()はtrain_cutoffキーのみを見る。
        "train_cutoff": str(pd.Timestamp(training_data_end_date).date()),
        "deployed_date": today,
        "validation_period": {
            "start": str(pd.Timestamp(oos_cutoff).date()) if oos_cutoff is not None else None,
            "end": str(pd.Timestamp(training_data_end_date).date()),
        },
        "validation_trades": validation_metrics["trades"],
        "validation_pf": round(validation_metrics["pf"], 3) if np.isfinite(validation_metrics["pf"]) else "inf",
        "validation_win_rate": round(validation_metrics["win_rate"], 2),
        "validation_drawdown": round(validation_metrics["max_dd_pct"], 2),
        # ★後方互換(旧キー、ModelMetaRoundTripTests等が参照): validation_*と同値。
        "oos_pf": round(validation_metrics["pf"], 3) if np.isfinite(validation_metrics["pf"]) else "inf",
        "oos_trades": validation_metrics["trades"],
        "oos_max_dd_pct": round(validation_metrics["max_dd_pct"], 2),
        "policy_hash": compute_policy_hash(),
        "deploy_reason": deploy_reason,
        "previous_model_id": (previous_meta or {}).get("model_id"),
    }
    safe_state.atomic_write_json(path or MODEL_META_FILE, meta)
    return meta


def incumbent_window_status(meta, oos_cutoff):
    """現行モデル(チャンピオン)にとって、今回のOOS区間が本当に未学習区間か。
    'safe'   : メタ情報があり、学習カットオフがOOS区間開始より前(重複無し)
    'leaky'  : メタ情報はあるが、学習カットオフがOOS区間に重なる(リーク)
    'unknown': メタ情報が無い(古いモデル、または初回)。安全側でleaky同様に扱う
    """
    if meta is None:
        return "unknown"
    try:
        train_cutoff = pd.Timestamp(meta["train_cutoff"])
    except Exception:
        return "unknown"
    return "safe" if train_cutoff < pd.Timestamp(oos_cutoff) else "leaky"


def evaluate_incumbent(ticker_frames, oos_dates, nikkei_ff, model_file=None):
    """現行model.pkl(チャンピオン)を、チャレンジャーと全く同じOOS区間・
    シミュレーションルールで評価する。読み込み/評価に失敗した場合はNone
    (=比較不能、呼び出し側で絶対ゲートのみにフォールバックする)。"""
    path = model_file or MODEL_FILE
    try:
        incumbent_model = joblib.load(path)
        trades = simulate_oos_top1(incumbent_model, ticker_frames, oos_dates, nikkei_ff)
        return compute_pf_metrics(trades)
    except Exception as e:
        print(f"⚠ 現行モデルの評価に失敗、比較不能として扱います: {e}")
        return None


def load_registered_incumbent_evaluation(report_file=None):
    """daily_retrain_report.csvの「最も新しいdeployed==True行」を、現行モデルが
    本番投入された時点で登録された評価(登録時OOS評価)として返す。

    現行モデルがevaluate_incumbent()で今回のOOS区間と同一条件で再評価できない
    場合(サイドカー無し/リーク=window_status!='safe')に、代わりにこの
    「投入した当時の記録」を参照するためのもの。行が無い/壊れている場合はNone。
    """
    path = report_file or RETRAIN_REPORT_FILE
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "deployed" not in df.columns:
        return None
    deployed_mask = df["deployed"].astype(str).str.strip().str.lower().isin(("true", "1"))
    deployed_rows = df[deployed_mask]
    if deployed_rows.empty:
        return None
    deployed_rows = deployed_rows.copy()
    deployed_rows["_date"] = pd.to_datetime(deployed_rows["date"], errors="coerce")
    deployed_rows = deployed_rows.sort_values("_date")
    row = deployed_rows.iloc[-1]
    try:
        trades = int(row["oos_trades"])
        pf_raw = row["oos_pf"]
        pf = float("inf") if str(pf_raw).strip().lower() == "inf" else float(pf_raw)
        max_dd_pct = float(row["oos_max_dd_pct"])
    except Exception:
        return None
    return {
        "date": str(row["date"]),
        "trades": trades,
        "pf": pf,
        "max_dd_pct": max_dd_pct,
        "reason": str(row.get("reason", "")),
    }


def decide_deploy(
    challenger_metrics,
    incumbent_metrics,
    window_status,
    incumbent_exists,
    registered_eval=None,
    min_trades=None,
    min_pf=None,
    max_dd_pct=None,
    margin_pf=None,
    max_dd_worse_pct=None,
):
    """チャンピオン/チャレンジャーのデプロイ判定(main()から分離、単体テスト用)。

    ・チャレンジャーの取引数が不足 → 既存挙動そのまま(現行モデル有りなら見送り、
      無ければ参考採用として投入)
    ・取引数は十分 →
        - 絶対ゲート(PF>=min_pf かつ 最大DD<=max_dd_pct)を通過 → 投入
        - 絶対ゲートは未通過でも、現行モデルの同一区間での成績と比較可能
          (window_status=='safe') かつ PFがmargin_pf以上上回りDDがmax_dd_worse_pct
          ポイント以上悪化していなければ → 投入(相対ゲート)
        - 絶対ゲート未通過・window_status!='safe'(比較不能)でも、registered_eval
          (現行モデルの投入時点の登録済みOOS評価)が渡された場合:
            - 登録時取引数がmin_trades未満(=現行モデルは投入時点でそもそも
              まともに評価されていなかった)なら、チャレンジャーが最低評価条件
              (取引数>=min_trades・最大DD<=max_dd_pct)を満たし、かつ登録時PFを
              厳密に上回れば投入(「PFがX改善」ではなく「現行が評価不能だった
              ため最低条件を満たす新モデルへ更新」という理由文にする)。
            - 登録時取引数がmin_trades以上なら、通常の相対ゲート(margin_pf/
              max_dd_worse_pct)を登録時実績との比較で適用する。
        - それ以外(現行モデルが無い/比較不能でregistered_evalも無い、または
          相対ゲート・登録時比較のいずれも満たさない) → 見送り
    """
    min_trades = MIN_OOS_TRADES if min_trades is None else min_trades
    min_pf = MIN_OOS_PF if min_pf is None else min_pf
    max_dd_pct = MAX_OOS_DD_PCT if max_dd_pct is None else max_dd_pct
    margin_pf = RETRAIN_CHAMPION_MARGIN_PF if margin_pf is None else margin_pf
    max_dd_worse_pct = RETRAIN_CHAMPION_MAX_DD_WORSE_PCT if max_dd_worse_pct is None else max_dd_worse_pct

    if challenger_metrics["trades"] < min_trades:
        if incumbent_exists:
            return False, f"OOS取引数不足({challenger_metrics['trades']}件<{min_trades}件)のため判定保留・前回モデルを継続使用"
        return True, f"OOS取引数不足({challenger_metrics['trades']}件<{min_trades}件)のため判定不能だが、既存モデル無し(初回投入前)のため参考採用として投入"

    absolute_pass = challenger_metrics["pf"] >= min_pf and abs(challenger_metrics["max_dd_pct"]) <= max_dd_pct

    if window_status == "safe" and incumbent_metrics is not None:
        beats_incumbent = (
            challenger_metrics["pf"] >= incumbent_metrics["pf"] + margin_pf
            and abs(challenger_metrics["max_dd_pct"]) <= abs(incumbent_metrics["max_dd_pct"]) + max_dd_worse_pct
        )
        if absolute_pass or beats_incumbent:
            via = "絶対ゲート" if absolute_pass else f"相対ゲート(現行PF={incumbent_metrics['pf']:.2f}比+{margin_pf:.2f}以上)"
            return True, (
                f"チャレンジャーがチャンピオンと同一OOS区間で比較の上{via}通過"
                f"(チャレンジャーPF={challenger_metrics['pf']:.2f} vs 現行PF={incumbent_metrics['pf']:.2f}, "
                f"チャレンジャー最大DD={challenger_metrics['max_dd_pct']:.1f}% vs 現行最大DD={incumbent_metrics['max_dd_pct']:.1f}%)"
            )
        return False, (
            f"チャレンジャーは絶対ゲート未通過かつチャンピオンを有意に上回れず、現行モデルを継続使用"
            f"(チャレンジャーPF={challenger_metrics['pf']:.2f} vs 現行PF={incumbent_metrics['pf']:.2f}, "
            f"チャレンジャー最大DD={challenger_metrics['max_dd_pct']:.1f}% vs 現行最大DD={incumbent_metrics['max_dd_pct']:.1f}%)"
        )

    if not absolute_pass and incumbent_exists and window_status != "safe" and registered_eval is not None:
        reg_trades = registered_eval["trades"]
        reg_pf = registered_eval["pf"]
        if reg_trades < min_trades:
            meets_min = (
                challenger_metrics["trades"] >= min_trades
                and abs(challenger_metrics["max_dd_pct"]) <= max_dd_pct
                and challenger_metrics["pf"] > reg_pf
            )
            if meets_min:
                return True, (
                    f"現行モデルは評価不能(登録時{reg_trades}件<{min_trades}件)のため、"
                    f"最低評価条件を満たす新モデルへ更新"
                    f"(チャレンジャー取引数={challenger_metrics['trades']}件, "
                    f"PF={challenger_metrics['pf']:.3f}, 最大DD={challenger_metrics['max_dd_pct']:.1f}% "
                    f"vs 登録時PF={reg_pf:.3f}[{registered_eval['date']}登録・{reg_trades}件])"
                )
            return False, (
                f"チャレンジャーは絶対ゲート未通過、現行モデルは評価不能(登録時{reg_trades}件<{min_trades}件)"
                f"のため登録時実績と比較したが最低評価条件(取引数>={min_trades}件・最大DD<={max_dd_pct}%・"
                f"登録時PF{reg_pf:.3f}超)を満たさず、現行モデルを継続使用"
                f"(チャレンジャー取引数={challenger_metrics['trades']}件, PF={challenger_metrics['pf']:.3f}, "
                f"最大DD={challenger_metrics['max_dd_pct']:.1f}%)"
            )

        beats_registered = (
            challenger_metrics["pf"] >= reg_pf + margin_pf
            and abs(challenger_metrics["max_dd_pct"]) <= abs(registered_eval["max_dd_pct"]) + max_dd_worse_pct
        )
        if beats_registered:
            return True, (
                f"チャレンジャーが現行モデルの登録時実績(投入時[{registered_eval['date']}]記録、"
                f"直接比較不能[{window_status}]のため代用)を相対ゲート(+{margin_pf:.2f}以上)で上回るため更新"
                f"(チャレンジャーPF={challenger_metrics['pf']:.3f} vs 登録時PF={reg_pf:.3f}, "
                f"チャレンジャー最大DD={challenger_metrics['max_dd_pct']:.1f}% vs "
                f"登録時最大DD={registered_eval['max_dd_pct']:.1f}%)"
            )
        return False, (
            f"チャレンジャーは絶対ゲート未通過かつ現行モデルの登録時実績([{registered_eval['date']}]記録)を"
            f"有意に上回れず、現行モデルを継続使用"
            f"(チャレンジャーPF={challenger_metrics['pf']:.3f} vs 登録時PF={reg_pf:.3f}, "
            f"チャレンジャー最大DD={challenger_metrics['max_dd_pct']:.1f}% vs "
            f"登録時最大DD={registered_eval['max_dd_pct']:.1f}%)"
        )

    # 現行モデルが無い、または比較不能(leaky/unknown)でregistered_evalも無い → 絶対ゲートのみで判定
    note = ""
    if incumbent_exists and window_status != "safe":
        note = f"(現行モデルはOOS区間が未学習と確認できず[{window_status}]比較不能のため絶対ゲートのみで判定)"
    if absolute_pass:
        return True, f"OOSゲート通過(PF={challenger_metrics['pf']:.2f}>={min_pf}, 最大DD={challenger_metrics['max_dd_pct']:.1f}%){note}"
    return False, f"OOSゲート未通過(PF={challenger_metrics['pf']:.2f}, 最大DD={challenger_metrics['max_dd_pct']:.1f}%){note}"


def recent_live_performance(days=30):
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return None
    if df.empty or "exit_date" not in df.columns:
        return None
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce")
    df = df.dropna(subset=["exit_date", "pnl"])
    if df.empty:
        return None
    cutoff = pd.Timestamp.now(tz=JST).tz_localize(None) - pd.Timedelta(days=days)
    recent = df[df["exit_date"] >= cutoff]
    if recent.empty:
        return {"trades": 0, "win_rate": None, "pnl": 0.0}
    win_rate = float((recent["pnl"] > 0).mean() * 100.0)
    return {"trades": int(len(recent)), "win_rate": win_rate, "pnl": float(recent["pnl"].sum())}


def append_retrain_report(row):
    new_df = pd.DataFrame([row])
    if os.path.exists(RETRAIN_REPORT_FILE):
        old_df = pd.read_csv(RETRAIN_REPORT_FILE)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(RETRAIN_REPORT_FILE, index=False, encoding="utf-8-sig")


def fit_rf(rows):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=7, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    model.fit(rows[FEATURES], rows["target"].astype(int))
    return model


def main():
    now = datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    print(f"=== 日次モデル再学習(Walk-Forward OOSゲート付き) {today} ===")

    if not retrain_window_open(now):
        print(
            f"⏳ 実行時刻 {now.strftime('%H:%M')} JSTはライブセッション想定時間帯"
            f"(08:40〜15:34 JST)のため再学習をスキップします(モデル差し替えなし・コミットなし)"
        )
        return

    if already_retrained_today(today):
        print(f"⏭ 本日({today})分の再学習は既に完了済みのためスキップします(コミットなし)")
        return

    ticker_frames, nikkei = build_universe(trader.TICKERS)
    all_dates = sorted(set().union(*[set(x.index) for x in ticker_frames.values()]))
    nikkei_ff = nikkei.reindex(all_dates).ffill()
    last_date = pd.Timestamp(all_dates[-1])
    oos_cutoff = last_date - pd.Timedelta(days=HOLDOUT_DAYS)
    oos_dates = [d for d in all_dates if pd.Timestamp(d) >= oos_cutoff]
    print(f"OOS区間: {oos_cutoff.date()} 〜 {last_date.date()}（{len(oos_dates)}営業日、学習からは完全除外）")

    train_cutoff = oos_cutoff - pd.Timedelta(days=HOLD_DAYS * 2)
    fit_rows = flatten_training_rows(ticker_frames, before_date=train_cutoff)
    print(f"学習データ行数(OOS区間+embargo{HOLD_DAYS}営業日相当分を除く): {len(fit_rows):,}")

    if len(fit_rows) < MIN_TRAIN_ROWS:
        msg = f"⚠ 学習データが{MIN_TRAIN_ROWS}行未満({len(fit_rows)}行)のため再学習を見送り"
        print(msg)
        notify(f"🟡 日次モデル再学習｜{today}\n{msg}\nmodel.pklは変更しません")
        return

    oos_model = fit_rf(fit_rows)
    oos_trades = simulate_oos_top1(oos_model, ticker_frames, oos_dates, nikkei_ff)
    metrics = compute_pf_metrics(oos_trades)
    print(
        f"OOSシミュレーション結果: 取引数={metrics['trades']} "
        f"PF={metrics['pf']:.3f} 勝率={metrics['win_rate']:.1f}% "
        f"最大DD={metrics['max_dd_pct']:.2f}%"
    )

    # ★追加(2026-09-30、チャンピオン/チャレンジャー方式): 従来はチャレンジャー
    # (今回学習したoos_model)を固定の絶対ゲート(PF>=MIN_OOS_PF)にのみ通す設計
    # だったため、現行モデル(チャンピオン)がどれほど悪くても「チャレンジャーが
    # 絶対ゲートを超えない限り現状維持」という比較が起きていた。現行model.pklを
    # 同一OOS区間・同一シミュレーションで評価できる場合(=現行モデルの学習カット
    # オフがOOS区間より前でリークが無い場合)は、その成績と比較した相対ゲートも
    # 追加で見る。比較できない場合(現行モデルが無い/メタ情報が無い古いモデル/
    # 学習カットオフがOOS区間に重なる)は、従来通り絶対ゲートのみで判定する。
    incumbent_exists = os.path.exists(MODEL_FILE)
    incumbent_meta = load_model_meta() if incumbent_exists else None
    window_status = incumbent_window_status(incumbent_meta, oos_cutoff) if incumbent_exists else "no_incumbent"
    incumbent_metrics = None
    if incumbent_exists and window_status == "safe" and metrics["trades"] >= MIN_OOS_TRADES:
        incumbent_metrics = evaluate_incumbent(ticker_frames, oos_dates, nikkei_ff)
        if incumbent_metrics is not None:
            print(
                f"チャンピオン(現行model.pkl)同一OOS区間シミュレーション結果: "
                f"取引数={incumbent_metrics['trades']} PF={incumbent_metrics['pf']:.3f} "
                f"勝率={incumbent_metrics['win_rate']:.1f}% 最大DD={incumbent_metrics['max_dd_pct']:.2f}%"
            )
        else:
            window_status = "unknown"

    # ★追加(②モデル更新ルール): 現行モデルが同一OOS区間で直接比較できない
    # (window_status != 'safe')場合、代わりに投入時点の登録済み評価
    # (daily_retrain_report.csvの直近deployed==True行)を参照する。
    registered_eval = None
    if incumbent_exists and window_status != "safe":
        registered_eval = load_registered_incumbent_evaluation()
        if registered_eval is not None:
            print(
                f"チャンピオン登録時評価(daily_retrain_report.csv、{registered_eval['date']}投入分)を参照: "
                f"取引数={registered_eval['trades']} PF={registered_eval['pf']:.3f} "
                f"最大DD={registered_eval['max_dd_pct']:.2f}%"
            )

    deploy, reason = decide_deploy(
        metrics, incumbent_metrics, window_status, incumbent_exists, registered_eval=registered_eval,
    )

    full_rows = flatten_training_rows(ticker_frames, before_date=None)
    full_rows.drop(columns=["date"]).to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")

    if deploy:
        if os.path.exists(MODEL_FILE):
            try:
                os.replace(MODEL_FILE, PREV_MODEL_FILE)
            except Exception as e:
                print(f"⚠ 旧model.pklの退避に失敗(続行): {e}")
        final_model = fit_rf(full_rows)
        joblib.dump(final_model, MODEL_FILE)
        save_model_meta(
            last_date, today, metrics,
            oos_cutoff=oos_cutoff, deploy_reason=reason, previous_meta=incumbent_meta,
        )
        print(f"✅ model.pkl 差し替え完了: {reason}")
    else:
        print(f"🟡 model.pkl 差し替え見送り: {reason}")

    live = recent_live_performance(days=30)
    live_text = "実績なし"
    if live and live["trades"] > 0:
        live_text = f"直近30日 取引{live['trades']}件 勝率{live['win_rate']:.1f}% 損益{live['pnl']:+,.0f}円"

    append_retrain_report({
        "date": today,
        "train_rows": len(full_rows),
        "oos_days": len(oos_dates),
        "oos_trades": metrics["trades"],
        "oos_pf": round(metrics["pf"], 3) if np.isfinite(metrics["pf"]) else "inf",
        "oos_win_rate": round(metrics["win_rate"], 2),
        "oos_max_dd_pct": round(metrics["max_dd_pct"], 2),
        "deployed": deploy,
        "reason": reason,
        "live_recent_trades": live["trades"] if live else 0,
        "live_recent_win_rate": round(live["win_rate"], 2) if live and live["win_rate"] is not None else "",
        "live_recent_pnl": round(live["pnl"], 2) if live else 0.0,
        # ★追加(2026-09-30、チャンピオン/チャレンジャー方式、後方互換のため末尾に追加):
        "incumbent_window_status": window_status,
        "incumbent_oos_trades": incumbent_metrics["trades"] if incumbent_metrics else "",
        "incumbent_oos_pf": (round(incumbent_metrics["pf"], 3) if incumbent_metrics and np.isfinite(incumbent_metrics["pf"]) else ("inf" if incumbent_metrics else "")),
        "incumbent_oos_win_rate": round(incumbent_metrics["win_rate"], 2) if incumbent_metrics else "",
        "incumbent_oos_max_dd_pct": round(incumbent_metrics["max_dd_pct"], 2) if incumbent_metrics else "",
        # ★追加(②モデル更新ルール、後方互換のため末尾に追加): 登録時評価を参照した場合のみ埋まる。
        "registered_eval_date": registered_eval["date"] if registered_eval else "",
        "registered_eval_trades": registered_eval["trades"] if registered_eval else "",
        "registered_eval_pf": (round(registered_eval["pf"], 3) if registered_eval and np.isfinite(registered_eval["pf"]) else ("inf" if registered_eval else "")),
        "registered_eval_max_dd_pct": round(registered_eval["max_dd_pct"], 2) if registered_eval else "",
    })

    notify(
        f"🧠 日次モデル再学習(Walk-Forward OOSゲート)｜{today}\n"
        f"OOS区間: 直近{HOLDOUT_DAYS}日｜取引{metrics['trades']}件\n"
        f"OOS PF: {metrics['pf']:.2f}｜勝率: {metrics['win_rate']:.1f}%｜最大DD: {metrics['max_dd_pct']:.1f}%\n"
        f"判定: {'✅ 本番差し替え' if deploy else '🟡 見送り'}({reason})\n"
        f"📈 {live_text}"
    )


if __name__ == "__main__":
    main()
