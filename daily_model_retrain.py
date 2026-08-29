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

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import daily_directional_top1 as trader

JST = ZoneInfo("Asia/Tokyo")

MODEL_FILE = trader.MODEL_FILE
TRAIN_FILE = trader.TRAIN_FILE
HISTORY_FILE = trader.HISTORY_FILE
FEATURES = trader.FEATURES
HOLD_DAYS = trader.HOLD_DAYS
TP_MULT = trader.TP_MULT
SL_MULT = trader.SL_MULT
ATR_TARGET_MULTIPLIER = 1.0

PREV_MODEL_FILE = "model_prev.pkl"
RETRAIN_REPORT_FILE = "daily_retrain_report.csv"

HOLDOUT_DAYS = int(os.getenv("RETRAIN_HOLDOUT_DAYS", "90"))
MIN_OOS_TRADES = int(os.getenv("RETRAIN_MIN_OOS_TRADES", "15"))
MIN_OOS_PF = float(os.getenv("RETRAIN_MIN_OOS_PF", "1.0"))
MAX_OOS_DD_PCT = float(os.getenv("RETRAIN_MAX_OOS_DD_PCT", "30.0"))
MIN_TRAIN_ROWS = int(os.getenv("RETRAIN_MIN_ROWS", "3000"))
DOWNLOAD_SLEEP = float(os.getenv("RETRAIN_DOWNLOAD_SLEEP", "0.15"))


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
    return ticker_frames


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


def simulate_oos_top1(model, ticker_frames, oos_dates):
    """OOS区間だけでTOP1を日次シミュレーションする。"""
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
                trades.append({
                    "entry_date": position["entry_date"], "exit_date": date,
                    "ticker": ticker, "direction": position["direction"],
                    "return_pct": ret, "reason": exit_reason,
                })
                position = None
            continue

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
            except Exception:
                continue

            long_s, short_s = trader.directional_score(row, up, down)
            direction = "BUY" if long_s >= short_s else "SHORT"
            score = max(long_s, short_s)
            price = float(row["Close"])
            if direction == "BUY":
                tp, sl = price + atr_abs * TP_MULT, price - atr_abs * SL_MULT
            else:
                tp, sl = price - atr_abs * TP_MULT, price + atr_abs * SL_MULT
            candidates.append({
                "ticker": ticker, "direction": direction, "score": score,
                "price": price, "tp": tp, "sl": sl,
            })

        if not candidates:
            continue
        candidates.sort(key=lambda z: z["score"], reverse=True)
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
    today = datetime.now(JST).strftime("%Y-%m-%d")
    print(f"=== 日次モデル再学習(Walk-Forward OOSゲート付き) {today} ===")

    ticker_frames = build_universe(trader.TICKERS)
    all_dates = sorted(set().union(*[set(x.index) for x in ticker_frames.values()]))
    last_date = pd.Timestamp(all_dates[-1])
    oos_cutoff = last_date - pd.Timedelta(days=HOLDOUT_DAYS)
    oos_dates = [d for d in all_dates if pd.Timestamp(d) >= oos_cutoff]
    print(f"OOS区間: {oos_cutoff.date()} 〜 {last_date.date()}（{len(oos_dates)}営業日、学習からは完全除外）")

    fit_rows = flatten_training_rows(ticker_frames, before_date=oos_cutoff)
    print(f"学習データ行数(OOS区間を除く): {len(fit_rows):,}")

    if len(fit_rows) < MIN_TRAIN_ROWS:
        msg = f"⚠ 学習データが{MIN_TRAIN_ROWS}行未満({len(fit_rows)}行)のため再学習を見送り"
        print(msg)
        trader.send(f"🟡 日次モデル再学習｜{today}\n{msg}\nmodel.pklは変更しません")
        return

    oos_model = fit_rf(fit_rows)
    oos_trades = simulate_oos_top1(oos_model, ticker_frames, oos_dates)
    metrics = compute_pf_metrics(oos_trades)
    print(
        f"OOSシミュレーション結果: 取引数={metrics['trades']} "
        f"PF={metrics['pf']:.3f} 勝率={metrics['win_rate']:.1f}% "
        f"最大DD={metrics['max_dd_pct']:.2f}%"
    )

    if metrics["trades"] < MIN_OOS_TRADES:
        deploy = True
        reason = f"OOS取引数不足({metrics['trades']}件<{MIN_OOS_TRADES}件)のため判定保留・参考採用"
    elif metrics["pf"] >= MIN_OOS_PF and abs(metrics["max_dd_pct"]) <= MAX_OOS_DD_PCT:
        deploy = True
        reason = f"OOSゲート通過(PF={metrics['pf']:.2f}>={MIN_OOS_PF}, 最大DD={metrics['max_dd_pct']:.1f}%)"
    else:
        deploy = False
        reason = f"OOSゲート未通過(PF={metrics['pf']:.2f}, 最大DD={metrics['max_dd_pct']:.1f}%)"

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
    })

    trader.send(
        f"🧠 日次モデル再学習(Walk-Forward OOSゲート)｜{today}\n"
        f"OOS区間: 直近{HOLDOUT_DAYS}日｜取引{metrics['trades']}件\n"
        f"OOS PF: {metrics['pf']:.2f}｜勝率: {metrics['win_rate']:.1f}%｜最大DD: {metrics['max_dd_pct']:.1f}%\n"
        f"判定: {'✅ 本番差し替え' if deploy else '🟡 見送り'}({reason})\n"
        f"📈 {live_text}"
    )


if __name__ == "__main__":
    main()
