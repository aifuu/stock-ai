"""
daily_model_retrain.py

毎営業日、取引開始前(JST 07:30目安)に1回だけ実行し、
daily_directional_top1.py が使う model.pkl を再学習する。

★設計方針(2026-08):
・特徴量計算は daily_directional_top1.py(=trader)の features()/rsi()/atr()/adx()
  をそのまま再利用する。ロジックを二重管理して食い違いが起きるのを防ぐため。
・ただし features() は「本日の最新行」でしか呼ばれない想定のため、
  日経225先物(future_return/future_ma5/future_rsi/future_gap)を
  0.0/50.0のプレースホルダで埋めている。学習データでは実際の
  先物値(1日ラグを入れて未来情報を使わない)を計算して上書きする。
・target(3クラス: 0=下落/1=中立/2=上昇)は walk_forward.py と同じ定義に
  揃える: 「HOLD_DAYS営業日後の騰落率」を「ATR比率×√HOLD_DAYS」を
  しきい値として3クラス化する。
・retrain候補モデルは、直近ホールドアウト期間で「これまでのmodel.pkl」と
  方向的中率を比較し、明確に悪化していない場合だけ本番差し替えする
  (悪いモデルに事故で入れ替わらないための安全ゲート)。
・directional_paper_history.csv(実際のペーパートレード結果)は、
  学習データそのものには混ぜず、"直近の実績"として daily_retrain_report.csv
  に記録し、人が継続的に監視できるようにする(重要: 精度の低い自己ラベルを
  学習に混ぜて悪循環になるのを避けるため、あえて学習には使わない)。
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
ATR_TARGET_MULTIPLIER = 1.0

PREV_MODEL_FILE = "model_prev.pkl"
RETRAIN_REPORT_FILE = "daily_retrain_report.csv"

HOLDOUT_DAYS = int(os.getenv("RETRAIN_HOLDOUT_DAYS", "60"))
MIN_ACCURACY_DROP = float(os.getenv("RETRAIN_MAX_ACC_DROP", "5.0"))
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


def build_features_with_target(ticker, nikkei, futures_df):
    """1銘柄分の特徴量とtargetを作り、全期間の学習行を返す。"""
    df = trader.download(ticker)
    if df is None or len(df) < 150:
        return None

    x = trader.features(df, nikkei)

    if futures_df is not None:
        aligned = futures_df.reindex(x.index).ffill()
        for col in ["future_return", "future_ma5", "future_rsi", "future_gap"]:
            x[col] = aligned[col]

    future_price = x["Close"].shift(-HOLD_DAYS)
    future_return = (future_price / x["Close"] - 1.0) * 100.0
    atr_threshold = x["atr_ratio"] * ATR_TARGET_MULTIPLIER * np.sqrt(HOLD_DAYS)

    x["target"] = np.select(
        [future_return <= -atr_threshold, future_return >= atr_threshold],
        [0, 2],
        default=1,
    )
    x["target_valid"] = future_price.notna()
    x["ticker"] = ticker

    x = x.dropna(subset=FEATURES)
    x = x[x["target_valid"]]
    if x.empty:
        return None

    return x[["ticker", "target"] + FEATURES].reset_index().rename(columns={"index": "date"})


def build_train_data():
    nikkei = trader.make_nikkei()
    if nikkei is None:
        raise RuntimeError("日経平均データが取得できず、学習データを作れません")

    futures_df = make_futures_features()
    frames = []

    for i, ticker in enumerate(trader.TICKERS, 1):
        part = build_features_with_target(ticker, nikkei, futures_df)
        if part is not None:
            frames.append(part)
            print(f"[{i}/{len(trader.TICKERS)}] {ticker}: {len(part):,}行")
        else:
            print(f"[{i}/{len(trader.TICKERS)}] {ticker}: スキップ(データ不足)")
        time.sleep(DOWNLOAD_SLEEP)

    if not frames:
        raise RuntimeError("有効な学習データが1件も作れませんでした")

    train_df = pd.concat(frames, ignore_index=True)
    train_df = train_df.dropna(subset=FEATURES + ["target"])
    return train_df


def directional_accuracy(model, df):
    """中立(1)を除いた行の上下方向的中率。"""
    directional = df[df["target"] != 1]
    if directional.empty:
        return None

    pred = model.predict(directional[FEATURES])
    same_direction = (
        ((pred == 2) & (directional["target"] == 2))
        | ((pred == 0) & (directional["target"] == 0))
    )
    return float(same_direction.mean() * 100.0)


def recent_live_performance(days=30):
    """ペーパートレード履歴の直近実績。学習には使用しない。"""
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
    return {
        "trades": int(len(recent)),
        "win_rate": win_rate,
        "pnl": float(recent["pnl"].sum()),
    }


def append_retrain_report(row):
    new_df = pd.DataFrame([row])
    if os.path.exists(RETRAIN_REPORT_FILE):
        old_df = pd.read_csv(RETRAIN_REPORT_FILE)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(RETRAIN_REPORT_FILE, index=False, encoding="utf-8-sig")


def main():
    today = datetime.now(JST).strftime("%Y-%m-%d")
    print(f"=== 日次モデル再学習 {today} ===")

    train_df = build_train_data()
    print(f"学習データ行数: {len(train_df):,}")

    if len(train_df) < MIN_TRAIN_ROWS:
        msg = f"⚠ 学習データが{MIN_TRAIN_ROWS}行未満({len(train_df)}行)のため再学習を見送り"
        print(msg)
        trader.send(f"🟡 日次モデル再学習｜{today}\n{msg}\nmodel.pklは変更しません")
        return

    train_df["date"] = pd.to_datetime(train_df["date"])
    cutoff = train_df["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    fit_part = train_df[train_df["date"] < cutoff]
    holdout_part = train_df[train_df["date"] >= cutoff]

    if len(fit_part) < MIN_TRAIN_ROWS or holdout_part.empty:
        print("⚠ ホールドアウト分割後のデータが不十分。全期間で学習のみ行い、精度比較をスキップします")
        fit_part, holdout_part = train_df, None

    new_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    new_model.fit(fit_part[FEATURES], fit_part["target"].astype(int))

    new_acc = directional_accuracy(new_model, holdout_part) if holdout_part is not None else None
    old_acc = None

    if holdout_part is not None and os.path.exists(MODEL_FILE):
        try:
            old_model = joblib.load(MODEL_FILE)
            if np.array_equal(old_model.classes_, np.array([0, 1, 2])):
                old_acc = directional_accuracy(old_model, holdout_part)
        except Exception as e:
            print(f"⚠ 既存model.pkl読み込み失敗(比較なしで続行): {e}")

    deploy = True
    reason = "初回学習またはホールドアウト比較なし"

    if new_acc is not None and old_acc is not None:
        drop = old_acc - new_acc
        if drop > MIN_ACCURACY_DROP:
            deploy = False
            reason = f"方向的中率が{drop:.1f}pt悪化({old_acc:.1f}%→{new_acc:.1f}%)のため見送り"
        else:
            reason = f"方向的中率 {old_acc:.1f}%→{new_acc:.1f}%（悪化{drop:+.1f}pt、許容範囲内）"

    # 学習データはモデル差し替えの有無に関係なく更新する。
    train_df.drop(columns=["date"]).to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")

    if deploy:
        if os.path.exists(MODEL_FILE):
            try:
                os.replace(MODEL_FILE, PREV_MODEL_FILE)
            except Exception as e:
                print(f"⚠ 旧model.pklの退避に失敗(続行): {e}")

        final_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        final_model.fit(train_df[FEATURES], train_df["target"].astype(int))
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
        "train_rows": len(train_df),
        "holdout_days": HOLDOUT_DAYS if holdout_part is not None else 0,
        "new_model_directional_acc": round(new_acc, 2) if new_acc is not None else "",
        "old_model_directional_acc": round(old_acc, 2) if old_acc is not None else "",
        "deployed": deploy,
        "reason": reason,
        "live_recent_trades": live["trades"] if live else 0,
        "live_recent_win_rate": round(live["win_rate"], 2) if live and live["win_rate"] is not None else "",
        "live_recent_pnl": round(live["pnl"], 2) if live else 0.0,
    })

    acc_text = (
        f"方向的中率: 旧{old_acc:.1f}% → 新{new_acc:.1f}%"
        if new_acc is not None and old_acc is not None
        else "方向的中率: 比較データ不足"
    )
    trader.send(
        f"🧠 日次モデル再学習｜{today}\n"
        f"学習データ: {len(train_df):,}行\n"
        f"{acc_text}\n"
        f"判定: {'✅ 本番差し替え' if deploy else '🟡 見送り'}({reason})\n"
        f"📈 {live_text}"
    )


if __name__ == "__main__":
    main()
