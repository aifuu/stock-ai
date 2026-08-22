import os
import warnings

import yfinance as yf
import pandas as pd
import numpy as np
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

warnings.filterwarnings("ignore")


# =========================================================
# Discord
# =========================================================

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send_discord(msg):
    if not WEBHOOK_URL:
        print("❌ Webhookなし(DISCORD_WEBHOOK未設定)")
        return

    if len(msg) > 1900:
        msg = msg[:1900] + "\n...(省略)"

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"content": msg},
            timeout=30
        )
        print("Discord status =", response.status_code)

        if response.status_code == 204:
            print("✅ Discord送信成功")
        else:
            print("❌ Discord送信失敗")
            print(response.text)

    except Exception as e:
        print("❌ Discord送信エラー:", e)


# =========================================================
# 設定
# =========================================================

TICKERS = [
    "7203.T", "7269.T", "285A.T", "9984.T", "4980.T",
    "8031.T", "8058.T", "9509.T", "9501.T", "8362.T",
    "8306.T", "5803.T", "6526.T", "6613.T",
]

COMPANY_NAMES = {
    "7203.T": "トヨタ自動車",
    "7269.T": "スズキ",
    "285A.T": "キオクシアHD",
    "9984.T": "ソフトバンクG",
    "4980.T": "デクセリアルズ",
    "8031.T": "三井物産",
    "8058.T": "三菱商事",
    "9509.T": "北海道電力",
    "9501.T": "東京電力HD",
    "8362.T": "福井銀行",
    "8306.T": "三菱UFJ",
    "5803.T": "フジクラ",
    "6526.T": "ソシオネクスト",
    "6613.T": "QDレーザ",
}

DATA_PERIOD = "5y"

TRAIN_DAYS = 500
RETRAIN_DAYS = 20
FORWARD_DAYS = 3
MAX_HOLD_DAYS = 5

DOWN_THRESHOLD = -1.5
UP_THRESHOLD = 1.5

TAKE_PROFIT = 0.08
STOP_LOSS = 0.04

MIN_AVG_VOLUME = 100000
MIN_DATA_REQUIRED = 600

# =========================================================
# 【運用上の重要な注意】買い推奨のしきい値(基準)
#
# buy_signal = (prediction == 2) and (up_prob >= BUY_UP_PROB_MIN)
#
# この BUY_UP_PROB_MIN = 0.0 は、今回のフィルター比較
# (PROBABILITY_THRESHOLDS)における「0% = 現状基準」の
# 意味も兼ねている。つまりこの値自体は、フィルター比較の
# 結果を見て直接書き換えるパラメータではない。
#
# 「実際にどのUP確率しきい値を使うべきか」は、
# PROBABILITY_THRESHOLDS の比較結果(BACKTEST側のみ)を
# 見たうえで、buy_signal の計算式を使う側(例:
# stock_scan.py の買い推奨ロジック)で個別に決定すること。
# =========================================================

BUY_UP_PROB_MIN = 0.0

# =========================================================
# 【完全OOS 運用ルール】
#
# OOS期間は「最終確認用」として扱う。
#
# 重要:
# OOSの結果を見て、以下を変更してはいけない。
#
# ・TAKE_PROFIT
# ・STOP_LOSS
# ・TRAIN_DAYS
# ・RETRAIN_DAYS
# ・FEATURES
# ・BUY_UP_PROB_MIN
# ・PROBABILITY_THRESHOLDS(今回の比較閾値そのもの)
# ・モデル構造
# ・その他売買ロジック
#
# これらをOOS結果を見ながら調整すると、
# そのOOS期間は実質的に「検証済みデータ」になる。
#
# パラメータ調整・閾値選定はBACKTEST期間の結果だけを
# 使う。OOSは最終確認として使用する。
#
# OOSを見て調整した場合、そのOOSは「消費済み」と考え、
# 次に新しいデータが蓄積されるまで真のOOS評価は行わない。
#
# 運用フロー:
#   ① BACKTESTを見る → 改良する
#   ② BACKTESTで納得できるところまで調整
#   ③ OOSを見る
#   ④ OOSが良ければ、初めて実戦投入を検討
#
# 本スクリプトは比較結果を表示するだけで、
# 自動で最適な閾値を選択する処理は一切行わない。
# =========================================================

OOS_DAYS = 90

# =========================================================
# 相場局面
# =========================================================

REGIME_MA_SHORT = 25
REGIME_MA_LONG = 75
REGIME_VOL_WINDOW = 20
REGIME_HIGH_VOL_PERCENTILE = 0.85
REGIME_TREND_BAND = 0.01

# =========================================================
# フォワードリターン
# =========================================================

FORWARD_HORIZONS = (30, 60, 90)

# =========================================================
# 今回比較するUP確率閾値
# 0% = 現状の基準(BUY_UP_PROB_MINと同じ意味)
# =========================================================

PROBABILITY_THRESHOLDS = [0, 40, 50, 60, 70]


# =========================================================
# 特徴量
# =========================================================

FEATURES = [
    "ret1",
    "ma25",
    "ma75",
    "vol_ratio",
    "rsi",
    "adx",
    "macd",
    "signal",
    "from_high",
    "from_low",
    "atr_ratio",
    "relative_strength",
    "bb_position",
    "bb_width",
    "obv_change",
    "nikkei_kairi25",
    "nikkei_rsi",
    "nikkei_macd",
    "nikkei_return_5d",
    "future_return",
    "future_ma5",
    "future_rsi",
    "future_gap",
]


# =========================================================
# 安全なデータ取得
# =========================================================

def safe_download(ticker, retries=3, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, **kwargs)

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df

            print(f"{ticker} 空データ (試行 {attempt}/{retries})")

        except Exception as e:
            print(f"{ticker} 取得失敗 (試行 {attempt}/{retries}): {e}")

    return None


# =========================================================
# RSI
# =========================================================

def calc_rsi(close, period=14):
    close = close.squeeze()
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(100)


# =========================================================
# ADX
# =========================================================

def calc_adx(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_dm_sm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_sm / atr.replace(0, np.nan)

    dx = (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    ) * 100

    return dx.ewm(alpha=1 / period, adjust=False).mean()


# =========================================================
# ATR
# =========================================================

def calc_atr(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


# =========================================================
# 株価特徴量
# =========================================================

def create_features(df):
    df = df.copy()

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["ret1"] = close.pct_change()

    df["ma25"] = close.rolling(25).mean()
    df["ma75"] = close.rolling(75).mean()

    df["vol_ratio"] = volume / volume.rolling(20).mean()

    df["rsi"] = calc_rsi(close)
    df["adx"] = calc_adx(df)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    high252 = close.rolling(252).max()
    low252 = close.rolling(252).min()

    df["from_high"] = (close / high252 - 1) * 100
    df["from_low"] = (close / low252 - 1) * 100

    atr = calc_atr(df)

    df["atr_ratio"] = atr / close

    df["_stock_ret5"] = close.pct_change(5)

    bb_ma20 = close.rolling(20).mean()
    bb_std20 = close.rolling(20).std()

    bb_upper = bb_ma20 + bb_std20 * 2
    bb_lower = bb_ma20 - bb_std20 * 2

    band_width = bb_upper - bb_lower

    df["bb_position"] = (close - bb_lower) / band_width.replace(0, np.nan)
    df["bb_width"] = band_width / bb_ma20.replace(0, np.nan) * 100

    direction = np.sign(close.diff())

    obv = (volume * direction).fillna(0).cumsum()

    df["obv_change"] = obv.diff(5) / volume.rolling(5).sum() * 100

    return df


# =========================================================
# 日経特徴量
# =========================================================

def create_nikkei_features(nikkei):
    nikkei = nikkei.copy()

    close = nikkei["Close"].squeeze()

    ma25 = close.rolling(25).mean()

    nikkei["nikkei_kairi25"] = (close - ma25) / ma25.replace(0, np.nan) * 100

    nikkei["nikkei_rsi"] = calc_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    nikkei["nikkei_macd"] = ema12 - ema26

    nikkei["nikkei_return_5d"] = close.pct_change(5) * 100
    nikkei["nikkei_ret5_raw"] = close.pct_change(5)

    return nikkei


# =========================================================
# 日経225先物
# =========================================================

def create_future_features(futures):
    futures = futures.copy()

    close = futures["Close"].squeeze()

    futures["future_return"] = close.pct_change()
    futures["future_ma5"] = close.rolling(5).mean()
    futures["future_rsi"] = calc_rsi(close)

    futures["future_gap"] = (close - close.shift(1)) / close.shift(1)

    # 翌日の予測時点で未来の情報にならないよう1日シフト
    for col in ["future_return", "future_ma5", "future_rsi", "future_gap"]:
        futures[col] = futures[col].shift(1)

    return futures


# =========================================================
# ターゲット
# =========================================================

def create_target(df):
    close = df["Close"].squeeze()

    future_price = close.shift(-FORWARD_DAYS)

    future_return = (future_price / close - 1) * 100

    target = np.select(
        [
            future_return <= DOWN_THRESHOLD,
            future_return >= UP_THRESHOLD,
        ],
        [0, 2],
        default=1
    )

    return pd.Series(target, index=df.index), future_return


# =========================================================
# 複数ホライズン
# =========================================================

def create_multi_horizon_returns(df, horizons=FORWARD_HORIZONS):
    close = df["Close"].squeeze()

    out = {}

    for h in horizons:
        future_price = close.shift(-h)
        out[f"fwd_return_{h}d"] = (future_price / close - 1) * 100

    return pd.DataFrame(out, index=df.index)


# =========================================================
# 相場局面
#
# ※ ボラティリティ順位(vol_rank)は5年分全体を通した
#   事後的な順位づけのため、「結果の分類・集計用」に
#   限定して使う。学習特徴量・売買判定には一切使わない
#   (取引ロジックへのリークではない)。
#   リアルタイムの相場フィルターとして使う場合は、
#   ローリング順位に作り直す必要がある。
# =========================================================

def classify_regime(nikkei):
    close = nikkei["Close"].squeeze()

    ma_short = close.rolling(REGIME_MA_SHORT).mean()
    ma_long = close.rolling(REGIME_MA_LONG).mean()

    daily_ret = close.pct_change()

    vol = daily_ret.rolling(REGIME_VOL_WINDOW).std() * 100

    # 注: これは「結果の分類・集計用」。学習特徴量・予測値には使わない。
    vol_rank = vol.rank(pct=True)

    trend_diff = (ma_short - ma_long) / ma_long.replace(0, np.nan)

    regime = pd.Series("レンジ", index=nikkei.index, dtype=object)

    regime[trend_diff >= REGIME_TREND_BAND] = "上昇"
    regime[trend_diff <= -REGIME_TREND_BAND] = "下落"
    regime[vol_rank >= REGIME_HIGH_VOL_PERCENTILE] = "急落・高ボラ"

    return regime


# =========================================================
# 流動性
# =========================================================

def liquidity_ok(df):
    volume = df["Volume"].squeeze()
    avg_volume = volume.tail(20).mean()

    return avg_volume >= MIN_AVG_VOLUME


# =========================================================
# 売買結果
# =========================================================

def evaluate_trade(df, entry_index, entry_price):
    future = df.iloc[entry_index + 1: entry_index + 1 + MAX_HOLD_DAYS]

    if len(future) == 0:
        return None

    take_profit = entry_price * (1 + TAKE_PROFIT)
    stop_loss = entry_price * (1 - STOP_LOSS)

    for day_number, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["High"])
        low = float(row["Low"])

        # 同日TP/SL両方に触れた場合は保守的にLOSS
        if low <= stop_loss and high >= take_profit:
            return {"result": "LOSS", "return": -STOP_LOSS * 100, "hold_days": day_number}

        if high >= take_profit:
            return {"result": "WIN", "return": TAKE_PROFIT * 100, "hold_days": day_number}

        if low <= stop_loss:
            return {"result": "LOSS", "return": -STOP_LOSS * 100, "hold_days": day_number}

    last_close = float(future.iloc[-1]["Close"])
    return_rate = (last_close / entry_price - 1) * 100

    result = "TIMEOUT_LOSS" if return_rate < 0 else "HOLD"

    return {"result": result, "return": return_rate, "hold_days": len(future)}


# =========================================================
# ウォークフォワード
# =========================================================

def run_walk_forward(ticker, df):

    print("")
    print("=" * 60)
    print(f"🚀 ウォークフォワード開始 {ticker} {COMPANY_NAMES.get(ticker, '')}")
    print("=" * 60)

    df = df.copy()

    target, future_return = create_target(df)

    df["target"] = target
    df["_future_return"] = future_return

    df = df.join(create_multi_horizon_returns(df))

    if len(df) < MIN_DATA_REQUIRED:
        print(f"{ticker}: データ不足 {len(df)}日")
        return []

    results = []

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    start = TRAIN_DAYS

    last_valid_index = len(df) - FORWARD_DAYS - MAX_HOLD_DAYS

    prediction_count = 0
    current = start

    while current < last_valid_index:

        train_start = current - TRAIN_DAYS
        train_end = current

        test_start = current
        test_end = min(current + RETRAIN_DAYS, last_valid_index)

        # -------------------------------------------------
        # 未来のtargetを学習に混ぜない
        # -------------------------------------------------

        safe_train_end = train_end - FORWARD_DAYS

        if safe_train_end <= train_start:
            current = test_end
            continue

        train_df = df.iloc[train_start:safe_train_end].copy()

        train_df = train_df.dropna(subset=FEATURES + ["target"])

        if len(train_df) < 100:
            current = test_end
            continue

        if train_df["target"].nunique() < 3:
            print(f"{ticker}: {df.index[current].date()} 3クラス不足")
            current = test_end
            continue

        X_train = train_df[FEATURES]
        y_train = train_df["target"].astype(int)

        model.fit(X_train, y_train)

        print(
            f"{ticker} | "
            f"学習 {df.index[train_start].date()} "
            f"～ {df.index[safe_train_end - 1].date()} "
            f"| 予測 {df.index[test_start].date()} "
            f"～ {df.index[test_end - 1].date()}"
        )

        for i in range(test_start, test_end):

            row = df.iloc[i]

            if row[FEATURES].isna().any():
                continue

            X_test = row[FEATURES].to_frame().T

            try:
                probabilities = model.predict_proba(X_test)[0]
                classes = list(model.classes_)

                down_prob = probabilities[classes.index(0)] if 0 in classes else 0
                flat_prob = probabilities[classes.index(1)] if 1 in classes else 0
                up_prob = probabilities[classes.index(2)] if 2 in classes else 0

                prediction = int(model.predict(X_test)[0])

            except Exception as e:
                print(f"{ticker} 予測失敗: {e}")
                continue

            actual = int(df.iloc[i]["target"])

            entry_price = float(df.iloc[i]["Close"])

            trade = evaluate_trade(df, i, entry_price)

            if trade is None:
                continue

            # 基準となる買いシグナル(0%なのでprediction==2なら対象)
            buy_signal = (
                prediction == 2
                and up_prob >= BUY_UP_PROB_MIN
            )

            # -------------------------------------------------
            # OOS / BACKTEST
            # -------------------------------------------------

            phase = "OOS" if i >= len(df) - OOS_DAYS else "BACKTEST"

            result_row = {
                "date": df.index[i].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "company": COMPANY_NAMES.get(ticker, ""),
                "phase": phase,

                "prediction": prediction,
                "actual": actual,
                "correct": int(prediction == actual),

                "down_probability": round(down_prob * 100, 2),
                "flat_probability": round(flat_prob * 100, 2),
                "up_probability": round(up_prob * 100, 2),

                "buy_signal": int(buy_signal),

                "price": round(entry_price, 2),

                "result": trade["result"],
                "return": round(trade["return"], 2),
                "hold_days": trade["hold_days"],

                "train_start": df.index[train_start].strftime("%Y-%m-%d"),
                "train_end": df.index[safe_train_end - 1].strftime("%Y-%m-%d"),
            }

            for h in FORWARD_HORIZONS:
                val = df.iloc[i][f"fwd_return_{h}d"]
                result_row[f"fwd_return_{h}d"] = (
                    round(float(val), 2) if pd.notna(val) else None
                )

            results.append(result_row)
            prediction_count += 1

        current = test_end

    print(f"✅ {ticker}: {prediction_count}件の予測完了")

    return results


# =========================================================
# 資産曲線
# =========================================================

def build_equity_curve(df, group_col="date"):
    if df.empty:
        return pd.Series(dtype=float), 0.0

    daily = df.groupby(group_col)["return"].mean().sort_index()

    equity = (1 + daily / 100).cumprod()

    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_drawdown = drawdown.min() * 100

    return equity, max_drawdown


# =========================================================
# 統計
# =========================================================

def compute_stats(df):

    if df is None or len(df) == 0:
        return None

    total = len(df)

    wins = (df["result"] == "WIN").sum()
    losses = df["result"].isin(["LOSS", "TIMEOUT_LOSS"]).sum()
    holds = (df["result"] == "HOLD").sum()

    decided = wins + losses
    win_rate = wins / decided * 100 if decided > 0 else 0

    profit_returns = df.loc[df["return"] > 0, "return"]
    loss_returns = df.loc[df["return"] < 0, "return"]

    avg_profit = profit_returns.mean() if len(profit_returns) > 0 else 0
    avg_loss = loss_returns.mean() if len(loss_returns) > 0 else 0

    gross_profit = profit_returns.sum()
    gross_loss = abs(loss_returns.sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    # 期待値(1トレードあたりの平均損益%) = return列の単純平均
    expectancy = df["return"].mean()

    equity, max_drawdown = build_equity_curve(df)

    if len(equity) > 0:
        final_equity = float(equity.iloc[-1])
        cumulative_return = (final_equity - 1) * 100
    else:
        final_equity = 1.0
        cumulative_return = 0.0

    return {
        "total": total,
        "wins": int(wins),
        "losses": int(losses),
        "holds": int(holds),
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "cumulative_return": cumulative_return,
        "equity_curve": equity,
    }


# =========================================================
# 1行統計
# =========================================================

def print_stats_oneline(label, stats):
    if stats is None:
        print(f"{label:30s} : データなし")
        return

    pf_text = "―" if np.isnan(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"

    print(
        f"{label:30s} : "
        f"件数={stats['total']:4d} "
        f"勝率={stats['win_rate']:5.1f}% "
        f"期待値={stats['expectancy']:+.2f}% "
        f"PF={pf_text:>5s} "
        f"DD={stats['max_drawdown']:6.1f}% "
        f"累積={stats['cumulative_return']:+7.1f}%"
    )


# =========================================================
# 詳細統計
# =========================================================

def print_stats(title, stats):

    print("")
    print("=" * 60)
    print(title)
    print("=" * 60)

    if stats is None:
        print("データなし")
        return

    pf_text = "―" if np.isnan(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"

    print(f"件数               : {stats['total']}")
    print(f"WIN                : {stats['wins']}")
    print(f"LOSS               : {stats['losses']}")
    print(f"HOLD               : {stats['holds']}")
    print(f"勝率               : {stats['win_rate']:.2f}%")
    print(f"平均利益           : +{stats['avg_profit']:.2f}%")
    print(f"平均損失           : {stats['avg_loss']:.2f}%")
    print(f"期待値(1トレード)  : {stats['expectancy']:+.2f}%")
    print(f"PF                 : {pf_text}")
    print(f"最大DD             : {stats['max_drawdown']:.2f}%")
    print(f"累積リターン       : {stats['cumulative_return']:+.2f}%")
    print(f"最終資産倍率       : {stats['final_equity']:.4f}")


# =========================================================
# フィルター比較
#
# ここが今回の追加部分。
#
# 基準:
#   prediction == 2
#
# 確率:
#   up_probability >= 0/40/50/60/70
#
# 上昇相場フィルター:
#   regime == "上昇"
#
# 自動最適化はしない。
#
# 【修正】「上昇相場のみ」はUP確率のしきい値に依存しない
# 統計のため、しきい値のループの外で1回だけ計算する。
# (ループ内に置くと、閾値の数だけ全く同じ統計が重複して
#  comparison_df に記録され、「閾値によって上昇相場のみの
#  数字が変わった」ように誤読される原因になっていた)
# =========================================================

def build_filter_comparison(result_df, thresholds=PROBABILITY_THRESHOLDS):

    rows = []

    def add_rows(threshold, filter_name, selected):

        if len(selected) == 0:
            return

        for phase in ["ALL", "BACKTEST", "OOS"]:

            if phase == "ALL":
                phase_df = selected
            else:
                phase_df = selected[selected["phase"] == phase]

            stats = compute_stats(phase_df)

            if stats is None:
                continue

            rows.append({
                "threshold": threshold,
                "filter": filter_name,
                "phase": phase,
                "total": stats["total"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "holds": stats["holds"],
                "win_rate": round(stats["win_rate"], 4),
                "expectancy": round(stats["expectancy"], 4),
                "profit_factor": (
                    round(stats["profit_factor"], 4)
                    if not np.isnan(stats["profit_factor"])
                    else np.nan
                ),
                "max_drawdown": round(stats["max_drawdown"], 4),
                "cumulative_return": round(stats["cumulative_return"], 4),
                "final_equity": round(stats["final_equity"], 6),
            })

    # 基本のAI買い条件(prediction == 2)
    base = result_df[result_df["prediction"] == 2].copy()

    # -------------------------------------------------
    # 「上昇相場のみ」は閾値非依存 → 1回だけ計算
    # threshold は None(「閾値なし」を示す)として記録
    # -------------------------------------------------
    uptrend_only_df = base[base["regime"] == "上昇"].copy()
    add_rows(None, "上昇相場のみ", uptrend_only_df)

    for threshold in thresholds:

        prob_df = base[base["up_probability"] >= threshold].copy()

        uptrend_df = prob_df[prob_df["regime"] == "上昇"].copy()

        add_rows(threshold, "確率のみ", prob_df)
        add_rows(threshold, "上昇相場+確率", uptrend_df)

    return pd.DataFrame(rows)


def _format_threshold_label(threshold):

    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        return "条件なし"

    return f"UP≥{int(threshold):2d}%"


# =========================================================
# 比較結果表示
# =========================================================

def print_filter_comparison(comparison_df):

    print("")
    print("=" * 110)
    print("🔬 AI買い条件フィルター比較")
    print("=" * 110)

    print("※ 0% = 現状基準。 自動で最適値は選択しません。")
    print("※「上昇相場のみ」はUP確率と無関係のため、条件は1種類のみ表示されます。")

    for filter_name in ["確率のみ", "上昇相場のみ", "上昇相場+確率"]:

        print("")
        print(f"【{filter_name}】")

        for phase in ["BACKTEST", "OOS"]:

            print(f"\n--- {phase} ---")

            subset = comparison_df[
                (comparison_df["filter"] == filter_name)
                & (comparison_df["phase"] == phase)
            ].copy()

            if subset.empty:
                print("データなし")
                continue

            for _, row in subset.iterrows():

                label = _format_threshold_label(row["threshold"])

                pf_val = row["profit_factor"]
                pf_text = "―" if pd.isna(pf_val) else f"{pf_val:.2f}"

                print(
                    f"{label} | "
                    f"件数={int(row['total']):4d} | "
                    f"勝率={row['win_rate']:5.1f}% | "
                    f"期待値={row['expectancy']:+.2f}% | "
                    f"PF={pf_text} | "
                    f"DD={row['max_drawdown']:.1f}% | "
                    f"累積={row['cumulative_return']:+.1f}%"
                )


# =========================================================
# Discord用比較
# =========================================================

def format_filter_comparison_for_discord(comparison_df):

    lines = [
        "🔬 フィルター比較",
        "※0%=現状基準 / 自動最適化なし"
    ]

    for phase in ["BACKTEST", "OOS"]:

        lines.append(f"\n【{phase}】")

        for filter_name in ["確率のみ", "上昇相場のみ", "上昇相場+確率"]:

            subset = comparison_df[
                (comparison_df["filter"] == filter_name)
                & (comparison_df["phase"] == phase)
            ]

            if subset.empty:
                continue

            lines.append(f"■ {filter_name}")

            for _, row in subset.iterrows():

                label = _format_threshold_label(row["threshold"])

                pf_val = row["profit_factor"]
                pf_text = "―" if pd.isna(pf_val) else f"{pf_val:.2f}"

                lines.append(
                    f"{label} "
                    f"件{int(row['total'])} "
                    f"勝率{row['win_rate']:.0f}% "
                    f"期待{row['expectancy']:+.2f}% "
                    f"PF{pf_text} "
                    f"DD{row['max_drawdown']:.0f}%"
                )

    return "\n".join(lines)


# =========================================================
# サマリーCSV
# =========================================================

def save_summary_csv(
    overall_stats,
    phase_stats,
    regime_stats,
    rank1_stats,
    buy_stats,
    comparison_df
):

    rows = []

    def add_row(label, stats):

        if stats is None:
            rows.append({"group": label, "data": "なし"})
            return

        row = {"group": label}

        for key, value in stats.items():
            if key == "equity_curve":
                continue
            row[key] = value

        rows.append(row)

    add_row("全体", overall_stats)

    for label in ["BACKTEST", "OOS"]:
        add_row(f"フェーズ_{label}", phase_stats.get(label))

    for label in ["上昇", "下落", "レンジ", "急落・高ボラ"]:
        add_row(f"相場局面_{label}", regime_stats.get(label))

    add_row("ランク1", rank1_stats)
    add_row("買い推奨", buy_stats)

    pd.DataFrame(rows).to_csv(
        "backtest_summary.csv", index=False, encoding="utf-8-sig"
    )

    comparison_df.to_csv(
        "filter_comparison.csv", index=False, encoding="utf-8-sig"
    )

    print("✅ backtest_summary.csv 保存")
    print("✅ filter_comparison.csv 保存")


# =========================================================
# フォワードリターン
# =========================================================

def format_forward_return_summary(df, horizons=FORWARD_HORIZONS):

    subset = df[df["buy_signal"] == 1]

    lines = ["\n📅 フォワードリターン"]

    if len(subset) == 0:
        lines.append("データなし")
        return "\n".join(lines)

    for h in horizons:

        col = f"fwd_return_{h}d"

        if col not in subset.columns:
            continue

        vals = subset[col].dropna()

        if len(vals) > 0:
            lines.append(f"{h}日: {vals.mean():+.2f}% (件数{len(vals)})")

    return "\n".join(lines)


# =========================================================
# メイン
# =========================================================

def main():

    print("")
    print("=" * 70)
    print("📊 ウォークフォワード・バックテスト")
    print("🔬 フィルター比較版")
    print("=" * 70)

    print(f"学習期間: {TRAIN_DAYS}")
    print(f"再学習間隔: {RETRAIN_DAYS}")
    print(f"予測期間: {FORWARD_DAYS}")
    print(f"TP: +{TAKE_PROFIT * 100:.1f}%")
    print(f"SL: -{STOP_LOSS * 100:.1f}%")
    print(f"OOS: 直近{OOS_DAYS}営業日")
    print(f"比較閾値: {PROBABILITY_THRESHOLDS}")

    # =====================================================
    # 日経
    # =====================================================

    print("\n📥 日経平均取得")

    nikkei = safe_download(
        "^N225", period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False
    )

    if nikkei is None:
        print("❌ 日経平均取得失敗")
        return

    regime_series = classify_regime(nikkei)

    nikkei = create_nikkei_features(nikkei)

    # =====================================================
    # 先物
    # =====================================================

    print("\n📥 日経225先物取得")

    futures = safe_download(
        "NIY=F", period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False
    )

    if futures is None:
        print("⚠ 先物取得失敗")
        futures = pd.DataFrame()
    else:
        futures = create_future_features(futures)

    # =====================================================
    # 全銘柄
    # =====================================================

    all_results = []
    skipped_liquidity = []

    for ticker in TICKERS:

        print("")
        print(f"📥 データ取得: {ticker}")

        stock = safe_download(
            ticker, period=DATA_PERIOD, interval="1d",
            auto_adjust=False, progress=False
        )

        # 【修正】取得失敗時のログを復元(以前のバージョンで
        # 抜け落ちていた。どの銘柄が失敗したか追えないと
        # デバッグしづらいため)
        if stock is None:
            print(f"{ticker}: データ取得失敗")
            continue

        if len(stock) < MIN_DATA_REQUIRED:
            print(f"{ticker}: データ不足 {len(stock)}日")
            continue

        if not liquidity_ok(stock):
            avg_volume = stock["Volume"].squeeze().tail(20).mean()
            print(f"⚠ {ticker}: 流動性不足 平均出来高={avg_volume:,.0f}")
            skipped_liquidity.append(ticker)
            continue

        stock = create_features(stock)

        stock = stock.join(
            nikkei[[
                "nikkei_kairi25",
                "nikkei_rsi",
                "nikkei_macd",
                "nikkei_return_5d",
                "nikkei_ret5_raw",
            ]],
            how="left"
        )

        if not futures.empty:
            stock = stock.join(
                futures[[
                    "future_return",
                    "future_ma5",
                    "future_rsi",
                    "future_gap",
                ]],
                how="left"
            )
        else:
            for col in ["future_return", "future_ma5", "future_rsi", "future_gap"]:
                stock[col] = np.nan

        stock["relative_strength"] = (
            stock["_stock_ret5"] - stock["nikkei_ret5_raw"]
        )

        stock = stock.dropna(subset=FEATURES + ["Close"])

        if len(stock) < MIN_DATA_REQUIRED:
            print(f"{ticker}: 特徴量作成後データ不足")
            continue

        ticker_results = run_walk_forward(ticker, stock)

        all_results.extend(ticker_results)

    if not all_results:
        print("❌ 結果なし")
        return

    result_df = pd.DataFrame(all_results)

    # =====================================================
    # 相場局面
    # =====================================================

    regime_lookup = {
        idx.strftime("%Y-%m-%d"): val
        for idx, val in regime_series.items()
    }

    result_df["regime"] = result_df["date"].map(regime_lookup)

    result_df.to_csv(
        "walk_forward_results.csv", index=False, encoding="utf-8-sig"
    )

    print("\n✅ walk_forward_results.csv 保存")

    # =====================================================
    # Accuracy / Precision
    # =====================================================

    accuracy = result_df["correct"].mean() * 100

    try:
        precision = precision_score(
            result_df["actual"],
            result_df["prediction"],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ) * 100
    except Exception:
        precision = 0

    print("")
    print("=" * 60)
    print("📊 ウォークフォワード結果")
    print("=" * 60)

    print(f"予測件数: {len(result_df)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")

    # =====================================================
    # 全体
    # =====================================================

    overall_stats = compute_stats(result_df)

    # =====================================================
    # ランク1
    # =====================================================

    rank1_idx = result_df.groupby("date")["up_probability"].idxmax()

    rank1_df = result_df.loc[rank1_idx].copy()

    rank1_stats = compute_stats(rank1_df)

    # =====================================================
    # 現状の買い推奨
    # =====================================================

    buy_df = result_df[result_df["buy_signal"] == 1].copy()

    buy_stats = compute_stats(buy_df)

    # =====================================================
    # 相場局面
    # =====================================================

    regime_stats = {}

    for label in ["上昇", "下落", "レンジ", "急落・高ボラ"]:
        regime_stats[label] = compute_stats(result_df[result_df["regime"] == label])

    # =====================================================
    # BT / OOS
    # =====================================================

    phase_stats = {}

    for label in ["BACKTEST", "OOS"]:
        phase_stats[label] = compute_stats(result_df[result_df["phase"] == label])

    print_stats("📊 全体", overall_stats)
    print_stats("🏆 ランク1", rank1_stats)
    print_stats("🟢 現状買い推奨", buy_stats)

    print("")
    print("=" * 70)
    print("🧪 BACKTEST vs OOS")
    print("=" * 70)

    print_stats_oneline("BACKTEST", phase_stats.get("BACKTEST"))
    print_stats_oneline("OOS", phase_stats.get("OOS"))

    print("")
    print("=" * 70)
    print("🌦 相場局面")
    print("=" * 70)

    for label in ["上昇", "下落", "レンジ", "急落・高ボラ"]:
        print_stats_oneline(label, regime_stats.get(label))

    # =====================================================
    # ★今回のメイン比較
    # =====================================================

    comparison_df = build_filter_comparison(result_df, PROBABILITY_THRESHOLDS)

    print_filter_comparison(comparison_df)

    # =====================================================
    # 銘柄別
    # =====================================================

    print("")
    print("=" * 70)
    print("📊 銘柄別成績")
    print("=" * 70)

    for ticker, group in result_df.groupby("ticker"):

        ticker_accuracy = group["correct"].mean() * 100

        wins = (group["result"] == "WIN").sum()
        losses = group["result"].isin(["LOSS", "TIMEOUT_LOSS"]).sum()
        decided = wins + losses

        win_rate = wins / decided * 100 if decided > 0 else 0

        print(
            f"{ticker:8s} "
            f"{COMPANY_NAMES.get(ticker, ''):10s} "
            f"Accuracy={ticker_accuracy:5.1f}% "
            f"勝率={win_rate:5.1f}% "
            f"平均={group['return'].mean():+.2f}% "
            f"件数={len(group)}"
        )

    # =====================================================
    # 資産曲線
    # =====================================================

    overall_stats["equity_curve"].to_csv(
        "portfolio_equity_curve.csv", header=["equity"], encoding="utf-8-sig"
    )

    # 【修正】rank1_statsがNoneになる可能性は理論上低いが、
    # 念のためbuy_statsと同様にNoneチェックを入れて安全に保存する
    if rank1_stats is not None:
        rank1_stats["equity_curve"].to_csv(
            "rank1_equity_curve.csv", header=["equity"], encoding="utf-8-sig"
        )

    if buy_stats is not None:
        buy_stats["equity_curve"].to_csv(
            "buy_equity_curve.csv", header=["equity"], encoding="utf-8-sig"
        )

    # =====================================================
    # CSV
    # =====================================================

    save_summary_csv(
        overall_stats,
        phase_stats,
        regime_stats,
        rank1_stats,
        buy_stats,
        comparison_df
    )

    # =====================================================
    # フォワード
    # =====================================================

    forward_summary_text = format_forward_return_summary(result_df)

    print("")
    print(forward_summary_text)

    # =====================================================
    # Discord
    # =====================================================

    def _pf_text(stats):
        if stats is None or np.isnan(stats["profit_factor"]):
            return "―"
        return f"{stats['profit_factor']:.2f}"

    overall_pf_text = _pf_text(overall_stats)

    discord = (
        "📊 AI WALK FORWARD BACKTEST\n"
        "━━━━━━━━━━━━━━━━\n"
        f"期間: {result_df['date'].min()} ～ {result_df['date'].max()}\n\n"
        f"最終資産倍率: {overall_stats['final_equity']:.4f}\n"
        f"累積: {overall_stats['cumulative_return']:+.2f}%\n"
        f"期待値: {overall_stats['expectancy']:+.2f}%\n"
        f"PF: {overall_pf_text}\n"
        f"最大DD: {overall_stats['max_drawdown']:.2f}%\n\n"
        "🧪 BACKTEST vs OOS\n"
        + (
            f"BT: 期待{phase_stats['BACKTEST']['expectancy']:+.2f}% "
            f"PF{_pf_text(phase_stats['BACKTEST'])}\n"
            if phase_stats.get("BACKTEST")
            else "BT: データなし\n"
        )
        + (
            f"OOS: 期待{phase_stats['OOS']['expectancy']:+.2f}% "
            f"PF{_pf_text(phase_stats['OOS'])}\n"
            if phase_stats.get("OOS")
            else "OOS: データなし\n"
        )
        + "\n"
        + format_filter_comparison_for_discord(comparison_df)
    )

    send_discord(discord)

    # =====================================================
    # 完了
    # =====================================================

    print("")
    print("=" * 70)
    print("✅ 完了")
    print("=" * 70)

    print("")
    print("生成ファイル:")
    print("  walk_forward_results.csv")
    print("  portfolio_equity_curve.csv")
    print("  rank1_equity_curve.csv")
    print("  buy_equity_curve.csv")
    print("  backtest_summary.csv")
    print("  filter_comparison.csv")


if __name__ == "__main__":
    main()
