import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

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
    "7203.T",
    "7269.T",
    "285A.T",
    "9984.T",
    "4980.T",
    "8031.T",
    "8058.T",
    "9509.T",
    "9501.T",
    "8362.T",
    "8306.T",
    "5803.T",
    "6526.T",
    "6613.T",
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


# =========================================================
# バックテスト設定
# =========================================================

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

BUY_UP_PROB_MIN = 0.0


# =========================================================
# 【新機能】完全OOS設定
#
# 各銘柄データの「直近 OOS_DAYS 営業日」を、
# ロジック調整・パラメータ選定に一切使っていない
# 完全なアウトオブサンプル期間として扱う。
# 判定は run_walk_forward 内で、その予測が
# 「データ全体の末尾から何営業日目か」だけを見て行う
# (通常のウォークフォワードと学習/推論の仕組み自体は同じ)。
# =========================================================

OOS_DAYS = 90


# =========================================================
# 【新機能】相場局面(レジーム)判定設定
# =========================================================

REGIME_MA_SHORT = 25
REGIME_MA_LONG = 75
REGIME_VOL_WINDOW = 20
REGIME_HIGH_VOL_PERCENTILE = 0.85
REGIME_TREND_BAND = 0.01  # MA同士のかい離がこの割合未満なら「レンジ」


# =========================================================
# 【新機能】フォワードリターン確認用ホライズン(営業日)
# =========================================================

FORWARD_HORIZONS = (30, 60, 90)


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

            df = yf.download(
                ticker,
                **kwargs
            )

            if df is not None and not df.empty:

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                return df

            print(
                f"{ticker} 空データ "
                f"(試行 {attempt}/{retries})"
            )

        except Exception as e:

            print(
                f"{ticker} 取得失敗 "
                f"(試行 {attempt}/{retries}): {e}"
            )

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

    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


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

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    return atr


# =========================================================
# 特徴量
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
# 日経平均特徴量
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

    futures["future_return"] = futures["future_return"].shift(1)
    futures["future_ma5"] = futures["future_ma5"].shift(1)
    futures["future_rsi"] = futures["future_rsi"].shift(1)
    futures["future_gap"] = futures["future_gap"].shift(1)

    return futures


# =========================================================
# ターゲット
#
# 0 = 下落
# 1 = 横ばい
# 2 = 上昇
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
# 【新機能】複数ホライズンのフォワードリターン
#
# 実際の売買判定(TP/SL/タイムアウト)とは別に、
# 「もし利確・損切を無視してN営業日そのまま
#  持ち続けていたら何%になっていたか」を
# 参考値として記録する。TAKE_PROFIT/STOP_LOSS
# ロジックの妥当性を後から検証する材料になる。
# =========================================================

def create_multi_horizon_returns(df, horizons=FORWARD_HORIZONS):

    close = df["Close"].squeeze()

    out = {}

    for h in horizons:

        future_price = close.shift(-h)

        out[f"fwd_return_{h}d"] = (
            (future_price / close - 1) * 100
        )

    return pd.DataFrame(out, index=df.index)


# =========================================================
# 【新機能】相場局面(レジーム)分類
#
# 日経平均のMA25/MA75トレンドと、20日ボラティリティの
# 分位点をもとに、各日を以下の4区分に分類する。
#
#   急落・高ボラ : 直近ボラティリティが全期間の上位15%
#   上昇         : MA25がMA75を一定以上上回る
#   下落         : MA25がMA75を一定以上下回る
#   レンジ       : 上記いずれにも該当しない
#
# ※ ボラティリティの分位点は全期間を通した順位づけの
#   ため、あくまで「事後的な期間分類・集計用」であり、
#   売買判定や学習の特徴量には使っていない
#   (取引ロジックへのリークではない)。
# =========================================================

def classify_regime(nikkei):

    close = nikkei["Close"].squeeze()

    ma_short = close.rolling(REGIME_MA_SHORT).mean()
    ma_long = close.rolling(REGIME_MA_LONG).mean()

    daily_ret = close.pct_change()

    vol = daily_ret.rolling(REGIME_VOL_WINDOW).std() * 100

    vol_rank = vol.rank(pct=True)

    trend_diff = (ma_short - ma_long) / ma_long.replace(0, np.nan)

    regime = pd.Series("レンジ", index=nikkei.index, dtype=object)

    regime[trend_diff >= REGIME_TREND_BAND] = "上昇"
    regime[trend_diff <= -REGIME_TREND_BAND] = "下落"

    # 高ボラは他の判定より優先
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

    # -------------------------------------------------
    # 【新機能】複数ホライズンのフォワードリターンを付与
    # -------------------------------------------------

    multi_horizon = create_multi_horizon_returns(df)

    df = df.join(multi_horizon)

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

        # =================================================
        # リーク対策
        # =================================================

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

        # =================================================
        # 予測
        # =================================================

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

            buy_signal = (
                prediction == 2
                and up_prob >= BUY_UP_PROB_MIN
            )

            # =================================================
            # 【新機能】完全OOS判定
            #
            # データ全体の末尾から OOS_DAYS 営業日以内なら
            # 「完全OOS」、それより前なら「BACKTEST」。
            # =================================================

            phase = (
                "OOS"
                if i >= len(df) - OOS_DAYS
                else "BACKTEST"
            )

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

            # -------------------------------------------------
            # 【新機能】フォワードリターン(30/60/90日)を追加
            # -------------------------------------------------

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
# 詳細統計(期待値を追加)
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
    loss_rate = losses / decided * 100 if decided > 0 else 0

    profit_returns = df.loc[df["return"] > 0, "return"]
    loss_returns = df.loc[df["return"] < 0, "return"]

    profit_trades = len(profit_returns)
    loss_trades = len(loss_returns)

    avg_profit = profit_returns.mean() if profit_trades > 0 else 0
    avg_loss = loss_returns.mean() if loss_trades > 0 else 0

    gross_profit = profit_returns.sum()
    gross_loss = abs(loss_returns.sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    # =====================================================
    # 【新機能】期待値(Expectancy)
    #
    # 1トレードあたりの平均期待損益(%)。
    # 勝率×平均利益 + 負け率×平均損失(平均損失は負値)。
    # HOLD(未決着・含み益)は決着ベースの勝率計算からは
    # 除外しているが、平均利益/平均損失自体はreturn>0/<0の
    # 全トレード(HOLD・TIMEOUT_LOSS含む)から算出している。
    # =====================================================

    expectancy = (
        (win_rate / 100) * avg_profit
        + (loss_rate / 100) * avg_loss
    )

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
        "profit_trades": profit_trades,
        "loss_trades": loss_trades,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "start_equity": 1.0,
        "final_equity": final_equity,
        "cumulative_return": cumulative_return,
        "equity_curve": equity,
    }


# =========================================================
# 統計表示(コンソール・詳細版)
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
    print(f"期待値(1トレード) : {stats['expectancy']:+.2f}%")
    print(f"プロフィットファクター : {pf_text}")
    print(f"最大ドローダウン   : {stats['max_drawdown']:.2f}%")
    print(f"累積リターン       : {stats['cumulative_return']:+.2f}%")
    print(f"最終資産倍率       : {stats['final_equity']:.4f}")


# =========================================================
# 統計表示(コンソール・1行版:相場局面/OOS比較用)
# =========================================================

def print_stats_oneline(label, stats):

    if stats is None:
        print(f"{label:12s} : データなし")
        return

    pf_text = "―" if np.isnan(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"

    print(
        f"{label:12s} : "
        f"件数={stats['total']:4d} "
        f"勝率={stats['win_rate']:5.1f}% "
        f"期待値={stats['expectancy']:+.2f}% "
        f"PF={pf_text:>5s} "
        f"最大DD={stats['max_drawdown']:6.1f}% "
        f"累積={stats['cumulative_return']:+7.1f}%"
    )


# =========================================================
# 【新機能】グループ別統計をまとめて計算
# (相場局面別・BT vs OOS など、group_colの値ごとに
#  compute_statsを実行する汎用ヘルパー)
# =========================================================

def compute_group_stats(df, group_col, labels):

    return {
        label: compute_stats(df[df[group_col] == label])
        for label in labels
    }


# =========================================================
# Discord用:詳細ブロック(複数行)
# =========================================================

def format_stats_block(title, stats):

    if stats is None:
        return f"\n{title}\nデータなし\n"

    pf_text = "―" if np.isnan(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"

    return (
        f"\n{title}\n"
        f"勝率: {stats['win_rate']:.1f}%\n"
        f"件数: {stats['total']}回\n"
        f"WIN: {stats['wins']} LOSS: {stats['losses']} HOLD: {stats['holds']}\n"
        f"平均利益: +{stats['avg_profit']:.2f}%\n"
        f"平均損失: {stats['avg_loss']:.2f}%\n"
        f"期待値: {stats['expectancy']:+.2f}%\n"
        f"PF: {pf_text}\n"
        f"最大DD: {stats['max_drawdown']:.1f}%\n"
        f"累積: {stats['cumulative_return']:+.1f}%\n"
    )


# =========================================================
# Discord用:1行ブロック(相場局面別・BT vs OOS用)
# =========================================================

def format_stats_oneline(label, stats):

    if stats is None:
        return f"{label}: データなし"

    pf_text = "―" if np.isnan(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"

    return (
        f"{label}: 勝率{stats['win_rate']:.0f}% "
        f"件数{stats['total']} "
        f"期待値{stats['expectancy']:+.2f}% "
        f"PF{pf_text} "
        f"DD{stats['max_drawdown']:.0f}% "
        f"累積{stats['cumulative_return']:+.0f}%"
    )


# =========================================================
# 【新機能】フォワードリターン(30/60/90日)まとめ
# =========================================================

def format_forward_return_summary(df, horizons=FORWARD_HORIZONS):

    subset = df[df["buy_signal"] == 1]

    lines = ["\n📅 フォワードリターン(買い推奨・平均)"]

    if len(subset) == 0:
        lines.append("データなし")
        return "\n".join(lines) + "\n"

    for h in horizons:

        col = f"fwd_return_{h}d"

        if col not in subset.columns:
            continue

        vals = subset[col].dropna()

        if len(vals) > 0:
            lines.append(
                f"{h}日: {vals.mean():+.2f}% (件数{len(vals)})"
            )
        else:
            lines.append(f"{h}日: データ不足")

    return "\n".join(lines) + "\n"


# =========================================================
# Discordメッセージ
# =========================================================

def build_discord_message(
    result_df,
    overall_stats,
    rank1_stats,
    buy_stats,
    regime_stats,
    phase_stats,
    forward_summary_text
):

    start_date = result_df["date"].min()
    end_date = result_df["date"].max()

    pf_text = (
        "―" if np.isnan(overall_stats["profit_factor"])
        else f"{overall_stats['profit_factor']:.2f}"
    )

    msg = ""

    msg += "📊 AI WALK FORWARD BACKTEST\n"
    msg += "━━━━━━━━━━━━━━━━\n"
    msg += f"期間: {start_date} ～ {end_date}\n\n"

    msg += (
        "💰 最終資産倍率\n"
        f"{overall_stats['start_equity']:.4f} → "
        f"{overall_stats['final_equity']:.4f}\n\n"
    )

    msg += f"📈 累積リターン\n{overall_stats['cumulative_return']:+.2f}%\n\n"
    msg += f"🎯 勝率\n{overall_stats['win_rate']:.2f}%\n\n"
    msg += f"📊 総件数\n{overall_stats['total']}件\n\n"
    msg += f"✅ WIN\n{overall_stats['wins']}件\n\n"
    msg += f"❌ LOSS\n{overall_stats['losses']}件\n\n"
    msg += f"⏸ HOLD\n{overall_stats['holds']}件\n\n"
    msg += f"💵 平均利益\n+{overall_stats['avg_profit']:.2f}%\n\n"
    msg += f"💸 平均損失\n{overall_stats['avg_loss']:.2f}%\n\n"
    msg += f"🧮 期待値(1トレード)\n{overall_stats['expectancy']:+.2f}%\n\n"
    msg += f"⚖️ プロフィットファクター\n{pf_text}\n\n"
    msg += f"📉 最大ドローダウン\n{overall_stats['max_drawdown']:.2f}%\n"

    msg += "━━━━━━━━━━━━━━━━\n"
    msg += "🧪 バックテスト vs 完全OOS\n"
    msg += format_stats_oneline("BT", phase_stats.get("BACKTEST")) + "\n"
    msg += format_stats_oneline("OOS", phase_stats.get("OOS")) + "\n"

    msg += "━━━━━━━━━━━━━━━━\n"
    msg += "🌦 相場局面別\n"
    for label in ["上昇", "下落", "レンジ", "急落・高ボラ"]:
        msg += format_stats_oneline(label, regime_stats.get(label)) + "\n"

    msg += forward_summary_text

    msg += "━━━━━━━━━━━━━━━━\n"

    msg += format_stats_block("🏆 ランク1", rank1_stats)
    msg += format_stats_block("🟢 買い推奨", buy_stats)

    return msg


# =========================================================
# 【新機能】統計サマリーCSV保存
#
# 全体・相場局面別・BT/OOS・ランク1・買い推奨の
# 各統計を1つの表にまとめて保存する。
# =========================================================

def save_summary_csv(
    overall_stats,
    regime_stats,
    phase_stats,
    rank1_stats,
    buy_stats
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

    for label in ["上昇", "下落", "レンジ", "急落・高ボラ"]:
        add_row(f"相場局面_{label}", regime_stats.get(label))

    for label in ["BACKTEST", "OOS"]:
        add_row(f"フェーズ_{label}", phase_stats.get(label))

    add_row("ランク1", rank1_stats)
    add_row("買い推奨", buy_stats)

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(
        "backtest_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("")
    print("✅ backtest_summary.csv 保存")


# =========================================================
# メイン
# =========================================================

def main():

    print("")
    print("=" * 60)
    print("📊 ウォークフォワード・バックテスト 拡張版")
    print("=" * 60)

    print(f"学習期間: {TRAIN_DAYS}営業日")
    print(f"再学習間隔: {RETRAIN_DAYS}営業日")
    print(f"予測期間: {FORWARD_DAYS}営業日")
    print(f"売買判定: 最大{MAX_HOLD_DAYS}営業日")
    print(f"利確: +{TAKE_PROFIT * 100:.1f}%")
    print(f"損切: -{STOP_LOSS * 100:.1f}%")
    print(f"流動性: {MIN_AVG_VOLUME:,}")
    print(f"完全OOS期間: 直近{OOS_DAYS}営業日")
    print(f"フォワード確認: {FORWARD_HORIZONS}営業日")
    print("")

    # =====================================================
    # 日経平均
    # =====================================================

    print("📥 日経平均取得")

    nikkei = safe_download(
        "^N225", period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False
    )

    if nikkei is None:
        print("❌ 日経平均取得失敗")
        return

    # -----------------------------------------------------
    # 【新機能】相場局面をここで先に算出しておく
    # (特徴量作成前の生のCloseを使うため、
    #  create_nikkei_featuresより先に実行)
    # -----------------------------------------------------

    regime_series = classify_regime(nikkei)

    nikkei = create_nikkei_features(nikkei)

    # =====================================================
    # 日経225先物
    # =====================================================

    print("📥 日経225先物取得")

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
        print("❌ バックテスト結果なし")
        return

    result_df = pd.DataFrame(all_results)

    # -----------------------------------------------------
    # 【新機能】相場局面(regime)をdateで紐付け
    # -----------------------------------------------------

    regime_lookup = {
        idx.strftime("%Y-%m-%d"): val
        for idx, val in regime_series.items()
    }

    result_df["regime"] = result_df["date"].map(regime_lookup)

    result_df.to_csv(
        "walk_forward_results.csv", index=False, encoding="utf-8-sig"
    )

    print("")
    print("✅ walk_forward_results.csv 保存")

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
    # 買い推奨
    # =====================================================

    buy_df = result_df[result_df["buy_signal"] == 1].copy()
    buy_stats = compute_stats(buy_df)

    # =====================================================
    # 【新機能】相場局面別
    # =====================================================

    REGIME_LABELS = ["上昇", "下落", "レンジ", "急落・高ボラ"]

    regime_stats = compute_group_stats(result_df, "regime", REGIME_LABELS)

    # =====================================================
    # 【新機能】バックテスト vs 完全OOS
    # =====================================================

    PHASE_LABELS = ["BACKTEST", "OOS"]

    phase_stats = compute_group_stats(result_df, "phase", PHASE_LABELS)

    # =====================================================
    # 全体表示
    # =====================================================

    print("")
    print("=" * 60)
    print("📊 ウォークフォワード結果")
    print("=" * 60)

    print(f"予測件数       : {len(result_df)}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print(f"Precision      : {precision:.2f}%")

    print_stats("📊 全体成績", overall_stats)
    print_stats("🏆 ランク1成績", rank1_stats)
    print_stats("🟢 買い推奨成績", buy_stats)

    print("")
    print("=" * 60)
    print("🧪 バックテスト vs 完全OOS")
    print("=" * 60)
    print_stats_oneline("BACKTEST", phase_stats.get("BACKTEST"))
    print_stats_oneline("OOS", phase_stats.get("OOS"))

    print("")
    print("=" * 60)
    print("🌦 相場局面別成績")
    print("=" * 60)
    for label in REGIME_LABELS:
        print_stats_oneline(label, regime_stats.get(label))

    # =====================================================
    # 銘柄別
    # =====================================================

    print("")
    print("=" * 60)
    print("📊 銘柄別成績")
    print("=" * 60)

    for ticker, group in result_df.groupby("ticker"):

        ticker_accuracy = group["correct"].mean() * 100

        ticker_wins = (group["result"] == "WIN").sum()
        ticker_losses = group["result"].isin(["LOSS", "TIMEOUT_LOSS"]).sum()
        ticker_trades = ticker_wins + ticker_losses

        ticker_win_rate = (
            ticker_wins / ticker_trades * 100 if ticker_trades > 0 else 0
        )

        ticker_avg_return = group["return"].mean()

        print(
            f"{ticker:8s} "
            f"{COMPANY_NAMES.get(ticker, ''):10s} "
            f"Accuracy={ticker_accuracy:5.1f}% "
            f"勝率={ticker_win_rate:5.1f}% "
            f"平均={ticker_avg_return:+.2f}% "
            f"件数={len(group)}"
        )

    # =====================================================
    # 資産曲線
    # =====================================================

    equity = overall_stats["equity_curve"]

    equity.to_csv(
        "portfolio_equity_curve.csv",
        header=["equity"],
        encoding="utf-8-sig"
    )

    print("")
    print("✅ portfolio_equity_curve.csv 保存")

    if rank1_stats is not None:
        rank1_stats["equity_curve"].to_csv(
            "rank1_equity_curve.csv", header=["equity"], encoding="utf-8-sig"
        )
        print("✅ rank1_equity_curve.csv 保存")

    if buy_stats is not None:
        buy_stats["equity_curve"].to_csv(
            "buy_equity_curve.csv", header=["equity"], encoding="utf-8-sig"
        )
        print("✅ buy_equity_curve.csv 保存")

    # =====================================================
    # 【新機能】統計サマリーCSV
    # =====================================================

    save_summary_csv(
        overall_stats,
        regime_stats,
        phase_stats,
        rank1_stats,
        buy_stats
    )

    # =====================================================
    # 流動性除外
    # =====================================================

    if skipped_liquidity:
        print("")
        print("⚠ 流動性フィルター除外:")
        for ticker in skipped_liquidity:
            print(f"  {ticker} {COMPANY_NAMES.get(ticker, '')}")

    # =====================================================
    # 【新機能】フォワードリターンまとめ
    # =====================================================

    forward_summary_text = format_forward_return_summary(result_df)

    print("")
    print(forward_summary_text)

    # =====================================================
    # Discord
    # =====================================================

    discord_msg = build_discord_message(
        result_df,
        overall_stats,
        rank1_stats,
        buy_stats,
        regime_stats,
        phase_stats,
        forward_summary_text
    )

    send_discord(discord_msg)

    # =====================================================
    # 完了
    # =====================================================

    print("")
    print("=" * 60)
    print("✅ ウォークフォワード・バックテスト完了")
    print("=" * 60)

    print("")
    print("生成ファイル:")
    print("  walk_forward_results.csv     (取引明細・regime/phase/fwd_return付き)")
    print("  portfolio_equity_curve.csv   (全体・日次資産曲線)")
    print("  rank1_equity_curve.csv       (ランク1・日次資産曲線)")
    print("  buy_equity_curve.csv         (買い推奨・日次資産曲線)")
    print("  backtest_summary.csv         (全グループ横断の統計サマリー)")


# =========================================================
# 実行
# =========================================================

if __name__ == "__main__":

    main()
