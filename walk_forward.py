import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# WALK-FORWARD BACKTEST
#
# stock_scan.py の本番ロジックを時系列で再現する。
#
# 【重要】
#
# 1. 予測日より未来のデータを学習に使わない
#
# 2. targetは「3営業日後」の価格から作成
#
# 3. targetが確定する日(target_date)が
#    prediction_dateより前の行だけを学習に使う
#
# 4. 3クラス分類
#    0 = 下落
#    1 = 横ばい
#    2 = 上昇
#
# 5. ATRベースtarget
#
# 6. RandomForestClassifier
#
# 7. テクニカル70% + AI確率30%
#
# 8. 日経MA25 > MA75のときだけ
#    「買い」「強い買い」を許可
#
# 9. ATR TP=3.0倍
#    ATR SL=1.5倍
#
# 10. 同日TP/SL両ヒットは保守的にLOSS
#
# 11. 【本番完全一致】流動性フィルターは、
#     本番(stock_scan.py)で「その日その銘柄が
#     流動性不足のときは train_data.csv に行が
#     追加されない」挙動と一致させる。
#     つまり学習データの選定にも liquid 条件を含める。
#     (当日の予測対象銘柄を絞る判定にも同じliquid列を使う)
#     過去の行自体は削除せず列として保持するので、
#     shift(-FORWARD_DAYS) の「3営業日後」の意味は崩れない。
#
# 12. target計算はFEATURESのdropnaより前に行う
#
# 13. 本番との比較を優先するため、
#     学習対象は「予測日より前にtargetが確定していて、
#     かつその日が流動的だった」全履歴を使う
# =========================================================


# =========================================================
# 基本設定
# =========================================================

START_DATE = os.getenv(
    "WF_START_DATE",
    "2021-01-01"
)

END_DATE = os.getenv(
    "WF_END_DATE",
    "2026-08-22"
)

# START_DATE以前に取得する学習履歴
# 本番との完全比較では十分な過去履歴を確保する。
HISTORY_YEARS = int(
    os.getenv(
        "WF_HISTORY_YEARS",
        "6"
    )
)

TOP_N = int(
    os.getenv(
        "WF_TOP_N",
        "3"
    )
)

FORWARD_DAYS = 3

ATR_TARGET_MULTIPLIER = 1.0

ATR_TP_MULTIPLIER = 3.0

ATR_SL_MULTIPLIER = 1.5

MIN_AVG_VOLUME = 300000

MIN_TRAIN_ROWS = 200


# =========================================================
# テスト開始時ウォームアップ
#
# START_DATEからすぐ予測せず、
# 252営業日分を予測対象から除外。
#
# ただし学習データそのものは
# START_DATEより前の履歴を使える。
# =========================================================

TEST_WARMUP_DAYS = int(
    os.getenv(
        "WF_WARMUP_DAYS",
        "252"
    )
)


# =========================================================
# モデル再学習間隔
#
# 1  = 毎営業日
# 20 = 20営業日ごと
#
# 本番との最終一致確認では1を推奨。
# 初回の速度確認は20でもよい。
# =========================================================

REFIT_EVERY_TRADING_DAYS = int(
    os.getenv(
        "WF_REFIT_EVERY_TRADING_DAYS",
        "20"
    )
)


# =========================================================
# RandomForest設定
# =========================================================

N_ESTIMATORS = 300

MAX_DEPTH = 7

RANDOM_STATE = 42


# =========================================================
# 比較用
#
# False:
# 本番と同じくスコア順TOP_N
#
# True:
# 「🔥 強い買い」「🟢 買い」だけを対象
# =========================================================

TRADE_ONLY_BUY_SIGNALS = (
    os.getenv(
        "WF_TRADE_ONLY_BUY",
        "false"
    ).lower()
    == "true"
)


# =========================================================
# 銘柄
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
    "6613.T"
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
# FEATURES
#
# stock_scan.py と同じ
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
    "relative_strength",
    "bb_position",
    "bb_width",
    "obv_change",
    "atr_ratio",
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
# safe_download
# =========================================================

def safe_download(
    ticker,
    retries=3,
    wait_sec=2,
    **kwargs
):

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            df = yf.download(
                ticker,
                **kwargs
            )

            if (
                df is not None
                and
                not df.empty
            ):

                if isinstance(
                    df.columns,
                    pd.MultiIndex
                ):

                    df.columns = (
                        df.columns
                        .get_level_values(0)
                    )

                return df

            print(
                f"{ticker}: "
                f"空データ "
                f"({attempt}/{retries})"
            )

        except Exception as e:

            print(
                f"{ticker}: "
                f"取得失敗 "
                f"({attempt}/{retries}): "
                f"{e}"
            )

        if attempt < retries:

            time.sleep(
                wait_sec
            )


    print(
        f"❌ {ticker}: "
        f"取得失敗"
    )

    return None


# =========================================================
# RSI
# =========================================================

def calc_rsi(
    close,
    period=14
):

    close = close.squeeze()

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta
    ).clip(
        lower=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss
    )

    rsi = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return rsi.where(
        avg_loss != 0,
        100
    )


# =========================================================
# ADX
# =========================================================

def calc_adx(
    df,
    period=14
):

    high = (
        df["High"]
        .squeeze()
    )

    low = (
        df["Low"]
        .squeeze()
    )

    close = (
        df["Close"]
        .squeeze()
    )

    prev_close = (
        close.shift(1)
    )

    tr = pd.concat(
        [
            high - low,
            (
                high
                -
                prev_close
            ).abs(),
            (
                low
                -
                prev_close
            ).abs(),
        ],
        axis=1
    ).max(
        axis=1
    )

    up_move = (
        high.diff()
    )

    down_move = (
        -low.diff()
    )

    plus_dm = np.where(
        (
            up_move
            >
            down_move
        )
        &
        (
            up_move > 0
        ),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (
            down_move
            >
            up_move
        )
        &
        (
            down_move > 0
        ),
        down_move,
        0.0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=high.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=high.index
    )

    atr = (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    plus_dm_sm = (
        plus_dm
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    minus_dm_sm = (
        minus_dm
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    plus_di = (
        100
        * plus_dm_sm
        /
        atr.replace(
            0,
            np.nan
        )
    )

    minus_di = (
        100
        * minus_dm_sm
        /
        atr.replace(
            0,
            np.nan
        )
    )

    dx = (
        (
            plus_di
            -
            minus_di
        ).abs()
        /
        (
            plus_di
            +
            minus_di
        ).replace(
            0,
            np.nan
        )
    ) * 100

    return (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )


# =========================================================
# 個別銘柄特徴量
# =========================================================

def create_features(
    df
):

    df = df.copy()

    close = (
        df["Close"]
        .squeeze()
    )

    volume = (
        df["Volume"]
        .squeeze()
    )


    # -----------------------------------------------------
    # 基本
    # -----------------------------------------------------

    df["ret1"] = (
        close.pct_change()
    )

    df["ma25"] = (
        close
        .rolling(25)
        .mean()
    )

    df["ma75"] = (
        close
        .rolling(75)
        .mean()
    )

    df["vol_ratio"] = (
        volume
        /
        volume
        .rolling(20)
        .mean()
    )

    df["rsi"] = (
        calc_rsi(close)
    )

    df["adx"] = (
        calc_adx(df)
    )


    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["macd"] = (
        ema12
        -
        ema26
    )

    df["signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    # -----------------------------------------------------
    # 52週高値 / 安値
    # -----------------------------------------------------

    high252 = (
        close
        .rolling(252)
        .max()
    )

    low252 = (
        close
        .rolling(252)
        .min()
    )

    df["high252"] = (
        high252
    )

    df["from_high"] = (
        close
        /
        high252
        -
        1
    ) * 100

    df["from_low"] = (
        close
        /
        low252
        -
        1
    ) * 100


    # -----------------------------------------------------
    # 相対強度用中間列
    # -----------------------------------------------------

    df["_stock_ret5"] = (
        close
        .pct_change(5)
    )


    # -----------------------------------------------------
    # Bollinger Band
    # -----------------------------------------------------

    bb_ma20 = (
        close
        .rolling(20)
        .mean()
    )

    bb_std20 = (
        close
        .rolling(20)
        .std()
    )

    bb_upper = (
        bb_ma20
        +
        bb_std20 * 2
    )

    bb_lower = (
        bb_ma20
        -
        bb_std20 * 2
    )

    df["bb_position"] = (
        (
            close
            -
            bb_lower
        )
        /
        (
            bb_upper
            -
            bb_lower
        )
    )

    df["bb_width"] = (
        (
            bb_upper
            -
            bb_lower
        )
        /
        bb_ma20
        *
        100
    )


    # -----------------------------------------------------
    # OBV
    # -----------------------------------------------------

    price_direction = np.sign(
        close.diff()
    )

    obv = (
        (
            volume
            *
            price_direction
        )
        .fillna(0)
        .cumsum()
    )

    df["obv_change"] = (
        obv.diff(5)
        /
        volume
        .rolling(5)
        .sum()
        *
        100
    )


    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    high = (
        df["High"]
        .squeeze()
    )

    low = (
        df["Low"]
        .squeeze()
    )

    prev_close = (
        close.shift(1)
    )

    tr = pd.concat(
        [
            high - low,
            (
                high
                -
                prev_close
            ).abs(),
            (
                low
                -
                prev_close
            ).abs(),
        ],
        axis=1
    ).max(
        axis=1
    )

    atr = (
        tr
        .rolling(14)
        .mean()
    )

    df["atr_ratio"] = (
        atr
        /
        close
        *
        100
    )


    return df


# =========================================================
# 日経特徴量
# =========================================================

def build_nikkei_features(
    nikkei
):

    nikkei = nikkei.copy()

    close = (
        nikkei["Close"]
        .squeeze()
    )

    nikkei["nikkei_ma25"] = (
        close
        .rolling(25)
        .mean()
    )

    nikkei["nikkei_ma75"] = (
        close
        .rolling(75)
        .mean()
    )

    nikkei["nikkei_kairi25"] = (
        (
            close
            -
            nikkei[
                "nikkei_ma25"
            ]
        )
        /
        nikkei[
            "nikkei_ma25"
        ]
        *
        100
    )

    nikkei["nikkei_uptrend"] = (
        nikkei[
            "nikkei_ma25"
        ]
        >
        nikkei[
            "nikkei_ma75"
        ]
    )

    nikkei["nikkei_rsi"] = (
        calc_rsi(
            close
        )
    )

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    nikkei["nikkei_macd"] = (
        ema12
        -
        ema26
    )

    nikkei["nikkei_return_5d"] = (
        close
        .pct_change(5)
        *
        100
    )

    nikkei["nikkei_ret5_raw"] = (
        close
        .pct_change(5)
    )

    return nikkei


# =========================================================
# 日経225先物
# =========================================================

def build_futures_features(
    futures
):

    futures = futures.copy()

    close = (
        futures["Close"]
        .squeeze()
    )

    futures["future_return"] = (
        close
        .pct_change()
    )

    futures["future_ma5"] = (
        close
        .rolling(5)
        .mean()
    )

    futures["future_rsi"] = (
        calc_rsi(
            close
        )
    )

    futures["future_gap"] = (
        (
            close
            -
            close.shift(1)
        )
        /
        close.shift(1)
    )


    # -----------------------------------------------------
    # stock_scan.pyと同じ1日ラグ
    # -----------------------------------------------------

    for col in [
        "future_return",
        "future_ma5",
        "future_rsi",
        "future_gap"
    ]:

        futures[col] = (
            futures[col]
            .shift(1)
        )


    return futures


# =========================================================
# 銘柄データ準備
#
# ★順番が重要
#
# 1. 株価データ
# 2. 個別特徴量
# 3. 日経JOIN
# 4. 先物JOIN
# 5. relative_strength
# 6. 流動性計算
# 7. target計算
# 8. target_date計算
# 9. 最後にFEATURESのNaN除外
#
# これにより、
# 「3営業日後」の意味を維持する。
# =========================================================

def prepare_symbol_data(
    ticker,
    nikkei,
    futures,
    start,
    end
):

    df = safe_download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
    )


    if (
        df is None
        or
        len(df) < 150
    ):

        return None


    df = create_features(
        df
    )


    # =====================================================
    # 日経特徴量
    # =====================================================

    df = df.join(
        nikkei[
            [
                "nikkei_ma25",
                "nikkei_ma75",
                "nikkei_kairi25",
                "nikkei_rsi",
                "nikkei_macd",
                "nikkei_return_5d",
                "nikkei_ret5_raw",
                "nikkei_uptrend",
            ]
        ],
        how="left"
    )


    # =====================================================
    # 先物特徴量
    # =====================================================

    df = df.join(
        futures[
            [
                "future_return",
                "future_ma5",
                "future_rsi",
                "future_gap",
            ]
        ],
        how="left"
    )


    # =====================================================
    # 相対強度
    # =====================================================

    df["relative_strength"] = (
        df["_stock_ret5"]
        -
        df["nikkei_ret5_raw"]
    )


    # =====================================================
    # 流動性
    #
    # ★行を削除しない
    #
    # 本番では最新日の流動性を確認して
    # その銘柄を今回の処理対象から外す。
    #
    # 過去データから行を消してしまうと、
    # 学習データ自体が変わってしまう。
    #
    # 【本番完全一致】ただし、本番の
    # train_data.csv は「その日その銘柄が流動的
    # だった日」しか行が追加されない仕様なので、
    # 学習データを選ぶ側(main loop)では
    # liquid列を条件に含める。
    # =====================================================

    df["avg_volume20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["liquid"] = (
        df["avg_volume20"]
        >=
        MIN_AVG_VOLUME
    )


    # =====================================================
    # target
    #
    # ★dropnaより前に計算
    #
    # 3営業日後の価格を使用。
    # =====================================================

    future_price = (
        df["Close"]
        .shift(
            -FORWARD_DAYS
        )
    )

    future_return = (
        future_price
        /
        df["Close"]
        -
        1
    ) * 100


    atr_threshold = (
        df["atr_ratio"]
        *
        ATR_TARGET_MULTIPLIER
        *
        np.sqrt(
            FORWARD_DAYS
        )
    )


    df["target"] = np.select(
        [
            future_return
            <=
            -atr_threshold,

            future_return
            >=
            atr_threshold,
        ],
        [
            0,
            2
        ],
        default=1
    )


    df["target_valid"] = (
        future_return.notna()
    )


    # =====================================================
    # target_date
    #
    # このtargetが何日後の価格で確定したか。
    # =====================================================

    target_date = (
        pd.Series(
            df.index,
            index=df.index
        )
        .shift(
            -FORWARD_DAYS
        )
    )

    df["target_date"] = (
        target_date
    )


    # =====================================================
    # 最後にFEATURESのNaN除外
    # =====================================================

    df = df.dropna(
        subset=FEATURES
    ).copy()


    if len(df) < 100:

        return None


    return df


# =========================================================
# シグナル計算
#
# stock_scan.py の calc_score() と合わせる
# =========================================================

def calculate_signal(
    df_row,
    up_prob,
    down_prob,
    flat_prob
):

    price = float(
        df_row["Close"]
    )

    rsi = float(
        df_row["rsi"]
    )

    macd = float(
        df_row["macd"]
    )

    signal = float(
        df_row["signal"]
    )

    ma25 = float(
        df_row["ma25"]
    )

    ma75 = float(
        df_row["ma75"]
    )

    vol_ratio = float(
        df_row["vol_ratio"]
    )


    # =====================================================
    # 52週高値
    # =====================================================

    high52 = float(
        df_row["high252"]
    )


    if (
        not np.isfinite(
            high52
        )
        or
        high52 <= 0
    ):

        distance = np.nan

    else:

        distance = (
            price
            /
            high52
            -
            1
        ) * 100


    # =====================================================
    # テクニカルスコア
    # =====================================================

    technical_score = 0


    if rsi < 35:

        technical_score += 25


    if macd > signal:

        technical_score += 25


    if ma25 > ma75:

        technical_score += 20


    if vol_ratio > 1.5:

        technical_score += 20


    if np.isfinite(
        distance
    ):

        if distance > -10:

            technical_score += 15

        elif distance > -20:

            technical_score += 8


    if float(
        df_row[
            "nikkei_rsi"
        ]
    ) > 50:

        technical_score += 5


    if float(
        df_row[
            "nikkei_return_5d"
        ]
    ) > 0:

        technical_score += 5


    # =====================================================
    # AIスコア
    # =====================================================

    technical_score_normalized = (
        technical_score
        /
        115.0
        *
        100
    )


    ai_score = (
        technical_score_normalized
        * 0.70
        +
        (
            up_prob
            *
            100
        )
        * 0.30
    )


    ai_score = max(
        0,
        min(
            100,
            ai_score
        )
    )


    # =====================================================
    # 確率
    # =====================================================

    up_percent = (
        up_prob * 100
    )

    down_percent = (
        down_prob * 100
    )

    flat_percent = (
        flat_prob * 100
    )


    # =====================================================
    # 拮抗・横ばい
    # =====================================================

    is_tie = (
        abs(
            up_percent
            -
            down_percent
        )
        <=
        1.0
    )

    is_flat_dominant = (
        flat_percent
        >=
        50.0
    )


    # =====================================================
    # 通常判定
    # =====================================================

    if is_flat_dominant:

        if (
            up_percent >= 50
            and
            up_percent > down_percent
        ):

            final_signal = (
                "🟡 監視(横ばい優勢)"
            )

        else:

            final_signal = (
                "🔴 買わない"
            )


    elif is_tie:

        final_signal = (
            "🟡 監視(拮抗)"
        )


    elif (
        up_percent >= 60
        and
        up_percent > down_percent
    ):

        final_signal = (
            "🔥 強い買い"
        )


    elif (
        up_percent >= 50
        and
        up_percent > down_percent
    ):

        final_signal = (
            "🟢 買い"
        )


    elif (
        up_percent >= 40
        and
        up_percent > down_percent
    ):

        final_signal = (
            "🟡 監視"
        )


    else:

        final_signal = (
            "🔴 買わない"
        )


    # =====================================================
    # 日経フィルター
    #
    # MA25 > MA75 でない場合、
    # 買い系を監視へ落とす。
    # =====================================================

    nikkei_uptrend = bool(
        df_row[
            "nikkei_uptrend"
        ]
    )


    if (
        not nikkei_uptrend
        and
        final_signal
        in [
            "🔥 強い買い",
            "🟢 買い"
        ]
    ):

        final_signal = (
            "🟡 監視(日経下落/レンジ)"
        )


    # =====================================================
    # ATR TP / SL
    # =====================================================

    atr_ratio = float(
        df_row[
            "atr_ratio"
        ]
    )


    take_profit = round(
        price
        *
        (
            1
            +
            (
                atr_ratio
                /
                100
            )
            *
            ATR_TP_MULTIPLIER
        ),
        0
    )


    stop_loss = round(
        price
        *
        (
            1
            -
            (
                atr_ratio
                /
                100
            )
            *
            ATR_SL_MULTIPLIER
        ),
        0
    )


    return {

        "score":
            ai_score,

        "signal":
            final_signal,

        "nikkei_uptrend":
            nikkei_uptrend,

        "technical_score":
            technical_score_normalized,

        "price":
            price,

        "take_profit":
            take_profit,

        "stop_loss":
            stop_loss,

        "up_prob":
            up_percent,

        "flat_prob":
            flat_percent,

        "down_prob":
            down_percent,

        "rsi":
            rsi,

        "vol":
            vol_ratio,
    }


# =========================================================
# 5営業日結果判定
#
# entry_dateの終値で入る。
# 翌営業日から最大5営業日を見る。
#
# 同日にTP/SL両方ヒット
# → LOSS
# =========================================================

def evaluate_trade(
    day_df,
    entry_date,
    entry_price,
    take_profit,
    stop_loss
):

    future = (
        day_df
        .loc[
            day_df.index
            >
            entry_date
        ]
        .head(5)
        .copy()
    )


    if future.empty:

        return (
            "NO_DATA",
            np.nan,
            0,
            np.nan
        )


    for hold_day, (_, row) in enumerate(
        future.iterrows(),
        start=1
    ):

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )


        # -------------------------------------------------
        # 同日両ヒット
        # -------------------------------------------------

        if (
            low <= stop_loss
            and
            high >= take_profit
        ):

            ret = (
                (
                    stop_loss
                    -
                    entry_price
                )
                /
                entry_price
                *
                100
            )

            return (
                "LOSS",
                ret,
                hold_day,
                stop_loss
            )


        # -------------------------------------------------
        # TP
        # -------------------------------------------------

        if high >= take_profit:

            ret = (
                (
                    take_profit
                    -
                    entry_price
                )
                /
                entry_price
                *
                100
            )

            return (
                "WIN",
                ret,
                hold_day,
                take_profit
            )


        # -------------------------------------------------
        # SL
        # -------------------------------------------------

        if low <= stop_loss:

            ret = (
                (
                    stop_loss
                    -
                    entry_price
                )
                /
                entry_price
                *
                100
            )

            return (
                "LOSS",
                ret,
                hold_day,
                stop_loss
            )


    # =====================================================
    # 5日後
    # =====================================================

    close_price = float(
        future.iloc[-1]["Close"]
    )


    ret = (
        (
            close_price
            -
            entry_price
        )
        /
        entry_price
        *
        100
    )


    if (
        len(future) >= 5
        and
        ret < 0
    ):

        result = (
            "TIMEOUT_LOSS"
        )

    else:

        result = (
            "HOLD"
        )


    return (
        result,
        ret,
        len(future),
        close_price
    )


# =========================================================
# モデル学習
# =========================================================

def fit_model(
    train_df
):

    usable = (
        train_df[
            train_df[
                "target_valid"
            ]
        ]
        .copy()
    )


    usable = (
        usable
        .dropna(
            subset=
            FEATURES
            +
            ["target"]
        )
    )


    usable["target"] = (
        usable[
            "target"
        ]
        .astype(int)
    )


    usable = usable[
        usable[
            "target"
        ].isin(
            [0, 1, 2]
        )
    ]


    if len(usable) < MIN_TRAIN_ROWS:

        return None


    if (
        usable[
            "target"
        ].nunique()
        <
        3
    ):

        return None


    model = (
        RandomForestClassifier(
            n_estimators=
                N_ESTIMATORS,

            max_depth=
                MAX_DEPTH,

            random_state=
                RANDOM_STATE,

            class_weight=
                "balanced",

            n_jobs=-1,
        )
    )


    model.fit(
        usable[FEATURES],
        usable["target"]
    )


    return model


# =========================================================
# 最大DD
# =========================================================

def max_drawdown(
    returns
):

    returns = (
        pd.Series(
            returns
        )
        .dropna()
    )


    if len(returns) == 0:

        return 0.0


    equity = (
        1
        +
        returns
        /
        100
    ).cumprod()


    peak = (
        equity
        .cummax()
    )


    dd = (
        equity
        /
        peak
        -
        1
    )


    return float(
        dd.min()
        * 100
    )


# =========================================================
# Profit Factor
# =========================================================

def profit_factor(
    returns
):

    returns = (
        pd.Series(
            returns
        )
        .dropna()
    )


    gains = (
        returns[
            returns > 0
        ]
        .sum()
    )


    losses = (
        -returns[
            returns < 0
        ]
        .sum()
    )


    if losses <= 0:

        if gains > 0:

            return np.inf

        return 0.0


    return float(
        gains
        /
        losses
    )


# =========================================================
# サマリー
# =========================================================

def summarize(
    result_df,
    label
):

    if result_df.empty:

        print(
            f"\n[{label}] "
            f"取引なし"
        )

        return {}


    decided = (
        result_df[
            result_df[
                "result"
            ].isin(
                [
                    "WIN",
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            )
        ]
        .copy()
    )


    wins = int(
        (
            decided[
                "result"
            ]
            ==
            "WIN"
        ).sum()
    )


    losses = int(
        (
            decided[
                "result"
            ].isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            )
        ).sum()
    )


    holds = int(
        (
            result_df[
                "result"
            ]
            ==
            "HOLD"
        ).sum()
    )


    signals = len(
        result_df
    )


    avg_return = float(
        result_df[
            "return"
        ].mean()
    )


    decided_count = (
        wins
        +
        losses
    )


    win_rate = (
        wins
        /
        decided_count
        *
        100
        if
        decided_count > 0
        else
        0.0
    )


    pf = (
        profit_factor(
            result_df[
                "return"
            ]
        )
    )


    mdd = (
        max_drawdown(
            result_df[
                "return"
            ]
        )
    )


    avg_hold_days = float(
        result_df[
            "hold_days"
        ].mean()
    )


    pf_text = (
        "inf"
        if np.isinf(pf)
        else
        f"{pf:.2f}"
    )


    print(
        "\n=============================="
    )

    print(
        f"📊 {label}"
    )

    print(
        "=============================="
    )

    print(
        f"シグナル数: "
        f"{signals}"
    )

    print(
        f"WIN: "
        f"{wins}"
    )

    print(
        f"LOSS/TIMEOUT_LOSS: "
        f"{losses}"
    )

    print(
        f"HOLD: "
        f"{holds}"
    )

    print(
        f"勝率: "
        f"{win_rate:.2f}%"
    )

    print(
        f"平均リターン: "
        f"{avg_return:.2f}%"
    )

    print(
        f"PF: "
        f"{pf_text}"
    )

    print(
        f"最大DD: "
        f"{mdd:.2f}%"
    )

    print(
        f"平均保有日数: "
        f"{avg_hold_days:.2f}"
    )


    return {

        "signals":
            signals,

        "wins":
            wins,

        "losses":
            losses,

        "holds":
            holds,

        "win_rate":
            win_rate,

        "avg_return":
            avg_return,

        "profit_factor":
            pf,

        "max_drawdown":
            mdd,

        "avg_hold_days":
            avg_hold_days,
    }


# =========================================================
# MAIN
# =========================================================

print(
    "==========================================="
)

print(
    "🤖 STOCK AI WALK-FORWARD BACKTEST"
)

print(
    "==========================================="
)

print(
    f"期間: "
    f"{START_DATE} ～ "
    f"{END_DATE}"
)

print(
    f"TOP_N: "
    f"{TOP_N}"
)

print(
    "日経MA25 > MA75フィルター: ON"
)

print(
    f"ATR TP: "
    f"{ATR_TP_MULTIPLIER}x"
)

print(
    f"ATR SL: "
    f"{ATR_SL_MULTIPLIER}x"
)

print(
    f"モデル再学習間隔: "
    f"{REFIT_EVERY_TRADING_DAYS}"
    "営業日"
)

print(
    f"予測開始ウォームアップ: "
    f"{TEST_WARMUP_DAYS}"
    "営業日"
)

print(
    "買いシグナル限定: "
    f"{TRADE_ONLY_BUY_SIGNALS}"
)


if (
    REFIT_EVERY_TRADING_DAYS
    <= 5
):

    print(
        "⚠ 再学習間隔が短いため、"
        "実行時間が長くなる可能性があります"
    )


# =========================================================
# 日付
# =========================================================

start_ts = (
    pd.Timestamp(
        START_DATE
    )
)

end_ts = (
    pd.Timestamp(
        END_DATE
    )
)


history_start = (
    start_ts
    -
    pd.DateOffset(
        years=
        HISTORY_YEARS
    )
)


print(
    f"データ取得開始: "
    f"{history_start.date()}"
)


# =========================================================
# 日経
# =========================================================

nikkei = safe_download(
    "^N225",
    start=
        history_start.strftime(
            "%Y-%m-%d"
        ),
    end=
        (
            end_ts
            +
            pd.Timedelta(
                days=1
            )
        ).strftime(
            "%Y-%m-%d"
        ),
    interval="1d",
    auto_adjust=True,
)


# =========================================================
# 日経先物
# =========================================================

futures = safe_download(
    "NIY=F",
    start=
        history_start.strftime(
            "%Y-%m-%d"
        ),
    end=
        (
            end_ts
            +
            pd.Timedelta(
                days=1
            )
        ).strftime(
            "%Y-%m-%d"
        ),
    interval="1d",
    auto_adjust=True,
)


if nikkei is None:

    raise RuntimeError(
        "日経平均データ取得失敗"
    )


if futures is None:

    raise RuntimeError(
        "日経225先物データ取得失敗"
    )


nikkei = (
    build_nikkei_features(
        nikkei
    )
)


futures = (
    build_futures_features(
        futures
    )
)


# =========================================================
# 各銘柄データ準備
# =========================================================

symbol_data = {}


for ticker in TICKERS:

    print(
        f"データ準備: "
        f"{ticker}"
    )


    df = prepare_symbol_data(
        ticker,
        nikkei,
        futures,
        history_start.strftime(
            "%Y-%m-%d"
        ),
        (
            end_ts
            +
            pd.Timedelta(
                days=1
            )
        ).strftime(
            "%Y-%m-%d"
        ),
    )


    if df is not None:

        symbol_data[
            ticker
        ] = df

        print(
            f"  {len(df)} rows"
        )

    else:

        print(
            "  スキップ"
        )


if not symbol_data:

    raise RuntimeError(
        "利用可能な銘柄データがありません"
    )


# =========================================================
# 予測対象日
# =========================================================

all_dates = sorted(
    set().union(
        *[
            set(
                df.index
            )
            for df in
            symbol_data.values()
        ]
    )
)


prediction_dates_all = [
    d
    for d in
    all_dates
    if
    start_ts
    <=
    pd.Timestamp(d)
    <=
    end_ts
]


# =========================================================
# ウォームアップ
# =========================================================

if (
    len(
        prediction_dates_all
    )
    >
    TEST_WARMUP_DAYS
):

    prediction_dates = (
        prediction_dates_all[
            TEST_WARMUP_DAYS:
        ]
    )

else:

    prediction_dates = []

    print(
        "⚠ ウォームアップ期間が"
        "予測期間以上です。"
    )


print(
    f"予測日数: "
    f"{len(prediction_dates)}"
)


# =========================================================
# WALK FORWARD
# =========================================================

results = []

model = None

last_fit_pos = (
    -10**9
)


for pos, prediction_date in enumerate(
    prediction_dates
):

    prediction_date = (
        pd.Timestamp(
            prediction_date
        )
    )


    if (
        pos % 20
        ==
        0
    ):

        print(
            f"進捗: "
            f"{pos + 1}"
            f"/"
            f"{len(prediction_dates)} "
            f"{prediction_date.date()}"
        )


    # =====================================================
    # 再学習判定
    # =====================================================

    need_refit = (
        model is None
        or
        (
            pos
            -
            last_fit_pos
        )
        >=
        REFIT_EVERY_TRADING_DAYS
    )


    # =====================================================
    # モデル再学習
    # =====================================================

    if need_refit:

        train_frames = []


        for ticker, df in (
            symbol_data.items()
        ):

            # =================================================
            # 学習データ
            #
            # 【本番完全一致】
            #
            # 本番stock_scan.pyでは、その日その銘柄が
            # 流動性不足なら train_data.csv に行が
            # 追加されない(continueでスキップされる)。
            # つまり本番のtrain_data.csvは「流動的だった
            # 日の行」しか蓄積されない。これと一致させる
            # ため、学習データの選定にも liquid 条件を
            # 含める。
            #
            # ★target_date < prediction_date
            # は絶対条件。
            # =================================================

            prior = df.loc[
                (
                    df.index
                    <
                    prediction_date
                )
                &
                (
                    df[
                        "target_date"
                    ]
                    <
                    prediction_date
                )
                &
                (
                    df["liquid"]
                )
            ]


            if prior.empty:

                continue


            usable_prior = (
                prior[
                    prior[
                        "target_valid"
                    ]
                ]
                .copy()
            )


            if usable_prior.empty:

                continue


            train_piece = (
                usable_prior[
                    FEATURES
                    +
                    [
                        "target"
                    ]
                ]
                .assign(
                    date=
                        usable_prior.index,

                    ticker=
                        ticker,

                    target_valid=
                        True,
                )
            )


            train_frames.append(
                train_piece
            )


        if not train_frames:

            model = None

            continue


        train_all = pd.concat(
            train_frames,
            ignore_index=True
        )


        train_all = (
            train_all
            .dropna(
                subset=
                FEATURES
                +
                [
                    "target"
                ]
            )
        )


        # -----------------------------------------------------
        # クラス確認
        # -----------------------------------------------------

        if len(
            train_all
        ) < MIN_TRAIN_ROWS:

            model = None

            print(
                "⚠ 学習データ不足: "
                f"{len(train_all)}件"
            )

            continue


        class_values = (
            pd.to_numeric(
                train_all[
                    "target"
                ],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
        )


        if not {
            0,
            1,
            2
        }.issubset(
            set(class_values)
        ):

            model = None

            print(
                "⚠ 3クラス不足"
            )

            continue


        # -----------------------------------------------------
        # 学習
        # -----------------------------------------------------

        model = fit_model(
            train_all
            .assign(
                target_valid=
                    True
            )
        )


        if model is None:

            continue


        last_fit_pos = (
            pos
        )


        print(
            f"  ✅ model refit "
            f"{prediction_date.date()} "
            f"train_rows={len(train_all)} "
            f"classes={list(model.classes_)}"
        )


    # =====================================================
    # モデルなし
    # =====================================================

    if model is None:

        continue


    # =====================================================
    # 当日候補
    # =====================================================

    candidates = []


    for ticker, df in (
        symbol_data.items()
    ):

        if (
            prediction_date
            not in
            df.index
        ):

            continue


        row = (
            df.loc[
                prediction_date
            ]
        )


        # =================================================
        # 本番と同じ流動性判定
        #
        # 「その日の平均出来高20日」が不足なら
        # その銘柄を今回の予測対象から外す。
        #
        # 学習履歴は削除しない。
        # =================================================

        if not bool(
            row[
                "liquid"
            ]
        ):

            continue


        # =================================================
        # 特徴量
        # =================================================

        x = (
            row[
                FEATURES
            ]
            .to_frame()
            .T
        )


        if (
            x.isna()
            .any()
            .any()
        ):

            continue


        # =================================================
        # predict_proba
        # =================================================

        try:

            proba = (
                model
                .predict_proba(
                    x
                )[0]
            )

            classes = list(
                model.classes_
            )


            if not all(
                cls in classes
                for cls
                in [
                    0,
                    1,
                    2
                ]
            ):

                continue


            down_prob = (
                proba[
                    classes.index(0)
                ]
            )

            flat_prob = (
                proba[
                    classes.index(1)
                ]
            )

            up_prob = (
                proba[
                    classes.index(2)
                ]
            )


        except Exception as e:

            print(
                f"predict失敗 "
                f"{ticker} "
                f"{prediction_date.date()}: "
                f"{e}"
            )

            continue


        # =================================================
        # シグナル
        # =================================================

        signal = (
            calculate_signal(
                row,
                up_prob,
                down_prob,
                flat_prob
            )
        )


        signal.update(
            {
                "date":
                    prediction_date.strftime(
                        "%Y-%m-%d"
                    ),

                "ticker":
                    ticker,

                "company":
                    COMPANY_NAMES.get(
                        ticker,
                        ""
                    ),
            }
        )


        candidates.append(
            signal
        )


    if not candidates:

        continue


    # =====================================================
    # スコア順
    # =====================================================

    candidates.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )


    # =====================================================
    # BUY限定モード
    # =====================================================

    if (
        TRADE_ONLY_BUY_SIGNALS
    ):

        candidates = [
            x
            for x
            in
            candidates
            if
            x[
                "signal"
            ]
            in
            [
                "🔥 強い買い",
                "🟢 買い",
            ]
        ]


    # =====================================================
    # TOP_N
    # =====================================================

    selected = (
        candidates[
            :TOP_N
        ]
    )


    # =====================================================
    # 5営業日評価
    # =====================================================

    for rank, s in enumerate(
        selected,
        start=1
    ):

        ticker = (
            s["ticker"]
        )


        df = (
            symbol_data[
                ticker
            ]
        )


        result, ret, hold_days, exit_price = (
            evaluate_trade(
                df,
                prediction_date,
                s["price"],
                s["take_profit"],
                s["stop_loss"],
            )
        )


        results.append(
            {
                "date":
                    s["date"],

                "ticker":
                    ticker,

                "rank":
                    rank,

                "score":
                    round(
                        s[
                            "score"
                        ],
                        2
                    ),

                "signal":
                    s[
                        "signal"
                    ],

                "nikkei_uptrend":
                    bool(
                        s[
                            "nikkei_uptrend"
                        ]
                    ),

                "up_prob":
                    round(
                        s[
                            "up_prob"
                        ],
                        2
                    ),

                "flat_prob":
                    round(
                        s[
                            "flat_prob"
                        ],
                        2
                    ),

                "down_prob":
                    round(
                        s[
                            "down_prob"
                        ],
                        2
                    ),

                "price":
                    s[
                        "price"
                    ],

                "take_profit":
                    s[
                        "take_profit"
                    ],

                "stop_loss":
                    s[
                        "stop_loss"
                    ],

                "return":
                    (
                        round(
                            float(
                                ret
                            ),
                            4
                        )
                        if
                        pd.notna(
                            ret
                        )
                        else
                        np.nan
                    ),

                "hold_days":
                    hold_days,

                "result":
                    result,

                "exit_price":
                    exit_price,
            }
        )


# =========================================================
# 結果0件
# =========================================================

if not results:

    raise RuntimeError(
        "バックテスト結果が0件です"
    )


# =========================================================
# 結果DataFrame
# =========================================================

results_df = pd.DataFrame(
    results
)


results_df["date"] = (
    pd.to_datetime(
        results_df[
            "date"
        ]
    )
)


results_df = (
    results_df
    .sort_values(
        [
            "date",
            "rank"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 保存
# =========================================================

results_df.to_csv(
    "walk_forward_results.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# TOP1～TOP_N
# =========================================================

for rank in range(
    1,
    TOP_N + 1
):

    summarize(
        results_df[
            results_df[
                "rank"
            ]
            ==
            rank
        ].copy(),

        f"TOP{rank}"
    )


# =========================================================
# 全体
# =========================================================

summary_all = (
    summarize(
        results_df,
        "全体"
    )
)


# =========================================================
# 日経上昇トレンド
# =========================================================

summary_up = (
    summarize(
        results_df[
            results_df[
                "nikkei_uptrend"
            ]
            ==
            True
        ].copy(),

        "日経MA25 > MA75"
    )
)


# =========================================================
# 日経非上昇
# =========================================================

summary_down = (
    summarize(
        results_df[
            results_df[
                "nikkei_uptrend"
            ]
            ==
            False
        ].copy(),

        "日経MA25 <= MA75"
    )
)


# =========================================================
# シグナル別
# =========================================================

SIGNALS = [
    "🔥 強い買い",
    "🟢 買い",
    "🟡 監視",
    "🟡 監視(日経下落/レンジ)",
    "🟡 監視(横ばい優勢)",
    "🟡 監視(拮抗)",
    "🔴 買わない",
]


for signal in SIGNALS:

    part = (
        results_df[
            results_df[
                "signal"
            ]
            ==
            signal
        ].copy()
    )


    summarize(
        part,
        signal
    )


# =========================================================
# AIスコア別
# =========================================================

for threshold in [
    60,
    70,
    80
]:

    part = (
        results_df[
            results_df[
                "score"
            ]
            >=
            threshold
        ].copy()
    )


    summarize(
        part,
        f"AIスコア {threshold}以上"
    )


# =========================================================
# 月別
# =========================================================

results_df["month"] = (
    results_df[
        "date"
    ]
    .dt
    .to_period("M")
    .astype(str)
)


monthly = (
    results_df
    .groupby(
        "month"
    )
    .agg(
        signals=(
            "ticker",
            "size"
        ),

        avg_return=(
            "return",
            "mean"
        ),

        total_return=(
            "return",
            "sum"
        ),

        wins=(
            "result",
            lambda s:
            (
                s
                ==
                "WIN"
            ).sum()
        ),

        losses=(
            "result",
            lambda s:
            s.isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            ).sum()
        ),

        holds=(
            "result",
            lambda s:
            (
                s
                ==
                "HOLD"
            ).sum()
        ),
    )
    .reset_index()
)


monthly["win_rate"] = np.where(
    (
        monthly[
            "wins"
        ]
        +
        monthly[
            "losses"
        ]
    )
    >
    0,

    monthly[
        "wins"
    ]
    /
    (
        monthly[
            "wins"
        ]
        +
        monthly[
            "losses"
        ]
    )
    *
    100,

    0
)


monthly.to_csv(
    "walk_forward_monthly.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 年別
# =========================================================

results_df["year"] = (
    results_df[
        "date"
    ]
    .dt
    .year
)


yearly = (
    results_df
    .groupby(
        "year"
    )
    .agg(
        signals=(
            "ticker",
            "size"
        ),

        avg_return=(
            "return",
            "mean"
        ),

        total_return=(
            "return",
            "sum"
        ),

        wins=(
            "result",
            lambda s:
            (
                s
                ==
                "WIN"
            ).sum()
        ),

        losses=(
            "result",
            lambda s:
            s.isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            ).sum()
        ),

        holds=(
            "result",
            lambda s:
            (
                s
                ==
                "HOLD"
            ).sum()
        ),
    )
    .reset_index()
)


yearly["win_rate"] = np.where(
    (
        yearly[
            "wins"
        ]
        +
        yearly[
            "losses"
        ]
    )
    >
    0,

    yearly[
        "wins"
    ]
    /
    (
        yearly[
            "wins"
        ]
        +
        yearly[
            "losses"
        ]
    )
    *
    100,

    0
)


yearly.to_csv(
    "walk_forward_yearly.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 年・日経トレンド別
# =========================================================

trend_yearly = (
    results_df
    .groupby(
        [
            "year",
            "nikkei_uptrend"
        ]
    )
    .agg(
        signals=(
            "ticker",
            "size"
        ),

        avg_return=(
            "return",
            "mean"
        ),

        total_return=(
            "return",
            "sum"
        ),

        wins=(
            "result",
            lambda s:
            (
                s
                ==
                "WIN"
            ).sum()
        ),

        losses=(
            "result",
            lambda s:
            s.isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            ).sum()
        ),
    )
    .reset_index()
)


trend_yearly["win_rate"] = np.where(
    (
        trend_yearly[
            "wins"
        ]
        +
        trend_yearly[
            "losses"
        ]
    )
    >
    0,

    trend_yearly[
        "wins"
    ]
    /
    (
        trend_yearly[
            "wins"
        ]
        +
        trend_yearly[
            "losses"
        ]
    )
    *
    100,

    0
)


trend_yearly.to_csv(
    "walk_forward_trend_yearly.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 最終サマリー
# =========================================================

print(
    "\n==========================================="
)

print(
    "📌 WALK-FORWARD FINAL SUMMARY"
)

print(
    "==========================================="
)

print(
    f"期間: "
    f"{START_DATE} ～ "
    f"{END_DATE}"
)

print(
    f"予測件数: "
    f"{len(results_df)}"
)

print(
    f"TOP_N: "
    f"{TOP_N}"
)

print(
    "日経MA25 > MA75: ON"
)

print(
    f"REFIT: "
    f"{REFIT_EVERY_TRADING_DAYS}営業日"
)

print(
    f"BUY限定: "
    f"{TRADE_ONLY_BUY_SIGNALS}"
)


if summary_all:

    pf_all = (
        summary_all[
            "profit_factor"
        ]
    )


    if np.isinf(
        pf_all
    ):

        pf_text = "inf"

    else:

        pf_text = (
            f"{pf_all:.2f}"
        )


    print(
        f"\n勝率: "
        f"{summary_all['win_rate']:.2f}%"
    )

    print(
        f"平均リターン: "
        f"{summary_all['avg_return']:.2f}%"
    )

    print(
        f"PF: "
        f"{pf_text}"
    )

    print(
        f"最大DD: "
        f"{summary_all['max_drawdown']:.2f}%"
    )


# =========================================================
# 追加比較
# =========================================================

if summary_up:

    print(
        "\n📈 日経上昇トレンド時"
    )

    print(
        f"勝率: "
        f"{summary_up['win_rate']:.2f}%"
    )

    print(
        f"平均リターン: "
        f"{summary_up['avg_return']:.2f}%"
    )

    print(
        f"PF: "
        f"{(
            'inf'
            if np.isinf(
                summary_up[
                    'profit_factor'
                ]
            )
            else
            f"{summary_up['profit_factor']:.2f}"
        )}"
    )

    print(
        f"最大DD: "
        f"{summary_up['max_drawdown']:.2f}%"
    )


if summary_down:

    print(
        "\n📉 日経非上昇トレンド時"
    )

    print(
        f"勝率: "
        f"{summary_down['win_rate']:.2f}%"
    )

    print(
        f"平均リターン: "
        f"{summary_down['avg_return']:.2f}%"
    )

    print(
        f"PF: "
        f"{(
            'inf'
            if np.isinf(
                summary_down[
                    'profit_factor'
                ]
            )
            else
            f"{summary_down['profit_factor']:.2f}"
        )}"
    )

    print(
        f"最大DD: "
        f"{summary_down['max_drawdown']:.2f}%"
    )


# =========================================================
# 保存ファイル
# =========================================================

print(
    "\n✅ walk_forward_results.csv 保存"
)

print(
    "✅ walk_forward_monthly.csv 保存"
)

print(
    "✅ walk_forward_yearly.csv 保存"
)

print(
    "✅ walk_forward_trend_yearly.csv 保存"
)
