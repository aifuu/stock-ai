import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier


# =====================
# Discord
# =====================
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send(msg):
    if not WEBHOOK_URL:
        print("❌ Webhookなし")
        return

    if len(msg) > 1900:
        msg = msg[:1900]

    r = requests.post(
        WEBHOOK_URL,
        json={"content": msg},
        timeout=30
    )

    print("Discord status =", r.status_code)

    if r.status_code == 204:
        print("✅ Discord送信成功")
    else:
        print("❌ Discord送信失敗")
        print(r.text)


# =====================
# 銘柄
# =====================
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


TRAIN_FILE = "train_data.csv"
MODEL_FILE = "model.pkl"


# =====================
# 3クラス分類設定
# =====================
DOWN_THRESHOLD = -1.5
UP_THRESHOLD = 1.5
FORWARD_DAYS = 3


# =====================
# 学習・予測で使う特徴量
# =====================
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
    "nikkei_kairi25",
    "nikkei_rsi",
    "nikkei_macd",
    "nikkei_return_5d",
    "future_return",
    "future_ma5",
    "future_rsi",
    "future_gap",
]


# =====================
# ダウンロード失敗時のリトライ
# =====================
def safe_download(ticker, retries=3, wait_sec=3, **kwargs):

    for attempt in range(1, retries + 1):

        try:
            df = yf.download(
                ticker,
                **kwargs
            )

            if df is not None and not df.empty:
                return df

            print(
                f"{ticker} 空データ "
                f"(試行{attempt}/{retries})"
            )

        except Exception as e:

            print(
                f"{ticker} 取得失敗 "
                f"(試行{attempt}/{retries}): {e}"
            )

        if attempt < retries:
            time.sleep(wait_sec)

    print(
        f"❌ {ticker} "
        f"リトライ{retries}回失敗"
    )

    return None


# =====================
# RSI
# =====================
def calc_rsi(close, period=14):

    close = close.squeeze()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


# =====================
# ADX
# =====================
def calc_adx(df, period=14):

    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
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

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_dm_sm = plus_dm.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    minus_dm_sm = minus_dm.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        plus_dm_sm /
        atr.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm_sm /
        atr.replace(0, np.nan)
    )

    dx = (
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    ) * 100

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return adx


# =====================
# 特徴量作成
# =====================
def create_features(df):

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["ret1"] = close.pct_change()

    df["ma25"] = (
        close.rolling(25).mean()
    )

    df["ma75"] = (
        close.rolling(75).mean()
    )

    df["vol_ratio"] = (
        volume /
        volume.rolling(20).mean()
    )

    df["rsi"] = calc_rsi(close)

    df["adx"] = calc_adx(df)

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["high252"] = (
        close.rolling(252).max()
    )

    df["low252"] = (
        close.rolling(252).min()
    )

    df["from_high"] = (
        close /
        df["high252"] -
        1
    ) * 100

    df["from_low"] = (
        close /
        df["low252"] -
        1
    ) * 100

    return df


# =====================
# AIスコア
# =====================
def calc_score(
    df,
    close,
    up_prob,
    down_prob,
    flat_prob
):

    price = float(
        close.dropna().iloc[-1]
    )

    rsi = float(
        df["rsi"].iloc[-1]
    )

    macd = float(
        df["macd"].iloc[-1]
    )

    signal = float(
        df["signal"].iloc[-1]
    )

    ma25 = float(
        df["ma25"].iloc[-1]
    )

    ma75 = float(
        df["ma75"].iloc[-1]
    )

    vol_ratio = float(
        df["vol_ratio"].iloc[-1]
    )

    high52 = float(
        close.rolling(252).max().iloc[-1]
    )

    distance = (
        price / high52 - 1
    ) * 100

    score = 0

    # RSI
    if rsi < 35:
        score += 25

    # MACD
    if macd > signal:
        score += 25

    # MA
    if ma25 > ma75:
        score += 20

    # 出来高
    if vol_ratio > 1.5:
        score += 20

    # 52週高値からの距離
    if distance > -10:
        score += 15

    elif distance > -20:
        score += 8

    # 日経RSI
    if float(
        df["nikkei_rsi"].iloc[-1]
    ) > 50:
        score += 5

    # 日経5日リターン
    if float(
        df["nikkei_return_5d"].iloc[-1]
    ) > 0:
        score += 5

    # =====================
    # 上昇確率を加点
    # =====================
    score += up_prob * 50

    # =====================
    # 利確 / 損切
    # =====================
    take_profit = round(
        price * 1.08,
        0
    )

    stop_loss = round(
        price * 0.96,
        0
    )

    return {

        "score": round(
            score,
            1
        ),

        "price": round(
            price,
            0
        ),

        "rsi": round(
            rsi,
            1
        ),

        "vol": round(
            vol_ratio,
            2
        ),

        "take_profit": take_profit,

        "stop_loss": stop_loss,

        "up_prob": round(
            up_prob * 100,
            1
        ),

        "flat_prob": round(
            flat_prob * 100,
            1
        ),

        "down_prob": round(
            down_prob * 100,
            1
        )
    }


# =========================
# 過去予測の5営業日以内結果判定
# =========================
def update_prediction_results():

    file = "prediction_history.csv"

    if not os.path.exists(file):

        print(
            "prediction_history.csv がありません"
        )

        return

    history = pd.read_csv(file)

    history["result"] = (
        history["result"]
        .astype("object")
    )

    history["return"] = pd.to_numeric(
        history["return"],
        errors="coerce"
    )

    history["hold_days"] = pd.to_numeric(
        history["hold_days"],
        errors="coerce"
    )

    required_columns = [
        "date",
        "ticker",
        "price",
        "take_profit",
        "stop_loss",
        "result",
        "return",
        "hold_days",
        "rank"
    ]

    for col in required_columns:

        if col not in history.columns:

            print(
                f"必要な列がありません: {col}"
            )

            return

    today = pd.Timestamp.now().normalize()

    # =========================
    # 過去予測を1件ずつ判定
    # =========================
    for i, row in history.iterrows():

        if (
            pd.notna(row["result"])
            and
            str(row["result"]).strip() != ""
        ):
            continue

        try:

            prediction_date = (
                pd.to_datetime(
                    row["date"]
                ).normalize()
            )

            ticker = str(
                row["ticker"]
            )

            entry_price = float(
                row["price"]
            )

            take_profit = float(
                row["take_profit"]
            )

            stop_loss = float(
                row["stop_loss"]
            )

        except Exception as e:

            print(
                f"履歴データ読み込みエラー: {e}"
            )

            continue

        # =========================
        # 予測日の翌営業日から5営業日
        # =========================
        business_days = pd.bdate_range(
            start=(
                prediction_date
                + pd.Timedelta(days=1)
            ),
            periods=5
        )

        if business_days[-1] > today:
            continue

        start_date = business_days[0]

        end_date = (
            business_days[-1]
            + pd.Timedelta(days=1)
        )

        data = safe_download(
            ticker,
            start=start_date.strftime(
                "%Y-%m-%d"
            ),
            end=end_date.strftime(
                "%Y-%m-%d"
            ),
            auto_adjust=False,
            progress=False
        )

        if data is None or data.empty:

            print(
                f"{ticker} 株価データなし"
            )

            continue

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )

        result = None
        return_rate = None
        hold_days = None

        check_days = min(
            5,
            len(data)
        )

        # =========================
        # 5営業日チェック
        # =========================
        for day_index in range(
            check_days
        ):

            day = data.iloc[
                day_index
            ]

            try:

                high = float(
                    day["High"]
                )

                low = float(
                    day["Low"]
                )

            except Exception:

                continue

            # =========================
            # 利確
            # =========================
            if high >= take_profit:

                result = "WIN"

                return_rate = (
                    (
                        take_profit
                        - entry_price
                    )
                    / entry_price
                    * 100
                )

                hold_days = (
                    day_index + 1
                )

                break

            # =========================
            # 損切
            # =========================
            if low <= stop_loss:

                result = "LOSS"

                return_rate = (
                    (
                        stop_loss
                        - entry_price
                    )
                    / entry_price
                    * 100
                )

                hold_days = (
                    day_index + 1
                )

                break

        # =========================
        # TP / SLに届かなかった
        # =========================
        if result is None:

            try:

                close_price = float(
                    data.iloc[
                        check_days - 1
                    ]["Close"]
                )

            except Exception:

                continue

            return_rate = (
                (
                    close_price
                    - entry_price
                )
                / entry_price
                * 100
            )

            hold_days = check_days

            # =========================
            # 5営業日経過＋マイナス
            # =========================
            if (
                hold_days >= 5
                and return_rate < 0
            ):

                result = "TIMEOUT_LOSS"

            else:

                result = "HOLD"

        history.at[
            i,
            "result"
        ] = result

        history.at[
            i,
            "return"
        ] = float(
            round(
                return_rate,
                2
            )
        )

        history.at[
            i,
            "hold_days"
        ] = float(
            hold_days
        )

        print(
            f"判定: "
            f"{prediction_date.date()} "
            f"{ticker} "
            f"{result} "
            f"{return_rate:.2f}% "
            f"{hold_days}日"
        )

    history.to_csv(
        file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "過去予測の結果判定完了"
    )


# =========================
# 過去予測結果更新
# =========================
update_prediction_results()


# =====================
# 日経平均
# =====================
nikkei = safe_download(
    "^N225",
    period="3y",
    interval="1d",
    auto_adjust=True
)

if nikkei is None:

    send(
        "❌ 日経平均データ取得失敗"
        "のため処理中断"
    )

    exit()

if isinstance(
    nikkei.columns,
    pd.MultiIndex
):

    nikkei.columns = (
        nikkei.columns
        .get_level_values(0)
    )

nikkei_close = (
    nikkei["Close"]
    .squeeze()
)

nikkei["nikkei_ma25"] = (
    nikkei_close
    .rolling(25)
    .mean()
)

nikkei["nikkei_kairi25"] = (
    (
        nikkei_close
        - nikkei["nikkei_ma25"]
    )
    / nikkei["nikkei_ma25"]
    * 100
)

nikkei["nikkei_rsi"] = (
    calc_rsi(nikkei_close)
)

ema12_n = (
    nikkei_close
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

ema26_n = (
    nikkei_close
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

nikkei["nikkei_macd"] = (
    ema12_n - ema26_n
)

nikkei["nikkei_return_5d"] = (
    nikkei_close.pct_change(5)
    * 100
)


# =====================
# 日経225先物
# =====================
futures = safe_download(
    "NIY=F",
    period="3y",
    interval="1d",
    auto_adjust=True
)

if futures is None:

    send(
        "❌ 先物データ取得失敗"
        "のため処理中断"
    )

    exit()

if isinstance(
    futures.columns,
    pd.MultiIndex
):

    futures.columns = (
        futures.columns
        .get_level_values(0)
    )

future_close = (
    futures["Close"]
    .squeeze()
)

futures["future_return"] = (
    future_close.pct_change()
)

futures["future_ma5"] = (
    future_close
    .rolling(5)
    .mean()
)

futures["future_rsi"] = (
    calc_rsi(future_close)
)

futures["future_gap"] = (
    (
        future_close
        - future_close.shift(1)
    )
    /
    future_close.shift(1)
)

# =====================
# 先物は1日ラグ
# =====================
futures[
    "future_return"
] = futures[
    "future_return"
].shift(1)

futures[
    "future_ma5"
] = futures[
    "future_ma5"
].shift(1)

futures[
    "future_rsi"
] = futures[
    "future_rsi"
].shift(1)

futures[
    "future_gap"
] = futures[
    "future_gap"
].shift(1)


# =====================
# 学習データ読み込み
# =====================
def load_training_data():

    if not os.path.exists(
        TRAIN_FILE
    ):

        return None, None

    df = pd.read_csv(
        TRAIN_FILE
    )

    if len(df) == 0:

        return None, None

    required_cols = (
        FEATURES + ["target"]
    )

    missing = [
        c
        for c in required_cols
        if c not in df.columns
    ]

    if missing:

        print(
            "⚠ train_data.csvの"
            "スキーマが古いため無視します"
        )

        print(
            f"不足列: {missing}"
        )

        return None, None

    df = df.dropna()

    if len(df) == 0:

        return None, None

    # =====================
    # 3クラス専用
    # =====================
    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )

    df = df[
        df["target"].isin(
            [0, 1, 2]
        )
    ].copy()

    if len(df) == 0:

        return None, None

    X = df[FEATURES]

    y = df["target"].astype(int)

    # =====================
    # 3クラス揃っているか確認
    # =====================
    if y.nunique() < 3:

        print(
            "⚠ 3クラス "
            "(0/1/2) が揃っていないため、"
            "既存学習データは使用しません"
        )

        return None, None

    return X, y


# =====================
# 学習データ保存
# =====================
def save_training_data(new_df):

    if os.path.exists(
        TRAIN_FILE
    ):

        old_df = pd.read_csv(
            TRAIN_FILE
        )

        # =====================
        # 旧2クラスデータを混ぜない
        # =====================
        if (
            "target" in old_df.columns
            and
            set(
                pd.to_numeric(
                    old_df["target"],
                    errors="coerce"
                )
                .dropna()
                .unique()
            )
            - {0, 1, 2}
        ):

            print(
                "⚠ 旧形式の学習データを"
                "破棄して3クラス用に再構築します"
            )

            old_df = pd.DataFrame()

        df_all = pd.concat(
            [
                old_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        df_all = new_df

    df_all = df_all.drop_duplicates(
        subset=[
            "date",
            "ticker"
        ],
        keep="last"
    )

    df_all.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✅ train_data.csv 更新: "
        f"{len(df_all)}件"
    )


# =====================
# モデル
# =====================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42,
    class_weight="balanced"
)

model_ready = False

results = []
all_data = []
all_train_rows = []


# =====================
# メイン処理
# =====================
for ticker in TICKERS:

    try:

        print(
            "解析中:",
            ticker
        )

        df = safe_download(
            ticker,
            period="3y",
            interval="1d",
            auto_adjust=True
        )

        if (
            df is None
            or
            len(df) < 150
        ):

            continue

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df = create_features(df)

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        close = (
            df["Close"]
            .squeeze()
        )

        volume = (
            df["Volume"]
            .squeeze()
        )

        # =====================
        # 日経特徴量
        # =====================
        df = df.join(
            nikkei[
                [
                    "nikkei_kairi25",
                    "nikkei_rsi",
                    "nikkei_macd",
                    "nikkei_return_5d"
                ]
            ],
            how="left"
        )

        # =====================
        # 先物特徴量
        # =====================
        df = df.join(
            futures[
                [
                    "future_return",
                    "future_ma5",
                    "future_rsi",
                    "future_gap"
                ]
            ],
            how="left"
        )

        df = df.dropna()

        print(
            ticker,
            "dropna後データ数",
            len(df)
        )

        # =====================
        # 3営業日後の騰落率
        # =====================
        future_price = (
            df["Close"]
            .shift(-FORWARD_DAYS)
        )

        future_return = (
            (
                future_price
                /
                df["Close"]
                - 1
            )
            * 100
        )

        # =====================
        # 3クラス分類
        #
        # 0 = 下落
        # 1 = 横ばい
        # 2 = 上昇
        # =====================
        df["target"] = np.select(
            [
                future_return
                <= DOWN_THRESHOLD,

                future_return
                >= UP_THRESHOLD
            ],
            [
                0,
                2
            ],
            default=1
        )

        # =====================
        # 最後の3日間は
        # 将来価格が存在しないため除外
        # =====================
        df = df[
            future_return.notna()
        ].copy()

        df["target"] = (
            df["target"]
            .astype(int)
        )

        if len(df) < 100:

            print(
                ticker,
                "学習データ不足"
            )

            continue

        print(
            ticker,
            "学習データ数=",
            len(df)
        )

        # =====================
        # クラス分布
        # =====================
        class_counts = (
            df["target"]
            .value_counts()
            .sort_index()
        )

        print(
            ticker,
            "クラス分布:",
            class_counts.to_dict()
        )

        # =====================
        # 3クラス全部必要
        # =====================
        if df["target"].nunique() < 3:

            print(
                ticker,
                "3クラス揃わないためスキップ"
            )

            continue

        X = df[FEATURES]

        y = df["target"]

        train_rows = X.copy()

        train_rows["target"] = (
            y.values
        )

        train_rows["date"] = (
            X.index.strftime(
                "%Y-%m-%d"
            )
        )

        train_rows["ticker"] = ticker

        all_train_rows.append(
            train_rows
        )

        # =====================
        # 最新データ
        # =====================
        latest = X.iloc[
            -1:
        ].copy()

        all_data.append(
            {
                "ticker": ticker,
                "latest": latest,
                "close": close,
                "df": df.copy()
            }
        )

        print(
            "保存:",
            ticker
        )

    except Exception as e:

        print(
            ticker,
            "エラー:",
            e
        )


# =====================
# 学習データ保存
# =====================
if all_train_rows:

    new_train_df = pd.concat(
        all_train_rows,
        ignore_index=True
    )

    save_training_data(
        new_train_df
    )


# =====================
# 学習データ読み込み
# =====================
X_all, y_all = (
    load_training_data()
)


# =====================
# 3クラスモデル学習
# =====================
if (
    X_all is not None
    and
    len(X_all) >= 100
    and
    y_all.nunique() == 3
):

    print(
        "====================="
    )

    print(
        "3クラス分類モデル学習開始"
    )

    print(
        f"学習データ数: {len(X_all)}"
    )

    print(
        "クラス分布:",
        y_all.value_counts()
        .sort_index()
        .to_dict()
    )

    model.fit(
        X_all,
        y_all
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    model_ready = True

    print(
        "✅ 3クラス分類モデル学習完了"
    )

    print(
        "model.classes_ =",
        model.classes_
    )

elif os.path.exists(
    MODEL_FILE
):

    try:

        model = joblib.load(
            MODEL_FILE
        )

        # =====================
        # 旧2クラスモデルを使用しない
        # =====================
        if not np.array_equal(
            model.classes_,
            np.array([0, 1, 2])
        ):

            print(
                "❌ 既存model.pklが"
                "3クラスモデルではありません"
            )

            print(
                "classes =",
                model.classes_
            )

            model_ready = False

        else:

            model_ready = True

            print(
                "⚠ 新規学習条件を満たさないため、"
                "前回の3クラスモデルを使用"
            )

    except Exception as e:

        print(
            "❌ 前回モデル読み込み失敗:",
            e
        )

else:

    print(
        "❌ 3クラス学習データ・"
        "モデルともになし"
    )

    print(
        "今回は予測をスキップ"
    )


# =====================
# 一括予測
# =====================
if model_ready:

    for item in all_data:

        latest = item["latest"]
        df = item["df"]
        close = item["close"]
        ticker = item["ticker"]

        try:

            # =====================
            # クラス別確率
            # =====================
            probabilities = (
                model.predict_proba(
                    latest
                )[0]
            )

            classes = list(
                model.classes_
            )

            # =====================
            # 0 = 下落
            # 1 = 横ばい
            # 2 = 上昇
            # =====================
            down_index = classes.index(
                0
            )

            flat_index = classes.index(
                1
            )

            up_index = classes.index(
                2
            )

            down_prob = (
                probabilities[
                    down_index
                ]
            )

            flat_prob = (
                probabilities[
                    flat_index
                ]
            )

            up_prob = (
                probabilities[
                    up_index
                ]
            )

        except (
            ValueError,
            IndexError,
            AttributeError
        ) as e:

            print(
                f"{ticker} "
                f"predict_proba失敗: {e}"
            )

            continue

        # =====================
        # スコア計算
        # =====================
        data = calc_score(
            df,
            close,
            up_prob,
            down_prob,
            flat_prob
        )

        results.append(
            {
                "ticker": ticker,

                "score": data["score"],

                "prob": data["up_prob"],

                "up_prob": data["up_prob"],

                "flat_prob": data["flat_prob"],

                "down_prob": data["down_prob"],

                "price": data["price"],

                "rsi": data["rsi"],

                "vol": data["vol"],

                "take_profit":
                    data["take_profit"],

                "stop_loss":
                    data["stop_loss"]
            }
        )

        print(
            f"{ticker} "
            f"score={data['score']} "
            f"下落={data['down_prob']:.1f}% "
            f"横ばい={data['flat_prob']:.1f}% "
            f"上昇={data['up_prob']:.1f}%"
        )


# =====================
# 結果
# =====================
if not results:

    send(
        "⚪ データなし"
    )

    exit()


results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

top = results[:3]


# =====================
# Discordメッセージ
# =====================
msg = (
    f"⏰ JST: "
    f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}"
    "\n\n"
)

msg += (
    "📊 AI株スキャン結果\n"
    "【3クラス分類】\n\n"
)


for i, r in enumerate(top):

    if r["score"] >= 60:

        rank = "🔥 強い買い"

    elif r["score"] >= 45:

        rank = "🟢 買い候補"

    elif r["score"] >= 35:

        rank = "🟡 監視"

    else:

        continue

    msg += f"""
━━━━━━━━━━━━━━
#{i+1} {r['ticker']} {COMPANY_NAMES.get(r['ticker'], '')}

{rank}

AIスコア: {r['score']}

下落確率: {r['down_prob']}%
横ばい確率: {r['flat_prob']}%
上昇確率: {r['up_prob']}%

買値: {r['price']}
利確: {r['take_profit']}
損切: {r['stop_loss']}

RSI: {r['rsi']}
出来高倍率: {r['vol']}
━━━━━━━━━━━━━━
"""


# =====================
# 予測履歴保存
# =====================
history_file = (
    "prediction_history.csv"
)

save_rows = []

for rank, r in enumerate(
    top,
    start=1
):

    save_rows.append(
        {
            "date":
                datetime.now(
                    ZoneInfo("Asia/Tokyo")
                ).strftime(
                    "%Y-%m-%d"
                ),

            "ticker":
                r["ticker"],

            "rank":
                rank,

            "score":
                r["score"],

            "probability":
                r["up_prob"],

            "down_probability":
                r["down_prob"],

            "flat_probability":
                r["flat_prob"],

            "up_probability":
                r["up_prob"],

            "price":
                r["price"],

            "take_profit":
                r["take_profit"],

            "stop_loss":
                r["stop_loss"],

            "result":
                "",

            "return":
                np.nan,

            "hold_days":
                np.nan
        }
    )


new_df = pd.DataFrame(
    save_rows
)


if os.path.exists(
    history_file
):

    old_df = pd.read_csv(
        history_file
    )

    df_all = pd.concat(
        [
            old_df,
            new_df
        ],
        ignore_index=True
    )

    df_all = df_all.drop_duplicates(
        subset=[
            "date",
            "ticker"
        ],
        keep="last"
    )

else:

    df_all = new_df


df_all.to_csv(
    history_file,
    index=False,
    encoding="utf-8-sig"
)


# =====================
# AI成績評価
# =====================
def show_ai_performance():

    file = (
        "prediction_history.csv"
    )

    if not os.path.exists(
        file
    ):

        return ""

    df = pd.read_csv(
        file
    )

    result_df = df[
        df["result"].notna()
        &
        (
            df["result"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    if len(result_df) == 0:

        return """
📊 AI実績

まだ判定データなし
"""


    result_df["return"] = (
        pd.to_numeric(
            result_df["return"],
            errors="coerce"
        )
    )

    result_df["hold_days"] = (
        pd.to_numeric(
            result_df["hold_days"],
            errors="coerce"
        )
    )

    result_df["rank"] = (
        pd.to_numeric(
            result_df["rank"],
            errors="coerce"
        )
    )

    total = len(
        result_df
    )

    wins = (
        result_df["result"]
        == "WIN"
    ).sum()

    losses = (
        result_df["result"]
        .isin(
            [
                "LOSS",
                "TIMEOUT_LOSS"
            ]
        )
    ).sum()

    holds = (
        result_df["result"]
        == "HOLD"
    ).sum()

    decided = (
        wins + losses
    )

    win_rate = (
        wins / decided * 100
        if decided > 0
        else 0
    )

    returns = (
        result_df["return"]
        .dropna()
    )

    if len(returns) > 0:

        avg_return = (
            returns.mean()
        )

        best = (
            returns.max()
        )

        worst = (
            returns.min()
        )

    else:

        avg_return = 0
        best = 0
        worst = 0


    days = (
        result_df["hold_days"]
        .dropna()
    )

    avg_days = (
        days.mean()
        if len(days) > 0
        else 0
    )


    rank_text = ""


    for rank in [1, 2, 3]:

        rank_df = (
            result_df[
                result_df["rank"]
                == rank
            ]
        )

        rank_total = len(
            rank_df
        )

        if rank_total == 0:

            rank_text += f"""
#{rank}位

データなし
"""

            continue


        rank_wins = (
            rank_df["result"]
            == "WIN"
        ).sum()

        rank_losses = (
            rank_df["result"]
            .isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            )
        ).sum()

        rank_holds = (
            rank_df["result"]
            == "HOLD"
        ).sum()

        rank_decided = (
            rank_wins
            + rank_losses
        )

        rank_win_rate = (
            rank_wins
            / rank_decided
            * 100
            if rank_decided > 0
            else 0
        )

        rank_returns = (
            rank_df["return"]
            .dropna()
        )

        if len(rank_returns) > 0:

            rank_avg_return = (
                rank_returns.mean()
            )

            rank_best = (
                rank_returns.max()
            )

            rank_worst = (
                rank_returns.min()
            )

        else:

            rank_avg_return = 0
            rank_best = 0
            rank_worst = 0


        rank_days = (
            rank_df["hold_days"]
            .dropna()
        )

        rank_avg_days = (
            rank_days.mean()
            if len(rank_days) > 0
            else 0
        )


        rank_text += f"""
#{rank}位

勝率: {rank_win_rate:.1f}%
WIN: {rank_wins}件
LOSS: {rank_losses}件
HOLD: {rank_holds}件
判定数: {rank_total}件

平均利益率: {rank_avg_return:.2f}%
平均保有日数: {rank_avg_days:.1f}日
最高利益: {rank_best:+.2f}%
最大損失: {rank_worst:.2f}%
"""


    return f"""
━━━━━━━━━━━━━━
📊 AI実績
（3クラス分類・TOP3・5営業日判定）
━━━━━━━━━━━━━━

【全体】

判定数: {total}件

勝ち: {wins}件
負け: {losses}件
HOLD: {holds}件

勝率: {win_rate:.1f}%

平均利益率: {avg_return:.2f}%
平均保有日数: {avg_days:.1f}日

最高利益: {best:+.2f}%
最大損失: {worst:.2f}%

━━━━━━━━━━━━━━
🏆 TOP3順位別
━━━━━━━━━━━━━━
{rank_text}
━━━━━━━━━━━━━━
"""


# =====================
# 最終表示・Discord送信
# =====================
performance = (
    show_ai_performance()
)

msg += performance

print(msg)

send(msg)
