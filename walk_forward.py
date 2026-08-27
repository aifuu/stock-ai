import os
import time
import warnings
 
import numpy as np
import pandas as pd
import requests
import yfinance as yf
 
from sklearn.ensemble import RandomForestClassifier
 
warnings.filterwarnings("ignore")
 
 
# =========================================================
# Discord
# =========================================================
 
WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK"
)
 
 
def send_discord(message):
 
    if not WEBHOOK_URL:
 
        print(
            "⚠ DISCORD_WEBHOOK が設定されていないため、"
            "Discord通知をスキップします"
        )
 
        return
 
    try:
 
        if len(message) > 1900:
            message = message[:1900]
 
        response = requests.post(
            WEBHOOK_URL,
            json={
                "content": message
            },
            timeout=30
        )
 
        print(
            f"Discord status = "
            f"{response.status_code}"
        )
 
        if response.status_code == 204:
 
            print(
                "✅ Discord送信成功"
            )
 
        else:
 
            print(
                "❌ Discord送信失敗"
            )
 
            print(
                response.text
            )
 
    except Exception as e:
 
        print(
            f"❌ Discord通知エラー: {e}"
        )
 
 
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
# ウォームアップ
# =========================================================
 
TEST_WARMUP_DAYS = int(
    os.getenv(
        "WF_WARMUP_DAYS",
        "252"
    )
)
 
 
# =========================================================
# 再学習間隔
# =========================================================
 
REFIT_EVERY_TRADING_DAYS = int(
    os.getenv(
        "WF_REFIT_EVERY_TRADING_DAYS",
        "20"
    )
)
 
 
# =========================================================
# RandomForest
# =========================================================
 
N_ESTIMATORS = 300
 
MAX_DEPTH = 7
 
RANDOM_STATE = 42
 
 
# =========================================================
# 比較対象
#
# ★改善点⑤
#
# UP確率のしきい値・日経フィルターに加えて、
# 「テスタ型モメンタムしきい値」を第3の比較軸として追加。
# TESTA_THRESHOLDS の None は「モメンタムフィルターなし
# (BASELINE)」を意味する。
# stock_scan.py本番では TESTA_MIN_SCORE_FOR_BUY=55 を
# 固定で使っているが、ここでは 55/65/75 を比較して、
# 「取引件数を確保しつつPF・DDが改善するライン」を探す。
# =========================================================
 
UP_THRESHOLDS = [
    50,
    55,
    60,
    65
]
 
NIKKEI_FILTER_OPTIONS = [
    True,
    False
]
 
TESTA_THRESHOLDS = [
    None,   # BASELINE(モメンタムフィルターなし)
    55,
    65,
    75,
]
 
 
# =========================================================
# DEV / VALIDATION / OOS
# =========================================================
 
OOS_DAYS = int(
    os.getenv(
        "WF_OOS_DAYS",
        "90"
    )
)
 
DEV_SPLIT_RATIO = float(
    os.getenv(
        "WF_DEV_SPLIT_RATIO",
        "0.6"
    )
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
    "6613.T",
    "6758.T",
    "6861.T",
    "6857.T",
    "8035.T",
    "6920.T",
    "6146.T",
    "6501.T",
    "6503.T",
    "6701.T",
    "6702.T",
    "6902.T",
    "6901.T",
    "7270.T",
    "7267.T",
    "7201.T",
    "7202.T",
    "7205.T",
    "7211.T",
    "7261.T",
    "7272.T",
    "8316.T",
    "8411.T",
    "8331.T",
    "8308.T",
    "8309.T",
    "8354.T",
    "8355.T",
    "7182.T",
    "7186.T",
    "8697.T",
    "8001.T",
    "8002.T",
    "8015.T",
    "2768.T",
    "8053.T",
    "8056.T",
    "8032.T",
    "8012.T",
    "8014.T",
    "8037.T",
    "9432.T",
    "9433.T",
    "9434.T",
    "9613.T",
    "9983.T",
    "4755.T",
    "4689.T",
    "6098.T",
    "2413.T",
    "3659.T",
    "4063.T",
    "4188.T",
    "4005.T",
    "4004.T",
    "4204.T",
    "4502.T",
    "4503.T",
    "4519.T",
    "4523.T",
    "4568.T",
    "5401.T",
    "5411.T",
    "5711.T",
    "5801.T",
    "5802.T",
    "5713.T",
    "6301.T",
    "6302.T",
    "6367.T",
    "7011.T",
    "7012.T",
    "7013.T",
    "9101.T",
    "9104.T",
    "9107.T",
    "9020.T",
    "9021.T",
    "9022.T",
    "8801.T",
    "8802.T",
    "2914.T",
    "3382.T",
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
# stock_scan.py と共通
#
# ★改善点②
#
# テスタ型モメンタム特徴量(ret5/ret20/ma25_slope5/
# volume_surge/breakout20/trend_alignment/momentum_score)と、
# 銘柄特性(volatility20/avg_volume_ratio)を追加して
# 本番と検証のFEATURESを完全一致させる。
# =========================================================
 
FEATURES = [
    # 基本
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
 
    # 相対強度
    "relative_strength",
 
    # テスタ型モメンタム
    "ret5",
    "ret20",
    "ma25_slope5",
    "volume_surge",
    "breakout20",
    "trend_alignment",
    "momentum_score",
 
    # ボリンジャーバンド
    "bb_position",
    "bb_width",
 
    # OBV
    "obv_change",
 
    # ATR
    "atr_ratio",
 
    # 銘柄特性(スケール非依存)
    "volatility20",
    "avg_volume_ratio",
 
    # 日経平均
    "nikkei_kairi25",
    "nikkei_rsi",
    "nikkei_macd",
    "nikkei_return_5d",
 
    # 日経225先物
    "future_return",
    "future_ma5",
    "future_rsi",
    "future_gap",
]
 
 
# =========================================================
# safe_download
#
# ★改善点⑥
#
# 指数バックオフ(2→4→8→16→32秒)に変更。
# =========================================================
 
def safe_download(
    ticker,
    retries=5,
    base_wait=2,
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
                and not df.empty
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
                f"{ticker}: 空データ "
                f"({attempt}/{retries})"
            )
 
        except Exception as e:
 
            print(
                f"{ticker}: 取得失敗 "
                f"({attempt}/{retries}): {e}"
            )
 
        if attempt < retries:
 
            wait_sec = (
                base_wait
                * (2 ** (attempt - 1))
            )
 
            print(
                f"{wait_sec}秒待機してリトライします..."
            )
 
            time.sleep(
                wait_sec
            )
 
    print(
        f"❌ {ticker}: 取得失敗"
    )
 
    return None
 
 
# =========================================================
# 複数銘柄の一括ダウンロード
#
# ★改善点⑥
#
# 銘柄ごとに個別アクセスするのではなく、
# まとめて1回のリクエストで取得を試み、
# 失敗した銘柄だけ個別にフォールバックする。
# =========================================================
 
def safe_download_batch(
    tickers,
    retries=5,
    base_wait=2,
    **kwargs
):
 
    for attempt in range(
        1,
        retries + 1
    ):
 
        try:
 
            df = yf.download(
                tickers,
                group_by="ticker",
                threads=True,
                **kwargs
            )
 
            if (
                df is not None
                and not df.empty
            ):
 
                return df
 
            print(
                f"一括取得: 空データ "
                f"({attempt}/{retries})"
            )
 
        except Exception as e:
 
            print(
                f"一括取得失敗 "
                f"({attempt}/{retries}): {e}"
            )
 
        if attempt < retries:
 
            wait_sec = (
                base_wait
                * (2 ** (attempt - 1))
            )
 
            print(
                f"{wait_sec}秒待機してリトライします..."
            )
 
            time.sleep(
                wait_sec
            )
 
    print(
        "❌ 一括取得 リトライ失敗"
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
# ATR
# =========================================================
 
def calc_atr(
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
 
    atr = (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )
 
    return atr
 
 
# =========================================================
# 個別銘柄特徴量
#
# ★改善点②
#
# stock_scan.py の create_features() と同じロジックに揃え、
# テスタ型モメンタム特徴量・銘柄特性特徴量を追加した。
# =========================================================
 
def create_features(df):
 
    df = df.copy()
 
    close = (
        df["Close"]
        .squeeze()
    )
 
    volume = (
        df["Volume"]
        .squeeze()
    )
 
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
 
    df["high252"] = high252
 
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
 
    df["_stock_ret5"] = (
        close
        .pct_change(5)
    )
 
    # -----------------------------------------------------
    # テスタ型モメンタム特徴量
    #
    # stock_scan.py と同一ロジック
    # -----------------------------------------------------
 
    df["ret5"] = (
        close.pct_change(5) * 100
    )
 
    df["ret20"] = (
        close.pct_change(20) * 100
    )
 
    df["ma25_slope5"] = (
        (
            df["ma25"]
            /
            df["ma25"].shift(5)
            - 1
        ) * 100
    )
 
    df["volume_surge"] = (
        volume
        /
        volume.rolling(5).mean()
    )
 
    rolling_high20 = (
        close.shift(1).rolling(20).max()
    )
 
    df["breakout20"] = (
        close / rolling_high20 - 1
    ) * 100
 
    df["trend_alignment"] = (
        (close > df["ma25"]).astype(int)
        +
        (df["ma25"] > df["ma75"]).astype(int)
        +
        (df["ma25_slope5"] > 0).astype(int)
    )
 
    momentum_score = pd.Series(
        0.0,
        index=df.index
    )
 
    momentum_score += np.where(
        close > df["ma25"], 20, 0
    )
 
    momentum_score += np.where(
        df["ma25"] > df["ma75"], 20, 0
    )
 
    momentum_score += np.where(
        df["ma25_slope5"] > 0, 15, 0
    )
 
    momentum_score += np.where(
        df["ret5"] > 0, 10, 0
    )
 
    momentum_score += np.where(
        df["ret20"] > 0, 10, 0
    )
 
    momentum_score += np.where(
        df["volume_surge"] >= 1.2, 10, 0
    )
 
    momentum_score += np.where(
        df["from_high"] >= -10, 10, 0
    )
 
    momentum_score += np.where(
        df["breakout20"] >= 0, 5, 0
    )
 
    df["momentum_score"] = momentum_score.clip(
        lower=0,
        upper=100
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
        volume
        *
        price_direction
    ).fillna(0).cumsum()
 
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
 
    atr = calc_atr(df)
 
    df["atr_ratio"] = (
        atr
        /
        close
        *
        100
    )
 
    # -----------------------------------------------------
    # 銘柄特性(スケール非依存)
    #
    # stock_scan.py と同一ロジック
    # -----------------------------------------------------
 
    df["avg_volume20"] = (
        volume.rolling(20).mean()
    )
 
    df["avg_volume60"] = (
        volume.rolling(60).mean()
    )
 
    df["volatility20"] = (
        df["ret1"].rolling(20).std() * 100
    )
 
    df["avg_volume_ratio"] = (
        df["avg_volume20"]
        /
        df["avg_volume60"].replace(0, np.nan)
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
            nikkei["nikkei_ma25"]
        )
        /
        nikkei["nikkei_ma25"]
        *
        100
    )
 
    nikkei["nikkei_uptrend"] = (
        nikkei["nikkei_ma25"]
        >
        nikkei["nikkei_ma75"]
    )
 
    nikkei["nikkei_rsi"] = (
        calc_rsi(close)
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
        calc_rsi(close)
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
 
    # 先物だけ1日ラグ
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
# ★改善点⑥
#
# 一括取得したデータをまず使い、
# 取得できなかった銘柄だけ個別にフォールバックする。
# =========================================================
 
def prepare_symbol_data(
    ticker,
    nikkei,
    futures,
    batch_price_data
):
 
    df = None
 
    if batch_price_data is not None:
 
        try:
 
            if isinstance(
                batch_price_data.columns,
                pd.MultiIndex
            ):
 
                top_level = (
                    batch_price_data.columns
                    .get_level_values(0)
                )
 
                if ticker in top_level:
 
                    sub_df = (
                        batch_price_data[ticker]
                        .dropna(how="all")
                        .copy()
                    )
 
                    if (
                        sub_df is not None
                        and len(sub_df) >= 150
                    ):
 
                        df = sub_df
 
        except Exception as e:
 
            print(
                f"{ticker} 一括データ抽出失敗: {e}"
            )
 
    if df is None:
 
        print(
            f"{ticker} 個別ダウンロードにフォールバック"
        )
 
        df = safe_download(
            ticker,
            start=history_start.strftime(
                "%Y-%m-%d"
            ),
            end=(
                end_ts
                +
                pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=True
        )
 
    if (
        df is None
        or
        len(df) < 150
    ):
 
        return None
 
    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
 
        df.columns = (
            df.columns
            .get_level_values(0)
        )
 
    df = create_features(
        df
    )
 
    # 日経
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
 
    # 先物
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
 
    # 相対強度
    df["relative_strength"] = (
        df["_stock_ret5"]
        -
        df["nikkei_ret5_raw"]
    )
 
    # 流動性
    df["liquid"] = (
        df["avg_volume20"]
        >=
        MIN_AVG_VOLUME
    )
 
    # -------------------------------------------------
    # target
    #
    # ★改善点④(確認)
    #
    # FORWARD_DAYS先の株価が未確定の行は
    # target_valid=False となり、fit_model() 側で
    # 学習データから除外される。「予測はできるが正解が
    # まだ分からない直近データ」を学習に混ぜない、という
    # stock_scan.py と同じ原則をここでも徹底する。
    # -------------------------------------------------
    future_price = (
        df["Close"]
        .shift(-FORWARD_DAYS)
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
 
    target_date = (
        pd.Series(
            df.index,
            index=df.index
        )
        .shift(-FORWARD_DAYS)
    )
 
    df["target_date"] = (
        target_date
    )
 
    # 最後に特徴量欠損除去
    df = df.dropna(
        subset=FEATURES
    ).copy()
 
    if len(df) < 100:
 
        return None
 
    return df
 
 
# =========================================================
# シグナル分類
#
# ★改善点⑤
#
# stock_scan.py の signal_category() と同じ分類ルール。
# 「実際に買う想定(buy)」「監視のみ(monitor)」
# 「対象外(no_buy)」を明示的に分ける。
# =========================================================
 
BUY_SIGNALS = {
    "🔥 強い買い",
    "🟢 買い",
}
 
 
def signal_category(signal):
 
    if signal in BUY_SIGNALS:
 
        return "buy"
 
    elif (
        isinstance(signal, str)
        and signal.startswith("🟡")
    ):
 
        return "monitor"
 
    else:
 
        return "no_buy"
 
 
# =========================================================
# シグナル作成
#
# ★改善点⑤
#
# stock_scan.py の calc_score() とAIスコアの重み付け
# (テクニカル/AI確率/テスタ型モメンタム)を揃えた。
# テスタ型モメンタムによる「買い→監視への格下げ」は
# ここでは行わず、testa_score を生の値として返し、
# 比較軸(TESTA_THRESHOLDS)側でフィルタリングする。
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
 
    high52 = float(
        df_row["high252"]
    )
 
    if (
        not np.isfinite(high52)
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
 
    technical_score = 0
 
    if rsi < 35:
        technical_score += 25
 
    if macd > signal:
        technical_score += 25
 
    if ma25 > ma75:
        technical_score += 20
 
    if vol_ratio > 1.5:
        technical_score += 20
 
    if np.isfinite(distance):
 
        if distance > -10:
 
            technical_score += 15
 
        elif distance > -20:
 
            technical_score += 8
 
    if float(
        df_row["nikkei_rsi"]
    ) > 50:
 
        technical_score += 5
 
    if float(
        df_row["nikkei_return_5d"]
    ) > 0:
 
        technical_score += 5
 
    technical_score_normalized = (
        technical_score
        /
        115.0
        *
        100
    )
 
    testa_score = float(
        df_row["momentum_score"]
    )
 
    # stock_scan.py と同じ配分
    BASE_TECH_WEIGHT = 0.525
    AI_WEIGHT = 0.225
    TESTA_WEIGHT = 0.25
 
    ai_score = (
        technical_score_normalized
        * BASE_TECH_WEIGHT
        +
        (
            up_prob
            *
            100
        )
        * AI_WEIGHT
        +
        testa_score
        * TESTA_WEIGHT
    )
 
    ai_score = max(
        0,
        min(
            100,
            ai_score
        )
    )
 
    up_percent = (
        up_prob * 100
    )
 
    down_percent = (
        down_prob * 100
    )
 
    flat_percent = (
        flat_prob * 100
    )
 
    # 表示用シグナル(参考ラベル。フィルタリングには使わない)
    if flat_percent >= 50:
 
        final_signal = (
            "🟡 監視(横ばい優勢)"
        )
 
    elif abs(
        up_percent
        -
        down_percent
    ) <= 1:
 
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
 
    else:
 
        final_signal = (
            "🟡 監視"
        )
 
    nikkei_uptrend = bool(
        df_row[
            "nikkei_uptrend"
        ]
    )
 
    if (
        not nikkei_uptrend
        and
        final_signal
        in (
            "🔥 強い買い",
            "🟢 買い",
        )
    ):
 
        final_signal = (
            "🟡 監視(日経下落/レンジ)"
        )
 
    # ATR TP / SL
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
 
        "category":
            signal_category(
                final_signal
            ),
 
        "testa_score":
            testa_score,
 
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
# 売買結果
#
# ★改善点①
#
# 「行数が5あるから5日」ではなく、
# 予測日から先の取引可能日が実際に5日分揃っているかを見る。
# TP/SLに未到達のままウィンドウが5日に満たない場合
# (バックテスト期間の終端に近い等)は、
# 中途半端な結果(HOLD/TIMEOUT_LOSS)として確定させず
# NO_DATA として集計から除外する。
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
 
        # 同日TP/SL両ヒット
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
 
        # TP
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
 
        # SL
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
 
    # -------------------------------------------------
    # ★5営業日分のウィンドウが揃っていない場合は判定保留
    # -------------------------------------------------
    if len(future) < 5:
 
        return (
            "NO_DATA",
            np.nan,
            len(future),
            np.nan
        )
 
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
# モデル
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
        usable["target"]
        .astype(int)
    )
 
    usable = usable[
        usable["target"]
        .isin(
            [0, 1, 2]
        )
    ]
 
    if len(usable) < MIN_TRAIN_ROWS:
 
        return None
 
    if (
        usable["target"]
        .nunique()
        <
        3
    ):
 
        return None
 
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
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
        *
        100
    )
 
 
# =========================================================
# PF
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
    label,
    print_result=True
):
 
    if result_df.empty:
 
        if print_result:
 
            print(
                f"\n[{label}] 取引なし"
            )
 
        return {}
 
    decided = (
        result_df[
            result_df["result"]
            .isin(
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
            decided["result"]
            ==
            "WIN"
        ).sum()
    )
 
    losses = int(
        (
            decided["result"]
            .isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            )
        ).sum()
    )
 
    holds = int(
        (
            result_df["result"]
            ==
            "HOLD"
        ).sum()
    )
 
    signals = len(
        result_df
    )
 
    avg_return = float(
        result_df["return"]
        .mean()
    )
 
    decided_count = (
        wins
        +
        losses
    )
 
    if decided_count > 0:
 
        win_rate = (
            wins
            /
            decided_count
            *
            100
        )
 
    else:
 
        win_rate = 0.0
 
    pf = profit_factor(
        result_df["return"]
    )
 
    mdd = max_drawdown(
        result_df["return"]
    )
 
    avg_hold_days = float(
        result_df["hold_days"]
        .mean()
    )
 
    if print_result:
 
        if np.isinf(pf):
 
            pf_text_local = "inf"
 
        else:
 
            pf_text_local = (
                f"{pf:.2f}"
            )
 
        print("")
        print(
            "=============================="
        )
        print(
            f"📊 {label}"
        )
        print(
            "=============================="
        )
        print(
            f"シグナル数: {signals}"
        )
        print(
            f"WIN: {wins}"
        )
        print(
            f"LOSS/TIMEOUT_LOSS: {losses}"
        )
        print(
            f"HOLD: {holds}"
        )
        print(
            f"勝率: {win_rate:.2f}%"
        )
        print(
            f"平均リターン: {avg_return:.2f}%"
        )
        print(
            f"PF: {pf_text_local}"
        )
        print(
            f"最大DD: {mdd:.2f}%"
        )
        print(
            f"平均保有日数: {avg_hold_days:.2f}"
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
# Discord用
# =========================================================
 
def pf_text(
    value
):
 
    if value is None:
 
        return "N/A"
 
    if np.isinf(value):
 
        return "inf"
 
    return f"{value:.2f}"
 
 
# =========================================================
# START
# =========================================================
 
print("")
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
    f"期間: {START_DATE} ～ {END_DATE}"
)
 
print(
    f"TOP_N: {TOP_N}"
)
 
print(
    f"再学習間隔: "
    f"{REFIT_EVERY_TRADING_DAYS}営業日"
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
    "比較:"
)
 
print(
    "UP50 / 55 / 60 / 65%"
)
 
print(
    "日経フィルター ON / OFF"
)
 
print(
    "テスタ型モメンタム BASELINE / 55 / 65 / 75"
)
 
 
# =========================================================
# 日付
# =========================================================
 
start_ts = pd.Timestamp(
    START_DATE
)
 
end_ts = pd.Timestamp(
    END_DATE
)
 
history_start = (
    start_ts
    -
    pd.DateOffset(
        years=HISTORY_YEARS
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
    start=history_start.strftime(
        "%Y-%m-%d"
    ),
    end=(
        end_ts
        +
        pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
    ),
    interval="1d",
    auto_adjust=True
)
 
if nikkei is None:
 
    raise RuntimeError(
        "日経平均データ取得失敗"
    )
 
 
# =========================================================
# 先物
# =========================================================
 
futures = safe_download(
    "NIY=F",
    start=history_start.strftime(
        "%Y-%m-%d"
    ),
    end=(
        end_ts
        +
        pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
    ),
    interval="1d",
    auto_adjust=True
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
# 全銘柄まとめて一括取得
#
# ★改善点⑥
# =========================================================
 
batch_price_data = safe_download_batch(
    TICKERS,
    start=history_start.strftime(
        "%Y-%m-%d"
    ),
    end=(
        end_ts
        +
        pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
    ),
    interval="1d",
    auto_adjust=True
)
 
 
# =========================================================
# 各銘柄準備
# =========================================================
 
symbol_data = {}
 
 
for ticker in TICKERS:
 
    print(
        f"データ準備: {ticker}"
    )
 
    df = prepare_symbol_data(
        ticker,
        nikkei,
        futures,
        batch_price_data
    )
 
    if df is not None:
 
        symbol_data[ticker] = df
 
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
            set(df.index)
            for df in symbol_data.values()
        ]
    )
)
 
prediction_dates_all = [
    d
    for d in all_dates
    if (
        start_ts
        <=
        pd.Timestamp(d)
        <=
        end_ts
    )
]
 
 
# =========================================================
# ウォームアップ
# =========================================================
 
if (
    len(prediction_dates_all)
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
        "予測期間以上です"
    )
 
 
print(
    f"予測日数: "
    f"{len(prediction_dates)}"
)
 
 
# =========================================================
# DEV / VALIDATION / OOS
# =========================================================
 
if (
    len(prediction_dates)
    >
    OOS_DAYS
):
 
    non_oos_dates = (
        prediction_dates[
            :-OOS_DAYS
        ]
    )
 
    oos_dates = (
        prediction_dates[
            -OOS_DAYS:
        ]
    )
 
else:
 
    non_oos_dates = []
 
    oos_dates = list(
        prediction_dates
    )
 
 
split_idx = int(
    len(non_oos_dates)
    *
    DEV_SPLIT_RATIO
)
 
dev_dates = (
    non_oos_dates[
        :split_idx
    ]
)
 
validation_dates = (
    non_oos_dates[
        split_idx:
    ]
)
 
phase_by_date = {}
 
 
for d in dev_dates:
 
    phase_by_date[
        pd.Timestamp(d)
    ] = "DEV"
 
 
for d in validation_dates:
 
    phase_by_date[
        pd.Timestamp(d)
    ] = "VALIDATION"
 
 
for d in oos_dates:
 
    phase_by_date[
        pd.Timestamp(d)
    ] = "OOS"
 
 
print(
    f"DEV: {len(dev_dates)}営業日"
)
 
print(
    f"VALIDATION: "
    f"{len(validation_dates)}営業日"
)
 
print(
    f"OOS: "
    f"{len(oos_dates)}営業日"
)
 
 
# =========================================================
# WALK FORWARD
#
# ここでは「候補を全部保存」。
# 条件の比較は後段で行う。
# =========================================================
 
candidate_history = []
 
model = None
 
last_fit_pos = -10**9
 
 
for pos, prediction_date in enumerate(
    prediction_dates
):
 
    prediction_date = pd.Timestamp(
        prediction_date
    )
 
    if pos % 20 == 0:
 
        print(
            f"進捗: "
            f"{pos + 1}/"
            f"{len(prediction_dates)} "
            f"{prediction_date.date()}"
        )
 
 
    # =====================================================
    # 再学習
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
 
 
    if need_refit:
 
        train_frames = []
 
 
        for ticker, df in (
            symbol_data.items()
        ):
 
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
                    ["target"]
                ]
                .assign(
                    date=
                    usable_prior.index,
 
                    ticker=
                    ticker,
 
                    target_valid=
                    True
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
                ["target"]
            )
        )
 
 
        if (
            len(train_all)
            <
            MIN_TRAIN_ROWS
        ):
 
            model = None
 
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
            set(
                class_values
            )
        ):
 
            model = None
 
            continue
 
 
        model = fit_model(
            train_all
            .assign(
                target_valid=True
            )
        )
 
 
        if model is None:
 
            continue
 
 
        last_fit_pos = pos
 
 
        print(
            f"  ✅ model refit "
            f"{prediction_date.date()} "
            f"train_rows="
            f"{len(train_all)}"
        )
 
 
    if model is None:
 
        continue
 
 
    # =====================================================
    # 当日の全候補
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
 
 
        row = df.loc[
            prediction_date
        ]
 
 
        # 流動性
        if not bool(
            row["liquid"]
        ):
 
            continue
 
 
        x = (
            row[FEATURES]
            .to_frame()
            .T
        )
 
 
        if (
            x.isna()
            .any()
            .any()
        ):
 
            continue
 
 
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
                for cls in [
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
 
 
        signal = calculate_signal(
            row,
            up_prob,
            down_prob,
            flat_prob
        )
 
 
        signal.update(
            {
                "date":
                    prediction_date,
 
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
    # AIスコア順
    # =====================================================
 
    candidates.sort(
        key=lambda x:
        x["score"],
        reverse=True
    )
 
 
    # =====================================================
    # 全候補保存
    # =====================================================
 
    for candidate in candidates:
 
        candidate_history.append(
            {
                "date":
                    candidate[
                        "date"
                    ],
 
                "ticker":
                    candidate[
                        "ticker"
                    ],
 
                "company":
                    candidate[
                        "company"
                    ],
 
                "score":
                    float(
                        candidate[
                            "score"
                        ]
                    ),
 
                "testa_score":
                    float(
                        candidate[
                            "testa_score"
                        ]
                    ),
 
                "up_prob":
                    float(
                        candidate[
                            "up_prob"
                        ]
                    ),
 
                "flat_prob":
                    float(
                        candidate[
                            "flat_prob"
                        ]
                    ),
 
                "down_prob":
                    float(
                        candidate[
                            "down_prob"
                        ]
                    ),
 
                "price":
                    float(
                        candidate[
                            "price"
                        ]
                    ),
 
                "take_profit":
                    float(
                        candidate[
                            "take_profit"
                        ]
                    ),
 
                "stop_loss":
                    float(
                        candidate[
                            "stop_loss"
                        ]
                    ),
 
                "nikkei_uptrend":
                    bool(
                        candidate[
                            "nikkei_uptrend"
                        ]
                    ),
 
                "rsi":
                    float(
                        candidate[
                            "rsi"
                        ]
                    ),
 
                "vol":
                    float(
                        candidate[
                            "vol"
                        ]
                    ),
            }
        )
 
 
# =========================================================
# 候補が無い場合
# =========================================================
 
if not candidate_history:
 
    raise RuntimeError(
        "ウォークフォワード候補が0件です"
    )
 
 
candidate_df = pd.DataFrame(
    candidate_history
)
 
 
candidate_df["date"] = (
    pd.to_datetime(
        candidate_df["date"]
    )
)
 
 
print("")
print(
    "✅ ウォークフォワード候補生成完了"
)
 
print(
    f"候補件数: "
    f"{len(candidate_df)}"
)
 
 
# =========================================================
# 戦略評価
#
# ★改善点⑤
#
# testa_threshold を追加。None ならモメンタムフィルターなし
# (BASELINE)、数値ならその値以上の momentum_score を持つ
# 候補だけを「買い」とみなす。
# =========================================================
 
def evaluate_strategy_variant(
    candidate_df,
    threshold,
    nikkei_filter,
    testa_threshold
):
 
    rows = []
 
 
    for prediction_date, day_candidates in (
        candidate_df
        .groupby(
            "date"
        )
    ):
 
        day_candidates = (
            day_candidates
            .copy()
        )
 
 
        # -------------------------------------------------
        # UP確率
        # -------------------------------------------------
 
        day_candidates = (
            day_candidates[
                day_candidates[
                    "up_prob"
                ]
                >=
                threshold
            ]
            .copy()
        )
 
 
        # -------------------------------------------------
        # UP > DOWN
        # -------------------------------------------------
 
        day_candidates = (
            day_candidates[
                day_candidates[
                    "up_prob"
                ]
                >
                day_candidates[
                    "down_prob"
                ]
            ]
            .copy()
        )
 
 
        # -------------------------------------------------
        # 横ばい優勢除外
        # -------------------------------------------------
 
        day_candidates = (
            day_candidates[
                day_candidates[
                    "flat_prob"
                ]
                <
                50
            ]
            .copy()
        )
 
 
        # -------------------------------------------------
        # 日経フィルター
        # -------------------------------------------------
 
        if nikkei_filter:
 
            day_candidates = (
                day_candidates[
                    day_candidates[
                        "nikkei_uptrend"
                    ]
                    ==
                    True
                ]
                .copy()
            )
 
 
        # -------------------------------------------------
        # ★テスタ型モメンタムフィルター
        # -------------------------------------------------
 
        if testa_threshold is not None:
 
            day_candidates = (
                day_candidates[
                    day_candidates[
                        "testa_score"
                    ]
                    >=
                    testa_threshold
                ]
                .copy()
            )
 
 
        if day_candidates.empty:
 
            continue
 
 
        # -------------------------------------------------
        # AIスコア順TOP_N
        # -------------------------------------------------
 
        selected = (
            day_candidates
            .sort_values(
                "score",
                ascending=False
            )
            .head(TOP_N)
        )
 
 
        # -------------------------------------------------
        # 売買結果
        # -------------------------------------------------
 
        for rank, (_, candidate) in enumerate(
            selected.iterrows(),
            start=1
        ):
 
            ticker = (
                candidate[
                    "ticker"
                ]
            )
 
 
            if ticker not in symbol_data:
 
                continue
 
 
            df = (
                symbol_data[
                    ticker
                ]
            )
 
 
            (
                result,
                ret,
                hold_days,
                exit_price
            ) = evaluate_trade(
                df,
                prediction_date,
                candidate["price"],
                candidate[
                    "take_profit"
                ],
                candidate[
                    "stop_loss"
                ]
            )
 
 
            if result == "NO_DATA":
 
                continue
 
 
            testa_label = (
                "BASE"
                if testa_threshold is None
                else f"TESTA{testa_threshold}"
            )
 
 
            rows.append(
                {
                    "date":
                        prediction_date,
 
                    "ticker":
                        ticker,
 
                    "company":
                        candidate[
                            "company"
                        ],
 
                    "rank":
                        rank,
 
                    "score":
                        candidate[
                            "score"
                        ],
 
                    "testa_score":
                        candidate[
                            "testa_score"
                        ],
 
                    "up_prob":
                        candidate[
                            "up_prob"
                        ],
 
                    "flat_prob":
                        candidate[
                            "flat_prob"
                        ],
 
                    "down_prob":
                        candidate[
                            "down_prob"
                        ],
 
                    "nikkei_uptrend":
                        candidate[
                            "nikkei_uptrend"
                        ],
 
                    "price":
                        candidate[
                            "price"
                        ],
 
                    "take_profit":
                        candidate[
                            "take_profit"
                        ],
 
                    "stop_loss":
                        candidate[
                            "stop_loss"
                        ],
 
                    "result":
                        result,
 
                    "return":
                        ret,
 
                    "hold_days":
                        hold_days,
 
                    "exit_price":
                        exit_price,
 
                    "phase":
                        phase_by_date.get(
                            prediction_date,
                            "UNKNOWN"
                        ),
 
                    "threshold":
                        threshold,
 
                    "nikkei_filter":
                        nikkei_filter,
 
                    "testa_threshold":
                        testa_threshold,
 
                    "strategy":
                        (
                            f"UP{threshold}_"
                            f"NIKKEI"
                            f"{'ON' if nikkei_filter else 'OFF'}_"
                            f"{testa_label}"
                        ),
                }
            )
 
 
    if not rows:
 
        return pd.DataFrame()
 
    return pd.DataFrame(
        rows
    )
 
 
# =========================================================
# 全条件比較
#
# UP_THRESHOLDS × NIKKEI_FILTER_OPTIONS × TESTA_THRESHOLDS
# =========================================================
 
strategy_detail_frames = []
 
strategy_summary_rows = []
 
 
for threshold in UP_THRESHOLDS:
 
    for nikkei_filter in (
        NIKKEI_FILTER_OPTIONS
    ):
 
        for testa_threshold in (
            TESTA_THRESHOLDS
        ):
 
            testa_label = (
                "BASE"
                if testa_threshold is None
                else f"TESTA{testa_threshold}"
            )
 
            strategy_name = (
                f"UP{threshold}_"
                f"NIKKEI"
                f"{'ON' if nikkei_filter else 'OFF'}_"
                f"{testa_label}"
            )
 
 
            print("")
            print(
                "=" * 60
            )
 
            print(
                f"検証中: "
                f"{strategy_name}"
            )
 
            print(
                "=" * 60
            )
 
 
            variant_df = (
                evaluate_strategy_variant(
                    candidate_df,
                    threshold,
                    nikkei_filter,
                    testa_threshold
                )
            )
 
 
            if not variant_df.empty:
 
                strategy_detail_frames.append(
                    variant_df
                )
 
 
            # ---------------------------------------------
            # 全体
            # ---------------------------------------------
 
            stats = summarize(
                variant_df,
                strategy_name,
                print_result=True
            )
 
 
            if stats:
 
                strategy_summary_rows.append(
                    {
                        "strategy":
                            strategy_name,
 
                        "threshold":
                            threshold,
 
                        "nikkei_filter":
                            nikkei_filter,
 
                        "testa_threshold":
                            testa_threshold
                            if testa_threshold
                            is not None
                            else "BASE",
 
                        "signals":
                            stats[
                                "signals"
                            ],
 
                        "wins":
                            stats[
                                "wins"
                            ],
 
                        "losses":
                            stats[
                                "losses"
                            ],
 
                        "holds":
                            stats[
                                "holds"
                            ],
 
                        "win_rate":
                            stats[
                                "win_rate"
                            ],
 
                        "avg_return":
                            stats[
                                "avg_return"
                            ],
 
                        "profit_factor":
                            stats[
                                "profit_factor"
                            ],
 
                        "max_drawdown":
                            stats[
                                "max_drawdown"
                            ],
 
                        "avg_hold_days":
                            stats[
                                "avg_hold_days"
                            ],
                    }
                )
 
            else:
 
                strategy_summary_rows.append(
                    {
                        "strategy":
                            strategy_name,
 
                        "threshold":
                            threshold,
 
                        "nikkei_filter":
                            nikkei_filter,
 
                        "testa_threshold":
                            testa_threshold
                            if testa_threshold
                            is not None
                            else "BASE",
 
                        "signals":
                            0,
 
                        "wins":
                            0,
 
                        "losses":
                            0,
 
                        "holds":
                            0,
 
                        "win_rate":
                            0.0,
 
                        "avg_return":
                            0.0,
 
                        "profit_factor":
                            0.0,
 
                        "max_drawdown":
                            0.0,
 
                        "avg_hold_days":
                            0.0,
                    }
                )
 
 
# =========================================================
# 結果DataFrame
# =========================================================
 
strategy_summary_df = pd.DataFrame(
    strategy_summary_rows
)
 
 
strategy_detail_df = (
    pd.concat(
        strategy_detail_frames,
        ignore_index=True
    )
    if strategy_detail_frames
    else
    pd.DataFrame()
)
 
 
# =========================================================
# 保存
# =========================================================
 
strategy_summary_df.to_csv(
    "walk_forward_strategy_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
 
 
candidate_df.to_csv(
    "walk_forward_all_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)
 
 
if not strategy_detail_df.empty:
 
    strategy_detail_df.to_csv(
        "walk_forward_strategy_details.csv",
        index=False,
        encoding="utf-8-sig"
    )
 
 
# =========================================================
# フェーズ別比較
#
# ★過剰最適化への配慮
#
# 取引件数(signals)が極端に少ない組み合わせ
# (例: TESTA75で取引数が数十件しかない等)は、
# VALIDATIONランキングの見た目上の勝率・PFが良くても
# 単独では採用判断しない。件数は必ず併記する。
# =========================================================
 
phase_rows = []
 
 
if not strategy_detail_df.empty:
 
    for (
        strategy_name,
        strategy_group
    ) in (
        strategy_detail_df
        .groupby(
            "strategy"
        )
    ):
 
        for (
            phase_name,
            phase_group
        ) in (
            strategy_group
            .groupby(
                "phase"
            )
        ):
 
            stats = summarize(
                phase_group,
                (
                    f"{strategy_name} "
                    f"{phase_name}"
                ),
                print_result=False
            )
 
 
            if not stats:
 
                continue
 
 
            phase_rows.append(
                {
                    "strategy":
                        strategy_name,
 
                    "phase":
                        phase_name,
 
                    "signals":
                        stats[
                            "signals"
                        ],
 
                    "wins":
                        stats[
                            "wins"
                        ],
 
                    "losses":
                        stats[
                            "losses"
                        ],
 
                    "holds":
                        stats[
                            "holds"
                        ],
 
                    "win_rate":
                        stats[
                            "win_rate"
                        ],
 
                    "avg_return":
                        stats[
                            "avg_return"
                        ],
 
                    "profit_factor":
                        stats[
                            "profit_factor"
                        ],
 
                    "max_drawdown":
                        stats[
                            "max_drawdown"
                        ],
 
                    "avg_hold_days":
                        stats[
                            "avg_hold_days"
                        ],
                }
            )
 
 
phase_strategy_df = pd.DataFrame(
    phase_rows
)
 
 
phase_strategy_df.to_csv(
    "walk_forward_strategy_phase_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
 
 
# =========================================================
# DEV / VALIDATIONランキング
# =========================================================
 
print("")
print(
    "=" * 80
)
 
print(
    "🧪 VALIDATIONランキング"
)
 
print(
    "=" * 80
)
 
 
validation_df = pd.DataFrame()
 
 
if not phase_strategy_df.empty:
 
    validation_df = (
        phase_strategy_df[
            phase_strategy_df["phase"]
            ==
            "VALIDATION"
        ]
        .copy()
    )
 
 
    if not validation_df.empty:
 
        validation_df = (
            validation_df
            .sort_values(
                [
                    "win_rate",
                    "profit_factor"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )
 
 
        for _, row in (
            validation_df
            .iterrows()
        ):
 
            print(
                f"{row['strategy']:26s} "
                f"件数={int(row['signals']):4d} "
                f"勝率={row['win_rate']:6.2f}% "
                f"平均={row['avg_return']:+6.2f}% "
                f"PF={pf_text(row['profit_factor'])} "
                f"DD={row['max_drawdown']:+6.2f}%"
            )
 
 
# =========================================================
# DEVランキング
# =========================================================
 
print("")
print(
    "=" * 80
)
 
print(
    "🧪 DEVランキング"
)
 
print(
    "=" * 80
)
 
 
dev_df = pd.DataFrame()
 
 
if not phase_strategy_df.empty:
 
    dev_df = (
        phase_strategy_df[
            phase_strategy_df["phase"]
            ==
            "DEV"
        ]
        .copy()
    )
 
 
    if not dev_df.empty:
 
        dev_df = (
            dev_df
            .sort_values(
                [
                    "win_rate",
                    "profit_factor"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )
 
 
        for _, row in (
            dev_df
            .iterrows()
        ):
 
            print(
                f"{row['strategy']:26s} "
                f"件数={int(row['signals']):4d} "
                f"勝率={row['win_rate']:6.2f}% "
                f"平均={row['avg_return']:+6.2f}% "
                f"PF={pf_text(row['profit_factor'])} "
                f"DD={row['max_drawdown']:+6.2f}%"
            )
 
 
# =========================================================
# OOSランキング
#
# 参考確認のみ
# 条件選定には使わない
# =========================================================
 
print("")
print(
    "=" * 80
)
 
print(
    "🧪 OOSランキング"
)
 
print(
    "=" * 80
)
 
 
oos_df = pd.DataFrame()
 
 
if not phase_strategy_df.empty:
 
    oos_df = (
        phase_strategy_df[
            phase_strategy_df["phase"]
            ==
            "OOS"
        ]
        .copy()
    )
 
 
    if not oos_df.empty:
 
        oos_df = (
            oos_df
            .sort_values(
                [
                    "win_rate",
                    "profit_factor"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )
 
 
        for _, row in (
            oos_df
            .iterrows()
        ):
 
            print(
                f"{row['strategy']:26s} "
                f"件数={int(row['signals']):4d} "
                f"勝率={row['win_rate']:6.2f}% "
                f"平均={row['avg_return']:+6.2f}% "
                f"PF={pf_text(row['profit_factor'])} "
                f"DD={row['max_drawdown']:+6.2f}%"
            )
 
 
# =========================================================
# 月別
# =========================================================
 
if not strategy_detail_df.empty:
 
    strategy_detail_df["month"] = (
        strategy_detail_df["date"]
        .dt
        .to_period("M")
        .astype(str)
    )
 
 
    monthly = (
        strategy_detail_df
        .groupby(
            [
                "strategy",
                "month"
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
                    s == "WIN"
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
                    s == "HOLD"
                ).sum()
            ),
        )
        .reset_index()
    )
 
 
    monthly["win_rate"] = np.where(
        (
            monthly["wins"]
            +
            monthly["losses"]
        )
        >
        0,
 
        monthly["wins"]
        /
        (
            monthly["wins"]
            +
            monthly["losses"]
        )
        *
        100,
 
        0
    )
 
 
    monthly.to_csv(
        "walk_forward_monthly.csv",
        index=False,
        encoding="utf-8-sig"
    )
 
 
# =========================================================
# 年別
# =========================================================
 
if not strategy_detail_df.empty:
 
    strategy_detail_df["year"] = (
        strategy_detail_df["date"]
        .dt
        .year
    )
 
 
    yearly = (
        strategy_detail_df
        .groupby(
            [
                "strategy",
                "year"
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
                    s == "WIN"
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
                    s == "HOLD"
                ).sum()
            ),
        )
        .reset_index()
    )
 
 
    yearly["win_rate"] = np.where(
        (
            yearly["wins"]
            +
            yearly["losses"]
        )
        >
        0,
 
        yearly["wins"]
        /
        (
            yearly["wins"]
            +
            yearly["losses"]
        )
        *
        100,
 
        0
    )
 
 
    yearly.to_csv(
        "walk_forward_yearly.csv",
        index=False,
        encoding="utf-8-sig"
    )
 
 
# =========================================================
# Discord
# =========================================================
 
discord_lines = []
 
 
discord_lines.append(
    "🧪 AI WALK FORWARD STRATEGY TEST"
)
 
discord_lines.append(
    "━━━━━━━━━━━━━━━━━━"
)
 
discord_lines.append(
    f"期間：{START_DATE} ～ {END_DATE}"
)
 
discord_lines.append(
    "比較：UP50/55/60/65% × 日経ON/OFF × テスタ型BASE/55/65/75"
)
 
discord_lines.append(
    f"再学習：{REFIT_EVERY_TRADING_DAYS}営業日"
)
 
discord_lines.append(
    f"ATR TP：{ATR_TP_MULTIPLIER}x"
)
 
discord_lines.append(
    f"ATR SL：{ATR_SL_MULTIPLIER}x"
)
 
discord_lines.append("")
 
 
# =========================================================
# Validation
# =========================================================
 
discord_lines.append(
    "【VALIDATION】"
)
 
 
if (
    validation_df
    is not None
    and
    not validation_df.empty
):
 
    for _, row in (
        validation_df
        .head(8)
        .iterrows()
    ):
 
        discord_lines.append(
            f"{row['strategy']} "
            f"件数={int(row['signals'])} "
            f"勝率={row['win_rate']:.2f}% "
            f"平均={row['avg_return']:+.2f}% "
            f"PF={pf_text(row['profit_factor'])}"
        )
 
else:
 
    discord_lines.append(
        "データなし"
    )
 
 
discord_lines.append("")
 
 
# =========================================================
# DEV
# =========================================================
 
discord_lines.append(
    "【DEV】"
)
 
 
if (
    dev_df
    is not None
    and
    not dev_df.empty
):
 
    for _, row in (
        dev_df
        .head(8)
        .iterrows()
    ):
 
        discord_lines.append(
            f"{row['strategy']} "
            f"件数={int(row['signals'])} "
            f"勝率={row['win_rate']:.2f}% "
            f"平均={row['avg_return']:+.2f}% "
            f"PF={pf_text(row['profit_factor'])}"
        )
 
else:
 
    discord_lines.append(
        "データなし"
    )
 
 
discord_lines.append("")
 
 
# =========================================================
# OOS
# =========================================================
 
discord_lines.append(
    "【OOS・参考確認】"
)
 
 
if (
    oos_df
    is not None
    and
    not oos_df.empty
):
 
    for _, row in (
        oos_df
        .head(8)
        .iterrows()
    ):
 
        discord_lines.append(
            f"{row['strategy']} "
            f"件数={int(row['signals'])} "
            f"勝率={row['win_rate']:.2f}% "
            f"平均={row['avg_return']:+.2f}% "
            f"PF={pf_text(row['profit_factor'])}"
        )
 
else:
 
    discord_lines.append(
        "データなし"
    )
 
 
discord_lines.append("")
 
 
# =========================================================
# 全体の条件ランキング
# =========================================================
 
discord_lines.append(
    "【全期間比較】"
)
 
 
if not strategy_summary_df.empty:
 
    ranking_all = (
        strategy_summary_df
        .sort_values(
            [
                "win_rate",
                "profit_factor"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(8)
    )
 
 
    for _, row in (
        ranking_all
        .iterrows()
    ):
 
        discord_lines.append(
            f"{row['strategy']} "
            f"件数={int(row['signals'])} "
            f"勝率={row['win_rate']:.2f}% "
            f"平均={row['avg_return']:+.2f}% "
            f"PF={pf_text(row['profit_factor'])}"
        )
 
else:
 
    discord_lines.append(
        "データなし"
    )
 
 
discord_lines.append("")
 
discord_lines.append(
    "📁 walk_forward_strategy_comparison.csv"
)
 
discord_lines.append(
    "📁 walk_forward_strategy_phase_comparison.csv"
)
 
discord_message = (
    "\n".join(
        discord_lines
    )
)
 
 
print("")
print(
    "==========================================="
)
 
print(
    "📨 戦略比較結果をDiscordへ送信"
)
 
print(
    "==========================================="
)
 
print(
    discord_message
)
 
 
send_discord(
    discord_message
)
 
 
# =========================================================
# 完了
# =========================================================
 
print("")
print(
    "==========================================="
)
 
print(
    "✅ WALK-FORWARD STRATEGY TEST 完了"
)
 
print(
    "==========================================="
)
 
print(
    "保存:"
)
 
print(
    "  walk_forward_all_candidates.csv"
)
 
print(
    "  walk_forward_strategy_comparison.csv"
)
 
print(
    "  walk_forward_strategy_details.csv"
)
 
print(
    "  walk_forward_strategy_phase_comparison.csv"
)
 
print(
    "  walk_forward_monthly.csv"
)
 
print(
    "  walk_forward_yearly.csv"
)
