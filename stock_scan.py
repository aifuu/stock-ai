import os
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAPER TRADE ONLY
#
# ★重要
# このファイルは1年間の検証用。
# 実際の証券会社への注文は一切行わない。
#
# AIシグナル
# ↓
# risk_manager.py
# ↓
# 仮想ポジション登録
# ↓
# 仮想決済
#
# =========================================================

from risk_manager import (
    risk_check,
    build_position_plan,
    register_position_open,
    register_position_close,
    risk_status_text,
    load_risk_state,
)


# =========================================================
# 基本設定
# =========================================================

POLICY_FILE = "strategy_policy.json"

TRAIN_FILE = "train_data.csv"
MODEL_FILE = "model.pkl"
HISTORY_FILE = "prediction_history.csv"


# =========================================================
# 自動戦略ポリシー
# =========================================================

DEFAULT_POLICY_UP_THRESHOLD = 50
DEFAULT_MIN_SCORE_FOR_BUY = 60
DEFAULT_NIKKEI_FILTER = False

DEFAULT_ATR_TP_MULTIPLIER = 3.0
DEFAULT_ATR_SL_MULTIPLIER = 1.5
DEFAULT_HOLD_DAYS = 5


def parse_bool(
    value,
    default=False
):

    if isinstance(
        value,
        bool
    ):

        return value

    if isinstance(
        value,
        (int, float)
    ):

        return bool(value)

    if isinstance(
        value,
        str
    ):

        text = value.strip().lower()

        if text in (
            "true",
            "1",
            "yes",
            "on"
        ):

            return True

        if text in (
            "false",
            "0",
            "no",
            "off"
        ):

            return False

    return default


def load_strategy_policy():

    default_policy = {

        "status":
            "DEFAULT",

        "up_threshold":
            DEFAULT_POLICY_UP_THRESHOLD,

        "min_score_for_buy":
            DEFAULT_MIN_SCORE_FOR_BUY,

        "nikkei_filter":
            DEFAULT_NIKKEI_FILTER,

        "atr_tp_multiplier":
            DEFAULT_ATR_TP_MULTIPLIER,

        "atr_sl_multiplier":
            DEFAULT_ATR_SL_MULTIPLIER,

        "hold_days":
            DEFAULT_HOLD_DAYS,
    }


    if not os.path.exists(
        POLICY_FILE
    ):

        print(
            "⚠ strategy_policy.jsonなし"
        )

        print(
            "→ デフォルト設定を使用"
        )

        return default_policy


    try:

        with open(
            POLICY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            policy = json.load(f)


        if not isinstance(
            policy,
            dict
        ):

            raise ValueError(
                "policyがdictではありません"
            )


        merged = default_policy.copy()

        merged.update(
            policy
        )


        if merged["status"] not in (
            "APPROVED",
            "DEFAULT"
        ):

            print(
                "⚠ policy status="
                f"{merged['status']}"
            )

            print(
                "→ デフォルト設定を使用"
            )

            return default_policy


        merged["up_threshold"] = int(
            merged["up_threshold"]
        )

        merged["min_score_for_buy"] = int(
            merged["min_score_for_buy"]
        )

        merged["nikkei_filter"] = parse_bool(
            merged["nikkei_filter"],
            DEFAULT_NIKKEI_FILTER
        )

        merged["atr_tp_multiplier"] = float(
            merged["atr_tp_multiplier"]
        )

        merged["atr_sl_multiplier"] = float(
            merged["atr_sl_multiplier"]
        )

        merged["hold_days"] = int(
            merged["hold_days"]
        )


        print("")
        print(
            "🤖 自動戦略ポリシー読み込み"
        )

        print(
            "status:",
            merged["status"]
        )

        print(
            "UP:",
            merged["up_threshold"]
        )

        print(
            "MIN SCORE:",
            merged["min_score_for_buy"]
        )

        print(
            "日経フィルター:",
            merged["nikkei_filter"]
        )

        print(
            "ATR TP:",
            merged["atr_tp_multiplier"]
        )

        print(
            "ATR SL:",
            merged["atr_sl_multiplier"]
        )

        print(
            "HOLD:",
            merged["hold_days"]
        )

        return merged


    except Exception as e:

        print(
            f"⚠ strategy_policy.json"
            f"読み込み失敗: {e}"
        )

        print(
            "→ デフォルト設定を使用"
        )

        return default_policy


STRATEGY_POLICY = (
    load_strategy_policy()
)


POLICY_UP_THRESHOLD = (
    STRATEGY_POLICY[
        "up_threshold"
    ]
)

MIN_SCORE_FOR_BUY = (
    STRATEGY_POLICY[
        "min_score_for_buy"
    ]
)

NIKKEI_FILTER_ENABLED = (
    STRATEGY_POLICY[
        "nikkei_filter"
    ]
)

ATR_TP_MULTIPLIER = (
    STRATEGY_POLICY[
        "atr_tp_multiplier"
    ]
)

ATR_SL_MULTIPLIER = (
    STRATEGY_POLICY[
        "atr_sl_multiplier"
    ]
)

HOLD_DAYS = (
    STRATEGY_POLICY[
        "hold_days"
    ]
)


# =========================================================
# PAPER TRADE SETTINGS
# =========================================================

PAPER_TRADING_ONLY = True

PAPER_INITIAL_CAPITAL = float(
    os.getenv(
        "AI_INITIAL_CAPITAL",
        "1000000"
    )
)

PAPER_MAX_POSITIONS = int(
    os.getenv(
        "AI_MAX_POSITIONS",
        "3"
    )
)

PAPER_LOG_FILE = (
    "paper_trade_log.csv"
)


# =========================================================
# Discord
# =========================================================

WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK"
)


def send(
    message
):

    if not WEBHOOK_URL:

        print(
            "⚠ Webhookなし"
        )

        return


    if len(message) > 1900:

        message = message[:1900]


    try:

        response = requests.post(
            WEBHOOK_URL,
            json={
                "content":
                    message
            },
            timeout=30
        )


        print(
            "Discord status =",
            response.status_code
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
            f"❌ Discord送信エラー: {e}"
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
    "6762.T",
    "7735.T",
    "6981.T",
    "4543.T",
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
    "6758.T": "ソニーグループ",
    "6861.T": "キーエンス",
    "6857.T": "アドバンテスト",
    "8035.T": "東京エレクトロン",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6501.T": "日立製作所",
    "6503.T": "三菱電機",
    "6701.T": "日本電気",
    "6702.T": "富士通",
    "6902.T": "デンソー",
    "6901.T": "澤藤電機",
    "7270.T": "SUBARU",
    "7267.T": "本田技研工業",
    "7201.T": "日産自動車",
    "7202.T": "いすゞ自動車",
    "7205.T": "日野自動車",
    "7211.T": "三菱自動車工業",
    "7261.T": "マツダ",
    "7272.T": "ヤマハ発動機",
    "8316.T": "三井住友FG",
    "8411.T": "みずほFG",
    "8331.T": "千葉銀行",
    "8308.T": "りそなHD",
    "8309.T": "三井住友トラスト",
    "8354.T": "ふくおかFG",
    "8355.T": "静岡銀行",
    "7182.T": "ゆうちょ銀行",
    "7186.T": "コンコルディアFG",
    "8697.T": "日本取引所グループ",
    "8001.T": "伊藤忠商事",
    "8002.T": "丸紅",
    "8015.T": "豊田通商",
    "2768.T": "双日",
    "8053.T": "住友商事",
    "8056.T": "BIPROGY",
    "8032.T": "日本紙パルプ商事",
    "8012.T": "長瀬産業",
    "8014.T": "蝶理",
    "8037.T": "カメイ",
    "9432.T": "NTT",
    "9433.T": "KDDI",
    "9434.T": "ソフトバンク",
    "9613.T": "NTTデータグループ",
    "9983.T": "ファーストリテイリング",
    "4755.T": "楽天グループ",
    "4689.T": "LINEヤフー",
    "6098.T": "リクルートHD",
    "2413.T": "エムスリー",
    "3659.T": "ネクソン",
    "4063.T": "信越化学工業",
    "4188.T": "三菱ケミカル",
    "4005.T": "住友化学",
    "4004.T": "レゾナックHD",
    "4204.T": "積水化学工業",
    "4502.T": "武田薬品",
    "4503.T": "アステラス製薬",
    "4519.T": "中外製薬",
    "4523.T": "エーザイ",
    "4568.T": "第一三共",
    "5401.T": "日本製鉄",
    "5411.T": "JFE",
    "5711.T": "三菱マテリアル",
    "5801.T": "古河電工",
    "5802.T": "住友電工",
    "5713.T": "住友金属鉱山",
    "6301.T": "コマツ",
    "6302.T": "住友重機械",
    "6367.T": "ダイキン",
    "7011.T": "三菱重工",
    "7012.T": "川崎重工",
    "7013.T": "IHI",
    "9101.T": "日本郵船",
    "9104.T": "商船三井",
    "9107.T": "川崎汽船",
    "9020.T": "JR東日本",
    "9021.T": "JR西日本",
    "9022.T": "JR東海",
    "8801.T": "三井不動産",
    "8802.T": "三菱地所",
}


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
    "relative_strength",
    "ret5",
    "ret20",
    "ma25_slope5",
    "volume_surge",
    "breakout20",
    "trend_alignment",
    "momentum_score",
    "bb_position",
    "bb_width",
    "obv_change",
    "atr_ratio",
    "volatility20",
    "avg_volume_ratio",
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
# ダウンロード
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
                f"{ticker} 空データ "
                f"({attempt}/{retries})"
            )


        except Exception as e:

            print(
                f"{ticker} 取得失敗 "
                f"({attempt}/{retries}): {e}"
            )


        if (
            attempt
            <
            retries
        ):

            wait_sec = (
                base_wait
                *
                (
                    2 **
                    (attempt - 1)
                )
            )

            print(
                f"{wait_sec}秒待機..."
            )

            time.sleep(
                wait_sec
            )


    print(
        f"❌ {ticker} 取得失敗"
    )

    return None


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
                and
                not df.empty
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


        if (
            attempt
            <
            retries
        ):

            wait_sec = (
                base_wait
                *
                (
                    2 **
                    (attempt - 1)
                )
            )

            print(
                f"{wait_sec}秒待機..."
            )

            time.sleep(
                wait_sec
            )


    print(
        "❌ 一括取得失敗"
    )

    return None


# =========================================================
# RSI
# =========================================================

def calc_rsi(
    close,
    period=14
):

    close = (
        close.squeeze()
    )

    delta = (
        close.diff()
    )

    gain = (
        delta.clip(
            lower=0
        )
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
        *
        plus_dm_sm
        /
        atr.replace(
            0,
            np.nan
        )
    )

    minus_di = (
        100
        *
        minus_dm_sm
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

    return (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )


# =========================================================
# 特徴量
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
        calc_rsi(
            close
        )
    )

    df["adx"] = (
        calc_adx(
            df
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

    df["high252"] = (
        high252
    )

    df["low252"] = (
        low252
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


    df["_stock_ret5"] = (
        close
        .pct_change(5)
    )


    # =====================================================
    # テスタ型モメンタム
    # =====================================================

    df["ret5"] = (
        close
        .pct_change(5)
        * 100
    )

    df["ret20"] = (
        close
        .pct_change(20)
        * 100
    )

    df["ma25_slope5"] = (
        (
            df["ma25"]
            /
            df["ma25"].shift(5)
            -
            1
        )
        * 100
    )

    df["volume_surge"] = (
        volume
        /
        volume
        .rolling(5)
        .mean()
    )

    rolling_high20 = (
        close
        .shift(1)
        .rolling(20)
        .max()
    )

    df["breakout20"] = (
        close
        /
        rolling_high20
        -
        1
    ) * 100

    df["trend_alignment"] = (
        (
            close
            >
            df["ma25"]
        ).astype(int)
        +
        (
            df["ma25"]
            >
            df["ma75"]
        ).astype(int)
        +
        (
            df["ma25_slope5"]
            >
            0
        ).astype(int)
    )


    momentum_score = pd.Series(
        0.0,
        index=df.index
    )


    momentum_score += np.where(
        close > df["ma25"],
        20,
        0
    )

    momentum_score += np.where(
        df["ma25"] > df["ma75"],
        20,
        0
    )

    momentum_score += np.where(
        df["ma25_slope5"] > 0,
        15,
        0
    )

    momentum_score += np.where(
        df["ret5"] > 0,
        10,
        0
    )

    momentum_score += np.where(
        df["ret20"] > 0,
        10,
        0
    )

    momentum_score += np.where(
        df["volume_surge"] >= 1.2,
        10,
        0
    )

    momentum_score += np.where(
        df["from_high"] >= -10,
        10,
        0
    )

    momentum_score += np.where(
        df["breakout20"] >= 0,
        5,
        0
    )

    df["momentum_score"] = (
        momentum_score
        .clip(
            0,
            100
        )
    )


    # =====================================================
    # Bollinger
    # =====================================================

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


    # =====================================================
    # OBV
    # =====================================================

    price_direction = np.sign(
        close.diff()
    )

    obv = (
        volume
        *
        price_direction
    ).fillna(0).cumsum()

    df["obv"] = obv

    df["obv_change"] = (
        obv.diff(5)
        /
        volume
        .rolling(5)
        .sum()
        *
        100
    )


    # =====================================================
    # ATR
    # =====================================================

    atr = calc_atr(
        df
    )

    df["atr_ratio"] = (
        atr
        /
        close
        *
        100
    )


    # =====================================================
    # 銘柄特性
    # =====================================================

    df["avg_volume20"] = (
        volume
        .rolling(20)
        .mean()
    )

    df["avg_volume60"] = (
        volume
        .rolling(60)
        .mean()
    )

    df["volatility20"] = (
        df["ret1"]
        .rolling(20)
        .std()
        *
        100
    )

    df["avg_volume_ratio"] = (
        df["avg_volume20"]
        /
        df["avg_volume60"].replace(
            0,
            np.nan
        )
    )


    return df


# =========================================================
# シグナル
# =========================================================

BUY_SIGNALS = {
    "🔥 強い買い",
    "🟢 買い",
}


def signal_category(
    signal
):

    if signal in BUY_SIGNALS:

        return "buy"

    if (
        isinstance(
            signal,
            str
        )
        and
        signal.startswith("🟡")
    ):

        return "monitor"

    return "no_buy"


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
        close
        .rolling(252)
        .max()
        .iloc[-1]
    )


    if (
        not np.isfinite(high52)
        or
        high52 <= 0
    ):

        distance = 0.0

    else:

        distance = (
            price
            /
            high52
            -
            1
        ) * 100


    technical_score = 0.0


    if rsi < 35:

        technical_score += 25


    if macd > signal:

        technical_score += 25


    if ma25 > ma75:

        technical_score += 20


    if vol_ratio > 1.5:

        technical_score += 20


    if distance > -10:

        technical_score += 15

    elif distance > -20:

        technical_score += 8


    nikkei_rsi = float(
        df[
            "nikkei_rsi"
        ].iloc[-1]
    )


    if (
        np.isfinite(
            nikkei_rsi
        )
        and
        nikkei_rsi > 50
    ):

        technical_score += 5


    nikkei_ret = float(
        df[
            "nikkei_return_5d"
        ].iloc[-1]
    )


    if (
        np.isfinite(
            nikkei_ret
        )
        and
        nikkei_ret > 0
    ):

        technical_score += 5


    testa_score = float(
        df[
            "momentum_score"
        ].iloc[-1]
    )


    technical_normalized = (
        technical_score
        /
        115.0
        *
        100
    )


    BASE_TECH_WEIGHT = 0.525
    AI_WEIGHT = 0.225
    TESTA_WEIGHT = 0.25


    ai_score = (
        technical_normalized
        *
        BASE_TECH_WEIGHT
        +
        (
            up_prob
            * 100
        )
        *
        AI_WEIGHT
        +
        testa_score
        *
        TESTA_WEIGHT
    )


    ai_score = max(
        0.0,
        min(
            100.0,
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
        flat_percent >= 50
    )


    try:

        nikkei_ma25 = float(
            df[
                "nikkei_ma25"
            ].iloc[-1]
        )

        nikkei_ma75 = float(
            df[
                "nikkei_ma75"
            ].iloc[-1]
        )

        nikkei_uptrend = (
            nikkei_ma25
            >
            nikkei_ma75
        )

    except Exception:

        nikkei_uptrend = False


    score_policy_ok = (
        ai_score
        >=
        MIN_SCORE_FOR_BUY
    )


    probability_policy_ok = (
        up_percent
        >=
        POLICY_UP_THRESHOLD
        and
        up_percent
        >
        down_percent
    )


    policy_buy_ok = (
        score_policy_ok
        and
        probability_policy_ok
    )


    # =====================================================
    # シグナル
    # =====================================================

    if is_flat_dominant:

        if policy_buy_ok:

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


    elif policy_buy_ok:

        if (
            up_percent
            >=
            max(
                POLICY_UP_THRESHOLD,
                60
            )
            and
            ai_score
            >=
            (
                MIN_SCORE_FOR_BUY
                +
                10
            )
        ):

            final_signal = (
                "🔥 強い買い"
            )

        else:

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
    # テスタ型モメンタム条件
    # =====================================================

    if final_signal in BUY_SIGNALS:

        if (
            testa_score
            <
            55
        ):

            final_signal = (
                "🟡 監視(モメンタム不足)"
            )


    # =====================================================
    # 日経フィルター
    # =====================================================

    if (
        NIKKEI_FILTER_ENABLED
        and
        not nikkei_uptrend
        and
        final_signal
        in
        BUY_SIGNALS
    ):

        final_signal = (
            "🟡 監視(日経下落/レンジ)"
        )


    # =====================================================
    # ATR
    # =====================================================

    atr_value = np.nan


    try:

        high = pd.to_numeric(
            df["High"],
            errors="coerce"
        )

        low = pd.to_numeric(
            df["Low"],
            errors="coerce"
        )

        close_series = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        prev_close = (
            close_series.shift(1)
        )

        true_range = pd.concat(
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
                ).abs()
            ],
            axis=1
        ).max(
            axis=1
        )

        atr_series = (
            true_range
            .rolling(
                14,
                min_periods=5
            )
            .mean()
        )

        atr_candidate = float(
            atr_series.iloc[-1]
        )


        if (
            np.isfinite(
                atr_candidate
            )
            and
            atr_candidate > 0
        ):

            atr_value = (
                atr_candidate
            )


    except Exception:

        pass


    if (
        not np.isfinite(
            atr_value
        )
        or
        atr_value <= 0
    ):

        atr_ratio_value = 2.0

    else:

        atr_ratio_value = (
            atr_value
            /
            price
            *
            100
        )


    take_profit = round(
        price
        *
        (
            1
            +
            (
                atr_ratio_value
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
                atr_ratio_value
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
            round(
                ai_score,
                1
            ),

        "signal":
            final_signal,

        "category":
            signal_category(
                final_signal
            ),

        "nikkei_uptrend":
            bool(
                nikkei_uptrend
            ),

        "technical_score":
            round(
                technical_normalized,
                1
            ),

        "testa_score":
            round(
                testa_score,
                1
            ),

        "ret5":
            round(
                float(
                    df["ret5"].iloc[-1]
                ),
                2
            ),

        "ret20":
            round(
                float(
                    df["ret20"].iloc[-1]
                ),
                2
            ),

        "ma25_slope5":
            round(
                float(
                    df[
                        "ma25_slope5"
                    ].iloc[-1]
                ),
                3
            ),

        "volume_surge":
            round(
                float(
                    df[
                        "volume_surge"
                    ].iloc[-1]
                ),
                2
            ),

        "breakout20":
            round(
                float(
                    df[
                        "breakout20"
                    ].iloc[-1]
                ),
                2
            ),

        "relative_strength":
            round(
                float(
                    df[
                        "relative_strength"
                    ].iloc[-1]
                )
                *
                100,
                2
            )
            if
            "relative_strength" in df.columns
            else
            0.0,

        "price":
            round(
                price,
                0
            ),

        "rsi":
            round(
                rsi,
                1
            ),

        "vol":
            round(
                vol_ratio,
                2
            ),

        "take_profit":
            take_profit,

        "stop_loss":
            stop_loss,

        "up_prob":
            round(
                up_percent,
                1
            ),

        "flat_prob":
            round(
                flat_percent,
                1
            ),

        "down_prob":
            round(
                down_percent,
                1
            ),
    }


# =========================================================
# 日経
# =========================================================

nikkei = safe_download(
    "^N225",
    period="3y",
    interval="1d",
    auto_adjust=True
)


if nikkei is None:

    send(
        "❌ 日経平均データ取得失敗"
    )

    raise SystemExit(1)


if isinstance(
    nikkei.columns,
    pd.MultiIndex
):

    nikkei.columns = (
        nikkei.columns
        .get_level_values(0)
    )


nikkei_close = (
    nikkei["Close"].squeeze()
)


nikkei["nikkei_ma25"] = (
    nikkei_close
    .rolling(25)
    .mean()
)

nikkei["nikkei_ma75"] = (
    nikkei_close
    .rolling(75)
    .mean()
)

nikkei["nikkei_kairi25"] = (
    (
        nikkei_close
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
    calc_rsi(
        nikkei_close
    )
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
    ema12_n
    -
    ema26_n
)

nikkei["nikkei_return_5d"] = (
    nikkei_close
    .pct_change(5)
    *
    100
)

nikkei["nikkei_ret5_raw"] = (
    nikkei_close
    .pct_change(5)
)


# =========================================================
# 先物
# =========================================================

futures = safe_download(
    "NIY=F",
    period="3y",
    interval="1d",
    auto_adjust=True
)


if futures is None:

    send(
        "❌ 先物データ取得失敗"
    )

    raise SystemExit(1)


if isinstance(
    futures.columns,
    pd.MultiIndex
):

    futures.columns = (
        futures.columns
        .get_level_values(0)
    )


future_close = (
    futures["Close"].squeeze()
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
    calc_rsi(
        future_close
    )
)

futures["future_gap"] = (
    (
        future_close
        -
        future_close.shift(1)
    )
    /
    future_close.shift(1)
)


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


# =========================================================
# 学習データ
# =========================================================

def load_training_data():

    if not os.path.exists(
        TRAIN_FILE
    ):

        return None, None


    df = pd.read_csv(
        TRAIN_FILE
    )


    if df.empty:

        return None, None


    required = (
        FEATURES
        +
        ["target"]
    )


    missing = [
        col
        for col
        in required
        if col not in df.columns
    ]


    if missing:

        print(
            "⚠ train_data.csv列不足"
        )

        print(
            missing
        )

        return None, None


    df = df.dropna(
        subset=required
    ).copy()


    if df.empty:

        return None, None


    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )


    df = df[
        df["target"].isin(
            [0, 1, 2]
        )
    ].copy()


    if df.empty:

        return None, None


    X = df[
        FEATURES
    ]

    y = df[
        "target"
    ].astype(int)


    if y.nunique() < 3:

        return None, None


    return X, y


# =========================================================
# バッチ取得
# =========================================================

batch_price_data = (
    safe_download_batch(
        TICKERS,
        period="3y",
        interval="1d",
        auto_adjust=True
    )
)


def get_ticker_ohlcv(
    ticker
):

    if batch_price_data is not None:

        try:

            if isinstance(
                batch_price_data.columns,
                pd.MultiIndex
            ):

                top_level = (
                    batch_price_data
                    .columns
                    .get_level_values(0)
                )


                if ticker in top_level:

                    sub_df = (
                        batch_price_data[
                            ticker
                        ]
                        .dropna(
                            how="all"
                        )
                        .copy()
                    )


                    if (
                        sub_df is not None
                        and
                        len(sub_df) >= 150
                    ):

                        return sub_df


        except Exception as e:

            print(
                f"{ticker} "
                f"一括抽出失敗: {e}"
            )


    return safe_download(
        ticker,
        period="3y",
        interval="1d",
        auto_adjust=True
    )


# =========================================================
# 日次結果判定
# =========================================================

def update_prediction_results():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return


    history = pd.read_csv(
        HISTORY_FILE
    )


    if history.empty:

        return


    required = [
        "date",
        "ticker",
        "price",
        "take_profit",
        "stop_loss",
        "result",
        "return",
        "hold_days",
    ]


    for col in required:

        if col not in history.columns:

            print(
                f"⚠ 必須列不足: {col}"
            )

            return


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


    today = pd.Timestamp.now(
        tz="Asia/Tokyo"
    ).tz_localize(None).normalize()


    for i, row in history.iterrows():

        # 判定済みはスキップ
        if (
            pd.notna(
                row["result"]
            )
            and
            str(
                row["result"]
            ).strip()
            != ""
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
                f"履歴読み込み失敗: {e}"
            )

            continue


        # -------------------------------------------------
        # HOLD_DAYS営業日
        # -------------------------------------------------

        business_days = pd.bdate_range(
            start=(
                prediction_date
                +
                pd.Timedelta(
                    days=1
                )
            ),
            periods=HOLD_DAYS
        )


        if (
            business_days.empty
            or
            business_days[-1]
            >
            today
        ):

            continue


        start_date = (
            business_days[0]
        )

        end_date = (
            business_days[-1]
            +
            pd.Timedelta(
                days=1
            )
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


        if (
            data is None
            or
            data.empty
        ):

            continue


        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )


        if len(data) < HOLD_DAYS:

            continue


        result = None
        return_rate = None
        hold_days = None
        exit_price = None


        for day_index in range(
            HOLD_DAYS
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


            # 同日TP/SL両方
            if (
                low <= stop_loss
                and
                high >= take_profit
            ):

                result = "LOSS"

                exit_price = (
                    stop_loss
                )

                return_rate = (
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

                hold_days = (
                    day_index + 1
                )

                break


            # TP
            if high >= take_profit:

                result = "WIN"

                exit_price = (
                    take_profit
                )

                return_rate = (
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

                hold_days = (
                    day_index + 1
                )

                break


            # SL
            if low <= stop_loss:

                result = "LOSS"

                exit_price = (
                    stop_loss
                )

                return_rate = (
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

                hold_days = (
                    day_index + 1
                )

                break


        # -------------------------------------------------
        # 5日終了
        # -------------------------------------------------

        if result is None:

            close_price = float(
                data.iloc[
                    HOLD_DAYS - 1
                ]["Close"]
            )


            exit_price = (
                close_price
            )


            return_rate = (
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


            hold_days = (
                HOLD_DAYS
            )


            if return_rate < 0:

                result = (
                    "TIMEOUT_LOSS"
                )

            else:

                result = "HOLD"


        # -------------------------------------------------
        # risk_manager
        #
        # ★ペーパートレード専用
        #
        # 実注文は一切しない。
        # -------------------------------------------------

        category = str(
            row.get(
                "category",
                ""
            )
        ).strip()


        if (
            PAPER_TRADING_ONLY
            and
            category == "buy"
            and
            exit_price is not None
        ):

            try:

                register_position_close(
                    ticker,
                    exit_price
                )

            except Exception as e:

                print(
                    f"⚠ risk決済反映失敗 "
                    f"{ticker}: {e}"
                )


        history.at[
            i,
            "result"
        ] = result


        history.at[
            i,
            "return"
        ] = round(
            float(
                return_rate
            ),
            2
        )


        history.at[
            i,
            "hold_days"
        ] = hold_days


        print(
            f"判定 "
            f"{prediction_date.date()} "
            f"{ticker} "
            f"{result} "
            f"{return_rate:+.2f}% "
            f"{hold_days}日"
        )


    history.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# =========================================================
# 起動時に過去予測を判定
# =========================================================

update_prediction_results()


# =========================================================
# モデル
# =========================================================

X_all, y_all = (
    load_training_data()
)


model = None
model_ready = False


if (
    X_all is not None
    and
    y_all is not None
    and
    len(X_all) >= 100
    and
    y_all.nunique() == 3
):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
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
        f"✅ 3クラスモデル学習完了 "
        f"rows={len(X_all)}"
    )


elif os.path.exists(
    MODEL_FILE
):

    try:

        model = joblib.load(
            MODEL_FILE
        )


        if np.array_equal(
            model.classes_,
            np.array([0, 1, 2])
        ):

            model_ready = True

            print(
                "✅ 前回の3クラスモデルを使用"
            )

        else:

            print(
                "❌ model.pklのクラスが不正"
            )

    except Exception as e:

        print(
            f"❌ model.pkl読み込み失敗: {e}"
        )


# =========================================================
# 銘柄解析
# =========================================================

all_data = []
stale_warnings = []


for ticker in TICKERS:

    try:

        print(
            "解析中:",
            ticker
        )


        df = get_ticker_ohlcv(
            ticker
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


        df = create_features(
            df
        )


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


        df["relative_strength"] = (
            df["_stock_ret5"]
            -
            df["nikkei_ret5_raw"]
        )


        df = df.dropna(
            subset=FEATURES
        ).copy()


        if len(df) < 100:

            continue


        latest_date = (
            df.index[-1]
        )


        days_since_latest = (
            pd.Timestamp.now()
            .normalize()
            -
            latest_date.normalize()
        ).days


        if (
            days_since_latest
            >
            7
        ):

            stale_warnings.append(
                f"{ticker}: "
                f"{latest_date.date()} "
                f"({days_since_latest}日前)"
            )


        avg_volume20 = float(
            df[
                "avg_volume20"
            ].iloc[-1]
        )


        if (
            not np.isfinite(
                avg_volume20
            )
            or
            avg_volume20 < 300000
        ):

            print(
                f"⚠ {ticker} 流動性不足"
            )

            continue


        latest = (
            df[
                FEATURES
            ]
            .iloc[-1:]
            .copy()
        )


        close = (
            df["Close"]
            .squeeze()
        )


        all_data.append(
            {

                "ticker":
                    ticker,

                "df":
                    df,

                "latest":
                    latest,

                "close":
                    close,

                "latest_date":
                    latest_date,

            }
        )


    except Exception as e:

        print(
            f"{ticker} エラー: {e}"
        )


# =========================================================
# 予測
# =========================================================

results = []


if model_ready:

    for item in all_data:

        ticker = (
            item["ticker"]
        )

        df = (
            item["df"]
        )

        latest = (
            item["latest"]
        )

        close = (
            item["close"]
        )

        latest_date = (
            item["latest_date"]
        )


        try:

            probabilities = (
                model
                .predict_proba(
                    latest
                )[0]
            )

            classes = list(
                model.classes_
            )


            down_prob = (
                probabilities[
                    classes.index(0)
                ]
            )

            flat_prob = (
                probabilities[
                    classes.index(1)
                ]
            )

            up_prob = (
                probabilities[
                    classes.index(2)
                ]
            )


            data = calc_score(
                df,
                close,
                up_prob,
                down_prob,
                flat_prob
            )


            data.update(
                {
                    "ticker":
                        ticker,

                    "company":
                        COMPANY_NAMES.get(
                            ticker,
                            ""
                        ),

                    "latest_date":
                        latest_date.strftime(
                            "%Y-%m-%d"
                        ),

                }
            )


            results.append(
                data
            )


        except Exception as e:

            print(
                f"{ticker} 予測失敗: {e}"
            )


if not results:

    send(
        "⚪ AI予測データなし"
    )

    raise SystemExit(0)


# =========================================================
# スコア順
# =========================================================

results = sorted(
    results,
    key=lambda x:
        x["score"],
    reverse=True
)


TOP_N = int(
    os.getenv(
        "TOP_N",
        "3"
    )
)


top = results[
    :TOP_N
]


# =========================================================
# ペーパーポジション登録
#
# ★実注文は絶対に行わない
# =========================================================

risk_before = risk_check()


for result in top:

    result["shares"] = 0

    result["risk_note"] = ""


    if (
        result["category"]
        !=
        "buy"
    ):

        result["risk_note"] = (
            "買いシグナル対象外"
        )

        continue


    if not PAPER_TRADING_ONLY:

        result["risk_note"] = (
            "PAPER_TRADING_ONLY=False"
        )

        continue


    current_state = (
        load_risk_state()
    )


    if result["ticker"] in (
        current_state.get(
            "positions",
            {}
        )
    ):

        result["risk_note"] = (
            "既存仮想ポジションあり"
        )

        continue


    check = risk_check()


    if not check[
        "trading_enabled"
    ]:

        result["risk_note"] = (
            "リスク管理により停止: "
            +
            str(
                check["reason"]
            )
        )

        continue


    try:

        plan = build_position_plan(
            check["capital"],
            result["ticker"],
            result["price"],
            result["take_profit"],
            result["stop_loss"]
        )


    except Exception as e:

        result["risk_note"] = (
            f"サイズ計算失敗: {e}"
        )

        continue


    shares = int(
        plan.get(
            "shares",
            0
        )
    )


    if shares <= 0:

        result["risk_note"] = (
            "仮想株数0"
        )

        continue


    # =====================================================
    # ★ここは「仮想登録」のみ
    # 証券会社APIは呼ばない
    # =====================================================

    state = register_position_open(
        result["ticker"],
        shares,
        result["price"]
    )


    if (
        result["ticker"]
        in
        state.get(
            "positions",
            {}
        )
    ):

        result["shares"] = shares

        result["risk_note"] = (
            "ペーパートレード登録"
        )

        print(
            f"📝 PAPER BUY "
            f"{result['ticker']} "
            f"{shares}株"
        )

    else:

        result["risk_note"] = (
            "仮想登録失敗"
        )


# =========================================================
# prediction_history.csv
# =========================================================

save_rows = []


for rank, result in enumerate(
    top,
    start=1
):

    save_rows.append(
        {

            "date":
                datetime.now(
                    ZoneInfo(
                        "Asia/Tokyo"
                    )
                ).strftime(
                    "%Y-%m-%d"
                ),

            "ticker":
                result[
                    "ticker"
                ],

            "rank":
                rank,

            "signal":
                result[
                    "signal"
                ],

            "category":
                result[
                    "category"
                ],

            "score":
                result[
                    "score"
                ],

            "testa_score":
                result[
                    "testa_score"
                ],

            "ret5":
                result[
                    "ret5"
                ],

            "ret20":
                result[
                    "ret20"
                ],

            "ma25_slope5":
                result[
                    "ma25_slope5"
                ],

            "volume_surge":
                result[
                    "volume_surge"
                ],

            "breakout20":
                result[
                    "breakout20"
                ],

            "relative_strength":
                result[
                    "relative_strength"
                ],

            "probability":
                result[
                    "up_prob"
                ],

            "down_probability":
                result[
                    "down_prob"
                ],

            "flat_probability":
                result[
                    "flat_prob"
                ],

            "up_probability":
                result[
                    "up_prob"
                ],

            "nikkei_uptrend":
                bool(
                    result[
                        "nikkei_uptrend"
                    ]
                ),

            "price":
                result[
                    "price"
                ],

            "take_profit":
                result[
                    "take_profit"
                ],

            "stop_loss":
                result[
                    "stop_loss"
                ],

            "shares":
                result.get(
                    "shares",
                    0
                ),

            "paper_trade":
                True,

            "risk_note":
                result.get(
                    "risk_note",
                    ""
                ),

            "data_date":
                result[
                    "latest_date"
                ],

            "result":
                "",

            "return":
                np.nan,

            "hold_days":
                np.nan,
        }
    )


new_df = pd.DataFrame(
    save_rows
)


if os.path.exists(
    HISTORY_FILE
):

    old_df = pd.read_csv(
        HISTORY_FILE
    )


    for col in (
        new_df.columns
    ):

        if col not in old_df.columns:

            old_df[col] = np.nan


    for col in (
        old_df.columns
    ):

        if col not in new_df.columns:

            new_df[col] = np.nan


    history_all = pd.concat(
        [
            old_df,
            new_df
        ],
        ignore_index=True
    )


    history_all = (
        history_all
        .drop_duplicates(
            subset=[
                "date",
                "ticker"
            ],
            keep="last"
        )
    )


else:

    history_all = new_df


history_all.to_csv(
    HISTORY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# Discord
# =========================================================

risk_after = (
    risk_check()
)


msg = []


msg.append(
    "📝 AI株AI ペーパートレード"
)

msg.append(
    "━━━━━━━━━━━━━━━━━━"
)

msg.append(
    "⚠️ 実注文なし"
)

msg.append(
    "1年間の検証用"
)

msg.append("")

msg.append(
    f"⏰ JST "
    f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}"
)

msg.append("")

msg.append(
    "🤖 戦略ポリシー"
)

msg.append(
    f"UP閾値: {POLICY_UP_THRESHOLD}%"
)

msg.append(
    f"最低スコア: {MIN_SCORE_FOR_BUY}"
)

msg.append(
    f"日経フィルター: "
    f"{'ON' if NIKKEI_FILTER_ENABLED else 'OFF'}"
)

msg.append(
    f"ATR TP: {ATR_TP_MULTIPLIER}x"
)

msg.append(
    f"ATR SL: {ATR_SL_MULTIPLIER}x"
)

msg.append(
    f"保有: {HOLD_DAYS}営業日"
)

msg.append("")

msg.append(
    risk_status_text()
)

msg.append("")

msg.append(
    "📊 AIシグナル"
)


for i, result in enumerate(
    top,
    start=1
):

    msg.append(
        ""
    )

    msg.append(
        "━━━━━━━━━━━━━━"
    )

    msg.append(
        f"#{i} "
        f"{result['ticker']} "
        f"{result['company']}"
    )

    msg.append(
        result["signal"]
    )

    msg.append(
        f"AIスコア: "
        f"{result['score']}"
    )

    msg.append(
        f"テスタ型: "
        f"{result['testa_score']}"
    )

    msg.append(
        f"上昇確率: "
        f"{result['up_prob']:.1f}%"
    )

    msg.append(
        f"横ばい確率: "
        f"{result['flat_prob']:.1f}%"
    )

    msg.append(
        f"下落確率: "
        f"{result['down_prob']:.1f}%"
    )

    msg.append(
        f"買値: "
        f"{result['price']:,.0f}"
    )

    msg.append(
        f"利確: "
        f"{result['take_profit']:,.0f}"
    )

    msg.append(
        f"損切: "
        f"{result['stop_loss']:,.0f}"
    )

    msg.append(
        f"仮想株数: "
        f"{result['shares']}株"
    )

    msg.append(
        f"状態: "
        f"{result['risk_note']}"
    )

    msg.append(
        f"データ日付: "
        f"{result['latest_date']}"
    )


if stale_warnings:

    msg.append("")

    msg.append(
        "⚠️ 古いデータ"
    )

    for warning in (
        stale_warnings
    ):

        msg.append(
            f"・{warning}"
        )


msg.append("")

msg.append(
    "📌 この結果は"
    "ペーパートレードであり、"
    "実際の発注は行っていません。"
)


msg.append(
    "📁 prediction_history.csv 更新"
)

msg.append(
    "📁 risk_state.json 更新"
)


final_message = (
    "\n".join(msg)
)


print(
    final_message
)


send(
    final_message
)

# =========================================================
# 月間損益監視・1年間ペーパートレード評価
#
# ★1年間の検証用
#
# 機能:
# 1. 月別損益を集計
# 2. 当月がプラスかマイナスか判定
# 3. マイナス月をDiscord警告
# 4. 2か月連続マイナスを強警告
# 5. 月間勝率 / PF / 平均利益率を表示
# 6. live_monthly_performance.csv 保存
#
# 注意:
# 実注文は一切行わない
# =========================================================

MONTHLY_NEGATIVE_ALERT = True

# 2か月連続マイナスなら強警告
CONSECUTIVE_NEGATIVE_MONTH_ALERT = 2

# 月間評価に必要な最低確定取引数
MIN_MONTHLY_TRADES_FOR_RELIABLE_ALERT = 5


def calculate_monthly_paper_performance():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return pd.DataFrame()


    try:

        df = pd.read_csv(
            HISTORY_FILE
        )

    except Exception as e:

        print(
            f"⚠ 月間評価読み込み失敗: {e}"
        )

        return pd.DataFrame()


    if df.empty:

        return pd.DataFrame()


    required_columns = [
        "date",
        "ticker",
        "category",
        "result",
        "return",
    ]


    for col in required_columns:

        if col not in df.columns:

            print(
                f"⚠ 月間評価に必要な列不足: {col}"
            )

            return pd.DataFrame()


    # -----------------------------------------------------
    # 買い推奨だけを評価
    # -----------------------------------------------------

    df = df[
        df["category"]
        ==
        "buy"
    ].copy()


    if df.empty:

        return pd.DataFrame()


    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    df["return"] = pd.to_numeric(
        df["return"],
        errors="coerce"
    )


    df["result"] = (
        df["result"]
        .astype(str)
        .str.strip()
    )


    df = df.dropna(
        subset=[
            "date",
            "return"
        ]
    ).copy()


    if df.empty:

        return pd.DataFrame()


    # -----------------------------------------------------
    # 確定済み取引のみ
    # -----------------------------------------------------

    df = df[
        df["result"].isin(
            [
                "WIN",
                "LOSS",
                "TIMEOUT_LOSS",
                "HOLD",
            ]
        )
    ].copy()


    if df.empty:

        return pd.DataFrame()


    df["month"] = (
        df["date"]
        .dt
        .to_period("M")
        .astype(str)
    )


    rows = []


    for month, group in (
        df.groupby("month")
    ):

        decided = group[
            group["result"].isin(
                [
                    "WIN",
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            )
        ].copy()


        wins = int(
            (
                decided["result"]
                ==
                "WIN"
            ).sum()
        )


        losses = int(
            decided[
                "result"
            ].isin(
                [
                    "LOSS",
                    "TIMEOUT_LOSS"
                ]
            ).sum()
        )


        trades = (
            wins
            +
            losses
        )


        returns = pd.to_numeric(
            decided["return"],
            errors="coerce"
        ).dropna()


        total_return = float(
            returns.sum()
        ) if len(returns) > 0 else 0.0


        avg_return = float(
            returns.mean()
        ) if len(returns) > 0 else 0.0


        if wins > 0:

            win_returns = returns[
                returns > 0
            ]

        else:

            win_returns = pd.Series(
                dtype=float
            )


        if losses > 0:

            loss_returns = returns[
                returns < 0
            ]

        else:

            loss_returns = pd.Series(
                dtype=float
            )


        gross_profit = float(
            win_returns.sum()
        ) if len(win_returns) > 0 else 0.0


        gross_loss = float(
            -loss_returns.sum()
        ) if len(loss_returns) > 0 else 0.0


        if gross_loss > 0:

            pf = (
                gross_profit
                /
                gross_loss
            )

        elif gross_profit > 0:

            pf = np.inf

        else:

            pf = 0.0


        win_rate = (
            wins
            /
            trades
            *
            100
            if trades > 0
            else 0.0
        )


        holds = int(
            (
                group["result"]
                ==
                "HOLD"
            ).sum()
        )


        rows.append(
            {
                "month":
                    month,

                "trades":
                    trades,

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

                "total_return":
                    total_return,

                "profit_factor":
                    pf,
            }
        )


    monthly_df = pd.DataFrame(
        rows
    )


    if monthly_df.empty:

        return monthly_df


    monthly_df = (
        monthly_df
        .sort_values(
            "month"
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # 月間プラス / マイナス
    # -----------------------------------------------------

    monthly_df["monthly_status"] = np.where(
        monthly_df["total_return"] > 0,
        "PLUS",
        np.where(
            monthly_df["total_return"] < 0,
            "MINUS",
            "FLAT"
        )
    )


    # -----------------------------------------------------
    # 連続マイナス月
    # -----------------------------------------------------

    consecutive = 0
    consecutive_values = []


    for status in (
        monthly_df["monthly_status"]
    ):

        if status == "MINUS":

            consecutive += 1

        else:

            consecutive = 0


        consecutive_values.append(
            consecutive
        )


    monthly_df[
        "consecutive_negative_months"
    ] = consecutive_values


    return monthly_df


def save_monthly_paper_report():

    monthly_df = (
        calculate_monthly_paper_performance()
    )


    if monthly_df.empty:

        print(
            "⚠ 月間ペーパー実績: データなし"
        )

        return monthly_df


    output_file = (
        "paper_monthly_performance.csv"
    )


    monthly_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"✅ 月間ペーパー実績保存: "
        f"{output_file}"
    )


    return monthly_df


def build_monthly_alert():

    monthly_df = (
        calculate_monthly_paper_performance()
    )


    if monthly_df.empty:

        return (
            "📊 月間ペーパー実績\n"
            "まだ確定取引がありません"
        )


    latest = (
        monthly_df.iloc[-1]
    )


    month = (
        latest["month"]
    )


    trades = int(
        latest["trades"]
    )


    wins = int(
        latest["wins"]
    )


    losses = int(
        latest["losses"]
    )


    win_rate = float(
        latest["win_rate"]
    )


    avg_return = float(
        latest["avg_return"]
    )


    total_return = float(
        latest["total_return"]
    )


    pf = latest[
        "profit_factor"
    ]


    if np.isinf(pf):

        pf_display = "inf"

    else:

        pf_display = (
            f"{float(pf):.2f}"
        )


    consecutive_negative = int(
        latest[
            "consecutive_negative_months"
        ]
    )


    if total_return > 0:

        status_text = (
            "🟢 月間プラス"
        )

    elif total_return < 0:

        status_text = (
            "🔴 月間マイナス"
        )

    else:

        status_text = (
            "🟡 月間±0"
        )


    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📅 月間ペーパー実績"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"対象月: {month}"
    )

    lines.append(
        f"状態: {status_text}"
    )

    lines.append(
        f"確定取引: {trades}件"
    )

    lines.append(
        f"WINS: {wins}件"
    )

    lines.append(
        f"LOSSES: {losses}件"
    )

    lines.append(
        f"勝率: {win_rate:.1f}%"
    )

    lines.append(
        f"平均利益率: {avg_return:+.2f}%"
    )

    lines.append(
        f"月間合計: {total_return:+.2f}%"
    )

    lines.append(
        f"PF: {pf_display}"
    )

    lines.append(
        f"連続マイナス月: "
        f"{consecutive_negative}ヶ月"
    )


    # -----------------------------------------------------
    # 少数サンプル注意
    # -----------------------------------------------------

    if (
        trades
        <
        MIN_MONTHLY_TRADES_FOR_RELIABLE_ALERT
    ):

        lines.append(
            ""
        )

        lines.append(
            "⚠️ 確定取引が少ないため"
        )

        lines.append(
            "月間成績は参考値です"
        )


    # -----------------------------------------------------
    # マイナス月警告
    # -----------------------------------------------------

    if (
        MONTHLY_NEGATIVE_ALERT
        and
        total_return < 0
    ):

        lines.append(
            ""
        )

        lines.append(
            "⚠️⚠️⚠️ 月間損失警告"
        )

        lines.append(
            "今月はペーパー収支がマイナスです"
        )


    # -----------------------------------------------------
    # 連続マイナス警告
    # -----------------------------------------------------

    if (
        consecutive_negative
        >=
        CONSECUTIVE_NEGATIVE_MONTH_ALERT
    ):

        lines.append(
            ""
        )

        lines.append(
            "🚨🚨🚨 戦略劣化警告"
        )

        lines.append(
            f"{consecutive_negative}"
            "ヶ月連続マイナスです"
        )

        lines.append(
            "実運用への移行は停止してください"
        )


    # -----------------------------------------------------
    # 直近3か月
    # -----------------------------------------------------

    recent = (
        monthly_df
        .tail(3)
    )


    lines.append(
        ""
    )

    lines.append(
        "【直近3ヶ月】"
    )


    for _, row in (
        recent.iterrows()
    ):

        row_pf = row[
            "profit_factor"
        ]


        if np.isinf(
            row_pf
        ):

            row_pf_text = "inf"

        else:

            row_pf_text = (
                f"{float(row_pf):.2f}"
            )


        lines.append(
            f"{row['month']} "
            f"件数={int(row['trades'])} "
            f"勝率={row['win_rate']:.1f}% "
            f"月間={row['total_return']:+.2f}% "
            f"PF={row_pf_text}"
        )


    return "\n".join(
        lines
    )


# =========================================================
# 月間評価実行
# =========================================================

monthly_paper_df = (
    save_monthly_paper_report()
)


monthly_alert_message = (
    build_monthly_alert()
)


print("")
print(
    monthly_alert_message
)


# =========================================================
# Discordへ月間評価を送信
# =========================================================

send(
    monthly_alert_message
)


print("")
print(
    "==========================================="
)

print(
    "✅ PAPER TRADE SCAN 完了"
)

print(
    "==========================================="
)

print(
    f"解析銘柄数: {len(all_data)}"
)

print(
    f"TOP_N: {TOP_N}"
)

print(
    f"ペーパートレード: {PAPER_TRADING_ONLY}"
)

print(
    f"仮想資金: "
    f"{risk_after['capital']:,.0f}円"
)

print(
    f"利用可能資金: "
    f"{risk_after['available_cash']:,.0f}円"
)

print(
    f"DD: "
    f"{risk_after['drawdown'] * 100:.2f}%"
)

print(
    f"連敗: "
    f"{risk_after['consecutive_losses']}"
)

print(
    "実注文: なし"
)
