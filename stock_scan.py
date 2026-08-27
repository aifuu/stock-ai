import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import joblib

import json

from sklearn.ensemble import RandomForestClassifier

# ===================== 
# 自動戦略ポリシー 
# ===================== 
POLICY_FILE = "strategy_policy.json" 
 
DEFAULT_POLICY_UP_THRESHOLD = 50 
DEFAULT_MIN_SCORE_FOR_BUY = 60 
DEFAULT_NIKKEI_FILTER = False 
 
DEFAULT_ATR_TP_MULTIPLIER = 3.0 
DEFAULT_ATR_SL_MULTIPLIER = 1.5 
DEFAULT_HOLD_DAYS = 5 
 

# =====================
# 安全なBoolean変換
#
# ★重要
#
# Python組み込みのbool()は、空文字列以外の文字列を
# 全てTrueと判定してしまう。
#
#   bool("false")  → True (誤り)
#   bool("0")      → True (誤り)
#
# strategy_policy.jsonの"nikkei_filter"が何らかの経路で
# 文字列型("false"等)になった場合、このバグにより
# 日経フィルターが意図せずONになってしまう危険がある。
# 明示的に文字列の中身を見て判定する。
# =====================
def parse_bool(value, default=False):

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):

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
        "status": "DEFAULT", 
 
        "up_threshold": DEFAULT_POLICY_UP_THRESHOLD, 
        "min_score_for_buy": DEFAULT_MIN_SCORE_FOR_BUY, 
 
        "nikkei_filter": DEFAULT_NIKKEI_FILTER, 
 
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
 
        # ===================== 
        # 安全チェック 
        # ===================== 
 
        if merged["status"] not in [ 
            "APPROVED", 
            "DEFAULT" 
        ]: 
 
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
            "⚠ strategy_policy.json" 
            f"読み込み失敗: {e}" 
        ) 
 
        print( 
            "→ デフォルト設定を使用" 
        ) 
 
        return default_policy 
 
 
STRATEGY_POLICY = load_strategy_policy() 
 
 
# ===================== 
# 本番で使用する設定 
# ===================== 
 
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
# Discord
# =========================================================
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send(msg):
    if not WEBHOOK_URL:
        print("❌ Webhookなし")
        return

    if len(msg) > 1900:
        msg = msg[:1900]

    try:
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

    except Exception as e:
        print("❌ Discord送信エラー:", e)


# =========================================================
# 銘柄
# =========================================================
# =========================================================
# 銘柄
#
# ★94銘柄構成(元14銘柄 + 追加80銘柄)
#
# 半導体・電機/自動車/銀行・金融/商社/通信/化学・医薬/
# 鉄鋼・非鉄/機械/海運/鉄道/不動産で業種分散させた
# AIスキャン対象の監視母集団。
#
# ※これは「買い推奨銘柄リスト」ではなく、AIが日々スコアリング
# する対象の母集団。実際に買うかどうかはcalc_score()の
# 判定(ポリシー・日経フィルター・テスタ型モメンタム等)で決まる。
#
# ※104銘柄のご依頼に対し、いただいたリストを数えたところ
# 追加分は90ではなく80銘柄でした(10銘柄×8グループ)。
# 14+80=94銘柄です。104ちょうどにしたい場合は追加10銘柄が
# 必要です。
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


# =========================================================
# ★COMPANY_NAMESはDiscordメッセージ表示用のラベルのみ。
#
# calc_score()等の判定ロジックには一切使用されないため、
# 万一名称が誤っていてもスキャン結果やシグナル判定には
# 影響しない。念のため確信度がやや低い銘柄には
# "(要確認)"を付けている。
# =========================================================
COMPANY_NAMES = {
    # ---- 元の14銘柄 ----
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

    # ---- 半導体・電機 ----
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

    # ---- 自動車 ----
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

    # ---- 銀行・金融 ----
    "8316.T": "三井住友フィナンシャルグループ",
    "8411.T": "みずほフィナンシャルグループ",
    "8331.T": "千葉銀行",
    "8308.T": "りそなホールディングス",
    "8309.T": "三井住友トラスト・ホールディングス",
    "8354.T": "ふくおかフィナンシャルグループ",
    "8355.T": "静岡銀行",
    "7182.T": "ゆうちょ銀行",
    "7186.T": "コンコルディア・フィナンシャルグループ",
    "8697.T": "日本取引所グループ",

    # ---- 商社 ----
    "8001.T": "伊藤忠商事",
    "8002.T": "丸紅",
    "8015.T": "豊田通商",
    "2768.T": "双日",
    "8053.T": "住友商事",
    "8056.T": "BIPROGY(要確認)",
    "8032.T": "日本紙パルプ商事(要確認)",
    "8012.T": "長瀬産業",
    "8014.T": "蝶理",
    "8037.T": "カメイ",

    # ---- 通信・ネット ----
    "9432.T": "日本電信電話(NTT)",
    "9433.T": "KDDI",
    "9434.T": "ソフトバンク",
    "9613.T": "NTTデータグループ",
    "9983.T": "ファーストリテイリング",
    "4755.T": "楽天グループ",
    "4689.T": "LINEヤフー",
    "6098.T": "リクルートホールディングス",
    "2413.T": "エムスリー",
    "3659.T": "ネクソン",

    # ---- 化学・医薬 ----
    "4063.T": "信越化学工業",
    "4188.T": "三菱ケミカルグループ",
    "4005.T": "住友化学",
    "4004.T": "レゾナックHD",
    "4204.T": "積水化学工業",
    "4502.T": "武田薬品工業",
    "4503.T": "アステラス製薬",
    "4519.T": "中外製薬",
    "4523.T": "エーザイ",
    "4568.T": "第一三共",

    # ---- 鉄鋼・非鉄・機械 ----
    "5401.T": "日本製鉄",
    "5411.T": "JFEホールディングス",
    "5711.T": "三菱マテリアル",
    "5801.T": "古河電気工業",
    "5802.T": "住友電気工業",
    "5713.T": "住友金属鉱山",
    "6301.T": "コマツ",
    "6302.T": "住友重機械工業",
    "6367.T": "ダイキン工業",
    "7011.T": "三菱重工業",

    # ---- 機械・海運・鉄道・不動産 ----
    "7012.T": "川崎重工業",
    "7013.T": "IHI",
    "9101.T": "日本郵船",
    "9104.T": "商船三井",
    "9107.T": "川崎汽船",
    "9020.T": "東日本旅客鉄道",
    "9021.T": "西日本旅客鉄道",
    "9022.T": "東海旅客鉄道",
    "8801.T": "三井不動産",
    "8802.T": "三菱地所",
}



TRAIN_FILE = "train_data.csv"
MODEL_FILE = "model.pkl"


# =========================================================
# 3クラス分類設定
# =========================================================
DOWN_THRESHOLD = -1.5
UP_THRESHOLD = 1.5
FORWARD_DAYS = 3


# =========================================================
# ATRターゲット設定
# =========================================================
ATR_TARGET_MULTIPLIER = 1.0


# =========================================================
# TOP件数
# =========================================================
TOP_N = int(os.getenv("TOP_N", "3"))


# =========================================================
# データ鮮度チェック
# =========================================================
STALE_DATA_WARNING_DAYS = 7


# =========================================================
# 流動性フィルター
# =========================================================
MIN_AVG_VOLUME = 300000


# =========================================================
# テスタ型モメンタム設定
#
# 「安いから買う」よりも、
# ・上昇トレンド
# ・直近の強さ
# ・出来高を伴う上昇
# ・高値圏での強さ
# ・日経に対する相対強度
# を数値化して評価する。
#
# 特定個人の非公開手法を再現するものではなく、
# 公開情報から一般化した「テスタ型」の検証用実装。
# =========================================================
TESTA_MOMENTUM_WEIGHT = 0.25
TESTA_MIN_SCORE_FOR_BUY = 55.0
TESTA_STRONG_SCORE = 75.0


# =========================================================
# 買い推奨 / 監視 の分類
#
# ★改善点⑤
#
# prediction_history.csv の成績集計を汚さないよう、
# 「実際に買う想定のシグナル」と「監視だけのシグナル」を
# ここで明示的に分離する。
#
# BUY_SIGNALS  … 実戦成績(勝率・PF等)の集計対象
# MONITOR_SIGNALS … 参考データとして記録するが勝率には含めない
# それ以外(🔴 買わない等) … 成績対象外
# =========================================================
BUY_SIGNALS = {
    "🔥 強い買い",
    "🟢 買い",
}


def signal_category(signal):
    if signal in BUY_SIGNALS:
        return "buy"
    elif isinstance(signal, str) and signal.startswith("🟡"):
        return "monitor"
    else:
        return "no_buy"


# =========================================================
# 学習・予測で使う特徴量
#
# ★改善点②
#
# 銘柄をOne-Hotで覚えさせるのではなく、
# 「その銘柄が今どんな性質か(出来高規模・変動率など)」を
# スケール非依存な形で特徴量に加える。
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
# ダウンロード失敗時のリトライ
#
# ★改善点⑥
#
# 指数バックオフ(2秒→4秒→8秒→16秒…)にして、
# yfinance側のレート制限に配慮する。
# =========================================================
def safe_download(ticker, retries=5, base_wait=2, **kwargs):

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

            wait_sec = base_wait * (2 ** (attempt - 1))

            print(
                f"{wait_sec}秒待機してリトライします..."
            )

            time.sleep(wait_sec)

    print(
        f"❌ {ticker} リトライ{retries}回失敗"
    )

    return None


# =========================================================
# 複数銘柄の一括ダウンロード
#
# ★改善点⑥
#
# 銘柄ごとに個別アクセスするのではなく、
# 可能な限りまとめて1回のリクエストで取得し、
# API呼び出し回数を減らす。
# 失敗した場合は呼び出し側で個別リトライにフォールバックする。
# =========================================================
def safe_download_batch(tickers, retries=5, base_wait=2, **kwargs):

    for attempt in range(1, retries + 1):

        try:
            df = yf.download(
                tickers,
                group_by="ticker",
                threads=True,
                **kwargs
            )

            if df is not None and not df.empty:
                return df

            print(
                f"一括取得: 空データ "
                f"(試行{attempt}/{retries})"
            )

        except Exception as e:

            print(
                f"一括取得失敗 "
                f"(試行{attempt}/{retries}): {e}"
            )

        if attempt < retries:

            wait_sec = base_wait * (2 ** (attempt - 1))

            print(
                f"{wait_sec}秒待機してリトライします..."
            )

            time.sleep(wait_sec)

    print(
        "❌ 一括取得 リトライ失敗"
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

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


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
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
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
        100
        * plus_dm_sm
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm_sm
        / atr.replace(0, np.nan)
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


# =========================================================
# 特徴量作成
# =========================================================
def create_features(df):

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # -----------------------------------------------------
    # 基本
    # -----------------------------------------------------
    df["ret1"] = close.pct_change()

    df["ma25"] = (
        close.rolling(25).mean()
    )

    df["ma75"] = (
        close.rolling(75).mean()
    )

    df["vol_ratio"] = (
        volume
        /
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
        df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    df["high252"] = (
        close.rolling(252).max()
    )

    df["low252"] = (
        close.rolling(252).min()
    )

    df["from_high"] = (
        close / df["high252"] - 1
    ) * 100

    df["from_low"] = (
        close / df["low252"] - 1
    ) * 100


    # -----------------------------------------------------
    # テスタ型モメンタム特徴量
    # -----------------------------------------------------
    df["ret5"] = (
        close.pct_change(5) * 100
    )

    df["ret20"] = (
        close.pct_change(20) * 100
    )

    # MA25が5日前より上向いているかを数値化
    df["ma25_slope5"] = (
        (
            df["ma25"]
            /
            df["ma25"].shift(5)
            - 1
        ) * 100
    )

    # 短期の出来高急増
    df["volume_surge"] = (
        volume
        /
        volume.rolling(5).mean()
    )

    # 20日高値をどれだけ上抜いているか
    rolling_high20 = (
        close.shift(1).rolling(20).max()
    )

    df["breakout20"] = (
        close / rolling_high20 - 1
    ) * 100

    # トレンド方向の整合性
    df["trend_alignment"] = (
        (close > df["ma25"]).astype(int)
        +
        (df["ma25"] > df["ma75"]).astype(int)
        +
        (df["ma25_slope5"] > 0).astype(int)
    )

    # 0～100のテスタ型モメンタムスコア
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
    # 相対強度用中間列
    # -----------------------------------------------------
    df["_stock_ret5"] = (
        close.pct_change(5)
    )


    # -----------------------------------------------------
    # ボリンジャーバンド
    # -----------------------------------------------------
    bb_ma20 = (
        close.rolling(20).mean()
    )

    bb_std20 = (
        close.rolling(20).std()
    )

    bb_upper = (
        bb_ma20
        + (bb_std20 * 2)
    )

    bb_lower = (
        bb_ma20
        - (bb_std20 * 2)
    )

    df["bb_position"] = (
        (close - bb_lower)
        /
        (bb_upper - bb_lower)
    )

    df["bb_width"] = (
        (bb_upper - bb_lower)
        /
        bb_ma20
        * 100
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
            * price_direction
        )
        .fillna(0)
        .cumsum()
    )

    df["obv"] = obv

    df["obv_change"] = (
        obv.diff(5)
        /
        volume.rolling(5).sum()
        * 100
    )


    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = (
        tr.rolling(14).mean()
    )

    df["atr_ratio"] = (
        atr
        / close
        * 100
    )


    # -----------------------------------------------------
    # ★改善点②: 銘柄特性(スケール非依存)
    #
    # volatility20      … 直近20日の日次リターンのばらつき(%)
    # avg_volume_ratio  … 直近20日平均出来高 / 直近60日平均出来高
    #                      (銘柄間の出来高規模差を吸収した「今が
    #                       出来高多いか少ないか」の相対指標)
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
# AIスコア・最終判定
#
# ★重要部分
#
# 日経MA25 > MA75
#     ↓
# 買い系シグナル許可
#
# 日経MA25 <= MA75
#     ↓
# 買い系シグナル禁止
#     ↓
# 「🟡 監視」に落とす
#
# AI確率・AIスコア自体は変更しない。
# =========================================================
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

    rsi = float(df["rsi"].iloc[-1])
    macd = float(df["macd"].iloc[-1])
    signal = float(df["signal"].iloc[-1])
    ma25 = float(df["ma25"].iloc[-1])
    ma75 = float(df["ma75"].iloc[-1])
    vol_ratio = float(df["vol_ratio"].iloc[-1])

    # -----------------------------------------------------
    # 52週高値からの距離
    # -----------------------------------------------------
    high52 = float(
        close.rolling(252).max().iloc[-1]
    )

    if not np.isfinite(high52) or high52 <= 0:
        distance = 0.0
    else:
        distance = (
            price / high52 - 1
        ) * 100

    # =====================================================
    # 従来テクニカルスコア
    # =====================================================
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

    # -----------------------------------------------------
    # 日経テクニカル
    # -----------------------------------------------------
    nikkei_rsi_value = float(
        df["nikkei_rsi"].iloc[-1]
    )

    if np.isfinite(nikkei_rsi_value):
        if nikkei_rsi_value > 50:
            technical_score += 5

    nikkei_return_value = float(
        df["nikkei_return_5d"].iloc[-1]
    )

    if np.isfinite(nikkei_return_value):
        if nikkei_return_value > 0:
            technical_score += 5

    # =====================================================
    # テスタ型モメンタム
    # =====================================================
    testa_score = float(
        df["momentum_score"].iloc[-1]
    )

    ret5 = float(
        df["ret5"].iloc[-1]
    )

    ret20 = float(
        df["ret20"].iloc[-1]
    )

    ma25_slope5 = float(
        df["ma25_slope5"].iloc[-1]
    )

    volume_surge = float(
        df["volume_surge"].iloc[-1]
    )

    breakout20 = float(
        df["breakout20"].iloc[-1]
    )

    relative_strength = float(
        df["relative_strength"].iloc[-1]
    )

    # =====================================================
    # スコア正規化
    # =====================================================
    MAX_TECHNICAL_SCORE = 115.0

    technical_score_normalized = (
        technical_score
        / MAX_TECHNICAL_SCORE
        * 100
    )

    # =====================================================
    # AI + テクニカル + テスタ
    # =====================================================
    BASE_TECH_WEIGHT = 0.525
    AI_WEIGHT = 0.225
    TESTA_WEIGHT = TESTA_MOMENTUM_WEIGHT

    ai_score = (
        technical_score_normalized
        * BASE_TECH_WEIGHT
        +
        (up_prob * 100)
        * AI_WEIGHT
        +
        testa_score
        * TESTA_WEIGHT
    )

    ai_score = max(
        0.0,
        min(100.0, ai_score)
    )

    # =====================================================
    # 確率
    # =====================================================
    up_percent = float(up_prob * 100)
    down_percent = float(down_prob * 100)
    flat_percent = float(flat_prob * 100)

    # =====================================================
    # 拮抗・横ばい判定
    # =====================================================
    FLAT_DOMINANT_THRESHOLD = 50.0
    TIE_MARGIN = 1.0

    is_tie = (
        abs(up_percent - down_percent)
        <= TIE_MARGIN
    )

    is_flat_dominant = (
        flat_percent >= FLAT_DOMINANT_THRESHOLD
    )

    # =====================================================
    # 日経トレンド
    # =====================================================
    nikkei_uptrend = False

    try:
        nikkei_ma25_value = float(
            df["nikkei_ma25"].iloc[-1]
        )

        nikkei_ma75_value = float(
            df["nikkei_ma75"].iloc[-1]
        )

        if (
            np.isfinite(nikkei_ma25_value)
            and np.isfinite(nikkei_ma75_value)
        ):
            nikkei_uptrend = (
                nikkei_ma25_value
                >
                nikkei_ma75_value
            )

    except Exception as e:
        print(
            "⚠ 日経トレンド判定失敗:",
            e
        )

    # =====================================================
    # ★ ポリシー条件
    # =====================================================
    #
    # MIN_SCORE_FOR_BUY:
    #   総合AIスコアの最低ライン
    #
    # POLICY_UP_THRESHOLD:
    #   AI上昇確率の最低ライン
    #
    # この2つを満たさない限り
    # 「買い」「強い買い」にはしない。
    # =====================================================

    score_policy_ok = (
        ai_score >= MIN_SCORE_FOR_BUY
    )

    probability_policy_ok = (
        up_percent >= POLICY_UP_THRESHOLD
        and
        up_percent > down_percent
    )

    policy_buy_ok = (
        score_policy_ok
        and
        probability_policy_ok
    )

    # =====================================================
    # 通常AI判定
    # =====================================================
    if is_flat_dominant:

        if policy_buy_ok:
            final_signal = "🟡 監視(横ばい優勢)"
        else:
            final_signal = "🔴 買わない"

    elif is_tie:

        final_signal = "🟡 監視(拮抗)"

    elif policy_buy_ok:

        # -------------------------------------------------
        # ポリシー条件を満たした銘柄だけ
        # 強い買い / 買いへ進む
        # -------------------------------------------------
        if (
            up_percent >= max(
                POLICY_UP_THRESHOLD,
                60.0
            )
            and
            ai_score >= (
                MIN_SCORE_FOR_BUY + 10
            )
        ):
            final_signal = "🔥 強い買い"

        else:
            final_signal = "🟢 買い"

    elif (
        up_percent >= 40
        and
        up_percent > down_percent
    ):

        final_signal = "🟡 監視"

    else:

        final_signal = "🔴 買わない"

    # =====================================================
    # ★ テスタ型フィルター
    # =====================================================
    testa_weak = (
        testa_score < TESTA_MIN_SCORE_FOR_BUY
    )

    if final_signal in (
        "🔥 強い買い",
        "🟢 買い"
    ):

        if testa_weak:

            final_signal = (
                "🟡 監視(モメンタム不足)"
            )

    # =====================================================
    # ★ 日経フィルター
    # =====================================================
    #
    # NIKKEI_FILTER_ENABLED = True
    # の場合のみ有効。
    #
    # 日経25MA <= 75MAなら
    # 買い → 監視へ落とす。
    # =====================================================
    if (
        NIKKEI_FILTER_ENABLED
        and
        not nikkei_uptrend
    ):

        if final_signal in (
            "🔥 強い買い",
            "🟢 買い"
        ):

            final_signal = (
                "🟡 監視(日経下落/レンジ)"
            )

    # =====================================================
    # ★ 最終スコア条件
    # =====================================================
    #
    # 何らかのフィルターによって
    # 買い条件を満たさなくなった場合、
    # 強い買い/買いを残さない。
    # =====================================================

    if final_signal in (
        "🔥 強い買い",
        "🟢 買い"
    ):

        if not score_policy_ok:
            final_signal = (
                "🟡 監視(スコア不足)"
            )

        elif not probability_policy_ok:
            final_signal = (
                "🟡 監視(上昇確率不足)"
            )

    # =====================================================
    # ★ ATRをOHLCから実計算
    # =====================================================
    #
    # atr_ratioを固定値として使用しない。
    #
    # True Range:
    #   High - Low
    #   |High - 前日Close|
    #   |Low  - 前日Close|
    #
    # 14日ATRを計算し、
    # 現在価格に対するATR比率から
    # TP / SLを決定する。
    # =====================================================

    atr_value = np.nan

    try:

        if all(
            col in df.columns
            for col in [
                "High",
                "Low",
                "Close"
            ]
        ):

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

            tr1 = high - low

            tr2 = (
                high - prev_close
            ).abs()

            tr3 = (
                low - prev_close
            ).abs()

            true_range = pd.concat(
                [
                    tr1,
                    tr2,
                    tr3
                ],
                axis=1
            ).max(axis=1)

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
                np.isfinite(atr_candidate)
                and
                atr_candidate > 0
            ):

                atr_value = (
                    atr_candidate
                )

    except Exception as e:

        print(
            "⚠ ATR計算失敗:",
            e
        )

    # =====================================================
    # ATR計算できなかった場合のみ
    # 既存のatr_ratioをフォールバック
    # =====================================================
    if (
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):

        try:

            if "atr_ratio" in df.columns:

                atr_ratio_fallback = float(
                    df["atr_ratio"].iloc[-1]
                )

                if (
                    np.isfinite(
                        atr_ratio_fallback
                    )
                    and
                    atr_ratio_fallback > 0
                ):

                    atr_value = (
                        price
                        * atr_ratio_fallback
                        / 100.0
                    )

        except Exception:
            pass

    # =====================================================
    # ATRがどうしても取れない場合
    # =====================================================
    #
    # 固定ATRを勝手に入れるのではなく、
    # TP/SLを現在価格から一定割合で作る。
    #
    # ※これはATRではない。
    # =====================================================
    if (
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):

        print(
            "⚠ ATRを取得できないため"
            "TP/SLを安全側のフォールバックで計算"
        )

        atr_ratio_value = 2.0

    else:

        atr_ratio_value = (
            atr_value
            / price
            * 100.0
        )

    # =====================================================
    # TP / SL
    # =====================================================
    take_profit = round(
        price
        * (
            1.0
            +
            (
                atr_ratio_value
                / 100.0
            )
            *
            ATR_TP_MULTIPLIER
        ),
        0
    )

    stop_loss = round(
        price
        * (
            1.0
            -
            (
                atr_ratio_value
                / 100.0
            )
            *
            ATR_SL_MULTIPLIER
        ),
        0
    )

    # =====================================================
    # 最終結果
    # =====================================================
    return {
        "score": round(
            ai_score,
            1
        ),

        "signal": final_signal,

        "category": signal_category(
            final_signal
        ),

        "nikkei_uptrend": bool(
            nikkei_uptrend
        ),

        "technical_score": round(
            technical_score_normalized,
            1
        ),

        "testa_score": round(
            testa_score,
            1
        ),

        "ret5": round(
            ret5,
            2
        ),

        "ret20": round(
            ret20,
            2
        ),

        "ma25_slope5": round(
            ma25_slope5,
            3
        ),

        "volume_surge": round(
            volume_surge,
            2
        ),

        "breakout20": round(
            breakout20,
            2
        ),

        "relative_strength": round(
            relative_strength * 100,
            2
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
            up_percent,
            1
        ),

        "flat_prob": round(
            flat_percent,
            1
        ),

        "down_prob": round(
            down_percent,
            1
        ),
    }

# =========================================================
# 過去予測のHOLD_DAYS営業日以内結果判定
#
# ★改善点①
#
# 「行数がHOLD_DAYSあるから確定」ではなく、
# 予測日からHOLD_DAYS営業日経過したかどうかで判定する。
# 祝日・データ欠測等で実際の取得行数がHOLD_DAYSに満たない場合は、
# 無理に判定を確定させず「判定保留」として次回に回す。
#
# ★修正点: HOLD_DAYSはstrategy_policy.jsonから読み込んだ値を使用。
#           以前は5固定だったため、ポリシー変更が反映されなかった。
# =========================================================
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


        # 予測日の翌営業日からHOLD_DAYS営業日分
        business_days = pd.bdate_range(
            start=(
                prediction_date
                +
                pd.Timedelta(days=1)
            ),
            periods=HOLD_DAYS
        )


        if business_days[-1] > today:
            # まだHOLD_DAYS営業日経過していない → 判定保留
            continue


        start_date = business_days[0]

        end_date = (
            business_days[-1]
            +
            pd.Timedelta(days=1)
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

            print(
                f"{ticker} 株価データなし(判定保留)"
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


        # ---------------------------------------------------
        # ★HOLD_DAYS営業日分のデータが揃っていない場合は判定保留
        #
        # 祝日・システム障害等でデータが欠測していると、
        # 本来HOLD_DAYS営業日経過していても実データが
        # HOLD_DAYS行未満のことがある。
        # ここで無理にHOLD/TIMEOUT_LOSSに確定させると
        # 「実際には未検証の結果」が成績に混入してしまうため、
        # 次回のcron実行に持ち越す。
        # ---------------------------------------------------
        if len(data) < HOLD_DAYS:

            print(
                f"{ticker} {prediction_date.date()} "
                f"判定保留: 営業日データ{len(data)}/{HOLD_DAYS}件しか取得できていません"
            )

            continue


        result = None
        return_rate = None
        hold_days = None


        check_days = HOLD_DAYS


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


            if high >= take_profit:

                result = "WIN"

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


            if low <= stop_loss:

                result = "LOSS"

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
                    -
                    entry_price
                )
                /
                entry_price
                *
                100
            )

            hold_days = check_days


            if (
                hold_days >= HOLD_DAYS
                and
                return_rate < 0
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


# =========================================================
# 過去予測結果更新
# =========================================================
update_prediction_results()


# =========================================================
# 日経225：市場トレンド判定
#
# ★本番フィルター
#
# 日経MA25 > MA75
#     → 上昇トレンド
#     → 買い系シグナル許可
#
# 日経MA25 <= MA75
#     → 下落/レンジ
#     → 買い系シグナル禁止
#
# 将来データは使用しない。
# =========================================================
nikkei = safe_download(
    "^N225",
    period="3y",
    interval="1d",
    auto_adjust=True
)


if nikkei is None:

    send(
        "❌ 日経平均データ取得失敗のため処理中断"
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
    nikkei["Close"].squeeze()
)


# ---------------------------------------------------------
# 日経MA25
# ---------------------------------------------------------
nikkei["nikkei_ma25"] = (
    nikkei_close
    .rolling(25)
    .mean()
)


# ---------------------------------------------------------
# 日経MA75
# ---------------------------------------------------------
nikkei["nikkei_ma75"] = (
    nikkei_close
    .rolling(75)
    .mean()
)


# ---------------------------------------------------------
# 日経乖離率
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# ★日経上昇トレンド
# ---------------------------------------------------------
nikkei["nikkei_uptrend"] = (
    nikkei["nikkei_ma25"]
    >
    nikkei["nikkei_ma75"]
)


# ---------------------------------------------------------
# 日経RSI
# ---------------------------------------------------------
nikkei["nikkei_rsi"] = (
    calc_rsi(
        nikkei_close
    )
)


# ---------------------------------------------------------
# 日経MACD
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 日経5日リターン
# ---------------------------------------------------------
nikkei["nikkei_return_5d"] = (
    nikkei_close.pct_change(5)
    * 100
)


# ---------------------------------------------------------
# 日経相対強度用中間列
# ---------------------------------------------------------
nikkei["nikkei_ret5_raw"] = (
    nikkei_close.pct_change(5)
)


# =========================================================
# 日経225先物
# =========================================================
futures = safe_download(
    "NIY=F",
    period="3y",
    interval="1d",
    auto_adjust=True
)


if futures is None:

    send(
        "❌ 先物データ取得失敗のため処理中断"
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
    futures["Close"].squeeze()
)


futures["future_return"] = (
    future_close.pct_change()
)

futures["future_ma5"] = (
    future_close.rolling(5).mean()
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


# ---------------------------------------------------------
# 先物は1日ラグ
# ---------------------------------------------------------
futures["future_return"] = (
    futures["future_return"].shift(1)
)

futures["future_ma5"] = (
    futures["future_ma5"].shift(1)
)

futures["future_rsi"] = (
    futures["future_rsi"].shift(1)
)

futures["future_gap"] = (
    futures["future_gap"].shift(1)
)


# =========================================================
# 学習データ読み込み
# =========================================================
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
        FEATURES
        +
        ["target"]
    )


    missing = [
        c
        for c in required_cols
        if c not in df.columns
    ]


    if missing:

        print(
            "⚠ train_data.csvのスキーマが古いため無視します"
        )

        print(
            f"不足列: {missing}"
        )

        return None, None


    df = df.dropna(
        subset=required_cols
    )


    if len(df) == 0:

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


    if len(df) == 0:

        return None, None


    X = df[FEATURES]

    y = df["target"].astype(
        int
    )


    if y.nunique() < 3:

        print(
            "⚠ 3クラス "
            "(0/1/2) が揃っていないため、"
            "既存学習データは使用しません"
        )

        return None, None


    return X, y


# =========================================================
# 時系列ホールドアウト検証
# =========================================================
def time_series_validation(
    X,
    y,
    test_ratio=0.20
):

    if (
        X is None
        or
        y is None
    ):

        return None


    if len(X) < 200:

        print(
            "⚠ 検証データ不足:",
            len(X)
        )

        return None


    split = int(
        len(X)
        *
        (1 - test_ratio)
    )


    train_X = X.iloc[
        :split
    ].copy()

    train_y = y.iloc[
        :split
    ].copy()

    test_X = X.iloc[
        split:
    ].copy()

    test_y = y.iloc[
        split:
    ].copy()


    if (
        train_y.nunique() < 3
        or
        test_y.nunique() < 3
    ):

        print(
            "⚠ 検証側で3クラスが揃っていません"
        )

        return None


    validation_model = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=42,
            class_weight="balanced"
        )
    )


    validation_model.fit(
        train_X,
        train_y
    )


    predictions = (
        validation_model.predict(
            test_X
        )
    )


    accuracy = (
        (
            predictions
            ==
            test_y.values
        ).mean()
        *
        100
    )


    print("")
    print("=====================")
    print("📊 時系列ホールドアウト検証")
    print("=====================")
    print(
        f"全データ: {len(X)}件"
    )
    print(
        f"学習: {len(train_X)}件"
    )
    print(
        f"検証: {len(test_X)}件"
    )
    print(
        f"Accuracy: {accuracy:.2f}%"
    )


    for cls in [0, 1, 2]:

        actual = (
            test_y == cls
        )

        predicted = (
            predictions == cls
        )

        tp = (
            actual & predicted
        ).sum()

        fp = (
            ~actual & predicted
        ).sum()


        precision = (
            tp
            /
            (tp + fp)
            *
            100
            if
            (tp + fp) > 0
            else 0
        )


        print(
            f"クラス{cls} Precision: "
            f"{precision:.2f}%"
        )


    return {
        "accuracy":
            round(
                accuracy,
                2
            ),

        "train_size":
            len(train_X),

        "test_size":
            len(test_X)
    }


# =========================================================
# 学習データ保存
# =========================================================
def save_training_data(
    new_df
):

    if os.path.exists(
        TRAIN_FILE
    ):

        old_df = pd.read_csv(
            TRAIN_FILE
        )


        if (
            "target"
            in old_df.columns
        ):

            target_values = (
                pd.to_numeric(
                    old_df["target"],
                    errors="coerce"
                )
                .dropna()
                .unique()
            )


            if (
                set(target_values)
                -
                {0, 1, 2}
            ):

                print(
                    "⚠ 旧形式の学習データを破棄して3クラス用に再構築します"
                )

                old_df = (
                    pd.DataFrame()
                )


        df_all = pd.concat(
            [
                old_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        df_all = new_df


    df_all = (
        df_all.drop_duplicates(
            subset=[
                "date",
                "ticker"
            ],
            keep="last"
        )
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


# =========================================================
# モデル
# =========================================================
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

stale_warnings = []


# =========================================================
# ★改善点⑥: 全銘柄まとめて一括取得
#
# 銘柄ごとに個別リクエストするのではなく、
# まず一括ダウンロードを試み、
# 取得できなかった銘柄だけ個別にフォールバックする。
# =========================================================
batch_price_data = safe_download_batch(
    TICKERS,
    period="3y",
    interval="1d",
    auto_adjust=True
)


def get_ticker_ohlcv(ticker):

    # 一括取得データから該当銘柄を抽出
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

                    if sub_df is not None and len(sub_df) >= 150:

                        return sub_df

        except Exception as e:

            print(
                f"{ticker} 一括データ抽出失敗: {e}"
            )

    # 一括取得で十分なデータが取れなかった銘柄は個別にリトライ
    print(
        f"{ticker} 個別ダウンロードにフォールバック"
    )

    return safe_download(
        ticker,
        period="3y",
        interval="1d",
        auto_adjust=True
    )


# =========================================================
# メイン処理
# =========================================================
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


        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )


        # =================================================
        # 日経特徴量
        # =================================================
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


        # =================================================
        # 先物特徴量
        # =================================================
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


        # =================================================
        # 相対強度
        # =================================================
        df["relative_strength"] = (
            df["_stock_ret5"]
            -
            df["nikkei_ret5_raw"]
        )


        df = df.dropna()


        print(
            ticker,
            "dropna後データ数",
            len(df)
        )


        if len(df) < 100:

            print(
                ticker,
                "データ不足"
            )

            continue


        # =================================================
        # 最新データ
        # =================================================
        full_df = df.copy()

        close = (
            full_df["Close"]
            .squeeze()
        )

        latest_date = (
            full_df.index[-1]
        )


        days_since_latest = (
            pd.Timestamp.now().normalize()
            -
            latest_date.normalize()
        ).days


        if (
            days_since_latest
            >
            STALE_DATA_WARNING_DAYS
        ):

            warn_text = (
                f"{ticker}: "
                f"最新データ "
                f"{latest_date.date()} "
                f"({days_since_latest}日前)"
            )

            print(
                f"⚠ {warn_text}"
            )

            stale_warnings.append(
                warn_text
            )


        # =================================================
        # 流動性フィルター
        # =================================================
        latest_avg_volume = (
            full_df[
                "avg_volume20"
            ].iloc[-1]
        )


        if (
            pd.isna(
                latest_avg_volume
            )
            or
            latest_avg_volume
            <
            MIN_AVG_VOLUME
        ):

            print(
                f"⚠ {ticker} 流動性不足 "
                f"平均出来高="
                f"{latest_avg_volume:.0f}"
            )

            continue


        # =================================================
        # ATRベース3クラスtarget
        #
        # ★改善点④(確認)
        #
        # FORWARD_DAYS(3営業日)先の株価がまだ確定していない
        # 直近の行は future_return_target が NaN になるため、
        # 下の notna() フィルターで自動的に学習データから
        # 除外される。つまり「予測はできるが正解がまだ
        # 分からない直近3日分」は絶対に学習に混ざらない。
        # =================================================
        future_price = (
            full_df["Close"]
            .shift(-FORWARD_DAYS)
        )


        future_return_target = (
            future_price
            /
            full_df["Close"]
            -
            1
        ) * 100


        atr_threshold = (
            full_df["atr_ratio"]
            *
            ATR_TARGET_MULTIPLIER
            *
            np.sqrt(
                FORWARD_DAYS
            )
        )


        train_df = (
            full_df.copy()
        )


        train_df["target"] = (
            np.select(
                [
                    future_return_target
                    <=
                    -atr_threshold,

                    future_return_target
                    >=
                    atr_threshold
                ],

                [
                    0,
                    2
                ],

                default=1
            )
        )


        # -------------------------------------------------
        # ★未来が確定していない行(直近FORWARD_DAYS日分)を除外
        # -------------------------------------------------
        train_df = (
            train_df[
                future_return_target.notna()
            ].copy()
        )


        train_df["target"] = (
            train_df["target"]
            .astype(int)
        )


        if len(train_df) < 100:

            print(
                ticker,
                "学習データ不足"
            )

            continue


        print(
            ticker,
            "学習データ数=",
            len(train_df)
        )


        class_counts = (
            train_df["target"]
            .value_counts()
            .sort_index()
        )


        print(
            ticker,
            "クラス分布:",
            class_counts.to_dict()
        )


        if (
            train_df["target"]
            .nunique()
            <
            3
        ):

            print(
                ticker,
                "3クラス揃わないためスキップ"
            )

            continue


        X = train_df[
            FEATURES
        ]

        y = train_df[
            "target"
        ]


        train_rows = X.copy()

        train_rows["target"] = (
            y.values
        )

        train_rows["date"] = (
            X.index.strftime(
                "%Y-%m-%d"
            )
        )

        train_rows["ticker"] = (
            ticker
        )


        all_train_rows.append(
            train_rows
        )


        # =================================================
        # 予測用最新行
        #
        # ★改善点④(確認)
        #
        # ここで使う latest 行(=今日時点の最新行)は、
        # target算出時にshift(-FORWARD_DAYS)でNaNになった行、
        # つまり上のtrain_dfには含まれていない未確定データ。
        # 「予測には使うが学習には使わない」を徹底している。
        # =================================================
        latest = (
            full_df[
                FEATURES
            ]
            .iloc[-1:]
            .copy()
        )


        all_data.append(
            {
                "ticker":
                    ticker,

                "latest":
                    latest,

                "close":
                    close,

                "df":
                    full_df,

                "latest_date":
                    latest_date,
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


# =========================================================
# 学習データ保存
# =========================================================
if all_train_rows:

    new_train_df = pd.concat(
        all_train_rows,
        ignore_index=True
    )

    save_training_data(
        new_train_df
    )


# =========================================================
# 学習データ読み込み
# =========================================================
X_all, y_all = (
    load_training_data()
)


# =========================================================
# 時系列ホールドアウト検証
# =========================================================
validation_result = None


if (
    X_all is not None
    and
    y_all is not None
):

    validation_result = (
        time_series_validation(
            X_all,
            y_all,
            test_ratio=0.20
        )
    )


# =========================================================
# 3クラスモデル学習
# =========================================================
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
        "❌ 3クラス学習データ・モデルともになし"
    )

    print(
        "今回は予測をスキップ"
    )


# =========================================================
# 特徴量重要度
# =========================================================
if model_ready:

    importances = (
        model.feature_importances_
    )


    if (
        len(importances)
        !=
        len(FEATURES)
    ):

        print("")
        print(
            "====================="
        )
        print(
            "⚠ 特徴量重要度スキップ"
        )
        print(
            "====================="
        )

        print(
            f"読み込んだモデルの特徴量数"
            f"({len(importances)})と"
            f"現在のFEATURES"
            f"({len(FEATURES)})が一致しません。"
        )

        print(
            "旧FEATURES構成のmodel.pklが使われている可能性があります。"
        )

    else:

        importance_df = pd.DataFrame(
            {
                "feature":
                    FEATURES,

                "importance":
                    importances
            }
        )


        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        print("")
        print(
            "====================="
        )
        print(
            "📊 特徴量重要度"
        )
        print(
            "====================="
        )


        for _, row in (
            importance_df.iterrows()
        ):

            print(
                f"{row['feature']:<25} "
                f"{row['importance']:.6f}"
            )


        importance_df.to_csv(
            "feature_importances.csv",
            index=False,
            encoding="utf-8-sig"
        )


        print(
            "✅ feature_importances.csv 保存"
        )


# =========================================================
# 一括予測
# =========================================================
if model_ready:

    for item in all_data:

        latest = item[
            "latest"
        ]

        df = item[
            "df"
        ]

        close = item[
            "close"
        ]

        ticker = item[
            "ticker"
        ]

        latest_date = item[
            "latest_date"
        ]


        try:

            probabilities = (
                model.predict_proba(
                    latest
                )[0]
            )

            classes = list(
                model.classes_
            )


            down_index = (
                classes.index(0)
            )

            flat_index = (
                classes.index(1)
            )

            up_index = (
                classes.index(2)
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


        # =================================================
        # 最終スコア・判定
        # =================================================
        data = calc_score(
            df,
            close,
            up_prob,
            down_prob,
            flat_prob
        )


        results.append(
            {
                "ticker":
                    ticker,

                "score":
                    data["score"],

                "signal":
                    data["signal"],

                "category":
                    data["category"],

                "nikkei_uptrend":
                    bool(
                        data.get(
                            "nikkei_uptrend",
                            False
                        )
                    ),

                "technical_score":
                    data[
                        "technical_score"
                    ],

                "testa_score":
                    data[
                        "testa_score"
                    ],

                "ret5":
                    data["ret5"],

                "ret20":
                    data["ret20"],

                "ma25_slope5":
                    data["ma25_slope5"],

                "volume_surge":
                    data["volume_surge"],

                "breakout20":
                    data["breakout20"],

                "relative_strength":
                    data["relative_strength"],

                "prob":
                    data["up_prob"],

                "up_prob":
                    data["up_prob"],

                "flat_prob":
                    data["flat_prob"],

                "down_prob":
                    data["down_prob"],

                "price":
                    data["price"],

                "rsi":
                    data["rsi"],

                "vol":
                    data["vol"],

                "take_profit":
                    data[
                        "take_profit"
                    ],

                "stop_loss":
                    data[
                        "stop_loss"
                    ],

                "latest_date":
                    latest_date.strftime(
                        "%Y-%m-%d"
                    ),
            }
        )


        print(
            f"{ticker} "
            f"score={data['score']} "
            f"testa={data['testa_score']} "
            f"判定={data['signal']} "
            f"区分={data['category']} "
            f"日経上昇トレンド="
            f"{'YES' if data.get('nikkei_uptrend', False) else 'NO'} "
            f"下落={data['down_prob']:.1f}% "
            f"横ばい={data['flat_prob']:.1f}% "
            f"上昇={data['up_prob']:.1f}% "
            f"データ日付="
            f"{latest_date.date()}"
        )


# =========================================================
# 結果なし
# =========================================================
if not results:

    send(
        "⚪ データなし"
    )

    exit()


# =========================================================
# スコア順
# =========================================================
results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)


top = results[
    :TOP_N
]


# =========================================================
# Discordメッセージ
# =========================================================
msg = (
    f"⏰ JST: "
    f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}"
    "\n\n"
)


# =========================================================
# データ鮮度警告
# =========================================================
if stale_warnings:

    msg += (
        "⚠️ データが古い可能性のある銘柄\n"
    )

    for w in stale_warnings:

        msg += (
            f"・{w}\n"
        )

    msg += "\n"


# =========================================================
# ★日経トレンド全体表示
# =========================================================
try:

    latest_nikkei_ma25 = float(
        nikkei[
            "nikkei_ma25"
        ].iloc[-1]
    )

    latest_nikkei_ma75 = float(
        nikkei[
            "nikkei_ma75"
        ].iloc[-1]
    )

    latest_nikkei_close = float(
        nikkei_close.iloc[-1]
    )

    latest_nikkei_uptrend = (
        latest_nikkei_ma25
        >
        latest_nikkei_ma75
    )

except Exception:

    latest_nikkei_ma25 = np.nan
    latest_nikkei_ma75 = np.nan
    latest_nikkei_close = np.nan
    latest_nikkei_uptrend = False


msg += (
    "📈 日経225トレンド判定\n"
)

msg += (
    f"日経終値: "
    f"{latest_nikkei_close:.0f}\n"
)

msg += (
    f"MA25: "
    f"{latest_nikkei_ma25:.0f}\n"
)

msg += (
    f"MA75: "
    f"{latest_nikkei_ma75:.0f}\n"
)

msg += (
    "日経上昇トレンド: "
    f"{'YES 🟢' if latest_nikkei_uptrend else 'NO 🔴'}\n"
)

msg += "\n"


msg += (
    "📊 AI株スキャン結果\n"
    "【3クラス分類】\n\n"
)


# =========================================================
# TOP銘柄表示
# =========================================================
for i, r in enumerate(top):

    rank = r["signal"]


    msg += f"""
━━━━━━━━━━━━━━
#{i+1} {r['ticker']} {COMPANY_NAMES.get(r['ticker'], '')}

{rank}

AIスコア: {r['score']}
テクニカル: {r['technical_score']}
テスタ型モメンタム: {r['testa_score']}
5日騰落率: {r['ret5']:+.2f}%
20日騰落率: {r['ret20']:+.2f}%
MA25傾き(5日): {r['ma25_slope5']:+.3f}%
出来高急増率: {r['volume_surge']:.2f}倍
20日高値突破: {r['breakout20']:+.2f}%
日経対比強度: {r['relative_strength']:+.2f}%
日経上昇トレンド: {'YES 🟢' if r['nikkei_uptrend'] else 'NO 🔴'}

下落確率: {r['down_prob']}%
横ばい確率: {r['flat_prob']}%
上昇確率: {r['up_prob']}%

買値: {r['price']}
利確: {r['take_profit']}
損切: {r['stop_loss']}

RSI: {r['rsi']}
出来高倍率: {r['vol']}
データ日付: {r['latest_date']}
━━━━━━━━━━━━━━
"""


# =========================================================
# 予測履歴保存
#
# ★改善点⑤
#
# 「買い推奨(buy)」「監視(monitor)」「対象外(no_buy)」の
# 区分(category)を必ず保存する。
# これにより後段の成績集計で「買い推奨だけの勝率」を
# 正確に出せるようにする。
# =========================================================
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
                    ZoneInfo(
                        "Asia/Tokyo"
                    )
                ).strftime(
                    "%Y-%m-%d"
                ),

            "ticker":
                r["ticker"],

            "rank":
                rank,

            "signal":
                r["signal"],

            "category":
                r["category"],

            "score":
                r["score"],

            "testa_score":
                r["testa_score"],

            "ret5":
                r["ret5"],

            "ret20":
                r["ret20"],

            "ma25_slope5":
                r["ma25_slope5"],

            "volume_surge":
                r["volume_surge"],

            "breakout20":
                r["breakout20"],

            "relative_strength":
                r["relative_strength"],

            "probability":
                r["up_prob"],

            "down_probability":
                r["down_prob"],

            "flat_probability":
                r["flat_prob"],

            "up_probability":
                r["up_prob"],

            # ★日経トレンド保存
            "nikkei_uptrend":
                bool(
                    r[
                        "nikkei_uptrend"
                    ]
                ),

            "price":
                r["price"],

            "take_profit":
                r[
                    "take_profit"
                ],

            "stop_loss":
                r[
                    "stop_loss"
                ],

            "data_date":
                r[
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
    history_file
):

    old_df = pd.read_csv(
        history_file
    )


    # -----------------------------------------------------
    # 旧CSVにsignal/categoryが存在しない行は
    # 「区分不明(旧データ)」として扱い、成績集計からは除外する。
    # -----------------------------------------------------
    if "category" not in old_df.columns:

        old_df["category"] = np.nan

    if "signal" not in old_df.columns:

        old_df["signal"] = np.nan


    df_all = pd.concat(
        [
            old_df,
            new_df
        ],
        ignore_index=True
    )


    df_all = (
        df_all.drop_duplicates(
            subset=[
                "date",
                "ticker"
            ],
            keep="last"
        )
    )

else:

    df_all = new_df


df_all.to_csv(
    history_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    "✅ prediction_history.csv 保存完了"
)


# =========================================================
# AI成績評価
#
# ★改善点⑤
#
# 「買い推奨(category == buy)」だけを実戦成績として集計する。
# 「監視(monitor)」は参考データとして別枠表示、
# 「対象外(no_buy / 区分不明の旧データ)」は集計から除外する。
# =========================================================
def _summarize(df):

    total = len(df)

    wins = (
        df["result"]
        == "WIN"
    ).sum()

    losses = (
        df["result"]
        .isin(
            [
                "LOSS",
                "TIMEOUT_LOSS"
            ]
        )
    ).sum()

    holds = (
        df["result"]
        == "HOLD"
    ).sum()

    decided = wins + losses

    win_rate = (
        wins
        /
        decided
        *
        100
        if decided > 0
        else 0
    )

    returns = (
        df["return"]
        .dropna()
    )

    if len(returns) > 0:

        avg_return = returns.mean()
        best = returns.max()
        worst = returns.min()

        gains = returns[returns > 0].sum()

        loss_sum = (
            -returns[returns < 0].sum()
        )

        profit_factor = (
            gains / loss_sum
            if loss_sum > 0
            else np.nan
        )

    else:

        avg_return = 0
        best = 0
        worst = 0
        profit_factor = np.nan

    days = (
        df["hold_days"]
        .dropna()
    )

    avg_days = (
        days.mean()
        if len(days) > 0
        else 0
    )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "holds": holds,
        "decided": decided,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "best": best,
        "worst": worst,
        "profit_factor": profit_factor,
        "avg_days": avg_days,
    }


def show_ai_performance():

    file = (
        "prediction_history.csv"
    )


    if not os.path.exists(file):

        return ""


    df = pd.read_csv(
        file
    )


    if "category" not in df.columns:

        df["category"] = np.nan


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

        return (
            "\n📊 AI実績\n\n"
            "まだ判定データなし\n"
        )


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


    # -----------------------------------------------------
    # 区分ごとに分離
    # -----------------------------------------------------
    buy_df = (
        result_df[
            result_df["category"] == "buy"
        ]
    )

    monitor_df = (
        result_df[
            result_df["category"] == "monitor"
        ]
    )


    buy_stat = _summarize(buy_df)
    monitor_stat = _summarize(monitor_df)


    pf_text = (
        f"{buy_stat['profit_factor']:.2f}"
        if pd.notna(buy_stat["profit_factor"])
        else "N/A"
    )


    # -----------------------------------------------------
    # 買い推奨の順位別内訳(rank 1..TOP_N)
    # -----------------------------------------------------
    rank_text = ""


    for rank in range(
        1,
        TOP_N + 1
    ):

        rank_df = (
            buy_df[
                buy_df["rank"]
                ==
                rank
            ]
        )


        rank_total = len(
            rank_df
        )


        if rank_total == 0:

            rank_text += (
                f"\n#{rank}位\n\n"
                "データなし(買い推奨なし)\n"
            )

            continue


        rank_stat = _summarize(rank_df)

        rank_pf_text = (
            f"{rank_stat['profit_factor']:.2f}"
            if pd.notna(rank_stat["profit_factor"])
            else "N/A"
        )


        rank_text += f"""
#{rank}位

勝率: {rank_stat['win_rate']:.1f}%
WIN: {rank_stat['wins']}件
LOSS: {rank_stat['losses']}件
HOLD: {rank_stat['holds']}件
判定数: {rank_stat['total']}件

平均利益率: {rank_stat['avg_return']:.2f}%
Profit Factor: {rank_pf_text}
平均保有日数: {rank_stat['avg_days']:.1f}日
最高利益: {rank_stat['best']:+.2f}%
最大損失: {rank_stat['worst']:.2f}%
"""


    return f"""
━━━━━━━━━━━━━━
📊 AI実績（買い推奨のみ集計）
（🔥強い買い・🟢買い / {HOLD_DAYS}営業日判定）
━━━━━━━━━━━━━━

【買い推奨 全体】

判定数: {buy_stat['total']}件

勝ち: {buy_stat['wins']}件
負け: {buy_stat['losses']}件
HOLD: {buy_stat['holds']}件

勝率: {buy_stat['win_rate']:.1f}%
Profit Factor: {pf_text}

平均利益率: {buy_stat['avg_return']:.2f}%
平均保有日数: {buy_stat['avg_days']:.1f}日

最高利益: {buy_stat['best']:+.2f}%
最大損失: {buy_stat['worst']:.2f}%

━━━━━━━━━━━━━━
🏆 買い推奨 順位別
━━━━━━━━━━━━━━
{rank_text}
━━━━━━━━━━━━━━
🟡 監視シグナル(参考データ・成績には含めない)
━━━━━━━━━━━━━━

判定数: {monitor_stat['total']}件
勝率(参考): {monitor_stat['win_rate']:.1f}%
平均利益率(参考): {monitor_stat['avg_return']:.2f}%

━━━━━━━━━━━━━━
"""


# =========================================================
# 最終表示・Discord送信
# =========================================================
performance = (
    show_ai_performance()
)


msg += performance


print(msg)


send(msg)
