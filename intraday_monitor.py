"""
intraday_monitor.py

東証の取引時間中、30分おきに実行する「場中監視モード」。

日足AI (ai_stock_scan.py) とは完全に独立したスクリプトで、
日足AIのロジックには一切影響を与えない。

やること:
1. 今日が東証営業日か判定(祝日・年末年始・土日はスキップ)
2. 今が寄り前/昼休み/引け後ならスキップ(ログのみ、Discordは送らない)
3. 前場・後場の間だけ、30分足データを取得して場中スコアを計算
4. スコア上位をDiscordに通知

【重要な注意点】
・yfinanceの30分足は取得できる期間や更新タイミングに制限があり、
  リアルタイムより数分遅延することがある。
・このスコアは「今この瞬間の値動きの強さ」を見るための
  ルールベースの指標であり、日足AI(RandomForest)の
  上昇確率とは別物。両者を厳密に合成した「場中専用AIモデル」を
  作る場合は、別途、場中データでの学習パイプラインが必要になる
  (このスクリプトはその前段階として、まずは特徴量を集めて
  スコアリングする構成にしてある)。
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import (
    TICKERS,
    COMPANY_NAMES,
    safe_download,
    calc_rsi,
    is_tse_trading_day,
    get_market_phase,
    send,
)


JST = ZoneInfo("Asia/Tokyo")

TOP_N = int(os.getenv("TOP_N", "3"))


# =====================
# 30分足データの前処理
# (タイムゾーンをJSTに統一し、本日分だけ切り出す)
# =====================
def prepare_intraday(df, now_jst):

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert(JST)

    today_df = df[df.index.date == now_jst.date()]

    return df, today_df


# =====================
# 日経平均・日経225先物の当日騰落率
# =====================
def get_market_context(now_jst):

    context = {
        "nikkei_change": 0.0,
        "futures_change": 0.0,
    }

    nikkei = safe_download(
        "^N225", period="5d", interval="30m", progress=False
    )

    if nikkei is not None and not nikkei.empty:

        _, nikkei_today = prepare_intraday(nikkei, now_jst)

        if len(nikkei_today) >= 1:

            open_price = float(nikkei_today["Open"].iloc[0])
            latest_price = float(nikkei_today["Close"].iloc[-1])

            if open_price > 0:
                context["nikkei_change"] = (
                    (latest_price / open_price) - 1
                ) * 100

    futures = safe_download(
        "NIY=F", period="5d", interval="30m", progress=False
    )

    if futures is not None and not futures.empty:

        _, futures_today = prepare_intraday(futures, now_jst)

        if len(futures_today) >= 1:

            open_price = float(futures_today["Open"].iloc[0])
            latest_price = float(futures_today["Close"].iloc[-1])

            if open_price > 0:
                context["futures_change"] = (
                    (latest_price / open_price) - 1
                ) * 100

    return context


# =====================
# 場中スコア計算(ルールベース、0〜100)
#
# 追加した特徴量:
# ・30分足の短期リターン(直近1本 / 直近2本=約1時間)
# ・30分足RSI
# ・30分足MACD
# ・出来高急増(本日の30分足平均に対する倍率)
# ・寄り付きからの騰落率
# ・当日高値からの位置
# ・日経平均の当日騰落
# ・日経225先物の当日騰落
# =====================
def calc_intraday_score(ticker_today, ticker_full, market_context):

    close_full = ticker_full["Close"].squeeze()

    if len(close_full) < 15:
        # RSI/MACDの計算に必要な本数が足りない
        return None

    rsi30 = float(calc_rsi(close_full).iloc[-1])

    ema12 = close_full.ewm(span=12, adjust=False).mean()
    ema26 = close_full.ewm(span=26, adjust=False).mean()
    macd30 = ema12 - ema26
    signal30 = macd30.ewm(span=9, adjust=False).mean()

    macd_latest = float(macd30.iloc[-1])
    signal_latest = float(signal30.iloc[-1])

    latest_close = float(close_full.iloc[-1])

    # 直近30分・直近1時間のリターン
    short_return_30m = float(close_full.pct_change().iloc[-1] * 100)

    if len(close_full) >= 3:
        short_return_1h = float(
            (close_full.iloc[-1] / close_full.iloc[-3] - 1) * 100
        )
    else:
        short_return_1h = 0.0

    # 本日データが薄い場合(寄り付き直後など)はそこまでで計算
    today_open = float(ticker_today["Open"].iloc[0])
    today_high = float(ticker_today["High"].max())
    today_low = float(ticker_today["Low"].min())

    open_change = (
        (latest_close / today_open - 1) * 100
        if today_open > 0
        else 0.0
    )

    near_high = (
        latest_close >= today_high * 0.98
        if today_high > 0
        else False
    )

    # 出来高急増(本日の直近本 vs 本日それまでの平均)
    vol_today = ticker_today["Volume"]

    if len(vol_today) >= 2:
        latest_vol = float(vol_today.iloc[-1])
        avg_vol = float(vol_today.iloc[:-1].mean())
        vol_surge = latest_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_surge = 1.0

    # =====================
    # スコア加点(合計100点満点)
    # =====================
    score = 0

    if rsi30 < 35:
        score += 15

    if macd_latest > signal_latest:
        score += 15

    if vol_surge > 1.5:
        score += 15

    if open_change > 0:
        score += 15

    if near_high:
        score += 10

    if market_context["nikkei_change"] > 0:
        score += 10

    if market_context["futures_change"] > 0:
        score += 10

    if short_return_30m > 0:
        score += 10

    score = max(0, min(100, score))

    # =====================
    # シグナル判定
    # =====================
    if score >= 70:
        signal = "🔥 強いモメンタム"
    elif score >= 55:
        signal = "🟢 上昇モメンタム"
    elif score >= 40:
        signal = "🟡 監視"
    else:
        signal = "🔴 弱い"

    return {
        "score": round(score, 1),
        "signal": signal,
        "price": round(latest_close, 1),
        "rsi30": round(rsi30, 1),
        "open_change": round(open_change, 2),
        "short_return_30m": round(short_return_30m, 2),
        "short_return_1h": round(short_return_1h, 2),
        "vol_surge": round(vol_surge, 2),
        "today_high": round(today_high, 1),
        "today_low": round(today_low, 1),
    }


def main():

    now = datetime.now(JST)

    # =====================
    # 東証営業日チェック
    # =====================
    if not is_tse_trading_day(now.date()):
        print(f"休場日({now.date()})のため終了")
        return

    # =====================
    # 取引時間チェック
    # (寄り前・昼休み・引け後はログのみでDiscordは送らない)
    # =====================
    phase = get_market_phase(now)

    if phase in ("before_open", "lunch", "after_close"):
        print(
            f"取引時間外(phase={phase}, "
            f"JST {now.strftime('%H:%M')})のためスキップ"
        )
        return

    print(f"場中監視開始 phase={phase} JST {now.strftime('%H:%M')}")

    market_context = get_market_context(now)

    print(
        "日経平均 当日騰落:", round(market_context["nikkei_change"], 2),
        "% / 日経225先物 当日騰落:",
        round(market_context["futures_change"], 2), "%"
    )

    results = []

    for ticker in TICKERS:

        try:
            intraday = safe_download(
                ticker, period="5d", interval="30m", progress=False
            )

            if intraday is None or intraday.empty:
                print(f"{ticker} 30分足データなし")
                continue

            full_df, today_df = prepare_intraday(intraday, now)

            if len(today_df) < 1:
                print(f"{ticker} 本日分の30分足データがまだありません")
                continue

            data = calc_intraday_score(today_df, full_df, market_context)

            if data is None:
                print(f"{ticker} データ不足のためスコア計算不可")
                continue

            results.append({"ticker": ticker, **data})

            print(
                f"{ticker} score={data['score']} 判定={data['signal']} "
                f"寄付比={data['open_change']}% "
                f"RSI30={data['rsi30']}"
            )

        except Exception as e:
            print(ticker, "エラー:", e)

    if not results:
        print("有効な結果なし。Discord送信をスキップ")
        return

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[:TOP_N]

    phase_label = {
        "morning": "前場",
        "afternoon": "後場",
    }.get(phase, phase)

    msg = (
        f"⏱ JST {now.strftime('%Y-%m-%d %H:%M')} "
        f"（{phase_label}・30分監視）\n\n"
        f"日経平均 当日騰落: {market_context['nikkei_change']:+.2f}%\n"
        f"日経225先物 当日騰落: {market_context['futures_change']:+.2f}%\n\n"
    )

    for i, r in enumerate(top):

        msg += f"""
━━━━━━━━━━━━━━
#{i+1} {r['ticker']} {COMPANY_NAMES.get(r['ticker'], '')}

{r['signal']}  スコア: {r['score']}

現在値: {r['price']}
寄付からの騰落: {r['open_change']:+.2f}%
直近30分リターン: {r['short_return_30m']:+.2f}%
直近1時間リターン: {r['short_return_1h']:+.2f}%

30分足RSI: {r['rsi30']}
出来高倍率(本日平均比): {r['vol_surge']}

本日高値: {r['today_high']} / 本日安値: {r['today_low']}
━━━━━━━━━━━━━━
"""

    print(msg)
    send(msg)


if __name__ == "__main__":
    main()
