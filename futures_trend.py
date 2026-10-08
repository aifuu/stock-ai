"""日経平均先物トレンド判定モジュール(案3: 日次の戦略切り替え判定専用)。

週次のレジーム検証(現物ベース4分類)はそのまま維持し、
このモジュールは「今日どちらのTOP1戦略(上昇用/下落用)を使うか」を
決めるためだけの軽量な日次シグナルを提供する。

判定ルール(たたき台。パラメータは後で調整前提):
- MA5 < MA20 -> 下落トレンド
- MA5 >= MA20 -> 上昇トレンド
- 前日比 <= -CRASH_THRESHOLDの場合は強制的に下落トレンド扱い(急落の取りこぼし防止)
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

# 日経225先物のyfinanceティッカー。取得できない場合は現物(^N225)にフォールバックする。
FUTURES_TICKER = os.getenv("FUTURES_TICKER", "NIY=F")
FALLBACK_TICKER = "^N225"

MA_SHORT = int(os.getenv("FUTURES_TREND_MA_SHORT", "5"))
MA_LONG = int(os.getenv("FUTURES_TREND_MA_LONG", "20"))
CRASH_THRESHOLD = float(os.getenv("FUTURES_TREND_CRASH_PCT", "0.015"))  # 1.5%

UP = "up"
DOWN = "down"


def _download(ticker, period="3mo"):
    try:
        d = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index)
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        return d.sort_index()
    except Exception:
        return None


def detect_futures_trend():
    """[非推奨・本番未使用] 当日の未確定足を含む旧判定。本番はdaily_decision経由の
    confirmed_trend()を使う(2026-10-08の場中トレンド反転事象を参照)。

先物(取得失敗時は現物)の直近終値からトレンドを判定する。

    戻り値: dict
      trend: "up" | "down"
      reason: 判定理由の短い文字列
      source: 実際に使ったティッカー("futures" | "cash_proxy")
      ma_short / ma_long / last_close / daily_change_pct: 診断用の数値(取得失敗時はNone)
    """
    d = _download(FUTURES_TICKER)
    source = "futures"
    if d is None or len(d) < MA_LONG + 1:
        d = _download(FALLBACK_TICKER)
        source = "cash_proxy"

    if d is None or len(d) < MA_LONG + 1:
        # データが全く取れない場合は安全側(下落トレンド扱い)に倒す。
        return {
            "trend": DOWN,
            "reason": "データ取得失敗のため安全側(下落トレンド扱い)",
            "source": "unavailable",
            "ma_short": None,
            "ma_long": None,
            "last_close": None,
            "daily_change_pct": None,
        }

    close = d["Close"].astype(float)
    ma_short = float(close.rolling(MA_SHORT).mean().iloc[-1])
    ma_long = float(close.rolling(MA_LONG).mean().iloc[-1])
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    daily_change_pct = (last_close / prev_close - 1.0) if prev_close else 0.0

    if daily_change_pct <= -CRASH_THRESHOLD:
        return {
            "trend": DOWN,
            "reason": f"急落補助条件: 前日比{daily_change_pct * 100:.2f}% <= -{CRASH_THRESHOLD * 100:.1f}%",
            "source": source,
            "ma_short": ma_short,
            "ma_long": ma_long,
            "last_close": last_close,
            "daily_change_pct": daily_change_pct,
        }

    trend = UP if ma_short >= ma_long else DOWN
    return {
        "trend": trend,
        "reason": f"MA{MA_SHORT}={ma_short:.1f} {'>=' if trend == UP else '<'} MA{MA_LONG}={ma_long:.1f}",
        "source": source,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "last_close": last_close,
        "daily_change_pct": daily_change_pct,
    }


def log_daily_trend(result, log_file="futures_trend_history.csv"):
    """週次の専用戦略採用判断に備え、日次判定結果を裏で蓄積するだけのロガー。

    本番の戦略選定には使わない(サンプル数が貯まるまでの記録専用)。
    """
    row = {
        "date": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d"),
        "trend": result.get("trend"),
        "reason": result.get("reason"),
        "source": result.get("source"),
        "ma_short": result.get("ma_short"),
        "ma_long": result.get("ma_long"),
        "last_close": result.get("last_close"),
        "daily_change_pct": result.get("daily_change_pct"),
    }
    df = pd.DataFrame([row])
    if os.path.exists(log_file):
        try:
            existing = pd.read_csv(log_file)
            if not (existing["date"] == row["date"]).any():
                df = pd.concat([existing, df], ignore_index=True)
            else:
                df = existing
        except Exception:
            pass
    df.to_csv(log_file, index=False, encoding="utf-8-sig")
    return row


def historical_trend_series(start, end, ticker=None):
    """検証(walk_forward)用に、指定期間の日次トレンド(up/down)系列を返す。

    本番の日次判定(detect_futures_trend)と同じMA5/MA20+急落補助条件を
    過去データに適用する。戻り値はDataFrame(index=date, columns=[trend])。
    先物データが取得できない期間は現物(^N225)で代替する。
    """
    t = ticker or FUTURES_TICKER
    pad_start = (pd.Timestamp(start) - pd.Timedelta(days=int(MA_LONG * 3))).strftime("%Y-%m-%d")
    pad_end = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    d = _download_range(t, pad_start, pad_end)
    source = "futures"
    if d is None or len(d) < MA_LONG + 2:
        d = _download_range(FALLBACK_TICKER, pad_start, pad_end)
        source = "cash_proxy"
    if d is None or len(d) < MA_LONG + 2:
        return pd.DataFrame(columns=["trend", "source"])

    frame = classify_close_series(d["Close"])
    out = pd.DataFrame({"trend": frame["trend"].values, "source": source}, index=d.index)
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    return out


def classify_close_series(close):
    """確定終値系列から日次トレンドを判定する唯一の実装。

    検証(historical_trend_series)と本番(confirmed_trend→daily_decision)の
    両方がこの関数を使うので、判定式が食い違うことはない。
    """
    close = pd.Series(close).astype(float)
    ma_short = close.rolling(MA_SHORT).mean()
    ma_long = close.rolling(MA_LONG).mean()
    daily_change_pct = close.pct_change()
    trend = np.where(ma_short >= ma_long, UP, DOWN)
    trend = np.where(daily_change_pct <= -CRASH_THRESHOLD, DOWN, trend)
    return pd.DataFrame(
        {"trend": trend, "ma_short": ma_short, "ma_long": ma_long, "daily_change_pct": daily_change_pct},
        index=close.index,
    )


def _confirmed_only(d, today):
    """当日(JST)以降の足を落とし、前日までの確定足だけにする。"""
    if d is None or d.empty:
        return d
    return d[d.index.normalize() < pd.Timestamp(today)]


def confirmed_trend(today, downloader=None):
    """tradingday=todayに使うトレンドを、前日までの確定終値だけで判定する。

    戻り値はdetect_futures_trend()と同じキー+trend_data_as_of(使った最後の確定日)。
    週次検証の対応付け(候補日dにtrend[d]、翌営業日に建てる)と一致する。
    """
    dl = downloader or _download
    d = _confirmed_only(dl(FUTURES_TICKER), today)
    source = "futures"
    if d is None or len(d) < MA_LONG + 1:
        d = _confirmed_only(dl(FALLBACK_TICKER), today)
        source = "cash_proxy"
    if d is None or len(d) < MA_LONG + 1:
        return {
            "trend": DOWN, "reason": "確定データ取得失敗のため安全側(下落・新規停止)",
            "source": "unavailable", "ma_short": None, "ma_long": None,
            "last_close": None, "daily_change_pct": None, "trend_data_as_of": None,
        }
    f = classify_close_series(d["Close"])
    last = f.iloc[-1]
    trend = str(last["trend"])
    chg = float(last["daily_change_pct"]) if pd.notna(last["daily_change_pct"]) else 0.0
    ms, ml = float(last["ma_short"]), float(last["ma_long"])
    if chg <= -CRASH_THRESHOLD:
        reason = f"急落補助条件: 前日確定比{chg * 100:.2f}% <= -{CRASH_THRESHOLD * 100:.1f}%"
    else:
        reason = f"MA{MA_SHORT}={ms:.1f} {'>=' if trend == UP else '<'} MA{MA_LONG}={ml:.1f}"
    return {
        "trend": trend, "reason": reason, "source": source,
        "ma_short": ms, "ma_long": ml,
        "last_close": float(d["Close"].astype(float).iloc[-1]),
        "daily_change_pct": chg,
        "trend_data_as_of": d.index[-1].strftime("%Y-%m-%d"),
    }


def intraday_price(today, source="futures"):
    """急落ブレーキ用の場中価格(当日の最新値)。当日足が無ければNone。"""
    ticker = FALLBACK_TICKER if source == "cash_proxy" else FUTURES_TICKER
    d = _download(ticker, period="5d")
    if d is None or d.empty:
        return None
    todays = d[d.index.normalize() == pd.Timestamp(today)]
    if todays.empty:
        return None
    return float(todays["Close"].astype(float).iloc[-1])


def _download_range(ticker, start, end):
    try:
        d = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index)
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        return d.sort_index()
    except Exception:
        return None


if __name__ == "__main__":
    res = detect_futures_trend()
    print(res)
    log_daily_trend(res)
