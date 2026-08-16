"""
common.py

日足AI (ai_stock_scan.py) とは独立した、30分監視モード専用の共通処理。
日足AIのロジックには一切手を入れず、こちらは別ファイルとして完結させる。
"""

import time
from datetime import date, datetime, time as dtime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import os
import jpholiday


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

    r = requests.post(WEBHOOK_URL, json={"content": msg}, timeout=30)

    print("Discord status =", r.status_code)

    if r.status_code == 204:
        print("✅ Discord送信成功")
    else:
        print("❌ Discord送信失敗")
        print(r.text)


# =====================
# 銘柄(日足AIと同じリスト)
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


# =====================
# ダウンロード失敗時のリトライ
# =====================
def safe_download(ticker, retries=3, wait_sec=3, **kwargs):

    for attempt in range(1, retries + 1):

        try:
            df = yf.download(ticker, **kwargs)

            if df is not None and not df.empty:
                return df

            print(f"{ticker} 空データ (試行{attempt}/{retries})")

        except Exception as e:
            print(f"{ticker} 取得失敗 (試行{attempt}/{retries}): {e}")

        if attempt < retries:
            time.sleep(wait_sec)

    print(f"❌ {ticker} リトライ{retries}回失敗")

    return None


# =====================
# RSI
# =====================
def calc_rsi(close, period=14):

    close = close.squeeze()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)

    return rsi


# =====================
# 東証営業日判定
#
# ・土日 → 休み
# ・祝日(jpholidayで判定) → 休み
# ・年末年始(12/31〜1/3) → 休み
#   (大晦日・1/2・1/3は祝日ではないが東証は休場のため個別に除外)
# =====================
def is_tse_trading_day(d: date) -> bool:

    if d.weekday() >= 5:
        return False

    if jpholiday.is_holiday(d):
        return False

    if (d.month == 12 and d.day == 31):
        return False

    if (d.month == 1 and d.day in (1, 2, 3)):
        return False

    return True


# =====================
# 現在の市場フェーズ判定
#
# before_open : 〜8:59  (寄り前。リアルタイム株価がまだ薄い/前場開始前)
# morning     : 9:00〜11:29 (前場)
# lunch       : 11:30〜12:29 (昼休み。板が動かない)
# afternoon   : 12:30〜15:30 (後場)
# after_close : 15:30より後
# =====================
def get_market_phase(now_jst: datetime) -> str:

    t = now_jst.time()

    if t < dtime(9, 0):
        return "before_open"

    if dtime(9, 0) <= t < dtime(11, 30):
        return "morning"

    if dtime(11, 30) <= t < dtime(12, 30):
        return "lunch"

    if dtime(12, 30) <= t <= dtime(15, 30):
        return "afternoon"

    return "after_close"
