"""Discord通知専用(ライフサイクル通知)。

売買ロジック・選定ロジックには一切触れない。既存の profit_top10_paper.discord_send()
による通知(エントリー/決済/毎サイクルの状況報告)はそのまま残し、本モジュールは
それとは別の第2チャンネル的な通知として、セッション開始・TOP1変更・エラーなどの
ライフサイクルイベントを即時通知し、それ以外の定期進捗は2時間に1回だけに絞る。

各ワーカー呼び出しは5分ごとに新しいPythonプロセスとして起動されるため、
「前回いつ送ったか」「前回のTOP1は何だったか」はメモリではなくディスクに
永続化する(paper_fast_entrypoint.pyのスキャンキャッシュと同じ考え方)。
"""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

TZ = ZoneInfo("Asia/Tokyo")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "").strip()
STATE_FILE = "discord_progress_state.json"
PROGRESS_INTERVAL_SECONDS = int(os.getenv("DISCORD_PROGRESS_INTERVAL_SECONDS", "7200"))


def now_jst():
    return datetime.now(TZ)


def _load_state():
    try:
        if not os.path.exists(STATE_FILE):
            return {}
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        print(f"⚠️ discord_progress state保存失敗: {exc}")


def send_discord(message, force=False):
    """force=Falseの通常進捗は2時間間隔(ディスク永続化)。送信失敗は握りつぶす
    (Discord通知は取引継続を止めるべきでないため)。"""
    if not WEBHOOK:
        return
    if not force:
        state = _load_state()
        last_ts = state.get("last_progress_ts")
        now_ts = now_jst().timestamp()
        if last_ts is not None:
            try:
                if now_ts - float(last_ts) < PROGRESS_INTERVAL_SECONDS:
                    return
            except (TypeError, ValueError):
                pass
        state["last_progress_ts"] = now_ts
        _save_state(state)
    try:
        requests.post(WEBHOOK, json={"content": message[:1950]}, timeout=15)
    except Exception as exc:
        print(f"⚠️ discord_progress送信失敗: {exc}")


def top1_changed(symbol, direction):
    """TOP1が前回呼び出し時から変わっていればTrueを返し、状態を更新する。
    空symbolはFalse固定(呼び出し側は候補が0件のときは呼ばない想定だが念のため)。"""
    symbol = str(symbol or "")
    direction = str(direction or "")
    if not symbol:
        return False
    state = _load_state()
    if state.get("last_top1_ticker") == symbol and state.get("last_top1_direction") == direction:
        return False
    state["last_top1_ticker"] = symbol
    state["last_top1_direction"] = direction
    _save_state(state)
    return True


def notify_start(session):
    send_discord(f"🚀 セッション開始｜{session}\n{now_jst():%Y-%m-%d %H:%M} JST", force=True)


def notify_selection_start():
    send_discord("🔍 銘柄選定開始", force=True)


def notify_top10(top10):
    lines = [
        f"{i}. {c.get('company', c.get('ticker', ''))}（{c.get('ticker', '')}）"
        f"{'買い' if str(c.get('direction', 'BUY')).upper() == 'BUY' else '空売り'}｜"
        f"score={float(c.get('score', 0) or 0):.1f}"
        for i, c in enumerate(top10[:10], 1)
    ]
    send_discord("📋 TOP10決定\n" + ("\n".join(lines) if lines else "候補なし"), force=True)


def notify_top1(symbol, direction, score):
    direction_jp = "買い" if str(direction).upper() == "BUY" else "空売り"
    send_discord(f"🥇 TOP1決定\n{symbol}｜{direction_jp}｜score={float(score):.1f}", force=True)


def notify_trade(symbol, direction, price):
    direction_jp = "買い" if str(direction).upper() == "BUY" else "空売り"
    send_discord(f"💹 取引実行\n{symbol}｜{direction_jp}｜価格 {float(price):,.1f}円", force=True)


def notify_exit(symbol, price, profit):
    emoji = "🟢" if float(profit) >= 0 else "🔴"
    send_discord(f"{emoji} 決済\n{symbol}｜価格 {float(price):,.1f}円｜損益 {float(profit):+,.0f}円", force=True)


def notify_progress(status):
    send_discord(status, force=False)


def notify_error(stage, error):
    send_discord(f"🚨 エラー発生\nstage={stage}\n{error}", force=True)
