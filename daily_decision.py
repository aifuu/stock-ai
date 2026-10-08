"""その日の売買判断(daily_decision.json)を1日1回だけ固定するモジュール(C案)。

背景(2026-10-08の事象):
  通常TOP10トラックは5分おきのtickごとに先物トレンドを再判定しており、
  判定に当日の未確定足が混ざっていた。場中に前日比-1.5%を割ったtickで
  急落補助条件により「下落」へ切り替わり、存在しないstrategy_policy_down.json
  の代わりに未検証の汎用strategy_policy.jsonへフォールバックして売買した。
  一方デイトレ別枠はfutures_trend_history.csvに朝一記録された「上昇」を
  読み続けていたため、2トラックが別のpolicyで動いていた。

このモジュールの約束:
  1. 判断は1日1回だけ作り、daily_decision.jsonに固定する。全トラックはこれだけを読む。
     既に当日の判断があれば絶対に作り直さない(ensure_decision)。
  2. trendは「前日までの確定終値」だけで判定する。当日の未確定足は使わない。
     これは週次検証(historical_trend_series)が候補日dにtrend[d](d終値まで)を
     対応させ、翌営業日に建てる構造と一致する。
  3. 場中の急落(前日確定終値比 <= -CRASH_THRESHOLD)はpolicyを変えず、
     intraday_crash_brake=Trueにして全トラックの新規エントリーを止める(当日中は解除しない)。
  4. その日のtrendに対応する承認済みpolicyが無い場合(現状のDOWN日)は
     entry_allowed=Falseで実売買なし。代わりにシャドー記録だけ残す。

判断ファイルの例:
{
  "date": "2026-10-08",
  "decision_time": "2026-10-08T08:30:12+09:00",
  "trend": "up",
  "trend_reason": "MA5=69715.0 >= MA20=66300.5",
  "trend_source": "futures",
  "trend_data_as_of": "2026-10-07",
  "policy_file": "strategy_policy_up.json",
  "policy_hash": "xxxxxxxxxxxx",
  "entry_allowed": true,
  "entry_block_reason": null,
  "shadow_policy_file": null,
  "intraday_crash_brake": false,
  ...
}
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import futures_trend

TZ = ZoneInfo("Asia/Tokyo")
DECISION_FILE = os.getenv("DAILY_DECISION_FILE", "daily_decision.json")
SHADOW_FILE = os.getenv("DOWN_DAY_SHADOW_FILE", "down_day_shadow_trades.csv")

POLICY_FILE = "strategy_policy.json"
POLICY_FILE_UP = "strategy_policy_up.json"
POLICY_FILE_DOWN = "strategy_policy_down.json"
SCHEMA_VERSION = 1

SHADOW_COLUMNS = [
    "date", "time", "ticker", "company", "direction", "price", "tp", "sl",
    "score", "up_probability", "down_probability", "profit_ev_pct", "top10_rank",
    "trend", "trend_data_as_of", "shadow_policy_file", "shadow_policy_hash",
    "entry_block_reason", "track",
]


def _now(now=None):
    if now is None:
        return datetime.now(TZ)
    return now if now.tzinfo else now.replace(tzinfo=TZ)


def policy_hash(path):
    """policyファイルの先頭12桁sha256(daytradeの既存policy_hashと同じ形式)。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _policy_for_trend(trend):
    return POLICY_FILE_DOWN if trend == futures_trend.DOWN else POLICY_FILE_UP


def compute_decision(now=None):
    """前日までの確定終値だけからその日の判断を組み立てる(ファイルには書かない)。"""
    now = _now(now)
    today = now.date()
    t = futures_trend.confirmed_trend(today)
    trend = t["trend"]
    candidate = _policy_for_trend(trend)

    decision = {
        "schema_version": SCHEMA_VERSION,
        "date": today.isoformat(),
        "decision_time": now.isoformat(timespec="seconds"),
        "trend": trend,
        "trend_reason": t.get("reason"),
        "trend_source": t.get("source"),
        "trend_data_as_of": t.get("trend_data_as_of"),
        "ma_short": t.get("ma_short"),
        "ma_long": t.get("ma_long"),
        "last_confirmed_close": t.get("last_close"),
        "last_confirmed_change_pct": t.get("daily_change_pct"),
        "policy_file": None,
        "policy_hash": None,
        "entry_allowed": False,
        "entry_block_reason": None,
        "shadow_policy_file": None,
        "shadow_policy_hash": None,
        "intraday_crash_brake": False,
        "crash_brake_time": None,
        "crash_brake_reason": None,
    }

    if t.get("source") == "unavailable":
        decision["entry_block_reason"] = "trend_data_unavailable"
    elif os.path.exists(candidate):
        decision["policy_file"] = candidate
        decision["policy_hash"] = policy_hash(candidate)
        decision["entry_allowed"] = True
    else:
        decision["entry_block_reason"] = f"no_approved_policy_for_{trend}"

    # 実売買しない日は、従来フォールバックしていた汎用policyで「もし買っていたら」を記録する。
    if not decision["entry_allowed"] and os.path.exists(POLICY_FILE):
        decision["shadow_policy_file"] = POLICY_FILE
        decision["shadow_policy_hash"] = policy_hash(POLICY_FILE)
    return decision


def _atomic_write(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _log_trend_history(decision):
    """週次の専用戦略検討用に、確定判定を従来のfutures_trend_history.csvへも1日1行記録。"""
    try:
        futures_trend.log_daily_trend({
            "trend": decision["trend"],
            "reason": decision.get("trend_reason"),
            "source": decision.get("trend_source"),
            "ma_short": decision.get("ma_short"),
            "ma_long": decision.get("ma_long"),
            "last_close": decision.get("last_confirmed_close"),
            "daily_change_pct": decision.get("last_confirmed_change_pct"),
        })
    except Exception as e:
        print(f"⚠️ futures_trend_history記録失敗(判断には影響なし): {e}")


def ensure_decision(now=None, path=None):
    """当日の判断が既にあればそれを返し、無ければ1回だけ作って固定する。

    08:30のpremarketで作るのが正規経路。ジョブ遅延などで未作成のまま場中に
    呼ばれた場合も、確定終値だけを使うので08:30に作った場合と同じ判定になる
    (decision_timeに実際の作成時刻が残るので監査で区別できる)。
    """
    path = path or DECISION_FILE
    now = _now(now)
    existing = _read(path)
    if existing and existing.get("date") == now.date().isoformat():
        return existing
    decision = compute_decision(now)
    _atomic_write(path, decision)
    _log_trend_history(decision)
    print(
        f"🗓 daily_decision作成: date={decision['date']} trend={decision['trend']} "
        f"data_as_of={decision['trend_data_as_of']} policy={decision['policy_file']} "
        f"entry_allowed={decision['entry_allowed']} block={decision['entry_block_reason']}"
    )
    return decision


def entry_status(decision):
    """(新規エントリー可否, 理由)。policyが朝の判断後に差し替わっていたら止める。"""
    if not decision.get("entry_allowed"):
        return False, decision.get("entry_block_reason") or "entry_not_allowed"
    if decision.get("intraday_crash_brake"):
        return False, "intraday_crash_brake"
    pf = decision.get("policy_file")
    if not pf or not os.path.exists(pf):
        return False, "policy_file_missing"
    if policy_hash(pf) != decision.get("policy_hash"):
        return False, "policy_changed_since_decision"
    return True, None


def update_crash_brake(now=None, path=None, price_fetcher=None):
    """場中の急落ブレーキを評価し、発動したら当日中は維持する(解除しない)。

    基準は判断ファイルのlast_confirmed_close(=trend_data_as_ofの確定終値)。
    取得失敗時はブレーキを変更しない(既存の状態を維持)。
    """
    path = path or DECISION_FILE
    now = _now(now)
    decision = ensure_decision(now, path)
    if decision.get("intraday_crash_brake"):
        return decision
    base = decision.get("last_confirmed_close")
    if not base:
        return decision
    fetch = price_fetcher or futures_trend.intraday_price
    try:
        price = fetch(now.date(), decision.get("trend_source"))
    except Exception as e:
        print(f"⚠️ 急落ブレーキ用の場中価格取得失敗(状態維持): {e}")
        return decision
    if price is None:
        return decision
    change = float(price) / float(base) - 1.0
    if change <= -futures_trend.CRASH_THRESHOLD:
        decision["intraday_crash_brake"] = True
        decision["crash_brake_time"] = now.isoformat(timespec="seconds")
        decision["crash_brake_reason"] = (
            f"場中 {price:.1f} / 前日確定 {float(base):.1f} = {change * 100:+.2f}% "
            f"<= -{futures_trend.CRASH_THRESHOLD * 100:.1f}%"
        )
        _atomic_write(path, decision)
        print(f"🛑 急落ブレーキ発動: {decision['crash_brake_reason']} → 全トラック新規停止(決済は継続)")
    return decision


def record_shadow(decision, candidate, now=None, track="profit_top10", path=None):
    """実売買しない日の「もし買っていたら」を記録する(同一銘柄は1日1回まで)。"""
    path = path or SHADOW_FILE
    now = _now(now)
    today = now.date().isoformat()
    ticker = str(candidate.get("ticker", "")).strip()
    if not ticker:
        return False
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("date") == today and r.get("ticker") == ticker and r.get("track") == track:
                        return False
        except Exception as e:
            print(f"⚠️ シャドー記録読込失敗(追記は継続): {e}")
    row = {
        "date": today, "time": now.strftime("%H:%M"), "ticker": ticker,
        "company": candidate.get("company", ""), "direction": str(candidate.get("direction", "BUY")).upper(),
        "price": candidate.get("price"), "tp": candidate.get("tp"), "sl": candidate.get("sl"),
        "score": candidate.get("score"), "up_probability": candidate.get("up_probability"),
        "down_probability": candidate.get("down_probability"), "profit_ev_pct": candidate.get("profit_ev_pct"),
        "top10_rank": candidate.get("top10_rank"), "trend": decision.get("trend"),
        "trend_data_as_of": decision.get("trend_data_as_of"),
        "shadow_policy_file": decision.get("shadow_policy_file"),
        "shadow_policy_hash": decision.get("shadow_policy_hash"),
        "entry_block_reason": decision.get("entry_block_reason"), "track": track,
    }
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHADOW_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"👻 シャドー記録: {ticker} score={candidate.get('score')} ({decision.get('entry_block_reason')})")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensure", action="store_true", help="当日の判断が無ければ1回だけ作成(既にあれば何もしない)")
    args = ap.parse_args(argv)
    d = ensure_decision()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
