#!/usr/bin/env python3
"""stock-ai 日次 回顧・展望・検証 ログツール (review-log ブランチ専用・標準ライブラリのみ)。

main の売買コード/状態ファイルには一切書き込まない。main のチェックアウトを読み取り、
このブランチの log/ と review_log.csv だけを書く。

  snapshot  --repo MAIN --date D            : log/D.json を作る(事実・前日展望の答え合わせ・実行間隔)
  add-review --date D --review FILE         : LLM が書いた回顧/展望/検証(JSON)を検証して log/D.json に追記
  summary                                   : log/*.json から review_log.csv を作り直す
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "log")
SUMMARY = os.path.join(HERE, "review_log.csv")
SCHEMA = 1

HISTORIES = {
    "top1": "profit_top10_paper_history.csv",
    "daytrade": "daytrade_tp5000_sl8000_history.csv",
    "macd_top": "nikkei_macd_dip_paper_history_top.csv",
    "macd_mid": "nikkei_macd_dip_paper_history_mid.csv",
    "macd_bottom": "nikkei_macd_dip_paper_history_bottom.csv",
}
MACD_STATES = {
    "macd_top": "nikkei_macd_dip_paper_state_top.json",
    "macd_mid": "nikkei_macd_dip_paper_state_mid.json",
    "macd_bottom": "nikkei_macd_dip_paper_state_bottom.json",
}
# 9/7 の欠損(58分)を検出し、平常日(最大5分前後)では鳴らない閾値。昼休みも bot は5分おきに記録する。
GAP_ALERT_MIN = 12
SESSION_START, SESSION_END = "09:00", "15:30"
TRACKS = list(HISTORIES)
TRENDS = ("up", "down", "unknown")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_csv(path):
    try:
        # 履歴CSVの一部は BOM 付き(先頭列 entry_date が読めなくなる)ので utf-8-sig
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pos(track, p):
    keys = ("ticker", "company", "direction", "entry_date", "entry_time", "entry_price",
            "shares", "tp", "sl", "hold_days", "current_price", "unrealized_pnl")
    out = {k: p.get(k) for k in keys if p.get(k) is not None}
    out["track"] = track
    if track.startswith("macd") and "direction" not in out:
        out["direction"] = "LONG"
    return out


def collect_positions(repo):
    positions = []
    st = _read_json(os.path.join(repo, "profit_top10_paper_state.json")) or {}
    for p in st.get("positions") or []:
        positions.append(_pos("top1", p))
    if st.get("pending_entry"):
        positions.append(dict(_pos("top1", st["pending_entry"]), pending=True))
    dt = _read_json(os.path.join(repo, "daytrade_tp5000_sl8000_state.json")) or {}
    for p in dt.get("positions") or []:
        positions.append(_pos("daytrade", p))
    if dt.get("pending"):
        positions.append(dict(_pos("daytrade", dt["pending"]), pending=True))
    for track, fn in MACD_STATES.items():
        s = _read_json(os.path.join(repo, fn)) or {}
        if s.get("position"):
            positions.append(_pos(track, s["position"]))
    return positions


def collect_capital(repo):
    out = {}
    st = _read_json(os.path.join(repo, "profit_top10_paper_state.json"))
    if st:
        out["top1"] = {"capital": st.get("capital"), "peak": st.get("peak")}
    for track, fn in MACD_STATES.items():
        s = _read_json(os.path.join(repo, fn))
        if s:
            out[track] = {"capital": s.get("capital"), "peak": s.get("peak")}
    for v in out.values():
        c, pk = _num(v.get("capital")), _num(v.get("peak"))
        v["dd_pct"] = round((pk - c) / pk * 100, 3) if c and pk else None
    return out


def new_history_rows(repo, date, prev_counts):
    """前回スナップショット以降に追加された決済行。前回が無ければ exit_date==date の行。
    TOP1 の HOLD_LIMIT は exit_date が期限日(過去日)で記録されるので、行数差分で拾う。"""
    counts, rows = {}, []
    for track, fn in HISTORIES.items():
        data = _read_csv(os.path.join(repo, fn))
        if data is None:
            counts[track] = None
            continue
        counts[track] = len(data)
        prev = (prev_counts or {}).get(track)
        if isinstance(prev, int) and 0 <= prev <= len(data):
            fresh = data[prev:]
        else:
            fresh = [r for r in data if r.get("exit_date") == date]
        for r in fresh:
            rows.append({
                "track": track, "ticker": r.get("ticker"), "company": r.get("company"),
                "direction": r.get("direction") or ("LONG" if track.startswith("macd") else None),
                "entry_date": r.get("entry_date"), "entry_time": r.get("entry_time"),
                "entry_price": _num(r.get("entry_price")),
                "exit_date": r.get("exit_date"), "exit_time": r.get("exit_time"),
                "exit_price": _num(r.get("exit_price")), "shares": _num(r.get("shares")),
                "result": r.get("result"), "pnl": _num(r.get("pnl")),
            })
    return counts, rows


def _git_bot_commit_times(date):
    """GitHub 公開 API から D の main への bot 記録コミット時刻(JST)。失敗時 None。"""
    since = f"{date}T00:00:00+09:00"
    until = f"{date}T16:30:00+09:00"
    times = []
    for page in range(1, 4):
        url = ("https://api.github.com/repos/aifuu/stock-ai/commits?per_page=100"
               f"&since={since.replace('+', '%2B')}&until={until.replace('+', '%2B')}&page={page}")
        try:
            raw = subprocess.run(["curl", "-sS", "--max-time", "30", "-H", "User-Agent: review-log", url],
                                 capture_output=True, timeout=60, check=True).stdout
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        for c in data:
            msg = (c.get("commit") or {}).get("message", "")
            if msg.startswith("chore: persist"):
                ts = c["commit"]["committer"]["date"].replace("Z", "+00:00")
                times.append(datetime.fromisoformat(ts).astimezone(JST))
        if len(data) < 100:
            break
    return sorted(times)


def is_trading_day(date):
    """main の common.is_tse_trading_day と同じ判定。jpholiday が無ければ None(不明)。"""
    d = datetime.fromisoformat(date).date()
    if d.weekday() >= 5 or (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)):
        return False
    try:
        import jpholiday
    except ImportError:
        return None
    return not jpholiday.is_holiday(d)


def cadence_from_times(times, date, trading_day="auto"):
    if trading_day == "auto":
        trading_day = is_trading_day(date)
    if trading_day is False:
        return {"status": "holiday", "commits": len(times or [])}
    if times is None:
        return {"status": "unverified", "reason": "GitHub API 取得失敗"}
    if not times and trading_day is None:
        return {"status": "no_session", "commits": 0,
                "reason": "記録0件。祝日判定不可(jpholiday 無し)なので休場か全停止かを確認すること"}
    day = datetime.fromisoformat(date).replace(tzinfo=JST)
    s = day.replace(hour=int(SESSION_START[:2]), minute=int(SESSION_START[3:]))
    e = day.replace(hour=int(SESSION_END[:2]), minute=int(SESSION_END[3:]))
    pts = [s] + [t for t in times if s < t < e] + [e]
    gaps = []
    for a, b in zip(pts, pts[1:]):
        m = (b - a).total_seconds() / 60
        if m > GAP_ALERT_MIN:
            gaps.append({"from": a.strftime("%H:%M"), "to": b.strftime("%H:%M"), "minutes": round(m)})
    max_gap = max(((b - a).total_seconds() / 60 for a, b in zip(pts, pts[1:])), default=None)
    return {
        "status": "gap" if gaps else "ok",
        "commits": len(times),
        "first": times[0].strftime("%H:%M") if times else None,
        "last": times[-1].strftime("%H:%M") if times else None,
        "max_gap_min": round(max_gap) if max_gap is not None else None,
        "gaps_over_threshold": gaps,
        "threshold_min": GAP_ALERT_MIN,
    }


def _prev_log(date):
    files = sorted(f for f in glob.glob(os.path.join(LOG_DIR, "*.json"))
                   if os.path.basename(f)[:10] < date)
    return _read_json(files[-1]) if files else None


def grade(prev, decision, positions, closed, date):
    """前営業日の展望(prev.review.outlook)を今日の事実で答え合わせする。"""
    if not prev:
        return {"status": "no_previous_log"}
    outlook = (prev.get("review") or {}).get("outlook")
    if not outlook:
        return {"status": "no_previous_outlook", "previous_date": prev.get("date")}
    g = {"status": "graded", "previous_date": prev.get("date")}
    pred = outlook.get("next_trend")
    actual = decision.get("trend") if decision.get("date") == date else None
    g["trend"] = {"predicted": pred, "actual": actual,
                  "hit": None if pred == "unknown" or actual is None else pred == actual}
    open_keys = {(p["track"], p.get("ticker")) for p in positions}
    closed_keys = {(r["track"], r.get("ticker")) for r in closed}
    exp = []
    for x in outlook.get("expected_closes") or []:
        k = (x.get("track"), x.get("ticker"))
        exp.append({"track": k[0], "ticker": k[1], "closed": k in closed_keys,
                    "still_open": k in open_keys})
    g["expected_closes"] = exp
    exp_keys = {(x["track"], x["ticker"]) for x in exp}
    g["unexpected_closes"] = [{"track": t, "ticker": k} for t, k in sorted(closed_keys - exp_keys)]
    graded = [x for x in exp]
    g["closes_hit"] = sum(1 for x in graded if x["closed"])
    g["closes_expected"] = len(graded)
    return g


def build_snapshot(repo, date, cadence_times=None, fetch=True):
    prev = _prev_log(date)
    decision = _read_json(os.path.join(repo, "daily_decision.json")) or {}
    positions = collect_positions(repo)
    counts, closed = new_history_rows(repo, date, (prev or {}).get("history_counts"))
    times = cadence_times if cadence_times is not None else (_git_bot_commit_times(date) if fetch else None)
    pnl = {}
    for r in closed:
        if r["pnl"] is not None:
            pnl[r["track"]] = round(pnl.get(r["track"], 0.0) + r["pnl"], 2)
    keep = ("date", "trend", "trend_reason", "trend_data_as_of", "ma_short", "ma_long",
            "last_confirmed_close", "last_confirmed_change_pct", "policy_file", "policy_hash",
            "entry_allowed", "entry_block_reason", "intraday_crash_brake", "crash_brake_reason")
    return {
        "schema": SCHEMA,
        "date": date,
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "decision": {k: decision.get(k) for k in keep},
        "closed_today": closed,
        "pnl_by_track": pnl,
        "losses": [r for r in closed if (r["pnl"] or 0) < 0],
        "positions": positions,
        "capital": collect_capital(repo),
        "history_counts": counts,
        "cadence": cadence_from_times(times, date),
        "previous_outlook_grade": grade(prev, decision, positions, closed, date),
        "review": None,
    }


def validate_review(rv, snap):
    errs = []
    for k in ("kaiko", "tenbou", "kensho"):
        if not isinstance(rv.get(k), str) or not rv[k].strip():
            errs.append(f"{k} は空でない文字列が必要")
    pv = rv.get("price_verified")
    if pv not in ("verified", "unverified", "not_needed"):
        errs.append("price_verified は verified/unverified/not_needed")
    if pv == "unverified" and not str(rv.get("price_unverified_reason") or "").strip():
        errs.append("unverified の場合 price_unverified_reason が必要")
    if pv == "not_needed" and snap.get("losses"):
        errs.append("損失がある日は not_needed にできない")
    o = rv.get("outlook")
    if not isinstance(o, dict):
        errs.append("outlook(dict) が必要")
    else:
        if o.get("next_trend") not in TRENDS:
            errs.append("outlook.next_trend は up/down/unknown")
        ec = o.get("expected_closes")
        if not isinstance(ec, list):
            errs.append("outlook.expected_closes は list")
        else:
            for x in ec:
                if not isinstance(x, dict) or x.get("track") not in TRACKS or not x.get("ticker"):
                    errs.append(f"expected_closes の要素が不正: {x!r} (track は {TRACKS})")
    return errs


def cmd_snapshot(a):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{a.date}.json")
    snap = build_snapshot(a.repo, a.date, fetch=not a.no_fetch)
    if snap["cadence"].get("status") == "holiday":
        # 休場日はログを作らない(翌営業日の答え合わせが前営業日の展望を参照できるように)
        print(json.dumps({"date": a.date, "holiday": True, "written": False}, ensure_ascii=False))
        return 0
    old = _read_json(path)
    if old and old.get("review"):
        snap["review"] = old["review"]  # 再実行しても書いた回顧は消さない
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: snap[k] for k in ("date", "pnl_by_track", "losses", "cadence",
                                           "previous_outlook_grade")}, ensure_ascii=False, indent=1))
    return 0


def cmd_add_review(a):
    path = os.path.join(LOG_DIR, f"{a.date}.json")
    snap = _read_json(path)
    if not snap:
        print(f"ERROR: {path} が無い。先に snapshot", file=sys.stderr)
        return 2
    rv = _read_json(a.review)
    if not isinstance(rv, dict):
        print("ERROR: review JSON が読めない", file=sys.stderr)
        return 2
    errs = validate_review(rv, snap)
    if errs:
        print("ERROR:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    snap["review"] = rv
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    print("OK", path)
    return 0


COLS = ["date", "trend", "trend_reason", "predicted_trend", "trend_hit", "closes_expected",
        "closes_hit", "unexpected_closes", "pnl_total", "pnl_top1", "pnl_daytrade", "pnl_macd_top",
        "pnl_macd_mid", "pnl_macd_bottom", "loss_count", "open_positions", "top1_dd_pct",
        "cadence", "max_gap_min", "price_verified", "has_review"]


def summary_row(s):
    g = s.get("previous_outlook_grade") or {}
    t = g.get("trend") or {}
    p = s.get("pnl_by_track") or {}
    rv = s.get("review") or {}
    d = s.get("decision") or {}
    return {
        "date": s.get("date"),
        "trend": d.get("trend") if d.get("date") == s.get("date") else "",
        "trend_reason": d.get("trend_reason") if d.get("date") == s.get("date") else "",
        "predicted_trend": t.get("predicted", ""),
        "trend_hit": "" if t.get("hit") is None else int(bool(t.get("hit"))),
        "closes_expected": g.get("closes_expected", ""),
        "closes_hit": g.get("closes_hit", ""),
        "unexpected_closes": len(g.get("unexpected_closes") or []) if g.get("status") == "graded" else "",
        "pnl_total": round(sum(p.values()), 2),
        **{f"pnl_{k}": p.get(k, 0) for k in TRACKS},
        "loss_count": len(s.get("losses") or []),
        "open_positions": len(s.get("positions") or []),
        "top1_dd_pct": ((s.get("capital") or {}).get("top1") or {}).get("dd_pct", ""),
        "cadence": (s.get("cadence") or {}).get("status", ""),
        "max_gap_min": (s.get("cadence") or {}).get("max_gap_min", ""),
        "price_verified": rv.get("price_verified", ""),
        "has_review": int(bool(rv)),
    }


def cmd_summary(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(LOG_DIR, "*.json"))):
        s = _read_json(f)
        if s:
            rows.append(summary_row(s))
    with open(SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    hits = [r["trend_hit"] for r in rows if r["trend_hit"] != ""]
    print(f"{len(rows)} days; trend hit {sum(hits)}/{len(hits)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--repo", required=True)
    s.add_argument("--date", default=datetime.now(JST).strftime("%Y-%m-%d"))
    s.add_argument("--no-fetch", action="store_true")
    r = sub.add_parser("add-review")
    r.add_argument("--date", default=datetime.now(JST).strftime("%Y-%m-%d"))
    r.add_argument("--review", required=True)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    return {"snapshot": cmd_snapshot, "add-review": cmd_add_review, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
