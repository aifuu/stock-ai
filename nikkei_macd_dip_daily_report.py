"""
日経MACD逆張り(単一指標) 日次保有・約定履歴レポート ― 実験的セカンドトラックの定期報告
====================================================================================

★このスクリプトは nikkei_macd_dip_paper.py(top/mid/bottomの3パターン並列トラック)の
状態ファイル(nikkei_macd_dip_paper_state_{top,mid,bottom}.json)と履歴ファイル
(nikkei_macd_dip_paper_history_{top,mid,bottom}.csv)を読むだけの完全に独立した
レポート専用スクリプト。売買判断は一切行わず、ファイルへの書き込みも一切行わない。
nikkei_macd_dip_paper.py 本体は一切呼び出さない・編集しない。

背景:
  nikkei_macd_dip_paper.py が送るDiscord通知は、エントリー/決済が実際に発生した
  回のみ(何も起きなければ通知なし)。そのため保有中のポジションが日々どうなって
  いるかが定期的には見えない。本スクリプトは毎朝(JST 9:05頃)と場中終了後
  (JST 15:40頃)の2回、変化の有無に関係なく必ず現在の保有状況(と場中終了後は
  本日の約定履歴も)をDiscordに送信する。

送信内容:
  - 朝(実行時刻のJST時が12時より前): 3パターンそれぞれの現在の保有状況
    (銘柄・株数・エントリー価格・エントリー日・TP/SL価格・保有上限日、
    ノーポジションならその旨)。
  - 場中終了後(実行時刻のJST時が12時以降): 上記の保有状況に加え、本日の
    約定履歴(各トラックの history csv から exit_date が本日の行、および
    state.json 上で entry_date が本日の保有中ポジションを抽出して一覧表示)。

保有上限日について:
  nikkei_macd_dip_paper.py の check_exit_intraday() は、TP/SLどちらも
  未到達の場合、エントリー日からの経過営業日数(pd.bdate_range、土日のみ
  除外・祝日は考慮しない)がHOLD_DAYS(3)に達した日の場中判定で強制決済
  (HOLD_LIMIT)する。本スクリプトはこのCI(pandas未インストール・requests
  のみ)でも動くよう、pd.bdate_rangeと完全に同一の結果になる純Python版の
  営業日カウント(add_business_days、土日のみ除外・祝日は考慮しない)を
  再実装し(pandas/pd.bdate_rangeとの一致は実装時に検証済み。
  nikkei_macd_dip_paper.py 本体はyfinance等の重い依存を持つため
  importしない方針を踏襲し、あえてimportしない)、保有中ポジションの
  entry_dateから強制決済予定日(保有上限日)を算出して表示する。
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

TZ = ZoneInfo("Asia/Tokyo")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
BASE_DIR = Path(__file__).resolve().parent

BASE_LABEL = "🔬 日経MACD逆張(別トラック)"

# nikkei_macd_dip_paper.py の HOLD_DAYS(最大保有営業日数)と同一の値。
# 変更する場合は両方のファイルを合わせて直すこと。
HOLD_DAYS = 3

WEEKDAY_LABELS_JA = ["月", "火", "水", "木", "金", "土", "日"]

# nikkei_macd_dip_paper.py の CONFIGS と同じファイル名・表示ラベルを、
# 依存を増やさないためここで読み取り専用の定数として再定義する
# (nikkei_macd_dip_paper.py 自体はimportしない = 本番の場中執行ロジックや
# yfinance/AIモデル読込などの重い依存を一切引き込まない)。
CONFIGS = [
    {
        "name": "top",
        "label": "top TP0.22/SL0.1257",
        "state_file": BASE_DIR / "nikkei_macd_dip_paper_state_top.json",
        "history_file": BASE_DIR / "nikkei_macd_dip_paper_history_top.csv",
    },
    {
        "name": "mid",
        "label": "mid TP1.25/SL0.7143",
        "state_file": BASE_DIR / "nikkei_macd_dip_paper_state_mid.json",
        "history_file": BASE_DIR / "nikkei_macd_dip_paper_history_mid.csv",
    },
    {
        "name": "bottom",
        "label": "bottom TP3.5/SL2",
        "state_file": BASE_DIR / "nikkei_macd_dip_paper_state_bottom.json",
        "history_file": BASE_DIR / "nikkei_macd_dip_paper_history_bottom.csv",
    },
]


def send(msg):
    text = str(msg)
    print(text)
    if not WEBHOOK_URL:
        print("⚠️ DISCORD_WEBHOOK 未設定のため送信スキップ")
        return False
    text = text if len(text) <= 1900 else text[:1897] + "..."
    try:
        r = requests.post(WEBHOOK_URL, json={"content": text}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"⚠️ Discord通知失敗: {exc}")
        return False


def load_state(cfg):
    """状態ファイルを読むだけ(存在しない/壊れている場合は None を返す)。"""
    path = cfg["state_file"]
    if not path.exists():
        return None
    try:
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"⚠️ {cfg['label']}: state読込失敗: {exc}")
        return None


def load_history_rows(cfg):
    """履歴csvを読むだけ(存在しない場合は空リスト)。"""
    path = cfg["history_file"]
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"⚠️ {cfg['label']}: history読込失敗: {exc}")
        return []


def fmt_money(x):
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_price(x):
    try:
        return f"{float(x):,.1f}"
    except (TypeError, ValueError):
        return "N/A"


def add_business_days(start_date, n):
    """start_date(date)から土日のみを除いてn営業日目(start_date自身が平日
    ならそれを1営業日目として含む)にあたる日付を返す。
    pd.bdate_range(pd.Timestamp(start_date), periods=n)[-1].date() と
    完全に同一の結果になる(祝日は考慮しない、純Python実装)。"""
    d = start_date
    counted = 0
    while True:
        if d.weekday() < 5:
            counted += 1
            if counted == n:
                return d
        d += timedelta(days=1)


def hold_limit_date(entry_date_str):
    """nikkei_macd_dip_paper.py の check_exit_intraday() と完全に同一の
    計算式(エントリー翌日からの営業日カウント、土日のみ除外・祝日は
    考慮しない)で、保有上限日(その日の場中判定で経過営業日数が
    HOLD_DAYSに達しHOLD_LIMIT決済となる日)を求める。"""
    entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    return add_business_days(entry_date + timedelta(days=1), HOLD_DAYS)


def hold_limit_line(entry_date_str, today):
    """保有上限日の表示行を組み立てる。today(date)が保有上限日に達している
    場合は本日中に強制決済される旨を強調する。"""
    try:
        limit_date = hold_limit_date(entry_date_str)
    except (TypeError, ValueError):
        return "保有上限: 算出不可(entry_date不正)"

    weekday_ja = WEEKDAY_LABELS_JA[limit_date.weekday()]
    limit_str = f"{limit_date.year}年{limit_date.month}月{limit_date.day}日({weekday_ja})"

    if limit_date <= today:
        return f"⚠️ 保有上限: {limit_str} ── 本日保有上限、場中判定で強制決済されます"
    return f"保有上限: {limit_str}まで(最大{HOLD_DAYS}営業日、この日の場中判定で強制決済)"


def position_block(cfg, state, today):
    """1トラック分の現在の保有状況ブロックを組み立てる。"""
    header = f"[{cfg['label']}]"
    if state is None:
        return f"{header}\n⚠️ 状態ファイル読込不可"

    position = state.get("position")
    capital = state.get("capital")
    if not position:
        return (
            f"{header}\n"
            f"ノーポジション\n"
            f"💰 仮想資産(別枠): {fmt_money(capital)}円"
        )

    ticker = position.get("ticker", "?")
    company = position.get("company", "?")
    shares = position.get("shares", "?")
    entry_price = position.get("entry_price")
    entry_date = position.get("entry_date", "?")
    tp = position.get("tp")
    sl = position.get("sl")
    invested = position.get("invested_amount")

    return (
        f"{header}\n"
        f"保有中: {ticker} {company}\n"
        f"エントリー {fmt_price(entry_price)}円｜{shares}株｜投資額 {fmt_money(invested)}円｜{entry_date}\n"
        f"TP {fmt_price(tp)}｜SL {fmt_price(sl)}\n"
        f"{hold_limit_line(entry_date, today)}\n"
        f"💰 仮想資産(別枠): {fmt_money(capital)}円"
    )


def today_trades_block(cfg, state, today_str):
    """本日の約定(新規エントリー・決済)を1トラック分組み立てる。
    エントリーはhistory csvにはまだ記録されない(決済時に初めて1行として
    追記される)ため、当日中に決済まで完了していないオープン中の新規建玉は
    state.json の position から検出し、当日中に決済まで完了した
    (エントリーも決済も同日)ものはhistory csvから検出する。"""
    lines = []

    position = (state or {}).get("position")
    if position and position.get("entry_date") == today_str:
        ticker = position.get("ticker", "?")
        company = position.get("company", "?")
        shares = position.get("shares", "?")
        entry_price = position.get("entry_price")
        lines.append(
            f"🆕 新規エントリー(保有中): {ticker} {company} {shares}株 @{fmt_price(entry_price)}円"
        )

    for row in load_history_rows(cfg):
        entry_date = row.get("entry_date")
        exit_date = row.get("exit_date")
        if entry_date != today_str and exit_date != today_str:
            continue
        ticker = row.get("ticker", "?")
        company = row.get("company", "?")
        shares = row.get("shares", "?")
        entry_price = row.get("entry_price")
        if entry_date == today_str and exit_date == today_str:
            prefix = "🆕→"
        elif entry_date == today_str:
            prefix = "🆕"
        else:
            prefix = ""
        if exit_date == today_str:
            exit_price = row.get("exit_price")
            result = row.get("result", "?")
            pnl = row.get("pnl")
            result_label = {"TP": "利確(TP)", "SL": "損切(SL)", "HOLD_LIMIT": "期限到達"}.get(
                result, result
            )
            try:
                pnl_str = f"{float(pnl):+,.0f}円"
            except (TypeError, ValueError):
                pnl_str = "N/A"
            lines.append(
                f"{prefix}✅ 決済({result_label}): {ticker} {company} {shares}株 "
                f"@{fmt_price(entry_price)}円→{fmt_price(exit_price)}円 損益 {pnl_str}"
            )
        else:
            lines.append(
                f"{prefix} 新規エントリー: {ticker} {company} {shares}株 @{fmt_price(entry_price)}円"
            )

    header = f"[{cfg['label']}]"
    if not lines:
        return f"{header}\n本日の約定なし"
    return header + "\n" + "\n".join(lines)


def build_message(now):
    today_str = now.strftime("%Y-%m-%d")
    is_morning = now.hour < 12

    states = {cfg["name"]: load_state(cfg) for cfg in CONFIGS}

    title = "朝の保有状況" if is_morning else "本日の取引まとめ"
    lines = [
        f"{BASE_LABEL}｜{title}(3パターン並列)",
        f"📅 {today_str} {now.strftime('%H:%M')} JST",
        "",
        "■ 現在の保有状況",
    ]
    today = now.date()
    for cfg in CONFIGS:
        lines.append("")
        lines.append(position_block(cfg, states[cfg["name"]], today))

    if not is_morning:
        lines.append("")
        lines.append("■ 本日の約定履歴")
        for cfg in CONFIGS:
            lines.append("")
            lines.append(today_trades_block(cfg, states[cfg["name"]], today_str))

    lines.append("")
    lines.append("⚠️ 実注文なし・読み取り専用レポート(本番システムとは完全に独立した実験トラック)")
    return "\n".join(lines)


def main():
    now = datetime.now(TZ)
    msg = build_message(now)
    send(msg)


if __name__ == "__main__":
    main()
