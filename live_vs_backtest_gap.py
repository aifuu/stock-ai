#!/usr/bin/env python3
"""ライブ・ペーパートレード vs adversarial_strategy_validator OOSバックテストの
ギャップ分析(read-only診断)。

重要な設計上の注意
------------------
adversarial_strategy_validator.py はモジュールレベルで実データのダウンロードや
重い計算を実行するため、このスクリプトは一切それを import しない
(確認済み: validator冒頭で候補CSVの存在チェック・yfinanceダウンロード・
候補生成がモジュールトップレベルで実行される。import = 即実行)。

このスクリプトは adversarial_strategy_validator.py の evaluate_trade() /
select_for_phase() のロジックをソースコードを読んで理解した上で、
**同じ規約を自前で再実装**している(下記 BACKTEST CONVENTION 参照)。

BACKTEST CONVENTION (adversarial_strategy_validator.py より書き起こし)
-----------------------------------------------------------------------
- エントリー: 候補日(date)の「price」列(walk_forward_all_candidates.csv生成時の
  その日の終値相当)でエントリーしたとみなす。本スクリプトではライブ約定日
  (entry_date)の日足Closeをこれに充てる(= 候補ファイルのprice列の実体は
  当日終値ベースであることをwalk_forward.py/daily_directional_top1.pyの
  生成ロジックから確認済み)。
- TP/SL: atr_ratio(ATR相当の%)× tp_multiplier / sl_multiplier で、
  entry×(1±atr_ratio/100×multiplier) として計算する。
  validatorは atr_ratio を候補ファイルの take_profit列から
  ((take_profit/price)-1)/3*100 という固定の式(TP倍率3.0固定を仮定)で
  逆算しているが、本スクリプトはライブ履歴行自身が持つ実際のポリシーの
  atr_tp_multiplier/atr_sl_multiplier(policy_file列、またはpolicy_file列が
  無い古い行は現行strategy_policy.json/strategy_policy_up.jsonで代替、
  という仮定を明示する)を使って同じ構造の式から atr_ratio を逆算する。
  これはvalidatorのTP倍率3.0固定という簡略化を、各トレード固有の実際の
  TP倍率に置き換えた一般化であり、ロジックの構造(entry×(1±atr_ratio/100×mult))
  自体はvalidatorと同一。
- 判定順序: 保有日数(hold_days、policyのhold_daysを使用)分の「エントリー日より
  後」の日足だけを見る(同日エントリーの値動きは一切参照しない=
  本番の「寄り後チャート確認してからエントリー」のような同日中の動きは
  バックテスト規約には存在しない)。各日について
    low<=stop and high>=take  -> SL優先(保守的、同日両方成立時はSL扱い)
    high>=take                -> TP
    low<=stop                 -> SL
  のいずれにも当たらず hold_days 分を使い切った場合は、最後の日の Close で
  決済(TIMEOUT_LOSS/HOLDに相当)。
- スリッページ: 片道0.1%(slippage=0.001)をリターンから引く。
  手数料(FEE_PCT = INTRADAY_FEE_RATE*2*100)はvalidator内では候補の優先順位
  (EV)計算にのみ使われ、実現リターン(evaluate_tradeの戻り値)には含まれない
  ことをソースで確認済み。つまり**validatorのバックテストP&Lには往復手数料が
  明示的に差し引かれていない**(片道0.1%のスリッページのみ)。

対して、ライブのdaytrade_tp5000_sl8000_paper.pyは全く異なる決済規約を使っている
(daytrade_tp5000_sl8000_paper.pyのモジュールdocstringより):
  - TP/SLは「往復手数料を差し引いた後にネットで+5,000円/-8,000円」になるよう
    価格から逆算した固定金額ルールであり、ATR倍数の%ルールではない。
  - 5分足で判定(日足ではない)。
  - 15:20 JSTで強制決済。
したがって daytrade track は「エントリー選定ルール(score/up_prob/policy閾値)」
はvalidatorと同じ系譜だが、「決済ルール」自体がvalidatorが検証した対象とは
構造的に異なる。これは本スクリプトが計算する「バックテスト規約での再現結果」と
ライブ結果の差の、最大かつ最も本質的な説明候補である(詳細はMarkdownレポート
を参照)。

入力(read-only):
  - profit_top10_paper_history.csv (TOP1トラック)
  - daytrade_tp5000_sl8000_history.csv (daytradeトラック)
  - strategy_policy.json / strategy_policy_up.json / strategy_policy_down.json

出力:
  - live_vs_backtest_gap.csv
  - live_vs_backtest_gap.md (日本語)

価格データ取得は外部ネットワーク(yfinance)に依存する。このサンドボックスには
ネットワークが無いため、取得関数は `fetch_daily_bars` という差し替え可能な
フックを経由する。ユニットテストではこれをモックしたフィクスチャ(合成した
日足OHLC)に差し替える。GitHub Actionsのworkflow_dispatch実行では実際の
yfinanceを使う。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

LIVE_FILES = {
    "top1": REPO_ROOT / "profit_top10_paper_history.csv",
    "daytrade": REPO_ROOT / "daytrade_tp5000_sl8000_history.csv",
}

SINCE_DATE = os.getenv("GAP_SINCE_DATE", "2026-09-01")
SLIPPAGE = 0.001  # validatorのevaluate_tradeと同じ片道0.1%
HOLD_LIMIT_BUG_FIX_DATE = "2026-10-06"  # profit_top10_paper.py HOLD_LIMIT rollbackバグ修正日

OUT_CSV = REPO_ROOT / "live_vs_backtest_gap.csv"
OUT_MD = REPO_ROOT / "live_vs_backtest_gap.md"


# ============================================================
# 価格データ取得フック (差し替え可能、既定はyfinance)
# ============================================================

def _default_fetch_daily_bars(ticker: str, start: str, end: str):
    """yfinanceから日足OHLCを取得する。戻り値は
    [{"date": "YYYY-MM-DD", "open":..,"high":..,"low":..,"close":..}, ...] の
    dateの昇順リスト。ネットワークが無い環境(このサンドボックス)では
    呼ばれない想定(ユニットテストは必ずfetch_daily_barsを差し替える)。
    """
    import yfinance as yf  # 遅延import: ネットワーク不要なテスト経路では一切読み込まない

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass
    out = []
    for idx, row in df.iterrows():
        out.append({
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx)[:10],
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
        })
    return out


fetch_daily_bars = _default_fetch_daily_bars


# ============================================================
# ポリシー読み込み
# ============================================================

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_policy_for_row(row: dict, track: str) -> dict:
    """ライブ履歴行から、バックテスト再現に使うポリシー(ATR倍率・hold_days)を
    決定する。daytrade行はpolicy_file列を持つのでそれを最優先で使う。
    TOP1(profit_top10_paper_history.csv)行はpolicy_file列を持たない古い
    スキーマなので、現行strategy_policy.jsonを仮定として使う(gap.mdに
    明示的に『仮定』と記載する)。
    """
    policy_file = (row.get("policy_file") or "").strip()
    assumption = None
    if not policy_file:
        policy_file = "strategy_policy.json"
        assumption = (
            "この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)"
            f"のため、現行の{policy_file}を代替として仮定した(当時実際に使われていた"
            "ポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。"
        )
    data = _load_json(REPO_ROOT / policy_file)
    if data is None:
        data = {"atr_tp_multiplier": 2.0, "atr_sl_multiplier": 1.0, "hold_days": 3}
        assumption = (assumption or "") + f" {policy_file} が読めないためデフォルト値を使用した。"
    return {
        "policy_file": policy_file,
        "atr_tp_multiplier": float(data.get("atr_tp_multiplier", 2.0)),
        "atr_sl_multiplier": float(data.get("atr_sl_multiplier", 1.0)),
        "hold_days": int(data.get("hold_days", 3) or 3),
        "assumption": assumption,
    }


# ============================================================
# バックテスト規約の再現 (adversarial_strategy_validator.evaluate_trade()相当)
# ============================================================

def backtest_convention_exit(entry_date: str, entry_price: float, direction: str,
                              atr_ratio_pct: float, tp_mult: float, sl_mult: float,
                              hold_days: int, bars: list):
    """validator.evaluate_trade()と同じロジックで、entry_date**より後**の
    日足だけを見てTP/SL/タイムアウトを判定する。barsは
    [{"date","open","high","low","close"}, ...] の昇順リスト。
    戻り値: (reason, return_pct_before_slippage, exit_date, exit_price, days_used)
    """
    future = [b for b in bars if b["date"] > entry_date]
    future = future[:max(1, hold_days)]
    if not future:
        return None

    if direction == "BUY":
        take = entry_price * (1 + atr_ratio_pct / 100 * tp_mult)
        stop = entry_price * (1 - atr_ratio_pct / 100 * sl_mult)
    else:
        take = entry_price * (1 - atr_ratio_pct / 100 * tp_mult)
        stop = entry_price * (1 + atr_ratio_pct / 100 * sl_mult)

    for day_no, b in enumerate(future, 1):
        high, low = b["high"], b["low"]
        if direction == "BUY":
            if low <= stop and high >= take:
                return ("SL_BOTH_SL_PRIORITY", (stop / entry_price - 1) * 100, b["date"], stop, day_no)
            if high >= take:
                return ("TP", (take / entry_price - 1) * 100, b["date"], take, day_no)
            if low <= stop:
                return ("SL", (stop / entry_price - 1) * 100, b["date"], stop, day_no)
        else:
            if high >= stop and low <= take:
                return ("SL_BOTH_SL_PRIORITY", (entry_price / stop - 1) * 100, b["date"], stop, day_no)
            if low <= take:
                return ("TP", (entry_price / take - 1) * 100, b["date"], take, day_no)
            if high >= stop:
                return ("SL", (entry_price / stop - 1) * 100, b["date"], stop, day_no)

    last = future[-1]
    close = last["close"]
    ret = ((close / entry_price - 1) * 100) if direction == "BUY" else ((entry_price / close - 1) * 100)
    reason = "TIMEOUT_LOSS" if ret < 0 else "HOLD"
    return (reason, ret, last["date"], close, len(future))


def derive_atr_ratio_pct(entry_price: float, tp_price: float, tp_mult: float) -> float:
    """ライブ行のtp価格と、そのトレードに使われたATR-TP倍率から、
    validatorと同じ構造(entry*(1+atr_ratio/100*mult)=tp)でatr_ratio(%)を逆算する。"""
    if entry_price <= 0 or tp_mult == 0:
        return 0.0
    return abs((tp_price / entry_price - 1) / tp_mult * 100.0)


# ============================================================
# CSV入出力
# ============================================================

def read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [r for r in reader if any((v or "").strip() for v in r.values())]


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


FORCED_CLOSE_BUDGET_BUG = "FORCED_CLOSE_BUDGET_BUG"


def process_track(track: str, since: str):
    path = LIVE_FILES[track]
    rows = read_csv_rows(path)
    out = []
    for row in rows:
        entry_date = (row.get("entry_date") or "").strip()
        exit_date = (row.get("exit_date") or "").strip()
        if not entry_date or entry_date < since:
            continue
        if not exit_date:
            continue  # 未決済(保有中)の行は対象外
        ticker = row.get("ticker", "")
        direction = (row.get("direction") or "BUY").strip().upper()
        entry_price = _to_float(row.get("entry_price"))
        live_exit_price = _to_float(row.get("exit_price"))
        live_result = (row.get("result") or "").strip()
        live_return_pct = _to_float(row.get("return_pct"))
        live_pnl = _to_float(row.get("pnl"))
        tp_price = _to_float(row.get("tp"))

        policy = resolve_policy_for_row(row, track)
        atr_ratio_pct = derive_atr_ratio_pct(entry_price, tp_price, policy["atr_tp_multiplier"])

        start = (datetime.strptime(entry_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (datetime.strptime(exit_date, "%Y-%m-%d") + timedelta(days=max(policy["hold_days"], 1) + 10)).strftime("%Y-%m-%d")
        bars = fetch_daily_bars(ticker, start, end)

        bt = None
        if bars and entry_price > 0:
            bt = backtest_convention_exit(
                entry_date, entry_price, direction, atr_ratio_pct,
                policy["atr_tp_multiplier"], policy["atr_sl_multiplier"],
                policy["hold_days"], bars,
            )

        causes = []
        if live_result == FORCED_CLOSE_BUDGET_BUG:
            causes.append("FORCED_CLOSE_BUDGET_BUG(資金不足による強制決済、タグ付きデータ)")
        if entry_date < HOLD_LIMIT_BUG_FIX_DATE and track == "top1":
            causes.append(f"pre-{HOLD_LIMIT_BUG_FIX_DATE} HOLD_LIMITロールバックバグの影響を受け得る期間のトレード")
        if policy["assumption"]:
            causes.append("policy_mismatch_or_unknown: " + policy["assumption"])
        if track == "daytrade":
            causes.append(
                "exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を"
                "5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の"
                "決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での"
                "参考値であり、ライブの実際の決済ルールそのものではない)"
            )

        bt_reason = bt[0] if bt else None
        bt_return_pct_raw = bt[1] if bt else None
        bt_return_pct = (bt_return_pct_raw - SLIPPAGE * 100) if bt_return_pct_raw is not None else None
        bt_exit_date = bt[2] if bt else None
        bt_exit_price = bt[3] if bt else None

        gap_pct = (live_return_pct - bt_return_pct) if bt_return_pct is not None else None

        if bt is None:
            causes.append("no_price_data: 日足データが取得できず再現不可(ネットワーク環境依存)")
        elif bt_reason and live_result and bt_reason.split("_")[0] != live_result.split("_")[0]:
            causes.append(
                f"exit_reason_diff: ライブ={live_result} / backtest規約={bt_reason}"
                "(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)"
            )

        out.append({
            "track": track,
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": direction,
            "policy_file": policy["policy_file"],
            "live_entry_price": entry_price,
            "live_exit_price": live_exit_price,
            "live_result": live_result,
            "live_return_pct": live_return_pct,
            "live_pnl": live_pnl,
            "backtest_atr_ratio_pct": round(atr_ratio_pct, 6),
            "backtest_exit_reason": bt_reason,
            "backtest_exit_date": bt_exit_date,
            "backtest_exit_price": bt_exit_price,
            "backtest_return_pct": round(bt_return_pct, 6) if bt_return_pct is not None else None,
            "gap_pct": round(gap_pct, 6) if gap_pct is not None else None,
            "candidate_causes": ";".join(causes),
        })
    return out


def write_csv(rows: list, path: Path):
    fieldnames = [
        "track", "ticker", "entry_date", "exit_date", "direction", "policy_file",
        "live_entry_price", "live_exit_price", "live_result", "live_return_pct", "live_pnl",
        "backtest_atr_ratio_pct", "backtest_exit_reason", "backtest_exit_date", "backtest_exit_price",
        "backtest_return_pct", "gap_pct", "candidate_causes",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def generate_report(rows: list) -> str:
    lines = []
    lines.append("# ライブ・ペーパートレード vs バックテスト ギャップ分析")
    lines.append("")
    n = len(rows)
    lines.append(
        f"⚠️ **サンプルサイズの注意**: 対象トレード数は合計 {n} 件(TOP1トラック + daytradeトラック、"
        f"{SINCE_DATE}以降の決済済みトレードのみ)。これは統計的に意味のある結論を出すには"
        "極めて少ないサンプルであり、以下の『原因の寄与度』はあくまで目視・定性的な傾向であって、"
        "厳密な統計的分解ではない。"
    )
    lines.append("")

    if not rows:
        lines.append("対象トレードがありません(履歴ファイルが空、または指定期間に決済済み行が無い)。")
        return "\n".join(lines)

    by_track = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)

    for track, trows in by_track.items():
        label = "TOP1 (profit_top10_paper_history.csv)" if track == "top1" else "daytrade (daytrade_tp5000_sl8000_history.csv)"
        lines.append(f"## {label} (n={len(trows)})")
        lines.append("")
        live_total = sum(r["live_return_pct"] for r in trows)
        bt_vals = [r["backtest_return_pct"] for r in trows if r["backtest_return_pct"] is not None]
        bt_total = sum(bt_vals) if bt_vals else None
        lines.append(f"- ライブ return_pct 合計: {live_total:.3f}pp (平均 {live_total/len(trows):.3f}pp/trade)")
        if bt_total is not None:
            lines.append(
                f"- バックテスト規約での再現 return_pct 合計: {bt_total:.3f}pp "
                f"(平均 {bt_total/len(bt_vals):.3f}pp/trade, n={len(bt_vals)}/{len(trows)}件で算出可能)"
            )
        else:
            lines.append("- バックテスト規約での再現値: 全件で価格データ取得不可(ネットワーク無し環境での実行)")
        lines.append("")
        for r in trows:
            lines.append(
                f"- {r['entry_date']} {r['ticker']} {r['direction']}: "
                f"live={r['live_result']}({r['live_return_pct']:.3f}pp) / "
                f"backtest規約={r['backtest_exit_reason']}({r['backtest_return_pct']}pp) / "
                f"gap={r['gap_pct']}pp / 原因候補=[{r['candidate_causes']}]"
            )
        lines.append("")

    lines.append("## 原因の順位付け(件数ベース、情報提供のみ)")
    lines.append("")
    cause_counter = {}
    for r in rows:
        for c in (r["candidate_causes"] or "").split(";"):
            c = c.strip()
            if not c:
                continue
            key = c.split(":")[0].split("(")[0].strip()
            cause_counter[key] = cause_counter.get(key, 0) + 1
    ranked = sorted(cause_counter.items(), key=lambda kv: -kv[1])
    if ranked:
        lines.append("| 原因候補 | 該当トレード数 | 全体に対する割合 |")
        lines.append("|---|---|---|")
        for cause, cnt in ranked:
            lines.append(f"| {cause} | {cnt} | {cnt/n*100:.0f}% |")
    else:
        lines.append("明確な原因タグに該当するトレードはありませんでした。")
    lines.append("")

    lines.append("## 選定の正当性(selection validity)について")
    lines.append("")
    lines.append(
        "ライブで選ばれた銘柄が、その取引日においてvalidatorの候補生成ルール"
        "(select_for_phase: up_prob/down_prob/flat_prob/score閾値+日経レジームゲート)"
        "のもとでも候補になっていたかを検証するには、その日のAI予測全銘柄分の"
        "up_prob/down_prob/flat_prob/scoreスナップショット(walk_forward_all_candidates.csv"
        "相当のその日の全候補、または少なくとも当日のprediction_history.csv全銘柄分)が"
        "必要である。しかしdaytrade_tp5000_sl8000_history.csvのtop10_rank列・"
        "top10_source列は『ライブがどのランクで選んだか』のみを記録しており、"
        "その日の全候補集合(落選した銘柄を含む全体)のスナップショットはこのリポジトリの"
        "どのファイルにも永続化されていない。したがって選定の正当性は本データからは"
        "検証不可能であり、検証するには当日全銘柄分の候補スナップショットを追加で"
        "保存するようワークフローを変更する必要がある(本タスクでは変更していない)。"
    )
    lines.append("")

    lines.append("## 手数料・スリッページの扱いの違い")
    lines.append("")
    lines.append(
        "- validator (adversarial_strategy_validator.evaluate_trade): リターンから片道0.1%"
        "(slippage=0.001)のみ減算。往復手数料(FEE_PCT)は候補優先順位(EV)計算にのみ使われ、"
        "実現P&Lには反映されない。"
    )
    lines.append(
        "- ライブ profit_top10_paper.py (TOP1): pnl = gross - (entry_price+exit_price)*shares*FEE_RATE "
        "で往復手数料を明示的に差し引く(FEE_RATE既定0.00055=0.055%/回、往復で約0.11%)。"
    )
    lines.append(
        "- ライブ daytrade_tp5000_sl8000_paper.py: TP/SL自体を『往復手数料差し引き後にネットで"
        "+5000円/-8000円』になるように価格から逆算しており、手数料は決済価格そのものに"
        "組み込まれている。"
    )
    lines.append(
        "- 結論: validatorのバックテストはライブより手数料に関して楽観的(片道0.1%のみ)であり、"
        "ライブは毎トレードで往復手数料(約0.11%)を追加で負担している。これは方向として"
        "『ライブがバックテストより悪く見える』ことの一部説明になるが、1トレードあたり"
        "0.11%程度であり、観測されているギャップの規模(1トレードあたり数%〜)全体を"
        "説明するには小さすぎる。"
    )
    lines.append("")

    lines.append("## HOLD_LIMITバグ (pre-2026-10-06) について")
    lines.append("")
    lines.append(
        "profit_top10_paper.py は2026-10-06のコミット(c4782fb)以前、yfinanceのTSE 5分足に"
        "15:25のバーが存在しないため、保有上限(hold_days)に達したポジションの"
        "HOLD_LIMIT決済が実質的に発火せず、ポジションが無期限に保有され続けるバグがあった"
        "(修正: 前営業日の日足を遡るロールバック分岐でHOLD_LIMIT決済するよう追加)。"
        f"profit_top10_paper_history.csv の{SINCE_DATE}以降の行はいずれもこの修正日より前の"
        "エントリーであり、このバグの影響を受けた可能性がある(影響の正確な定量化には"
        "当時のstateファイルの保有履歴が必要で、本データからは『影響を受け得る期間だった』"
        "ことまでしか確認できない)。"
    )
    lines.append("")

    lines.append("## FORCED_CLOSE_BUDGET_BUG タグについて")
    lines.append("")
    tagged = [r for r in rows if "FORCED_CLOSE_BUDGET_BUG" in (r["candidate_causes"] or "")]
    lines.append(
        f"- {len(tagged)}件がFORCED_CLOSE_BUDGET_BUGタグ付きだった"
        "(資金不足等により通常の決済ルールではなく強制決済されたことを示す履歴上のタグ)。"
        "これらの行はvalidatorが想定する正常な決済経路を経ていないため、"
        "バックテストとの比較対象として本質的に公平ではない。"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    rows = []
    for track in ("top1", "daytrade"):
        rows.extend(process_track(track, SINCE_DATE))
    write_csv(rows, OUT_CSV)
    report = generate_report(rows)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
