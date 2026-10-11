#!/usr/bin/env python3
"""週次オプティマイザの「検証失敗内訳」診断スクリプト。

読み取り専用(read-only)。adversarial_strategy_validator.py は
import/実行すると実際にネットワークダウンロードや重い計算を行う
モジュールなので、このスクリプトは一切 import しない。
代わりに adversarial_strategy_validator.py のソースコードをテキストとして
正規表現で読み、ゲート定数(閾値)だけを抽出する(_extract_gate_constants)。

入力(すべて read-only、リポジトリ直下に既存のCSV/JSONを読むだけ):
  - adversarial_validation_results_{up,down}.csv
  - adversarial_oos_results_{up,down}.csv
  - adversarial_final_candidates_{up,down}.csv
  - multi_oos_folds_{up,down}.csv
  - multi_oos_fold_funnel_{up,down}.csv (参考情報として読む場合がある)
  - strategy_policy.json / strategy_policy_up.json / strategy_policy_down.json (承認済みポリシー)

出力:
  - validation_failure_breakdown.csv
  - validation_failure_summary.md (日本語)

このスクリプトは adversarial_strategy_validator.py / 各種 strategy_policy*.json /
policy_manual_overrides.json を一切書き換えない。読むだけ。
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VALIDATOR_SRC = REPO_ROOT / "adversarial_strategy_validator.py"

TRENDS = ["up", "down"]

OUT_CSV = REPO_ROOT / "validation_failure_breakdown.csv"
OUT_MD = REPO_ROOT / "validation_failure_summary.md"


# ============================================================
# 1. ゲート定数の抽出 (validatorをimportせず、ソースをテキストとして正規表現で読む)
# ============================================================

def _extract_gate_constants(src_text: str) -> dict:
    """adversarial_strategy_validator.pyのソーステキストから、validation_pass/
    oos_passの合否判定に使われる定数をすべて正規表現で抽出する。

    対応する2つの書き方:
      NAME = <float_or_int_literal>
      NAME = int(os.getenv("ENV_VAR", "<literal>"))
      NAME = float(os.getenv("ENV_VAR", "<literal>"))
      NAME = float(os.getenv("ENV_VAR") or "<literal>")  (念のため両方許容)
    """
    patterns = {
        "MIN_VALIDATION_TRADES": r'^MIN_VALIDATION_TRADES\s*=\s*int\(os\.getenv\("WF_MIN_VALIDATION_TRADES",\s*"(\d+)"\)\)',
        "MIN_PF_LOWER": r'^MIN_PF_LOWER\s*=\s*([0-9.]+)',
        "MIN_RETURN_LOWER": r'^MIN_RETURN_LOWER\s*=\s*([0-9.]+)',
        "MAX_VALIDATION_DD": r'^MAX_VALIDATION_DD\s*=\s*([0-9.]+)',
        "MIN_ANNUAL_SIGNALS": r'^MIN_ANNUAL_SIGNALS\s*=\s*int\(os\.getenv\("WF_MIN_ANNUAL_SIGNALS",\s*"(\d+)"\)\)',
        "MIN_OOS_TRADES": r'^MIN_OOS_TRADES\s*=\s*int\(os\.getenv\("WF_MIN_OOS_TRADES",\s*"(\d+)"\)\)',
        "MIN_OOS_PF": r'^MIN_OOS_PF\s*=\s*([0-9.]+)',
        "MIN_OOS_AVG_RETURN": r'^MIN_OOS_AVG_RETURN\s*=\s*([0-9.]+)',
        "MIN_OOS_TO_VALIDATION_PF": r'^MIN_OOS_TO_VALIDATION_PF\s*=\s*([0-9.]+)',
        "MAX_VALIDATION_PF_FOR_RATIO": r'^MAX_VALIDATION_PF_FOR_RATIO\s*=\s*float\(os\.getenv\("WF_MAX_VALIDATION_PF_FOR_RATIO",\s*"([0-9.]+)"\)\)',
        "MIN_MONTHLY_POSITIVE_RATIO": r'^MIN_MONTHLY_POSITIVE_RATIO\s*=\s*float\(os\.getenv\("WF_MIN_MONTHLY_POSITIVE_RATIO",\s*"([0-9.]+)"\)\)',
        "MAX_OOS_DD": r'^MAX_OOS_DD\s*=\s*float\(os\.getenv\("WF_MAX_OOS_DD",\s*"([0-9.]+)"\)\)',
    }
    out = {}
    for name, pat in patterns.items():
        m = re.search(pat, src_text, re.MULTILINE)
        if not m:
            raise RuntimeError(
                f"adversarial_strategy_validator.py から定数 {name} を抽出できませんでした"
                "(正規表現がソースの書き方と一致しません。validator側の表記が変わった可能性)。"
            )
        out[name] = float(m.group(1))
    return out


def load_gate_constants() -> dict:
    text = VALIDATOR_SRC.read_text(encoding="utf-8")
    return _extract_gate_constants(text)


# ============================================================
# 2. ゲート判定ロジック (adversarial_strategy_validator.py 580行目付近 /
#    594-596行目付近の validation_pass / oos_pass の論理式をそのまま再現する)
# ============================================================

def _f(row, key, default=float("nan")):
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def evaluate_validation_gates(row: dict, gates: dict) -> dict:
    """1行(strategy)に対し、validation_passを構成する各ゲートの合否をboolで返す。"""
    signals = _f(row, "validation_signals", 0.0)
    pf = _f(row, "validation_pf", 0.0)
    avg_return = _f(row, "validation_avg_return", 0.0)
    dd = _f(row, "validation_dd", 0.0)
    annual_signals = _f(row, "validation_annual_signals", 0.0)
    monthly_positive_ratio = _f(row, "validation_monthly_positive_ratio", 0.0)
    avg_lower = _f(row, "validation_avg_lower", float("nan"))

    return {
        "gate_signals": signals >= gates["MIN_VALIDATION_TRADES"],
        "gate_pf": pf >= gates["MIN_PF_LOWER"],
        "gate_avg_return": avg_return > gates["MIN_RETURN_LOWER"],
        "gate_dd": dd >= -gates["MAX_VALIDATION_DD"],
        "gate_annual_signals": annual_signals >= gates["MIN_ANNUAL_SIGNALS"],
        "gate_monthly_positive_ratio": monthly_positive_ratio >= gates["MIN_MONTHLY_POSITIVE_RATIO"] * 100,
        "gate_avg_lower": (avg_lower > 0) if avg_lower == avg_lower else False,  # NaN -> False (bootstrap未計算=不合格)
    }


def evaluate_oos_gates(row: dict, gates: dict) -> dict:
    """1行(strategy)に対し、oos_passを構成する各ゲートの合否をboolで返す。"""
    signals = _f(row, "oos_signals", 0.0)
    pf = _f(row, "oos_pf", 0.0)
    avg_return = _f(row, "oos_avg_return", 0.0)
    monthly_positive_ratio = _f(row, "oos_monthly_positive_ratio", 0.0)
    dd = _f(row, "oos_dd", 0.0)
    compound = _f(row, "oos_compound_return", 0.0)
    validation_pf = _f(row, "validation_pf", 0.0)

    pf_ratio_raw = row.get("oos_pf_ratio", "")
    if pf_ratio_raw in ("", None):
        pf_ratio = (pf / validation_pf) if validation_pf else float("nan")
    else:
        try:
            pf_ratio = float(pf_ratio_raw)
        except ValueError:
            pf_ratio = float("nan")

    ratio_ok = (pf_ratio == pf_ratio and pf_ratio >= gates["MIN_OOS_TO_VALIDATION_PF"]) or (
        validation_pf > gates["MAX_VALIDATION_PF_FOR_RATIO"]
    )

    return {
        "gate_signals": signals >= gates["MIN_OOS_TRADES"],
        "gate_pf": pf >= gates["MIN_OOS_PF"],
        "gate_avg_return": avg_return > gates["MIN_OOS_AVG_RETURN"],
        "gate_ratio": ratio_ok,
        "gate_monthly_positive_ratio": monthly_positive_ratio >= gates["MIN_MONTHLY_POSITIVE_RATIO"] * 100,
        "gate_dd": dd >= -gates["MAX_OOS_DD"],
        "gate_compound": compound > 0,
    }


GATE_LABELS = {
    "gate_signals": "トレード数不足",
    "gate_pf": "PF不足",
    "gate_avg_return": "平均リターン不足",
    "gate_dd": "最大DD超過",
    "gate_annual_signals": "年率シグナル数不足",
    "gate_monthly_positive_ratio": "月次プラス比率不足",
    "gate_avg_lower": "ブートストラップ下限<=0",
    "gate_ratio": "OOS/Validation PF比不足",
    "gate_compound": "OOS複利リターン<=0",
}


# ============================================================
# 3. CSV読み込みユーティリティ (空/ヘッダのみのファイルにも対応)
# ============================================================

def read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            rows = [r for r in reader if any((v or "").strip() for v in r.values())]
            return rows
    except Exception:
        return []


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ============================================================
# 4. breakdown本体
# ============================================================

PARAM_COLS = ["strategy", "up", "score", "nikkei", "tp", "sl", "hold"]


def build_breakdown_rows(gates: dict):
    out_rows = []
    raw_rows_by_key = {}
    for trend in TRENDS:
        for stage, fname_tpl, evaluator, gate_cols in (
            ("validation", "adversarial_validation_results_{}.csv", evaluate_validation_gates,
             ["gate_signals", "gate_pf", "gate_avg_return", "gate_dd", "gate_annual_signals",
              "gate_monthly_positive_ratio", "gate_avg_lower"]),
            ("oos", "adversarial_oos_results_{}.csv", evaluate_oos_gates,
             ["gate_signals", "gate_pf", "gate_avg_return", "gate_ratio",
              "gate_monthly_positive_ratio", "gate_dd", "gate_compound"]),
        ):
            path = REPO_ROOT / fname_tpl.format(trend)
            rows = read_csv_rows(path)
            raw_rows_by_key[(trend, stage)] = rows
            for row in rows:
                gate_results = evaluator(row, gates)
                passed = all(gate_results[g] for g in gate_cols)
                reasons = [GATE_LABELS[g] for g in gate_cols if not gate_results[g]]
                out = {"trend": trend, "stage": stage}
                for c in PARAM_COLS:
                    out[c] = row.get(c, "")
                out.update(gate_results)
                out["pass_computed"] = passed
                stored_pass_col = f"{stage}_pass"
                out["pass_stored"] = row.get(stored_pass_col, "")
                out["failure_reasons"] = ";".join(reasons) if reasons else ""
                out_rows.append(out)
    return out_rows, raw_rows_by_key


def write_breakdown_csv(rows: list, path: Path):
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("trend,stage,strategy,pass_computed,failure_reasons\n")
        return
    all_gate_cols = sorted({k for r in rows for k in r.keys() if k.startswith("gate_")})
    fieldnames = ["trend", "stage"] + PARAM_COLS + all_gate_cols + [
        "pass_computed", "pass_stored", "failure_reasons",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================
# 5. 集計・レポート生成
# ============================================================

def _signature(row: dict, gate_cols: list) -> tuple:
    return tuple(row.get(g) for g in gate_cols) + (row.get("pass_computed"),)


def _load_fold_trend(trend: str):
    path = REPO_ROOT / f"multi_oos_folds_{trend}.csv"
    rows = read_csv_rows(path)
    out = []
    for r in rows:
        try:
            out.append((r.get("fold"), r.get("oos_end"), int(float(r.get("final_pass_count", 0)))))
        except (TypeError, ValueError):
            out.append((r.get("fold"), r.get("oos_end"), r.get("final_pass_count")))
    return out


def _load_approved_policies():
    policies = []
    for fname, trend_hint in (
        ("strategy_policy.json", None),
        ("strategy_policy_up.json", "up"),
        ("strategy_policy_down.json", "down"),
    ):
        data = read_json(REPO_ROOT / fname)
        if data:
            data = dict(data)
            data["_file"] = fname
            data["_trend_hint"] = trend_hint
            policies.append(data)
    return policies


def _parse_strategy_params(name: str):
    """strategy_name (例: UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3) から
    up/score/nikkei/tp/sl/holdを抜き出す。マッチしない場合はNoneを返す。"""
    m = re.match(
        r"UP(?P<up>\d+)_SCORE(?P<score>\d+)_NIKKEI(?P<nikkei>ON|OFF)_TP(?P<tp>[0-9.]+)_SL(?P<sl>[0-9.]+)_H(?P<hold>\d+)",
        name,
    )
    if not m:
        return None
    d = m.groupdict()
    return {
        "up": int(d["up"]), "score": int(d["score"]), "nikkei": d["nikkei"] == "ON",
        "tp": float(d["tp"]), "sl": float(d["sl"]), "hold": int(d["hold"]),
    }


def _find_policy_in_oos(trend: str, policy_params: dict, oos_rows: list):
    """exact match (up/score/nikkei/tp/sl/hold全て一致) を探す。
    無ければ up/score/nikkeiのみ一致する行(tp/sl/hold違い=nearest relative)を探す。
    どちらも無ければNoneを返す。"""
    if policy_params is None or not oos_rows:
        return None, "none"
    for row in oos_rows:
        try:
            if (int(float(row.get("up", -1))) == policy_params["up"]
                    and int(float(row.get("score", -1))) == policy_params["score"]
                    and (str(row.get("nikkei", "")).strip().lower() in ("true", "1")) == policy_params["nikkei"]
                    and abs(float(row.get("tp", -1)) - policy_params["tp"]) < 1e-6
                    and abs(float(row.get("sl", -1)) - policy_params["sl"]) < 1e-6
                    and int(float(row.get("hold", -1))) == policy_params["hold"]):
                return row, "exact"
        except (TypeError, ValueError):
            continue
    for row in oos_rows:
        try:
            if (int(float(row.get("up", -1))) == policy_params["up"]
                    and int(float(row.get("score", -1))) == policy_params["score"]
                    and (str(row.get("nikkei", "")).strip().lower() in ("true", "1")) == policy_params["nikkei"]):
                return row, "nearest(up/score/nikkei一致,tp/sl/hold違い)"
        except (TypeError, ValueError):
            continue
    return None, "not_found"


def generate_report(gates: dict, breakdown_rows: list, raw_rows_by_key: dict) -> str:
    lines = []
    lines.append("# 検証失敗内訳サマリー (validation_failure_breakdown)")
    lines.append("")
    lines.append(
        "本レポートは read-only の診断であり、記載の『この gate を無視した場合に通過したはず"
        "の件数』は**あくまで参考情報**であって、いかなるゲート・ポリシー・validator・"
        "署名付きpolicyファイルを緩める提案・推奨では一切ない。実際のゲート定数・判定ロジックは"
        "adversarial_strategy_validator.py 側で厳格に維持されている。"
    )
    lines.append("")
    lines.append("## 抽出されたゲート定数(adversarial_strategy_validator.pyのソースから正規表現で抽出)")
    lines.append("")
    for k in sorted(gates):
        lines.append(f"- `{k}` = {gates[k]}")
    lines.append("")

    by_key = defaultdict(list)
    for r in breakdown_rows:
        by_key[(r["trend"], r["stage"])].append(r)

    stage_gate_cols = {
        "validation": ["gate_signals", "gate_pf", "gate_avg_return", "gate_dd", "gate_annual_signals",
                        "gate_monthly_positive_ratio", "gate_avg_lower"],
        "oos": ["gate_signals", "gate_pf", "gate_avg_return", "gate_ratio",
                "gate_monthly_positive_ratio", "gate_dd", "gate_compound"],
    }

    for trend in TRENDS:
        lines.append(f"## {trend.upper()}トレンド")
        lines.append("")
        for stage in ("validation", "oos"):
            rows = by_key.get((trend, stage), [])
            lines.append(f"### {stage.upper()}")
            if not rows:
                lines.append("")
                lines.append("- 対象行なし(ファイルが空、またはヘッダのみ)。")
                lines.append("")
                continue
            gate_cols = stage_gate_cols[stage]
            n = len(rows)
            n_pass = sum(1 for r in rows if r["pass_computed"])
            n_fail = n - n_pass
            lines.append("")
            lines.append(f"- 候補数 n = {n} / 合格 = {n_pass} / 不合格 = {n_fail}")
            lines.append("")
            lines.append("| gate | 不合格件数 | この gate のみ無視した場合に追加で通過したはずの件数(情報提供のみ) |")
            lines.append("|---|---|---|")
            for g in gate_cols:
                fail_n = sum(1 for r in rows if not r[g])
                would_pass_if_ignored = sum(
                    1 for r in rows
                    if not r[g] and all(r[g2] for g2 in gate_cols if g2 != g)
                )
                lines.append(f"| {GATE_LABELS[g]} ({g}) | {fail_n} | {would_pass_if_ignored} |")
            lines.append("")

            combo_counter = Counter(r["failure_reasons"] for r in rows if r["failure_reasons"])
            if combo_counter:
                lines.append("上位の失敗ゲート組み合わせ(多い順、上位5件):")
                lines.append("")
                for combo, cnt in combo_counter.most_common(5):
                    lines.append(f"- 「{combo}」: {cnt}件")
                lines.append("")

            sig_counter = Counter(_signature(r, gate_cols) for r in rows)
            n_distinct = len(sig_counter)
            lines.append(
                f"全{n}件の戦略のうち、ゲート判定結果(全gateの合否パターン)が完全に一致する"
                f"重複を除くと、識別可能な『結果パターン』の種類は {n_distinct} 種類"
                f"(多くの UP20/25/30 等の行が実質的に同一結果の重複であることを示す)。"
            )
            lines.append("")

            mismatches = [r for r in rows if str(r.get("pass_stored", "")).strip().lower() in ("true", "false")
                          and (str(r["pass_stored"]).strip().lower() == "true") != r["pass_computed"]]
            if mismatches:
                lines.append(
                    f"⚠️ 注意: {len(mismatches)}件で、本スクリプトが再計算した合否と"
                    f"元CSVに保存されていた{stage}_pass列が一致しませんでした"
                    "(ゲート定数の抽出漏れ、またはvalidator側ロジック変更の可能性。要確認)。"
                )
                lines.append("")

        lines.append("### fold毎の最終合格数の推移 (multi_oos_folds_{trend}.csv)".format(trend=trend))
        fold_trend = _load_fold_trend(trend)
        if fold_trend:
            seq = ", ".join(f"fold{f}={c}" for f, _end, c in fold_trend)
            lines.append("")
            lines.append(f"- {seq}")
        else:
            lines.append("")
            lines.append("- multi_oos_folds_{}.csv が見つからないか空です。".format(trend))
        lines.append("")

        lines.append("### 承認済みポリシーの最新OOS成績")
        lines.append("")
        oos_rows = raw_rows_by_key.get((trend, "oos"), [])
        policies = _load_approved_policies()
        matched_any = False
        for pol in policies:
            strategy_name = pol.get("strategy_name", "")
            params = _parse_strategy_params(strategy_name)
            if params is None:
                continue
            # strategy_policy_up.json は明示的にUPトレンド用、strategy_policy.jsonは
            # trend分離前の"all"相当(up_threshold等から見てBUY/UP寄りのパラメータだが
            # trend専用ではない)。down向けの承認済みポリシーファイルは現在存在しない
            # (strategy_policy_down.json が無いことをコード側で確認済み)。
            if trend == "down" and pol["_file"] != "strategy_policy_down.json":
                continue
            matched_any = True
            found_row, match_kind = _find_policy_in_oos(trend, params, oos_rows)
            lines.append(f"- `{pol['_file']}` (strategy_name=`{strategy_name}`):")
            if found_row is None:
                lines.append(
                    f"  - 最新fold({trend})の候補群(DEV top50由来)に、up/score/nikkeiが一致する"
                    "行が1件も存在しません。この承認済みパラメータ組み合わせは今回のfoldの"
                    "stability_objective上位50に入らなかったため、最新OOSでの直接の成績は"
                    "このファイル群からは導出できません(validatorの再実行が必要であり、本スクリプトは"
                    "それを行いません)。"
                )
            else:
                lines.append(f"  - 一致方法: {match_kind}")
                lines.append(
                    f"  - oos_signals={found_row.get('oos_signals')}, oos_pf={found_row.get('oos_pf')}, "
                    f"oos_avg_return={found_row.get('oos_avg_return')}, oos_dd={found_row.get('oos_dd')}, "
                    f"oos_monthly_positive_ratio={found_row.get('oos_monthly_positive_ratio')}, "
                    f"oos_pf_ratio={found_row.get('oos_pf_ratio')}, oos_pass={found_row.get('oos_pass')}"
                )
        if not matched_any:
            lines.append(f"- {trend}トレンド専用の承認済みポリシーファイルは存在しません。")
        lines.append("")

        if trend == "down":
            lines.append("### DOWN固有の参考指標(ゼロ取引月・アクティブ月のみの positive ratio)")
            lines.append("")
            lines.append(
                "- adversarial_validation_results_down.csv / adversarial_oos_results_down.csv には"
                "`validation_monthly_positive_ratio`(全暦月に対する比率、ゼロ取引月を0%として含む)"
                "と`validation_months`/`validation_positive_months`という**集計済みの件数**のみが"
                "保存されており、各月が「ゼロ取引だったのか」「取引はあったが負けただったのか」を"
                "区別する月別の明細(per-month trade count)はこのリポジトリのどのCSVにも"
                "保存されていません。"
            )
            lines.append(
                "- そのため『ゼロ取引月の件数』『アクティブ月(取引が1件以上あった月)のみに限定した"
                "positive ratio』は、現在保存されているファイルからは**導出不可能**です。"
            )
            lines.append(
                "- これを導出するには、adversarial_strategy_validator.py側で"
                "(validatorを変更せずにこのスクリプトから直接得るのではなく、将来の拡張として)"
                "月ごとのtrade件数(signals_per_month)を追加でCSV出力する必要があります"
                "(stats()関数内のmonthly Series、または trade_diagnostics の日付をdown trend別に"
                "月次集計したものを追加の出力列として export することが考えられます)。"
                "本タスクの制約により、validator自体は変更していません。"
            )
            lines.append("")

    return "\n".join(lines)


def main():
    gates = load_gate_constants()
    rows, raw_rows_by_key = build_breakdown_rows(gates)
    write_breakdown_csv(rows, OUT_CSV)
    report = generate_report(gates, rows, raw_rows_by_key)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
