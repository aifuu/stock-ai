"""all_candidates_paper.py (独立6ヶ月データ収集トラック) の緊急データ修復ツール。

事故で紛れ込んだ特定のtrade_id(複数可、最大5件)を、state.json(open
positions)と月次生データ(all-candidates-paper-dataタグの各csv.gz)から
取り除く。それ以外のポジション・トレード行・last_completed_run_dateなど
のフィールドには一切触れない。

all_candidates_paper.py(以下「本体」)が持つRelease I/O・state検証・
state昇格・月次アセット読み書き・集計関数をそのまま再利用する
(import all_candidates_repair as acr; acr側でロジックを複製しない)。
本体(all_candidates_paper.py)の動作は一切変更しない。

安全策:
  - dry_run(既定True)ではネットワーク書き込み(Releaseアップロード)を
    一切行わない。何が変わるかの完全な差分だけを表示する。
  - dry_run=Falseで実際に変更する場合、変更対象になるアセットのバイト
    単位のバックアップを「他の変更を行う前に」同じRelease内へ
    `<元のアセット名>.pre-repair-<UTC yyyymmddHHMMSS>`という名前で
    アップロードし、再ダウンロードしてsha256を照合したうえで初めて
    本体の変更に着手する。
  - 与えられたtrade_idのうちどれか1つでもstate/月次データのどこにも
    見つからなければ、何もアップロードせずに例外で失敗する(全件一致が
    前提。部分適用はしない)。
  - 1回の呼び出しで削除できるtrade_idは最大5件まで。
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import all_candidates_paper as acp

MAX_TRADE_IDS = 5

BACKUP_SUFFIX_FMT = "%Y%m%d%H%M%S"


class RepairRefusedError(RuntimeError):
    """引数検証(件数超過)で拒否した場合に送出。"""


class TradeIdNotFoundError(RuntimeError):
    """指定されたtrade_idのいずれかがstate/月次データのどこにも見つからない場合に送出。"""


def parse_trade_ids(raw):
    ids = [t.strip() for t in raw.split(",") if t.strip()]
    return ids


def _utc_backup_suffix(now=None):
    now = now or datetime.now(timezone.utc)
    return now.strftime(BACKUP_SUFFIX_FMT)


# =====================================================================
# 対象アセットのダウンロード
# =====================================================================

def _download_state_and_bak(work_dir):
    """本体のfetch_state()(primary検証・破損時bak復旧を含む)をそのまま
    再利用して現行stateを取得する。加えて、修復前の完全なスナップショットを
    残す目的でstate.bak.jsonも(fetch_state()がbak復旧を使わなかった場合でも)
    ローカルへ明示的にダウンロードしておく。
    """
    state, source = acp.fetch_state(work_dir, notify=None)
    if source == "initialized_empty":
        raise RuntimeError(f"{acp.STATE_ASSET_NAME}が存在しません(修復対象のstateがない)")

    bak_path = os.path.join(work_dir, acp.STATE_BAK_ASSET_NAME)
    bak_result = acp.download_release_asset(acp.RELEASE_TAG_STATE, acp.STATE_BAK_ASSET_NAME, bak_path)
    bak_exists = bak_result == "ok"
    return state, bak_exists


def _download_all_monthly_dfs(work_dir):
    """all-candidates-paper-dataタグの全月次csv.gzをダウンロードし、
    {month_key: (local_path, DataFrame)}を返す。
    """
    asset_names = [
        name for name in acp.list_release_assets(acp.RELEASE_TAG_DATA)
        if name.startswith("all_candidates_") and name.endswith(".csv.gz")
    ]
    result = {}
    for asset_name in asset_names:
        month_key = asset_name[len("all_candidates_"):len(asset_name) - len(".csv.gz")]
        local_path = os.path.join(work_dir, asset_name)
        dl_result = acp.download_release_asset(acp.RELEASE_TAG_DATA, asset_name, local_path)
        if dl_result != "ok":
            continue
        import gzip
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
        result[month_key] = (local_path, df)
    return result


# =====================================================================
# 検索・除去(純粋関数、Release I/Oなし)
# =====================================================================

def locate_trade_ids(trade_ids, state, monthly_dfs):
    """各trade_idについて、stateのpositionsに存在するか / どの月次dfに
    存在するかを返す。{trade_id: {"state": bool, "months": [month_key,...]}}
    """
    state_ids = {p.get("trade_id") for p in state.get("positions", [])}
    locations = {}
    for tid in trade_ids:
        months = [mk for mk, (_, df) in monthly_dfs.items() if "trade_id" in df.columns and (df["trade_id"] == tid).any()]
        locations[tid] = {"state": tid in state_ids, "months": months}
    return locations


def remove_from_state(state, trade_ids):
    """stateのpositionsからtrade_idsに合致する行を除去する。last_completed_run_date
    その他のフィールドは一切変更しない。(new_state, removed_positions)を返す。
    """
    trade_id_set = set(trade_ids)
    positions = state.get("positions", [])
    removed = [p for p in positions if p.get("trade_id") in trade_id_set]
    kept = [p for p in positions if p.get("trade_id") not in trade_id_set]
    new_state = dict(state)
    new_state["positions"] = kept
    return new_state, removed


def remove_from_monthly_df(df, trade_ids):
    """月次DataFrameからtrade_idsに合致する行を除去する。(new_df, removed_rows_df)を返す。"""
    trade_id_set = set(trade_ids)
    mask = df["trade_id"].isin(trade_id_set)
    removed_df = df[mask].copy()
    kept_df = df[~mask].copy()
    return kept_df, removed_df


# =====================================================================
# バックアップ(修正前に必ず実行し、sha256で検証する)
# =====================================================================

def backup_and_verify_asset(tag, asset_name, local_path, suffix, work_dir):
    """local_pathの内容をバイト単位でコピーし、`<asset_name>.pre-repair-<suffix>`
    という別名で同じReleaseへアップロード、再ダウンロードしてsha256を照合する。
    照合に失敗した場合は例外を送出する(=本体の変更に絶対に進ませない)。
    戻り値: アップロードしたバックアップアセット名。
    """
    backup_asset_name = f"{asset_name}.pre-repair-{suffix}"
    backup_local_path = os.path.join(work_dir, backup_asset_name)
    with open(local_path, "rb") as src, open(backup_local_path, "wb") as dst:
        dst.write(src.read())

    acp.ensure_release_exists(
        tag, tag,
        "Machine-updated research state (not a software release). "
        "Managed by all_candidates_repair.py / all_candidates_repair.yml.",
    )
    acp.upload_release_asset(tag, backup_local_path)

    verify_path = os.path.join(work_dir, f"_verify_{backup_asset_name}")
    verify_result = acp.download_release_asset(tag, backup_asset_name, verify_path)
    if verify_result != "ok":
        raise RuntimeError(f"バックアップ{backup_asset_name}の再ダウンロード検証に失敗しました(status={verify_result})")
    expected_sha = acp.sha256_file(local_path)
    actual_sha = acp.sha256_file(verify_path)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"バックアップ{backup_asset_name}のsha256不一致: expected={expected_sha} actual={actual_sha}"
        )
    print(f"✅ backup verified: {asset_name} -> {backup_asset_name} (sha256={expected_sha[:12]}...)")
    return backup_asset_name


# =====================================================================
# 差分レポート
# =====================================================================

def format_position(p):
    return (
        f"trade_id={p.get('trade_id')} ticker={p.get('ticker')} direction={p.get('direction')} "
        f"entry_date={p.get('entry_date')} entry_time={p.get('entry_time')} rank={p.get('rank')} "
        f"hold_days={p.get('hold_days')} policy_file={p.get('policy_file')}"
    )


def build_diff_report(trade_ids, reason, dry_run, removed_positions, removed_by_month,
                       state_position_count_before, state_position_count_after,
                       last_completed_run_date, assets_to_backup):
    lines = []
    lines.append("=== all_candidates_repair diff ===")
    lines.append(f"reason: {reason}")
    lines.append(f"dry_run: {dry_run}")
    lines.append(f"target trade_ids (n={len(trade_ids)}):")
    for tid in trade_ids:
        lines.append(f"  - {tid}")
    lines.append("")
    lines.append(f"state positions removed (n={len(removed_positions)}):")
    for p in removed_positions:
        lines.append(f"  - {format_position(p)}")
    lines.append(
        f"state positions unaffected: {state_position_count_after} of {state_position_count_before} (unchanged)"
    )
    lines.append(f"last_completed_run_date: unchanged ({last_completed_run_date})")
    lines.append("")
    lines.append("monthly data rows removed:")
    if not removed_by_month:
        lines.append("  (none)")
    for month_key, removed_df in removed_by_month.items():
        asset_name = acp.month_asset_name(month_key + "-01")
        lines.append(f"  [{month_key}] {asset_name}: {len(removed_df)} row(s) removed")
        for _, row in removed_df.iterrows():
            lines.append(
                f"    - trade_id={row.get('trade_id')} ticker={row.get('ticker')} "
                f"direction={row.get('direction')} entry_date={row.get('entry_date')} "
                f"exit_reason={row.get('exit_reason')}"
            )
    lines.append("")
    if dry_run:
        lines.append("assets that WOULD be backed up + modified (dry_run=True: nothing was uploaded):")
    else:
        lines.append("assets backed up + modified:")
    for tag, asset_name in assets_to_backup:
        lines.append(f"  - [{tag}] {asset_name}")
    lines.append("=== end diff ===")
    return "\n".join(lines)


# =====================================================================
# オーケストレーション
# =====================================================================

def run_repair(trade_ids, reason, dry_run=True, work_dir="."):
    if not trade_ids:
        raise RepairRefusedError("trade_idsが空です")
    if len(trade_ids) > MAX_TRADE_IDS:
        raise RepairRefusedError(
            f"1回の呼び出しで削除できるtrade_idは最大{MAX_TRADE_IDS}件までです(指定={len(trade_ids)}件)"
        )
    if len(set(trade_ids)) != len(trade_ids):
        raise RepairRefusedError(f"trade_idsに重複があります: {trade_ids}")

    print(f"📥 state取得中... (work_dir={work_dir})")
    state, bak_exists = _download_state_and_bak(work_dir)
    print("📥 月次データ取得中...")
    monthly_dfs = _download_all_monthly_dfs(work_dir)
    print(f"  取得済み月: {sorted(monthly_dfs.keys())}")

    locations = locate_trade_ids(trade_ids, state, monthly_dfs)
    not_found = [tid for tid, loc in locations.items() if not loc["state"] and not loc["months"]]
    if not_found:
        raise TradeIdNotFoundError(
            f"以下のtrade_idはstate.positionsにも月次データのどこにも見つかりませんでした"
            f"(何もアップロードせずに中断します): {not_found}"
        )
    for tid, loc in locations.items():
        print(f"🔎 {tid}: state={loc['state']} months={loc['months']}")

    state_position_count_before = len(state.get("positions", []))
    new_state, removed_positions = remove_from_state(state, trade_ids)
    state_changed = len(removed_positions) > 0

    removed_by_month = {}
    repaired_monthly_dfs = {}
    for month_key, (local_path, df) in monthly_dfs.items():
        kept_df, removed_df = remove_from_monthly_df(df, trade_ids)
        repaired_monthly_dfs[month_key] = (local_path, kept_df)
        if len(removed_df) > 0:
            removed_by_month[month_key] = removed_df

    assets_to_backup = []
    if state_changed:
        assets_to_backup.append((acp.RELEASE_TAG_STATE, acp.STATE_ASSET_NAME))
    for month_key in removed_by_month:
        assets_to_backup.append((acp.RELEASE_TAG_DATA, acp.month_asset_name(month_key + "-01")))

    report = build_diff_report(
        trade_ids, reason, dry_run, removed_positions, removed_by_month,
        state_position_count_before, len(new_state["positions"]),
        state.get("last_completed_run_date"), assets_to_backup,
    )
    print(report)

    result = {
        "trade_ids": trade_ids, "reason": reason, "dry_run": dry_run,
        "removed_positions": removed_positions, "removed_by_month": removed_by_month,
        "report": report,
    }

    if dry_run:
        print("🧪 dry_run=True: 何もアップロード/変更しませんでした")
        return result

    if not assets_to_backup:
        print("ℹ️ 変更対象アセットがないため、これ以上の処理はありません")
        return result

    suffix = _utc_backup_suffix()
    print(f"💾 変更前バックアップをアップロード中 (suffix={suffix})...")
    for tag, asset_name in assets_to_backup:
        local_path = os.path.join(work_dir, asset_name)
        backup_and_verify_asset(tag, asset_name, local_path, suffix, work_dir)

    if state_changed:
        # ★重要: state_path(まだ修復前の内容のまま)を先に上書きしては
        # ならない。promote_and_upload_state()自身が「ディスク上の既存
        # state.jsonを.bakへ昇格 → 新state.jsonを書き込み」の順で行う
        # (これが2世代保存の仕組み)ため、ここで先に新state.jsonを書き込むと
        # 昇格される内容が既に新state.jsonになってしまい、.bakが修復前の
        # スナップショットにならない(=2世代保存が壊れる)。
        print("⬆️ 修復済みstateをアップロード中 (validate-then-promote)...")
        acp.promote_and_upload_state(new_state, work_dir=work_dir, upload=True)

    for month_key in removed_by_month:
        local_path, kept_df = repaired_monthly_dfs[month_key]
        kept_df = kept_df.sort_values(["date", "ticker", "direction"]).reset_index(drop=True)
        import gzip
        with gzip.open(local_path, "wt", encoding="utf-8") as f:
            kept_df.to_csv(f, index=False)
        print(f"⬆️ 修復済み月次データをアップロード中: {os.path.basename(local_path)}")
        acp.upload_release_asset(acp.RELEASE_TAG_DATA, local_path)

    print("📊 集計CSVを再計算中...")
    all_frames = [
        acp.rows_to_dataframe(df.to_dict("records"))
        for _, df in repaired_monthly_dfs.values()
    ]
    all_trades = pd.concat(all_frames, ignore_index=True) if all_frames else acp.rows_to_dataframe([])
    daily_summary = acp.compute_daily_summary(all_trades)
    monthly_summary = acp.compute_monthly_summary(daily_summary)
    acp.write_summary_csv(daily_summary, os.path.join(work_dir, acp.DAILY_SUMMARY_FILE))
    acp.write_summary_csv(monthly_summary, os.path.join(work_dir, acp.MONTHLY_SUMMARY_FILE))
    print(f"✅ {acp.DAILY_SUMMARY_FILE} / {acp.MONTHLY_SUMMARY_FILE} を再計算・上書きしました")

    result["daily_summary"] = daily_summary
    result["monthly_summary"] = monthly_summary
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-ids", required=True, help="カンマ区切りのtrade_idリスト(最大5件)")
    parser.add_argument("--reason", required=True, help="修復理由(ログ・差分レポートに記録される)")
    parser.add_argument("--dry-run", default="true", help="'true'(既定)なら何もアップロードせず差分表示のみ")
    parser.add_argument("--work-dir", default=".")
    return parser


def _str_to_bool(s):
    return str(s).strip().lower() in ("true", "1", "yes", "on")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    trade_ids = parse_trade_ids(args.trade_ids)
    dry_run = _str_to_bool(args.dry_run)
    try:
        run_repair(trade_ids, args.reason, dry_run=dry_run, work_dir=args.work_dir)
    except (RepairRefusedError, TradeIdNotFoundError) as e:
        print(f"❌ 修復を拒否/中断しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 all_candidates_repair失敗: {e}")
        raise


if __name__ == "__main__":
    main()
