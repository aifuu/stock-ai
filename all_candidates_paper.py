"""独立6ヶ月データ収集トラック(ALL/TOP1/TOP3/TOP5比較用)。

profit_top10_paper.py (以下「live」) はTOP1候補だけを取引し、それ以外の
承認policy通過候補を毎日捨てている。このモジュールは、liveから
scan()とselect_policy_file()の2関数だけを読み取り専用で再利用し、
「その日承認policyを通過した全候補」を仮想的にペーパートレードして、
6ヶ月後にALL/TOP1/TOP3/TOP5および閾値帯別のパフォーマンスを比較できる
データを蓄積する。

liveのSTATE_FILE/HISTORY_FILEには一切書き込まない。状態・生トレード
データはすべてGitHub Releaseアセット(このモジュール専用のタグ)に保存し、
git管理下に置くのは軽量な集計CSV2本(all_candidates_daily_summary.csv /
all_candidates_monthly_summary.csv)だけ。

policyはlive の strategy_policy*.json ではなく、このリポジトリに
一度だけコミットされた「凍結」コピー(all_candidates_frozen_policy*.json)
を使う。select_policy_file()はliveの先物トレンド判定ロジックをそのまま
再利用して「今日どちらの凍結policyを使うか」だけを決める
(DOWN/フォールバックは常に正規の凍結policyへ倒す。将来liveに
strategy_policy_down.jsonが実際に作られても、この研究トラックは
6ヶ月間ずっとフォールバック=all_candidates_frozen_policy.jsonを使い続ける)。

★研究モデル凍結(承認ベース、run()冒頭): git管理下のresearch_model_freeze_
approval.json(approved=true・expected_model_id指定)があり、現行の
directional_model_meta.json(2026-09-12投入のレガシーモデルには無い)の
model_idがそれと一致し、参考採用型のdeploy_reasonでなくvalidation_trades
が最低基準を満たす場合にのみ、その時点のdirectional_model.pklをGitHub
Release(タグ all-candidates-research-model)へ凍結保存する。一度凍結
したら自動では二度と変更しない。凍結後は毎回そのRelease資産をダウンロード
してsha256を検証し、profit_top10_paper.load_modelをこのプロセス内だけ
一時的に差し替えてscan()に使う(profit_top10_paper.pyは変更しない)。
ダウンロード/検証に失敗した場合はliveモデルへフォールバックせず必ず
例外で停止する。承認ファイルはこのタスクでは作成しない(＝当面は凍結
されず、現行liveモデルでデータ収集を続ける)。

★モデル識別(全trade行にmodel_id/model_versionを記録): 各ポジション/
トレード行には、その候補選定に実際に使われたモデルのmodel_id(sha256
先頭16桁、凍結中は凍結モデルのそれ)とmodel_version(directional_model_
meta.jsonのtraining_date、凍結中は凍結時点のtraining_date)を記録する。
directional_model_meta.jsonが無い/model_idが現行pklと一致しない場合
(＝2026-09-12投入のレガシーモデル)はmodel_version="legacy-20260912"と
定義する。過去に記録済みの行(この列が導入される前の行)はこれらの列が
欠損(NaN)のままになるが、それらは全て定義上このレガシー2026-09-12モデル
によるものとして扱う(rewriteはしない)。dedupキー(trade_id)は従来通り
date|ticker|direction|data_date|policy_hashのままで、model_id/model_version
はキーに含めない。

★設計メモ(エントリー価格は「引け値」): このワークフローは東証の大引け
(15:30 JST)後、15:20ではなく16:05 JSTに実行される(理由は
.github/workflows/all_candidates_paper.ymlのcronコメント参照:
決済判定に必要な15:25以降の5分足が確実に存在するようにするため)。
そのためscan()が返す候補価格は実質的に「その日の引け値」であり、
live(profit_top10_paper.py、TOP1のみを寄り後の朝一セッションで
約09:00に約定)のエントリー時刻とは異なる。ALL/TOP1/TOP3/TOP5は
このトラック内で同じscan()呼び出し・同じ実行時刻から作られるため、
バケット間の比較(このモジュールが本来比較したい対象)は内部的に
一貫している。liveとこのトラックの絶対リターンを直接比較する場合は、
エントリー時刻が異なる(このトラックは約引け値、liveは約始値)ことを
踏まえること。
"""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

import daily_model_retrain as dmr
import profit_top10_paper as live_p10
from common import count_tse_trading_days, is_tse_trading_day, tse_trading_days_between
from daily_directional_top1 import download
from profit_top10_paper import scan, select_policy_file

TZ = ZoneInfo("Asia/Tokyo")

FROZEN_POLICY_FILE = "all_candidates_frozen_policy.json"
FROZEN_POLICY_FILE_UP = "all_candidates_frozen_policy_up.json"
# select_policy_file()がこのファイル名を返した場合だけ「上昇用」を使う
# (profit_top10_paper.POLICY_FILE_UPと同じ値。同モジュールからのimportは
# scan/select_policy_fileの2関数のみに限定されているため、値はここで
# 独立して持つ)。
LIVE_POLICY_FILE_UP_NAME = "strategy_policy_up.json"

STATE_ASSET_NAME = "all_candidates_paper_state.json"
STATE_BAK_ASSET_NAME = "all_candidates_paper_state.bak.json"
RELEASE_TAG_STATE = "all-candidates-paper-state"
RELEASE_TAG_DATA = "all-candidates-paper-data"
RELEASE_BASE_URL = "https://github.com/aifuu/stock-ai/releases/download"

DAILY_SUMMARY_FILE = "all_candidates_daily_summary.csv"
MONTHLY_SUMMARY_FILE = "all_candidates_monthly_summary.csv"
SUMMARY_HEADER_COMMENT = (
    "# equal_weight_pnl_jpy is an APPROXIMATION: it simulates splitting a fixed "
    "¥1,000,000 equally across that day's/month's entry-cohort candidates in "
    "each bucket and summing weight*return_pct/100 for the ones that have closed "
    "so far. It is NOT a true continuous equity curve (holds span multiple days, "
    "there is no shared/compounding capital pool), and rows for cohorts with "
    "still-open positions update on later runs as those trades close. "
    "Underlying trade rows (all_candidates_YYYY-MM.csv.gz) carry model_id/"
    "model_version identifying which directional model produced each candidate; "
    "rows predating that column (or with it blank) are defined as the legacy "
    "2026-09-12 model and are never rewritten. Once a research model is frozen "
    "(see all_candidates_paper.py docstring), later rows keep using the frozen "
    "model_id even after the live model changes."
)

BUCKETS = ("ALL", "TOP1", "TOP3", "TOP5")
EQUAL_WEIGHT_CAPITAL = 1_000_000.0

# ★決定(ライブと独立に再実装。mark_and_closeは再利用しない): このトラックは
# 銘柄ごとに独立した想定元本を使う(共有資本の奪い合いをしない)ため、
# 決済ロジックはliveのFEE_RATE/FORCED_EXIT定数をそのまま複製する
# (profit_top10_paper.pyからはscan/select_policy_fileの2関数以外は
# importしない、という制約のため)。値自体はliveと同じにして決済判定が
# 一致するようにする。
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))
FORCED_EXIT = dtime(15, 25)

# ★④研究モデル凍結(承認ベース)関連定数。モジュールdocstring参照。
RESEARCH_MODEL_FREEZE_APPROVAL_FILE = "research_model_freeze_approval.json"
RELEASE_TAG_RESEARCH_MODEL = "all-candidates-research-model"
LIVE_MODEL_FILE = "directional_model.pkl"
LIVE_MODEL_META_FILE = "directional_model_meta.json"
# ★⑤モデル識別: directional_model_meta.jsonが無い/model_id不一致の場合の定義上のバージョン名。
LEGACY_MODEL_VERSION = "legacy-20260912"


class StateCorruptError(RuntimeError):
    """state.jsonとstate.bak.jsonの両方が読み込めない/存在しないときに送出。"""


class ReleaseFetchError(RuntimeError):
    """Releaseアセットの取得が(404以外の理由で)リトライ後も失敗したときに送出。"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def compute_model_id(path):
    """model_id = 指定パスのモデルファイルのバイト列のsha256先頭16桁。
    daily_model_retrain.compute_model_id()と同じ桁数(16桁)で揃える。"""
    return sha256_file(path)[:16]


def current_model_identity(state, work_dir="."):
    """今回のscan()候補が実際に使うモデルの(model_id, model_version)を返す。

    ・凍結済み(state['frozen_model_id']あり) → 凍結モデルのid/training_dateを
      常に優先する(④: liveが後で変わっても凍結モデルを使い続けるため)。
    ・未凍結 → 現行のdirectional_model.pklを直接ハッシュしたものをmodel_idとし、
      directional_model_meta.jsonが存在してそのmodel_idと一致すればtraining_
      dateをmodel_versionとする。メタが無い/不一致(2026-09-12投入のレガシー
      モデル)ならLEGACY_MODEL_VERSIONとする。
    """
    if state.get("frozen_model_id"):
        return state["frozen_model_id"], state.get("frozen_model_training_date") or LEGACY_MODEL_VERSION

    live_path = os.path.join(work_dir, LIVE_MODEL_FILE)
    if not os.path.exists(live_path):
        return None, LEGACY_MODEL_VERSION
    model_id = compute_model_id(live_path)
    model_version = LEGACY_MODEL_VERSION
    meta_path = os.path.join(work_dir, LIVE_MODEL_META_FILE)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and meta.get("model_id") == model_id and meta.get("training_date"):
                model_version = meta["training_date"]
        except Exception:
            pass
    return model_id, model_version


def load_frozen_policy(path):
    """凍結policyファイルを読み込み、scan()が期待する型(float/bool/int)へ
    変換する。liveのload_policy()とは異なり、HMAC署名検証は一切行わない
    (署名検証コードには触れない、という制約のため)。凍結ファイルは
    生成時点でliveのAPPROVED済みpolicyをバイト単位で複製したものであり、
    ここで再検証する対象ではない。
    """
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    required = [
        "up_threshold", "min_score_for_buy", "nikkei_filter",
        "atr_tp_multiplier", "atr_sl_multiplier", "hold_days",
    ]
    missing = [k for k in required if k not in p]
    if missing:
        raise RuntimeError("凍結policy不足: " + ",".join(missing))
    p["up_threshold"] = float(p["up_threshold"])
    p["min_score_for_buy"] = float(p["min_score_for_buy"])
    p["nikkei_filter"] = str(p["nikkei_filter"]).lower() in ("true", "1", "yes", "on")
    p["atr_tp_multiplier"] = float(p["atr_tp_multiplier"])
    p["atr_sl_multiplier"] = float(p["atr_sl_multiplier"])
    p["hold_days"] = int(p["hold_days"])
    return p


def choose_frozen_policy_file(live_policy_file):
    """live側のselect_policy_file()が返したファイル名から、使う凍結ファイルを
    決める。'...up.json'を選んだ場合だけ凍結up、それ以外(DOWN・フォール
    バックいずれも)は凍結normalへ倒す(このトラックのDOWN方針は6ヶ月間
    固定で、将来liveにstrategy_policy_down.jsonができても変わらない)。
    """
    if live_policy_file == LIVE_POLICY_FILE_UP_NAME:
        return FROZEN_POLICY_FILE_UP
    return FROZEN_POLICY_FILE


def discord_send(message):
    webhook = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook:
        print(f"[all_candidates_paper] {message}")
        return False
    try:
        import requests
        r = requests.post(webhook, json={"content": message[:1950]}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"⚠️ all_candidates_paper Discord通知失敗: {e}")
        return False


# =====================================================================
# GitHub Release I/O (subprocess経由。テストはすべてsubprocess.runを
# モックし、実ネットワーク呼び出しは一切行わない)
# =====================================================================

def _curl_download_with_status(url, dest_path, timeout=30):
    """curlでダウンロードし、(http_status:int|None, error:Exception|None)を返す。

    http_statusがNoneなのはcurl自体が完走できなかった場合(ネットワーク断・
    タイムアウトなど)のみ。HTTPレベルの404/5xxはstatusとして返る
    (-fを付けないため、非2xxでもcurlの終了コードは0)。
    -Lを付けてリダイレクトを追跡する: GitHub Releaseの既存アセットへの
    downloadURLは常に署名付きの release-assets.githubusercontent.com へ
    302リダイレクトするため、-Lが無いとstatusが302のまま返り(200でも
    404でもない)、既存アセットのダウンロードが常に非404失敗として
    リトライ→ハード失敗になってしまう。statusはリダイレクト追跡後の
    最終ステータス。
    """
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(timeout), "-w", "%{http_code}",
             "-o", dest_path, url],
            capture_output=True, text=True, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, exc
    if proc.returncode != 0:
        return None, RuntimeError(f"curl failed rc={proc.returncode} stderr={proc.stderr.strip()}")
    status_text = (proc.stdout or "").strip()
    try:
        status = int(status_text)
    except ValueError:
        return None, RuntimeError(f"curl returned non-numeric status: {status_text!r}")
    return status, None


def download_release_asset(tag, asset_name, dest_path, retries=3, retry_wait=5,
                            base_url=None):
    """Releaseアセットをダウンロードする。

    戻り値: 'ok' (ダウンロード成功) | '404' (アセットが存在しない=正常系)
    例外: ReleaseFetchError -- 404以外の失敗がretries回とも続いた場合
    (ネットワーク断・5xxなど)。この場合は絶対に空状態へフォールバック
    してはならない。
    """
    base_url = base_url or RELEASE_BASE_URL
    url = f"{base_url}/{tag}/{asset_name}"
    last_err = None
    for attempt in range(1, retries + 1):
        status, err = _curl_download_with_status(url, dest_path)
        if err is None and status == 200:
            return "ok"
        if err is None and status == 404:
            return "404"
        last_err = err if err is not None else RuntimeError(f"HTTP {status}")
        print(f"⚠️ {asset_name} download失敗 (試行{attempt}/{retries}): {last_err}")
        if attempt < retries:
            time.sleep(retry_wait)
    raise ReleaseFetchError(
        f"{asset_name}のダウンロードが{retries}回とも失敗しました(404ではない): {last_err}"
    )


def ensure_release_exists(tag, title, notes):
    subprocess.run(
        ["gh", "release", "create", tag, "--title", title, "--notes", notes, "--prerelease"],
        capture_output=True, text=True, check=False,
    )


def upload_release_asset(tag, local_path):
    proc = subprocess.run(
        ["gh", "release", "upload", tag, local_path, "--clobber"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh release upload失敗 ({local_path} -> {tag}): {proc.stderr.strip()}")


def list_release_assets(tag, retries=3, retry_wait=3):
    """指定タグの既存アセット名一覧を返す。Releaseが無ければ空リスト。

    defense-in-depth: gh CLI呼び出し自体の一時的な失敗や、GitHub側の
    eventual-consistencyラグ(直前のアップロードがまだ一覧に反映されて
    いない)に備えて数回リトライする。根本対策ではない -- 同一run内で
    自分がアップロードした月をこの一覧で再取得することに依存しない
    (run()側でメモリ上のデータを使う)のが本筋の修正。
    """
    for attempt in range(1, retries + 1):
        proc = subprocess.run(
            ["gh", "release", "view", tag, "--json", "assets", "-q", ".assets[].name"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            return [line for line in proc.stdout.splitlines() if line.strip()]
        if attempt < retries:
            time.sleep(retry_wait)
    return []


# =====================================================================
# ④研究モデル凍結(承認ベース)。モジュールdocstring参照。
# =====================================================================

def _load_freeze_approval(path=None):
    p = path or RESEARCH_MODEL_FREEZE_APPROVAL_FILE
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_live_model_meta(path=None):
    p = path or LIVE_MODEL_META_FILE
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or "model_id" not in data:
        return None
    return data


def evaluate_freeze_eligibility(state, work_dir=".", min_validation_trades=None):
    """凍結の4条件をすべて評価する。(eligible: bool, reason: str, meta: dict|None)を返す。

    条件(すべて満たす場合のみeligible=True):
      (1) 承認ファイル(RESEARCH_MODEL_FREEZE_APPROVAL_FILE)がgit管理下に存在し、
          approved=true かつ expected_model_id指定
      (2) 現行のdirectional_model_meta.jsonが存在する(＝2026-09-12投入の
          レガシーモデルではない)、かつそのmodel_idが(1)のexpected_model_idと一致
      (3) その投入理由(deploy_reason)が'参考採用'型ではなく、validation_tradesが
          最低評価基準(RETRAIN_MIN_OOS_TRADES)以上
      (4) stateにまだ凍結済みモデルが無い(re-freeze禁止)
    どれか1つでも欠ければeligible=Falseで、失敗した条件を明記したreasonを返す。
    """
    min_validation_trades = (
        dmr.MIN_OOS_TRADES if min_validation_trades is None else min_validation_trades
    )

    if state.get("frozen_model_id"):
        return False, "既に研究モデルを凍結済みのため対象外(re-freezeは行わない)", None

    approval = _load_freeze_approval(os.path.join(work_dir, RESEARCH_MODEL_FREEZE_APPROVAL_FILE))
    if approval is None or not approval.get("approved") or not approval.get("expected_model_id"):
        return (
            False,
            f"承認ファイル({RESEARCH_MODEL_FREEZE_APPROVAL_FILE})が無い/approved=trueでない/"
            "expected_model_id未指定のため対象外",
            None,
        )

    meta = _load_live_model_meta(os.path.join(work_dir, LIVE_MODEL_META_FILE))
    if meta is None:
        return (
            False,
            f"現行モデルに{LIVE_MODEL_META_FILE}(サイドカー)が無く"
            "(2026-09-12投入のレガシーモデル相当)、model_idを確認できないため対象外",
            None,
        )

    if meta.get("model_id") != approval.get("expected_model_id"):
        return (
            False,
            f"承認ファイルのexpected_model_id({approval.get('expected_model_id')})と"
            f"現行モデルのmodel_id({meta.get('model_id')})が不一致のため対象外",
            None,
        )

    deploy_reason = str(meta.get("deploy_reason") or "")
    if "参考採用" in deploy_reason:
        return False, f"現行モデルの投入理由が参考採用型('{deploy_reason}')のため対象外", None

    try:
        validation_trades = int(meta.get("validation_trades"))
    except (TypeError, ValueError):
        return False, "現行モデルのvalidation_trades情報が不正/欠損のため対象外", None
    if validation_trades < min_validation_trades:
        return (
            False,
            f"現行モデルのvalidation_trades({validation_trades})が"
            f"最低評価基準({min_validation_trades})未満のため対象外",
            None,
        )

    return True, "全条件を満たすため凍結対象", meta


def freeze_research_model(meta, work_dir=".", upload=True, now=None):
    """evaluate_freeze_eligibility()でeligible=Trueと判定されたmetaを元に、
    現行directional_model.pklをGitHub Release(all-candidates-research-model)
    へアップロードし、アップロード後に再ダウンロード・sha256照合して初めて
    成功とする(all_candidates_repair.pyのbackup_and_verify_assetと同じ
    validate-then-promoteスタイル)。戻り値はstateへマージするフィールドのdict。
    """
    now = now or datetime.now(TZ)
    model_id = meta["model_id"]
    date_str = now.strftime("%Y%m%d")
    asset_name = f"research_model_{date_str}_{model_id}.joblib"
    meta_asset_name = f"research_model_{date_str}_{model_id}.meta.json"

    live_pkl_path = os.path.join(work_dir, LIVE_MODEL_FILE)
    expected_sha = sha256_file(live_pkl_path)
    local_asset_path = os.path.join(work_dir, asset_name)
    shutil.copyfile(live_pkl_path, local_asset_path)

    local_meta_path = os.path.join(work_dir, meta_asset_name)
    with open(local_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if upload:
        ensure_release_exists(
            RELEASE_TAG_RESEARCH_MODEL, RELEASE_TAG_RESEARCH_MODEL,
            "Machine-frozen research model snapshot (not a software release). "
            "Managed by all_candidates_paper.py. Frozen once, approval-gated, "
            "and never re-frozen automatically.",
        )
        upload_release_asset(RELEASE_TAG_RESEARCH_MODEL, local_asset_path)
        upload_release_asset(RELEASE_TAG_RESEARCH_MODEL, local_meta_path)

        verify_path = os.path.join(work_dir, f"_verify_{asset_name}")
        verify_result = download_release_asset(RELEASE_TAG_RESEARCH_MODEL, asset_name, verify_path)
        if verify_result != "ok":
            raise RuntimeError(f"凍結モデル{asset_name}の再ダウンロード検証に失敗しました(status={verify_result})")
        actual_sha = sha256_file(verify_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"凍結モデル{asset_name}のsha256不一致: expected={expected_sha} actual={actual_sha}"
            )
        print(f"✅ 研究モデル凍結完了・検証済み: {asset_name} (sha256={expected_sha[:12]}...)")

    return {
        "frozen_model_id": model_id,
        "frozen_model_asset": asset_name,
        "frozen_model_meta_asset": meta_asset_name,
        "frozen_model_training_date": meta.get("training_date"),
        "frozen_model_sha256": expected_sha,
        "frozen_at": now.isoformat(),
    }


def load_frozen_model(state, work_dir="."):
    """凍結済みモデルをReleaseからダウンロードしsha256を検証してロードする。

    ダウンロード失敗・sha256不一致は必ず例外を送出する(liveモデルへの
    フォールバックは絶対に行わない、という④の要件)。
    """
    import joblib

    asset_name = state["frozen_model_asset"]
    expected_sha = state.get("frozen_model_sha256")
    local_path = os.path.join(work_dir, f"_frozen_{asset_name}")
    result = download_release_asset(RELEASE_TAG_RESEARCH_MODEL, asset_name, local_path)
    if result != "ok":
        raise RuntimeError(
            f"凍結モデル{asset_name}のダウンロードに失敗しました(status={result})。"
            "liveモデルへのフォールバックは行いません"
        )
    actual_sha = sha256_file(local_path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"凍結モデル{asset_name}のsha256不一致: expected={expected_sha} actual={actual_sha}。"
            "liveモデルへのフォールバックは行いません"
        )
    return joblib.load(local_path)


def _scan_with_model(policy, model):
    """指定モデルを使ってscan()を呼ぶ。profit_top10_paper.load_model()を
    このプロセス内だけ一時的に差し替える(profit_top10_paper.pyは変更しない)。
    """
    original = live_p10.load_model
    live_p10.load_model = lambda: model
    try:
        return scan(policy, limit=None)
    finally:
        live_p10.load_model = original


# =====================================================================
# State (open positions) 永続化
# =====================================================================

def default_state():
    return {"positions": []}


def _validate_state(data):
    return isinstance(data, dict) and isinstance(data.get("positions"), list)


def _load_and_validate_state(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not _validate_state(data):
        raise ValueError(f"{path}: 想定外の形式(positionsがlistのdictではない)")
    return data


def fetch_state(work_dir=".", notify=discord_send):
    """state.jsonをReleaseから取得する。

    戻り値: (state:dict, source:str) source ∈ {'initialized_empty','primary','bak_recovered'}
    例外: StateCorruptError -- primaryもbakも読めない/検証失敗。
          ReleaseFetchError -- 404以外の理由でダウンロード自体が失敗
          (download_release_asset内でretries尽き後に送出。ここでは
          キャッチせず、そのまま呼び出し元へハード失敗として伝播させる)。
    """
    state_path = os.path.join(work_dir, STATE_ASSET_NAME)
    bak_path = os.path.join(work_dir, STATE_BAK_ASSET_NAME)

    primary_result = download_release_asset(RELEASE_TAG_STATE, STATE_ASSET_NAME, state_path)
    if primary_result == "404":
        return default_state(), "initialized_empty"

    try:
        data = _load_and_validate_state(state_path)
        return data, "primary"
    except Exception as primary_exc:
        bak_result = download_release_asset(RELEASE_TAG_STATE, STATE_BAK_ASSET_NAME, bak_path)
        if bak_result == "404":
            raise StateCorruptError(
                f"{STATE_ASSET_NAME}が破損しており、{STATE_BAK_ASSET_NAME}も存在しません: {primary_exc!r}"
            )
        try:
            data = _load_and_validate_state(bak_path)
        except Exception as bak_exc:
            raise StateCorruptError(
                f"{STATE_ASSET_NAME}と{STATE_BAK_ASSET_NAME}が両方とも読み込めません: "
                f"primary={primary_exc!r} bak={bak_exc!r}"
            )
        _notify(notify, f"⚠️ {STATE_ASSET_NAME}が破損していたため{STATE_BAK_ASSET_NAME}から復元しました: {primary_exc!r}")
        return data, "bak_recovered"


def _notify(notify, message):
    if notify is None:
        return
    try:
        notify(message)
    except Exception as exc:
        print(f"⚠️ all_candidates_paper通知失敗(無視して継続): {exc}")


def promote_and_upload_state(new_state, work_dir=".", upload=True):
    """新state.jsonを書き込み、直前まで良好だった既存state.jsonを.bakへ
    昇格させてからReleaseへアップロードする(validate-before-promote)。
    """
    serialized = json.dumps(new_state, ensure_ascii=False, indent=2)
    json.loads(serialized)  # 書き込み前に必ずパースできることを確認する

    state_path = os.path.join(work_dir, STATE_ASSET_NAME)
    bak_path = os.path.join(work_dir, STATE_BAK_ASSET_NAME)

    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                json.load(f)
        except Exception:
            pass  # 壊れている場合は.bakを上書きしない(safe_state.pyと同じ方針)
        else:
            shutil.copyfile(state_path, bak_path)

    with open(state_path, "w", encoding="utf-8") as f:
        f.write(serialized)

    if upload:
        ensure_release_exists(
            RELEASE_TAG_STATE, RELEASE_TAG_STATE,
            "Machine-updated research state (not a software release). "
            "Managed by all_candidates_paper.py / all_candidates_paper.yml.",
        )
        upload_release_asset(RELEASE_TAG_STATE, state_path)
        if os.path.exists(bak_path):
            upload_release_asset(RELEASE_TAG_STATE, bak_path)


# =====================================================================
# 決済ロジック(profit_top10_paper.mark_and_closeと同じ判定ルールを
# 使う独立実装。liveのSTATE/HISTORYには一切書き込まない)
# =====================================================================

def download_5m(ticker):
    """profit_top10_paper.download_5mと同じロジックの独立コピー
    (profit_top10_paper.pyからはscan/select_policy_file以外import禁止のため)。"""
    import yfinance as yf
    try:
        d = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False, threads=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        idx = pd.to_datetime(d.index)
        idx = idx.tz_convert(TZ).tz_localize(None) if getattr(idx, "tz", None) is not None else idx.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
        d.index = idx
        return d.sort_index()
    except Exception:
        return None


def evaluate_exits(positions, now, download_fn=None, download_5m_fn=None):
    """liveのmark_and_close()と同じ決済判定(直近営業日を日足で遡って
    TP/SL未検知を確認 → 当日5分足でTP/SL → 保有上限到達時のHOLD_LIMIT)を
    独立に再実装する。資本管理・持ち高更新は行わず、(remaining, closed)の
    2リストを返すだけの純粋関数に近い形にする。
    """
    download_fn = download_fn or download
    download_5m_fn = download_5m_fn or download_5m
    remaining, closed = [], []
    for p in positions:
        ep = float(p["entry_price"])
        direction = p.get("direction", "BUY")
        entry_date = pd.Timestamp(p["entry_date"])
        hold_limit = max(1, int(p.get("hold_days") or 1))
        exit_price = reason = exit_dt = None

        prior_days = tse_trading_days_between(entry_date + pd.Timedelta(days=1), pd.Timestamp(now.date()) - pd.Timedelta(days=1))
        if len(prior_days) > 0:
            daily = download_fn(p["ticker"], period="3mo")
            if daily is not None and not daily.empty:
                for ts, b in daily[daily.index.normalize().isin(prior_days)].iterrows():
                    hi, lo = float(b["High"]), float(b["Low"])
                    if direction == "BUY":
                        if lo <= p["sl"] and hi >= p["tp"]:
                            exit_price, reason = float(p["sl"]), "SL"
                        elif hi >= p["tp"]:
                            exit_price, reason = float(p["tp"]), "TP"
                        elif lo <= p["sl"]:
                            exit_price, reason = float(p["sl"]), "SL"
                    else:
                        if hi >= p["sl"] and lo <= p["tp"]:
                            exit_price, reason = float(p["sl"]), "SL"
                        elif lo <= p["tp"]:
                            exit_price, reason = float(p["tp"]), "TP"
                        elif hi >= p["sl"]:
                            exit_price, reason = float(p["sl"]), "SL"
                    if reason:
                        exit_dt = ts
                        break
                    # ★決定(rollbackでのHOLD_LIMIT、liveのmark_and_closeとの
                    # 意図的な小さな差分): liveは5分おきに走るため、HOLD_LIMITは
                    # 常に「当日」の15:25以降の5分足で判定される。このトラックは
                    # 1日1回(15:30引け後)しか走らないため、ワークフローがある日
                    # 丸ごとスキップされると、その日の5分足はもう二度と取得でき
                    # ず、当日ループのHOLD_LIMIT判定(下のforループ)では検知
                    # できない。そのためここで、その日(ts)時点の保有営業日数が
                    # 上限に達していれば、TP/SLがこの同じ日に成立していない
                    # (=reasonがまだNone)ことを条件に、その日の日足Closeで
                    # HOLD_LIMIT決済にする。liveは毎回当日ループを回せるため
                    # この分岐は不要(=liveには存在しない、このトラック固有の
                    # 差分)。exit_dtは15:25足のcloseの近似としてその日の15:30に
                    # 固定する(liveのFORCED_EXIT=15:25の5分足closeにいちばん
                    # 近い、このトラックで取得できる価格が日足Closeであるため)。
                    held_at_ts = count_tse_trading_days(entry_date + pd.Timedelta(days=1), ts)
                    if held_at_ts >= hold_limit:
                        exit_price, reason = float(b["Close"]), "HOLD_LIMIT"
                        exit_dt = pd.Timestamp.combine(pd.Timestamp(ts).date(), dtime(15, 30))
                        break

        held = count_tse_trading_days(entry_date + pd.Timedelta(days=1), pd.Timestamp(now.date()))
        if exit_price is None:
            d = download_5m_fn(p["ticker"])
            if d is not None and not d.empty:
                bars = d[d.index.date == now.date()]
                for ts, b in bars.iterrows():
                    if ts.time() < dtime(9, 0):
                        continue
                    hi, lo = float(b["High"]), float(b["Low"])
                    if direction == "BUY":
                        if lo <= p["sl"]:
                            exit_price, reason = float(p["sl"]), "SL"
                        elif hi >= p["tp"]:
                            exit_price, reason = float(p["tp"]), "TP"
                    else:
                        if hi >= p["sl"]:
                            exit_price, reason = float(p["sl"]), "SL"
                        elif lo <= p["tp"]:
                            exit_price, reason = float(p["tp"]), "TP"
                    if reason:
                        exit_dt = ts
                        break
                    if held >= hold_limit and ts.time() >= FORCED_EXIT:
                        exit_price, reason = float(b["Close"]), "HOLD_LIMIT"
                        exit_dt = ts
                        break

        if exit_price is None:
            remaining.append(p)
            continue

        exit_date_str = str(pd.Timestamp(exit_dt).date()) if exit_dt is not None else str(now.date())
        exit_time_str = pd.Timestamp(exit_dt).strftime("%H:%M") if exit_dt is not None else now.strftime("%H:%M")
        gross_pct = ((exit_price - ep) / ep * 100) if direction == "BUY" else ((ep - exit_price) / ep * 100)
        # ★決定(手数料の扱い): このトラックは銘柄ごとに独立元本で株数を
        # 持たないため、liveのように株数×価格で手数料pnlを引くことができない。
        # liveのFEE_RATEをそのまま「往復定率(エントリー・決済の2回分)」として
        # パーセンテージに直接適用する近似にする(株数に依存しないぶん、
        # liveのreturn_pctと厳密一致はしないが、桁感は揃う)。
        return_pct = gross_pct - FEE_RATE * 100 * 2

        row = dict(p)
        row.update(
            exit_price=exit_price, exit_time=exit_time_str, exit_date=exit_date_str,
            exit_reason=reason, return_pct=return_pct,
        )
        closed.append(row)
    return remaining, closed


# =====================================================================
# 新規エントリー構築(dedup + 独立元本、共有資本の奪い合いをしない)
# =====================================================================

def build_trade_id(entry_date, ticker, direction, data_date, policy_hash):
    return f"{entry_date}|{ticker}|{direction}|{data_date}|{policy_hash}"


def build_new_positions(known_trade_ids, candidates, today, policy, policy_file,
                         policy_hash, trend_down_flag, model_id=None, model_version=None):
    """scan()が返した全候補(既にexpected_value_pct,score降順=rankそのもの)から、
    まだ持っていないtrade_idの分だけ新規ポジションを作る。

    ★決定(dedupキーのentry_timestamp): 仕様の「date|ticker|direction|
    entry_timestamp|policy_hash」のentry_timestampに実行時刻(HH:MM)を
    使うと、同日中にワークフローがリトライされた場合に時刻が変わって
    しまいdedupが機能しない(=再実行で重複エントリーする)。そのため
    ここではscan()候補が持つ'data_date'(その銘柄の直近日足の日付。
    同一営業日中は再実行しても不変)をentry_timestamp相当として使う。
    実行時刻そのものは参考情報としてentry_time列に残すが、dedupキーには
    含めない。

    ★⑤モデル識別: model_id/model_versionは今回のcandidates(scan()呼び出し)が
    実際に使ったモデルを示す(current_model_identity()参照)。dedupキー
    (trade_id)には含めない。
    """
    new_positions = []
    for rank, c in enumerate(candidates, start=1):
        data_date = c.get("data_date", today)
        trade_id = build_trade_id(today, c["ticker"], c["direction"], data_date, policy_hash)
        if trade_id in known_trade_ids:
            continue
        new_positions.append({
            "trade_id": trade_id,
            "date": today,
            "ticker": c["ticker"],
            "direction": c["direction"],
            "rank": rank,
            "score": c.get("score"),
            "up_probability": c.get("up_probability"),
            "down_probability": c.get("down_probability"),
            "nikkei_filter": bool(policy.get("nikkei_filter")),
            "policy_file": policy_file,
            "policy_hash": policy_hash,
            "trend_down_flag": bool(trend_down_flag),
            "entry_price": c["price"],
            "tp": c["tp"],
            "sl": c["sl"],
            "hold_days": int(policy["hold_days"]),
            "entry_date": today,
            "entry_time": datetime.now(TZ).strftime("%H:%M"),
            "model_id": model_id,
            "model_version": model_version,
        })
    return new_positions


# =====================================================================
# 月次生データセット(GitHub Release: all-candidates-paper-data)
# =====================================================================

TRADE_COLUMNS = [
    "trade_id", "date", "entry_date", "ticker", "direction", "rank", "score",
    "up_probability", "down_probability", "nikkei_filter", "policy_file",
    "policy_hash", "trend_down_flag", "entry_price", "tp", "sl", "hold_days",
    "entry_time", "exit_price", "exit_time", "exit_date", "exit_reason", "return_pct",
    # ★⑤追加(モデル識別): 欠損(NaN)の行はレガシー2026-09-12モデル扱い
    # (モジュールdocstring・SUMMARY_HEADER_COMMENT参照)。過去行は書き換えない。
    "model_id", "model_version",
]


def rows_to_dataframe(rows):
    df = pd.DataFrame(rows)
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[TRADE_COLUMNS]


def month_asset_name(date_str):
    return f"all_candidates_{date_str[:7]}.csv.gz"


def append_month_rows(rows_df, month_key, work_dir=".", upload=True):
    """指定月(YYYY-MM)の既存アセットを取得し、trade_idでdedupして追記・
    再アップロードする。存在しない月(404)は新規アセットとして扱う。
    """
    asset_name = month_asset_name(month_key + "-01")
    local_path = os.path.join(work_dir, asset_name)
    result = download_release_asset(RELEASE_TAG_DATA, asset_name, local_path)
    if result == "ok":
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            existing = pd.read_csv(f)
        combined = pd.concat([existing, rows_df], ignore_index=True)
    else:
        combined = rows_df
    combined = combined.drop_duplicates(subset=["trade_id"], keep="last")
    combined = combined.sort_values(["date", "ticker", "direction"]).reset_index(drop=True)
    with gzip.open(local_path, "wt", encoding="utf-8") as f:
        combined.to_csv(f, index=False)
    if upload:
        ensure_release_exists(
            RELEASE_TAG_DATA, RELEASE_TAG_DATA,
            "Machine-updated research data (not a software release). "
            "Managed by all_candidates_paper.py / all_candidates_paper.yml. "
            "One asset per calendar month (all_candidates_YYYY-MM.csv.gz).",
        )
        upload_release_asset(RELEASE_TAG_DATA, local_path)
    return combined


def append_trade_rows(rows, work_dir=".", upload=True):
    """closed+still-openの全行を、各行のentry月(date列)ごとにグルーピングして
    append_month_rowsへ渡す。1回の呼び出しで複数月にまたがってもよい。
    """
    if not rows:
        return {}
    df = rows_to_dataframe(rows)
    df["_month"] = df["date"].astype(str).str.slice(0, 7)
    out = {}
    for month_key, group in df.groupby("_month"):
        out[month_key] = append_month_rows(group.drop(columns=["_month"]), month_key, work_dir=work_dir, upload=upload)
    return out


def download_all_months(work_dir=".", months=None):
    """all-candidates-paper-dataタグの全月次アセット(または指定monthsのみ)を
    ダウンロードして連結したDataFrameを返す。1件もなければ空DataFrame。
    """
    if months is None:
        assets = list_release_assets(RELEASE_TAG_DATA)
        months = sorted({name[len("all_candidates_"):len("all_candidates_") + 7] for name in assets if name.startswith("all_candidates_") and name.endswith(".csv.gz")})
    frames = []
    for month_key in months:
        asset_name = month_asset_name(month_key + "-01")
        local_path = os.path.join(work_dir, asset_name)
        result = download_release_asset(RELEASE_TAG_DATA, asset_name, local_path)
        if result != "ok":
            continue
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            frames.append(pd.read_csv(f))
    if not frames:
        return rows_to_dataframe([])
    return pd.concat(frames, ignore_index=True)


# =====================================================================
# 集計(ALL/TOP1/TOP3/TOP5、日次・月次)
# =====================================================================

def _bucket_frame(df, bucket):
    if bucket == "ALL":
        return df
    n = int(bucket[len("TOP"):])
    return df[df["rank"] <= n]


def compute_daily_summary(all_trades_df):
    """全トレード行(open+closed混在)から日次(=エントリー日コホート)の
    ALL/TOP1/TOP3/TOP5集計を作る。純粋関数(Release I/Oなし)、fixtureで
    直接テストできる。
    """
    if all_trades_df.empty:
        return pd.DataFrame(columns=["date", "bucket", "candidate_count", "trades_closed", "win_rate", "avg_return_pct", "equal_weight_pnl_jpy"])
    df = all_trades_df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    rows = []
    for date_val, day_df in df.groupby("date"):
        for bucket in BUCKETS:
            b = _bucket_frame(day_df, bucket)
            candidate_count = len(b)
            closed = b[b["exit_price"].notna()]
            trades_closed = len(closed)
            win_rate = float((closed["return_pct"] > 0).mean()) if trades_closed else None
            avg_return_pct = float(closed["return_pct"].mean()) if trades_closed else None
            if candidate_count > 0 and trades_closed > 0:
                weight = EQUAL_WEIGHT_CAPITAL / candidate_count
                equal_weight_pnl = float((weight * closed["return_pct"] / 100).sum())
            else:
                equal_weight_pnl = None
            rows.append({
                "date": date_val, "bucket": bucket, "candidate_count": candidate_count,
                "trades_closed": trades_closed, "win_rate": win_rate,
                "avg_return_pct": avg_return_pct, "equal_weight_pnl_jpy": equal_weight_pnl,
            })
    return pd.DataFrame(rows).sort_values(["date", "bucket"]).reset_index(drop=True)


def compute_monthly_summary(daily_summary_df):
    """日次集計を暦月に丸めた集計(候補数・決済数は合計、勝率・平均リターンは
    決済件数で加重平均、equal_weight_pnl_jpyはその月の日次近似値の単純合計)。
    """
    if daily_summary_df.empty:
        return pd.DataFrame(columns=["month", "bucket", "candidate_count", "trades_closed", "win_rate", "avg_return_pct", "equal_weight_pnl_jpy"])
    df = daily_summary_df.copy()
    df["month"] = df["date"].astype(str).str.slice(0, 7)
    rows = []
    for (month_val, bucket), g in df.groupby(["month", "bucket"]):
        candidate_count = int(g["candidate_count"].sum())
        trades_closed = int(g["trades_closed"].sum())
        closed_g = g[g["trades_closed"] > 0]
        if trades_closed > 0:
            win_rate = float((closed_g["win_rate"] * closed_g["trades_closed"]).sum() / trades_closed)
            avg_return_pct = float((closed_g["avg_return_pct"] * closed_g["trades_closed"]).sum() / trades_closed)
        else:
            win_rate = None
            avg_return_pct = None
        equal_weight_pnl = float(g["equal_weight_pnl_jpy"].dropna().sum()) if g["equal_weight_pnl_jpy"].notna().any() else None
        rows.append({
            "month": month_val, "bucket": bucket, "candidate_count": candidate_count,
            "trades_closed": trades_closed, "win_rate": win_rate,
            "avg_return_pct": avg_return_pct, "equal_weight_pnl_jpy": equal_weight_pnl,
        })
    return pd.DataFrame(rows).sort_values(["month", "bucket"]).reset_index(drop=True)


def write_summary_csv(df, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(SUMMARY_HEADER_COMMENT + "\n")
        df.to_csv(f, index=False)


# =====================================================================
# 実行オーケストレーション
# =====================================================================

def run(now=None, work_dir=".", upload=True):
    now = now or datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")

    if not (now.weekday() < 5 and is_tse_trading_day(now.date())):
        print(f"\U0001f4a4 all_candidates_paper {today}: 東証休場日のためスキップ")
        return {"today": today, "skipped": "not_a_trading_day"}

    # ★決定(実行時刻ウィンドウガード、2026-09-29のインシデントを受けて):
    # 起動トリガーがschedule/workflow_dispatchのどちらであっても、日付が
    # またいで遅延発火した場合に「今日」を誤って翌営業日の寄り前として
    # 処理してしまう事故があった(2026-09-28分がJST 00:18に発火し、
    # 2026-09-29のコホートとして誤って処理された)。このトラックは大引け
    # (15:30 JST)後の1日1回処理を前提にしており、決済判定
    # (evaluate_exits)も15:25以降の5分足の存在に依存するため、
    # 15:35 JSTより前、または日付が変わった後(翌日0:00〜15:34)に
    # 走った場合は、その時点のnow.date()を「今日」として処理しては
    # ならない。ここで弾けば、何がトリガーであっても、runは必ず
    # 自分が処理すべき営業日(大引け後、日付が変わる前)を処理する
    # ことが保証される。stateへは一切触れずに返す。
    if not (dtime(15, 35) <= now.time() <= dtime(23, 59)):
        print(
            f"⏳ all_candidates_paper {today}: 実行時刻 {now.strftime('%H:%M')} JSTが"
            "受付ウィンドウ(15:35〜23:59 JST)外のためスキップ"
            "(大引け後・日付が変わる前にしか正しく処理できないため)"
        )
        return {"today": today, "skipped": "outside_run_window"}

    frozen_policies = {
        FROZEN_POLICY_FILE: load_frozen_policy(os.path.join(work_dir, FROZEN_POLICY_FILE)),
        FROZEN_POLICY_FILE_UP: load_frozen_policy(os.path.join(work_dir, FROZEN_POLICY_FILE_UP)),
    }
    policy_hashes = {
        FROZEN_POLICY_FILE: sha256_file(os.path.join(work_dir, FROZEN_POLICY_FILE)),
        FROZEN_POLICY_FILE_UP: sha256_file(os.path.join(work_dir, FROZEN_POLICY_FILE_UP)),
    }

    live_policy_file, trend_result = select_policy_file()
    trend_down_flag = trend_result.get("trend") == "down"
    frozen_policy_file = choose_frozen_policy_file(live_policy_file)
    policy = frozen_policies[frozen_policy_file]
    policy_hash = policy_hashes[frozen_policy_file]

    state, state_source = fetch_state(work_dir)
    print(f"\U0001f4e5 state取得: source={state_source}")

    if state.get("last_completed_run_date") == today:
        print(f"⏭ all_candidates_paper {today}: 本日は既に実行済みのためスキップ")
        return {"today": today, "skipped": "already_completed_today"}

    # ★④研究モデル凍結(承認ベース)。trading-day/window/same-dayゲート・state
    # 取得の直後、scan()より前に評価する。凍結済みなら毎回その凍結モデルを
    # ダウンロード・sha256検証して使う(失敗時はハード失敗、liveへのフォール
    # バックはしない)。未凍結なら4条件を評価し、満たせば今回のみ凍結する。
    frozen_model = None
    if state.get("frozen_model_id"):
        frozen_model = load_frozen_model(state, work_dir=work_dir)
        print(f"🧊 凍結済み研究モデルを使用: {state['frozen_model_asset']}")
    else:
        eligible, freeze_note, freeze_meta = evaluate_freeze_eligibility(state, work_dir=work_dir)
        if eligible:
            freeze_fields = freeze_research_model(freeze_meta, work_dir=work_dir, upload=upload, now=now)
            state.update(freeze_fields)
            print(f"🧊 研究モデルを新規凍結しました: {freeze_fields['frozen_model_asset']}")
        else:
            print(f"ℹ️ 研究モデル凍結見送り({freeze_note})、現行liveモデルでデータ収集を継続")

    model_id, model_version = current_model_identity(state, work_dir=work_dir)

    remaining, closed = evaluate_exits(state.get("positions", []), now)

    if frozen_model is not None:
        candidates, scanned = _scan_with_model(policy, frozen_model)
    else:
        candidates, scanned = scan(policy, limit=None)

    known_ids = {p["trade_id"] for p in remaining} | {p["trade_id"] for p in closed}
    new_positions = build_new_positions(
        known_ids, candidates, today, policy, frozen_policy_file, policy_hash, trend_down_flag,
        model_id=model_id, model_version=model_version,
    )

    # ★決定(集計はこのrunの再リストに依存しない): list_release_assets()は
    # このrunがまだ何もアップロードしていない、いま この時点で呼ぶ。
    # これより後でRELEASE_TAG_DATAへアップロードする月は、GitHub側の
    # eventual-consistencyでこの一覧に載らないことがあるため
    # (=直前にアップロードした月が本人からすら見えない)、その月の集計には
    # この一覧を使わない。
    prior_month_assets = list_release_assets(RELEASE_TAG_DATA)
    prior_months = sorted({
        name[len("all_candidates_"):len("all_candidates_") + 7]
        for name in prior_month_assets
        if name.startswith("all_candidates_") and name.endswith(".csv.gz")
    })

    # ★state.update()ではなく明示的なdictにする(positions/last_completed_run_date
    # 以外の未知フィールドを無限に引きずらないため)。frozen_model_*フィールドは
    # 凍結済み/今回凍結した場合のみstateに存在するので、それだけ明示的に引き継ぐ。
    new_state = {"positions": remaining + new_positions, "last_completed_run_date": today}
    for key in (
        "frozen_model_id", "frozen_model_asset", "frozen_model_meta_asset",
        "frozen_model_training_date", "frozen_model_sha256", "frozen_at",
    ):
        if key in state:
            new_state[key] = state[key]
    promote_and_upload_state(new_state, work_dir=work_dir, upload=upload)

    rows_to_append = closed + remaining + new_positions
    appended_by_month = append_trade_rows(rows_to_append, work_dir=work_dir, upload=upload)

    # append_trade_rows()が返すDataFrameは、各月についてアップロード前に
    # 取得した既存アセット + 今回の行 を既にメモリ上でマージ済みなので、
    # 今回触った月はそれをそのまま使う(再ダウンロード不要=再リストの
    # ラグの影響を受けない)。今回触っていない月だけ、run開始時に取得した
    # prior_monthsに基づいてダウンロードする(このrunでは書き込んでいない
    # ので再リストのレースは起きない)。
    other_months = [m for m in prior_months if m not in appended_by_month]
    other_trades = download_all_months(work_dir=work_dir, months=other_months)
    all_trades = pd.concat(
        list(appended_by_month.values()) + [other_trades], ignore_index=True,
    ) if appended_by_month else other_trades
    daily_summary = compute_daily_summary(all_trades)
    monthly_summary = compute_monthly_summary(daily_summary)
    write_summary_csv(daily_summary, os.path.join(work_dir, DAILY_SUMMARY_FILE))
    write_summary_csv(monthly_summary, os.path.join(work_dir, MONTHLY_SUMMARY_FILE))

    print(
        f"✅ all_candidates_paper {today}: scanned={scanned} candidates={len(candidates)} "
        f"opened={len(new_positions)} closed={len(closed)} open_total={len(new_state['positions'])} "
        f"policy={frozen_policy_file}"
    )
    return {
        "today": today, "candidates": candidates, "opened": new_positions, "closed": closed,
        "state": new_state, "daily_summary": daily_summary, "monthly_summary": monthly_summary,
    }


def main():
    try:
        run()
    except Exception as e:
        discord_send(f"\U0001f6a8 all_candidates_paper失敗: {e}")
        raise


if __name__ == "__main__":
    main()
