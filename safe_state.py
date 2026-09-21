"""共通の安全な状態ファイル読み書きヘルパー。

各ペーパートレードモジュール(profit_top10_paper.py, daily_directional_top1.py,
nikkei_macd_dip_paper.py, risk_manager.py)のstate(JSON)/履歴(CSV)ファイルの
読み込み・書き込みを一箇所にまとめる。JSONのスキーマ(キー構成)には一切
関与しない ―― ここは「壊れたファイルをどう安全に扱うか」だけを担当する。

設計:
  - 保存は必ずアトミック(同ディレクトリに.tmpを書いてfsyncしてからos.replace)。
    置き換え前の(読めていた)旧ファイルは<name>.bakとして1世代分残す。
  - 読み込みは「ファイルが無い」(初回起動などの正常系。挙動は変更しない)と
    「ファイルはあるが壊れている」を区別する。壊れている場合は<name>.bakへの
    フォールバックを試み、それも駄目なら元ファイルのコピーを
    <name>.corrupt-<UTC timestamp>として隔離し、呼び出し元から渡された
    通知関数(各モジュール既存のDiscord通知など)でエラーを知らせたうえで
    StateCorruptErrorを送出する(通知自体の失敗はこの関数からは伝播しない)。
  - .bak/.corrupt-*はgit管理下に置かない(.gitignore参照)。
"""

import itertools
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd


class StateCorruptError(RuntimeError):
    """状態ファイルとその.bakの両方が読み込み不能なときに送出される。"""


_quarantine_seq = itertools.count()


def _utc_stamp():
    """隔離ファイル名用のUTCタイムスタンプ。1秒内の複数回失敗でも衝突(上書き)
    しないよう、マイクロ秒とプロセス内カウンタを付与して一意にする。"""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%S')}.{now.microsecond:06d}Z-{next(_quarantine_seq):04d}"


def _notify(notify, message):
    if notify is None:
        return
    try:
        notify(message)
    except Exception as exc:
        print(f"⚠️ safe_state通知失敗(無視して継続): {exc}")


def atomic_write_json(path, data):
    """dataをpathへアトミックに書き込む。置き換え前のpathは<path>.bakへ複製する。

    ただし置き換え前のpathが壊れている(有効なJSONとして読めない)場合は、
    その壊れた内容で既存の.bak(＝直近の正常な状態)を上書きしてしまわないよう
    .bakへの複製をスキップする(「壊れている→.bakから復元→保存」という流れで
    安全網である.bak自体を壊さないため)。
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
        except Exception as exc:
            print(f"⚠️ {path}: 内容が壊れているため.bak更新をスキップします(既存の.bakを保持): {exc}")
        else:
            try:
                shutil.copyfile(path, path + ".bak")
            except OSError as exc:
                print(f"⚠️ {path}: .bak作成失敗(無視して継続): {exc}")
    os.replace(tmp, path)


def _read_json(path, validate):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if validate is not None and not validate(data):
        raise ValueError(f"{path}: 検証失敗(想定外の形式)")
    return data


def load_json_state(path, notify=None, label=None, validate=None):
    """状態JSONを読み込む。

    戻り値:
      - pathが存在しない -> None (呼び出し元は自分のデフォルトを使う。既存の挙動を変えない)
      - pathが正常に読める -> パース済みdict/リスト
      - pathが壊れているが<path>.bakが読める -> .bakの内容(通知して復元)
    例外:
      - StateCorruptError: pathも.bakも読めない(隔離コピー作成+通知を試みたうえで送出)
    """
    label = label or path
    if not os.path.exists(path):
        return None
    try:
        return _read_json(path, validate)
    except Exception as primary_exc:
        bak_path = path + ".bak"
        bak_exc = None
        if os.path.exists(bak_path):
            try:
                data = _read_json(bak_path, validate)
                _notify(notify, f"⚠️ {label}: 状態ファイル破損を検知、.bakから復元しました({primary_exc})")
                return data
            except Exception as exc:
                bak_exc = exc
        quarantine = f"{path}.corrupt-{_utc_stamp()}"
        try:
            shutil.copyfile(path, quarantine)
        except OSError as exc:
            print(f"⚠️ {path}: 隔離コピー作成失敗: {exc}")
            quarantine = None
        _notify(
            notify,
            f"🚨 {label}: 状態ファイルと.bakが両方とも読み込めません。取引を中断します。"
            f" primary_error={primary_exc!r} bak_error={bak_exc!r} quarantine={quarantine}",
        )
        raise StateCorruptError(
            f"{label}: 状態ファイルが破損し.bakも利用不能です(primary={primary_exc!r}, bak={bak_exc!r})"
        )


def _recovery_path(path):
    if path.endswith(".csv"):
        return path[:-4] + ".recovery.csv"
    return path + ".recovery.csv"


def safe_append_history(path, row, notify=None, label=None, encoding="utf-8-sig"):
    """履歴CSVに1行追記する。

    健全系(pathが無い、またはpd.read_csvで読める)は従来と完全に同じ出力
    (同じ列・順序・エンコーディング)になる。pathは存在するが読み込めない場合、
    元ファイルは一切上書きせずに<path>.corrupt-<UTC timestamp>へ複製し、
    新しい行は<path>を.csv→.recovery.csvに置き換えたファイルへ追記する
    (ポジションは既にメモリ上で決済済みのため、このtickは中断しない)。
    """
    label = label or path
    new_row = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
        except Exception as primary_exc:
            quarantine = f"{path}.corrupt-{_utc_stamp()}"
            try:
                shutil.copyfile(path, quarantine)
            except OSError as exc:
                print(f"⚠️ {path}: 隔離コピー作成失敗: {exc}")
                quarantine = None
            recovery = _recovery_path(path)
            header = not os.path.exists(recovery)
            new_row.to_csv(recovery, mode="a", index=False, header=header, encoding=encoding)
            _notify(
                notify,
                f"🚨 {label}: 履歴CSVが破損しています。元ファイルは上書きせず{quarantine}に隔離し、"
                f"新しい行は{recovery}に退避しました({primary_exc!r})。",
            )
            return
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False, encoding=encoding)
