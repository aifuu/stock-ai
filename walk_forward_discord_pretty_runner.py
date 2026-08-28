"""
WALK-FORWARD Discord表示改善ランナー

walk_forward.py / walk_forward_fast_runner.py の計算ロジックは変更しない。
Discordへ送信する本文だけを日本語・見やすい表示へ変換する。
"""

from pathlib import Path
import re
import runpy
import requests

BASE = Path(__file__).resolve().with_name("walk_forward_fast_runner.py")

_original_post = requests.post


def strategy_name_jp(name: str) -> str:
    """内部戦略名をDiscord表示用の日本語へ変換。"""
    s = str(name)
    m = re.fullmatch(r"UP(\d+)_NIKKEI(ON|OFF)_(BASE|TESTA(55|65|75))", s)
    if not m:
        return s

    up = m.group(1)
    nikkei = "ON" if m.group(2) == "ON" else "OFF"
    mode = m.group(3)
    if mode == "BASE":
        testa = "BASE"
    else:
        testa = f"テスタ型{m.group(5)}"

    return f"上昇確率{up}%｜日経フィルター{nikkei}｜{testa}"


def pretty_content(msg: str) -> str:
    s = str(msg)

    replacements = [
        ("🧪 AI WALK FORWARD STRATEGY TEST", "🧪 AIウォークフォワード戦略テスト"),
        ("AI WALK FORWARD STRATEGY TEST", "AIウォークフォワード戦略テスト"),
        ("【VALIDATION】", "【検証期間】"),
        ("【DEV】", "【開発・戦略選択期間】"),
        ("【OOS・参考確認】", "【OOS・未使用期間の参考確認】"),
        ("【OOS】", "【OOS・未使用期間】"),
        ("【全期間比較】", "【全期間比較】"),
        ("比較：", "比較："),
        ("再学習：", "再学習間隔："),
        ("ATR TP：", "ATR利確："),
        ("ATR SL：", "ATR損切："),
        ("件数=", "取引件数="),
        ("平均=", "平均リターン="),
        ("PF=", "利益倍率(PF)="),
        ("データなし", "データなし"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)

    # 戦略名だけを表示用に日本語化。CSVや内部ロジックには一切触れない。
    s = re.sub(
        r"\bUP(\d+)_NIKKEI(ON|OFF)_(BASE|TESTA(?:55|65|75))\b",
        lambda m: strategy_name_jp(m.group(0)),
        s,
    )

    # 見出しの視認性を改善。
    s = s.replace("━━━━━━━━━━━━━━━━━━", "━━━━━━━━━━━━━━━━━━")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def patched_post(url, *args, **kwargs):
    # Discord webhookだけを対象にする。
    if "discord.com/api/webhooks/" in str(url):
        payload = kwargs.get("json")
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            payload = dict(payload)
            payload["content"] = pretty_content(payload["content"])
            kwargs["json"] = payload

    return _original_post(url, *args, **kwargs)


requests.post = patched_post


if __name__ == "__main__":
    runpy.run_path(str(BASE), run_name="__main__")
