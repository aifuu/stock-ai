"""Adversarial Validation のDiscord表示を日本語で読みやすくする実行ラッパー。"""
from pathlib import Path
import os
import re
import runpy
import requests

BASE = Path(__file__).resolve().with_name("adversarial_strategy_validator.py")
TMP = Path(__file__).resolve().with_name(".adversarial_pretty_tmp.py")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def pretty_content(msg):
    replacements = [
        ("🛡️ ADVERSARIAL STRATEGY VALIDATOR", "🛡️ AI戦略｜敵対的検証レポート"),
        ("AI ADVERSARIAL VALIDATION", "🛡️ AI戦略｜厳格検証"),
        ("探索対象期間:", "検証期間："),
        ("期間：", "検証期間："),
        ("探索数:", "探索した戦略数："),
        ("N_eff近似:", "実質的な独立戦略数："),
        ("DEV候補:", "開発期間の候補："),
        ("Validation PASS:", "検証合格："),
        ("Validation HOLD:", "検証保留："),
        ("OOS PASS:", "未知データ合格："),
        ("Final PASS:", "最終合格："),
        ("⚠ HOLDは本番投入せずPaper Tradeへ", "⚠️ 保留（HOLD）は本番投入せず、紙上取引で監視"),
        ("VALIDATION", "検証期間"),
        ("DEV", "開発期間"),
        ("OOS", "未知データ検証"),
        ("PASS", "合格"),
        ("HOLD", "保留"),
        ("件数=", "件数＝"),
        ("勝率=", "勝率＝"),
        ("平均=", "平均利益率＝"),
        ("PF=", "利益倍率＝"),
        ("全期間比較", "全期間の比較"),
        ("探索数", "探索した戦略数"),
    ]
    for old, new in replacements:
        msg = msg.replace(old, new)

    # 戦略コードを人間が読める日本語へ
    def strategy_name(m):
        up = m.group(1)
        market = "日経フィルターON" if m.group(2) == "ON" else "日経フィルターOFF"
        style = m.group(3)
        style_text = {
            "BASE": "基本戦略",
            "TESTA55": "テスタ型55",
            "TESTA65": "テスタ型65",
            "TESTA75": "テスタ型75",
        }.get(style, style)
        return f"上昇確率{up}%以上｜{market}｜{style_text}"

    msg = re.sub(
        r"UP(45|50|55|60|65)_NIKKEI(ON|OFF)_(BASE|TESTA55|TESTA65|TESTA75)",
        strategy_name,
        msg,
    )

    # 数値の意味を日本語化（既に日本語化されたものはそのまま）
    msg = msg.replace("勝率：", "勝率＝")
    msg = re.sub(r"\n{4,}", "\n\n", msg)
    return msg.strip()


def send_pretty(msg):
    if not WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOKなし")
        return
    content = pretty_content(msg)
    # 4000文字を超えないよう分割。各メッセージは日本語タイトル付き。
    chunks = [content[i:i + 3800] for i in range(0, len(content), 3800)] or ["検証結果なし"]
    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "embeds": [{
                "title": "🛡️ AI戦略｜敵対的検証レポート" + (f"（{idx}/{len(chunks)}）" if len(chunks) > 1 else ""),
                "description": chunk,
                "color": 15158332,
            }]
        }
        try:
            r = requests.post(WEBHOOK_URL, json=payload, timeout=30)
            print("Discord status =", r.status_code)
            if r.status_code != 204:
                print("❌ Discord送信失敗:", r.text)
        except Exception as e:
            print("❌ Discord送信エラー:", e)


def build_source():
    source = BASE.read_text(encoding="utf-8")
    pattern = re.compile(r"(?ms)^def send_discord\(msg\):\n.*?(?=^def safe_download\()")
    replacement = '''def send_discord(msg):
    send_pretty(msg)


def safe_download('''
    source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError("adversarial_strategy_validator.py のDiscord送信関数を特定できません")
    return source


if __name__ == "__main__":
    TMP.write_text(build_source(), encoding="utf-8")
    try:
        runpy.run_path(str(TMP), run_name="__main__")
    finally:
        try:
            TMP.unlink()
        except FileNotFoundError:
            pass
