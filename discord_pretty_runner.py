"""
Discord表示改善用ランナー
stock_scan.py の計算・判定ロジックは変更せず、実行時だけDiscord送信部分を
Embed形式＋日本語表示に差し替える。
"""

from pathlib import Path
import runpy
import requests
import os
import re

BASE = Path(__file__).resolve().with_name("stock_scan.py")
TMP = Path(__file__).resolve().with_name(".stock_scan_pretty_tmp.py")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def pretty_content(msg):
    replacements = [
        ("⏰ JST:", "🕐 実行時刻："),
        ("📈 日経225トレンド判定", "🇯🇵 市場環境｜日経225"),
        ("日経終値:", "終値："),
        ("MA25:", "25日移動平均："),
        ("MA75:", "75日移動平均："),
        ("日経上昇トレンド:", "上昇トレンド："),
        ("YES 🟢", "🟢 上昇トレンド"),
        ("NO 🔴", "🔴 上昇トレンドではない"),
        ("📊 AI株スキャン結果", "🤖 AI株スキャン｜本日の結論"),
        ("【3クラス分類】", "【下落・横ばい・上昇の3分類】"),
        ("AIスコア:", "AI総合スコア："),
        ("テクニカル:", "テクニカル評価："),
        ("テスタ型モメンタム:", "モメンタム評価："),
        ("5日騰落率:", "直近5日："),
        ("20日騰落率:", "直近20日："),
        ("MA25傾き(5日):", "25日線の傾き："),
        ("出来高急増率:", "出来高増加："),
        ("20日高値突破:", "20日高値からの位置："),
        ("日経対比強度:", "日経に対する強さ："),
        ("下落確率:", "🔴 下落確率："),
        ("横ばい確率:", "🟡 横ばい確率："),
        ("上昇確率:", "🟢 上昇確率："),
        ("買値:", "💰 現在値："),
        ("利確:", "🎯 利確目安："),
        ("損切:", "🛑 損切目安："),
        ("RSI:", "RSI："),
        ("出来高倍率:", "出来高倍率："),
        ("データ日付:", "データ日："),
        ("📊 AI実績（買い推奨のみ集計）", "📊 AI実績｜買い推奨のみ"),
        ("勝率:", "勝率："),
        ("Profit Factor:", "利益倍率："),
        ("平均利益率:", "平均利益率："),
        ("平均保有日数:", "平均保有日数："),
        ("最高利益:", "最高利益："),
        ("最大損失:", "最大損失："),
        ("🏆 買い推奨 順位別", "🏆 買い推奨ランキング"),
        ("🟡 監視シグナル(参考データ・成績には含めない)", "🟡 監視シグナル｜参考値・成績対象外"),
        ("判定数:", "判定数："),
        ("勝ち:", "勝ち："),
        ("負け:", "負け："),
        ("HOLD:", "保留："),
        ("買わない", "見送り"),
        ("監視(拮抗)", "監視｜判断が拮抗"),
        ("監視(日経下落/レンジ)", "監視｜日経が上昇トレンドではない"),
        ("監視(モメンタム不足)", "監視｜モメンタム不足"),
        ("監視(スコア不足)", "監視｜スコア不足"),
        ("監視(上昇確率不足)", "監視｜上昇確率不足"),
        ("⚠️ データが古い可能性のある銘柄", "⚠️ データ鮮度に注意"),
    ]
    for old, new in replacements:
        msg = msg.replace(old, new)
    msg = msg.replace("━━━━━━━━━━━━━━\n", "")
    msg = msg.replace("━━━━━━━━━━━━━━━━━━\n", "")
    msg = re.sub(r"\n{4,}", "\n\n", msg)
    return msg.strip()


def send_pretty(msg):
    if not WEBHOOK_URL:
        print("❌ Webhookなし")
        return
    content = pretty_content(msg)
    if len(content) <= 3900:
        payload = {
            "embeds": [{
                "title": "🤖 AI株スキャン｜日本語レポート",
                "description": content,
                "color": 3447003,
            }]
        }
    else:
        chunks = [content[i:i + 1900] for i in range(0, len(content), 1900)]
        payload = {"content": "🤖 AI株スキャン｜日本語レポート\n" + "\n\n".join(chunks)}
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=30)
        print("Discord status =", r.status_code)
        if r.status_code == 204:
            print("✅ Discord送信成功（日本語・見やすい表示）")
        else:
            print("❌ Discord送信失敗")
            print(r.text)
    except Exception as e:
        print("❌ Discord送信エラー:", e)


def build_source():
    source = BASE.read_text(encoding="utf-8")

    # stock_scan.py の send() は複数行定義になっているため、
    # 「def send(msg):」という完全一致に依存せず、関数本体から
    # 次の「# 銘柄」セクション直前までを安全に差し替える。
    pattern = re.compile(
        r"(?ms)^def\s+send\s*\(\s*[^)]*\s*\):\s*.*?(?=^#\s*=+\n#\s*銘柄\s*$)"
    )

    replacement = '''def send(message):
    send_pretty(message)


'''

    source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError("stock_scan.py の send() 差し替え位置を特定できません")
    return source


if __name__ == "__main__":
    TMP.write_text(build_source(), encoding="utf-8")
    try:
        runpy.run_path(
            str(TMP),
            run_name="__main__",
            init_globals={"send_pretty": send_pretty},
        )
    finally:
        try:
            TMP.unlink()
        except FileNotFoundError:
            pass
