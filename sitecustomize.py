"""GitHub Actions 起動時に自動ロードされる軽量なDiscord表示補助。"""
import re
import os
import subprocess
import sys

try:
    import requests
    _original_post = requests.post

    def _pretty_adversarial_post(url, *args, **kwargs):
        payload = kwargs.get("json")
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            msg = payload["content"]
            if "ADVERSARIAL" in msg or "AI ADVERSARIAL VALIDATION" in msg:
                replacements = [
                    ("🛡️ ADVERSARIAL STRATEGY VALIDATOR", "🛡️ AI戦略｜敵対的検証レポート"),
                    ("AI ADVERSARIAL VALIDATION", "🛡️ AI戦略｜厳格検証"),
                    ("期間：", "検証期間："),
                    ("探索数：", "探索した戦略数："),
                    ("探索数:", "探索した戦略数："),
                    ("N_eff近似：", "実質的な独立戦略数："),
                    ("N_eff近似:", "実質的な独立戦略数："),
                    ("DEV候補：", "開発期間の候補："),
                    ("DEV候補:", "開発期間の候補："),
                    ("Validation PASS：", "検証合格："),
                    ("Validation PASS:", "検証合格："),
                    ("Validation HOLD：", "検証保留："),
                    ("Validation HOLD:", "検証保留："),
                    ("OOS PASS：", "未知データ合格："),
                    ("OOS PASS:", "未知データ合格："),
                    ("Final PASS：", "最終合格："),
                    ("Final PASS:", "最終合格："),
                    ("⚠ HOLDは本番投入せずPaper Tradeへ", "⚠️ 保留（HOLD）は本番投入せず、紙上取引で監視"),
                    ("全期間比較", "全期間の比較"),
                ]
                for old, new in replacements:
                    msg = msg.replace(old, new)

                def strategy_name(m):
                    up = m.group(1)
                    market = "日経フィルターON" if m.group(2) == "ON" else "日経フィルターOFF"
                    style = {
                        "BASE": "基本戦略",
                        "TESTA55": "テスタ型55",
                        "TESTA65": "テスタ型65",
                        "TESTA75": "テスタ型75",
                    }.get(m.group(3), m.group(3))
                    return f"上昇確率{up}%以上｜{market}｜{style}"

                msg = re.sub(
                    r"UP(45|50|55|60|65)_NIKKEI(ON|OFF)_(BASE|TESTA55|TESTA65|TESTA75)",
                    strategy_name,
                    msg,
                )
                msg = re.sub(r"\n{4,}", "\n\n", msg).strip()
                kwargs["json"] = {
                    "embeds": [{
                        "title": "🛡️ AI戦略｜敵対的検証レポート",
                        "description": msg[:3900],
                        "color": 15158332,
                    }]
                }
        return _original_post(url, *args, **kwargs)

    requests.post = _pretty_adversarial_post
except Exception:
    # 表示補助が失敗してもAI本体を止めない。
    pass

# The production paper workflow historically invokes paper_fast_entrypoint.py.
# Route only the normal execution of that legacy command to the forced 1/3/5-day
# TOP1 trader; keep --analysis-only untouched for the pre-09:30 analysis phase.
if os.path.basename(sys.argv[0]) == "paper_fast_entrypoint.py" and "--analysis-only" not in sys.argv:
    rc = subprocess.call([sys.executable, "multi_hold_paper.py", *sys.argv[1:]])
    os._exit(rc)
