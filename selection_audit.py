import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import profit_top10_paper as pt

TZ = ZoneInfo('Asia/Tokyo')
AUDIT_FILE = 'selection_audit.csv'
TOP10 = 10


def finite(v, default=0.0):
    try:
        x = float(v)
        return x if x == x and x not in (float('inf'), float('-inf')) else default
    except Exception:
        return default


def run():
    now = datetime.now(TZ)
    today = now.strftime('%Y-%m-%d')
    policy = pt.load_policy()

    # 実際の本番売買(profit_top10_paper.py)と完全に同一の候補生成・フィルタ・
    # 期待値ソートロジックをそのまま再利用する。以前はここで別の独自スコア式
    # (出来高比率35%を含む2段階ファネル)を再実装しており、実際の売買ロジックと
    # 一致しない監査結果を出していた。
    cands, scanned = pt.scan(policy)
    if not cands:
        raise RuntimeError('選定候補が0件')

    top10 = cands[:TOP10]
    top1 = top10[0]

    rows = []
    for i, c in enumerate(top10):
        out = dict(c)
        out['rank'] = i + 1
        out['gap_vs_top1_score'] = finite(top1['score']) - finite(c['score'])
        out['gap_vs_top1_expected_value_pct'] = finite(top1['expected_value_pct']) - finite(c['expected_value_pct'])
        rows.append(out)

    audit = pd.DataFrame(rows)
    audit.insert(0, 'generated_at', now.isoformat())
    audit.to_csv(AUDIT_FILE, index=False, encoding='utf-8-sig')

    msg = [
        '🔍 TOP1選定監査(実際の売買ロジックと完全一致)',
        f'📅 {today} {now:%H:%M} JST',
        f'対象ユニバース: {len(pt.TICKERS)}銘柄｜取得成功: {scanned}',
        f'🥇 最終TOP1: {top1["ticker"]} {top1["company"]}（{top1["direction"]}）｜'
        f'スコア {finite(top1["score"]):.2f}｜期待値 {finite(top1["expected_value_pct"]):+.2f}%',
        '',
        '順位 | 銘柄 | 方向 | AI上昇% | AI下落% | スコア | 期待値% | TOP1差',
    ]
    for i, c in enumerate(top10):
        msg.append(
            f'{i + 1} | {c["ticker"]} {c["company"]} | {c["direction"]} | '
            f'{finite(c["up_probability"]):.1f}% | {finite(c["down_probability"]):.1f}% | '
            f'{finite(c["score"]):.2f} | {finite(c["expected_value_pct"]):+.2f}% | '
            f'{finite(top1["score"]) - finite(c["score"]):+.2f}'
        )
    msg += [
        '',
        '🧠 TOP1の根拠:',
        f'方向: {top1["direction"]}｜スコア: {finite(top1["score"]):.2f}｜'
        f'期待値: {finite(top1["expected_value_pct"]):+.2f}%',
        f'買った基準: {top1.get("buy_reason", "")}',
        f'📁 詳細: {AUDIT_FILE}',
    ]
    text = '\n'.join(msg)
    if os.getenv('DISCORD_WEBHOOK'):
        pt.discord_send(text)
    print(text)


if __name__ == '__main__':
    run()
