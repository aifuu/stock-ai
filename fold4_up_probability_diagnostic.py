import pandas as pd
import numpy as np
from pathlib import Path

P = Path('walk_forward_all_candidates.csv')
OUT = Path('fold4_up_probability_diagnostic.csv')

if not P.exists():
    raise SystemExit('❌ walk_forward_all_candidates.csv がありません')

df = pd.read_csv(P)
if 'date' not in df.columns:
    raise SystemExit('❌ date列がありません')
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
df = df.dropna(subset=['date']).copy()

dates = sorted(df['date'].drop_duplicates())
if len(dates) < 239:
    raise SystemExit(f'❌ unique dates不足: {len(dates)} < 239')

oos_dates = pd.DatetimeIndex(dates[-239:])
oos = df[df['date'].isin(oos_dates)].copy()

# 候補CSVで使われる可能性のある確率列を自動検出。
prob_candidates = ['prob_up', 'up_probability', 'probability_up', 'up_prob', 'p_up', 'up_prob_pct', 'prob']
prob_col = next((c for c in prob_candidates if c in oos.columns), None)

print('============================================================')
print('🔎 FOLD 4 OOS UP-PROBABILITY DIAGNOSTIC')
print('ゲート条件・探索閾値・policyは変更しない。診断のみ。')
print('============================================================')
print(f'OOS期間: {oos_dates[0].date()} -> {oos_dates[-1].date()} ({len(oos_dates)}営業日)')
print(f'RAW_OOS: {len(oos):,}')
print('numeric columns:', ', '.join(oos.select_dtypes(include=np.number).columns.tolist()))

if prob_col is None:
    # 候補CSVに確率列が保存されていない場合は、列名を明示して終了。
    print('❌ UP確率列を候補CSVから特定できませんでした。')
    print('利用可能列:', ', '.join(oos.columns.tolist()))
    raise SystemExit(2)

x = pd.to_numeric(oos[prob_col], errors='coerce')
# 0-1表現なら百分率へ変換。
if x.notna().any() and x.max() <= 1.0:
    pct = x * 100.0
    scale = '0-1 -> %'
else:
    pct = x
    scale = '%'

oos['_up_pct'] = pct
valid = oos['_up_pct'].dropna()
print(f'probability column: {prob_col} ({scale})')
print(f'valid probability rows: {len(valid):,} / {len(oos):,}')

if len(valid):
    qs = valid.quantile([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99])
    print('--- UP probability distribution (%) ---')
    print(f'min={valid.min():.4f} max={valid.max():.4f} mean={valid.mean():.4f} median={valid.median():.4f}')
    for q,v in qs.items():
        print(f'p{int(q*100):02d}={v:.4f}')

    print('--- threshold counts ---')
    for t in [20,25,30,35,40,42.5,45,47.5,50,52.5,55,60,65,70,75,80]:
        n = int((valid >= t).sum())
        print(f'UP>={t:>4}: {n:>6} ({100*n/len(valid):6.2f}%)')

# 日次分布。8/19・8/20など急落日の確認に使う。
daily = oos.groupby('date').agg(
    rows=('_up_pct','size'),
    valid_prob=('_up_pct', lambda s: int(s.notna().sum())),
    up45=('_up_pct', lambda s: int((s >= 45).sum())),
    up50=('_up_pct', lambda s: int((s >= 50).sum())),
    mean_up=('_up_pct','mean'),
    median_up=('_up_pct','median'),
).reset_index()
print('--- last 20 OOS dates ---')
print(daily.tail(20).to_string(index=False))

daily.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'診断CSV: {OUT} / {len(daily)} rows')
