#!/usr/bin/env python3
"""Forced daily TOP1 paper trading with independent 1/3/5 business-day hold buckets.

Every business day after 09:30 exactly one TOP1 is selected. The daily TOP1 is
used for any hold bucket that is currently empty. The 1d bucket therefore trades
every business day; the 3d/5d buckets wait until their previous position is
fully closed, then use that day's TOP1 (no separate re-selection). TP/SL exits
are handled by the same entry conditions when supplied by the selected signal;
the hold period is the maximum target close.
"""
import json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import requests
import paper_fast_entrypoint as paper
import paper_risk_policy

TZ=ZoneInfo('Asia/Tokyo')
STATE_FILE='multi_hold_paper_state.json'
HISTORY_FILE='multi_hold_paper_history.csv'
INITIAL_CAPITAL=float(os.getenv('AI_INITIAL_CAPITAL','1000000'))
HOLDS=(1,3,5)
WEBHOOK_URL=os.getenv('DISCORD_WEBHOOK')
MARKET_CLOSE_MINUTES=15*60+35
MIN_AVG_VOLUME=300_000
# ★修正(2026-09): profit_top10_paper.pyと同じ手数料率を往復(entry+exit)で適用する。
# これまでmulti_hold_paperは手数料を一切計上しておらず、月次収益率が実態より
# 良く見えていた。
FEE_RATE=paper_risk_policy.FEE_RATE

def now(): return datetime.now(TZ)
def today(): return now().date().isoformat()

def _default_state():
    return {
        'capitals':{str(h):INITIAL_CAPITAL for h in HOLDS},
        'peaks':{str(h):INITIAL_CAPITAL for h in HOLDS},
        'daily_start_capitals':{str(h):INITIAL_CAPITAL for h in HOLDS},
        'risk_date':None,
        'positions':[],'trades_today':0,'last_entry_date':None,
    }

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            s=json.load(open(STATE_FILE,encoding='utf-8'))
            s.setdefault('capitals',{str(h):INITIAL_CAPITAL for h in HOLDS})
            # ★修正(2026-09): 日次損失上限/最大DDのリスク判定に必要な状態を追加。
            # 旧state.jsonにはこれらのキーが無いため、既存資産をそのままpeak/
            # 当日開始資産として初期化する(いきなり停止判定にならないように)。
            s.setdefault('peaks',dict(s['capitals']))
            s.setdefault('daily_start_capitals',dict(s['capitals']))
            s.setdefault('risk_date',None)
            s.setdefault('positions',[]); s.setdefault('trades_today',0); s.setdefault('last_entry_date',None)
            return s
        except Exception: pass
    return _default_state()

def save_state(s):
    tmp=STATE_FILE+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f: json.dump(s,f,ensure_ascii=False,indent=2)
    os.replace(tmp,STATE_FILE)

def reset_daily_risk(s,d):
    """新しい営業日になったら、各バケットの当日開始資産をリセットする。"""
    if s.get('risk_date')!=d:
        s['risk_date']=d
        s['daily_start_capitals']={h:s['capitals'].get(h,INITIAL_CAPITAL) for h in s['capitals']}

def risk_check(s,h):
    """指定バケットについて、共通リスクポリシー(日次損失上限・最大DD)を判定する。"""
    h=str(h)
    bucket_state={
        'capital':s['capitals'].get(h,INITIAL_CAPITAL),
        'daily_start_capital':s['daily_start_capitals'].get(h,INITIAL_CAPITAL),
        'peak':s['peaks'].get(h,INITIAL_CAPITAL),
        # multi_holdは1バケットにつき同時1ポジションのみ・取引数上限は対象外。
        'positions':[p for p in s['positions'] if str(p.get('hold_days'))==h],
        'trades_today':0,
    }
    ok,reason=paper_risk_policy.evaluate(bucket_state)
    return ok,reason

def daily_bars(ticker):
    d=yf.download(ticker,period='6mo',interval='1d',auto_adjust=True,progress=False,threads=False)
    if d is None or d.empty:return None
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
    return d

def target_date(entry_date,hold):
    return pd.bdate_range(pd.Timestamp(entry_date),periods=hold)[-1].date().isoformat()

def append(rows):
    if not rows:return
    d=pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        try:d=pd.concat([pd.read_csv(HISTORY_FILE),d],ignore_index=True)
        except Exception:pass
    d.to_csv(HISTORY_FILE,index=False,encoding='utf-8-sig')

def notify(text):
    if not WEBHOOK_URL:return
    try: requests.post(WEBHOOK_URL,json={'content':text},timeout=10).raise_for_status()
    except Exception as e: print(f'⚠️ Discord通知失敗: {e}')

def close_due(s):
    remaining=[]; closed=[]
    n=now(); minute=n.hour*60+n.minute
    for p in s['positions']:
        target=pd.Timestamp(p['exit_date']).date()
        if target>n.date() or (target==n.date() and minute<MARKET_CLOSE_MINUTES):
            remaining.append(p); continue
        d=daily_bars(p['ticker'])
        if d is None or pd.Timestamp(p['exit_date']) not in d.index:
            remaining.append(p); continue
        exit_price=float(d.loc[pd.Timestamp(p['exit_date']),'Close'])
        entry=float(p['entry_price']); sign=1 if p['direction']=='BUY' else -1
        # ★修正(2026-09): 往復(entry+exit)の手数料を計上する。profit_top10_paper.py
        # と同じFEE_RATEを使い、これまで無視されていたコストを反映する。
        ret=sign*(exit_price/entry-1.0)-FEE_RATE*2
        h=str(p['hold_days'])
        new_capital=float(s['capitals'].get(h,INITIAL_CAPITAL))*(1+ret)
        s['capitals'][h]=new_capital
        s.setdefault('peaks',{})[h]=max(float(s.get('peaks',{}).get(h,new_capital)),new_capital)
        closed.append({**p,'exit_price':exit_price,'return_pct':ret*100,'status':'CLOSED','exit_reason':'TARGET_CLOSE'})
    s['positions']=remaining
    append(closed)
    return closed

def _passes_liquidity_and_trend(ticker, direction):
    """Final hard gate: average volume and direction-specific daily trend."""
    d=daily_bars(ticker)
    if d is None or len(d)<75:return False, None
    close=pd.to_numeric(d['Close'],errors='coerce').dropna()
    volume=pd.to_numeric(d['Volume'],errors='coerce').dropna()
    if len(close)<75 or len(volume)<20:return False, None
    price=float(close.iloc[-1]); ma25=float(close.rolling(25).mean().iloc[-1]); ma75=float(close.rolling(75).mean().iloc[-1]); ma25_prev=float(close.rolling(25).mean().iloc[-6]); avg20=float(volume.tail(20).mean())
    if avg20<MIN_AVG_VOLUME:return False, None
    uptrend=price>ma25>ma75 and ma25>ma25_prev
    downtrend=price<ma25<ma75 and ma25<ma25_prev
    if direction=='BUY' and not uptrend:return False, None
    if direction=='SHORT' and not downtrend:return False, None
    return True, d

def choose_top1():
    policy=paper.loop.app.load_policy()
    candidates,_=paper.scan_progressive_with_prefilter(policy)
    if not candidates: raise RuntimeError('TOP1候補が生成されませんでした')
    for c in candidates:
        ticker=str(c.get('ticker') or c.get('symbol') or '')
        direction=str(c.get('direction','BUY')).upper()
        if direction not in ('BUY','SHORT'):continue
        ok,_=_passes_liquidity_and_trend(ticker,direction)
        if ok:
            return c
    raise RuntimeError('TOP1候補はあるが、平均出来高30万株以上かつ方向一致トレンドを満たす銘柄がありません')

def open_daily(s):
    d=today()
    if s.get('last_entry_date')==d:
        print('⏭ 本日の強制TOP1売買は既に成立済み')
        return
    if now().weekday()>=5:
        print('⏭ 土日なので取引しません')
        return

    # 銘柄選定は毎日1回だけ。3日/5日枠のための別選定は行わない。
    c=choose_top1()
    ticker=str(c.get('ticker') or c.get('symbol') or '')
    if not ticker: raise RuntimeError('TOP1 ticker missing')
    direction=str(c.get('direction','BUY')).upper()
    if direction not in ('BUY','SHORT'): direction='BUY'
    ok,df=_passes_liquidity_and_trend(ticker,direction)
    if not ok: raise RuntimeError(f'{ticker}: final liquidity/trend gate failed')
    entry=float(df.iloc[-1]['Close'])
    company=c.get('company') or c.get('name') or ticker

    # その日のTOP1を、空いている保有枠にだけ入れる。
    # 1日枠は毎営業日必ず空くため毎日売買。
    # 3日/5日枠は前ポジションの決済後、その決済日のTOP1を使う。
    reset_daily_risk(s,d)
    active_holds={int(p.get('hold_days',0)) for p in s['positions']}
    opened=[]; risk_blocked=[]
    for h in HOLDS:
        if h in active_holds:
            continue
        # ★修正(2026-09): バケットごとに日次損失上限・最大DDを判定する。
        # これまでpaper_risk_policyへの接続が無く、連敗・急落時の停止機構が
        # 一切存在しなかった。
        ok,reason=risk_check(s,h)
        if not ok:
            risk_blocked.append((h,reason))
            print(f'⛔ {h}日枠: リスクポリシーにより新規エントリー停止: {reason}')
            continue
        p={'entry_date':d,'ticker':ticker,'company':company,'direction':direction,'entry_price':entry,'hold_days':h,'exit_date':target_date(d,h),'score':float(c.get('score',0) or 0),'up_probability':float(c.get('up_probability',0) or 0),'down_probability':float(c.get('down_probability',0) or 0),'forced_entry':True}
        s['positions'].append(p)
        opened.append(h)

    # 1日枠が毎日成立していることを状態上も保証する。ただし、リスクポリシーで
    # 意図的に停止した場合や既に保有中の場合は例外としない。
    if 1 not in opened and 1 not in active_holds and 1 not in {h for h,_ in risk_blocked}:
        raise RuntimeError('1日枠が空いていないため、本日の毎日売買を成立できません')

    s['last_entry_date']=d
    s['trades_today']=len(opened)
    if not opened:
        blocked_text='・'.join(f'{h}日枠({reason})' for h,reason in risk_blocked)
        msg=f'⛔ 本日はリスクポリシーにより全枠エントリー見送り: {blocked_text}'
        print(msg); notify(msg)
        return
    opened_text='・'.join(f'{h}日' for h in opened)
    waiting=[h for h in HOLDS if h not in opened and h not in {rh for rh,_ in risk_blocked}]
    waiting_text=('｜保有中で待機: '+'・'.join(f'{h}日枠' for h in waiting)) if waiting else ''
    blocked_text=('｜リスク停止: '+'・'.join(f'{h}日枠' for h,_ in risk_blocked)) if risk_blocked else ''
    msg=(f'🔥 強制TOP1成立｜{company} ({ticker})｜{direction}\n'
         f'本日のTOP1を空いている枠へ登録: {opened_text}{waiting_text}{blocked_text}\n'
         f'Entry {entry:.2f}｜Score {float(c.get("score",0) or 0):.1f}｜UP {float(c.get("up_probability",0) or 0):.1f}%｜DOWN {float(c.get("down_probability",0) or 0):.1f}%')
    print(msg); notify(msg)

def main():
    s=load_state()
    # ★修正(2026-09): 当日分の決済を反映する前に当日開始資産を確定させる
    # (close_due()より後に呼ぶと、その日の決済損益が「当日開始」に混入する)。
    reset_daily_risk(s,today())
    closed=close_due(s)
    for p in closed: print(f"✅ {p['hold_days']}日決済 {p['ticker']} {p['return_pct']:+.2f}%")
    n=now(); minute=n.hour*60+n.minute
    if 570<=minute<930: open_daily(s)
    else: print(f'⏸ 取引時間外 {n:%H:%M JST}')
    save_state(s)
    print('📊 1日資産: ¥{:,.0f} | 3日資産: ¥{:,.0f} | 5日資産: ¥{:,.0f}'.format(*[s['capitals'][str(h)] for h in HOLDS]))
    print(f"📌 保有ポジション: {len(s['positions'])} | 本日売買バケット: {s.get('trades_today',0)}/3")

if __name__=='__main__': main()
