#!/usr/bin/env python3
"""Forced daily TOP1 paper trading with independent 1/3/5 business-day hold buckets.

Every business day after 09:30 exactly one TOP1 is selected and all three buckets
(1d, 3d, 5d) enter the SAME ticker/direction. No post-selection gate may cancel
the entry. Each bucket is independent and exits at its target business-day close.
"""
import json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import requests
import paper_fast_entrypoint as paper

TZ=ZoneInfo('Asia/Tokyo')
STATE_FILE='multi_hold_paper_state.json'
HISTORY_FILE='multi_hold_paper_history.csv'
INITIAL_CAPITAL=float(os.getenv('AI_INITIAL_CAPITAL','1000000'))
HOLDS=(1,3,5)
WEBHOOK_URL=os.getenv('DISCORD_WEBHOOK')
MARKET_CLOSE_MINUTES=15*60+35

def now(): return datetime.now(TZ)
def today(): return now().date().isoformat()

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            s=json.load(open(STATE_FILE,encoding='utf-8'))
            s.setdefault('capitals',{str(h):INITIAL_CAPITAL for h in HOLDS})
            s.setdefault('positions',[]); s.setdefault('trades_today',0); s.setdefault('last_entry_date',None)
            return s
        except Exception: pass
    return {'capitals':{str(h):INITIAL_CAPITAL for h in HOLDS},'positions':[],'trades_today':0,'last_entry_date':None}

def save_state(s):
    tmp=STATE_FILE+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f: json.dump(s,f,ensure_ascii=False,indent=2)
    os.replace(tmp,STATE_FILE)

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
        ret=sign*(exit_price/entry-1.0)
        h=str(p['hold_days'])
        s['capitals'][h]=float(s['capitals'].get(h,INITIAL_CAPITAL))*(1+ret)
        closed.append({**p,'exit_price':exit_price,'return_pct':ret*100,'status':'CLOSED','exit_reason':'TARGET_CLOSE'})
    s['positions']=remaining
    append(closed)
    return closed

def choose_top1():
    policy=paper.loop.app.load_policy()
    candidates,_=paper.scan_progressive_with_prefilter(policy)
    if not candidates: raise RuntimeError('TOP1候補が生成されませんでした')
    return candidates[0]

def open_daily(s):
    d=today()
    if s.get('last_entry_date')==d:
        print('⏭ 本日の強制TOP1売買は既に成立済み')
        return
    if now().weekday()>=5:
        print('⏭ 土日なので取引しません')
        return
    c=choose_top1()
    ticker=str(c.get('ticker') or c.get('symbol') or '')
    if not ticker: raise RuntimeError('TOP1 ticker missing')
    direction=str(c.get('direction','BUY')).upper()
    if direction not in ('BUY','SHORT'): direction='BUY'
    df=daily_bars(ticker)
    if df is None or df.empty: raise RuntimeError(f'{ticker}: price download failed')
    entry=float(df.iloc[-1]['Close'])
    company=c.get('company') or c.get('name') or ticker
    for h in HOLDS:
        s['positions'].append({'entry_date':d,'ticker':ticker,'company':company,'direction':direction,'entry_price':entry,'hold_days':h,'exit_date':target_date(d,h),'score':float(c.get('score',0) or 0),'up_probability':float(c.get('up_probability',0) or 0),'down_probability':float(c.get('down_probability',0) or 0),'forced_entry':True})
    s['last_entry_date']=d; s['trades_today']=3
    msg=(f'🔥 強制TOP1成立｜{company} ({ticker})｜{direction}\n'
         f'同一銘柄を1日・3日・5日の3バケットへ必ず登録\n'
         f'Entry {entry:.2f}｜Score {float(c.get("score",0) or 0):.1f}｜UP {float(c.get("up_probability",0) or 0):.1f}%｜DOWN {float(c.get("down_probability",0) or 0):.1f}%')
    print(msg); notify(msg)

def main():
    s=load_state()
    closed=close_due(s)
    for p in closed: print(f"✅ {p['hold_days']}日決済 {p['ticker']} {p['return_pct']:+.2f}%")
    n=now(); minute=n.hour*60+n.minute
    if 570<=minute<930: open_daily(s)
    else: print(f'⏸ 取引時間外 {n:%H:%M JST}')
    save_state(s)
    print('📊 1日資産: ¥{:,.0f} | 3日資産: ¥{:,.0f} | 5日資産: ¥{:,.0f}'.format(*[s['capitals'][str(h)] for h in HOLDS]))
    print(f"📌 保有ポジション: {len(s['positions'])} | 本日売買バケット: {s.get('trades_today',0)}/3")

if __name__=='__main__': main()
