"""백테스트 1125건에 대해 신호일 기준 5일 누적 외국인/기관 매수 데이터 수집 후 검증."""
import urllib.request as ureq
import re, time, json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import math

def fetch_frgn_history(ticker, max_page=50):
    """한 종목의 외국인/기관 일별 이력 (~3년치) — 모든 페이지 모음."""
    out = []
    for page in range(1, max_page + 1):
        url = f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
        try:
            req = ureq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with ureq.urlopen(req, timeout=10) as r:
                html = r.read().decode("cp949", errors="replace")
            trs = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
            page_rows = 0
            for tr in trs:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
                if len(cells) < 7: continue
                cleaned = [re.sub(r'<[^>]+>','', c).replace('\xa0','').replace(',','').strip() for c in cells]
                if cleaned and re.match(r'\d{4}\.\d{2}\.\d{2}', cleaned[0]):
                    try:
                        date_str = cleaned[0].replace('.','-')
                        organ = int(cleaned[5].replace('+','')) if cleaned[5] not in ('','-') else 0
                        foreign = int(cleaned[6].replace('+','')) if cleaned[6] not in ('','-') else 0
                        out.append({"date": date_str, "organ": organ, "foreign": foreign})
                        page_rows += 1
                    except (ValueError, IndexError):
                        pass
            if page_rows == 0:
                break  # 빈 페이지 → 끝
        except Exception:
            break
    return ticker, out

def collect_all(tickers, max_workers=15):
    print(f"[1] 수집: {len(tickers)} tickers, {max_workers} workers")
    t0 = time.time()
    history_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_frgn_history, tk): tk for tk in tickers}
        done = 0
        for fut in as_completed(futures):
            tk, hist = fut.result()
            history_map[tk] = hist
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(tickers)} ({time.time()-t0:.0f}s)")
    print(f"  완료: {len(history_map)} tickers, {time.time()-t0:.0f}s")
    return history_map

def get_5day_supply(history, signal_date_str):
    """signal_date 기준 직전 5거래일(또는 시그널일 포함 5일) 외국인/기관 누적."""
    if not history: return None
    sd = pd.to_datetime(signal_date_str)
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= sd].sort_values('date', ascending=False).head(5)
    if len(df) < 3: return None
    return {
        'foreigner_5d': int(df['foreign'].sum()),
        'organ_5d': int(df['organ'].sum()),
        'foreign_inst_combo_days': int(((df['foreign']>0)&(df['organ']>0)).sum()),
        'foreign_buy_days': int((df['foreign']>0).sum()),
        'organ_buy_days': int((df['organ']>0).sum()),
    }

def new_score_with_supply(row, supply):
    """새 시스템 entry_score 시뮬 (supply 포함)."""
    s = 0
    tier = row.get('tier',''); ip = row.get('is_prime', False)
    s += 22 if (tier=='CORE' and ip) else 15 if tier=='CORE' else 18 if (tier=='WATCH' and ip) else 10 if tier=='WATCH' else 5 if tier=='EARLY' else 0
    # K-NN 0 (백테스트 데이터 한계)
    # supply 30점
    sup_sc = 0
    if supply:
        f5 = supply.get('foreigner_5d', 0)
        if f5 >= 500_000: sup_sc += 8
        elif f5 >= 100_000: sup_sc += 6
        elif f5 > 0: sup_sc += 3
        o5 = supply.get('organ_5d', 0)
        if o5 >= 500_000: sup_sc += 7
        elif o5 >= 100_000: sup_sc += 5
        elif o5 > 0: sup_sc += 2
        c = supply.get('foreign_inst_combo_days', 0)
        if c >= 4: sup_sc += 8
        elif c >= 3: sup_sc += 5
        elif c >= 2: sup_sc += 3
        elif c >= 1: sup_sc += 1
        if supply.get('foreign_buy_days',0) >= 4: sup_sc += 4
        elif supply.get('foreign_buy_days',0) >= 3: sup_sc += 2
    else:
        sup_sc = 10
    s += sup_sc
    s += 8  # gap (보수적)
    if row.get('vc_confirmed'): s += 3
    if row.get('stage1_ok'): s += 3
    if row.get('retest'): s += 2
    if row.get('penalty',0) == 0: s += 4
    if row.get('regime') == 'BEAR': s -= 10
    elif row.get('regime') == 'NEUTRAL': s -= 3
    return max(0, min(100, s))

def new_action(s, regime='BULL'):
    if regime == 'BEAR':
        if s >= 80: return 'BUY'
        elif s >= 65: return 'SMALL_BUY'
        return 'HOLD'
    if s >= 75: return 'STRONG_BUY'
    if s >= 60: return 'BUY'
    if s >= 45: return 'SMALL_BUY'
    return 'HOLD'

def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return 0, 1.0
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(s1/n1 + s2/n2)
    if se == 0: return 0, 1.0
    t = (a.mean() - b.mean()) / se
    p = 2 * (1 - 0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return t, p

def main():
    df = pd.read_csv('backtest_history.csv')
    print(f"[0] 백테스트 데이터: {len(df)}건, {df['ticker'].nunique()} tickers")

    tickers = df['ticker'].astype(str).str.zfill(6).unique().tolist()
    history = collect_all(tickers, max_workers=15)

    print(f"[2] supply 매핑")
    supply_results = []
    for _, row in df.iterrows():
        tk = str(row['ticker']).zfill(6)
        sd = str(row['signal_date'])
        hist = history.get(tk)
        sup = get_5day_supply(hist, sd) if hist else None
        supply_results.append(sup)
    df['supply'] = supply_results
    n_with_supply = sum(1 for s in supply_results if s is not None)
    print(f"  supply 매핑된 행: {n_with_supply}/{len(df)}")

    print(f"[3] 새 시스템 entry_score (supply 포함) 계산")
    df['entry_new_supply'] = df.apply(lambda r: new_score_with_supply(r, r['supply']), axis=1)
    df['action_new_supply'] = df.apply(lambda r: new_action(r['entry_new_supply'], r.get('regime','BULL')), axis=1)

    print()
    print("=== 결과: NEW system (supply 포함) ===")
    for act in ['STRONG_BUY','BUY','SMALL_BUY','HOLD']:
        sub = df[df['action_new_supply']==act]
        if len(sub) == 0:
            print(f"  {act:11s} n=0")
            continue
        ret = sub['ret_20d'].mean()
        win = (sub['ret_20d']>0).mean() * 100
        p20 = sub['reached_20pct'].mean() * 100
        mdd = sub['mdd_60d'].mean()
        print(f"  {act:11s} n={len(sub):4d}  ret_20d={ret:+.2f}%  win={win:.1f}%  +20%={p20:.1f}%  MDD={mdd:.2f}%")

    print()
    # OLD system 비교 (supply 없음)
    def old_score(r):
        t=r.get('tier',''); ip=r.get('is_prime',False); s=0
        s += 35 if (t=='CORE' and ip) else 20 if t=='CORE' else 25 if (t=='WATCH' and ip) else 15 if t=='WATCH' else 5 if t=='EARLY' else 0
        s += 20 + (3 if r.get('vc_confirmed') else 0) + (4 if r.get('penalty',0)==0 else 0)
        return s
    def old_action(s):
        if s >= 85: return 'STRONG_BUY'
        if s >= 70: return 'BUY'
        if s >= 55: return 'SMALL_BUY'
        return 'HOLD'
    df['action_old'] = df.apply(old_score, axis=1).apply(old_action)
    print("=== 결과: OLD system (참고) ===")
    for act in ['STRONG_BUY','BUY','SMALL_BUY','HOLD']:
        sub = df[df['action_old']==act]
        if len(sub) == 0:
            print(f"  {act:11s} n=0")
            continue
        ret = sub['ret_20d'].mean()
        win = (sub['ret_20d']>0).mean() * 100
        mdd = sub['mdd_60d'].mean()
        print(f"  {act:11s} n={len(sub):4d}  ret_20d={ret:+.2f}%  win={win:.1f}%  MDD={mdd:.2f}%")

    print()
    # 통계 검정 — 새 BUY+ vs OLD BUY+
    new_buy = df[df['action_new_supply'].isin(['STRONG_BUY','BUY','SMALL_BUY'])]
    old_buy = df[df['action_old'].isin(['STRONG_BUY','BUY','SMALL_BUY'])]
    print(f"[4] 통계 검정: NEW(supply) BUY+ n={len(new_buy)}, OLD BUY+ n={len(old_buy)}")
    t,p = welch(new_buy['ret_20d'], old_buy['ret_20d'])
    print(f"  ret_20d  NEW={new_buy['ret_20d'].mean():.2f} OLD={old_buy['ret_20d'].mean():.2f}  t={t:.2f} p={p:.4f}  sig={p<0.05}")
    t,p = welch(new_buy['mdd_60d'], old_buy['mdd_60d'])
    print(f"  MDD      NEW={new_buy['mdd_60d'].mean():.2f} OLD={old_buy['mdd_60d'].mean():.2f}  t={t:.2f} p={p:.4f}  sig={p<0.05}")

    # 최강 등급(STRONG_BUY) 검정
    new_strong = df[df['action_new_supply']=='STRONG_BUY']
    if len(new_strong) >= 5:
        baseline = df[~df['action_new_supply'].isin(['STRONG_BUY'])]
        t,p = welch(new_strong['ret_20d'], baseline['ret_20d'])
        print(f"  NEW STRONG_BUY ret={new_strong['ret_20d'].mean():.2f} vs 나머지={baseline['ret_20d'].mean():.2f}  t={t:.2f} p={p:.4f}")

    df[['signal_date','ticker','name','entry_new_supply','action_new_supply','ret_20d','mdd_60d']].to_csv('supply_validation.csv', index=False)
    print("\n저장: supply_validation.csv")

if __name__ == "__main__":
    main()
