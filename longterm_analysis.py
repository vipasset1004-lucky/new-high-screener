"""백테스트 1125건 → 1년/2년/전체 추적. 진짜 5배 후보 식별 알고리즘."""
import pandas as pd, numpy as np, math, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from new_high_screener import fetch_daily
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=" * 70)
bt = pd.read_csv('backtest_history.csv')
print(f"신호 {len(bt)}건, unique tickers={bt['ticker'].nunique()}")

tickers = bt['ticker'].astype(str).str.zfill(6).unique().tolist()
print(f"\n[1] {len(tickers)} tickers OHLCV 병렬 fetch...")
df_map = {}
def _fetch(tk):
    try:
        return tk, fetch_daily(tk, is_korean=True)
    except Exception:
        return tk, None
with ThreadPoolExecutor(max_workers=15) as ex:
    futures = {ex.submit(_fetch, tk): tk for tk in tickers}
    for fut in as_completed(futures):
        tk, df = fut.result()
        if df is not None:
            df_map[tk] = df
print(f"  -> fetched {len(df_map)}")

print("\n[2] 각 신호일별 장기 metric 계산...")
results = []
for _, row in bt.iterrows():
    tk = str(row['ticker']).zfill(6)
    sd = pd.to_datetime(row['signal_date'])
    df = df_map.get(tk)
    if df is None: continue
    after = df[df.index >= sd]
    if len(after) < 60: continue
    entry = float(after['Close'].iloc[0])
    n_after = len(after)

    def at(n):
        if n_after < n: return None
        return (after['Close'].iloc[n-1]/entry - 1) * 100

    def peak_until(n):
        if n_after < n: return None
        return (after['High'].iloc[:n].max()/entry - 1) * 100

    peak_all = (after['High'].max()/entry - 1) * 100

    results.append({
        'signal_date': sd.date(), 'ticker': tk, 'name': row['name'],
        'days_avail': n_after,
        'ret_60d': at(60), 'ret_120d': at(120), 'ret_252d': at(252), 'ret_504d': at(504),
        'peak_60d': peak_until(60), 'peak_120d': peak_until(120),
        'peak_252d': peak_until(252), 'peak_504d': peak_until(504),
        'peak_all': peak_all,
        'reach_50':  bool((after['High'] >= entry * 1.5).any()),
        'reach_100': bool((after['High'] >= entry * 2.0).any()),
        'reach_200': bool((after['High'] >= entry * 3.0).any()),
        'reach_500': bool((after['High'] >= entry * 6.0).any()),
        # 원본 신호
        'score': row['score'], 'grade': row['grade'], 'tier': row['tier'],
        'is_prime': row['is_prime'], 'type': row['type'],
        'rvol': row['rvol'], 'rsi': row['rsi'], 'regime': row['regime'],
        'penalty': row['penalty'], 'is_ath': row['is_ath'],
        'first_breakout': row['first_breakout'], 'retest': row['retest'],
        'stage1_ok': row['stage1_ok'], 'vc_confirmed': row['vc_confirmed'], 'killer': row['killer'],
    })

df_lt = pd.DataFrame(results)
df_lt.to_csv('backtest_longterm.csv', index=False)
print(f"  -> saved: backtest_longterm.csv ({len(df_lt)} rows)")
print()
print(f"  보유기간 평균 {df_lt['days_avail'].mean():.0f}일 ({df_lt['days_avail'].mean()/252:.1f}년)")
print(f"  보유기간 최대 {df_lt['days_avail'].max()}일 ({df_lt['days_avail'].max()/252:.1f}년)")

# === 분석 ===
print()
print("=" * 70)
print("[3] Distribution of long-term outcomes (any peak in available period)")
print(f"  reached +50%:  {df_lt['reach_50'].mean()*100:5.1f}%")
print(f"  reached +100%: {df_lt['reach_100'].mean()*100:5.1f}%")
print(f"  reached +200%: {df_lt['reach_200'].mean()*100:5.1f}%")
print(f"  reached +500%: {df_lt['reach_500'].mean()*100:5.1f}%")

print()
print("[4] 1년/2년 시점별 평균 수익률 (관측 가능한 행만)")
for col in ['ret_60d', 'ret_120d', 'ret_252d', 'ret_504d']:
    sub = df_lt[df_lt[col].notna()]
    print(f"  {col:10s}: n={len(sub):4d}  avg={sub[col].mean():+6.1f}%  median={sub[col].median():+6.1f}%")

for col in ['peak_60d', 'peak_120d', 'peak_252d', 'peak_504d']:
    sub = df_lt[df_lt[col].notna()]
    print(f"  {col:10s}: n={len(sub):4d}  avg_peak={sub[col].mean():+6.1f}%  median_peak={sub[col].median():+6.1f}%")

# === 신호별 +200% / +500% 도달율 ===
print()
print("=" * 70)
print("[5] 신호별 +200% / +500% 도달율 (lift over baseline)")
base200 = df_lt['reach_200'].mean()*100
base500 = df_lt['reach_500'].mean()*100
print(f"baseline: +200%={base200:.1f}%, +500%={base500:.1f}%")
print()
print(f"{'signal':40s} n      +100%   +200%   +500%   lift_200  lift_500")
print("-" * 90)
def rep(name, mask):
    sub = df_lt[mask]
    if len(sub) < 15: return
    p100 = sub['reach_100'].mean()*100
    p200 = sub['reach_200'].mean()*100
    p500 = sub['reach_500'].mean()*100
    print(f"  {name:40s} {len(sub):4d}   {p100:5.1f}%  {p200:5.1f}%  {p500:5.1f}%   {p200-base200:+5.1f}pp  {p500-base500:+5.1f}pp")

# 단일 신호
rep('PRIME (is_prime=True)',           df_lt['is_prime']==True)
rep('grade=S (PRIME-S)',               df_lt['grade']=='S')
rep('is_ath',                          df_lt['is_ath']==True)
rep('vc_confirmed',                    df_lt['vc_confirmed']==True)
rep('first_breakout',                  df_lt['first_breakout']==True)
rep('killer (FB+RVOL2x)',              df_lt['killer']==True)
rep('retest',                          df_lt['retest']==True)
rep('stage1_ok',                       df_lt['stage1_ok']==True)
rep('penalty == 0',                    df_lt['penalty']==0)
rep('rvol >= 2.0',                     df_lt['rvol']>=2.0)
rep('rsi < 65',                        df_lt['rsi']<65)
rep('regime=BULL',                     df_lt['regime']=='BULL')
rep('type=PULLBACK',                   df_lt['type']=='PULLBACK_REBREAK')
rep('type=TREND',                      df_lt['type']=='TREND_CONTINUE')
rep('score >= 80',                     df_lt['score']>=80)
print()

# 조합
print("=== 조합 (n>=20) ===")
combos = [
    ('PRIME + vc',                  (df_lt['is_prime']==True)&(df_lt['vc_confirmed']==True)),
    ('PRIME + ATH',                 (df_lt['is_prime']==True)&(df_lt['is_ath']==True)),
    ('PRIME + ATH + vc',            (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['vc_confirmed']==True)),
    ('PRIME + ATH + BULL',          (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['regime']=='BULL')),
    ('PRIME + ATH + vc + BULL',     (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['vc_confirmed']==True)&(df_lt['regime']=='BULL')),
    ('S grade + ATH + vc + BULL',   (df_lt['grade']=='S')&(df_lt['is_ath']==True)&(df_lt['vc_confirmed']==True)&(df_lt['regime']=='BULL')),
    ('PRIME + first_breakout',      (df_lt['is_prime']==True)&(df_lt['first_breakout']==True)),
    ('PRIME + killer',              (df_lt['is_prime']==True)&(df_lt['killer']==True)),
    ('PRIME + first_breakout + ATH',(df_lt['is_prime']==True)&(df_lt['first_breakout']==True)&(df_lt['is_ath']==True)),
    ('vc + ATH',                    (df_lt['vc_confirmed']==True)&(df_lt['is_ath']==True)),
    ('vc + first_breakout',         (df_lt['vc_confirmed']==True)&(df_lt['first_breakout']==True)),
    ('PRIME + pen=0 + PULLBACK',    (df_lt['is_prime']==True)&(df_lt['penalty']==0)&(df_lt['type']=='PULLBACK_REBREAK')),
    ('PRIME + pen=0 + ATH',         (df_lt['is_prime']==True)&(df_lt['penalty']==0)&(df_lt['is_ath']==True)),
    ('PRIME + ATH + first_breakout',(df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['first_breakout']==True)),
    ('PRIME + ATH + first_breakout + vc',(df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['first_breakout']==True)&(df_lt['vc_confirmed']==True)),
]
for name, mask in combos:
    rep(name, mask)

# 통계 검정
print()
print("=" * 70)
print("[6] 통계 검정 — 최강 조합 vs baseline")
def proportion_test(p1, n1, p2, n2):
    pp = (p1*n1+p2*n2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    z = (p2-p1)/se if se>0 else 0
    pv = 2*(1 - 0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return z, pv

# 최강 조합 자동 탐색 — top n by +200% rate
strong_signals = []
for sig_name, mask in [
    ('PRIME', df_lt['is_prime']==True),
    ('PRIME+vc', (df_lt['is_prime']==True)&(df_lt['vc_confirmed']==True)),
    ('PRIME+ATH+vc', (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['vc_confirmed']==True)),
    ('PRIME+ATH+BULL', (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['regime']=='BULL')),
    ('PRIME+ATH+first_breakout+vc', (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['first_breakout']==True)&(df_lt['vc_confirmed']==True)),
    ('PRIME+ATH+vc+BULL', (df_lt['is_prime']==True)&(df_lt['is_ath']==True)&(df_lt['vc_confirmed']==True)&(df_lt['regime']=='BULL')),
]:
    sub = df_lt[mask]
    if len(sub) < 15: continue
    p200 = sub['reach_200'].mean()
    base = df_lt[~mask]
    base200_ = base['reach_200'].mean()
    z, pv = proportion_test(base200_, len(base), p200, len(sub))
    print(f"  {sig_name:35s} n={len(sub):4d}  +200%={p200*100:.1f}% vs base {base200_*100:.1f}%  z={z:.2f} p={pv:.4f} sig={pv<0.05}")
