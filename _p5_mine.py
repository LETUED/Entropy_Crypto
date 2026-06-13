import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\dglst\Desktop\dev\Entropy_Crypto\docs\ver_p5.csv')

# t-columns are ALREADY OBI-direction-adjusted % moves (positive = moved in signal direction).
# So "direction correct @5min" = t300s > 0.
# EV模型: 신호방향으로 t300s 만큼 움직였다고 가정(부호조정됨).
# 왕복 수수료: Maker 0.08%, Taker 0.20%. t컬럼 단위는 %.
TARGET = 't300s'
FEE_MAKER = 0.08
FEE_TAKER = 0.20

df = df[df[TARGET].notna()].copy()
df['hour'] = df['time'].str.slice(0, 2)

# entropy buckets
def ent_bucket(e):
    if 0.45 <= e < 0.6: return 'E[0.45-0.6)'
    if 0.6 <= e < 0.7:  return 'E[0.6-0.7)'
    if 0.7 <= e <= 0.75: return 'E[0.7-0.75]'
    return 'E_other'
df['ent_b'] = df['entropy'].apply(ent_bucket)

# tps buckets (quantile-ish fixed thresholds)
def tps_bucket(t):
    if t < 12: return 'TPS[<12)'
    if t < 16: return 'TPS[12-16)'
    if t < 22: return 'TPS[16-22)'
    return 'TPS[>=22)'
df['tps_b'] = df['tps'].apply(tps_bucket)

# signal_duration buckets
def dur_bucket(d):
    if d < 6: return 'DUR[<6)'
    if d < 9: return 'DUR[6-9)'
    return 'DUR[>=9)'
df['dur_b'] = df['signal_duration_s'].apply(dur_bucket)

print("TPS quantiles:", df['tps'].quantile([0,.25,.5,.75,1]).round(2).tolist())
print("DUR quantiles:", df['signal_duration_s'].quantile([0,.25,.5,.75,1]).round(2).tolist())
print("Entropy buckets:\n", df['ent_b'].value_counts())

def stats(sub):
    n = len(sub)
    if n == 0: return None
    x = sub[TARGET]
    wr = (x > 0).mean() * 100
    p50 = x.median()
    p90 = x.quantile(0.90)
    mean_move = x.mean()
    ev_maker = mean_move - FEE_MAKER
    ev_taker = mean_move - FEE_TAKER
    return dict(n=n, WR=round(wr,1), p50=round(p50,4), p90=round(p90,4),
                mean=round(mean_move,4), EVm=round(ev_maker,4), EVt=round(ev_taker,4))

# ---- BASELINE ----
print("\n=== BASELINE (all) ===")
print(stats(df))

# ---- SINGLE FILTERS ----
print("\n=== SINGLE FILTER GRID ===")
single_rows = []
for dim in ['hour','symbol','ent_b','tps_b','dur_b']:
    for val, sub in df.groupby(dim):
        s = stats(sub)
        s['dim'] = dim; s['val'] = str(val)
        single_rows.append(s)
sdf = pd.DataFrame(single_rows).sort_values('EVm', ascending=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(sdf[['dim','val','n','WR','p50','p90','mean','EVm','EVt']].to_string(index=False))

# ---- DOUBLE FILTERS ----
print("\n=== DOUBLE FILTER GRID (n>=8 shown, sorted EVm) ===")
dims = ['hour','symbol','ent_b','tps_b','dur_b']
double_rows = []
from itertools import combinations
for d1, d2 in combinations(dims, 2):
    for (v1, v2), sub in df.groupby([d1, d2]):
        s = stats(sub)
        if s is None: continue
        s['combo'] = f"{d1}={v1} & {d2}={v2}"
        s['n_raw'] = s['n']
        double_rows.append(s)
ddf = pd.DataFrame(double_rows)
ddf_show = ddf[ddf['n'] >= 8].sort_values('EVm', ascending=False)
print(ddf_show[['combo','n','WR','p50','p90','mean','EVm','EVt']].head(30).to_string(index=False))

print("\n=== TOP positive-EVm DOUBLE combos n>=20 (overfit-safe) ===")
safe = ddf[ddf['n'] >= 20].sort_values('EVm', ascending=False)
print(safe[['combo','n','WR','p50','p90','mean','EVm','EVt']].head(15).to_string(index=False))

print("\n=== Count combos with positive EVm by sample size ===")
for thr in [8, 15, 20, 30]:
    sub = ddf[ddf['n'] >= thr]
    pos = (sub['EVm'] > 0).sum()
    print(f"  n>={thr}: {len(sub)} combos, {pos} have EVm>0")

# also single best n>=20
print("\n=== SINGLE combos n>=20 sorted EVm ===")
print(sdf[sdf['n']>=20][['dim','val','n','WR','p50','p90','mean','EVm','EVt']].to_string(index=False))
