import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


df = pd.read_csv(os.path.join(DATA_DIR, 'rolling_var_es.csv'))
df['Date'] = pd.to_datetime(df['Date'])

alpha_var = 0.01 
T = len(df)

models = ['Normal', 't', 'HS', 'EWMA', 'FHS']
results = []
space_data = {}

for m in models:
    var_c = f'VaR_{m}'
    es_c = f'ES_{m}'
    
    # 1. Kupiec
    v = df['Loss'] > df[var_c]
    i_hat = v.sum()
    p = alpha_var
    i_0 = T * p
    
    var_z = T * p * (1 - p)
    z_stat = (i_hat - i_0) / np.sqrt(var_z)
    pval = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    # 2. Spaces
    v_idx = df.index[v].tolist()
    if len(v_idx) > 0: space_data[m] = np.diff([0] + v_idx)
    else: space_data[m] = []
        
    # 3. ES
    L_violations = df.loc[v, 'Loss']
    es_v = df.loc[v, es_c]
    
    if i_hat > 1: 
        K = (L_violations - es_v) / es_v
        Kmean = K.mean()
        Kstd = K.std(ddof=1) 
        Kse = Kstd / np.sqrt(i_hat)
        
        tstat = Kmean / Kse if Kse > 0 else np.nan
        ptst = 2 * (1 - stats.t.cdf(np.abs(tstat), df=i_hat-1))
        
        aes = L_violations.mean()
        pes = es_v.mean()
    else:
        tstat, ptst, aes, pes = np.nan, np.nan, np.nan, np.nan
        
    # years
    vpy = i_hat / (T / 252)
        
    results.append({
        'Model': m,
        'Expected Violations': round(i_0, 2),
        'Actual Violations': i_hat,
        'Violations/Year': round(vpy, 2),
        'VaR Z-stat': round(z_stat, 4),
        'VaR p-value': round(pval, 4),
        'Expected ES (viols)': round(pes, 2) if not np.isnan(pes) else np.nan,
        'Actual ES (viols)': round(aes, 2) if not np.isnan(aes) else np.nan,
        'ES t-stat': round(tstat, 4) if not np.isnan(tstat) else np.nan,
        'ES p-value': round(ptst, 4) if not np.isnan(ptst) else np.nan
    })

res = pd.DataFrame(results)
print(res.to_string())

csv_path = os.path.join(DATA_DIR, 'backtest_results.csv')
res.to_csv(csv_path, index=False)

# Make plots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, m in enumerate(models):
    ax = axes[i]
    spaces = space_data.get(m, [])
    if len(spaces) > 0:
        n = len(spaces)
        spaces = np.sort(spaces)
        p_emp = np.arange(1, n + 1) / (n + 1) 
        q = stats.expon.ppf(p_emp, scale=1/alpha_var)
        
        ax.scatter(q, spaces, color='blue', alpha=0.5)
        
        mx = max(q[-1] if n>0 else 0, spaces[-1] if n>0 else 0)
        ax.plot([0, mx], [0, mx], color='red', linestyle='--')
        ax.set_title(m)
        ax.grid(True)
    else:
        ax.set_title(m)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'spacings_qq_plots.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Loss'], label='Loss', color='gray', linewidth=1)
plt.plot(df['Date'], df['VaR_FHS'], color='orange')
plt.plot(df['Date'], df['ES_FHS'], color='red')
plt.plot(df['Date'], df['VaR_Normal'], color='blue', linestyle='--')

fhs_v = df['Loss'] > df['VaR_FHS']
if fhs_v.any():
    plt.scatter(df.loc[fhs_v, 'Date'], df.loc[fhs_v, 'Loss'], color='red', s=20)

plt.axhline(0, color='black')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_vs_var_es.png'))
plt.close()
