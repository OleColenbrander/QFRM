import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
DATA_DIR  = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

ALPHA_VAR = 0.01
MODELS    = ['Normal', 't', 'HS', 'EWMA', 'FHS']


def run_backtesting():
    df = pd.read_csv(os.path.join(DATA_DIR, 'rolling_var_es.csv'))
    df['Date'] = pd.to_datetime(df['Date'])

    T       = len(df)
    p       = ALPHA_VAR
    results = []
    space_data = {}

    for m in MODELS:
        var_c = f'VaR_{m}'
        es_c  = f'ES_{m}'

        # 1. Kupiec POF
        v    = df['Loss'] > df[var_c]
        i_hat = v.sum()
        i_0   = T * p
        z_stat = (i_hat - i_0) / np.sqrt(T * p * (1 - p))
        pval   = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # 2. Violation spacings (for dependence QQ plot)
        v_idx = df.index[v].tolist()
        space_data[m] = np.diff([0] + v_idx) if len(v_idx) > 0 else []

        # 3. ES test
        L_violations = df.loc[v, 'Loss']
        es_v         = df.loc[v, es_c]

        if i_hat > 1:
            K     = (L_violations - es_v) / es_v
            Kmean = K.mean()
            Kse   = K.std(ddof=1) / np.sqrt(i_hat)
            tstat = Kmean / Kse if Kse > 0 else np.nan
            ptst  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=i_hat - 1))
            aes   = L_violations.mean()
            pes   = es_v.mean()
        else:
            tstat, ptst, aes, pes = np.nan, np.nan, np.nan, np.nan

        vpy = i_hat / (T / 252)
        results.append({
            'Model':                 m,
            'Expected Violations':   round(i_0, 2),
            'Actual Violations':     i_hat,
            'Violations/Year':       round(vpy, 2),
            'VaR Z-stat':            round(z_stat, 4),
            'VaR p-value':           round(pval, 4),
            'Expected ES (viols)':   round(pes, 2) if not np.isnan(pes) else np.nan,
            'Actual ES (viols)':     round(aes, 2) if not np.isnan(aes) else np.nan,
            'ES t-stat':             round(tstat, 4) if not np.isnan(tstat) else np.nan,
            'ES p-value':            round(ptst, 4) if not np.isnan(ptst) else np.nan,
        })

    res = pd.DataFrame(results)
    print(res.to_string())
    res.to_csv(os.path.join(DATA_DIR, 'backtest_results.csv'), index=False)

    # ── Plot: violation spacing QQ plots ──────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, m in enumerate(MODELS):
        ax     = axes[i]
        spaces = space_data.get(m, [])
        if len(spaces) > 0:
            n      = len(spaces)
            spaces = np.sort(spaces)
            p_emp  = np.arange(1, n + 1) / (n + 1)
            q      = stats.expon.ppf(p_emp, scale=1 / ALPHA_VAR)
            mx     = max(q[-1], spaces[-1])
            ax.scatter(q, spaces, color='blue', alpha=0.5)
            ax.plot([0, mx], [0, mx], color='red', linestyle='--')
            ax.grid(True)
        ax.set_title(m)
        ax.set_xlabel("Theoretical quantile")
        ax.set_ylabel("Observed spacing" if i == 0 else "")

    fig.suptitle("Violation Spacing QQ Plots (exponential iid test)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'spacings_qq_plots.png'), dpi=150)
    plt.close()

    # ── Plot: realized loss vs. VaR/ES for all models ─────────────────
    colours = {"Normal": "#2196F3", "t": "#4CAF50", "HS": "#FF9800",
               "EWMA": "#9C27B0", "FHS": "#F44336"}
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(df['Date'], 0, df['Loss'], where=df['Loss'] > 0,
                    color='lightgray', alpha=0.6, label='Daily loss')
    for m in MODELS:
        ax.plot(df['Date'], df[f'VaR_{m}'], color=colours[m],
                linewidth=0.9, alpha=0.85, label=f'VaR {m}')

    fhs_v = df['Loss'] > df['VaR_FHS']
    if fhs_v.any():
        ax.scatter(df.loc[fhs_v, 'Date'], df.loc[fhs_v, 'Loss'],
                   color='red', s=18, zorder=4, label='FHS violation')

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title("Realized Loss vs. Rolling 99% VaR (all models)", fontweight="bold")
    ax.set_ylabel("Dollar Loss ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_vs_var_es.png'), dpi=150)
    plt.close()

    return res


if __name__ == "__main__":
    run_backtesting()
