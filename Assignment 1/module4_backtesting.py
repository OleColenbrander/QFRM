import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "fixed_window_var_es.csv"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"
METRICS_CSV = DATA_DIR / "out_of_sample_metrics.csv"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_VAR = 0.99
p_var = 1.0 - ALPHA_VAR 

def load_data():
    preds = pd.read_csv(RESULTS_CSV, index_col=0, header=[0, 1, 2], parse_dates=True)
    losses = pd.read_csv(LOSSES_CSV, index_col=0, parse_dates=True)['Loss']
    
    # assign 1D series to 3-level MultiIndex dataframe (note the choice previously was for clear overview of individual assets)
    df = preds.copy()
    df[('Realized', 'Portfolio', 'Loss')] = losses
    df = df.dropna(subset=[('Realized', 'Portfolio', 'Loss')])
    
    return df, preds

def run_backtesting():
    df, original_preds = load_data()
    L = df[('Realized', 'Portfolio', 'Loss')]
    
    T_out = len(df)
    models = original_preds.columns.levels[0].tolist() 
    i_0 = T_out * p_var
    
    var_results = []
    es_results = []
    spaces_dict = {}
    
    fig_loss, axes_loss = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), sharex=True)
    if len(models) == 1:
        axes_loss = [axes_loss]

    for m_idx, m in enumerate(models):
        var_col = (m, 'Portfolio', 'VaR_99')
        es_col  = (m, 'Portfolio', 'ES_975')
        
        VaR = df[var_col]
        ES  = df[es_col]
        
        # 1 VaR Validation
        violations = L > VaR
        N = violations.sum()
        vpy = N / (T_out / 252.0)
        
        binom_res = stats.binomtest(k=int(N), n=T_out, p=p_var, alternative='two-sided')
        var_pval = binom_res.pvalue
        
        var_results.append({
            'Model': m,
            'Expected Hits': round(i_0, 2),
            'Actual Hits': N,
            'Hits / Year': round(vpy, 2),
            'Binomial p-value': round(var_pval, 4)
        })
        
        # 2 ES Validation 
        violating_days = df.index[violations]
        L_viol = L.loc[violating_days]
        ES_viol = ES.loc[violating_days]
        
        if N >= 2:
            K = (L_viol - ES_viol) / ES_viol
            t_res = stats.ttest_1samp(K, popmean=0)
            es_tstat = t_res.statistic
            es_pval = t_res.pvalue
        else:
            es_tstat = np.nan
            es_pval = np.nan
            
        es_results.append({
            'Model': m,
            'N': N,
            'Actual ES': round(L_viol.mean(), 2) if N > 0 else np.nan,
            'Expected ES': round(ES_viol.mean(), 2) if N > 0 else np.nan,
            'ES t-statistic': round(es_tstat, 4) if pd.notna(es_tstat) else "N/A",
            'ES p-value': round(es_pval, 4) if pd.notna(es_pval) else "N/A"
        })
        
        # 3    Spacings
        v_idx = np.where(violations)[0] 
        if len(v_idx) > 0:
            spacings = np.diff(np.concatenate(([0], v_idx)))
            spaces_dict[m] = spacings
        else:
            spaces_dict[m] = []
            
        # Plotting Loss vs VaR
        ax = axes_loss[m_idx]
        ax.plot(df.index, L, color='grey', label='Loss', linewidth=1)
        ax.plot(df.index, VaR, color='orange', label='VaR_99', linewidth=1.2)
        ax.plot(df.index, ES, color='blue', linestyle='--', label='ES_975', linewidth=1.2)
        
        if N > 0:
            ax.scatter(violating_days, L_viol, color='red', marker='o', label='Violation', zorder=5)
            
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f"{m} (Hits: {N})")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    fig_loss.tight_layout()
    fig_loss.savefig(PLOTS_DIR / 'loss_vs_var_es_fixed.png')
    plt.close(fig_loss)

    # 4      QQ Plots for Spacings
    scale_param = 1.0 / p_var
    fig_qq, axes_qq = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    if len(models) == 1:
        axes_qq = [axes_qq]
        
    for i, m in enumerate(models):
        ax = axes_qq[i]
        spaces = spaces_dict.get(m, [])
        n_spaces = len(spaces)
        if n_spaces > 0:
            sorted_spaces = np.sort(spaces)
            p_emp = np.arange(1, n_spaces + 1) / (n_spaces + 1)
            q_theor = stats.expon.ppf(p_emp, scale=scale_param)
            
            ax.scatter(q_theor, sorted_spaces, color='blue', alpha=0.6)
            mx = max(max(q_theor), max(sorted_spaces)) if n_spaces > 0 else 100
            ax.plot([0, mx], [0, mx], color='red', linestyle='--')
        
        ax.set_title(m)
        ax.set_xlabel('Theoretical Quantiles (Exp)')
        if i == 0:
            ax.set_ylabel('Empirical Spacings')
        ax.grid(True, alpha=0.3)

    fig_qq.tight_layout()
    fig_qq.savefig(PLOTS_DIR / 'spacings_qq_fixed.png')
    plt.close(fig_qq)

    
    df_var = pd.DataFrame(var_results)
    df_es = pd.DataFrame(es_results)
    
    print("=" * 60)
    print("TABLE 1: VaR Statistics (Binomial Test)")
    print("=" * 60)
    print(df_var.to_string(index=False))
    print("\n")
    print("=" * 60)
    print("TABLE 2: ES Statistics (1-Sample T-test on K-residuals)")
    print("=" * 60)
    print(df_es.to_string(index=False))
    print("\nPlots saved to 'data/plots/'.")
    
    # veilig opslaagn
    df_var.set_index('Model', inplace=True)
    df_es.set_index('Model', inplace=True)
    combined = df_var.join(df_es.drop(columns=['N']))
    combined.to_csv(METRICS_CSV)
    print(f"Metrics saved to {METRICS_CSV.name}")

if __name__ == "__main__":
    run_backtesting()
