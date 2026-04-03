import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Base directory for Assignment 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting Backtest Analysis (Module 5)...")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'rolling_var_es.csv'))
df['Date'] = pd.to_datetime(df['Date'])

alpha_var = 0.01 # 99% VaR, tail probability 0.01
T = len(df)

models = ['Normal', 't', 'HS', 'EWMA', 'FHS']
results = []
spacing_data = {}

for model in models:
    var_col = f'VaR_{model}'
    es_col = f'ES_{model}'
    
    # --- 1. VaR Binomial Test (Kupiec) ---
    # Indicator: 1 if Loss > VaR
    violations = df['Loss'] > df[var_col]
    I_hat = violations.sum()
    p = alpha_var
    I_0 = T * p
    
    # Z-test statistic
    var_Z = T * p * (1 - p)
    z_stat = (I_hat - I_0) / np.sqrt(var_Z)
    p_val_binom = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    # --- 2. Spacing Test ---
    # Find indices of violations
    violation_indices = df.index[violations].tolist()
    if len(violation_indices) > 0:
        spacings = np.diff([0] + violation_indices)
        spacing_data[model] = spacings
    else:
        spacing_data[model] = []
        
    # --- 3. Expected Shortfall Test (Rescaled Residuals) ---
    loss_violations = df.loc[violations, 'Loss']
    es_violations = df.loc[violations, es_col]
    
    if I_hat > 1: # We need at least 2 observations to calculate sample stdev
        # rescaled residuals K_t = (L_t - ES_t) / ES_t (assuming mu = 0)
        K = (loss_violations - es_violations) / es_violations
        K_mean = K.mean()
        K_std = K.std(ddof=1) # Sample standard deviation
        K_se = K_std / np.sqrt(I_hat)
        
        t_stat = K_mean / K_se if K_se > 0 else np.nan
        p_val_t = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=I_hat-1))
        
        actual_es = loss_violations.mean()
        pred_es = es_violations.mean()
    else:
        t_stat, p_val_t, actual_es, pred_es = np.nan, np.nan, np.nan, np.nan
        
    # --- 4. Violations per year ---
    # 252 trading days per year
    viols_per_year = I_hat / (T / 252)
        
    results.append({
        'Model': model,
        'Expected Violations': round(I_0, 2),
        'Actual Violations': I_hat,
        'Violations/Year': round(viols_per_year, 2),
        'VaR Z-stat': round(z_stat, 4),
        'VaR p-value': round(p_val_binom, 4),
        'Expected ES (viols)': round(pred_es, 2) if not np.isnan(pred_es) else np.nan,
        'Actual ES (viols)': round(actual_es, 2) if not np.isnan(actual_es) else np.nan,
        'ES t-stat': round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
        'ES p-value': round(p_val_t, 4) if not np.isnan(p_val_t) else np.nan
    })

# Form a DataFrame for reporting
results_df = pd.DataFrame(results)

# Print nice table to console
print("\n" + "="*95)
print(" "*35 + "BACKTESTING RESULTS")
print("="*95)
print(results_df.to_string(index=False))
print("="*95 + "\n")

# Save to CSV
results_csv_path = os.path.join(DATA_DIR, 'backtest_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to: {results_csv_path}")

# ==========================================
# Graphical Outputs
# ==========================================

print("Generating plots...")

# 1. Spacing QQ-Plots (1x5 Grid as requested)
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Spacing Test QQ-Plots vs Theoretical Exponential Distribution (Exp(λ=0.01))', fontsize=16, y=1.05)

for i, model in enumerate(models):
    ax = axes[i]
    spacings = spacing_data.get(model, [])
    if len(spacings) > 0:
        n = len(spacings)
        sorted_spacings = np.sort(spacings)
        # Empirical CDF probabilities
        p_empirical = np.arange(1, n + 1) / (n + 1) 
        # Theoretical quantiles
        theoretical_quantiles = stats.expon.ppf(p_empirical, scale=1/alpha_var)
        
        ax.scatter(theoretical_quantiles, sorted_spacings, color='blue', alpha=0.7, edgecolor='black')
        
        # Reference 45 degree line
        mx = max(theoretical_quantiles[-1] if n>0 else 0, sorted_spacings[-1] if n>0 else 0)
        ax.plot([0, mx], [0, mx], color='red', linestyle='--', label='Y=X (Perfect Match)')
        
        ax.set_title(f'{model} Spacings')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Spacings (Days)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
    else:
        ax.set_title(f'{model}\n(No Violations)')

plt.tight_layout()
spacings_plot_path = os.path.join(PLOTS_DIR, 'spacings_qq_plots.png')
plt.savefig(spacings_plot_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved {spacings_plot_path}")

# 2. Daily Loss vs predictions Plot
plt.figure(figsize=(15, 7))

# Plot P&L (Loss is inverted, but VaR is positive)
plt.plot(df['Date'], df['Loss'], label='Realized Loss ($L_t$)', color='gray', alpha=0.6, linewidth=1)

# Plot FHS VaR & ES
plt.plot(df['Date'], df['VaR_FHS'], label='VaR (99%) - GARCH FHS', color='orange', linewidth=1.5)
plt.plot(df['Date'], df['ES_FHS'], label='ES (97.5%) - GARCH FHS', color='red', linewidth=1.5)

# Plot Normal VaR as a static comparison (it changes slowly)
plt.plot(df['Date'], df['VaR_Normal'], label='VaR (99%) - Normal Variance-Covariance', color='blue', linewidth=1.5, linestyle='--')

# Highlight violations for FHS
fhs_violations = df['Loss'] > df['VaR_FHS']
if fhs_violations.any():
    plt.scatter(df.loc[fhs_violations, 'Date'], df.loc[fhs_violations, 'Loss'], 
                color='darkred', s=40, zorder=5, label='FHS Violation', edgecolors='white')

plt.axhline(0, color='black', linewidth=0.8)
plt.title('Out-of-Sample Portfolio Loss vs VaR & ES Predictions', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Loss (USD)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
loss_vs_var_plot_path = os.path.join(PLOTS_DIR, 'loss_vs_var_es.png')
plt.savefig(loss_vs_var_plot_path, dpi=300)
plt.close()
print(f"  Saved {loss_vs_var_plot_path}")

print("Module 5 completely executed.")
