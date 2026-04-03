import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm

# Suppress warnings from arch/scipy during rolling estimates
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = Path(__file__).parent / "data"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"
OUTPUT_CSV = DATA_DIR / "rolling_var_es.csv"

WINDOW_SIZE = 1000
ALPHA = 0.99
ALPHA_ES = 0.975

# Quantiles for theoretical distributions
Z_VAR = stats.norm.ppf(ALPHA)
Z_ES  = stats.norm.ppf(ALPHA_ES) # Note: For Normal ES formula we use the matching ALPHA_ES quantile or just ALPHA? 
# Theory says ES_alpha uses alpha. If VaR is 99%, ES is usually also 99%, but 
# the prompt specifically says "1-daagse 99% Value-at-Risk (VaR) en 97,5% Expected Shortfall (ES)". 
# We need to make sure we use alpha=0.975 for ES and alpha=0.99 for VaR.
Z_ES_QUANTILE = stats.norm.ppf(ALPHA_ES)

def ewma_volatility(losses, lam=0.94):
    """
    Applies EWMA filter to a series of losses to find the variance forecast for T+1.
    Assumes mean is 0.
    """
    var = np.var(losses) # Initialize with sample variance
    for x in losses:
        var = lam * var + (1 - lam) * (x ** 2)
    return np.sqrt(var)

def compute_rolling_risk():
    print(f"Loading data from {LOSSES_CSV}...")
    df = pd.read_csv(LOSSES_CSV, index_col=0, parse_dates=True)
    losses = df["Loss"].values
    dates = df.index
    
    T = len(losses)
    if T <= WINDOW_SIZE:
        raise ValueError(f"Dataset length ({T}) is smaller than or equal to window size ({WINDOW_SIZE}).")
        
    print(f"Total days: {T}. Rolling window: {WINDOW_SIZE}. Iterations: {T - WINDOW_SIZE}")
    
    # Pre-allocate output arrays
    out_dates = dates[WINDOW_SIZE:]
    realized_loss = losses[WINDOW_SIZE:]
    
    # 1. Normal
    var_norm = np.zeros(T - WINDOW_SIZE)
    es_norm  = np.zeros(T - WINDOW_SIZE)
    
    # 2. Student-t
    var_t = np.zeros(T - WINDOW_SIZE)
    es_t  = np.zeros(T - WINDOW_SIZE)
    
    # 3. Historical Simulation
    var_hs = np.zeros(T - WINDOW_SIZE)
    es_hs  = np.zeros(T - WINDOW_SIZE)
    
    # 4. EWMA
    var_ewma = np.zeros(T - WINDOW_SIZE)
    es_ewma  = np.zeros(T - WINDOW_SIZE)
    
    # 5. GARCH / FHS
    var_fhs = np.zeros(T - WINDOW_SIZE)
    es_fhs  = np.zeros(T - WINDOW_SIZE)

    # Convert alpha for formulas
    p_var = ALPHA
    p_es = ALPHA_ES
    
    # Pre-calculate Normal ES multiplier for Normal and EWMA
    # ES_alpha = mu + sigma * (phi(Z_alpha) / (1 - alpha))
    norm_es_mult = stats.norm.pdf(Z_ES_QUANTILE) / (1 - ALPHA_ES)

    print("Running rolling window estimations...")
    for t in tqdm(range(WINDOW_SIZE, T), desc="Rolling Window"):
        i = t - WINDOW_SIZE # index in our output arrays
        
        # Historical window data [t-WINDOW_SIZE, t-1]
        window = losses[t-WINDOW_SIZE:t]
        
        # 1. Normal VC
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        var_norm[i] = mu + sigma * Z_VAR
        es_norm[i] = mu + sigma * norm_es_mult
        
        # 2. Student-t
        # Fit univariate Student-t to the window losses
        df_t, loc_t, scale_t = stats.t.fit(window)
        t_var_q = stats.t.ppf(ALPHA, df_t)
        t_es_q = stats.t.ppf(ALPHA_ES, df_t)
        var_t[i] = loc_t + scale_t * t_var_q
        # ES Student-t formula
        term1 = stats.t.pdf(t_es_q, df_t) / (1 - ALPHA_ES)
        term2 = (df_t + t_es_q**2) / (df_t - 1)
        es_t[i] = loc_t + scale_t * term1 * term2
        
        # 3. Historical Simulation
        # Sort window
        sorted_window = np.sort(window)
        # VaR index k = n * (1 - alpha) from the top
        k_var = int(np.ceil(WINDOW_SIZE * (1 - ALPHA))) - 1
        var_hs[i] = sorted_window[-(k_var+1)]
        # ES index
        k_es = int(np.ceil(WINDOW_SIZE * (1 - ALPHA_ES))) - 1
        var_es_thresh = sorted_window[-(k_es+1)]
        # Average of losses strictly exceeding the ES threshold (or equal) => Empirical ES
        exceedances = window[window > var_es_thresh]
        if len(exceedances) == 0:
            es_hs[i] = var_hs[i] # Fallback if empty array
        else:
            es_hs[i] = np.mean(exceedances)
            
        # 4. EWMA
        sigma_ewma = ewma_volatility(window, lam=0.94)
        var_ewma[i] = sigma_ewma * Z_VAR # assuming mu=0
        es_ewma[i] = sigma_ewma * norm_es_mult # assuming mu=0
        
        # 5. GARCH / Filtered Historical Simulation (FHS)
        # We model the mean as 0 (or constant) and fit GARCH(1,1)
        # Rescale losses for optimizer stability (arch package prefers variance ~ 1.0)
        scale_garch = np.std(window)
        scaled_window = window / scale_garch
        
        am = arch_model(scaled_window, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        res = am.fit(disp='off')
        
        # Forecast var for T+1
        garch_forecast = res.forecast(horizon=1, align='origin')
        sigma_garch_scaled = np.sqrt(garch_forecast.variance.values[-1, 0])
        sigma_t1 = sigma_garch_scaled * scale_garch
        
        # Extract standardized residuals
        std_resid = res.resid / res.conditional_volatility
        # FHS VaR & ES using standard residuals
        sorted_resid = np.sort(std_resid)
        resid_var_99 = sorted_resid[-(int(np.ceil(WINDOW_SIZE * (1 - ALPHA))))]
        resid_var_975 = sorted_resid[-(int(np.ceil(WINDOW_SIZE * (1 - ALPHA_ES))))]
        
        resid_exceedances = std_resid[std_resid > resid_var_975]
        resid_es_975 = np.mean(resid_exceedances) if len(resid_exceedances) > 0 else resid_var_975
        
        var_fhs[i] = sigma_t1 * resid_var_99
        es_fhs[i]  = sigma_t1 * resid_es_975

    # Compile results
    results_df = pd.DataFrame({
        "Loss": realized_loss,
        "VaR_Normal": var_norm,
        "ES_Normal": es_norm,
        "VaR_t": var_t,
        "ES_t": es_t,
        "VaR_HS": var_hs,
        "ES_HS": es_hs,
        "VaR_EWMA": var_ewma,
        "ES_EWMA": es_ewma,
        "VaR_FHS": var_fhs,
        "ES_FHS": es_fhs,
    }, index=out_dates)
    
    results_df.to_csv(OUTPUT_CSV)
    print(f"\nRolling predictions stored successfully in {OUTPUT_CSV}")

if __name__ == "__main__":
    compute_rolling_risk()
