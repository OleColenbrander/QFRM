import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"
OUTPUT_CSV = DATA_DIR / "rolling_var_es.csv"


WINDOW_SIZE = 1000
ALPHA = 0.99
ALPHA_ES = 0.975

# gets normal q values
Z_VAR = stats.norm.ppf(ALPHA)
Z_ES_QUANTILE = stats.norm.ppf(ALPHA_ES)


def ewma_volatility(losses, lam=0.94):
    var = np.var(losses) 
    for x in losses:
        var = lam * var + (1 - lam) * (x ** 2)
    return np.sqrt(var)

def compute_rolling_risk():
    df = pd.read_csv(LOSSES_CSV, index_col=0, parse_dates=True)
    losses = df["Loss"].values
    dates = df.index
    
    T = len(losses)
    
    out_dates = dates[WINDOW_SIZE:]
    realized_loss = losses[WINDOW_SIZE:]
    
    var_norm = np.zeros(T - WINDOW_SIZE)
    es_norm  = np.zeros(T - WINDOW_SIZE)
    var_t = np.zeros(T - WINDOW_SIZE)
    es_t  = np.zeros(T - WINDOW_SIZE)
    var_hs = np.zeros(T - WINDOW_SIZE)
    es_hs  = np.zeros(T - WINDOW_SIZE)
    var_ewma = np.zeros(T - WINDOW_SIZE)
    es_ewma  = np.zeros(T - WINDOW_SIZE)
    var_fhs = np.zeros(T - WINDOW_SIZE)
    es_fhs  = np.zeros(T - WINDOW_SIZE)

    # calc multiplier 
    norm_es_mult = stats.norm.pdf(Z_ES_QUANTILE) / (1 - ALPHA_ES)

    for t in tqdm(range(WINDOW_SIZE, T)):
        i = t - WINDOW_SIZE 
        window = losses[t-WINDOW_SIZE:t]
        
        # 1. Normal
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        var_norm[i] = mu + sigma * Z_VAR
        es_norm[i] = mu + sigma * norm_es_mult
        
        # 2. T
        df_t, loc_t, scale_t = stats.t.fit(window)
        t_var_q = stats.t.ppf(ALPHA, df_t)
        t_es_q = stats.t.ppf(ALPHA_ES, df_t)
        var_t[i] = loc_t + scale_t * t_var_q
        es_t[i] = loc_t + scale_t * (stats.t.pdf(t_es_q, df_t) / (1 - ALPHA_ES)) * ((df_t + t_es_q**2) / (df_t - 1))
        
        # 3. HS
        sorted_window = np.sort(window)
        k_var = int(np.ceil(WINDOW_SIZE * (1 - ALPHA))) - 1
        var_hs[i] = sorted_window[-(k_var+1)]
        k_es = int(np.ceil(WINDOW_SIZE * (1 - ALPHA_ES))) - 1
        v_thresh = sorted_window[-(k_es+1)]
        exceeds = window[window > v_thresh]
        
        if len(exceeds) == 0:
            es_hs[i] = var_hs[i]
        else:
            es_hs[i] = np.mean(exceeds)
            
        # 4. EWMA
        s_ewma = ewma_volatility(window, lam=0.94)
        var_ewma[i] = s_ewma * Z_VAR 
        es_ewma[i] = s_ewma * norm_es_mult 
        
        # 5. GARCH
        scale_garch = np.std(window)
        scaled_w = window / scale_garch
        
        am = arch_model(scaled_w, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        res = am.fit(disp='off')
        
        forecast = res.forecast(horizon=1, align='origin')
        sig_scaled = np.sqrt(forecast.variance.values[-1, 0])
        sig_t1 = sig_scaled * scale_garch
        
        std_r = res.resid / res.conditional_volatility
        sort_r = np.sort(std_r)
        
        r_var = sort_r[-(int(np.ceil(WINDOW_SIZE * (1 - ALPHA))))]
        r_es = sort_r[-(int(np.ceil(WINDOW_SIZE * (1 - ALPHA_ES))))]
        
        exc = std_r[std_r > r_es]
        es_r_val = np.mean(exc) if len(exc) > 0 else r_var
        
        var_fhs[i] = sig_t1 * r_var
        es_fhs[i]  = sig_t1 * es_r_val

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


if __name__ == "__main__":
    compute_rolling_risk()
