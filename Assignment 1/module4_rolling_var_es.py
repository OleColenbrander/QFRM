import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    plot_rolling_var_es(results_df)


def plot_rolling_var_es(df=None):
    if df is None:
        df = pd.read_csv(OUTPUT_CSV, index_col=0, parse_dates=True)

    PLOTS_DIR = DATA_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    models   = ["Normal", "t", "HS", "EWMA", "FHS"]
    colours  = {"Normal": "#2196F3", "t": "#4CAF50", "HS": "#FF9800",
                "EWMA": "#9C27B0", "FHS": "#F44336"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)

    # ── crisis shading ──────────────────────────────────────────────
    crisis_periods = [
        ("2020-02-20", "2020-04-30", "#FFE0E0", "COVID-19"),
        ("2025-04-01", "2025-06-30", "#FFF3CD", "2025/4 shock"),
        ("2026-02-15", "2026-03-31", "#E8F5E9", "2026/3 shock"),
    ]
    for start, end, colour, label in crisis_periods:
        for ax in (ax1, ax2):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color=colour, alpha=0.45, label=label)

    # ── top panel: realized loss + VaR lines ────────────────────────
    ax1.fill_between(df.index, 0, df["Loss"], where=df["Loss"] > 0,
                     color="lightgray", alpha=0.6, label="Daily loss")
    for m in models:
        ax1.plot(df.index, df[f"VaR_{m}"], color=colours[m],
                 linewidth=0.9, label=f"VaR {m}", alpha=0.85)

    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("Dollar Loss ($)")
    ax1.set_title("Rolling 1-Day 99% VaR: All Models vs. Realized Loss\n"
                  "(1000-day window, shaded = crisis periods)", fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.legend(fontsize=7.5, loc="upper left", ncol=4)
    ax1.grid(True, alpha=0.25)

    # ── bottom panel: FHS violation markers ─────────────────────────
    violations = df["Loss"] > df["VaR_FHS"]
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.scatter(df.index[violations], df.loc[violations, "Loss"] - df.loc[violations, "VaR_FHS"],
                color="#F44336", s=12, zorder=3, label="FHS violation")
    ax2.set_ylabel("Excess ($)")
    ax2.set_title("FHS VaR Violations", fontweight="bold", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.25)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    path = PLOTS_DIR / "17_rolling_var_es.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    compute_rolling_risk()
