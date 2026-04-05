import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from pathlib import Path

from module1_data import RATE_TICKERS, load_or_build
from module2_portfolio import build_portfolio, WEIGHTS, PORTFOLIO_VALUE, LOAN_DURATION
import warnings

warnings.filterwarnings("ignore")

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "fixed_window_var_es.csv"
PLOTS_DIR   = DATA_DIR / "plots"

ALPHA_VAR     = 0.99
ALPHA_ES      = 0.975
NU_CANDIDATES = [3, 4, 5, 6]
EWMA_LAMBDA   = 0.94
SPLIT_DATE    = "2025-01-01"

def build_sensitivity_vector(tickers):
    # build the a vector
    a = []
    for t in tickers:
        if t in RATE_TICKERS: a.append(-LOAN_DURATION * WEIGHTS[t] * PORTFOLIO_VALUE / 100.0)
        else: a.append(WEIGHTS[t] * PORTFOLIO_VALUE)
    return np.array(a)

def normal_vc(ret_in, ret_out, a):
    # fit in-sample
    mu = ret_in.mean().values
    C = ret_in.cov().values
    q_var = stats.norm.ppf(ALPHA_VAR)
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)

    out_dates = ret_out.index
    T_out = len(out_dates)
    cols = {}

    for i, t in enumerate(ret_in.columns):
        mu_L  = -a[i] * mu[i]
        std_L = abs(a[i]) * np.sqrt(C[i, i])
        
        cols[(t, "VaR_99")] = np.full(T_out, mu_L + std_L * q_var)
        cols[(t, "ES_975")] = np.full(T_out, mu_L + std_L * es_scale)

    mu_Lp = float(-a @ mu)
    std_Lp = np.sqrt(float(a @ C @ a))
    cols[("Portfolio", "VaR_99")] = np.full(T_out, mu_Lp + std_Lp * q_var)
    cols[("Portfolio", "ES_975")] = np.full(T_out, mu_Lp + std_Lp * es_scale)
    
    return pd.DataFrame(cols, index=out_dates)

def plot_qq_student_t(std_losses):
    n = len(std_losses)
    emp = np.sort(std_losses.values)
    p = (np.arange(1, n + 1) - 0.5) / n

    scores = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, nu in zip(axes.flat, NU_CANDIDATES):
        th = stats.t.ppf(p, df=nu)
        mask = (p > 0.005) & (p < 0.995)
        scores[nu] = float(np.mean(np.abs(emp[mask] - th[mask])))

        lo, hi = min(th.min(), emp.min()), max(th.max(), emp.max())
        ax.scatter(th, emp, s=2)
        ax.plot([lo, hi], [lo, hi], color="red")
        ax.set_title(f"QQ Student t({nu})")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_qq_student_t_fixed_window.png")
    plt.close()
    
    best_nu = min(scores, key=scores.get)
    return best_nu

def student_t(ret_in, ret_out, a, losses_in):
    # fit in-sample
    mu = ret_in.mean().values
    C = ret_in.cov().values

    std_loss = (losses_in - losses_in.mean()) / losses_in.std()
    nu = plot_qq_student_t(std_loss)

    adj = np.sqrt((nu - 2.0) / nu)
    q_var = stats.t.ppf(ALPHA_VAR, df=nu)
    q_es = stats.t.ppf(ALPHA_ES, df=nu)
    es_t_coeff = stats.t.pdf(q_es, df=nu) / (1.0 - ALPHA_ES) * (nu + q_es ** 2) / (nu - 1.0)

    out_dates = ret_out.index
    T_out = len(out_dates)
    cols = {}

    for i, t in enumerate(ret_in.columns):
        mu_L = -a[i] * mu[i]
        scale_L = abs(a[i]) * np.sqrt(C[i, i]) * adj
        cols[(t, "VaR_99")] = np.full(T_out, mu_L + scale_L * q_var)
        cols[(t, "ES_975")] = np.full(T_out, mu_L + scale_L * es_t_coeff)

    mu_Lp = float(-a @ mu)
    scale_Lp = np.sqrt(float(a @ C @ a)) * adj
    cols[("Portfolio", "VaR_99")] = np.full(T_out, mu_Lp + scale_Lp * q_var)
    cols[("Portfolio", "ES_975")] = np.full(T_out, mu_Lp + scale_Lp * es_t_coeff)

    return pd.DataFrame(cols, index=out_dates)

def historical_simulation(pnl_in, pnl_out, losses_in):
    out_dates = pnl_out.index
    T_out = len(out_dates)
    cols = {}
    
    for t in pnl_in.columns:
        comp_loss = -pnl_in[t]
        es_thresh = float(comp_loss.quantile(ALPHA_ES))
        tail = comp_loss[comp_loss > es_thresh]
        
        var_v = float(comp_loss.quantile(ALPHA_VAR))
        es_v = float(tail.mean()) if len(tail) > 0 else es_thresh
        
        cols[(t, "VaR_99")] = np.full(T_out, var_v)
        cols[(t, "ES_975")] = np.full(T_out, es_v)

    es_thresh = float(losses_in.quantile(ALPHA_ES))
    tail_port = losses_in[losses_in > es_thresh]
    
    cols[("Portfolio", "VaR_99")] = np.full(T_out, float(losses_in.quantile(ALPHA_VAR)))
    cols[("Portfolio", "ES_975")] = np.full(T_out, float(tail_port.mean()) if len(tail_port) > 0 else es_thresh)
    
    return pd.DataFrame(cols, index=out_dates)

def garch_ccc(ret_in, ret_out, a):
    tickers = list(ret_in.columns)
    
    # In-sample estimation
    params_dict = {}
    stds_in = {}
    
    last_sig2 = {}
    last_resid2 = {}
    
    for t in tickers:
        series = ret_in[t].values * 100.0
        try:
            res = arch_model(series, mean="Constant", vol="Garch", p=1, q=1).fit(disp="off")
            stds_in[t] = np.asarray(res.std_resid)
            
            omega = res.params['omega']
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            mu = res.params.get('mu', series.mean())
            
            params_dict[t] = (omega, alpha, beta, mu)
            
            # save state for out of sample period simulation
            last_sig2[t] = res.conditional_volatility.values[-1] ** 2
            last_resid2[t] = res.resid.values[-1] ** 2
        except:
            # fallback to un-filtered
            s = series.std()
            m = series.mean()
            params_dict[t] = (s**2 * (1 - 0.9 - 0.05), 0.05, 0.9, m)
            stds_in[t] = (series - m) / s
            last_sig2[t] = s**2
            last_resid2[t] = 0.0

    R = pd.DataFrame(stds_in).corr().values
    
    out_dates = ret_out.index
    T_out = len(out_dates)
    
    q_var = stats.norm.ppf(ALPHA_VAR)
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)
    mu_in = ret_in.mean().values
    
    cols = {}
    for t in tickers + ["Portfolio"]:
        cols[(t, "VaR_99")] = np.zeros(T_out)
        cols[(t, "ES_975")] = np.zeros(T_out)
        
    for idx_t in range(T_out):
        h = np.zeros(len(tickers))
        new_last_sig2 = {}
        new_last_resid2 = {}
        
        for i, t in enumerate(tickers):
            omega, alpha, beta, mu_garch = params_dict[t]
            # OOS update
            sig2_pred = omega + alpha * last_resid2[t] + beta * last_sig2[t]
            h[i] = np.sqrt(sig2_pred) / 100.0
            
            real_resid = (ret_out[t].iloc[idx_t] * 100.0) - mu_garch
            new_last_sig2[t] = sig2_pred
            new_last_resid2[t] = real_resid ** 2
            
            mu_L = -a[i] * mu_in[i]
            std_L = abs(a[i]) * h[i]
            cols[(t, "VaR_99")][idx_t] = mu_L + std_L * q_var
            cols[(t, "ES_975")][idx_t] = mu_L + std_L * es_scale
            
        last_sig2 = new_last_sig2
        last_resid2 = new_last_resid2
        
        H = np.diag(h) @ R @ np.diag(h)
        mu_Lp = float(-a @ mu_in)
        std_Lp = np.sqrt(float(a @ H @ a))
        cols[("Portfolio", "VaR_99")][idx_t] = mu_Lp + std_Lp * q_var
        cols[("Portfolio", "ES_975")][idx_t] = mu_Lp + std_Lp * es_scale
        
    return pd.DataFrame(cols, index=out_dates)

def fhs_ewma(ret_in, ret_out, pnl_in, pnl_out, a):
    tickers = list(ret_in.columns)
    lam = EWMA_LAMBDA

    def extract_ewma_in_sample(series):
        var = np.empty(len(series))
        var[0] = series[0] ** 2
        for t in range(1, len(series)):
            var[t] = lam * var[t - 1] + (1.0 - lam) * series[t - 1] ** 2
        return var, series[-1]

    out_dates = ret_out.index
    T_out = len(out_dates)
    cols = {}
    
    var_states = {}
    prev_r = {}
    q_z_var = {}
    q_z_es = {}

    for i, t in enumerate(tickers):
        f = ret_in[t].values
        var_s, last_r = extract_ewma_in_sample(f)
        sig_s = np.sqrt(var_s)
        
        var_states[t] = var_s[-1]
        prev_r[t] = last_r
        
        lz = -np.sign(a[i]) * (f / sig_s)
        
        q_es = float(np.quantile(lz, ALPHA_ES))
        tail = lz[lz > q_es]
        
        q_z_var[t]  = float(np.quantile(lz, ALPHA_VAR))
        q_z_es[t]   = float(tail.mean()) if len(tail) > 0 else float(q_es)

        cols[(t, "VaR_99")] = np.zeros(T_out)
        cols[(t, "ES_975")] = np.zeros(T_out)

    # In sample portfolio ewma extraction
    port_in = pnl_in.sum(axis=1).values
    port_var_s, p_last_r = extract_ewma_in_sample(port_in)
    port_var_state = port_var_s[-1]
    
    lz_port = -(port_in / np.sqrt(port_var_s))
    q_es_p = float(np.quantile(lz_port, ALPHA_ES))
    tail_p = lz_port[lz_port > q_es_p]
    
    q_z_var["Portfolio"] = float(np.quantile(lz_port, ALPHA_VAR))
    q_z_es["Portfolio"]  = float(tail_p.mean()) if len(tail_p) > 0 else float(q_es_p)
    
    cols[("Portfolio", "VaR_99")] = np.zeros(T_out)
    cols[("Portfolio", "ES_975")] = np.zeros(T_out)

    # OOS walk 
    for idx_t in range(T_out):
        # Predict volatility for day t using known variance of t-1
        port_sig_pred = np.sqrt(lam * port_var_state + (1.0 - lam) * p_last_r ** 2)
        cols[("Portfolio", "VaR_99")][idx_t] = port_sig_pred * q_z_var["Portfolio"]
        cols[("Portfolio", "ES_975")][idx_t] = port_sig_pred * q_z_es["Portfolio"]
        
        port_out_r = pnl_out.sum(axis=1).iloc[idx_t]
        port_var_state = port_sig_pred ** 2
        p_last_r = port_out_r
        
        for i, t in enumerate(tickers):
            sig_pred = np.sqrt(lam * var_states[t] + (1.0 - lam) * prev_r[t] ** 2)
            fc = abs(a[i]) * sig_pred
            
            cols[(t, "VaR_99")][idx_t] = fc * q_z_var[t]
            cols[(t, "ES_975")][idx_t] = fc * q_z_es[t]
            
            var_states[t] = sig_pred ** 2
            prev_r[t] = ret_out[t].iloc[idx_t]

    return pd.DataFrame(cols, index=out_dates)

def estimate_var_es(force=False):
    _, ret = load_or_build(force)
    pnl, _, losses = build_portfolio(force)
    
    a = build_sensitivity_vector(ret.columns)
    
    # Split Data into in_sample and out_of_sample 
    ret_in  = ret[ret.index < SPLIT_DATE]
    ret_out = ret[ret.index >= SPLIT_DATE]
    pnl_in  = pnl[pnl.index < SPLIT_DATE]
    pnl_out = pnl[pnl.index >= SPLIT_DATE]
    losses_in = losses[losses.index < SPLIT_DATE]
    
    if len(ret_out) == 0:
        print("No out-of-sample data. Please check SPLIT_DATE.")
        return
        
    print(f"In-sample: {len(ret_in)} days, Out-of-sample: {len(ret_out)} days")
        
    res = {
        "Normal": normal_vc(ret_in, ret_out, a),
        "Student-t": student_t(ret_in, ret_out, a, losses_in),
        "Historical": historical_simulation(pnl_in, pnl_out, losses_in),
        "GARCH-CCC": garch_ccc(ret_in, ret_out, a),
        "FHS-EWMA": fhs_ewma(ret_in, ret_out, pnl_in, pnl_out, a),
    }

    df_list = []
    for k, v in res.items():
        v.columns = pd.MultiIndex.from_tuples([(k, c[0], c[1]) for c in v.columns])
        df_list.append(v)
        
    summary = pd.concat(df_list, axis=1)
    summary.to_csv(RESULTS_CSV)
    
    print(f"Fixed Window out-of-sample mapping completed -> {RESULTS_CSV.name}")
    return summary

if __name__ == "__main__":
    estimate_var_es()
