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
RESULTS_CSV = DATA_DIR / "var_es_results.csv"
PLOTS_DIR   = DATA_DIR / "plots"


ALPHA_VAR     = 0.99
ALPHA_ES      = 0.975
NU_CANDIDATES = [3, 4, 5, 6]
EWMA_LAMBDA   = 0.94

def build_sensitivity_vector(tickers):
    # build the a vector
    a = []
    for t in tickers:
        if t in RATE_TICKERS: a.append(-LOAN_DURATION * WEIGHTS[t] * PORTFOLIO_VALUE / 100.0)
        else: a.append(WEIGHTS[t] * PORTFOLIO_VALUE)
    return np.array(a)

def normal_vc(returns, a):
    # meth 1
    mu = returns.mean().values
    C = returns.cov().values
    q_var = stats.norm.ppf(ALPHA_VAR)
    
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)

    rows = {}
    for i, t in enumerate(returns.columns):
        mu_L  = -a[i] * mu[i]
        std_L = abs(a[i]) * np.sqrt(C[i, i])
        rows[t] = {"VaR_99": mu_L + std_L * q_var, "ES_975": mu_L + std_L * es_scale}

    mu_Lp = float(-a @ mu)
    std_Lp = np.sqrt(float(a @ C @ a))
    rows["Portfolio"] = {"VaR_99": mu_Lp + std_Lp * q_var, "ES_975": mu_Lp + std_Lp * es_scale}
    return pd.DataFrame(rows).T


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
    fig.savefig(PLOTS_DIR / "05_qq_student_t.png")
    plt.close()
    
    best_nu = min(scores, key=scores.get)
    return best_nu

def student_t(returns, a, losses):
    # meth 2
    mu = returns.mean().values
    C = returns.cov().values

    std_loss = (losses - losses.mean()) / losses.std()
    nu = plot_qq_student_t(std_loss)

    adj = np.sqrt((nu - 2.0) / nu)
    q_var = stats.t.ppf(ALPHA_VAR, df=nu)
    q_es = stats.t.ppf(ALPHA_ES, df=nu)
    es_t_coeff = stats.t.pdf(q_es, df=nu) / (1.0 - ALPHA_ES) * (nu + q_es ** 2) / (nu - 1.0)

    rows = {}
    for i, t in enumerate(returns.columns):
        mu_L = -a[i] * mu[i]
        scale_L = abs(a[i]) * np.sqrt(C[i, i]) * adj
        rows[t] = {"VaR_99": mu_L + scale_L * q_var, "ES_975": mu_L + scale_L * es_t_coeff}

    mu_Lp = float(-a @ mu)
    scale_Lp = np.sqrt(float(a @ C @ a)) * adj
    rows["Portfolio"] = {"VaR_99": mu_Lp + scale_Lp * q_var, "ES_975": mu_Lp + scale_Lp * es_t_coeff}

    return pd.DataFrame(rows).T


def historical_simulation(pnl, losses):
    # meth 3
    rows = {}
    for t in pnl.columns:
        comp_loss = -pnl[t]
        es_thresh = float(comp_loss.quantile(ALPHA_ES))
        tail = comp_loss[comp_loss > es_thresh]
        
        var_v = float(comp_loss.quantile(ALPHA_VAR))
        es_v = float(tail.mean()) if len(tail) > 0 else es_thresh
        rows[t] = {"VaR_99": var_v, "ES_975": es_v}

    es_thresh = float(losses.quantile(ALPHA_ES))
    tail_port = losses[losses > es_thresh]
    rows["Portfolio"] = {"VaR_99": float(losses.quantile(ALPHA_VAR)), "ES_975": float(tail_port.mean()) if len(tail_port) > 0 else es_thresh}
    return pd.DataFrame(rows).T


def garch_ccc(returns, a):
    # meth 4
    tickers = list(returns.columns)
    fcsts = {}
    stds = {}

    for t in tickers:
        series = returns[t].values * 100.0
        try:
            res = arch_model(series, mean="Constant", vol="Garch", p=1, q=1).fit(disp="off")
            stds[t] = np.asarray(res.std_resid)
            fcsts[t] = np.sqrt(float(np.asarray(res.forecast(horizon=1, reindex=False).variance).flat[-1])) / 100.0
        except:
            fcsts[t] = float(returns[t].std())
            stds[t] = ((returns[t] - returns[t].mean()) / returns[t].std()).values

    R = pd.DataFrame(stds, index=returns.index).corr().values
    h = np.array([fcsts[t] for t in tickers])
    H = np.diag(h) @ R @ np.diag(h)

    mu = returns.mean().values
    q_var = stats.norm.ppf(ALPHA_VAR)
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)

    rows = {}
    for i, t in enumerate(tickers):
        mu_L = -a[i] * mu[i]
        std_L = abs(a[i]) * h[i]
        rows[t] = {"VaR_99": mu_L + std_L * q_var, "ES_975": mu_L + std_L * es_scale}

    mu_Lp = float(-a @ mu)
    std_Lp = np.sqrt(float(a @ H @ a))
    rows["Portfolio"] = {"VaR_99": mu_Lp + std_Lp * q_var, "ES_975": mu_Lp + std_Lp * es_scale}

    return pd.DataFrame(rows).T


def fhs_ewma(returns, pnl, a):
    # meth 5
    tickers = list(returns.columns)
    lam = EWMA_LAMBDA

    def do_ewma(s):
        var = np.empty(len(s))
        var[0] = s[0] ** 2
        for t in range(1, len(s)):
            var[t] = lam * var[t - 1] + (1.0 - lam) * s[t - 1] ** 2
        return np.sqrt(var), np.sqrt(lam * var[-1] + (1.0 - lam) * s[-1] ** 2)

    rows = {}
    for i, t in enumerate(tickers):
        f = returns[t].values
        sig, sig_n = do_ewma(f)
        lz = -np.sign(a[i]) * (f / sig)
        fc = abs(a[i]) * sig_n
        
        q_es = float(np.quantile(lz, ALPHA_ES))
        tail = lz[lz > q_es]
        
        rows[t] = {"VaR_99": float(fc * np.quantile(lz, ALPHA_VAR)), "ES_975": float(fc * tail.mean()) if len(tail) > 0 else float(fc * q_es)}

    port = pnl.sum(axis=1).values
    p_sig, p_sig_n = do_ewma(port)
    lz_port = -(port / p_sig)
    q_es_p = float(np.quantile(lz_port, ALPHA_ES))
    tail_p = lz_port[lz_port > q_es_p]
    
    rows["Portfolio"] = {"VaR_99": float(p_sig_n * np.quantile(lz_port, ALPHA_VAR)), "ES_975": float(p_sig_n * tail_p.mean()) if len(tail_p) > 0 else float(p_sig_n * q_es_p)}

    return pd.DataFrame(rows).T



def estimate_var_es(force=False):
    _, ret = load_or_build(force)
    pnl, _, losses = build_portfolio(force)
    
    a = build_sensitivity_vector(ret.columns)
    
    res = {
        "Normal": normal_vc(ret, a),
        "Student-t": student_t(ret, a, losses),
        "Historical": historical_simulation(pnl, losses),
        "GARCH-CCC": garch_ccc(ret, a),
        "FHS-EWMA": fhs_ewma(ret, pnl, a),
    }

    df_list = []
    for k, v in res.items():
        v.columns = pd.MultiIndex.from_product([[k], v.columns])
        df_list.append(v)
        
    summary = pd.concat(df_list, axis=1)
    summary.to_csv(RESULTS_CSV)
    
    return summary

if __name__ == "__main__":
    estimate_var_es()
