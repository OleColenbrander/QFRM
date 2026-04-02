"""
Module 3: VaR & ES Estimation (1-Day Horizon)
==============================================

Confidence levels
-----------------
  VaR : 99%   (alpha = 0.99)
  ES  : 97.5% (alpha = 0.975)

Both are computed for every portfolio constituent and for the aggregate portfolio.

Methods implemented
-------------------
1. Normal Variance-Covariance  (MFE 2015, Section 9.2.2)
2. Multivariate Student-t      (MFE 2015, Section 9.2.2)
3. Historical Simulation       (MFE 2015, Section 9.2.3)
4. GARCH(1,1) CCC              (MFE 2015, Section 14.2.2)
5. Filtered Historical Simulation with EWMA  (MFE 2015, Section 9.2.4)

Sensitivity vector
------------------
The dollar profit-and-loss of each position is:
    PnL_i  = a_i * f_i

where f_i is the risk factor (log-return for equities, yield change in p.p. for ^IRX) and the sensitivity a_i is:
    a_i    = w_i * V                     for equities / index
    a_IRX  = -D_mod * w_IRX * V / 100   for the loan proxy (yield change in p.p.)

All VaR and ES figures are expressed as positive dollar amounts representing the loss at the stated confidence level.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from arch import arch_model

from module1_data import RATE_TICKERS, load_or_build
from module2_portfolio import build_portfolio, WEIGHTS, PORTFOLIO_VALUE, LOAN_DURATION

warnings.filterwarnings("ignore")


DATA_DIR    = Path(__file__).parent / "data"
PLOTS_DIR   = DATA_DIR / "plots"
RESULTS_CSV = DATA_DIR / "var_es_results.csv"

ALPHA_VAR     = 0.99
ALPHA_ES      = 0.975
NU_CANDIDATES = [3, 4, 5, 6]
EWMA_LAMBDA   = 0.94


def build_sensitivity_vector(tickers):
    a = []
    for t in tickers:
        if t in RATE_TICKERS:
            a.append(-LOAN_DURATION * WEIGHTS[t] * PORTFOLIO_VALUE / 100.0)
        else:
            a.append(WEIGHTS[t] * PORTFOLIO_VALUE)
    return np.array(a)



# Method 1: Normal Variance-Covariance
def normal_vc(returns, a):
    print("Method 1 -- Normal Variance-Covariance ...")
    tickers  = list(returns.columns)
    mu       = returns.mean().values
    C        = returns.cov().values
    q_var    = stats.norm.ppf(ALPHA_VAR)
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)

    rows = {}
    for i, ticker in enumerate(tickers):
        mu_L  = -a[i] * mu[i]
        std_L = abs(a[i]) * np.sqrt(C[i, i])
        rows[ticker] = {"VaR_99": mu_L + std_L * q_var, "ES_975": mu_L + std_L * es_scale}

    mu_Lp  = float(-a @ mu)
    std_Lp = np.sqrt(float(a @ C @ a))
    rows["Portfolio"] = {"VaR_99": mu_Lp + std_Lp * q_var, "ES_975": mu_Lp + std_Lp * es_scale}

    return pd.DataFrame(rows).T



# Method 2: Multivariate Student-t
def plot_qq_student_t(std_losses):
    n   = len(std_losses)
    emp = np.sort(std_losses.values)
    p   = (np.arange(1, n + 1) - 0.5) / n

    mad_scores = {}
    fig, axes  = plt.subplots(2, 2, figsize=(13, 10))

    for ax, nu in zip(axes.flat, NU_CANDIDATES):
        th   = stats.t.ppf(p, df=nu)
        mask = (p > 0.005) & (p < 0.995)
        mad  = float(np.mean(np.abs(emp[mask] - th[mask])))
        mad_scores[nu] = mad

        lo, hi = min(th.min(), emp.min()), max(th.max(), emp.max())
        ax.scatter(th, emp, s=2, alpha=0.45, color="navy")
        ax.plot([lo, hi], [lo, hi], color="firebrick", linewidth=1.2, label="45-degree line")
        ax.set_title("QQ vs t(%d)  |  MAD = %.4f" % (nu, mad), fontsize=10)
        ax.set_xlabel("Theoretical t(%d) Quantiles" % nu)
        ax.set_ylabel("Empirical Quantiles (standardised loss)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle("QQ-Plots: Standardised Portfolio Losses vs Student-t(nu)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_qq_student_t.png", dpi=150)
    plt.close(fig)
    print("  Student-t QQ plots saved.")
    print("  MAD scores:", {nu: round(v, 5) for nu, v in mad_scores.items()})

    return min(mad_scores, key=mad_scores.get)


def student_t(returns, a, losses):
    print("Method 2 -- Multivariate Student-t ...")
    tickers = list(returns.columns)
    mu      = returns.mean().values
    C       = returns.cov().values

    std_losses = (losses - losses.mean()) / losses.std()
    nu         = plot_qq_student_t(std_losses)
    print(f"  Selected nu = {nu} (lowest QQ MAD)")

    scale_adj  = np.sqrt((nu - 2.0) / nu)
    q_var      = stats.t.ppf(ALPHA_VAR, df=nu)
    q_es       = stats.t.ppf(ALPHA_ES, df=nu)
    es_t_coeff = stats.t.pdf(q_es, df=nu) / (1.0 - ALPHA_ES) * (nu + q_es ** 2) / (nu - 1.0)

    rows = {}
    for i, ticker in enumerate(tickers):
        mu_L    = -a[i] * mu[i]
        scale_L = abs(a[i]) * np.sqrt(C[i, i]) * scale_adj
        rows[ticker] = {"VaR_99": mu_L + scale_L * q_var, "ES_975": mu_L + scale_L * es_t_coeff}

    mu_Lp    = float(-a @ mu)
    scale_Lp = np.sqrt(float(a @ C @ a)) * scale_adj
    rows["Portfolio"] = {"VaR_99": mu_Lp + scale_Lp * q_var, "ES_975": mu_Lp + scale_Lp * es_t_coeff}

    return pd.DataFrame(rows).T

# Method 3: Historical Simulation
def historical_simulation(pnl, losses):
    print("Method 3 -- Historical Simulation ...")
    rows = {}
    for ticker in pnl.columns:
        comp_loss = -pnl[ticker]
        es_thresh = float(comp_loss.quantile(ALPHA_ES))
        tail      = comp_loss[comp_loss > es_thresh]
        rows[ticker] = {
            "VaR_99": float(comp_loss.quantile(ALPHA_VAR)),
            "ES_975": float(tail.mean()) if len(tail) > 0 else es_thresh,
        }

    es_thresh = float(losses.quantile(ALPHA_ES))
    tail_port = losses[losses > es_thresh]
    rows["Portfolio"] = {
        "VaR_99": float(losses.quantile(ALPHA_VAR)),
        "ES_975": float(tail_port.mean()) if len(tail_port) > 0 else es_thresh,
    }

    return pd.DataFrame(rows).T

# Method 4: GARCH(1,1) CCC
def garch_ccc(returns, a):
    print("Method 4 -- GARCH(1,1) CCC ...")
    tickers        = list(returns.columns)
    garch_forecasts = {}
    std_resid_dict  = {}

    for ticker in tickers:
        series = returns[ticker].values * 100.0
        try:
            res = arch_model(series, mean="Constant", vol="Garch", p=1, q=1,
                             dist="normal").fit(disp="off", show_warning=False)
            std_resid_dict[ticker]  = np.asarray(res.std_resid)
            h_next = np.sqrt(float(np.asarray(
                res.forecast(horizon=1, reindex=False).variance).flat[-1])) / 100.0
            garch_forecasts[ticker] = h_next
            print(f"  GARCH [OK] {ticker:<8}  h_{{T+1}} = {h_next:.6f}")
        except Exception as exc:
            print(f"  GARCH FAILED for {ticker}: {exc} -- using sample std")
            garch_forecasts[ticker] = float(returns[ticker].std())
            std_resid_dict[ticker]  = ((returns[ticker] - returns[ticker].mean())
                                       / returns[ticker].std()).values

    R     = pd.DataFrame(std_resid_dict, index=returns.index).corr().values
    h_vec = np.array([garch_forecasts[t] for t in tickers])
    H     = np.diag(h_vec) @ R @ np.diag(h_vec)

    mu       = returns.mean().values
    q_var    = stats.norm.ppf(ALPHA_VAR)
    es_scale = stats.norm.pdf(stats.norm.ppf(ALPHA_ES)) / (1.0 - ALPHA_ES)

    rows = {}
    for i, ticker in enumerate(tickers):
        mu_L  = -a[i] * mu[i]
        std_L = abs(a[i]) * h_vec[i]
        rows[ticker] = {"VaR_99": mu_L + std_L * q_var, "ES_975": mu_L + std_L * es_scale}

    mu_Lp  = float(-a @ mu)
    std_Lp = np.sqrt(float(a @ H @ a))
    rows["Portfolio"] = {"VaR_99": mu_Lp + std_Lp * q_var, "ES_975": mu_Lp + std_Lp * es_scale}

    return pd.DataFrame(rows).T

# Method 5: Filtered Historical Simulation with EWMA
def fhs_ewma(returns, pnl, a):
    print("Method 5 -- Filtered Historical Simulation (EWMA) ...")
    tickers = list(returns.columns)
    lam     = EWMA_LAMBDA

    def ewma_vol(series):
        var    = np.empty(len(series))
        var[0] = series[0] ** 2
        for t in range(1, len(series)):
            var[t] = lam * var[t - 1] + (1.0 - lam) * series[t - 1] ** 2
        forecast = lam * var[-1] + (1.0 - lam) * series[-1] ** 2
        return np.sqrt(var), np.sqrt(forecast)

    rows = {}
    for i, ticker in enumerate(tickers):
        f                = returns[ticker].values
        sigma, sigma_next = ewma_vol(f)
        lz               = -np.sign(a[i]) * (f / sigma)
        dollar_fc        = abs(a[i]) * sigma_next
        q_es             = float(np.quantile(lz, ALPHA_ES))
        tail             = lz[lz > q_es]
        rows[ticker] = {
            "VaR_99": float(dollar_fc * np.quantile(lz, ALPHA_VAR)),
            "ES_975": float(dollar_fc * tail.mean()) if len(tail) > 0 else float(dollar_fc * q_es),
        }

    port_arr              = pnl.sum(axis=1).values
    p_sigma, p_sigma_next = ewma_vol(port_arr)
    lz_port               = -(port_arr / p_sigma)
    q_es_p                = float(np.quantile(lz_port, ALPHA_ES))
    tail_p                = lz_port[lz_port > q_es_p]
    rows["Portfolio"] = {
        "VaR_99": float(p_sigma_next * np.quantile(lz_port, ALPHA_VAR)),
        "ES_975": float(p_sigma_next * tail_p.mean()) if len(tail_p) > 0 else float(p_sigma_next * q_es_p),
    }

    return pd.DataFrame(rows).T

# Plots
def plot_comparison(summary):
    methods  = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]
    var_vals = [summary.loc["Portfolio", (m, "VaR_99")] for m in methods]
    es_vals  = [summary.loc["Portfolio", (m, "ES_975")] for m in methods]

    x   = np.arange(len(methods))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w / 2, var_vals, w, label="VaR 99%",  color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w / 2, es_vals,  w, label="ES 97.5%", color="firebrick",  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Dollar Loss ($)", fontsize=10)
    ax.set_title("1-Day Portfolio VaR (99%) and ES (97.5%) by Method  |  V0 = $1,000,000",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"${h:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_var_es_comparison.png", dpi=150)
    plt.close(fig)
    print("  Comparison chart saved.")


def plot_component_heatmap(summary, tickers):
    methods  = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]
    entities = tickers + ["Portfolio"]

    matrix = pd.DataFrame(index=entities, columns=methods, dtype=float)
    for entity in entities:
        for m in methods:
            try:
                matrix.loc[entity, m] = summary.loc[entity, (m, "VaR_99")]
            except KeyError:
                matrix.loc[entity, m] = np.nan

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities, fontsize=10)
    plt.colorbar(im, ax=ax, label="VaR 99% ($)")

    for i in range(len(entities)):
        for j in range(len(methods)):
            ax.text(j, i, f"${float(matrix.iloc[i, j]):,.0f}",
                    ha="center", va="center", fontsize=7.5, color="black")

    ax.set_title("VaR 99% by Asset and Method ($)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_var_heatmap.png", dpi=150)
    plt.close(fig)
    print("  VaR heatmap saved.")


def estimate_var_es(force_rebuild=False):
    _, returns      = load_or_build(force_rebuild=force_rebuild)
    pnl, _, losses  = build_portfolio(force_rebuild=force_rebuild)
    tickers         = list(returns.columns)
    a               = build_sensitivity_vector(tickers)

    results = {
        "Normal":     normal_vc(returns, a),
        "Student-t":  student_t(returns, a, losses),
        "Historical": historical_simulation(pnl, losses),
        "GARCH-CCC":  garch_ccc(returns, a),
        "FHS-EWMA":   fhs_ewma(returns, pnl, a),
    }

    frames = []
    for method_name, df in results.items():
        df.columns = pd.MultiIndex.from_product([[method_name], df.columns])
        frames.append(df)
    summary = pd.concat(frames, axis=1)

    plot_comparison(summary)
    plot_component_heatmap(summary, tickers)
    summary.to_csv(RESULTS_CSV)
    print(f"Results saved -> {RESULTS_CSV}")

    return summary


if __name__ == "__main__":
    estimate_var_es()
