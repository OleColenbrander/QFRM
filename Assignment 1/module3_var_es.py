import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from pathlib import Path

from module1_data import RATE_TICKERS, load_or_build
from module2_portfolio import build_portfolio, compute_component_pnl, WEIGHTS, PORTFOLIO_VALUE, LOAN_DURATION
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


def _select_nu(std_losses):
    """Select best-fitting Student-t degrees of freedom using MAE on QQ deviations."""
    vals = std_losses.values if hasattr(std_losses, "values") else np.asarray(std_losses)
    n = len(vals)
    emp = np.sort(vals)
    p = (np.arange(1, n + 1) - 0.5) / n
    mask = (p > 0.005) & (p < 0.995)
    scores = {}
    for nu in NU_CANDIDATES:
        th = stats.t.ppf(p, df=nu)
        scores[nu] = float(np.mean(np.abs(emp[mask] - th[mask])))
    return min(scores, key=scores.get)


def plot_qq_student_t(std_losses):
    n = len(std_losses)
    emp = np.sort(std_losses.values)
    p = (np.arange(1, n + 1) - 0.5) / n

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, nu in zip(axes.flat, NU_CANDIDATES):
        th = stats.t.ppf(p, df=nu)
        lo, hi = min(th.min(), emp.min()), max(th.max(), emp.max())
        ax.scatter(th, emp, s=2)
        ax.plot([lo, hi], [lo, hi], color="red")
        ax.set_title(f"QQ Student t({nu})")

    best_nu = _select_nu(std_losses)
    fig.suptitle(f"Best fit: t({best_nu})", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_qq_student_t.png")
    plt.close()
    return best_nu


def student_t(returns, a, losses, plot_qq=True):
    # meth 2
    mu = returns.mean().values
    C = returns.cov().values

    std_loss = (losses - losses.mean()) / losses.std()
    nu = plot_qq_student_t(std_loss) if plot_qq else _select_nu(std_loss)

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
    # meth 4: GARCH(1,1)-CCC with Filtered Historical Simulation
    # Step 1: fit per-asset GARCH(1,1) to obtain (a) next-period volatility
    #   forecast sigma_{i,T+1} and (b) standardised residuals z_{i,t}.
    # Step 2: apply FHS — rescale every historical day using today's forecast
    #   vol: L_fhs_t = -a^T (mu + sigma_fcst * z_t).  The joint distribution of
    #   z_t across assets naturally preserves the CCC correlation structure, so
    #   no explicit correlation matrix is needed for the quantile step.
    # Step 3: read off empirical quantiles of L_fhs for VaR and ES.
    tickers = list(returns.columns)
    fcsts   = {}   # sigma_{i,T+1} — next-period forecast vols
    stds    = {}   # z_{i,t}       — standardised GARCH residuals

    for t in tickers:
        series = returns[t].values * 100.0
        try:
            res     = arch_model(series, mean="Constant", vol="Garch", p=1, q=1).fit(disp="off")
            stds[t] = np.asarray(res.std_resid)
            fcsts[t] = np.sqrt(
                float(np.asarray(res.forecast(horizon=1, reindex=False).variance).flat[-1])
            ) / 100.0
        except Exception:
            fcsts[t] = float(returns[t].std())
            stds[t]  = ((returns[t] - returns[t].mean()) / returns[t].std()).values

    # (T x n) matrix of standardised residuals, (n,) forecast vols
    Z          = np.column_stack([stds[t] for t in tickers])
    sigma_fcst = np.array([fcsts[t] for t in tickers])
    mu         = returns.mean().values

    # FHS portfolio losses: L_t = -a^T mu  -  (a * sigma_fcst)^T z_t
    L_port = -float(a @ mu) - Z @ (a * sigma_fcst)

    rows = {}
    for i, t in enumerate(tickers):
        # Per-asset FHS loss: -a_i*(mu_i + sigma_i_fcst * z_{i,t})
        L_asset  = -a[i] * mu[i] - a[i] * sigma_fcst[i] * stds[t]
        q_es     = float(np.quantile(L_asset, ALPHA_ES))
        tail     = L_asset[L_asset > q_es]
        rows[t]  = {
            "VaR_99": float(np.quantile(L_asset, ALPHA_VAR)),
            "ES_975": float(tail.mean()) if len(tail) > 0 else q_es,
        }

    q_es_p   = float(np.quantile(L_port, ALPHA_ES))
    tail_p   = L_port[L_port > q_es_p]
    rows["Portfolio"] = {
        "VaR_99": float(np.quantile(L_port, ALPHA_VAR)),
        "ES_975": float(tail_p.mean()) if len(tail_p) > 0 else q_es_p,
    }

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



def plot_var_es_comparison(summary):
    methods = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]
    entities = list(summary.index)
    x = np.arange(len(entities))
    w = 0.15
    colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, measure, title in zip(axes, ["VaR_99", "ES_975"], ["VaR 99%", "ES 97.5%"]):
        for i, (m, c) in enumerate(zip(methods, colours)):
            vals = [summary.loc[e, (m, measure)] for e in entities]
            ax.bar(x + (i - 2) * w, vals, w, label=m, color=c, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(entities, rotation=15, fontsize=9)
        ax.set_title(f"Portfolio {title} by Method", fontweight="bold")
        ax.set_ylabel("Dollar Loss ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("VaR & ES Comparison Across Methods and Entities", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_var_es_comparison.png", dpi=150)
    plt.close(fig)

    # plot 07: heatmap (portfolio row only, methods × measures)
    port_data = pd.DataFrame(
        {m: {measure: summary.loc["Portfolio", (m, measure)] for measure in ["VaR_99", "ES_975"]}
         for m in methods}
    ).T
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    import matplotlib.cm as cm
    im = ax2.imshow(port_data.values, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["VaR 99%", "ES 97.5%"])
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    for i in range(len(methods)):
        for j, col in enumerate(["VaR_99", "ES_975"]):
            ax2.text(j, i, f"${port_data.iloc[i, j]:,.0f}", ha="center", va="center", fontsize=9)
    fig2.colorbar(im, ax=ax2, label="Dollar Loss ($)")
    ax2.set_title("Portfolio VaR & ES Heatmap by Method", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "07_var_heatmap.png", dpi=150)
    plt.close(fig2)


# Named analysis periods — add or remove entries to change what is compared.
# GARCH is skipped for periods shorter than MIN_OBS_GARCH observations.
PERIODS = {
    "Full":          ("2016-01-01", "2026-03-31"),
    "Pre-COVID":     ("2016-01-01", "2020-02-19"),
    "COVID crisis":  ("2020-02-20", "2020-06-30"),
    "Post-COVID":    ("2020-07-01", "2024-12-31"),
    "Pre-2025":      ("2016-01-01", "2024-12-31"),
    "Shock 2025/4":  ("2025-04-01", "2025-12-31"),
    "Shock 2026/3":  ("2026-02-01", "2026-03-31"),  # ~40 obs; GARCH auto-skipped
}
MIN_OBS_GARCH = 250


def compare_named_periods(force=False):
    """Run all 5 VaR/ES methods on each named period and compare results."""
    _, ret_full = load_or_build(force)

    methods  = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]
    colours  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    all_res  = {}   # period_name -> {method -> portfolio VaR_99, ES_975}
    rows_csv = []

    print("\n" + "=" * 70)
    print("  Named Period Comparison — Portfolio 99% VaR & 97.5% ES")
    print("=" * 70)

    for period_name, (start, end) in PERIODS.items():
        ret = ret_full[start:end]
        n   = len(ret)
        if n < 30:
            print(f"  Skipping '{period_name}': only {n} obs")
            continue

        pnl    = compute_component_pnl(ret)
        losses = (-pnl.sum(axis=1)).rename("Loss")
        a      = build_sensitivity_vector(ret.columns)

        res = {}
        res["Normal"]    = normal_vc(ret, a)
        res["Student-t"] = student_t(ret, a, losses, plot_qq=False)
        res["Historical"]= historical_simulation(pnl, losses)
        res["GARCH-CCC"] = garch_ccc(ret, a) if n >= MIN_OBS_GARCH else None
        res["FHS-EWMA"]  = fhs_ewma(ret, pnl, a)

        all_res[period_name] = {}
        print(f"\n  {period_name}  ({start} → {end}, n={n})")
        print(f"  {'Method':<14} {'VaR 99%':>10}  {'ES 97.5%':>10}")
        print(f"  {'-'*38}")
        for m in methods:
            if res[m] is None:
                print(f"  {m:<14} {'—':>10}  {'—':>10}  (skipped: n<{MIN_OBS_GARCH})")
                all_res[period_name][m] = (np.nan, np.nan)
            else:
                v = res[m].loc["Portfolio", "VaR_99"]
                e = res[m].loc["Portfolio", "ES_975"]
                print(f"  {m:<14} ${v:>9,.0f}  ${e:>9,.0f}")
                all_res[period_name][m] = (v, e)
                rows_csv.append({"Period": period_name, "Start": start, "End": end,
                                 "N_obs": n, "Method": m, "VaR_99": v, "ES_975": e})

    # ── Save CSV ────────────────────────────────────────────────────────────────
    pd.DataFrame(rows_csv).to_csv(DATA_DIR / "period_comparison_results.csv", index=False)

    # ── Plot 16: grouped bar chart (periods on x-axis, bars per method) ─────────
    valid_periods = [p for p in PERIODS if p in all_res]
    x = np.arange(len(valid_periods))
    n_m = len(methods)
    w   = 0.8 / n_m

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, measure_idx, measure_label in zip(axes, [0, 1], ["VaR 99%", "ES 97.5%"]):
        for i, (m, c) in enumerate(zip(methods, colours)):
            vals = [all_res[p][m][measure_idx] for p in valid_periods]
            offset = (i - (n_m - 1) / 2) * w
            ax.bar(x + offset, vals, w, label=m, color=c, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_periods, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Dollar Loss ($)")
        ax.set_title(f"Portfolio {measure_label} by Period & Method", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("VaR & ES Across Market Regimes\n"
                 "(full / pre-COVID / COVID crisis / post-COVID / pre-2025 / 2025 shock)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "16_period_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: 16_period_comparison.png")


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

    plot_var_es_comparison(summary)
    compare_named_periods(force)

    return summary

if __name__ == "__main__":
    estimate_var_es()
