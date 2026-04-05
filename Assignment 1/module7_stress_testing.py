

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

from module1_data import RATE_TICKERS, load_or_build
from module2_portfolio import build_portfolio, WEIGHTS, PORTFOLIO_VALUE, LOAN_DURATION

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
OUTPUT_CSV = DATA_DIR / "stress_test_results.csv"

# ── Parameters ─────────────────────────────────────────────────────────────────
ALPHA = 0.99

EQUITY_TICKERS   = ["AAPL", "MSFT", "ASML.AS", "^GSPC"]  
FX_TICKER        = "EURUSD"                                 
RATE_TICKER      = "^IRX"

# Shocks
EQUITY_SHOCKS    = [-0.40, -0.20, +0.20, +0.40]            # 
FX_SHOCKS        = [-0.20, -0.10, +0.10, +0.20]            
RATE_SHOCKS      = [-3.0,  -2.0,  +2.0,  +3.0]            
COMMODITY_SHOCKS = [-0.40, -0.20, +0.20, +0.40]            
# Note: the portfolio holds NO commodity position


# ── Core functions ─────────────────────────────────────────────────────────────

def hs_var(losses: np.ndarray, alpha: float = ALPHA) -> float:
    """Historical-simulation VaR: empirical alpha-quantile of the loss series."""
    arr = np.asarray(losses, dtype=float)
    return float(np.nanquantile(arr, alpha))


def recompute_losses(returns: pd.DataFrame) -> np.ndarray:
    pnl_cols = {}
    for ticker in WEIGHTS:
        if ticker not in returns.columns:
            continue
        notional = WEIGHTS[ticker] * PORTFOLIO_VALUE
        if ticker in RATE_TICKERS:
            pnl_cols[ticker] = -LOAN_DURATION * notional * (returns[ticker] / 100.0)
        else:
            pnl_cols[ticker] = notional * returns[ticker]
    pnl           = pd.DataFrame(pnl_cols, index=returns.index)
    portfolio_pnl = pnl.sum(axis=1)
    return (-portfolio_pnl).values


def strip_nan(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


# ── Shock applicators ──────────────────────────────────────────────────────────

def _shock_indices(n: int, n_shocks: int) -> np.ndarray:
    """Return n_shocks indices spread evenly across a series of length n."""
    return np.round(np.linspace(0, n - 1, n_shocks)).astype(int)


def equity_shocked(returns: pd.DataFrame, shock: float, n_shocks: int = 5) -> pd.DataFrame:
    """
    Add a log-return shock to all equity tickers on n_shocks evenly-spaced days.
    Models the stress event occurring n_shocks times across the historical window
    (e.g., n_shocks=5 over ~10 years ≈ once every two years).
    """
    r = returns.copy()
    idx = _shock_indices(len(r), n_shocks)
    for t in EQUITY_TICKERS:
        if t in r.columns:
            r[t].iloc[idx] += shock
    return r


def fx_shocked(returns: pd.DataFrame, shock: float, n_shocks: int = 5) -> pd.DataFrame:
    """
    Add a EUR/USD shock to the EURUSD column on n_shocks evenly-spaced days.
    ASML.AS carries only the local EUR stock return; the EUR/USD FX return
    is modelled as a separate risk factor (EURUSD column, same $200k notional).
    """
    r = returns.copy()
    idx = _shock_indices(len(r), n_shocks)
    if FX_TICKER in r.columns:
        r[FX_TICKER].iloc[idx] += shock
    return r


def rate_shocked(returns: pd.DataFrame, shock_pp: float, n_shocks: int = 5) -> pd.DataFrame:
    """
    Apply a parallel yield shift of 'shock_pp' pp to ^IRX on n_shocks
    evenly-spaced days across the historical window.
    """
    r = returns.copy()
    idx = _shock_indices(len(r), n_shocks)
    if RATE_TICKER in r.columns:
        r[RATE_TICKER].iloc[idx] += shock_pp
    return r


# ── Instantaneous P&L (analytic) ───────────────────────────────────────────────

def inst_pnl_equity(shock: float) -> float:
    """Dollar P&L of a uniform equity shock hitting today's portfolio."""
    return sum(WEIGHTS[t] * PORTFOLIO_VALUE * shock for t in EQUITY_TICKERS if t in WEIGHTS)


def inst_pnl_fx(shock: float) -> float:
    """Dollar P&L of an FX shock on the ASML.AS position."""
    return WEIGHTS[FX_TICKER] * PORTFOLIO_VALUE * shock


def inst_pnl_rate(shock_pp: float) -> float:
    """Dollar P&L of a parallel rate shift on the ^IRX position."""
    notional = WEIGHTS[RATE_TICKER] * PORTFOLIO_VALUE
    return -LOAN_DURATION * notional * (shock_pp / 100.0)


# ── Stress test runner ─────────────────────────────────────────────────────────

def run_stress_tests(returns: pd.DataFrame, baseline_losses: np.ndarray,
                     n_shocks: int = 5) -> pd.DataFrame:

    base_var = hs_var(baseline_losses)
    rows = []

    # ── 1. Equity shocks ───────────────────────────────────────────────────────
    for s in EQUITY_SHOCKS:
        ipnl        = inst_pnl_equity(s)
        s_losses    = recompute_losses(equity_shocked(returns, s, n_shocks=n_shocks))
        s_var       = hs_var(s_losses)
        label       = f"{'+' if s >= 0 else ''}{s*100:.0f}%"
        rows.append(_row("Equity (all)", label, ", ".join(EQUITY_TICKERS),
                         base_var, s_var, ipnl))

    # ── 2. FX shocks ──────────────────────────────────────────────────────────
    for s in FX_SHOCKS:
        ipnl        = inst_pnl_fx(s)
        s_losses    = recompute_losses(fx_shocked(returns, s, n_shocks=n_shocks))
        s_var       = hs_var(s_losses)
        severity    = "major" if abs(s) <= 0.10 else "extreme"
        label       = f"{'+' if s >= 0 else ''}{s*100:.0f}% [{severity}]"
        rows.append(_row("FX – EUR/USD", label, FX_TICKER,
                         base_var, s_var, ipnl))

    # ── 3. Interest rate shocks ───────────────────────────────────────────────
    for s in RATE_SHOCKS:
        ipnl        = inst_pnl_rate(s)
        s_losses    = recompute_losses(rate_shocked(returns, s, n_shocks=n_shocks))
        s_var       = hs_var(s_losses)
        label       = f"{'+' if s >= 0 else ''}{s:.0f} pp"
        rows.append(_row("Interest Rate (^IRX)", label, RATE_TICKER,
                         base_var, s_var, ipnl))

    # ── 4. Commodity shocks ───────────────────────────────────────────────────
    # The portfolio has no commodity positions.
    for s in COMMODITY_SHOCKS:
        label = f"{'+' if s >= 0 else ''}{s*100:.0f}%"
        rows.append({
            "Category":           "Commodity",
            "Shock":              label,
            "Tickers":            "— no exposure —",
            "Inst_PnL":           0.0,
            "Baseline_VaR":       base_var,
            "Stressed_VaR":       base_var,
            "Delta_VaR":          0.0,
            "Delta_VaR_pct":      0.0,
            "Analytic_Delta_VaR": 0.0,
            "Residual":           0.0,
        })

    return pd.DataFrame(rows)


def _row(category, shock_label, tickers, base_var, s_var, ipnl):
    delta = s_var - base_var
    pct   = delta / base_var * 100 if base_var != 0 else np.nan
    return {
        "Category":      category,
        "Shock":         shock_label,
        "Tickers":       tickers,
        "Inst_PnL":      round(ipnl,     2),
        "Baseline_VaR":  round(base_var, 2),
        "Stressed_VaR":  round(s_var,    2),
        "Delta_VaR":     round(delta,    2),
        "Delta_VaR_pct": round(pct,      2),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_DOLLAR = mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
_COLORS = {"pos": "#d32f2f", "neg": "#1565c0", "zero": "#9e9e9e"}


def _bar_color(v):
    if v > 0:  return _COLORS["pos"]
    if v < 0:  return _COLORS["neg"]
    return _COLORS["zero"]


def plot_tornado(results: pd.DataFrame) -> None:
    """
    Tornado chart: horizontal bars showing ΔVaR for every stressed scenario,
    sorted by absolute impact.  Red = VaR increases (risk worsens),
    Blue = VaR decreases (risk improves).
    """
    df = results.copy()
    df["Label"] = df["Category"] + "  " + df["Shock"]
    df = df.reindex(df["Delta_VaR"].abs().sort_values().index)

    colors = [_bar_color(v) for v in df["Delta_VaR"]]
    fig, ax = plt.subplots(figsize=(13, max(6, len(df) * 0.48)))
    bars = ax.barh(df["Label"], df["Delta_VaR"], color=colors, alpha=0.85, height=0.7)

    ax.axvline(0, color="black", linewidth=1.0)
    for bar, val in zip(bars, df["Delta_VaR"]):
        offset = 500 if val >= 0 else -500
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"${val:+,.0f}", va="center", ha=ha, fontsize=8)

    ax.set_xlabel("ΔVaR = Stressed VaR − Baseline VaR  ($)", fontsize=10)
    ax.set_title("Stress-Test Tornado Chart  |  99% Historical-Simulation VaR\n"
                 "Red = risk increases  |  Blue = risk decreases",
                 fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(_DOLLAR)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "12_stress_tornado.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_by_category(results: pd.DataFrame, baseline_var: float) -> None:
    cats = list(results["Category"].unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 5), sharey=False)
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        sub    = results[results["Category"] == cat].reset_index(drop=True)
        x      = np.arange(len(sub))
        width  = 0.35

        ax.axhline(baseline_var, color="black", linewidth=1.2,
                   linestyle="--", label=f"Baseline VaR ${baseline_var:,.0f}")
        for i, row in sub.iterrows():
            color = _bar_color(row["Delta_VaR"])
            ax.bar(i, row["Stressed_VaR"], color=color, alpha=0.75, width=0.6)
            ax.text(i, row["Stressed_VaR"] + baseline_var * 0.01,
                    f"${row['Stressed_VaR']:,.0f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Shock"], fontsize=8, rotation=15)
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.set_ylabel("99% VaR ($)" if cat == cats[0] else "")
        ax.yaxis.set_major_formatter(_DOLLAR)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Stressed 99% HS VaR by Scenario Category  |  V₀ = $1,000,000",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "13_stress_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_distribution_overlay(returns: pd.DataFrame,
                               baseline_losses: np.ndarray,
                               baseline_var: float) -> None:

    scenarios = [
        ("Equity −40%",  recompute_losses(equity_shocked(returns, -0.40)), "#b71c1c"),
        ("Equity −20%",  recompute_losses(equity_shocked(returns, -0.20)), "#e57373"),
        ("FX −10%",      recompute_losses(fx_shocked(returns,    -0.10)), "#0288d1"),
        ("FX −20%",      recompute_losses(fx_shocked(returns,    -0.20)), "#01579b"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    # Baseline
    ax.hist(strip_nan(baseline_losses), bins=80, density=True,
            color="silver", alpha=0.6, label="Baseline", zorder=2)
    ax.axvline(baseline_var, color="black", linewidth=1.8, linestyle="--",
               label=f"Baseline VaR ${baseline_var:,.0f}", zorder=5)

    for label, losses, color in scenarios:
        arr  = strip_nan(losses)
        svar = hs_var(arr)
        ax.hist(arr, bins=80, density=True, color=color, alpha=0.30,
                label=f"{label}  VaR=${svar:,.0f}", zorder=3)
        ax.axvline(svar, color=color, linewidth=1.4, linestyle=":", zorder=4)

    ax.set_xlabel("Portfolio Loss ($)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Baseline vs. Stressed Loss Distributions  |  99% HS VaR",
                 fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(_DOLLAR)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = PLOTS_DIR / "14_stress_distribution_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_var_sensitivity(results: pd.DataFrame) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cat, shocks, xlabel in [
        (axes[0], "Equity (all)",  EQUITY_SHOCKS, "Equity Price Shock (%)"),
        (axes[1], "FX – EUR/USD",  FX_SHOCKS,     "EUR/USD Move (%)"),
    ]:
        sub    = results[results["Category"] == cat].copy()
        s_pct  = [float(s.split("%")[0].split("[")[0]) for s in sub["Shock"]]
        s_var  = sub["Stressed_VaR"].values
        base   = sub["Baseline_VaR"].iloc[0]

        ax.plot(s_pct, s_var,     "o-", color="firebrick",  linewidth=2, markersize=6,
                label="Stressed VaR")
        ax.axhline(base,            color="black",     linewidth=1.2, linestyle="--",
                   label=f"Baseline VaR ${base:,.0f}")
        ax.axhline(0,               color="gray",      linewidth=0.8, linestyle=":")

        # Shade regions
        ax.fill_between(s_pct, 0, base, alpha=0.05, color="green",
                         label="Risk-free zone (VaR < 0)")

        for sp, sv in zip(s_pct, s_var):
            ax.annotate(f"${sv:,.0f}", (sp, sv),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("99% VaR ($)", fontsize=10)
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(_DOLLAR)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VaR Sensitivity to Equity and FX Stress Shocks",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "15_stress_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_results(results: pd.DataFrame) -> None:
    pd.set_option("display.float_format", "${:,.2f}".format)
    categories = results["Category"].unique()

    for cat in categories:
        sub = results[results["Category"] == cat].copy()
        print(f"\n{'─'*72}")
        print(f"  {cat}")
        print(f"{'─'*72}")
        print(f"{'Shock':<22} {'Inst. P&L':>14} {'Base VaR':>12} "
              f"{'Stressed VaR':>14} {'ΔVaR':>12} {'ΔVaR%':>8}")
        print("  " + "·" * 70)
        for _, row in sub.iterrows():
            delta_str = f"${row['Delta_VaR']:+,.0f}"
            pct_str   = f"{row['Delta_VaR_pct']:+.1f}%" if not np.isnan(row["Delta_VaR_pct"]) else "—"
            print(f"  {row['Shock']:<20} ${row['Inst_PnL']:>13,.0f} "
                  f"${row['Baseline_VaR']:>11,.0f} "
                  f"${row['Stressed_VaR']:>13,.0f} "
                  f"{delta_str:>12} {pct_str:>8}")

    pd.reset_option("display.float_format")


def print_interpretation(results: pd.DataFrame) -> None:
    base_var = results["Baseline_VaR"].iloc[0]

    eq_40_down = results[
        (results["Category"] == "Equity (all)") & (results["Shock"] == "-40%")
    ]["Delta_VaR"].values[0]

    fx_10_down = results[
        (results["Category"] == "FX – EUR/USD") &
        results["Shock"].str.startswith("-10%")
    ]["Delta_VaR"].values[0]

    rate_2_up = results[
        (results["Category"] == "Interest Rate (^IRX)") & (results["Shock"] == "+2 pp")
    ]["Delta_VaR"].values[0]

    eq_exposure  = sum(WEIGHTS[t] for t in EQUITY_TICKERS) * PORTFOLIO_VALUE
    fx_exposure  = WEIGHTS[FX_TICKER] * PORTFOLIO_VALUE
    rate_dv01    = LOAN_DURATION * WEIGHTS[RATE_TICKER] * PORTFOLIO_VALUE / 100.0  # per 1 pp
    print(f"{'═'*72}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run(force_rebuild: bool = False) -> pd.DataFrame:
    print("=" * 72)
    print("Module 7 – Stress Testing")
    print("=" * 72)

    _, returns            = load_or_build(force_rebuild=force_rebuild)
    pnl, _, losses        = build_portfolio(force_rebuild=force_rebuild)
    baseline_losses       = strip_nan(losses.values.astype(float))
    baseline_var          = hs_var(baseline_losses)

    print(f"\nBaseline 99% HS VaR (1-day): ${baseline_var:,.2f}")
    print(f"Sample size (clean obs)     : {len(baseline_losses):,}")

    print("\nRunning stress scenarios ...")
    results = run_stress_tests(returns, losses.values.astype(float), n_shocks=5)

    print_results(results)
    print_interpretation(results)

    print("\nGenerating plots ...")
    plot_tornado(results)
    plot_by_category(results, baseline_var)
    plot_distribution_overlay(returns, baseline_losses, baseline_var)
    plot_var_sensitivity(results)

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved -> {OUTPUT_CSV}")

    return results


if __name__ == "__main__":
    run()
