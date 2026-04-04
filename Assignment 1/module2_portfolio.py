"""
Module 2: Portfolio Setup
=========================

Portfolio Composition & Weight Justification
---------------------------------------------
The portfolio holds five instruments with fixed weights that sum to 1.0 (could be adapted):

    AAPL     0.25   (equity)
    MSFT     0.25   (equity)
    ASML.AS  0.20   (foreign equity, EUR)
    ^GSPC    0.20   (equity index)
    ^IRX     0.10   (floating-rate loan proxy)

Weight rationale:

1. AAPL (25%) and MSFT (25%): The two largest constituents of the S&P 500 by
   market capitalisation over the sample period.  Equal weighting between them
   avoids an arbitrary tilt toward either and is consistent with a large-cap
   growth allocation.

2. ASML.AS (20%): Replaces GOOGL to introduce foreign currency exposure (EUR) 
   and test stress-testing scenarios for FX rates, as outlined in the assignment.
   The return integrates both local stock performance and EUR/USD fluctuations.

3. ^GSPC (20%): A broad-market index position (e.g. an S&P 500 ETF/futures)
   provides systematic risk exposure, introduces diversification across the
   ~500 remaining index constituents, and grounds the portfolio in a
   widely-used risk benchmark.  It also gives the portfolio explicit exposure
   to index-level shocks that individual stock VaR measures might understate.

4. ^IRX / Loan (10%): A floating-rate loan component proxied by the 13-week
   US Treasury Bill yield.  The 10% allocation is intentionally modest: the
   interest-rate risk factor is structurally different from equity risk, and
   a smaller weight avoids the loan dominating the portfolio P&L during
   rate-spike episodes while still ensuring the portfolio has meaningful
   exposure to interest-rate uncertainty, as required by the assignment.

Weights are fixed throughout the full sample period.  No rebalancing is
modelled; this is standard practice for a first-pass VaR/ES analysis and
keeps the attribution of risk changes to market moves rather than to
portfolio composition changes.

Loan Component Modelling
------------------------
The mark-to-market daily P&L of a fixed-income position is approximated by
the first-order (duration) expansion:

    dP_loan,t = -D_mod * V_loan * (Delta_y_t / 100)

where:
    D_mod    = modified duration of the loan (years)
    V_loan   = notional value of the loan position ($)
    Delta_y_t = daily change in the ^IRX yield (in percentage points)

We use D_mod = 0.25, matching the 91-day (13-week) maturity of the T-Bill
proxy.  This is consistent with treating the loan as a floating-rate
instrument that reprices every quarter: the interest-rate sensitivity between
reset dates is captured by the time to the next repricing divided by 365.

Loss Definition
---------------
Following the lectures, the daily loss is the negative of the daily portfolio P&L:

    L_t = -PnL_t

A positive L_t means the portfolio lost money on day t.  VaR and ES are then defined as quantiles of the loss distribution.

Starting Portfolio Value
------------------------
    V_0 = $1,000,000
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from module1_data import load_or_build, RATE_TICKERS


DATA_DIR   = Path(__file__).parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"

PORTFOLIO_VALUE = 1_000_000.0
LOAN_DURATION   = 0.25

WEIGHTS = {
    "AAPL":    0.25,
    "MSFT":    0.25,
    "ASML.AS": 0.20,
    "^GSPC":   0.20,
    "^IRX":    0.10,
}


def compute_component_pnl(returns):
    pnl = pd.DataFrame(index=returns.index)
    for ticker in returns.columns:
        notional = WEIGHTS[ticker] * PORTFOLIO_VALUE
        if ticker in RATE_TICKERS:
            pnl[ticker] = -LOAN_DURATION * notional * (returns[ticker] / 100.0)
        else:
            pnl[ticker] = notional * returns[ticker]
    return pnl


def plot_pnl(pnl, portfolio_pnl):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for col in pnl.columns:
        ax1.plot(pnl.index, pnl[col], linewidth=0.5, alpha=0.8, label=col)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Daily Component P&L ($)", fontsize=11)
    ax1.set_ylabel("P&L ($)")
    ax1.legend(fontsize=8, ncol=5)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2.fill_between(portfolio_pnl.index, portfolio_pnl.clip(lower=0), 0,
                     color="seagreen", alpha=0.6, label="Gain")
    ax2.fill_between(portfolio_pnl.index, portfolio_pnl.clip(upper=0), 0,
                     color="firebrick", alpha=0.6, label="Loss")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Daily Portfolio P&L ($)", fontsize=11)
    ax2.set_ylabel("P&L ($)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Portfolio P&L  |  V0 = $1,000,000", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_portfolio_pnl.png", dpi=150)
    plt.close(fig)
    print("  P&L chart saved.")


def plot_loss_distribution(losses, portfolio_pnl):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    mu, sigma = losses.mean(), losses.std()

    ax = axes[0, 0]
    ax.plot(losses.index, losses, linewidth=0.5, color="firebrick", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Daily Loss Time Series ($)", fontsize=10)
    ax.set_ylabel("Loss ($)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0, 1]
    ax.hist(losses, bins=80, density=True, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    x = np.linspace(losses.min(), losses.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="firebrick", linewidth=1.5, label="Normal fit")
    ax.set_title("Loss Distribution with Normal Overlay", fontsize=10)
    ax.set_xlabel("Loss ($)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    standardised = (losses - mu) / sigma
    empirical_q  = np.sort(standardised.values)
    normal_q     = stats.norm.ppf(np.linspace(0.001, 0.999, len(standardised)))
    q_range      = [min(normal_q.min(), empirical_q.min()), max(normal_q.max(), empirical_q.max())]
    ax.scatter(normal_q, empirical_q, s=2, color="navy", alpha=0.5)
    ax.plot(q_range, q_range, color="firebrick", linewidth=1.2, label="45-degree line")
    ax.set_title("QQ-Plot of Standardised Losses vs Normal", fontsize=10)
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    cum_pnl = portfolio_pnl.cumsum()
    ax.plot(cum_pnl.index, cum_pnl, linewidth=0.9, color="navy")
    ax.fill_between(cum_pnl.index, cum_pnl, 0, where=(cum_pnl < 0),
                    color="firebrick", alpha=0.4, label="Cumulative loss")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative Portfolio P&L ($)", fontsize=10)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Loss Distribution Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_loss_distribution.png", dpi=150)
    plt.close(fig)
    print("  Loss distribution chart saved.")


def build_portfolio(force_rebuild=False):
    _, returns = load_or_build()

    pnl           = compute_component_pnl(returns)
    portfolio_pnl = pnl.sum(axis=1).rename("Portfolio_PnL")
    losses        = (-portfolio_pnl).rename("Loss")

    output = pnl.copy()
    output["Portfolio_PnL"] = portfolio_pnl
    output["Loss"]          = losses
    output.to_csv(LOSSES_CSV)
    print(f"Portfolio P&L and losses saved -> {LOSSES_CSV}")

    plot_pnl(pnl, portfolio_pnl)
    plot_loss_distribution(losses, portfolio_pnl)

    return pnl, portfolio_pnl, losses


if __name__ == "__main__":
    build_portfolio()
