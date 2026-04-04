import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from module1_data import load_or_build, RATE_TICKERS

DATA_DIR   = Path(__file__).parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"

# basic vars
PORTFOLIO_VALUE = 1_000_000.0
LOAN_DURATION   = 0.25 # approx 3 months

# weights for the assets
WEIGHTS = {
    "AAPL":    0.25,
    "MSFT":    0.25,
    "ASML.AS": 0.20,
    "^GSPC":   0.20,
    "^IRX":    0.10,
}

def compute_component_pnl(returns):
    pnl = pd.DataFrame(index=returns.index)
    for t in returns.columns:
        # calc notional per asset
        notional = WEIGHTS[t] * PORTFOLIO_VALUE
        
        if t in RATE_TICKERS:
            # loan duration formula for PNL
            pnl[t] = -LOAN_DURATION * notional * (returns[t] / 100.0)
        else:
            pnl[t] = notional * returns[t]
            
    return pnl

def plot_pnl(pnl, portfolio_pnl):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    for c in pnl.columns:
        ax1.plot(pnl.index, pnl[c], label=c)
    ax1.axhline(0, color="black")
    ax1.legend()
    ax1.grid(True)

    # plot the total pnl 
    ax2.plot(portfolio_pnl.index, portfolio_pnl)
    ax2.axhline(0, color="black")
    ax2.legend(['Total PnL'])
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_portfolio_pnl.png")
    plt.close()


def plot_loss_distribution(losses, portfolio_pnl):
    # distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    mu = losses.mean()
    sigma = losses.std()

    axes[0, 0].plot(losses.index, losses, color="red")
    axes[0, 0].grid(True)

    axes[0, 1].hist(losses, bins=50, density=True)
    
    # fit a normal dist
    x = np.linspace(losses.min(), losses.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), color="red")

    # qq plot against standard normal
    std = (losses - mu) / sigma
    emp_q = np.sort(std.values)
    norm_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(std)))
    
    axes[1, 0].scatter(norm_q, emp_q, s=2)
    axes[1, 0].plot([-3, 3], [-3, 3], color="red") # 45 dev line
    axes[1, 0].grid(True)

    
    cum = portfolio_pnl.cumsum()
    axes[1, 1].plot(cum.index, cum)
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_loss_distribution.png")
    plt.close()

def build_portfolio(force_rebuild=False):
    _, ret = load_or_build()

    pnl = compute_component_pnl(ret)
    total_pnl = pnl.sum(axis=1).rename("Portfolio_PnL")
    
    # loss is negative pnl
    losses = (-total_pnl).rename("Loss")

    out = pnl.copy()
    out["Portfolio_PnL"] = total_pnl
    out["Loss"] = losses
    out.to_csv(LOSSES_CSV)
    
    plot_pnl(pnl, total_pnl)
    plot_loss_distribution(losses, total_pnl)

    return pnl, total_pnl, losses


if __name__ == "__main__":
    build_portfolio()
