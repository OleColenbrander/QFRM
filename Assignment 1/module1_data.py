"""
Module 1: Data Acquisition & Preprocessing
=========================

Purpose
-------
Fetch, clean, synchronise, and compute returns for a multi-asset portfolio:
  - Equities  : AAPL, MSFT, GOOGL
  - Index     : ^GSPC (S&P 500)
  - Loan proxy: ^IRX  (13-week US T-Bill yield, floating-rate loan proxy;
                        yield *changes* replace log-returns for this series)

Sample period: 2016-01-01 to 2026-03-31  (~10 years of daily data)

Design
------
All logic lives in PortfolioDataPipeline so downstream modules simply call
    prices, returns = load_or_build()

Synchronisation & Cleaning Methodology
---------------------------------------
  - We use forward-fill (last-observation-carried-forward) to handle missing
    values. Missing days are attributable to public holidays where one exchange
    is closed while others remain open. On such days the asset price is
    unchanged, so the return is zero.
  - The number of filled observations is logged for transparency.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

START_DATE = "2016-01-01"
END_DATE   = "2026-03-31"

EQUITY_TICKERS = ("AAPL", "MSFT", "GOOGL", "^GSPC")
RATE_TICKERS   = ("^IRX",)
ALL_TICKERS    = EQUITY_TICKERS + RATE_TICKERS

DATA_DIR    = Path(__file__).parent / "data"
PRICES_CSV  = DATA_DIR / "raw_prices.csv"
RETURNS_CSV = DATA_DIR / "returns.csv"
PLOTS_DIR   = DATA_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    print(f"Downloading data from Yahoo Finance ({START_DATE} to {END_DATE}) ...")
    df = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    prices = df["Close"][list(ALL_TICKERS)]
    prices.index = pd.to_datetime(prices.index)
    print(f"  Downloaded {len(prices)} rows.")
    return prices.sort_index()


def synchronise(prices):
    n_missing = int(prices.isna().sum().sum())
    synced    = prices.ffill()
    print(f"Synchronised: {n_missing} missing values forward-filled.")
    return synced


def compute_returns(prices):
    returns = pd.DataFrame(index=prices.index[1:])
    for ticker in EQUITY_TICKERS:
        returns[ticker] = np.log(prices[ticker] / prices[ticker].shift(1)).iloc[1:]
    for ticker in RATE_TICKERS:
        returns[ticker] = prices[ticker].diff().iloc[1:]
    print(f"Returns computed: {len(returns)} observations.")
    return returns


def plot_prices(prices):
    eq_cols = list(EQUITY_TICKERS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    normalised = 100.0 * prices[eq_cols].div(prices[eq_cols].iloc[0], axis=1)
    for col in normalised.columns:
        ax1.plot(normalised.index, normalised[col], linewidth=0.9, label=col)
    ax1.set_title("Normalised Prices (Base=100 on %s)" % str(prices.index[0].date()), fontsize=11)
    ax1.set_ylabel("Index Level")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2.plot(prices.index, prices["^IRX"], color="steelblue", linewidth=0.9,
             label="^IRX (13-wk T-Bill yield, %)")
    ax2.set_title("13-Week T-Bill Yield (%)", fontsize=11)
    ax2.set_ylabel("Yield (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Portfolio Assets  |  %s to %s" % (START_DATE, END_DATE),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_prices.png", dpi=150)
    plt.close(fig)
    print("  Price chart saved.")


def plot_returns(returns):
    cols = list(returns.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.8 * len(cols)), sharex=True)

    for ax, col in zip(axes, cols):
        color  = "firebrick" if col in RATE_TICKERS else "navy"
        ylabel = "Delta yield (p.p.)" if col in RATE_TICKERS else "Log-return"
        ax.plot(returns.index, returns[col], linewidth=0.45, color=color, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(col, fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Daily Returns / Yield Changes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_returns.png", dpi=150)
    plt.close(fig)
    print("  Returns chart saved.")


def load_or_build(force_rebuild=False):
    if not force_rebuild and PRICES_CSV.exists():
        print("Loading cached prices from disk ...")
        raw = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    else:
        raw = download_data()

    prices  = synchronise(raw)
    returns = compute_returns(prices)

    prices.to_csv(PRICES_CSV)
    returns.to_csv(RETURNS_CSV)
    print(f"Data saved to {PRICES_CSV} and {RETURNS_CSV}.")

    plot_prices(prices)
    plot_returns(returns)

    return prices, returns


if __name__ == "__main__":
    load_or_build(force_rebuild=True)
