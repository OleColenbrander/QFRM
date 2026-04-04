import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

START_DATE = "2016-01-01"
END_DATE   = "2026-03-31"

EQUITY_TICKERS = ("AAPL", "MSFT", "ASML.AS", "^GSPC")
RATE_TICKERS   = ("^IRX",)
FX_TICKERS     = ("EURUSD=X",)

ALL_TICKERS = EQUITY_TICKERS + RATE_TICKERS + FX_TICKERS

# make dirs
DATA_DIR    = Path(__file__).parent / "data"
PRICES_CSV  = DATA_DIR / "raw_prices.csv"
RETURNS_CSV = DATA_DIR / "returns.csv"
PLOTS_DIR   = DATA_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    print(f"Downloading from {START_DATE} to {END_DATE}")
    df = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    prices = df["Close"][list(ALL_TICKERS)]
    prices.index = pd.to_datetime(prices.index)
    
    print(f"Downloaded {len(prices)} rows.")
    return prices.sort_index()

def synchronise(prices):
    # fill missing values because sometimes markets are closed
    n_missing = int(prices.isna().sum().sum())
    synced = prices.ffill()
    print(f"Fixed {n_missing} missing values")
    return synced



def compute_returns(prices):
    # calculate logs
    returns = pd.DataFrame(index=prices.index[1:])
    
    for t in EQUITY_TICKERS:
        returns[t] = np.log(prices[t] / prices[t].shift(1)).iloc[1:]
        
    # add fx risk to ASML 
    if "ASML.AS" in returns.columns and "EURUSD=X" in prices.columns:
        fx = np.log(prices["EURUSD=X"] / prices["EURUSD=X"].shift(1)).iloc[1:]
        returns["ASML.AS"] = returns["ASML.AS"] + fx

    for t in RATE_TICKERS:
        returns[t] = prices[t].diff().iloc[1:]
        
    returns = returns.dropna()
    print(f"returns calculated, len: {len(returns)}")
    return returns

def plot_prices(prices):
    eq = list(EQUITY_TICKERS)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # normalize starts at 100
    norm = 100.0 * prices[eq].div(prices[eq].iloc[0], axis=1)
    for c in norm.columns:
        a1.plot(norm.index, norm[c], label=c)
    a1.legend()
    a1.grid(True)

    a2.plot(prices.index, prices["^IRX"], label="IRX")
    a2.legend()
    a2.grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_prices.png")
    plt.close(fig)


def plot_returns(returns):
    col_lst = list(returns.columns)
    fig, axes = plt.subplots(len(col_lst), 1, figsize=(10, 2 * len(col_lst)), sharex=True)
    
    for ax, c in zip(axes, col_lst):
        if c in RATE_TICKERS: c_color = "red" 
        else: c_color = "blue"
        ax.plot(returns.index, returns[c], color=c_color)
        ax.axhline(0, color="black")
        ax.set_title(c)
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_returns.png")
    plt.close(fig)

def load_or_build(force_rebuild=False):
    if not force_rebuild and PRICES_CSV.exists():
        raw = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    else:
        raw = download_data()
        
    prices  = synchronise(raw)
    returns = compute_returns(prices)

    prices.to_csv(PRICES_CSV)
    returns.to_csv(RETURNS_CSV)

    plot_prices(prices)
    plot_returns(returns)

    return prices, returns

if __name__ == "__main__":
    load_or_build(force_rebuild=True)
