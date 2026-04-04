"""
Module 6: Multi-Day VaR – Historical Simulation vs. Square-Root-of-Time Rule
=============================================================================

Objective
---------
Construct 1-, 5- and 10-day 99% VaR using non-overlapping return blocks at the
respective frequency (direct historical simulation), then compare with the
scaled 1-day VaR obtained via the square-root-of-time (SQRT) rule:

    VaR(h) ≈ VaR(1) × √h

The comparison reveals whether the iid / normality assumption embedded in the
SQRT rule holds for this portfolio.

Approach
--------
1. Non-overlapping h-day losses:
   - Form non-overlapping blocks of h consecutive daily portfolio P&L values.
   - Sum each block → h-day P&L; negate → h-day loss.
   - The 99th percentile of the resulting empirical distribution is the
     direct h-day historical-simulation VaR.

2. Square-root-of-time (SQRT) rule:
   - Scale the 1-day historical-simulation VaR by √h.

Both approaches are applied at the total portfolio level and per component.
The ratio  HS_VaR(h) / (VaR(1) × √h)  quantifies how well the SQRT rule fits:
  ratio > 1  → SQRT rule underestimates true multi-day risk (fat tails / autocorrelation)
  ratio < 1  → SQRT rule overestimates
  ratio ≈ 1  → SQRT rule is a decent approximation
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from module2_portfolio import build_portfolio

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
OUTPUT_CSV = DATA_DIR / "multiday_var_results.csv"

ALPHA    = 0.99
HORIZONS = [1, 5, 10]


# ── Core functions ─────────────────────────────────────────────────────────────

def historical_var(losses: np.ndarray, alpha: float = ALPHA) -> float:
    """99th-percentile empirical quantile (historical-simulation VaR)."""
    return float(np.quantile(losses, alpha))


def build_nonoverlapping_losses(daily_losses: np.ndarray, h: int) -> np.ndarray:
    """
    Aggregate daily losses into non-overlapping h-day blocks by summing.
    Tail observations that do not complete a full block are discarded.
    """
    n_blocks = len(daily_losses) // h
    blocks   = daily_losses[: n_blocks * h].reshape(n_blocks, h)
    return blocks.sum(axis=1)


def compute_multiday_var(pnl: pd.DataFrame, losses: pd.Series) -> pd.DataFrame:
    """
    Compute direct HS VaR and SQRT-rule VaR for every horizon and entity.

    Returns a tidy DataFrame with columns:
        Horizon | Entity | N_obs | HS_VaR | SQRT_VaR | Ratio_HS_over_SQRT
    """
    entities = list(pnl.columns) + ["Portfolio"]

    # Drop NaN rows per component before aggregating.
    # Individual series (e.g. ASML.AS, ^IRX) may have NaN on dates where that
    # exchange was closed; pnl.sum(axis=1) skips them silently (skipna=True)
    # so the portfolio series is clean while individual component arrays are not.
    # np.quantile propagates NaN, so we must strip it per series first.
    daily_loss: dict[str, np.ndarray] = {}
    for c in pnl.columns:
        arr = -pnl[c].values.astype(float)
        daily_loss[c] = arr[~np.isnan(arr)]
    port_arr = losses.values.astype(float)
    daily_loss["Portfolio"] = port_arr[~np.isnan(port_arr)]

    # 1-day HS VaR used as the base for the SQRT rule
    var1 = {e: historical_var(daily_loss[e]) for e in entities}

    rows = []
    for h in HORIZONS:
        for entity in entities:
            h_losses = build_nonoverlapping_losses(daily_loss[entity], h)
            hs_var   = historical_var(h_losses)
            sqrt_var = var1[entity] * np.sqrt(h)
            ratio    = hs_var / sqrt_var if sqrt_var != 0 else np.nan
            rows.append({
                "Horizon":            h,
                "Entity":             entity,
                "N_obs":              len(h_losses),
                "HS_VaR":             hs_var,
                "SQRT_VaR":           sqrt_var,
                "Ratio_HS_over_SQRT": ratio,
            })

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_portfolio_comparison(results: pd.DataFrame) -> None:
    """Grouped bar chart: direct HS VaR vs. SQRT-rule VaR for each horizon."""
    port = results[results["Entity"] == "Portfolio"].set_index("Horizon")

    x = np.arange(len(HORIZONS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w / 2, [port.loc[h, "HS_VaR"]   for h in HORIZONS], w,
                   label="Direct HS VaR",    color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w / 2, [port.loc[h, "SQRT_VaR"] for h in HORIZONS], w,
                   label="SQRT-of-time VaR", color="firebrick",  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}-day" for h in HORIZONS], fontsize=11)
    ax.set_ylabel("Dollar Loss ($)", fontsize=10)
    ax.set_title("Portfolio 99% VaR: Direct HS vs. Square-Root-of-Time Rule\n"
                 "V₀ = $1,000,000", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    for bar in list(bars1) + list(bars2):
        h_val = bar.get_height()
        ax.annotate(f"${h_val:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h_val),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = PLOTS_DIR / "08_multiday_var_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_loss_distributions(daily_losses: np.ndarray, results: pd.DataFrame) -> None:
    """
    One histogram per horizon showing the empirical h-day loss distribution
    with both VaR estimates marked as vertical lines.
    """
    port_rows = results[results["Entity"] == "Portfolio"].set_index("Horizon")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, h in zip(axes, HORIZONS):
        h_losses = build_nonoverlapping_losses(daily_losses, h)
        ax.hist(h_losses, bins=40, density=True, color="silver",
                edgecolor="white", linewidth=0.3,
                label=f"n = {len(h_losses)} blocks")

        hs_var   = port_rows.loc[h, "HS_VaR"]
        sqrt_var = port_rows.loc[h, "SQRT_VaR"]

        ax.axvline(hs_var,   color="steelblue", linewidth=2.0,
                   linestyle="--", label=f"HS VaR = ${hs_var:,.0f}")
        ax.axvline(sqrt_var, color="firebrick",  linewidth=2.0,
                   linestyle="-.",  label=f"SQRT VaR = ${sqrt_var:,.0f}")

        ax.set_title(f"{h}-Day Portfolio Losses", fontsize=10, fontweight="bold")
        ax.set_xlabel("Loss ($)")
        ax.set_ylabel("Density" if h == 1 else "")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Non-Overlapping h-Day Loss Distributions with 99% VaR Estimates",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "09_multiday_loss_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_ratio_by_entity(results: pd.DataFrame, entities: list) -> None:
    """
    Grouped bar chart of the ratio  HS_VaR / SQRT_VaR  for each entity and
    horizon.  A ratio of 1 (dashed line) means the SQRT rule is exact.
    """
    x       = np.arange(len(entities))
    width   = 0.25
    offsets = [-width, 0, width]
    palette = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for h, offset, color in zip(HORIZONS, offsets, palette):
        subset = results[results["Horizon"] == h].set_index("Entity")
        ratios = [subset.loc[e, "Ratio_HS_over_SQRT"] for e in entities]
        ax.bar(x + offset, ratios, width, label=f"{h}-day", color=color, alpha=0.82)

    ax.axhline(1.0, color="black", linewidth=1.4, linestyle="--",
               label="Ratio = 1  (SQRT rule exact)")
    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=9, rotation=15)
    ax.set_ylabel("HS VaR / SQRT-of-time VaR", fontsize=10)
    ax.set_title("Ratio of Direct HS VaR to Square-Root-of-Time VaR by Entity & Horizon",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "10_multiday_var_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_scaling_curve(results: pd.DataFrame) -> None:
    """
    Line chart for the portfolio: actual HS VaR vs. SQRT-scaled VaR across
    horizons, plus the theoretical √h scaling starting from VaR(1).
    """
    port  = results[results["Entity"] == "Portfolio"].set_index("Horizon")
    var1  = port.loc[1, "HS_VaR"]
    h_arr = np.linspace(1, 10, 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1, 5, 10], [port.loc[h, "HS_VaR"]   for h in HORIZONS],
            "o-", color="steelblue", linewidth=2.0, markersize=7,
            label="Direct HS VaR")
    ax.plot([1, 5, 10], [port.loc[h, "SQRT_VaR"] for h in HORIZONS],
            "s--", color="firebrick", linewidth=2.0, markersize=7,
            label="SQRT-of-time rule")
    ax.plot(h_arr, var1 * np.sqrt(h_arr),
            color="firebrick", linewidth=0.8, alpha=0.45,
            label="√h curve (continuous)")

    ax.set_xlabel("Horizon (days)", fontsize=10)
    ax.set_ylabel("99% VaR ($)", fontsize=10)
    ax.set_title("Portfolio 99% VaR vs. Horizon: HS vs. SQRT Rule",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xticks([1, 5, 10])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "11_multiday_var_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Diagnostics print ──────────────────────────────────────────────────────────

def print_interpretation(results: pd.DataFrame) -> None:
    port = results[results["Entity"] == "Portfolio"].set_index("Horizon")
    print("\n" + "─" * 60)
    print("  Square-Root-of-Time Rule – Interpretation")
    print("─" * 60)
    for h in [5, 10]:
        r = port.loc[h, "Ratio_HS_over_SQRT"]
        direction = "UNDERESTIMATES" if r > 1 else "OVERESTIMATES"
        print(f"  {h}-day horizon:  HS VaR / SQRT VaR = {r:.4f}"
              f"  →  SQRT rule {direction} true risk by {abs(r - 1)*100:.1f}%")
    print()
    print("  Interpretation:")
    print("  A ratio > 1 indicates the SQRT rule underestimates multi-day VaR.")
    print("  This typically arises from fat-tailed returns (Student-t behaviour)")
    print("  and/or volatility clustering (GARCH effects), which violate the iid")
    print("  normality assumption that the SQRT rule requires.")
    print("─" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(force_rebuild: bool = False) -> pd.DataFrame:
    print("=" * 60)
    print("Module 6 – Multi-Day VaR (HS vs. Square-Root-of-Time)")
    print("=" * 60)

    pnl, _, losses = build_portfolio(force_rebuild=force_rebuild)
    entities       = list(pnl.columns) + ["Portfolio"]

    results = compute_multiday_var(pnl, losses)

    # ── Summary tables ─────────────────────────────────────────────
    print("\n--- 99% VaR by Horizon and Entity ---")
    pivot = results.pivot_table(
        index="Entity", columns="Horizon",
        values=["HS_VaR", "SQRT_VaR", "Ratio_HS_over_SQRT"]
    ).reindex(entities)
    with pd.option_context("display.float_format", "${:,.2f}".format):
        print(pivot[["HS_VaR", "SQRT_VaR"]].to_string())

    print("\n--- Ratio HS_VaR / SQRT_VaR ---")
    ratio_pivot = results.pivot_table(
        index="Entity", columns="Horizon",
        values="Ratio_HS_over_SQRT"
    ).reindex(entities)
    print(ratio_pivot.to_string(float_format="{:.4f}".format))

    print_interpretation(results)

    # ── Plots ──────────────────────────────────────────────────────
    print("Generating plots ...")
    plot_portfolio_comparison(results)
    plot_loss_distributions(losses.values, results)
    plot_ratio_by_entity(results, entities)
    plot_scaling_curve(results)

    # ── Save results ───────────────────────────────────────────────
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved -> {OUTPUT_CSV}")

    return results


if __name__ == "__main__":
    run()
