
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
    return float(np.quantile(losses, alpha))


def build_nonoverlapping_losses(daily_losses: np.ndarray, h: int) -> np.ndarray:
    n_blocks = len(daily_losses) // h
    blocks   = daily_losses[: n_blocks * h].reshape(n_blocks, h)
    return blocks.sum(axis=1)


def compute_multiday_var(pnl: pd.DataFrame, losses: pd.Series) -> pd.DataFrame:
    entities = list(pnl.columns) + ["Portfolio"]

    # Drop NaN rows per component before aggregating.
    # Individual series (e.g. ASML.AS, ^IRX) may have NaN on dates where that
    # exchange was closed; pnl.sum(axis=1) skips them(skipna=True)
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
