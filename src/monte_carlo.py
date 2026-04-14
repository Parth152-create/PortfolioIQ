import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import N_MONTE_CARLO, RISK_FREE_RATE

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_simulation(ann_returns: pd.Series, cov_matrix: pd.DataFrame,
                   n_portfolios: int = N_MONTE_CARLO) -> pd.DataFrame:
    """
    Randomly generate N portfolios and compute their return/volatility/sharpe.
    This creates the cloud of points behind the efficient frontier.
    """
    n_assets = len(ann_returns)
    results = []

    for _ in range(n_portfolios):
        # random weights that sum to 1
        weights = np.random.dirichlet(np.ones(n_assets))

        ret = np.dot(weights, ann_returns)
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (ret - RISK_FREE_RATE) / vol

        results.append({
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe,
            "weights": weights
        })

    return pd.DataFrame(results)


def plot_monte_carlo(simulation: pd.DataFrame, frontier: pd.DataFrame,
                     max_sharpe_point: dict, min_var_point: dict, save=True):
    fig, ax = plt.subplots(figsize=(12, 7))

    # simulation cloud
    sc = ax.scatter(
        simulation["volatility"], simulation["return"],
        c=simulation["sharpe"], cmap="viridis",
        alpha=0.3, s=8, zorder=1, label=f"{len(simulation):,} random portfolios"
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # efficient frontier line on top
    ax.plot(
        frontier["volatility"], frontier["return"],
        color="white", linewidth=2.5, zorder=3, label="Efficient frontier"
    )

    # max sharpe star
    ax.scatter(
        max_sharpe_point["volatility"], max_sharpe_point["return"],
        marker="*", color="gold", s=400, zorder=5,
        edgecolors="black", linewidths=0.5, label="Max Sharpe"
    )

    # min variance diamond
    ax.scatter(
        min_var_point["volatility"], min_var_point["return"],
        marker="D", color="cyan", s=120, zorder=5,
        edgecolors="black", linewidths=0.5, label="Min Variance"
    )

    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.set_title(f"Monte Carlo Simulation — {len(simulation):,} Portfolios\nPortfolioIQ",
                 fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "monte_carlo.png"
        plt.savefig(path, dpi=150)
        print(f"[saved] {path}")
    plt.show()


def get_best_simulated(simulation: pd.DataFrame,
                        ann_returns: pd.Series) -> pd.Series:
    """Return the highest Sharpe portfolio from the simulation."""
    best_idx = simulation["sharpe"].idxmax()
    best = simulation.iloc[best_idx]
    tickers = ann_returns.index.tolist()
    print("\n--- Best Simulated Portfolio ---")
    print(f"  Return:     {best['return']:.2%}")
    print(f"  Volatility: {best['volatility']:.2%}")
    print(f"  Sharpe:     {best['sharpe']:.4f}")
    for t, w in zip(tickers, best["weights"]):
        if w > 0.01:
            print(f"  {t}: {w:.1%}")
    return best