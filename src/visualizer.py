import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_efficient_frontier(frontier: pd.DataFrame, max_sharpe_point: dict,
                             min_var_point: dict, save=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    # frontier scatter colored by Sharpe
    sc = ax.scatter(
        frontier["volatility"], frontier["return"],
        c=frontier["sharpe"], cmap="viridis", s=20, zorder=2
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # max sharpe star
    ax.scatter(
        max_sharpe_point["volatility"], max_sharpe_point["return"],
        marker="*", color="gold", s=300, zorder=5, label="Max Sharpe"
    )

    # min variance diamond
    ax.scatter(
        min_var_point["volatility"], min_var_point["return"],
        marker="D", color="cyan", s=100, zorder=5, label="Min Variance"
    )

    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier — PortfolioIQ")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "efficient_frontier.png"
        plt.savefig(path, dpi=150)
        print(f"[saved] {path}")
    plt.show()


def plot_allocation(weights, tickers, title="Portfolio Allocation", save=True):
    # filter out near-zero weights
    data = [(t, w) for t, w in zip(tickers, weights) if w > 0.001]
    labels, values = zip(*data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax1.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    ax1.set_title(f"{title} — Pie")

    # bar chart
    ax2.barh(labels, [v * 100 for v in values], color=colors)
    ax2.set_xlabel("Weight (%)")
    ax2.set_title(f"{title} — Weights")
    ax2.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax2.text(v * 100 + 0.5, i, f"{v:.1%}", va="center", fontsize=9)

    plt.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + ".png"
        path = PLOTS_DIR / fname
        plt.savefig(path, dpi=150)
        print(f"[saved] {path}")
    plt.show()


def plot_correlation_heatmap(daily_returns: pd.DataFrame, save=True):
    corr = daily_returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title("Asset Correlation Matrix — PortfolioIQ")
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "correlation_heatmap.png"
        plt.savefig(path, dpi=150)
        print(f"[saved] {path}")
    plt.show()


def plot_cumulative_returns(weights, daily_returns: pd.DataFrame,
                             label="Portfolio", save=True):
    port_ret = (daily_returns * weights).sum(axis=1)
    cumulative = (1 + port_ret).cumprod()

    # equal weight benchmark
    eq_weights = np.ones(len(daily_returns.columns)) / len(daily_returns.columns)
    eq_ret = (daily_returns * eq_weights).sum(axis=1)
    eq_cumulative = (1 + eq_ret).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative, label=label, linewidth=2)
    ax.plot(eq_cumulative.index, eq_cumulative, label="Equal Weight",
            linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_title("Cumulative Returns — PortfolioIQ")
    ax.set_ylabel("Portfolio Value (starting at 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "cumulative_returns.png"
        plt.savefig(path, dpi=150)
        print(f"[saved] {path}")
    plt.show()