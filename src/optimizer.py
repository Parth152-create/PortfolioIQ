import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.config import RISK_FREE_RATE, N_FRONTIER_POINTS


def portfolio_performance(weights, ann_returns, cov_matrix):
    """Return (annualized_return, annualized_volatility) for a weight vector."""
    ret = np.dot(weights, ann_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return ret, vol


def sharpe_ratio(weights, ann_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    ret, vol = portfolio_performance(weights, ann_returns, cov_matrix)
    return (ret - risk_free_rate) / vol


def _base_constraints_and_bounds(n):
    """Weights sum to 1, each weight between 0 and 1 (long-only)."""
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    return constraints, bounds


def minimum_variance_portfolio(ann_returns, cov_matrix):
    """Find the portfolio with the lowest possible volatility."""
    n = len(ann_returns)
    constraints, bounds = _base_constraints_and_bounds(n)

    result = minimize(
        fun=lambda w: portfolio_performance(w, ann_returns, cov_matrix)[1],
        x0=np.ones(n) / n,           # start from equal weights
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def max_sharpe_portfolio(ann_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """Find the portfolio that maximizes the Sharpe ratio."""
    n = len(ann_returns)
    constraints, bounds = _base_constraints_and_bounds(n)

    result = minimize(
        fun=lambda w: -sharpe_ratio(w, ann_returns, cov_matrix, risk_free_rate),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def efficient_frontier(ann_returns, cov_matrix, n_points=N_FRONTIER_POINTS):
    """
    Sweep target returns from min-variance to max-return,
    solving for minimum volatility at each target.
    Returns a DataFrame with columns: return, volatility, sharpe.
    """
    n = len(ann_returns)
    _, bounds = _base_constraints_and_bounds(n)

    # range of target returns to sweep
    min_ret = min_variance_return(ann_returns, cov_matrix)
    max_ret = float(ann_returns.max())
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, ann_returns) - t}
        ]
        result = minimize(
            fun=lambda w: portfolio_performance(w, ann_returns, cov_matrix)[1],
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            ret, vol = portfolio_performance(result.x, ann_returns, cov_matrix)
            sr = (ret - RISK_FREE_RATE) / vol
            frontier.append({
                "return": round(ret, 6),
                "volatility": round(vol, 6),
                "sharpe": round(sr, 6),
                "weights": result.x
            })

    return pd.DataFrame(frontier)


def min_variance_return(ann_returns, cov_matrix):
    """Helper — get the return of the minimum variance portfolio."""
    result = minimum_variance_portfolio(ann_returns, cov_matrix)
    ret, _ = portfolio_performance(result.x, ann_returns, cov_matrix)
    return ret


def get_allocation_df(weights, tickers):
    """Format weights into a clean DataFrame for display/export."""
    df = pd.DataFrame({
        "ticker": tickers,
        "weight": weights,
        "weight_pct": [f"{w * 100:.2f}%" for w in weights]
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    return df