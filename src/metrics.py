import numpy as np
import pandas as pd
from src.config import RISK_FREE_RATE


def sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate=RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio from daily return series."""
    excess = portfolio_returns - risk_free_rate / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def sortino_ratio(portfolio_returns: pd.Series, risk_free_rate=RISK_FREE_RATE) -> float:
    """Like Sharpe but only penalizes downside volatility."""
    excess = portfolio_returns - risk_free_rate / 252
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return float((excess.mean() / downside) * np.sqrt(252))


def max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Largest peak-to-trough decline in the portfolio's cumulative value.
    Returns a negative number e.g. -0.34 means a 34% drawdown.
    """
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def value_at_risk(portfolio_returns: pd.Series, confidence=0.95) -> float:
    """
    Historical VaR — the loss you'd expect not to exceed
    on 95% of trading days. Returns a negative number.
    """
    return float(np.percentile(portfolio_returns, (1 - confidence) * 100))


def portfolio_daily_returns(weights, daily_returns: pd.DataFrame) -> pd.Series:
    """Weighted sum of daily returns across all tickers."""
    return (daily_returns * weights).sum(axis=1)


def full_metrics(weights, daily_returns: pd.DataFrame) -> dict:
    """Run all metrics at once. Returns a clean summary dict."""
    port_ret = portfolio_daily_returns(weights, daily_returns)
    return {
        "sharpe_ratio":   round(sharpe_ratio(port_ret), 4),
        "sortino_ratio":  round(sortino_ratio(port_ret), 4),
        "max_drawdown":   f"{max_drawdown(port_ret):.2%}",
        "var_95":         f"{value_at_risk(port_ret):.2%}",
        "annual_return":  f"{port_ret.mean() * 252:.2%}",
        "annual_vol":     f"{port_ret.std() * np.sqrt(252):.2%}",
    }