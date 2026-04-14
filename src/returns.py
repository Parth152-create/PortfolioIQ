import pandas as pd
import numpy as np


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Percentage change day over day."""
    return prices.pct_change().dropna()


def annualized_returns(daily: pd.DataFrame) -> pd.Series:
    """Mean daily return scaled to 252 trading days."""
    return daily.mean() * 252


def covariance_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return daily.cov() * 252


def correlation_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation — useful for the heatmap."""
    return daily.corr()