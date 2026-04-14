from src.data_loader import fetch_prices
from src.returns import daily_returns, annualized_returns, covariance_matrix
from src.optimizer import (max_sharpe_portfolio, minimum_variance_portfolio,
                            efficient_frontier, portfolio_performance)
from src.metrics import full_metrics
from src.visualizer import (plot_allocation, plot_correlation_heatmap,
                             plot_cumulative_returns)
from src.monte_carlo import run_simulation, plot_monte_carlo, get_best_simulated
from src.report import save_allocation

prices = fetch_prices()
daily = daily_returns(prices)
ann_ret = annualized_returns(daily)
cov = covariance_matrix(daily)
tickers = prices.columns.tolist()

ms = max_sharpe_portfolio(ann_ret, cov)
mv = minimum_variance_portfolio(ann_ret, cov)
frontier = efficient_frontier(ann_ret, cov)

ms_ret, ms_vol = portfolio_performance(ms.x, ann_ret, cov)
mv_ret, mv_vol = portfolio_performance(mv.x, ann_ret, cov)

ms_metrics = full_metrics(ms.x, daily)
mv_metrics = full_metrics(mv.x, daily)

print("\n--- Max Sharpe Metrics ---")
for k, v in ms_metrics.items():
    print(f"  {k}: {v}")

print("\n--- Min Variance Metrics ---")
for k, v in mv_metrics.items():
    print(f"  {k}: {v}")

# save JSON reports
save_allocation(ms.x, tickers, ms_metrics, portfolio_type="max_sharpe")
save_allocation(mv.x, tickers, mv_metrics, portfolio_type="min_variance")

# plots
print("\n[monte carlo] Running 10,000 simulations...")
simulation = run_simulation(ann_ret, cov)
get_best_simulated(simulation, ann_ret)
plot_monte_carlo(
    simulation, frontier,
    max_sharpe_point={"return": ms_ret, "volatility": ms_vol},
    min_var_point={"return": mv_ret, "volatility": mv_vol}
)
plot_allocation(ms.x, tickers, title="Max Sharpe Portfolio")
plot_allocation(mv.x, tickers, title="Min Variance Portfolio")
plot_correlation_heatmap(daily)
plot_cumulative_returns(ms.x, daily, label="Max Sharpe")