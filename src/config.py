import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

TICKERS = _cfg["portfolio"]["tickers"]
START_DATE = _cfg["portfolio"]["start_date"]
END_DATE = _cfg["portfolio"]["end_date"]

RISK_FREE_RATE = _cfg["optimization"]["risk_free_rate"]
N_FRONTIER_POINTS = _cfg["optimization"]["n_frontier_points"]
N_MONTE_CARLO = _cfg["optimization"]["n_monte_carlo"]

CACHE_DIR = ROOT / _cfg["data"]["cache_dir"]
RAW_DIR = ROOT / _cfg["data"]["raw_dir"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)