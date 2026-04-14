import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.config import TICKERS, START_DATE, END_DATE, RISK_FREE_RATE

REPORTS_DIR = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_allocation(weights, tickers, metrics: dict,
                    portfolio_type: str = "max_sharpe"):
    output = {
        "portfolio_type": portfolio_type,
        "generated_at": datetime.now().isoformat(),
        "config": {
            "tickers": tickers,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "risk_free_rate": RISK_FREE_RATE
        },
        "metrics": metrics,
        "allocation": {
            ticker: round(float(w), 6)
            for ticker, w in zip(tickers, weights)
            if w > 0.0001
        }
    }

    path = REPORTS_DIR / f"{portfolio_type}_allocation.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[saved] {path}")
    return output