import yfinance as yf
import pandas as pd
from pathlib import Path
from src.config import TICKERS, START_DATE, END_DATE, CACHE_DIR


def fetch_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for given tickers.
    Uses local CSV cache to avoid repeated API calls.
    """
    cache_file = CACHE_DIR / f"prices_{'_'.join(sorted(tickers))}_{start}_{end}.csv"

    if cache_file.exists() and not force_refresh:
        print(f"[cache] Loading prices from {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"[yfinance] Fetching {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # extract Close prices only
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw[["Close"]]
        df.columns = tickers

    df = _clean(df)

    df.to_csv(cache_file)
    print(f"[cache] Saved to {cache_file.name}")

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with too many NaNs, then forward-fill the rest."""
    threshold = 0.1  # drop ticker if >10% of rows are missing
    df = df.dropna(axis=1, thresh=int(len(df) * (1 - threshold)))
    df = df.ffill().bfill()
    return df


def get_price_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick sanity check — date range, count, nulls per ticker."""
    summary = pd.DataFrame({
        "start": df.apply(lambda c: c.first_valid_index()),
        "end": df.apply(lambda c: c.last_valid_index()),
        "trading_days": df.count(),
        "missing": df.isna().sum(),
        "first_price": df.iloc[0],
        "last_price": df.iloc[-1],
    })
    return summary