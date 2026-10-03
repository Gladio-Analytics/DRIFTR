def get_price_data(ticker: str):
    import yfinance as yf

    # auto_adjust=True folds splits/dividends into "Close" (no separate
    # "Adj Close" column in current yfinance) — this IS the adjusted close.
    # Ticker.history() (not yf.download()) is used deliberately: download()
    # now returns MultiIndex columns even for a single ticker, which is an
    # unnecessary complication when we only ever fetch one ticker at a time.
    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    prices = df["Close"].astype(float)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices.index = prices.index.tz_localize(None)
    prices.index.name = "date"
    return prices.to_frame(name=ticker)


def to_log_rets(price_df):
    import numpy as np
    return np.log(price_df).diff().dropna()


def get_last_price(price_df):
    return float(price_df.iloc[:, 0].astype(float).dropna().iloc[-1])
