def get_price_data(ticker: str, retries: int = 3, backoff_seconds: float = 1.5):
    import time
    import yfinance as yf

    # auto_adjust=True folds splits/dividends into "Close" (no separate
    # "Adj Close" column in current yfinance) — this IS the adjusted close.
    # Ticker.history() (not yf.download()) is used deliberately: download()
    # now returns MultiIndex columns even for a single ticker, which is an
    # unnecessary complication when we only ever fetch one ticker at a time.
    #
    # Retried with backoff: Yahoo Finance's "crumb" auth token gets rate-limited
    # (HTTP 429) more readily from shared-IP hosts like cloud platforms than
    # from a local machine — a transient failure there often clears within a
    # few seconds, so a short retry is worth it before giving up.
    df = None
    last_exc = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if not df.empty:
                break
        except Exception as e:
            last_exc = e
            df = None
        if attempt < retries - 1:
            time.sleep(backoff_seconds * (attempt + 1))

    if df is None or df.empty:
        detail = f" ({last_exc})" if last_exc is not None else ""
        raise ValueError(f"No price data found for ticker '{ticker}'{detail}.")

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
