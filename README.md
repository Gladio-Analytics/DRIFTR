# DRIFTR

A Streamlit app for exploring a stock's historical price/return behavior and
simulating where it could go over the next year, across six different
stochastic models (Piecewise GBM, Jump-Diffusion, Heston, Trend +
Mean-Reversion, Regime-Switching, GARCH) and a library of narrative scenarios
(Crash & Recovery, Boom & Bust, a ticker-calibrated Black Swan, etc.).

## Running locally

```
pip install -r requirements.txt
streamlit run DRIFTR_app.py
```

No API key needed — price history comes from Yahoo Finance via `yfinance`.

## Files

- `DRIFTR_app.py` — the Streamlit app itself.
- `DRIFTR_FETCH.py` — pulls price history and converts it to log returns.
- `DRIFTR_METHODS.py` — calibrates each of the six engines from a ticker's history.
- `DRIFTR_SIM.py` — simulates price paths for each engine, with or without a scenario.
- `DRIFTR_SCENARIOS.py` — the scenario library and the phase-schedule machinery.
- `DRIFTR_DATA.py` — descriptive statistics (empirical and simulated).
- `DRIFTR_VIS.py` — all the Plotly charts and tables.
- `.streamlit/config.toml` — forces the light theme.
