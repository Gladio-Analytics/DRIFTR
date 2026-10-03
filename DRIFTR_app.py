import numpy as np
import streamlit as st

from DRIFTR_FETCH import get_price_data, to_log_rets
from DRIFTR_DATA import make_empirical_price_paths, terminal_path_stats, simulated_terminal_stats
from DRIFTR_METHODS import make_simulation_priors_df, estimate_all_simulation_inputs
from DRIFTR_SIM import simulate_all, returns_to_prices
from DRIFTR_SCENARIOS import SCENARIO_LIBRARY, resolve_phase_schedule, build_black_swan_scenario
from DRIFTR_VIS import (plot_price_and_log_rets, plot_empirical_price_paths_simple, plot_simulated_price_paths,
                        plot_stats_table, plot_phase_shading)

st.set_page_config(page_title="DRIFTR", layout="wide")

SIM_ENGINE = "TrendMR"  # the engine with a friendly parameter write-up; also included in the "all engines" sim below
ALL_ENGINES = ["PiecewiseGBM", "Heston", "JumpDiffusion", "TrendMR", "RegimeSwitching", "GARCH"]
SIM_N = 5000
SIM_T = 252


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ticker(ticker):
    price_df = get_price_data(ticker)
    log_rets_df = to_log_rets(price_df)
    return price_df, log_rets_df


@st.cache_data(show_spinner=False)
def _empirical_analysis(log_rets_df):
    emp_paths = make_empirical_price_paths(log_rets_df)
    term_stats = terminal_path_stats(log_rets_df, emp_paths)
    return emp_paths, term_stats


def _simulate_with_progress(params, methods, seed, scenario, _progress=None, _span=(0.0, 1.0)):
    """Simulates each engine one at a time (rather than one simulate_all call
    across all methods) purely so real, per-engine progress can be reported —
    np.random.default_rng() passes an existing Generator through unchanged
    (verified: it continues the stream, it does not reseed), so splitting the
    work up this way draws the exact same kind of randomness as a single call
    would, just with a progress tick after each engine. `_span` remaps this
    function's own 0..1 progress into a sub-range of a larger shared bar
    (see _run_simulation, which uses this for its second half)."""
    lo, hi = _span

    def _tick(frac, text):
        if _progress:
            _progress(lo + frac * (hi - lo), text)

    rng = np.random.default_rng(seed)
    results = {"_T": SIM_T, "_N": SIM_N, "_params": params.copy()}
    n = len(methods)
    # n+2 steps: one per engine, plus prices, plus stats.
    for i, m in enumerate(methods):
        _tick(i / (n + 2), f"Simulating {m} ({i+1}/{n})...")
        sub = simulate_all(params, T=SIM_T, N=SIM_N, rng=rng, methods=m, scenario=scenario)
        results[m] = sub[m]
    _tick(n / (n + 2), "Computing prices...")
    prices = returns_to_prices(results, params)
    _tick((n + 1) / (n + 2), "Computing statistics...")
    sim_stats = simulated_terminal_stats(results)
    _tick(1.0, "Done.")
    return results, prices, sim_stats


@st.cache_data(show_spinner=False)
def _run_simulation(price_df, log_rets_df, seed, _progress=None):
    priors = make_simulation_priors_df(price_df, log_rets_df)

    # Calibrate each engine independently so one brittle engine on an unusual
    # ticker (e.g. GARCH's near-IGARCH check, RegimeSwitching needing enough
    # history) doesn't take down the whole simulation — just that one engine.
    # Calibration gets the first half of the bar, simulate/price/stats the
    # second half (via _simulate_with_progress's _span) — GARCH's multi-start
    # optimizer alone can take over a second, so a single "Calibrating..." tick
    # spanning all 6 engines would otherwise sit motionless long enough to
    # look stuck; six separate ticks keep something visibly moving throughout.
    params = None
    failed = {}
    n_engines = len(ALL_ENGINES)
    for i, engine in enumerate(ALL_ENGINES):
        if _progress:
            _progress(i / n_engines * 0.5, f"Calibrating {engine} ({i+1}/{n_engines})...")
        try:
            params = estimate_all_simulation_inputs(log_rets_df, priors, methods=engine, existing_params_df=params)
        except Exception as e:
            failed[engine] = str(e)

    if params is None or params.shape[1] == 0:
        raise RuntimeError("None of the engines could be calibrated for this ticker.")

    methods = list(params.columns)
    results, prices, sim_stats = _simulate_with_progress(
        params, methods, seed, None, _progress=_progress, _span=(0.5, 1.0))
    return params, results, prices, sim_stats, failed


st.title("DRIFTR")
st.caption("Enter a ticker to see its historical price/return behavior and what that history implies about where it could go over the next year.")

# A session-state figure cache keyed by cheap string identifiers (ticker,
# engine, scenario name), rather than @st.cache_data on the plot functions
# directly — that would mean re-hashing large dict-of-DataFrame arguments
# (prices/params/results/sim_stats) on every single Streamlit rerun, which
# happens on every widget interaction. Figures only ever depend on which
# ticker/engine/scenario they're for, so a plain string key is exact and free.
if "figs" not in st.session_state:
    st.session_state.figs = {}


def _cached_fig(key, builder):
    if key not in st.session_state.figs:
        st.session_state.figs[key] = builder()
    return st.session_state.figs[key]


if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = None

ticker_input = st.text_input("Ticker", value="", placeholder="One ticker, e.g. MSFT").strip().upper()
if st.button("Get Stock Data", type="primary") and ticker_input:
    # The old placeholder ("e.g. MSFT, JNJ, GS") read like an instruction to
    # enter several comma-separated tickers rather than three separate
    # examples — guard against that literally being typed in and sent to
    # yfinance as one (invalid) symbol, which just produces a cryptic 404.
    if any(c in ticker_input for c in (",", " ", ";")):
        st.error("Please enter a single ticker (e.g. 'MSFT'), not multiple tickers.")
    else:
        st.session_state.active_ticker = ticker_input

ticker = st.session_state.active_ticker

if not ticker:
    st.info("Enter a ticker above and click 'Get Stock Data' to get started.")
    st.stop()

bar = st.progress(0.0, text=f"Fetching {ticker}...")
try:
    price_df, log_rets_df = _load_ticker(ticker)
except Exception as e:
    bar.empty()
    st.error(f"Couldn't load data for '{ticker}': {e}")
    st.stop()

bar.progress(0.3, text="Computing empirical price paths...")
emp_paths, term_stats = _empirical_analysis(log_rets_df)

bar.progress(0.7, text="Building charts...")
price_rets_fig = _cached_fig(f"price_rets::{ticker}", lambda: plot_price_and_log_rets(price_df, log_rets_df))
empirical_paths_fig = _cached_fig(f"empirical_paths::{ticker}",
                                  lambda: plot_empirical_price_paths_simple(log_rets_df, emp_paths))
empirical_stats_fig = _cached_fig(f"empirical_stats::{ticker}", lambda: plot_stats_table(term_stats))
bar.progress(1.0, text="Done.")
bar.empty()

st.header(f"{ticker}: Price & Returns")
st.plotly_chart(price_rets_fig, width="stretch")

st.header(f"{ticker}: Empirical Price Paths")
st.caption("Every historical 21/63/126/252-day stretch of this stock's actual returns, replayed as if starting today.")
st.plotly_chart(empirical_paths_fig, width="stretch")

st.header(f"{ticker}: Terminal Value Distribution")
st.caption("Where each stretch above ended up, summarized by horizon.")
st.plotly_chart(empirical_stats_fig, width="stretch")

st.divider()
st.header("Simulate Future Prices")
st.caption("A one-year-ahead Monte Carlo simulation, calibrated from this stock's own history — no scripted scenario, just \"keep doing what it's been doing.\"")

_SIM_KEYS = ("params", "results", "prices", "sim_stats", "failed")

if "sim" not in st.session_state:
    st.session_state.sim = {}

if st.button("Simulate Future Prices", type="primary"):
    bar = st.progress(0.0, text="Starting...")
    try:
        values = _run_simulation(price_df, log_rets_df, seed=hash(ticker) % (2**31),
                                 _progress=lambda frac, text: bar.progress(frac, text=text))
        st.session_state.sim[ticker] = dict(zip(_SIM_KEYS, values))
    except Exception as e:
        st.error(f"Couldn't simulate '{ticker}': {e}")
    finally:
        bar.empty()

# Named dict + explicit key check rather than a positional tuple: if this
# code's return shape changes later (as it just did, adding "failed"), a
# stale entry from a previous app version is silently dropped here instead
# of crashing on unpack — just re-click Simulate instead of restarting
# the whole Streamlit session.
sim_entry = st.session_state.sim.get(ticker)
if sim_entry is not None and (not isinstance(sim_entry, dict) or not all(k in sim_entry for k in _SIM_KEYS)):
    # isinstance check must short-circuit first: an even older stale entry
    # could be a plain tuple containing a DataFrame, and "k in a_tuple" does
    # an equality scan that chokes on comparing a string to a DataFrame.
    del st.session_state.sim[ticker]
    sim_entry = None

if sim_entry is not None:
    params, results, prices, sim_stats, failed = (sim_entry[k] for k in _SIM_KEYS)
    available_engines = list(params.columns)

    if failed:
        with st.expander(f"{len(failed)} of {len(ALL_ENGINES)} engines couldn't be calibrated for {ticker}"):
            for engine, msg in failed.items():
                st.write(f"**{engine}**: {msg}")

    if SIM_ENGINE in available_engines:
        p = params[SIM_ENGINE]
        mu_m, sigma_m = float(p["mu_m"]), float(p["sigma_m"])
        kappa, sigma_x, x0 = float(p["kappa"]), float(p["sigma_x"]), float(p["x0"])
        half_life_days = np.log(2) / kappa * 252
        wobble = sigma_x / np.sqrt(2 * kappa)

        st.subheader("Estimated Parameters")
        st.caption(f"In plain English, this is what the {SIM_ENGINE} model learned from this stock's history "
                   "(the other engines below don't have a plain-English write-up yet):")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Expected annual return", f"{mu_m*100:+.1f}%",
                 help="The long-term trend this stock's price has been climbing (or falling) along, annualized.")
        c2.metric("Trend uncertainty", f"{sigma_m*100:.1f}%/yr",
                 help="How much that long-term trend itself wobbles year to year — a higher number means the trend is less reliable.")
        c3.metric("Typical wobble", f"{wobble*100:.1f}%",
                 help="How far the price typically strays from its trend before snapping back, at any given time.")
        c4.metric("Pull-back half-life", f"{half_life_days:.0f} trading days",
                 help="How long it takes for a stray move away from the trend to fade halfway back, on average.")
        c5.metric("Currently vs. trend", f"{x0*100:+.1f}%",
                 help="Where the price sits right now relative to its own trend line — positive means it's running above trend.")

    st.subheader("Simulated Price Paths")
    st.caption("Use the dropdown (top right) to switch between engines.")
    st.plotly_chart(_cached_fig(f"sim_paths::{ticker}",
                                lambda: plot_simulated_price_paths(prices, params, ticker=ticker, thin=0.2)),
                    width="stretch")

    st.subheader("Simulated Terminal Value Distribution")
    default_idx = available_engines.index(SIM_ENGINE) if SIM_ENGINE in available_engines else 0
    chosen_engine = st.selectbox("Engine", available_engines, index=default_idx, key="stats_engine_select")
    st.plotly_chart(_cached_fig(f"sim_stats_table::{ticker}::{chosen_engine}",
                                lambda: plot_stats_table(sim_stats[chosen_engine])), width="stretch")

    st.divider()
    st.header("Run a Scenario")
    st.caption("Same calibrated model, conditioned on a specific narrative instead of \"business as usual.\"")

    scenario_names = [s for s in SCENARIO_LIBRARY if s != "Baseline"] + ["Black Swan"]
    default_scenario_idx = scenario_names.index("Crash & Recovery") if "Crash & Recovery" in scenario_names else 0
    chosen_scenario = st.selectbox("Scenario", scenario_names, index=default_scenario_idx,
                                   help="Black Swan is calibrated from THIS ticker's own worst historical day, made a bit worse — "
                                        "it isn't a fixed scenario like the others.")

    # Black Swan is the one scenario that isn't a fixed entry in SCENARIO_LIBRARY
    # — "worse than the historical worst case" is inherently ticker-specific.
    if chosen_scenario == "Black Swan":
        try:
            scenario_spec = build_black_swan_scenario(log_rets_df)
        except Exception as e:
            st.error(f"Couldn't build a Black Swan scenario for {ticker}: {e}")
            scenario_spec = None
    else:
        scenario_spec = SCENARIO_LIBRARY[chosen_scenario]

    _SCEN_KEYS = ("results", "prices", "sim_stats")
    if "scenario_sim" not in st.session_state:
        st.session_state.scenario_sim = {}

    scenario_key = f"{ticker}::{chosen_scenario}"

    if scenario_spec is not None and st.button("Run Scenario", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            seed = hash(scenario_key) % (2**31)
            scen_results, scen_prices, scen_stats = _simulate_with_progress(
                params, available_engines, seed, scenario_spec,
                _progress=lambda frac, text: bar.progress(frac, text=text))
            st.session_state.scenario_sim[scenario_key] = dict(zip(_SCEN_KEYS, (scen_results, scen_prices, scen_stats)))
        except Exception as e:
            st.error(f"Couldn't simulate scenario '{chosen_scenario}': {e}")
        finally:
            bar.empty()

    scen_entry = st.session_state.scenario_sim.get(scenario_key)
    if scen_entry is not None and (not isinstance(scen_entry, dict) or not all(k in scen_entry for k in _SCEN_KEYS)):
        del st.session_state.scenario_sim[scenario_key]
        scen_entry = None

    if scen_entry is not None:
        scen_results, scen_prices, scen_stats = (scen_entry[k] for k in _SCEN_KEYS)

        def _build_scenario_fig():
            fig = plot_simulated_price_paths(scen_prices, params, ticker=ticker, thin=0.2)
            return plot_phase_shading(fig, resolve_phase_schedule(scenario_spec, SIM_T))

        st.subheader(f"Simulated Price Paths — {chosen_scenario}")
        st.caption("Shaded bands mark the scenario's phases. Use the dropdown (top right) to switch engines.")
        st.plotly_chart(_cached_fig(f"scenario_paths::{scenario_key}", _build_scenario_fig), width="stretch")

        st.subheader("Baseline vs. Scenario — Terminal Value Distribution")
        comp_default = available_engines.index(SIM_ENGINE) if SIM_ENGINE in available_engines else 0
        comp_engine = st.selectbox("Engine", available_engines, index=comp_default, key="scenario_stats_engine_select")
        col_a, col_b = st.columns(2)
        with col_a:
            # Distinct key namespace from the un-titled table above — this one
            # carries a "Baseline" title, so it can't share a cache slot with it.
            st.plotly_chart(_cached_fig(f"sim_stats_table_titled::{ticker}::{comp_engine}",
                                        lambda: plot_stats_table(sim_stats[comp_engine], title="Baseline")),
                            width="stretch")
        with col_b:
            st.plotly_chart(_cached_fig(f"scenario_stats_table::{scenario_key}::{comp_engine}",
                                        lambda: plot_stats_table(scen_stats[comp_engine], title=chosen_scenario)),
                            width="stretch")
