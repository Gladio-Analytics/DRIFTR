import numpy as np
import pandas as pd
from scipy.stats import skewnorm


# --------------------------------------------------
# Helpers
# --------------------------------------------------

_SKIP_DIFF = {"drift_sample", "transition_matrix", "regime_labels"}


def _df(arr, T, N):
    return pd.DataFrame(
        arr,
        index=pd.RangeIndex(1, T + 1, name="step"),
        columns=pd.RangeIndex(0, N, name="path"),
    )


def _params_changed(old_params, new_params, method):
    if method not in old_params.columns or method not in new_params.columns:
        return True
    old_col = old_params[method].dropna()
    new_col = new_params[method].dropna()
    if set(old_col.index) != set(new_col.index):
        return True
    for key in old_col.index:
        if key in _SKIP_DIFF:
            continue
        o, n = old_col[key], new_col[key]
        try:
            if not np.isclose(float(o), float(n)):
                return True
        except (TypeError, ValueError):
            if o != n:
                return True
    return False


def _draw_z(shock_dist, shock_a, shock_loc, shock_scale, shape, rng):
    if shock_dist == "normal":
        return rng.standard_normal(shape)
    seed  = int(rng.integers(0, 2 ** 31))
    Z     = skewnorm.rvs(shock_a, loc=shock_loc, scale=shock_scale,
                         size=shape, random_state=seed)
    delta = shock_a / np.sqrt(1 + shock_a ** 2)
    mean  = shock_loc + shock_scale * delta * np.sqrt(2 / np.pi)
    var   = shock_scale ** 2 * (1 - (2 * delta ** 2) / np.pi)
    return (Z - mean) / np.sqrt(max(var, 1e-12))


# --------------------------------------------------
# Component drawers — all path arrays returned as DataFrames
# --------------------------------------------------

def draw_components_piecewise_gbm(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import resolve_phase_schedule, apply_scheduled_jumps

    rng = np.random.default_rng(rng)
    g   = params_df["PiecewiseGBM"]
    mu_calib    = float(g["mu"])
    mu_se       = float(g["mu_se"])
    sigma_calib = float(g["sigma"])
    dt          = float(g["dt"])

    sched = resolve_phase_schedule(scenario, T)
    if sched is None:
        mu_sched, vol_sched, jump_events = np.full(T, mu_calib), np.full(T, sigma_calib), []
        # Parameter uncertainty: each path draws ONE persistent drift offset
        # from the calibration's own standard error (not re-randomized every
        # step — that would just look like extra noise, not "we don't know
        # the true drift"). Suppressed under a named scenario, whose targets
        # are prescribed, not estimated.
        mu_jitter = (mu_se * rng.standard_normal(N))[None, :]
    else:
        mu_sched, vol_sched, jump_events = sched["mu_sched"], sched["vol_sched"], sched["jump_events"]
        mu_jitter = 0.0

    drift = (mu_sched[:, None] + mu_jitter - 0.5 * vol_sched[:, None] ** 2) * dt
    vol   = vol_sched[:, None] * np.sqrt(dt)
    Z     = rng.standard_normal((T, N))
    J     = apply_scheduled_jumps(jump_events, T, N, rng)
    eps   = drift + vol * Z + J

    return {
        "Z": _df(Z, T, N), "J": _df(J, T, N), "eps": _df(eps, T, N),
        "mu_sched": mu_sched, "vol_sched": vol_sched, "dt": dt,
    }


def draw_components_jumpdiffusion(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import resolve_phase_schedule, apply_scheduled_jumps

    rng = np.random.default_rng(rng)
    g   = params_df["JumpDiffusion"]
    mu_calib         = float(g["mu"])
    mu_se            = float(g["mu_se"])
    sigma_calib      = float(g["sigma"])
    dt               = float(g["dt"])
    jump_lambda_year = float(g["jump_lambda_year"])
    jump_mean_calib  = float(g["jump_mean"])
    jump_sd_calib    = float(g["jump_sd"])

    sched = resolve_phase_schedule(scenario, T)
    if sched is None:
        mu_sched, vol_sched = np.full(T, mu_calib), np.full(T, sigma_calib)
        mu_jitter = (mu_se * rng.standard_normal(N))[None, :]
    else:
        mu_sched, vol_sched = sched["mu_sched"], sched["vol_sched"]
        mu_jitter = 0.0

    drift = (mu_sched[:, None] + mu_jitter - 0.5 * vol_sched[:, None] ** 2) * dt
    vol   = vol_sched[:, None] * np.sqrt(dt)
    Z     = rng.standard_normal((T, N))

    if sched is None:
        # Ambient Poisson-thinned jump process from calibrated params.
        lam_daily = jump_lambda_year / 252.0
        J      = np.zeros((T, N))
        occurs = rng.random((T, N)) < lam_daily
        n_occ  = int(occurs.sum())
        if n_occ > 0:
            J[occurs] = rng.normal(jump_mean_calib, jump_sd_calib, n_occ)
    else:
        # A named scenario scripts its own jumps; the ambient process is
        # suppressed so the narrative isn't polluted by uncontrolled jumps.
        J = apply_scheduled_jumps(sched["jump_events"], T, N, rng)

    eps = drift + vol * Z + J
    return {"Z": _df(Z, T, N), "J": _df(J, T, N), "eps": _df(eps, T, N)}


def draw_components_heston(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import resolve_phase_schedule

    rng     = np.random.default_rng(rng)
    h       = params_df["Heston"]
    mu_calib    = float(h["mu"])
    mu_se       = float(h["mu_se"])
    v0          = float(h["v0"])
    theta_calib = float(h["theta"])
    kappa_calib = float(h["heston_kappa"])
    sigma_v     = float(h["sigma_v"])
    rho         = float(h["rho"])
    dt          = float(h["dt"])
    rho2    = np.sqrt(max(1.0 - rho ** 2, 0.0))
    sdt     = np.sqrt(dt)

    sched = resolve_phase_schedule(scenario, T)
    if sched is None:
        # One persistent per-path drift offset (parameter uncertainty), not
        # re-randomized every step. theta_t/kappa_t stay at their calibrated
        # constants in this cheap-fix scope.
        mu_path = mu_calib + mu_se * rng.standard_normal(N)
        theta_t = np.full(T, theta_calib)
        kappa_t = np.full(T, kappa_calib)
    else:
        mu_t    = sched["mu_sched"]
        theta_t = sched["vol_sched"] ** 2
        # Duration floor: a calibrated kappa of 2-5/yr is too slow to move
        # variance meaningfully toward a new theta within a short scenario
        # phase, so floor kappa by the phase's own duration (half-life ~ 1/3
        # of the phase) whenever that's faster than the calibrated value.
        boundaries = sched["boundaries"]
        kappa_t    = np.empty(T)
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            dur_years = max((e - s) * dt, 1 / 252)
            kappa_t[s:e] = max(kappa_calib, 3.0 / dur_years)

    V_arr     = np.empty((T, N))
    SIGMA_arr = np.empty((T, N))
    Z1_arr    = np.empty((T, N))
    Zs_arr    = np.empty((T, N))
    EPS_arr   = np.empty((T, N))

    v = np.full(N, max(v0, 1e-12))

    for t in range(T):
        v_pos        = np.maximum(v, 0.0)
        sv           = np.sqrt(v_pos)
        z1           = rng.standard_normal(N)
        z2           = rng.standard_normal(N)
        zs           = rho * z1 + rho2 * z2
        v            = np.maximum(v_pos + kappa_t[t] * (theta_t[t] - v_pos) * dt + sigma_v * sv * z1 * sdt, 0.0)
        SIGMA_arr[t] = sv
        V_arr[t]     = v
        Z1_arr[t]    = z1
        Zs_arr[t]    = zs
        mu_now       = mu_path if sched is None else mu_t[t]
        EPS_arr[t]   = (mu_now - 0.5 * v_pos) * dt + sv * zs * sdt

    return {
        "V": _df(V_arr, T, N), "sigma": _df(SIGMA_arr, T, N),
        "Z1": _df(Z1_arr, T, N), "Zs": _df(Zs_arr, T, N), "eps": _df(EPS_arr, T, N),
        "mu": mu_calib, "dt": dt, "sdt": sdt,
    }


def draw_components_trendmr(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import resolve_phase_schedule

    rng = np.random.default_rng(rng)
    p   = params_df["TrendMR"]
    mu_m_calib    = float(p["mu_m"])
    mu_m_se       = float(p["mu_m_se"])
    sigma_m_calib = float(p["sigma_m"])
    kappa         = float(p["kappa"])
    sigma_x_calib = float(p["sigma_x"])
    x0            = float(p["x0"])
    dt            = float(p["dt"])
    sqdt          = np.sqrt(dt)

    sched = resolve_phase_schedule(scenario, T)
    if sched is None:
        # One persistent per-path trend-drift offset (parameter uncertainty).
        mu_m_path   = mu_m_calib + mu_m_se * rng.standard_normal(N)
        sigma_m_t   = np.full(T, sigma_m_calib)
        sigma_x_t   = np.full(T, sigma_x_calib)
        jump_events = []
    else:
        mu_m_t   = sched["mu_sched"]
        base_vol = np.sqrt(sigma_m_calib ** 2 + sigma_x_calib ** 2)
        scale_t  = sched["vol_sched"] / max(base_vol, 1e-8)
        sigma_m_t = sigma_m_calib * scale_t
        sigma_x_t = sigma_x_calib * scale_t
        jump_events = sched["jump_events"]

    X_arr   = np.empty((T, N))
    DM_arr  = np.empty((T, N))
    RET_arr = np.empty((T, N))
    x = np.full(N, x0)

    for t in range(T):
        dx = -kappa * x * dt + sigma_x_t[t] * sqdt * rng.standard_normal(N)
        # A crash here is a one-time LEVEL shock into x (not the return) —
        # kappa then pulls it back out on its own, which is this engine's
        # native "recovery" mechanic; no separate recovery-phase logic needed.
        for ev in jump_events:
            jm, js = float(ev["jump_mean"]), float(ev["jump_sd"])
            if "prob_at_phase_start" in ev and ev["start_step"] == t:
                occ = rng.random(N) < float(ev["prob_at_phase_start"])
                if occ.any():
                    dx[occ] += rng.normal(jm, js, occ.sum())
            elif "lambda_annual" in ev and ev["start_step"] <= t < ev["end_step"]:
                occ = rng.random(N) < (float(ev["lambda_annual"]) / 252.0)
                if occ.any():
                    dx[occ] += rng.normal(jm, js, occ.sum())
        x = x + dx
        mu_m_now = mu_m_path if sched is None else mu_m_t[t]
        dm = mu_m_now * dt + sigma_m_t[t] * sqdt * rng.standard_normal(N)
        X_arr[t]   = x
        DM_arr[t]  = dm
        RET_arr[t] = dm + dx

    return {"x": _df(X_arr, T, N), "dm": _df(DM_arr, T, N), "eps": _df(RET_arr, T, N)}


def draw_components_garch(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import resolve_phase_schedule

    rng = np.random.default_rng(rng)
    p   = params_df["GARCH"]
    omega_calib   = float(p["garch_omega"])
    alpha         = float(p["garch_alpha"])
    beta          = float(p["garch_beta"])
    h0            = float(p["garch_h0"])
    mu_step_calib = float(p["garch_mu_step"])
    mu_step_se    = float(p["garch_mu_se"])
    dt            = float(p["dt"])
    shock_dist    = str(p["garch_shock_dist"])
    shock_a, shock_loc, shock_scale = p["shock_a"], p["shock_loc"], p["shock_scale"]

    sched       = resolve_phase_schedule(scenario, T)
    persistence = min(alpha + beta, 0.995)
    if sched is None:
        # One persistent per-path drift offset (parameter uncertainty),
        # reshaped to (1,N) so it broadcasts against the (T,N) shock array
        # the same way compose_returns_garch already broadcasts a (T,1).
        mu_component = (mu_step_calib + mu_step_se * rng.standard_normal(N)).reshape(1, N)
        omega_t, jump_events = np.full(T, omega_calib), []
    else:
        mu_component = (sched["mu_sched"] * dt).reshape(T, 1)                   # annualized -> daily
        omega_t      = (sched["vol_sched"] ** 2) * dt * (1 - persistence)       # annualized vol -> daily-variance omega
        jump_events  = sched["jump_events"]

    Z = _draw_z(shock_dist, shock_a, shock_loc, shock_scale, (T, N), rng)

    SHOCK_arr = np.empty((T, N))
    H_arr     = np.empty((T, N))
    h = np.full(N, max(h0, 1e-12))

    for t in range(T):
        H_arr[t] = h
        shock    = np.sqrt(h) * Z[t]
        # Scripted crash = a forced |shock| at the phase-start step, which
        # elevates h via alpha*shock**2 and decays naturally per beta
        # afterward — a sustained vol regime is handled separately by omega_t.
        for ev in jump_events:
            if "prob_at_phase_start" in ev and ev["start_step"] == t:
                occ = rng.random(N) < float(ev["prob_at_phase_start"])
                if occ.any():
                    shock[occ] = float(ev["jump_mean"])
        SHOCK_arr[t] = shock
        h = np.maximum(omega_t[t] + alpha * shock ** 2 + beta * h, 1e-12)

    return {"H": _df(H_arr, T, N), "Z": _df(Z, T, N), "shock": _df(SHOCK_arr, T, N), "mu_component": mu_component}


def draw_components_regimeswitching(params_df, T=252, N=10000, rng=None, scenario=None):
    from DRIFTR_SCENARIOS import adapt_scenario_regimeswitching, apply_scheduled_jumps_per_path

    rng = np.random.default_rng(rng)
    p   = params_df["RegimeSwitching"]
    dt  = float(p["dt"])
    regimes = list(p["regime_labels"])

    if scenario is None:
        # True Baseline: genuine calibrated Markov-chain regime switching —
        # paths can enter/exit regimes stochastically on their own.
        P           = np.asarray(p["transition_matrix"], dtype=float)
        mu_table    = np.array([float(p[f"mu_{g}"]) for g in regimes])
        mu_se_table = np.array([float(p[f"mu_{g}_se"]) for g in regimes])
        sigma_table = np.array([float(p[f"sigma_{g}"]) for g in regimes])
        pos         = {g: i for i, g in enumerate(regimes)}
        cumP        = np.cumsum(P, axis=1)

        # One persistent per-path drift-uncertainty factor: whenever a path
        # is in regime g, its realized drift is mu_table[g] + z*mu_se_table[g]
        # — the same z for that path throughout, so "we don't know Crash's
        # true mean" stays fixed for a given hypothetical path rather than
        # re-randomizing every step (which would just look like extra noise).
        z = rng.standard_normal(N)

        s = np.full(N, pos[str(p["initial_regime"])], dtype=int)
        REGIME = np.empty((T, N), dtype=int)
        MU     = np.empty((T, N))
        SIGMA  = np.empty((T, N))

        for t in range(T):
            REGIME[t] = s
            MU[t]     = mu_table[s] + z * mu_se_table[s]
            SIGMA[t]  = sigma_table[s]
            u = rng.random(N)
            s = (u[:, None] > cumP[s]).sum(axis=1)

        eps = (MU - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * rng.standard_normal((T, N))
        return {"regime": _df(REGIME, T, N), "mu": _df(MU, T, N), "sigma": _df(SIGMA, T, N), "eps": _df(eps, T, N)}

    # Named scenario: per-path stochastic phase timing, forced to follow the
    # scenario's phase order (see DRIFTR_SCENARIOS.adapt_scenario_regimeswitching).
    adapted    = adapt_scenario_regimeswitching(scenario, T, N, rng)
    mu_annual  = adapted["mu_annual"]
    vol_annual = adapted["vol_annual"]
    phase_id   = adapted["phase_id"]

    Z   = rng.standard_normal((T, N))
    eps = (mu_annual - 0.5 * vol_annual ** 2) * dt + vol_annual * np.sqrt(dt) * Z
    J   = apply_scheduled_jumps_per_path(adapted["jump_events"], T, N, rng)
    eps = eps + J

    return {"regime": _df(phase_id, T, N), "mu": _df(mu_annual, T, N),
            "sigma": _df(vol_annual, T, N), "eps": _df(eps, T, N)}


# --------------------------------------------------
# Return composers
# --------------------------------------------------

def compose_returns_piecewise_gbm(c, T, N):
    return c["eps"]


def compose_returns_jumpdiffusion(c, T, N):
    return c["eps"]


def compose_returns_heston(c, T, N):
    return c["eps"]


def compose_returns_trendmr(c, T, N):
    return c["eps"]


def compose_returns_garch(c, T, N):
    shock = c["shock"].to_numpy()
    # mu_component is (1,N) in Baseline (per-path jittered drift) or (T,1)
    # under a scenario (shared per-step schedule) — both broadcast against
    # the (T,N) shock array the same way.
    mu_component = np.asarray(c["mu_component"])
    return _df(shock + mu_component, T, N)


def compose_returns_regimeswitching(c, T, N):
    return c["eps"]


_DRAWERS = {
    "PiecewiseGBM":    draw_components_piecewise_gbm,
    "JumpDiffusion":   draw_components_jumpdiffusion,
    "Heston":          draw_components_heston,
    "TrendMR":         draw_components_trendmr,
    "GARCH":           draw_components_garch,
    "RegimeSwitching": draw_components_regimeswitching,
}

_COMPOSERS = {
    "PiecewiseGBM":    compose_returns_piecewise_gbm,
    "JumpDiffusion":   compose_returns_jumpdiffusion,
    "Heston":          compose_returns_heston,
    "TrendMR":         compose_returns_trendmr,
    "GARCH":           compose_returns_garch,
    "RegimeSwitching": compose_returns_regimeswitching,
}


# --------------------------------------------------
# Orchestrators
# --------------------------------------------------

def draw_all_components(params_df, T=252, N=10000, rng=None, methods=None, scenario=None):
    rng = np.random.default_rng(rng)
    if methods is None:
        methods = [m for m in _DRAWERS if m in params_df.columns]
    if isinstance(methods, str):
        methods = [methods]
    bad = [m for m in methods if m not in _DRAWERS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    components = {"_params": params_df.copy(), "_T": T, "_N": N, "_scenario": scenario}
    for m in methods:
        components[m] = _DRAWERS[m](params_df, T=T, N=N, rng=rng, scenario=scenario)
    return components


def simulate_all_from_components(components):
    T       = components["_T"]
    N       = components["_N"]
    methods = [m for m in components if not m.startswith("_")]
    results = {"_params": components["_params"].copy(), "_T": T, "_N": N}
    for m in methods:
        results[m] = _COMPOSERS[m](components[m], T, N)
    return results


def simulate_all(params_df, T=252, N=10000, rng=None, methods=None, scenario=None):
    components = draw_all_components(params_df, T=T, N=N, rng=rng, methods=methods, scenario=scenario)
    return simulate_all_from_components(components)


def update_components(components, params_df, scenario=None):
    old_params = components["_params"]
    T          = components["_T"]
    N          = components["_N"]
    rng        = np.random.default_rng()
    methods    = [m for m in _DRAWERS if m in params_df.columns]
    to_run     = [m for m in methods if _params_changed(old_params, params_df, m)]

    if not to_run:
        print("No parameter changes detected — nothing to re-draw.")
        return components

    updated                = dict(components)
    updated["_params"]     = params_df.copy()
    updated["_scenario"]   = scenario
    for m in to_run:
        print(f"  Re-drawing {m}")
        updated[m] = _DRAWERS[m](params_df, T=T, N=N, rng=rng, scenario=scenario)
    return updated


# --------------------------------------------------
# Prices
# --------------------------------------------------

def returns_to_prices(results, params_df):
    price_results = {}
    for m, rets_df in results.items():
        if m.startswith("_"):
            continue
        S0               = float(params_df.loc["S0", m])
        prices           = S0 * np.exp(rets_df.cumsum(axis=0))
        price_results[m] = prices
    return price_results
