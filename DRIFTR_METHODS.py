def make_simulation_priors_df(price_df, log_rets_df):
    import numpy as np
    import pandas as pd

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()
    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()
    if price_df.shape[1] != 1 or log_rets_df.shape[1] != 1:
        raise ValueError("price_df and log_rets_df must each contain exactly one ticker.")
    if str(price_df.columns[0]) != str(log_rets_df.columns[0]):
        raise ValueError("price_df and log_rets_df must contain the same ticker.")

    S0 = float(price_df.iloc[:, 0].astype(float).dropna().iloc[-1])

    out = pd.DataFrame({
        "GBM": {
            "S0": S0, "dt": 1/252, "method": "mle",
        },
        "Heston": {
            "S0": S0, "dt": 1/252, "rv_window": 21, "use_mu": True,
        },
        "JumpDiffusion": {
            "S0": S0, "dt": 1/252,
            "jump_threshold_mult": 3.0, "jump_lambda_prior": 2.0,
            "jump_mean_prior": -0.01, "jump_sd_prior": 0.02, "jump_shrink": 0.5,
        },
        "TrendMR": {
            "S0": S0, "dt": 1/252, "trend_window": 126,
            "kappa_min": 0.5, "kappa_max": 60.0, "sigma_floor": 1e-4,
        },
        "RegimeSwitching": {
            "S0": S0, "dt": 1/252, "regime_window": 21, "min_regime_days": 10,
            "recovery_window": 15, "shrink_pseudo_n": 30.0,
            "sigma_min": 0.003, "sigma_max": 0.05,
        },
        "GARCH": {
            "S0": S0, "dt": 1/252, "shock_method": "skewnorm",
        },
    })
    out.index.name = "parameter"
    return out


def _get_r(log_rets_df):
    import pandas as pd
    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()
    if log_rets_df.shape[1] != 1:
        raise ValueError("log_rets_df must contain exactly one ticker.")
    return log_rets_df.iloc[:, 0].astype(float).dropna()


def _ar1_ols(series):
    import numpy as np
    y     = series.iloc[1:].to_numpy()
    x     = series.shift(1).iloc[1:].to_numpy()
    X     = np.column_stack([np.ones_like(x), x])
    c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ np.array([c, phi])
    sigma = float(resid.std(ddof=2))
    bar   = float(c / (1 - phi)) if abs(1 - phi) > 1e-8 else float(series.mean())
    return float(phi), bar, sigma


def _garch11_qmle(e):
    import numpy as np
    from scipy.optimize import minimize
    v   = max(float(np.var(e, ddof=1)), 1e-12)
    eps = 1e-6

    def negll(par):
        omega, alpha, beta = par
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e100
        h    = np.empty(len(e))
        h[0] = max(omega / max(1 - alpha - beta, 1e-6), 1e-12)
        for t in range(1, len(e)):
            h[t] = max(omega + alpha * e[t-1]**2 + beta * h[t-1], 1e-12)
        return 0.5 * np.sum(np.log(h) + e**2 / h)

    # SLSQP's hard 1e100 penalty wall at the feasibility boundary can make its
    # finite-difference gradient estimate spuriously report "incompatible"
    # inequality constraints near some starting points (observed on jump-heavy
    # return series) — retry from a few starts and keep the best successful fit.
    starts = [(0.05, 0.90), (0.10, 0.80), (0.15, 0.70), (0.08, 0.60), (0.20, 0.50)]
    best = None
    for a0, b0 in starts:
        w0  = max(v * (1 - a0 - b0), 1e-8)
        res = minimize(negll, x0=[w0, a0, b0], method="SLSQP",
                       bounds=[(1e-12, np.inf), (0, 1-eps), (0, 1-eps)],
                       constraints=[{"type": "ineq", "fun": lambda x: 1-eps-x[1]-x[2]}],
                       options={"maxiter": 800, "ftol": 1e-9, "disp": False})
        if res.success and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        raise RuntimeError("GARCH(1,1) QMLE failed to converge from any starting point.")
    omega, alpha, beta = [float(z) for z in best.x]
    h0 = float(omega / max(1 - alpha - beta, 1e-6))
    return omega, alpha, beta, h0


def _garch_std_resid(e, omega, alpha, beta, h0):
    import numpy as np
    h    = np.empty(len(e))
    h[0] = max(h0, 1e-12)
    for t in range(1, len(e)):
        h[t] = max(omega + alpha * e[t-1]**2 + beta * h[t-1], 1e-12)
    return e / np.sqrt(h)


def estimate_gbm_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd

    r              = _get_r(log_rets_df).to_numpy()
    p              = simulation_priors_df["GBM"]
    S0, dt, method = float(p["S0"]), float(p["dt"]), str(p["method"])

    if len(r) < 2:
        raise ValueError("Need at least 2 observations.")
    if method not in ("mle", "unbiased"):
        raise ValueError("method must be 'mle' or 'unbiased'.")

    ddof      = 0 if method == "mle" else 1
    mean_step = float(r.mean())
    var_step  = float(r.var(ddof=ddof))
    sigma     = float(np.sqrt(var_step / dt))
    mu        = float(mean_step / dt + 0.5 * sigma ** 2)
    years     = len(r) * dt
    mu_se     = float(sigma / np.sqrt(years))  # SE(mu) = sigma / sqrt(years of history)

    return pd.Series({
        "S0": S0, "dt": dt, "mu": mu, "sigma": sigma, "mu_se": mu_se,
        "mu_log_step": (mu - 0.5 * sigma**2) * dt,
        "sigma_step":  sigma * np.sqrt(dt),
        "mean_step":   mean_step, "var_step": var_step,
    }, name="GBM", dtype=object)


def estimate_piecewise_gbm_inputs(log_rets_df, simulation_priors_df):
    # Piecewise GBM shares GBM's calibrated (mu, sigma) as its single default
    # "phase" — the schedule only matters once a scenario is attached at
    # simulation time (see DRIFTR_SCENARIOS.resolve_phase_schedule).
    s = estimate_gbm_inputs(log_rets_df, simulation_priors_df)
    return s.rename("PiecewiseGBM")


def estimate_heston_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd

    rets              = _get_r(log_rets_df)
    p                 = simulation_priors_df["Heston"]
    S0, dt, rv_window, use_mu = float(p["S0"]), float(p["dt"]), int(p["rv_window"]), bool(p["use_mu"])

    if len(rets) < rv_window + 5:
        raise ValueError("Not enough history for Heston calibration.")

    v_series = (rets.rolling(rv_window).var(ddof=1) / dt).dropna()
    if len(v_series) < 3:
        raise ValueError("Realized variance series too short.")

    v0, theta = float(v_series.iloc[-1]), float(v_series.mean())
    v_t, v_lag = v_series.iloc[1:], v_series.shift(1).iloc[1:]
    x, y = v_lag.to_numpy(), v_t.to_numpy()

    x_var = float(np.var(x, ddof=1))
    if x_var <= 0:
        raise ValueError("Cannot estimate Heston mean reversion.")

    b            = np.cov(x, y, ddof=1)[0, 1] / x_var
    a            = y.mean() - b * x.mean()
    heston_kappa = float(-np.log(float(np.clip(b, 1e-6, 0.999999))) / dt)
    eps_norm     = (y - (a + b * x)) / (np.sqrt(np.maximum(x, 1e-12)) * np.sqrt(dt))
    sigma_v      = float(np.std(eps_norm, ddof=1))

    idx    = v_t.index.intersection(rets.index)
    x_al   = v_lag.loc[idx].to_numpy()
    vt_al  = v_t.loc[idx].to_numpy()
    eps_al = (vt_al - (a + b * x_al)) / (np.sqrt(np.maximum(x_al, 1e-12)) * np.sqrt(dt))
    z_r    = rets.loc[idx].to_numpy() / np.sqrt(np.maximum(x_al * dt, 1e-12))
    rho    = float(np.clip(np.corrcoef(z_r, eps_al)[0, 1], -0.999, 0.999))
    mu     = float(rets.mean() / dt) if use_mu else 0.0
    years  = len(rets) * dt
    mu_se  = float(np.sqrt(max(theta, 0.0)) / np.sqrt(years))  # theta stands in for "sigma" here

    return pd.Series({
        "S0": S0, "dt": dt, "mu": mu, "mu_se": mu_se, "v0": v0, "theta": theta,
        "heston_kappa": heston_kappa, "sigma_v": sigma_v, "rho": rho, "rv_window": rv_window,
    }, name="Heston", dtype=object)


def estimate_jumpdiffusion_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd

    r = _get_r(log_rets_df)
    p = simulation_priors_df["JumpDiffusion"]

    S0                  = float(p["S0"])
    dt                  = float(p["dt"])
    jump_threshold_mult = float(p["jump_threshold_mult"])
    jump_lambda_prior   = float(p["jump_lambda_prior"])
    jump_mean_prior     = float(p["jump_mean_prior"])
    jump_sd_prior       = float(p["jump_sd_prior"])
    jump_shrink         = float(p["jump_shrink"])

    if len(r) < 30:
        raise ValueError("Need at least 30 observations for Jump-Diffusion calibration.")

    jump_thresh = jump_threshold_mult * float(np.median(np.abs(r.to_numpy())))
    jump_mask   = r.abs() > jump_thresh
    jump_obs    = r[jump_mask]
    diffusive   = r[~jump_mask]

    if len(diffusive) < 10:
        raise ValueError("Too few non-jump observations left to fit the diffusive component.")

    mean_step = float(diffusive.mean())
    var_step  = float(diffusive.var(ddof=1))
    sigma     = float(np.sqrt(var_step / dt))
    mu        = float(mean_step / dt + 0.5 * sigma ** 2)
    diffusive_years = max(len(diffusive) * dt, 1e-6)
    mu_se     = float(sigma / np.sqrt(diffusive_years))

    years            = max(len(r) / 252, 1e-6)
    jump_lam_raw     = len(jump_obs) / years
    jump_lambda_year = float((1 - jump_shrink) * jump_lam_raw + jump_shrink * jump_lambda_prior)
    jump_mean_raw    = float(jump_obs.mean()) if len(jump_obs) > 0 else 0.0
    jump_sd_raw      = float(jump_obs.std(ddof=1)) if len(jump_obs) > 1 else 0.0
    jump_mean        = float((1 - jump_shrink) * jump_mean_raw + jump_shrink * jump_mean_prior)
    jump_sd          = float((1 - jump_shrink) * jump_sd_raw   + jump_shrink * jump_sd_prior)

    return pd.Series({
        "S0": S0, "dt": dt, "mu": mu, "sigma": sigma, "mu_se": mu_se,
        "jump_lambda_year": jump_lambda_year, "jump_mean": jump_mean, "jump_sd": jump_sd,
        "jump_threshold_mult": jump_threshold_mult, "n_jumps_detected": len(jump_obs),
    }, name="JumpDiffusion", dtype=object)


def estimate_trendmr_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd

    r = _get_r(log_rets_df)
    p = simulation_priors_df["TrendMR"]

    S0           = float(p["S0"])
    dt           = float(p["dt"])
    trend_window = int(p["trend_window"])
    kappa_min    = float(p["kappa_min"])
    kappa_max    = float(p["kappa_max"])
    sigma_floor  = float(p["sigma_floor"])

    if len(r) < trend_window + 30:
        raise ValueError("Not enough history for Trend+Mean-Reversion calibration.")

    logp = r.cumsum()
    m    = logp.rolling(trend_window, min_periods=trend_window).mean()
    x    = (logp - m).dropna()

    if len(x) < 10:
        raise ValueError("Not enough deviation observations after trend-windowing.")

    # x_t's long-run mean is fixed at 0 by construction (it's a deviation from
    # its own trailing trend) — deliberately discard _ar1_ols's intercept/mean.
    phi, _, sigma_resid = _ar1_ols(x)
    phi   = float(np.clip(phi, 1e-3, 0.999))
    kappa = float(np.clip(-np.log(phi) / dt, kappa_min, kappa_max))
    sigma_x = sigma_resid * np.sqrt(2 * kappa / max(1 - np.exp(-2 * kappa * dt), 1e-8))
    sigma_x = float(max(sigma_x, sigma_floor))

    dm      = m.diff().dropna()
    mu_m    = float(dm.mean() / dt)
    sigma_m = float(max(dm.std(ddof=1) / np.sqrt(dt), sigma_floor))
    x0      = float(x.iloc[-1])
    years   = len(dm) * dt
    mu_m_se = float(sigma_m / np.sqrt(years))

    return pd.Series({
        "S0": S0, "dt": dt, "trend_window": trend_window,
        "mu_m": mu_m, "mu_m_se": mu_m_se, "sigma_m": sigma_m,
        "kappa": kappa, "sigma_x": sigma_x, "x0": x0,
    }, name="TrendMR", dtype=object)


def estimate_regimeswitching_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd

    r = _get_r(log_rets_df)
    p = simulation_priors_df["RegimeSwitching"]

    S0               = float(p["S0"])
    dt               = float(p["dt"])
    regime_window    = int(p["regime_window"])
    min_regime_days  = int(p["min_regime_days"])
    recovery_window  = int(p["recovery_window"])
    shrink_pseudo_n  = float(p["shrink_pseudo_n"])
    sigma_min        = float(p["sigma_min"])
    sigma_max        = float(p["sigma_max"])

    if len(r) < regime_window + 30:
        raise ValueError("Not enough history for regime-switching calibration.")

    # Shifted by 1 day: day t's regime is classified using the window through
    # t-1, never including r_t itself. Without this, a day's own return feeds
    # both its regime label and the regime's attributed mean return, which
    # mechanically inflates e.g. "Bull" regime's realized mean (you're
    # partly rediscovering the selection criterion). This also matches how
    # the model is used going forward: diagnose the regime as of now, then
    # simulate future returns from it.
    roll_mu  = (r.rolling(regime_window, min_periods=regime_window).mean() * 252).shift(1).dropna()
    roll_sig = (r.rolling(regime_window, min_periods=regime_window).std(ddof=1) * np.sqrt(252)).shift(1).dropna()
    idx      = roll_mu.index.intersection(roll_sig.index)
    mu_ann, sig_ann = roll_mu.loc[idx], roll_sig.loc[idx]

    q90_sig, q40_sig = sig_ann.quantile([0.90, 0.40])
    q75_mu,  q25_mu  = mu_ann.quantile([0.75, 0.25])

    # Assigned lowest-priority-first so later overwrites act like an if/elif
    # chain with Crash checked first, then Bull, then Bear, then Sideways.
    label = pd.Series("Normal", index=idx)
    label[sig_ann <= q40_sig] = "Sideways"
    label[(mu_ann <= q25_mu) & (sig_ann < q90_sig)] = "Bear"
    label[(mu_ann >= q75_mu) & (sig_ann < q90_sig)] = "Bull"
    label[(sig_ann >= q90_sig) & (mu_ann <= 0)]     = "Crash"

    # Recovery overlay: Bull/Normal days within recovery_window trading days
    # after the most recent Crash/Bear day, with positive proxy drift.
    is_trough   = label.isin(["Crash", "Bear"]).to_numpy()
    positions   = np.arange(len(label))
    trough_pos  = np.where(is_trough, positions, -1)
    last_trough = pd.Series(trough_pos, index=label.index).replace(-1, np.nan).ffill()
    has_trough  = last_trough.notna().to_numpy()
    days_since  = positions - np.nan_to_num(last_trough.to_numpy(), nan=0.0)
    recovery_mask = (
        has_trough
        & (days_since <= recovery_window)
        & (mu_ann.to_numpy() > 0)
        & label.isin(["Bull", "Normal"]).to_numpy()
    )
    label[recovery_mask] = "Recovery"

    # Small-sample guard: thin regimes merge into "Normal".
    counts_by_label = label.value_counts()
    thin = counts_by_label[counts_by_label < min_regime_days].index
    label[label.isin(thin)] = "Normal"

    regimes = ["Bull", "Normal", "Bear", "Crash", "Recovery", "Sideways"]

    mean_step_g  = float(r.mean())
    var_step_g   = float(r.var(ddof=1))
    sigma_global = float(np.sqrt(var_step_g / dt))
    mu_global    = float(mean_step_g / dt + 0.5 * sigma_global ** 2)

    r_aligned = r.loc[label.index]

    out = {
        "S0": S0, "dt": dt,
        "regime_labels": regimes,
        "initial_regime": str(label.iloc[-1]),
    }
    for g in regimes:
        rg  = r_aligned[label == g]
        n_g = len(rg)
        if n_g > 0:
            mean_g      = float(rg.mean())
            var_g       = float(rg.var(ddof=1)) if n_g > 1 else 0.0
            sigma_g_raw = float(np.sqrt(max(var_g, 0.0) / dt))
            mu_g_raw    = float(mean_g / dt + 0.5 * sigma_g_raw ** 2)
        else:
            sigma_g_raw, mu_g_raw = sigma_global, mu_global
        w        = shrink_pseudo_n / (shrink_pseudo_n + n_g)
        mu_g     = w * mu_global + (1 - w) * mu_g_raw
        sigma_g  = float(np.clip(w * sigma_global + (1 - w) * sigma_g_raw, sigma_min, sigma_max))
        years_g  = max(n_g, 1) * dt
        mu_g_se  = float(sigma_g / np.sqrt(years_g))  # thin regimes (e.g. Crash) get a wider jitter
        out[f"mu_{g}"]    = mu_g
        out[f"mu_{g}_se"] = mu_g_se
        out[f"sigma_{g}"] = sigma_g

    # Daily-frequency, Laplace-smoothed transition matrix over the fixed regime set.
    pos    = {g: i for i, g in enumerate(regimes)}
    codes  = label.map(pos).to_numpy()
    k      = len(regimes)
    counts = np.zeros((k, k))
    np.add.at(counts, (codes[:-1], codes[1:]), 1)
    eps_smooth = 0.5
    P = (counts + eps_smooth) / (counts + eps_smooth).sum(axis=1, keepdims=True)
    out["transition_matrix"] = P

    return pd.Series(out, name="RegimeSwitching", dtype=object)


def estimate_garch_inputs(log_rets_df, simulation_priors_df):
    import numpy as np
    import pandas as pd
    from scipy.stats import skewnorm

    r = _get_r(log_rets_df).to_numpy()
    p = simulation_priors_df["GARCH"]

    S0           = float(p["S0"])
    dt           = float(p["dt"])
    shock_method = str(p["shock_method"]).lower()

    if len(r) < 30:
        raise ValueError("Need at least 30 observations for GARCH calibration.")

    mean_step = float(r.mean())
    e         = r - mean_step

    garch_omega, garch_alpha, garch_beta, garch_h0 = _garch11_qmle(e)
    persistence = garch_alpha + garch_beta
    if persistence >= 0.999:
        raise ValueError(
            f"GARCH(1,1) fit is near-integrated (alpha+beta={persistence:.4f}); "
            "variance-target retargeting under a scenario will be unreliable for this ticker."
        )

    z = _garch_std_resid(e, garch_omega, garch_alpha, garch_beta, garch_h0)

    if shock_method == "skewnorm" and len(z) >= 50:
        a_sn, loc_sn, scale_sn = skewnorm.fit(z)
        garch_shock_dist = "skewnorm"
        shock_a, shock_loc, shock_scale = float(a_sn), float(loc_sn), float(scale_sn)
    else:
        garch_shock_dist = "normal"
        shock_a = shock_loc = shock_scale = np.nan

    # garch_mu_step is a raw DAILY mean (unlike GBM's annualized mu), so its
    # SE is the plain SE of the sample mean — no annualization.
    garch_mu_se = float(np.std(r, ddof=1) / np.sqrt(len(r)))

    return pd.Series({
        "S0": S0, "dt": dt,
        "garch_omega": garch_omega, "garch_alpha": garch_alpha,
        "garch_beta": garch_beta, "garch_h0": garch_h0,
        "garch_mu_step": mean_step, "garch_mu_annual": mean_step * 252, "garch_mu_se": garch_mu_se,
        "garch_shock_dist": garch_shock_dist,
        "shock_a": shock_a, "shock_loc": shock_loc, "shock_scale": shock_scale,
    }, name="GARCH", dtype=object)


_ESTIMATORS = {
    "GBM":             estimate_gbm_inputs,
    "PiecewiseGBM":    estimate_piecewise_gbm_inputs,
    "Heston":          estimate_heston_inputs,
    "JumpDiffusion":   estimate_jumpdiffusion_inputs,
    "TrendMR":         estimate_trendmr_inputs,
    "RegimeSwitching": estimate_regimeswitching_inputs,
    "GARCH":           estimate_garch_inputs,
}

_PARAM_ORDER = [
    "S0", "dt",
    "mu", "mu_se", "sigma", "mu_log_step", "sigma_step", "mean_step", "var_step",
    "v0", "theta", "heston_kappa", "sigma_v", "rho", "rv_window",
    "jump_lambda_year", "jump_mean", "jump_sd", "jump_threshold_mult", "n_jumps_detected",
    "trend_window", "mu_m", "mu_m_se", "sigma_m", "kappa", "sigma_x", "x0",
    "regime_labels", "transition_matrix", "initial_regime",
    "mu_Bull", "mu_Bull_se", "sigma_Bull", "mu_Normal", "mu_Normal_se", "sigma_Normal",
    "mu_Bear", "mu_Bear_se", "sigma_Bear", "mu_Crash", "mu_Crash_se", "sigma_Crash",
    "mu_Recovery", "mu_Recovery_se", "sigma_Recovery", "mu_Sideways", "mu_Sideways_se", "sigma_Sideways",
    "garch_omega", "garch_alpha", "garch_beta", "garch_h0",
    "garch_mu_step", "garch_mu_annual", "garch_mu_se", "garch_shock_dist",
    "shock_a", "shock_loc", "shock_scale",
]


def estimate_all_simulation_inputs(log_rets_df, simulation_priors_df, methods=None, existing_params_df=None):
    import pandas as pd

    valid   = list(_ESTIMATORS.keys())
    methods = valid if methods is None else ([methods] if isinstance(methods, str) else methods)
    bad     = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Valid: {valid}")

    new_cols  = [_ESTIMATORS[m](log_rets_df, simulation_priors_df) for m in methods]
    new_df    = pd.concat(new_cols, axis=1)

    if existing_params_df is None:
        params_df = new_df.copy()
    else:
        params_df = existing_params_df.drop(columns=[m for m in methods if m in existing_params_df.columns])
        params_df = pd.concat([params_df, new_df], axis=1)

    ordered   = [r for r in _PARAM_ORDER if r in params_df.index]
    remaining = [r for r in params_df.index if r not in ordered]
    params_df = params_df.loc[ordered + remaining]
    params_df = params_df[[m for m in valid if m in params_df.columns]]
    params_df.index.name = "parameter"
    return params_df
