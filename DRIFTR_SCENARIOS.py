# Scenario layer: a scenario is an ordered list of narrative "phases", each
# carrying a target annualized drift/vol (and optionally a scripted jump).
# `scenario=None` is the "Baseline" sentinel everywhere in DRIFTR_SIM — it
# means "run the engine's own calibrated constant-parameter behavior,
# unchanged." A named scenario's targets always override an engine's
# calibrated constants (never blend with them), so the same scenario is an
# apples-to-apples comparison across all six engines.

SCENARIO_LIBRARY = {
    "Baseline": None,
    "Steady Growth": {"phases": [
        {"name": "Growth", "duration_frac": 1.0, "drift_annual": 0.10, "vol_annual": 0.14},
    ]},
    "Choppy Uptrend": {"phases": [
        {"name": "Grind Up 1", "duration_frac": 0.25, "drift_annual": 0.14, "vol_annual": 0.16},
        {"name": "Pullback 1", "duration_frac": 0.25, "drift_annual": -0.05, "vol_annual": 0.22},
        {"name": "Grind Up 2", "duration_frac": 0.25, "drift_annual": 0.16, "vol_annual": 0.17},
        {"name": "Pullback 2", "duration_frac": 0.25, "drift_annual": -0.04, "vol_annual": 0.20},
    ]},
    "Momentum Rally": {"phases": [
        {"name": "Breakout", "duration_frac": 0.3, "drift_annual": 0.25, "vol_annual": 0.18},
        {"name": "Euphoria", "duration_frac": 0.3, "drift_annual": 0.45, "vol_annual": 0.26},
        {"name": "Cooling",  "duration_frac": 0.4, "drift_annual": 0.05, "vol_annual": 0.30},
    ]},
    "Sideways": {"phases": [
        {"name": "Range-bound", "duration_frac": 1.0, "drift_annual": 0.0, "vol_annual": 0.15},
    ]},
    "Slow Decline": {"phases": [
        {"name": "Decline", "duration_frac": 1.0, "drift_annual": -0.08, "vol_annual": 0.16},
    ]},
    "Crash": {"phases": [
        {"name": "Calm",          "duration_frac": 0.15, "drift_annual": 0.02,  "vol_annual": 0.12},
        {"name": "Crash",         "duration_frac": 0.15, "drift_annual": -0.60, "vol_annual": 0.55,
         "jump": {"prob_at_phase_start": 1.0, "jump_mean": -0.09, "jump_sd": 0.02}},
        {"name": "Aftermath",     "duration_frac": 0.30, "drift_annual": -0.10, "vol_annual": 0.40},
        {"name": "Stabilization", "duration_frac": 0.40, "drift_annual": 0.03,  "vol_annual": 0.20},
    ]},
    "Crash & Recovery": {"phases": [
        {"name": "Pre-Crash",     "duration_frac": 0.10, "drift_annual": 0.02,  "vol_annual": 0.12},
        {"name": "Crash",         "duration_frac": 0.10, "drift_annual": -0.55, "vol_annual": 0.55,
         "jump": {"prob_at_phase_start": 1.0, "jump_mean": -0.08, "jump_sd": 0.02}},
        {"name": "Stabilization", "duration_frac": 0.20, "drift_annual": -0.02, "vol_annual": 0.30},
        {"name": "Recovery",      "duration_frac": 0.30, "drift_annual": 0.35,  "vol_annual": 0.22},
        {"name": "Normal",        "duration_frac": 0.30, "drift_annual": 0.09,  "vol_annual": 0.16},
    ]},
    "Boom & Bust": {"phases": [
        {"name": "Boom",      "duration_frac": 0.35, "drift_annual": 0.35,  "vol_annual": 0.18},
        {"name": "Peak",      "duration_frac": 0.15, "drift_annual": 0.15,  "vol_annual": 0.30},
        {"name": "Bust",      "duration_frac": 0.20, "drift_annual": -0.50, "vol_annual": 0.50,
         "jump": {"prob_at_phase_start": 1.0, "jump_mean": -0.07, "jump_sd": 0.02}},
        {"name": "Aftermath", "duration_frac": 0.30, "drift_annual": -0.05, "vol_annual": 0.30},
    ]},
}


def build_black_swan_scenario(log_rets_df, severity=1.25):
    """Unlike everything in SCENARIO_LIBRARY, "worse than the historical worst
    case" is inherently ticker-specific — there's no universal number for it.

    Reuses the static "Crash" scenario's own phase shape and magnitudes
    (already validated to produce a severe-but-plausible year, not a runaway
    one) and changes exactly one thing: the scripted jump is calibrated to
    THIS ticker's own worst-ever single day, amplified by `severity`.

    An earlier version also scaled the phase drift/vol themselves from this
    ticker's worst realized month/volatility spike — numerically that blew
    up, because the engine's own Ito correction term (drift - 0.5*vol^2)
    multiplies an already-amplified vol against itself, and stacking that on
    top of an equally-amplified jump and a continued-severe aftermath
    compounds far past "a bit worse than history." A single amplified jump is
    the clean, bounded, literal reading of "worse than the worst day this
    stock has ever had" — it doesn't need a second amplified lever alongside it.
    """
    import pandas as pd

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()
    r = log_rets_df.iloc[:, 0].astype(float).dropna()
    if len(r) < 30:
        raise ValueError("Not enough history to calibrate a Black Swan scenario.")

    worst_day = float(r.min())
    jump_mean = worst_day * severity
    jump_sd   = float(r.std(ddof=1)) * 1.5

    return {"phases": [
        {"name": "Calm",          "duration_frac": 0.15, "drift_annual": 0.02,  "vol_annual": 0.12},
        {"name": "Black Swan",    "duration_frac": 0.15, "drift_annual": -0.60, "vol_annual": 0.55,
         "jump": {"prob_at_phase_start": 1.0, "jump_mean": jump_mean, "jump_sd": jump_sd}},
        {"name": "Aftermath",     "duration_frac": 0.30, "drift_annual": -0.10, "vol_annual": 0.40},
        {"name": "Stabilization", "duration_frac": 0.40, "drift_annual": 0.03,  "vol_annual": 0.20},
    ]}


def _largest_remainder_round(fracs, total):
    """Round `fracs * total` to integers that sum exactly to `total`, with
    every entry forced to >=1 (raises if there are more phases than steps)."""
    import numpy as np

    fracs = np.asarray(fracs, dtype=float)
    k = len(fracs)
    if total < k:
        raise ValueError(f"Horizon T={total} is too short for {k} scenario phases (need >=1 step each).")

    fracs  = fracs / fracs.sum()
    raw    = fracs * total
    counts = np.floor(raw).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1

    zero_idx = np.where(counts == 0)[0]
    for i in zero_idx:
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[i] = 1
    return counts


def resolve_phase_schedule(scenario, T):
    """Shared deterministic (T,)-length schedule used by every engine except
    Regime-Switching's scenario-conditioned path. Returns None for Baseline."""
    import numpy as np

    if scenario is None:
        return None

    phases = scenario["phases"]
    fracs  = [p["duration_frac"] for p in phases]
    if abs(sum(fracs) - 1.0) > 0.05:
        raise ValueError(f"Scenario phase duration_frac values sum to {sum(fracs):.3f}, expected ~1.0.")

    durations  = _largest_remainder_round(fracs, T)
    boundaries = np.concatenate([[0], np.cumsum(durations)]).tolist()

    mu_sched  = np.empty(T)
    vol_sched = np.empty(T)
    phase_id  = np.empty(T, dtype=int)
    jump_events = []

    for i, p in enumerate(phases):
        s, e = boundaries[i], boundaries[i + 1]
        mu_sched[s:e]  = float(p["drift_annual"])
        vol_sched[s:e] = float(p["vol_annual"])
        phase_id[s:e]  = i
        if "jump" in p:
            ev = dict(p["jump"])
            ev["start_step"]   = int(s)
            ev["end_step"]     = int(e)
            ev["phase_index"]  = i
            jump_events.append(ev)

    return {
        "mu_sched": mu_sched, "vol_sched": vol_sched, "phase_id": phase_id,
        "phase_names": [p["name"] for p in phases],
        "boundaries": boundaries,
        "jump_events": jump_events,
    }


def adapt_scenario_regimeswitching(scenario, T, N, rng, concentration=8.0):
    """Regime-Switching is the one engine where scenario phases get per-path
    stochastic timing (gamma-distributed durations around each phase's target
    length) rather than one shared schedule — every path tells the same
    narrative but lives through it on its own clock, which is what makes
    10,000 paths "conditional on Crash & Recovery" meaningfully different
    from each other instead of differing only by return noise."""
    import numpy as np

    rng = np.random.default_rng(rng)
    phases = scenario["phases"]
    k = len(phases)
    if T < k:
        raise ValueError(f"Horizon T={T} is too short for {k} scenario phases (need >=1 step each).")

    fracs = np.array([p["duration_frac"] for p in phases], dtype=float)
    fracs = fracs / fracs.sum()
    expected_days = fracs * T

    raw = rng.gamma(shape=concentration, scale=expected_days / concentration, size=(N, k))
    norm = raw / raw.sum(axis=1, keepdims=True) * T

    counts = np.floor(norm).astype(int)
    remainder = T - counts.sum(axis=1)
    frac_part = norm - counts
    rank = np.argsort(np.argsort(-frac_part, axis=1), axis=1)
    counts += (rank < remainder[:, None]).astype(int)

    # repair any zero-duration phases per path (rare with concentration>=8)
    for n in np.where((counts == 0).any(axis=1))[0]:
        for i in np.where(counts[n] == 0)[0]:
            donor = int(np.argmax(counts[n]))
            counts[n, donor] -= 1
            counts[n, i] = 1

    bounds = np.concatenate([np.zeros((N, 1), dtype=int), np.cumsum(counts, axis=1)], axis=1)  # (N, k+1)

    steps      = np.arange(T)
    bounds_upper = bounds[:, 1:]                                       # (N, k)
    cond       = steps[:, None, None] >= bounds_upper[None, :, :]      # (T, N, k)
    phase_id   = np.clip(cond.sum(axis=2), 0, k - 1).astype(int)       # (T, N)

    drift_vals = np.array([float(p["drift_annual"]) for p in phases])
    vol_vals   = np.array([float(p["vol_annual"]) for p in phases])
    mu_annual  = drift_vals[phase_id]
    vol_annual = vol_vals[phase_id]

    jump_events = []
    for i, p in enumerate(phases):
        if "jump" in p:
            ev = dict(p["jump"])
            ev["phase_index"]        = i
            ev["start_step_per_path"] = bounds[:, i].copy()
            ev["end_step_per_path"]   = bounds[:, i + 1].copy()
            jump_events.append(ev)

    return {
        "phase_id": phase_id, "mu_annual": mu_annual, "vol_annual": vol_annual,
        "phase_names": [p["name"] for p in phases],
        "jump_events": jump_events,
    }


def apply_scheduled_jumps(jump_events, T, N, rng):
    """Shared-schedule (single path-independent timing) jump injector — used
    by Piecewise GBM, Jump-Diffusion, and Regime-Switching's Baseline is
    exempt (it uses its own ambient process instead). Returns a (T,N)
    additive log-return shock array."""
    import numpy as np

    rng = np.random.default_rng(rng)
    J = np.zeros((T, N))
    for ev in jump_events:
        jump_mean, jump_sd = float(ev["jump_mean"]), float(ev["jump_sd"])
        if "prob_at_phase_start" in ev:
            prob   = float(ev["prob_at_phase_start"])
            occurs = rng.random(N) < prob
            if occurs.any():
                J[ev["start_step"], occurs] += rng.normal(jump_mean, jump_sd, occurs.sum())
        elif "lambda_annual" in ev:
            lam_daily = float(ev["lambda_annual"]) / 252.0
            for t in range(ev["start_step"], ev["end_step"]):
                occurs = rng.random(N) < lam_daily
                if occurs.any():
                    J[t, occurs] += rng.normal(jump_mean, jump_sd, occurs.sum())
        else:
            raise ValueError("Scenario jump spec needs 'prob_at_phase_start' or 'lambda_annual'.")
    return J


def apply_scheduled_jumps_per_path(jump_events, T, N, rng):
    """Per-path variant for Regime-Switching's scenario-conditioned timing,
    where each jump event's phase-start step differs by path."""
    import numpy as np

    rng = np.random.default_rng(rng)
    J = np.zeros((T, N))
    for ev in jump_events:
        jump_mean, jump_sd = float(ev["jump_mean"]), float(ev["jump_sd"])
        starts = ev["start_step_per_path"]
        if "prob_at_phase_start" in ev:
            prob   = float(ev["prob_at_phase_start"])
            occurs = rng.random(N) < prob
            idx    = np.where(occurs)[0]
            if len(idx):
                J[starts[idx], idx] += rng.normal(jump_mean, jump_sd, len(idx))
        elif "lambda_annual" in ev:
            ends      = ev["end_step_per_path"]
            lam_daily = float(ev["lambda_annual"]) / 252.0
            for t in range(T):
                active = (t >= starts) & (t < ends)
                if not active.any():
                    continue
                occurs = active & (rng.random(N) < lam_daily)
                if occurs.any():
                    J[t, occurs] += rng.normal(jump_mean, jump_sd, occurs.sum())
        else:
            raise ValueError("Scenario jump spec needs 'prob_at_phase_start' or 'lambda_annual'.")
    return J
