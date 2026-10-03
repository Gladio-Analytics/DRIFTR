def describe_log_rets(log_rets_df, ann_days=252):
    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis, jarque_bera

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    results = []

    for ticker in log_rets_df.columns:
        r = log_rets_df[ticker].astype(float).dropna()
        if len(r) < 2:
            raise ValueError(f"Not enough observations for {ticker}.")

        mean_daily  = r.mean()
        std_daily   = r.std(ddof=1)
        var_daily   = r.var(ddof=1)
        years       = len(r) / ann_days
        total_return = np.exp(r.sum()) - 1
        cagr        = np.exp(r.sum() / years) - 1

        q = r.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        skewness     = skew(r, bias=False)
        excess_kurt  = kurtosis(r, fisher=True,  bias=False)
        regular_kurt = kurtosis(r, fisher=False, bias=False)
        jb           = jarque_bera(r)

        risk = {}
        for level in [0.90, 0.95, 0.99]:
            cutoff       = r.quantile(1 - level)
            pct          = int(level * 100)
            risk[f"var_{pct}"] = -cutoff
            risk[f"es_{pct}"]  = -r[r <= cutoff].mean()

        wealth          = np.exp(r.cumsum())
        running_max     = wealth.cummax()
        drawdown        = wealth / running_max - 1
        max_drawdown    = drawdown.min()
        max_drawdown_date = drawdown.idxmin()

        positive_share  = (r > 0).mean()
        negative_share  = (r < 0).mean()
        zero_share      = (r == 0).mean()
        downside_dev    = np.sqrt(np.mean(np.minimum(r, 0) ** 2))

        acf = {}
        for lag in [1, 5, 10, 20]:
            acf[f"acf_ret_{lag}"]    = r.autocorr(lag=lag)
            acf[f"acf_sq_ret_{lag}"] = (r ** 2).autocorr(lag=lag)

        results.append({
            "ticker":           ticker,
            "n":                len(r),
            "years":            round(years, 2),
            "mean_daily":       mean_daily,
            "std_daily":        std_daily,
            "var_daily":        var_daily,
            "total_return":     total_return,
            "cagr":             cagr,
            "q01": q[0.01], "q05": q[0.05], "q10": q[0.10],
            "q25": q[0.25], "q50": q[0.50], "q75": q[0.75],
            "q90": q[0.90], "q95": q[0.95], "q99": q[0.99],
            "skew":             skewness,
            "excess_kurtosis":  excess_kurt,
            "kurtosis":         regular_kurt,
            "jarque_bera":      jb.statistic,
            "jarque_bera_pvalue": jb.pvalue,
            "positive_share":   positive_share,
            "negative_share":   negative_share,
            "zero_share":       zero_share,
            "downside_dev":     downside_dev,
            "max_drawdown":     max_drawdown,
            "max_drawdown_date": max_drawdown_date,
            **risk,
            **acf,
        })

    return pd.DataFrame(results).set_index("ticker").T


def make_empirical_price_paths(log_rets_df, horizons=(21, 63, 126, 252)):
    import numpy as np
    import pandas as pd

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    r   = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    out = {}

    for h in horizons:
        wins      = np.lib.stride_tricks.sliding_window_view(r, h)
        cum       = np.hstack([np.ones((wins.shape[0], 1)), np.exp(wins.cumsum(axis=1))])
        steps     = np.arange(0, h + 1)
        out[h]    = pd.DataFrame(cum, columns=steps)

    return out


def terminal_path_stats(log_rets_df, empirical_paths, horizons=(21, 63, 126, 252), t=None):
    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis, jarque_bera

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    r            = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    all_horizons = list(horizons)
    if t is not None:
        all_horizons = all_horizons + [int(t)]

    rows = []

    for h in all_horizons:
        P        = empirical_paths[h].to_numpy(float) if h in empirical_paths else None
        if P is None:
            wins = np.lib.stride_tricks.sliding_window_view(r, h)
            tv   = wins.sum(axis=1)
        else:
            tv = P[:, -1] - 1.0   # terminal return

        x  = tv[np.isfinite(tv)]
        q  = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        jb = jarque_bera(x)

        risk = {}
        for level in [0.90, 0.95, 0.99]:
            cutoff          = np.quantile(x, 1 - level)
            tail            = x[x <= cutoff]
            pct             = int(level * 100)
            risk[f"var_{pct}"] = float(-cutoff)
            risk[f"es_{pct}"]  = float(-tail.mean()) if len(tail) > 0 else np.nan

        rows.append({
            "horizon":          h,
            "n_paths":          len(x),
            "mean":             float(x.mean()),
            "median":           float(np.median(x)),
            "std":              float(x.std(ddof=1)),
            "min":              float(x.min()),
            "max":              float(x.max()),
            "q01": float(q[0]), "q05": float(q[1]), "q10": float(q[2]),
            "q25": float(q[3]), "q50": float(q[4]), "q75": float(q[5]),
            "q90": float(q[6]), "q95": float(q[7]), "q99": float(q[8]),
            "skew":             float(skew(x, bias=False)),
            "excess_kurtosis":  float(kurtosis(x, fisher=True,  bias=False)),
            "kurtosis":         float(kurtosis(x, fisher=False, bias=False)),
            "jarque_bera":      float(jb.statistic),
            "jarque_bera_pvalue": float(jb.pvalue),
            "positive_share":   float(np.mean(x > 0)),
            "negative_share":   float(np.mean(x < 0)),
            "downside_dev":     float(np.sqrt(np.mean(np.minimum(x, 0) ** 2))),
            **risk,
        })

    return pd.DataFrame(rows).set_index("horizon")


def simulated_terminal_stats(results, horizons=(21, 63, 126, 252), t=None):
    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis, jarque_bera

    all_horizons = list(horizons)
    if t is not None:
        all_horizons = all_horizons + [int(t)]

    methods = [m for m in results if not m.startswith("_")]
    out     = {}

    for m in methods:
        rets_df = results[m]
        rows    = []

        for h in all_horizons:
            if h > len(rets_df):
                raise ValueError(f"Horizon {h} exceeds simulation length {len(rets_df)}.")

            x  = rets_df.iloc[:h].sum(axis=0).to_numpy(float)
            x  = x[np.isfinite(x)]
            q  = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            jb = jarque_bera(x)

            risk = {}
            for level in [0.90, 0.95, 0.99]:
                cutoff          = np.quantile(x, 1 - level)
                tail            = x[x <= cutoff]
                pct             = int(level * 100)
                risk[f"var_{pct}"] = float(-cutoff)
                risk[f"es_{pct}"]  = float(-tail.mean()) if len(tail) > 0 else np.nan

            rows.append({
                "horizon":          h,
                "n_paths":          len(x),
                "mean":             float(x.mean()),
                "median":           float(np.median(x)),
                "std":              float(x.std(ddof=1)),
                "min":              float(x.min()),
                "max":              float(x.max()),
                "q01": float(q[0]), "q05": float(q[1]), "q10": float(q[2]),
                "q25": float(q[3]), "q50": float(q[4]), "q75": float(q[5]),
                "q90": float(q[6]), "q95": float(q[7]), "q99": float(q[8]),
                "skew":             float(skew(x, bias=False)),
                "excess_kurtosis":  float(kurtosis(x, fisher=True,  bias=False)),
                "kurtosis":         float(kurtosis(x, fisher=False, bias=False)),
                "jarque_bera":      float(jb.statistic),
                "jarque_bera_pvalue": float(jb.pvalue),
                "positive_share":   float(np.mean(x > 0)),
                "negative_share":   float(np.mean(x < 0)),
                "downside_dev":     float(np.sqrt(np.mean(np.minimum(x, 0) ** 2))),
                **risk,
            })

        out[m] = pd.DataFrame(rows).set_index("horizon")

    return out


def compare_terminal_returns(results, log_rets_df, horizons=(21, 63, 126, 252), t=None):
    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis, jarque_bera

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    all_horizons = list(horizons)
    if t is not None:
        all_horizons = all_horizons + [int(t)]

    def _calc_stats(x, label):
        x  = np.asarray(x, dtype=float)
        x  = x[np.isfinite(x)]
        q  = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        jb = jarque_bera(x)
        risk = {}
        for level in [0.90, 0.95, 0.99]:
            cutoff          = np.quantile(x, 1 - level)
            tail            = x[x <= cutoff]
            pct             = int(level * 100)
            risk[f"var_{pct}"] = float(-cutoff)
            risk[f"es_{pct}"]  = float(-tail.mean()) if len(tail) > 0 else np.nan
        return {
            "label": label, "n": len(x),
            "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
            "q01": float(q[0]), "q05": float(q[1]), "q10": float(q[2]),
            "q25": float(q[3]), "q50": float(q[4]), "q75": float(q[5]),
            "q90": float(q[6]), "q95": float(q[7]), "q99": float(q[8]),
            "skew": float(skew(x, bias=False)),
            "excess_kurtosis": float(kurtosis(x, fisher=True,  bias=False)),
            "kurtosis":        float(kurtosis(x, fisher=False, bias=False)),
            "jarque_bera": float(jb.statistic), "jarque_bera_pvalue": float(jb.pvalue),
            "positive_share": float(np.mean(x > 0)), "negative_share": float(np.mean(x < 0)),
            "downside_dev": float(np.sqrt(np.mean(np.minimum(x, 0) ** 2))),
            **risk,
        }

    r   = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    out = {}

    for m in [m for m in results if not m.startswith("_")]:
        rets_df = results[m]
        rows    = []
        for h in all_horizons:
            if h > len(rets_df):
                raise ValueError(f"Horizon {h} exceeds simulation length {len(rets_df)}.")
            sim_cum = rets_df.iloc[:h].sum(axis=0).to_numpy()
            rows.append(_calc_stats(sim_cum, label=f"Sim H={h}"))
            emp_cum = np.lib.stride_tricks.sliding_window_view(r, h).sum(axis=1)
            rows.append(_calc_stats(emp_cum, label=f"Emp H={h}"))
        out[m] = pd.DataFrame(rows).set_index("label")

    return out
