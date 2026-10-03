_ENGINE_COLORS = {
    "PiecewiseGBM":    "#1f77b4",
    "JumpDiffusion":   "#d62728",
    "Heston":          "#ff7f0e",
    "TrendMR":         "#2ca02c",
    "GARCH":           "#9467bd",
    "RegimeSwitching": "#8c564b",
}


def plot_price_and_log_rets(price_df, log_rets_df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ticker = str(price_df.columns[0])
    fig    = make_subplots(rows=1, cols=2,
                           subplot_titles=[f"{ticker} Price", f"{ticker} Log Returns"],
                           horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df.iloc[:, 0],
                             mode="lines", name="Price", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=log_rets_df.index, y=log_rets_df.iloc[:, 0],
                             mode="lines", name="Log Return", line=dict(color="#ff7f0e", width=0.8)), row=1, col=2)
    fig.update_layout(title=f"{ticker}", template="plotly_white",
                      height=450, width=1400, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18, x=0))
    fig.update_yaxes(title_text="Price",      row=1, col=1)
    fig.update_yaxes(title_text="Log Return", row=1, col=2)
    return fig


def plot_empirical_price_paths(log_rets_df, empirical_paths, bins=60,
                                show_t_dropdown=True, path_opacity=0.12):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    ticker   = str(log_rets_df.columns[0])
    horizons = (21, 63, 126, 252)
    colors   = {21: "#1f77b4", 63: "#ff7f0e", 126: "#2ca02c", 252: "#d62728"}
    selected_color = "#9467bd"
    r        = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    ncols    = 5 if show_t_dropdown else 4
    btitles  = ["H = 21 days","H = 63 days","H = 126 days","H = 252 days"]
    if show_t_dropdown:
        btitles.append("Selected t")

    fig = make_subplots(rows=2, cols=ncols,
                        specs=[[{"colspan": ncols}] + [None]*(ncols-1), [{} for _ in range(ncols)]],
                        subplot_titles=[""] + btitles,
                        vertical_spacing=0.14, horizontal_spacing=0.035)

    P    = empirical_paths[252].to_numpy(float)
    days = np.arange(253)
    segment_bounds = [(0,21),(21,63),(63,126),(126,252)]
    segment_labels = ["Days 0–21","Days 22–63","Days 64–126","Days 127–252"]

    for (s, e), H, label in zip(segment_bounds, horizons, segment_labels):
        seg    = P[:, s:e+1]
        x_seg  = np.tile(np.append(days[s:e+1], np.nan), P.shape[0])
        y_seg  = np.hstack([seg, np.full((seg.shape[0],1), np.nan)]).ravel()
        fig.add_trace(go.Scattergl(x=x_seg, y=y_seg, mode="lines",
                                   line=dict(color=colors[H], width=1),
                                   opacity=path_opacity, name=label, hoverinfo="skip"), row=1, col=1)

    for arr, name, dash, w in [
        (np.nanpercentile(P,10,axis=0), "P10",     "dash",  2.5),
        (np.nanpercentile(P,50,axis=0), "Median",  "solid", 3.0),
        (np.nanpercentile(P,90,axis=0), "P90",     "dash",  2.5),
    ]:
        fig.add_trace(go.Scattergl(x=days, y=arr, mode="lines",
                                   line=dict(color="black", width=w, dash=dash),
                                   name=name, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scattergl(x=days, y=np.zeros(len(days)), mode="lines",
                               line=dict(color="black", width=1.5, dash="dot"),
                               name="Zero", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=days, y=np.ones(len(days)), mode="lines",
                               line=dict(color="black", width=2),
                               name="Baseline", hoverinfo="skip"), row=1, col=1)

    for j, H in enumerate(horizons, start=1):
        wins    = np.lib.stride_tricks.sliding_window_view(r, H)
        H_rets  = wins.sum(axis=1)
        counts, edges = np.histogram(H_rets, bins=bins, density=True)
        centers = (edges[:-1]+edges[1:])/2
        fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges)*0.95,
                             marker=dict(color=colors[H], line=dict(color="black",width=0.4)),
                             opacity=0.80, showlegend=False), row=2, col=j)
        for val, dash in [(np.percentile(H_rets,10),"dash"),
                          (np.percentile(H_rets,50),"solid"),
                          (np.percentile(H_rets,90),"dash")]:
            fig.add_vline(x=val, line_width=3, line_dash=dash, line_color="black", row=2, col=j)
        fig.update_xaxes(title_text=f"{H}-day log return", row=2, col=j)

    n_fixed = len(fig.data)

    if show_t_dropdown:
        dynamic = {}
        y_min   = float(np.nanmin(P))
        y_max   = float(np.nanmax(P))

        for t in range(1, 253):
            wins_t  = np.lib.stride_tricks.sliding_window_view(r, t)
            t_rets  = wins_t.sum(axis=1)
            counts, edges = np.histogram(t_rets, bins=bins, density=True)
            centers = (edges[:-1]+edges[1:])/2
            p10, p50, p90 = np.percentile(t_rets, [10,50,90])
            hist_ymax = counts.max() * 1.05

            fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges)*0.95,
                                 marker=dict(color=selected_color, line=dict(color="black",width=0.5)),
                                 opacity=0.80, visible=False, showlegend=False), row=2, col=5)
            hi = len(fig.data)-1

            for val, dash in [(p10,"dash"),(p50,"solid"),(p90,"dash")]:
                fig.add_trace(go.Scatter(x=[val,val], y=[0,hist_ymax], mode="lines",
                                         line=dict(color="black",width=3,dash=dash),
                                         visible=False, showlegend=False, hoverinfo="skip"), row=2, col=5)

            fig.add_trace(go.Scattergl(x=[t,t], y=[y_min,y_max], mode="lines",
                                       line=dict(color="black",width=5,dash="dash"),
                                       visible=False, showlegend=False, hoverinfo="skip"), row=1, col=1)
            vli = len(fig.data)-1
            dynamic[t] = (hi, hi+1, hi+2, hi+3, vli)

        n_total = len(fig.data)
        buttons = []
        vis_off = [False]*n_total
        for i in range(n_fixed): vis_off[i] = True
        buttons.append(dict(label="Off", method="update",
                            args=[{"visible": vis_off}, {"title": f"{ticker}: Empirical price paths"}]))
        for t in range(1, 253):
            vis = [False]*n_total
            for i in range(n_fixed): vis[i] = True
            for idx in dynamic[t]: vis[idx] = True
            buttons.append(dict(label=f"{t} days", method="update",
                                args=[{"visible": vis},
                                      {"title": f"{ticker}: Empirical price paths — t = {t}"}]))

        fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down",
                                            x=1.0, xanchor="right", y=1.14, yanchor="top",
                                            showactive=True)])
        fig.update_xaxes(title_text="Selected t-day log return", row=2, col=5)

    fig.update_layout(title=f"{ticker}: Empirical price paths", template="plotly_white",
                      height=900, width=1600 if show_t_dropdown else 1400,
                      barmode="overlay",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hovermode=False)
    fig.update_xaxes(title_text="Trading days", range=[0,252], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative gross return (start = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    return fig


def plot_empirical_price_paths_simple(log_rets_df, empirical_paths, bins=60, path_opacity=0.12):
    """Same as plot_empirical_price_paths, minus the per-day 'Selected t'
    dropdown (which builds ~1,000 extra traces for a feature most callers
    don't need). Kept as a separate function rather than a flag so the
    full version stays available for whoever wants the t-scrubber later."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    ticker   = str(log_rets_df.columns[0])
    horizons = (21, 63, 126, 252)
    colors   = {21: "#1f77b4", 63: "#ff7f0e", 126: "#2ca02c", 252: "#d62728"}
    r        = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    btitles  = ["H = 21 days", "H = 63 days", "H = 126 days", "H = 252 days"]

    fig = make_subplots(rows=2, cols=4,
                        specs=[[{"colspan": 4}, None, None, None], [{}, {}, {}, {}]],
                        subplot_titles=[""] + btitles,
                        vertical_spacing=0.14, horizontal_spacing=0.035)

    P    = empirical_paths[252].to_numpy(float)
    days = np.arange(253)
    segment_bounds = [(0,21),(21,63),(63,126),(126,252)]
    segment_labels = ["Days 0–21","Days 22–63","Days 64–126","Days 127–252"]

    for (s, e), H, label in zip(segment_bounds, horizons, segment_labels):
        seg    = P[:, s:e+1]
        x_seg  = np.tile(np.append(days[s:e+1], np.nan), P.shape[0])
        y_seg  = np.hstack([seg, np.full((seg.shape[0],1), np.nan)]).ravel()
        fig.add_trace(go.Scattergl(x=x_seg, y=y_seg, mode="lines",
                                   line=dict(color=colors[H], width=1),
                                   opacity=path_opacity, name=label, hoverinfo="skip"), row=1, col=1)

    for arr, name, dash, w in [
        (np.nanpercentile(P,10,axis=0), "P10",     "dash",  2.5),
        (np.nanpercentile(P,50,axis=0), "Median",  "solid", 3.0),
        (np.nanpercentile(P,90,axis=0), "P90",     "dash",  2.5),
    ]:
        fig.add_trace(go.Scattergl(x=days, y=arr, mode="lines",
                                   line=dict(color="black", width=w, dash=dash),
                                   name=name, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scattergl(x=days, y=np.zeros(len(days)), mode="lines",
                               line=dict(color="black", width=1.5, dash="dot"),
                               name="Zero", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=days, y=np.ones(len(days)), mode="lines",
                               line=dict(color="black", width=2),
                               name="Baseline", hoverinfo="skip"), row=1, col=1)

    for j, H in enumerate(horizons, start=1):
        wins    = np.lib.stride_tricks.sliding_window_view(r, H)
        H_rets  = wins.sum(axis=1)
        counts, edges = np.histogram(H_rets, bins=bins, density=True)
        centers = (edges[:-1]+edges[1:])/2
        fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges)*0.95,
                             marker=dict(color=colors[H], line=dict(color="black",width=0.4)),
                             opacity=0.80, showlegend=False), row=2, col=j)
        for val, dash in [(np.percentile(H_rets,10),"dash"),
                          (np.percentile(H_rets,50),"solid"),
                          (np.percentile(H_rets,90),"dash")]:
            fig.add_vline(x=val, line_width=3, line_dash=dash, line_color="black", row=2, col=j)
        fig.update_xaxes(title_text=f"{H}-day log return", row=2, col=j)

    fig.update_layout(title=f"{ticker}: Empirical price paths", template="plotly_white",
                      height=900, width=1400,
                      barmode="overlay",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hovermode=False)
    fig.update_xaxes(title_text="Trading days", range=[0,252], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative gross return (start = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    return fig


def plot_terminal_path_stats_table(terminal_stats_df, ticker=None):
    import pandas as pd
    import plotly.graph_objects as go

    df      = terminal_stats_df.T
    headers = ["Statistic"] + [f"H = {h}" for h in df.columns]
    cells   = [df.index.tolist()] + [df[h].round(4).tolist() for h in df.columns]
    fig     = go.Figure(go.Table(
        header=dict(values=headers, fill_color="#2c2c2c",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=cells,
                   fill_color=[["#f9f9f9" if i%2==0 else "white" for i in range(len(df.index))]],
                   font=dict(size=11), align="left"),
    ))
    fig.update_layout(
        title="Terminal path stats" + (f" ({ticker})" if ticker else ""),
        height=900, width=1000,
    )
    return fig


def plot_stats_table(stats_df, title=None):
    """A deliberately-styled stats table (horizon x statistic) that looks
    right on a light page, with per-row number formatting — the replacement
    for both plot_terminal_path_stats_table's dark header (which clashed with
    a dark Streamlit theme) and plain st.dataframe's generic look."""
    import numpy as np
    import plotly.graph_objects as go

    df      = stats_df.T  # stats as rows, horizons as columns
    headers = ["Statistic"] + [f"H = {h}" for h in df.columns]

    def _fmt(stat_name, v):
        if not np.isfinite(v):
            return "—"
        if stat_name == "n_paths":
            return f"{v:,.0f}"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:,.4f}"

    cells = [df.index.tolist()] + [
        [_fmt(stat, v) for stat, v in zip(df.index, df[h])] for h in df.columns
    ]

    n_rows  = len(df.index)
    n_cols  = len(df.columns)
    row_colors = ["#f3f4f6" if i % 2 == 0 else "#ffffff" for i in range(n_rows)]

    fig = go.Figure(go.Table(
        columnwidth=[1.7] + [1.0] * n_cols,
        header=dict(values=headers, fill_color="#1f2937",
                   font=dict(color="white", size=13), align=["left"] + ["right"] * n_cols,
                   height=32),
        cells=dict(values=cells, fill_color=[row_colors],
                  font=dict(size=12, color="#111827"), align=["left"] + ["right"] * n_cols,
                  height=26),
    ))
    fig.update_layout(
        title=title, paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=min(900, 70 + 26 * n_rows),
    )
    return fig


def plot_simulated_paths(results, bins=60, path_opacity=0.04, thin=0.5, ticker=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    methods  = [m for m in results if not m.startswith("_")]
    T        = results["_T"]
    horizons = (21, 63, 126, 252)
    colors   = {21:"#1f77b4", 63:"#ff7f0e", 126:"#2ca02c", 252:"#d62728"}
    segment_bounds = [(0,21),(21,63),(63,126),(126,252)]
    segment_labels = ["Days 0–21","Days 22–63","Days 64–126","Days 127–252"]

    if not (0 < thin <= 1):
        raise ValueError("thin must be in (0, 1].")

    fig = make_subplots(rows=2, cols=4,
                        specs=[[{"colspan":4},None,None,None],[{},{},{},{}]],
                        subplot_titles=["","H = 21 days","H = 63 days","H = 126 days","H = 252 days"],
                        vertical_spacing=0.14, horizontal_spacing=0.035)

    days          = np.arange(0, T+1)
    method_ranges = {}

    for m in methods:
        R        = results[m].to_numpy(float)
        cum      = np.hstack([np.zeros((R.shape[1],1)), R.cumsum(axis=0).T])
        N        = cum.shape[0]
        n_thin   = max(1, int(N * thin))
        idx      = np.random.choice(N, size=n_thin, replace=False)
        cum_thin = cum[idx, :]
        start    = len(fig.data)

        for (s, e), H, label in zip(segment_bounds, horizons, segment_labels):
            seg    = cum_thin[:, s:e+1]
            y_flat = np.hstack([seg, np.full((n_thin,1), np.nan)]).ravel()
            x_flat = np.tile(np.append(days[s:e+1], np.nan), n_thin)
            fig.add_trace(go.Scattergl(x=x_flat, y=y_flat, mode="lines",
                                       line=dict(color=colors[H], width=0.5),
                                       opacity=path_opacity, name=label, hoverinfo="skip"), row=1, col=1)

        for arr, name, dash, w in [
            (np.nanpercentile(cum,10,axis=0), "P10",    "dash",  2.5),
            (np.nanpercentile(cum,50,axis=0), "Median", "solid", 3.0),
            (np.nanpercentile(cum,90,axis=0), "P90",    "dash",  2.5),
        ]:
            fig.add_trace(go.Scattergl(x=days, y=arr, mode="lines",
                                       line=dict(color="black", width=w, dash=dash),
                                       name=name, hoverinfo="skip"), row=1, col=1)

        fig.add_trace(go.Scattergl(x=days, y=np.zeros(T+1), mode="lines",
                                   line=dict(color="black", width=1.5, dash="dot"),
                                   name="Zero", hoverinfo="skip"), row=1, col=1)

        for j, H in enumerate(horizons, start=1):
            terminal         = cum[:, H]
            counts, edges    = np.histogram(terminal, bins=bins, density=True)
            centers          = (edges[:-1]+edges[1:])/2
            fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges)*0.95,
                                 marker=dict(color=colors[H], line=dict(color="black",width=0.4)),
                                 opacity=0.80, showlegend=False), row=2, col=j)
            # Real traces, not fig.add_vline shapes — shapes are layout-level
            # and ignore the per-method visibility toggle below, so every
            # method's markers would otherwise stack up on screen at once.
            ymax = counts.max() * 1.05 if len(counts) else 1.0
            for val, dash in [(np.percentile(terminal,10),"dash"),
                              (np.percentile(terminal,50),"solid"),
                              (np.percentile(terminal,90),"dash")]:
                fig.add_trace(go.Scatter(x=[val,val], y=[0,ymax], mode="lines",
                                         line=dict(color="black", width=3, dash=dash),
                                         showlegend=False, hoverinfo="skip"), row=2, col=j)

        method_ranges[m] = (start, len(fig.data))

    n_total = len(fig.data)
    first_m = methods[0]
    s0, e0  = method_ranges[first_m]
    for i in range(n_total):
        fig.data[i].visible = (s0 <= i < e0)

    buttons = []
    for m in methods:
        s, e    = method_ranges[m]
        visible = [s <= i < e for i in range(n_total)]
        buttons.append(dict(label=m, method="update",
                            args=[{"visible": visible},
                                  {"title": f"{m}" + (f" ({ticker})" if ticker else "") + " — Simulated paths"}]))

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down",
                          x=0.0, xanchor="left", y=1.10, yanchor="top", showactive=True)],
        title=f"{first_m}" + (f" ({ticker})" if ticker else "") + " — Simulated paths",
        template="plotly_white", height=900, width=1400,
        barmode="overlay", hovermode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Trading days", range=[0,T], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative log-return", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    for j, H in enumerate(horizons, start=1):
        fig.update_xaxes(title_text=f"{H}-day log return", row=2, col=j)
    return fig


def plot_simulated_price_paths(prices, params_df, bins=60, path_opacity=0.04, thin=0.5, ticker=None):
    """Same structure as plot_simulated_paths, but plotted in actual price
    level (starting from the stock's current price) rather than cumulative
    log-return (starting from zero) — more intuitive for a retail audience.
    Engine switcher dropdown lives top-right, matching plot_empirical_price_paths."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    methods  = [m for m in prices if not m.startswith("_")]
    T        = len(prices[methods[0]])
    horizons = (21, 63, 126, 252)
    colors   = {21: "#1f77b4", 63: "#ff7f0e", 126: "#2ca02c", 252: "#d62728"}
    segment_bounds = [(0,21),(21,63),(63,126),(126,252)]
    segment_labels = ["Days 0–21","Days 22–63","Days 64–126","Days 127–252"]

    if not (0 < thin <= 1):
        raise ValueError("thin must be in (0, 1].")

    fig = make_subplots(rows=2, cols=4,
                        specs=[[{"colspan":4},None,None,None],[{},{},{},{}]],
                        subplot_titles=["","H = 21 days","H = 63 days","H = 126 days","H = 252 days"],
                        vertical_spacing=0.14, horizontal_spacing=0.035)

    days          = np.arange(0, T+1)
    method_ranges = {}

    for m in methods:
        S0  = float(params_df.loc["S0", m])
        P   = prices[m].to_numpy(float).T                    # (N, T)
        cum = np.hstack([np.full((P.shape[0], 1), S0), P])   # prepend day-0 = current price
        N        = cum.shape[0]
        n_thin   = max(1, int(N * thin))
        idx      = np.random.choice(N, size=n_thin, replace=False)
        cum_thin = cum[idx, :]
        start    = len(fig.data)

        for (s, e), H, label in zip(segment_bounds, horizons, segment_labels):
            seg    = cum_thin[:, s:e+1]
            y_flat = np.hstack([seg, np.full((n_thin,1), np.nan)]).ravel()
            x_flat = np.tile(np.append(days[s:e+1], np.nan), n_thin)
            fig.add_trace(go.Scattergl(x=x_flat, y=y_flat, mode="lines",
                                       line=dict(color=colors[H], width=0.5),
                                       opacity=path_opacity, name=label, hoverinfo="skip"), row=1, col=1)

        for arr, name, dash, w in [
            (np.nanpercentile(cum,10,axis=0), "P10",    "dash",  2.5),
            (np.nanpercentile(cum,50,axis=0), "Median", "solid", 3.0),
            (np.nanpercentile(cum,90,axis=0), "P90",    "dash",  2.5),
        ]:
            fig.add_trace(go.Scattergl(x=days, y=arr, mode="lines",
                                       line=dict(color="black", width=w, dash=dash),
                                       name=name, hoverinfo="skip"), row=1, col=1)

        fig.add_trace(go.Scattergl(x=days, y=np.full(T+1, S0), mode="lines",
                                   line=dict(color="black", width=1.5, dash="dot"),
                                   name="Current Price", hoverinfo="skip"), row=1, col=1)

        for j, H in enumerate(horizons, start=1):
            terminal         = cum[:, H]
            counts, edges    = np.histogram(terminal, bins=bins, density=True)
            centers          = (edges[:-1]+edges[1:])/2
            fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges)*0.95,
                                 marker=dict(color=colors[H], line=dict(color="black",width=0.4)),
                                 opacity=0.80, showlegend=False), row=2, col=j)
            # Real traces (not fig.add_vline shapes) so these respond to the
            # dropdown's visibility toggle below — add_vline shapes are layout-
            # level and stay on screen for every method at once otherwise,
            # which is what produced "too many vertical bars" with 6 engines.
            ymax = counts.max() * 1.05 if len(counts) else 1.0
            for val, dash in [(np.percentile(terminal,10),"dash"),
                              (np.percentile(terminal,50),"solid"),
                              (np.percentile(terminal,90),"dash")]:
                fig.add_trace(go.Scatter(x=[val,val], y=[0,ymax], mode="lines",
                                         line=dict(color="black", width=3, dash=dash),
                                         showlegend=False, hoverinfo="skip"), row=2, col=j)

        method_ranges[m] = (start, len(fig.data))

    n_total = len(fig.data)
    first_m = methods[0]
    s0, e0  = method_ranges[first_m]
    for i in range(n_total):
        fig.data[i].visible = (s0 <= i < e0)

    buttons = []
    for m in methods:
        s, e    = method_ranges[m]
        visible = [s <= i < e for i in range(n_total)]
        buttons.append(dict(label=m, method="update",
                            args=[{"visible": visible},
                                  {"title": f"{m}" + (f" ({ticker})" if ticker else "") + " — Simulated price paths"}]))

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down",
                          x=1.0, xanchor="right", y=1.10, yanchor="top", showactive=True)],
        title=f"{first_m}" + (f" ({ticker})" if ticker else "") + " — Simulated price paths",
        template="plotly_white", height=900, width=1400,
        barmode="overlay", hovermode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Trading days", range=[0,T], row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    for j, H in enumerate(horizons, start=1):
        fig.update_xaxes(title_text=f"{H}-day price", row=2, col=j)
    return fig


def plot_simulated_stats_table(sim_stats, ticker=None):
    import plotly.graph_objects as go

    methods = list(sim_stats.keys())
    first_m = methods[0]

    def _make_table(m):
        df      = sim_stats[m].T
        headers = ["Statistic"] + [f"H = {h}" for h in df.columns]
        cells   = [df.index.tolist()] + [df[h].round(4).tolist() for h in df.columns]
        return go.Table(
            header=dict(values=headers, fill_color="#2c2c2c",
                        font=dict(color="white", size=12), align="left"),
            cells=dict(values=cells,
                       fill_color=[["#f9f9f9" if i%2==0 else "white" for i in range(len(df.index))]],
                       font=dict(size=11), align="left"),
        )

    fig    = go.Figure()
    ranges = {}
    for m in methods:
        start = len(fig.data)
        fig.add_trace(_make_table(m))
        ranges[m] = (start, len(fig.data))

    for i in range(len(fig.data)):
        fig.data[i].visible = (i == 0)

    buttons = []
    for m in methods:
        s, e    = ranges[m]
        visible = [s <= i < e for i in range(len(fig.data))]
        buttons.append(dict(label=m, method="update",
                            args=[{"visible": visible},
                                  {"title": f"Simulated terminal return stats — {m}" +
                                            (f" ({ticker})" if ticker else "")}]))

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down",
                          x=1.0, xanchor="right", y=1.10, yanchor="top", showactive=True)],
        title=f"Simulated terminal return stats — {first_m}" + (f" ({ticker})" if ticker else ""),
        height=900, width=1000,
    )
    return fig


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _fan_traces(df, name, color):
    import numpy as np
    import plotly.graph_objects as go

    arr   = df.to_numpy(float)
    steps = df.index.to_numpy()
    p10   = np.nanpercentile(arr, 10, axis=1)
    p50   = np.nanpercentile(arr, 50, axis=1)
    p90   = np.nanpercentile(arr, 90, axis=1)
    rgb   = _hex_to_rgb(color)
    fill  = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)"
    return [
        go.Scatter(x=steps, y=p90, mode="lines", line=dict(color=color, width=0.8, dash="dash"),
                   name=f"{name} P90", hoverinfo="skip"),
        go.Scatter(x=steps, y=p10, mode="lines", line=dict(color=color, width=0.8, dash="dash"),
                   fill="tonexty", fillcolor=fill, name=f"{name} P10", hoverinfo="skip"),
        go.Scatter(x=steps, y=p50, mode="lines", line=dict(color=color, width=2),
                   name=f"{name} P50", hoverinfo="skip"),
    ]


def _hist_trace(df, name, color, nbins=80, skip_zeros=False):
    import numpy as np
    import plotly.graph_objects as go

    vals = df.to_numpy(float).ravel()
    if skip_zeros:
        vals = vals[vals != 0.0]
    vals = vals[np.isfinite(vals)]
    counts, edges = np.histogram(vals, bins=nbins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return go.Bar(x=centers, y=counts, width=np.diff(edges) * 0.95,
                  marker=dict(color=color, line=dict(color="black", width=0.3)),
                  opacity=0.8, name=name, hoverinfo="skip")


def plot_components_heston(components, ticker=None):
    from plotly.subplots import make_subplots

    c   = components["Heston"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Variance V", "Vol σ = √V", "Return shocks ε", "Correlated shocks Zs"],
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for tr in _fan_traces(c["V"],     "V",  "#1f77b4"): fig.add_trace(tr, row=1, col=1)
    for tr in _fan_traces(c["sigma"], "σ",  "#ff7f0e"): fig.add_trace(tr, row=1, col=2)
    fig.add_trace(_hist_trace(c["eps"], "ε",  "#2ca02c"), row=2, col=1)
    fig.add_trace(_hist_trace(c["Zs"],  "Zs", "#9467bd"), row=2, col=2)
    fig.update_layout(title="Heston components" + (f" ({ticker})" if ticker else ""),
                      template="plotly_white", height=700, width=1200,
                      hovermode=False, barmode="overlay",
                      legend=dict(orientation="h", y=-0.12, x=0))
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="Variance",   row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=1, col=2)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)
    return fig


def plot_components_trendmr(components, ticker=None):
    from plotly.subplots import make_subplots

    c   = components["TrendMR"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Deviation x (mean-reverting)", "Trend increment dm", "Total return ε = dm + dx"],
                        horizontal_spacing=0.07)
    for tr in _fan_traces(c["x"], "x", "#2ca02c"): fig.add_trace(tr, row=1, col=1)
    fig.add_trace(_hist_trace(c["dm"],  "dm", "#1f77b4"), row=1, col=2)
    fig.add_trace(_hist_trace(c["eps"], "ε",  "#9467bd"), row=1, col=3)
    fig.update_layout(title="Trend + Mean-Reversion components" + (f" ({ticker})" if ticker else ""),
                      template="plotly_white", height=450, width=1500,
                      hovermode=False, barmode="overlay",
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="Deviation x", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Value", row=1, col=3)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=3)
    return fig


def plot_components_jumpdiffusion(components, ticker=None):
    from plotly.subplots import make_subplots

    c   = components["JumpDiffusion"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Return shocks ε", "Jump component J (non-zero)", "Jump frequency per step"],
                        horizontal_spacing=0.07)
    import numpy as np
    import plotly.graph_objects as go

    fig.add_trace(_hist_trace(c["eps"], "ε", "#1f77b4"), row=1, col=1)
    fig.add_trace(_hist_trace(c["J"], "J (non-zero)", "#d62728", skip_zeros=True), row=1, col=2)
    J_arr = c["J"].to_numpy(float)
    freq  = (J_arr != 0.0).mean(axis=1)
    steps = c["J"].index.to_numpy()
    fig.add_trace(go.Scatter(x=steps, y=freq, mode="lines",
                             line=dict(color="#8c564b", width=1.5),
                             name="Jump freq", hoverinfo="skip"), row=1, col=3)
    fig.update_layout(title="Jump-Diffusion components" + (f" ({ticker})" if ticker else ""),
                      template="plotly_white", height=450, width=1500,
                      hovermode=False, barmode="overlay",
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Step",  row=1, col=3)
    fig.update_yaxes(title_text="Density",    row=1, col=1)
    fig.update_yaxes(title_text="Density",    row=1, col=2)
    fig.update_yaxes(title_text="Frac paths", row=1, col=3)
    return fig


def plot_components_garch(components, ticker=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    c = components["GARCH"]
    mu_component = np.asarray(c["mu_component"])
    # Baseline: (1,N), one jittered drift per path (constant over time) ->
    # show the across-path distribution. Scenario: (T,1), one schedule shared
    # by every path (constant across paths) -> show it as a line over time.
    is_per_path = mu_component.shape[0] == 1

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Conditional variance H",
                                        "Drift across paths" if is_per_path else "Drift μ_t (shared across paths)",
                                        "Shocks = √H·Z", "Raw shocks Z"],
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for tr in _fan_traces(c["H"], "H", "#1f77b4"): fig.add_trace(tr, row=1, col=1)

    if is_per_path:
        counts, edges = np.histogram(mu_component.ravel(), bins=60, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges) * 0.95,
                             marker=dict(color="#ff7f0e", line=dict(color="black", width=0.3)),
                             opacity=0.8, name="μ (per path)", hoverinfo="skip"), row=1, col=2)
        fig.update_xaxes(title_text="Daily drift", row=1, col=2)
        fig.update_yaxes(title_text="Density",     row=1, col=2)
    else:
        steps = c["H"].index.to_numpy()
        fig.add_trace(go.Scatter(x=steps, y=mu_component[:, 0], mode="lines",
                                 line=dict(color="#ff7f0e", width=2), name="μ_t", hoverinfo="skip"), row=1, col=2)
        fig.update_xaxes(title_text="Step",  row=1, col=2)
        fig.update_yaxes(title_text="Drift", row=1, col=2)

    fig.add_trace(_hist_trace(c["shock"], "shock = √H·Z", "#2ca02c"), row=2, col=1)
    fig.add_trace(_hist_trace(c["Z"],     "Z",            "#9467bd"), row=2, col=2)
    fig.update_layout(title="GARCH components" + (f" ({ticker})" if ticker else ""),
                      template="plotly_white", height=700, width=1200,
                      hovermode=False, barmode="overlay",
                      legend=dict(orientation="h", y=-0.12, x=0))
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)
    return fig


def plot_components_regimeswitching(components, params_df, ticker=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    c       = components["RegimeSwitching"]
    regimes = list(params_df["RegimeSwitching"]["regime_labels"])
    regime_arr = c["regime"].to_numpy(int)
    steps      = c["regime"].index.to_numpy()
    T, N       = regime_arr.shape

    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#ff7f0e"]
    occ = np.stack([(regime_arr == i).mean(axis=1) for i in range(len(regimes))], axis=1)  # (T, k)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regime occupancy share over time", "Return shocks ε"],
                        horizontal_spacing=0.08)

    cum = np.zeros(T)
    for i, name in enumerate(regimes):
        fig.add_trace(go.Scatter(x=steps, y=cum + occ[:, i], mode="lines",
                                 line=dict(width=0.5, color=palette[i % len(palette)]),
                                 fill="tonexty" if i > 0 else "tozeroy",
                                 name=name, hoverinfo="skip"), row=1, col=1)
        cum = cum + occ[:, i]

    fig.add_trace(_hist_trace(c["eps"], "ε", "#1f77b4"), row=1, col=2)
    fig.update_layout(title="Regime-Switching components" + (f" ({ticker})" if ticker else ""),
                      template="plotly_white", height=500, width=1300,
                      hovermode=False, barmode="overlay",
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="Share of paths", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    return fig


def plot_phase_shading(fig, sched, row=1, col=1, y0=None, y1=None, opacity=0.08):
    """Overlay vertical bands marking each scenario phase on an existing
    figure. `sched` is the dict returned by DRIFTR_SCENARIOS.resolve_phase_schedule
    (pass None for Baseline — this is then a no-op)."""
    if sched is None:
        return fig

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    boundaries = sched["boundaries"]
    names      = sched["phase_names"]

    for i, name in enumerate(names):
        fig.add_vrect(
            x0=boundaries[i], x1=boundaries[i + 1],
            fillcolor=palette[i % len(palette)], opacity=opacity, line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation=dict(font_size=10),
            row=row, col=col,
        )
    return fig


def plot_scenario_comparison(results, scenario_name=None, percentiles=(10, 50, 90), ticker=None):
    """Overlay percentile bands for the same scenario across all engines
    present in `results`, directly answering "how sensitive is the simulated
    distribution to model choice?" for a given narrative."""
    import numpy as np
    import plotly.graph_objects as go

    methods = [m for m in results if not m.startswith("_")]
    T       = results["_T"]
    days    = np.arange(0, T + 1)

    fig = go.Figure()
    p_lo, p_mid, p_hi = percentiles

    for m in methods:
        color = _ENGINE_COLORS.get(m, "#7f7f7f")
        R     = results[m].to_numpy(float)
        cum   = np.vstack([np.zeros((1, R.shape[1])), R.cumsum(axis=0)])  # (T+1, N)

        lo  = np.percentile(cum, p_lo, axis=1)
        mid = np.percentile(cum, p_mid, axis=1)
        hi  = np.percentile(cum, p_hi, axis=1)
        rgb = _hex_to_rgb(color)

        fig.add_trace(go.Scatter(x=days, y=hi, mode="lines",
                                 line=dict(color=color, width=0.8, dash="dash"),
                                 name=f"{m} P{p_hi}", hoverinfo="skip", legendgroup=m))
        fig.add_trace(go.Scatter(x=days, y=lo, mode="lines",
                                 line=dict(color=color, width=0.8, dash="dash"),
                                 fill="tonexty", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)",
                                 name=f"{m} P{p_lo}", hoverinfo="skip", legendgroup=m))
        fig.add_trace(go.Scatter(x=days, y=mid, mode="lines",
                                 line=dict(color=color, width=2.5),
                                 name=f"{m} P{p_mid}", legendgroup=m))

    fig.add_trace(go.Scatter(x=days, y=np.zeros(T + 1), mode="lines",
                             line=dict(color="black", width=1, dash="dot"),
                             name="Zero", hoverinfo="skip"))

    title = "Scenario comparison across engines"
    if scenario_name:
        title += f" — {scenario_name}"
    if ticker:
        title += f" ({ticker})"

    fig.update_layout(title=title, template="plotly_white", height=650, width=1300,
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(title_text="Trading days")
    fig.update_yaxes(title_text="Cumulative log-return")
    return fig


def plot_empirical_at_t(log_rets_df, empirical_paths, t, bins=60, path_opacity=0.12, ticker=None):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import skew, kurtosis

    if isinstance(log_rets_df, pd.Series):
        log_rets_df = log_rets_df.to_frame()

    ticker   = ticker or str(log_rets_df.columns[0])
    r        = log_rets_df.iloc[:, 0].astype(float).dropna().to_numpy()
    wins     = np.lib.stride_tricks.sliding_window_view(r, t)
    cum_log  = np.hstack([np.zeros((wins.shape[0], 1)), wins.cumsum(axis=1)])
    terminal = cum_log[:, -1]
    days     = np.arange(0, t + 1)
    colors   = {21: "#1f77b4", 63: "#ff7f0e", 126: "#2ca02c", 252: "#d62728"}
    color    = colors.get(t, "#9467bd")

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.35, 0.10],
        specs=[[{}], [{}], [{"type": "table"}]],
        subplot_titles=[
            f"Empirical cumulative log-return paths up to t = {t}",
            f"Terminal log-return distribution at t = {t}",
            "",
        ],
        vertical_spacing=0.08,
    )

    nan_col = np.full((cum_log.shape[0], 1), np.nan)
    y_flat  = np.hstack([cum_log, nan_col]).ravel()
    x_flat  = np.tile(np.append(days, np.nan), cum_log.shape[0])
    fig.add_trace(go.Scattergl(
        x=x_flat, y=y_flat, mode="lines",
        line=dict(color=color, width=0.8),
        opacity=path_opacity, name="Paths", hoverinfo="skip",
    ), row=1, col=1)

    for arr, name, dash, w in [
        (np.nanpercentile(cum_log, 10, axis=0), "P10",    "dash",  2.0),
        (np.nanpercentile(cum_log, 50, axis=0), "Median", "solid", 2.5),
        (np.nanpercentile(cum_log, 90, axis=0), "P90",    "dash",  2.0),
    ]:
        fig.add_trace(go.Scattergl(
            x=days, y=arr, mode="lines",
            line=dict(color="black", width=w, dash=dash),
            name=name, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=days, y=np.zeros(len(days)), mode="lines",
        line=dict(color="black", width=1.5, dash="dot"),
        name="Zero", hoverinfo="skip",
    ), row=1, col=1)

    counts, edges = np.histogram(terminal, bins=bins, density=True)
    centers       = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centers, y=counts, width=np.diff(edges) * 0.95,
        marker=dict(color=color, line=dict(color="black", width=0.3)),
        opacity=0.8, name=f"t={t}", showlegend=False,
    ), row=2, col=1)

    p10, p50, p90 = np.percentile(terminal, [10, 50, 90])
    for val, dash in [(p10, "dash"), (p50, "solid"), (p90, "dash")]:
        fig.add_vline(x=val, line_width=2.5, line_dash=dash, line_color="black", row=2, col=1)

    stats_labels = ["n", "mean", "std", "skew", "kurt", "P5", "P25", "P50", "P75", "P95"]
    stats_values = [
        f"{len(terminal):,}",
        f"{terminal.mean():.4f}",
        f"{terminal.std(ddof=1):.4f}",
        f"{skew(terminal, bias=False):.3f}",
        f"{kurtosis(terminal, fisher=True, bias=False):.3f}",
        f"{np.percentile(terminal,  5):.4f}",
        f"{np.percentile(terminal, 25):.4f}",
        f"{np.percentile(terminal, 50):.4f}",
        f"{np.percentile(terminal, 75):.4f}",
        f"{np.percentile(terminal, 95):.4f}",
    ]
    fig.add_trace(go.Table(
        header=dict(values=stats_labels, fill_color="#2c2c2c",
                    font=dict(color="white", size=11), align="center"),
        cells=dict(values=[[v] for v in stats_values],
                   fill_color="#f9f9f9", font=dict(size=11), align="center"),
    ), row=3, col=1)

    fig.update_layout(
        title=f"{ticker} — Empirical paths at t = {t}",
        template="plotly_white", height=900, width=1200,
        hovermode=False, barmode="overlay",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(title_text="Day",                   row=1, col=1)
    fig.update_yaxes(title_text="Cumulative log-return", row=1, col=1)
    fig.update_xaxes(title_text="Terminal log-return",   row=2, col=1)
    fig.update_yaxes(title_text="Density",               row=2, col=1)
    return fig


def plot_simulated_at_t(results, t, bins=60, path_opacity=0.04, thin=0.3, ticker=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import skew, kurtosis

    methods = [m for m in results if not m.startswith("_")]
    T       = results["_T"]

    if t > T:
        raise ValueError(f"t={t} exceeds simulation length T={T}.")

    days = np.arange(0, t + 1)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.35, 0.15],
        specs=[[{}], [{}], [{"type": "table"}]],
        subplot_titles=[
            f"Simulated cumulative log-return paths up to t = {t}",
            f"Terminal log-return distribution at t = {t}",
            "",
        ],
        vertical_spacing=0.08,
    )

    method_ranges = {}

    for m in methods:
        color    = _ENGINE_COLORS.get(m, "#9467bd")
        R        = results[m].to_numpy(float)
        cum      = np.hstack([np.zeros((R.shape[1], 1)), R.cumsum(axis=0).T])
        cum_t    = cum[:, :t + 1]
        terminal = cum[:, t]
        N        = cum.shape[0]
        start    = len(fig.data)

        n_thin  = max(1, int(N * thin))
        idx     = np.random.choice(N, size=n_thin, replace=False)
        seg     = cum_t[idx, :]
        nan_col = np.full((n_thin, 1), np.nan)
        y_flat  = np.hstack([seg, nan_col]).ravel()
        x_flat  = np.tile(np.append(days, np.nan), n_thin)

        fig.add_trace(go.Scattergl(
            x=x_flat, y=y_flat, mode="lines",
            line=dict(color=color, width=0.5),
            opacity=path_opacity, name=m, hoverinfo="skip",
        ), row=1, col=1)

        for arr, name, dash, w in [
            (np.nanpercentile(cum_t, 10, axis=0), "P10",    "dash",  1.5),
            (np.nanpercentile(cum_t, 50, axis=0), "Median", "solid", 2.5),
            (np.nanpercentile(cum_t, 90, axis=0), "P90",    "dash",  1.5),
        ]:
            fig.add_trace(go.Scattergl(
                x=days, y=arr, mode="lines",
                line=dict(color=color, width=w, dash=dash),
                name=name, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

        fig.add_trace(go.Scattergl(
            x=days, y=np.zeros(t + 1), mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            name="Zero", hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

        counts, edges = np.histogram(terminal, bins=bins, density=True)
        centers       = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=np.diff(edges) * 0.95,
            marker=dict(color=color, line=dict(color="black", width=0.3)),
            opacity=0.8, name=m, showlegend=False,
        ), row=2, col=1)

        p10, p50, p90 = np.percentile(terminal, [10, 50, 90])
        for val, dash in [(p10, "dash"), (p50, "solid"), (p90, "dash")]:
            fig.add_trace(go.Scatter(
                x=[val, val], y=[0, counts.max() * 1.05],
                mode="lines", line=dict(color="black", width=2, dash=dash),
                hoverinfo="skip", showlegend=False,
            ), row=2, col=1)

        stats_labels = ["n", "mean", "std", "skew", "kurt", "P5", "P25", "P50", "P75", "P95"]
        stats_values = [
            f"{N:,}",
            f"{terminal.mean():.4f}",
            f"{terminal.std(ddof=1):.4f}",
            f"{skew(terminal, bias=False):.3f}",
            f"{kurtosis(terminal, fisher=True, bias=False):.3f}",
            f"{np.percentile(terminal,  5):.4f}",
            f"{np.percentile(terminal, 25):.4f}",
            f"{np.percentile(terminal, 50):.4f}",
            f"{np.percentile(terminal, 75):.4f}",
            f"{np.percentile(terminal, 95):.4f}",
        ]
        fig.add_trace(go.Table(
            header=dict(values=stats_labels, fill_color="#2c2c2c",
                        font=dict(color="white", size=11), align="center"),
            cells=dict(values=[[v] for v in stats_values],
                       fill_color="#f9f9f9", font=dict(size=11), align="center"),
        ), row=3, col=1)

        method_ranges[m] = (start, len(fig.data))

    n_total = len(fig.data)
    first_m = methods[0]
    s0, e0  = method_ranges[first_m]
    for i in range(n_total):
        fig.data[i].visible = (s0 <= i < e0)

    buttons = []
    for m in methods:
        s, e    = method_ranges[m]
        visible = [s <= i < e for i in range(n_total)]
        buttons.append(dict(
            label=m, method="update",
            args=[{"visible": visible},
                  {"title": (ticker or "") + f" — Simulated paths at t = {t} — {m}"}],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=1.0, xanchor="right", y=1.08, yanchor="top", showactive=True,
        )],
        title=(ticker or "") + f" — Simulated paths at t = {t} — {first_m}",
        template="plotly_white", height=900, width=1200,
        hovermode=False, barmode="overlay",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(title_text="Step",                  row=1, col=1)
    fig.update_yaxes(title_text="Cumulative log-return", row=1, col=1)
    fig.update_xaxes(title_text="Terminal log-return",   row=2, col=1)
    fig.update_yaxes(title_text="Density",               row=2, col=1)
    return fig
