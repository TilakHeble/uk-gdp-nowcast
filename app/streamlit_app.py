"""UK GDP nowcasting dashboard.

Reads the precomputed tables in outputs/tables (no model fitting at runtime).
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
H_LABEL = {-1: "Before the quarter", 0: "End of month 1", 1: "End of month 2",
           2: "End of month 3", 3: "Month after the quarter"}
GROUP_COLOURS = {
    "Output (real time)": "#1f77b4", "GDP": "#7f7f7f", "Retail": "#2ca02c", "Trade": "#9467bd",
    "Labour market": "#ff7f0e", "Prices": "#d62728", "Money and markets": "#8c564b",
}

st.set_page_config(page_title="UK GDP nowcast", layout="wide")


@st.cache_data
def load(name, **kw):
    return pd.read_csv(T / name, **kw)


live = load("live_nowcast.csv")
rmse = load("rmse_first_estimate.csv", index_col=0)
rmse.columns = rmse.columns.astype(int)
pi = load("predictions_with_intervals.csv")
dm = load("dm_vs_ar.csv")
sel = load("selection_frequency.csv")
contrib = load("live_contributions.csv")
rob = load("robustness.csv")
la = load("lookahead.csv")
rev = load("gdp_revisions.csv", index_col=0)
nq = load("n_quarters.csv", index_col=0).iloc[:, 0]

st.title("Nowcasting UK GDP with regularised regression")
st.caption("Pseudo-real-time backtest 2006-2026 against ONS first estimates. Real-time GDP and "
           "output data from the ONS real-time database and revisions triangles; other indicators "
           "from ONS and the Bank of England. Open Government Licence v3.0.")

tabs = st.tabs(["Live nowcast", "Backtest", "Accuracy", "What drives it", "Revisions", "Robustness"])

# ------------------------------------------------------------------------------------- live
with tabs[0]:
    st.subheader("Nowcasts made with data published by the end of October 2026")
    for q in sorted(live.quarter.unique()):
        d = live[live.quarter == q]
        h = int(d.h.iloc[0])
        c = d[d.model == "Combination"].iloc[0]
        b = d[d.model == "Bridge"].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{q}: combination nowcast", f"{c.nowcast:.2f}%",
                    help="Equal-weight average of the five bridge-anchored regularised models")
        col2.metric("80% interval", f"{c.lower80:.2f}% to {c.upper80:.2f}%",
                    help="From the model's own past out-of-sample errors (last 40 quarters)")
        col3.metric("Bridge equation", f"{b.nowcast:.2f}%")
        st.caption(f"Horizon: {H_LABEL[h]}."
                   + (" ONS publishes its first estimate for this quarter on 12 November 2026."
                      if q == "2026Q3" else ""))
        show = d[["model", "nowcast", "lower80", "upper80"]].set_index("model").round(2)
        with st.expander(f"All models for {q}"):
            st.dataframe(show, width="stretch")
        cc = contrib[(contrib.quarter == q) & (contrib.model == "Bridge + ridge")]
        if len(cc):
            fig = go.Figure(go.Bar(x=cc.contribution, y=cc.group, orientation="h",
                                   marker_color=[GROUP_COLOURS.get(g, "#999") for g in cc.group]))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                              title=f"{q}: what moves the Bridge + ridge nowcast away from its baseline (pp)")
            st.plotly_chart(fig, width="stretch")
        st.divider()

# ------------------------------------------------------------------------------------- backtest
with tabs[1]:
    c1, c2 = st.columns([1, 3])
    h = c1.selectbox("Horizon", list(H_LABEL), index=3, format_func=lambda k: H_LABEL[k])
    models = sorted(pi.model.unique())
    chosen = c1.multiselect("Models", models, default=["Bridge", "Combination", "AR"])
    show_band = c1.checkbox("80% interval for the first model", value=True)
    d = pi[pi.h == h].copy()
    d["date"] = pd.PeriodIndex(d.quarter, freq="Q").to_timestamp()
    fig = go.Figure()
    act = d.drop_duplicates("quarter").sort_values("date")
    fig.add_trace(go.Scatter(x=act.date, y=act.actual, name="ONS first estimate",
                             line=dict(color="black", width=3)))
    for i, m in enumerate(chosen):
        s = d[d.model == m].sort_values("date")
        if i == 0 and show_band:
            fig.add_trace(go.Scatter(x=pd.concat([s.date, s.date[::-1]]),
                                     y=pd.concat([s.upper, s.lower[::-1]]), fill="toself",
                                     fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                                     name=f"{m} 80% interval"))
        fig.add_trace(go.Scatter(x=s.date, y=s.nowcast, name=m))
    fig.update_layout(height=480, yaxis=dict(title="q/q growth (%)", range=[-3, 3]),
                      margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(fig, width="stretch")
    c2.caption("COVID quarters (2020-21) go off-scale. Zoom out with the toolbar to see them.")

# ------------------------------------------------------------------------------------- accuracy
with tabs[2]:
    st.subheader("Root mean squared error against the ONS first estimate (pp), excluding 2020Q1-2021Q2")
    st.dataframe(rmse.rename(columns=H_LABEL).style.format("{:.3f}")
                 .highlight_min(axis=0, color="#c7e9c0"), width="stretch")
    st.caption(f"Horizons -1 to 2: {nq.loc[2]} quarters from 2006Q1. Month after the quarter: "
               f"{nq.loc[3]} quarters, from 2018Q3 only (before mid-2018 ONS published its first "
               "estimate within a month of the quarter, before this origin).")
    hh = st.selectbox("Diebold-Mariano tests against the AR benchmark at horizon", list(H_LABEL),
                      index=3, format_func=lambda k: H_LABEL[k], key="dmh")
    st.dataframe(dm[dm.h == hh].drop(columns="h").sort_values("rel_rmse").set_index("model")
                 .round(3), width="stretch")
    st.caption("rel_rmse below 1 means more accurate than AR. Positive DM statistic favours the model.")

# ------------------------------------------------------------------------------------- drivers
with tabs[3]:
    c1, c2 = st.columns(2)
    m = c1.selectbox("Sparse model", sorted(sel.model.unique()))
    hh = c2.selectbox("Horizon ", sorted(sel.h.unique()), index=3, format_func=lambda k: H_LABEL[k])
    s = sel[(sel.model == m) & (sel.h == hh)].head(25)
    fig = go.Figure(go.Bar(x=s.frequency[::-1], y=(s.name + " (" + s.code + ")")[::-1],
                           orientation="h",
                           marker_color=[GROUP_COLOURS.get(g, "#999") for g in s.group[::-1]]))
    fig.update_layout(height=650, xaxis=dict(title="share of backtest origins selected", range=[0, 1]),
                      margin=dict(l=10, r=10, t=30, b=10),
                      title="Selection frequency across the backtest (top 25 indicators)")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------- revisions
with tabs[4]:
    st.subheader("ONS revises GDP growth substantially after the first estimate")
    r = rev.copy()
    r.index = pd.PeriodIndex(r.index, freq="Q").to_timestamp()
    r = r[~r.index.year.isin([2020]) & ~((r.index.year == 2021) & (r.index.month <= 4))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=r.first_estimate, name="First estimate"))
    fig.add_trace(go.Scatter(x=r.index, y=r.latest_estimate, name="Latest estimate"))
    fig.add_trace(go.Bar(x=r.index, y=r.revision, name="Revision", opacity=0.4))
    fig.update_layout(height=450, yaxis_title="q/q growth (%)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    rr = rev.loc["2006Q1":].drop(index=[f"{y}Q{k}" for y in (2020, 2021) for k in (1, 2, 3, 4)
                                       if f"{y}Q{k}" <= "2021Q2"], errors="ignore").revision
    st.write(f"Since 2006 (excluding COVID), the average revision is {rr.mean():+.2f} pp and the "
             f"root mean square revision is {np.sqrt((rr**2).mean()):.2f} pp. A nowcast judged "
             "against the first estimate is partly judged against noise that ONS later removes.")

# ------------------------------------------------------------------------------------- robustness
with tabs[5]:
    st.subheader("How results change under alternative designs (RMSE vs first estimate)")
    v = st.selectbox("Variant", sorted(rob.variant.unique()))
    d = rob[rob.variant == v].pivot(index="model", columns="h", values="rmse")
    base = rob[rob.variant == v].pivot(index="model", columns="h", values="main_rmse")
    st.dataframe(pd.concat({"variant": d, "main design": base}, axis=1).round(3),
                 width="stretch")
    st.markdown("""
* **lookahead_latest_vintage_output**: today's revised monthly GDP and sector output instead of
  real-time output, still judged against the first estimate. Errors rise at the end of the quarter:
  revised output data match the first estimate less well than the data ONS actually used.
* **early_indicators_only**: no official output or trade data; retail, labour market, prices,
  money and markets only.
* **covid_in_training**: 2020Q1-2021Q2 kept in the training data.
* **rolling_window_40q**: train on the latest 40 quarters only.
* **one_se_rule**: the largest penalty within one standard error of the best CV error (sparser,
  more heavily shrunk models).
""")
    st.subheader("The cost of look-ahead bias")
    st.write("RMSE (pp) of the same models under four designs. The conventional design uses today's "
             "revised data as inputs, training target and yardstick, as most published backtests do.")
    lm = st.selectbox("Model", ["Bridge", "AR", "Ridge", "Lasso", "Bridge + ridge"], key="lam")
    t = la[la.model == lm].pivot(index="design", columns="h", values="rmse").rename(columns=H_LABEL)
    st.dataframe(t.round(3), width="stretch")
    st.caption("In the month after the quarter, a conventional backtest reports roughly half the error "
               "a forecaster can actually achieve: revised monthly GDP nearly adds up to revised "
               "quarterly GDP. Revised output data also predict the *first* estimate worse than "
               "real-time data do, because ONS builds its first estimate from first-published data.")
