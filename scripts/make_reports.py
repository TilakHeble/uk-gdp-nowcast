"""Turn backtest outputs into the tables and figures used by the docs and dashboard.

    python scripts/make_reports.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nowcast import evaluate as E  # noqa: E402
from nowcast.config import FIGURES, HORIZON_LABELS, HORIZONS, OUTPUTS, PROCESSED  # noqa: E402
from nowcast.explain import contributions  # noqa: E402
from nowcast.features import build_all  # noqa: E402

TABLES = OUTPUTS / "tables"
SEL_MODELS = ["Lasso", "Group lasso", "Bridge + lasso", "Bridge + group lasso"]
KEY_MODELS = ["AR", "Bridge", "Ridge", "Group lasso", "Bridge + ridge",
              "Bridge + group lasso", "Combination"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUTPUTS / "backtest_predictions.csv")
    p = E.prepare(raw)
    pl = E.prepare(raw, target="latest_estimate")

    # ---------------------------------------------------------------- accuracy
    E.rmse_table(p).to_csv(TABLES / "rmse_first_estimate.csv")
    E.rmse_table(p, exclude_covid=False).to_csv(TABLES / "rmse_first_estimate_incl_covid.csv")
    E.rmse_table(pl).to_csv(TABLES / "rmse_latest_estimate.csv")
    E.n_quarters(p).to_csv(TABLES / "n_quarters.csv")
    E.dm_table(p).to_csv(TABLES / "dm_vs_ar.csv", index=False)
    E.dm_table(p, benchmark="Bridge").to_csv(TABLES / "dm_vs_bridge.csv", index=False)
    E.dm_table(pl).to_csv(TABLES / "dm_vs_ar_latest.csv", index=False)
    for h in HORIZONS:
        E.period_table(p, h).to_csv(TABLES / f"rmse_by_period_h{h}.csv")

    # ---------------------------------------------------------------- intervals
    pi = E.add_intervals(p)
    pi.drop(columns=["q"]).to_csv(TABLES / "predictions_with_intervals.csv", index=False)
    E.coverage_table(pi).to_csv(TABLES / "interval_coverage_80.csv")

    # ---------------------------------------------------------------- selection
    with open(OUTPUTS / "backtest_coefficients.pkl", "rb") as f:
        coefs = pickle.load(f)
    sel = []
    for m in SEL_MODELS:
        for h in HORIZONS:
            s = E.selection_frequency(coefs, model=m, h=h)
            if len(s):
                s.insert(0, "h", h)
                s.insert(0, "model", m)
                sel.append(s)
    sel = pd.concat(sel)
    sel.to_csv(TABLES / "selection_frequency.csv", index=False)

    # ---------------------------------------------------------------- live nowcast
    live = pd.read_csv(OUTPUTS / "live_nowcast.csv")
    lc = live[live.model.isin(E.COMBINATION_MEMBERS)].groupby(["quarter", "h", "origin"],
                                                              as_index=False).nowcast.mean()
    lc["model"] = "Combination"
    live = pd.concat([live, lc], ignore_index=True)
    rows = []
    for _, r in live.iterrows():
        past = pi[(pi.model == r.model) & (pi.h == r.h) & ~pi.covid].sort_values("q")
        past = past[past.q <= pd.Period(r.quarter, "Q") - 2].error.tail(40)
        lo, hi = (np.quantile(past, [0.1, 0.9]) if len(past) >= 12 else (np.nan, np.nan))
        rows.append(dict(r, lower80=r.nowcast + lo, upper80=r.nowcast + hi, n_errors=len(past)))
    live = pd.DataFrame(rows)
    live.to_csv(TABLES / "live_nowcast.csv", index=False)

    B = build_all(HORIZONS)
    with open(OUTPUTS / "live_coefficients.pkl", "rb") as f:
        live_coefs = pickle.load(f)
    contrib = []
    for q, h, cdict in live_coefs:
        for m, coef in cdict.items():
            c = contributions(B, pd.Period(q, "Q"), h, coef)
            for g, v in c.items():
                contrib.append(dict(quarter=q, h=h, model=m, group=g, contribution=v))
    pd.DataFrame(contrib).to_csv(TABLES / "live_contributions.csv", index=False)

    # ---------------------------------------------------------------- robustness
    rob = []
    main_rmse = E.rmse_table(p, common_sample=False)
    for f in sorted(OUTPUTS.glob("robust_*.csv")):
        if "conventional" in f.stem:      # judged against the latest estimate: see lookahead.csv
            continue
        r = E.prepare(pd.read_csv(f))
        t = E.rmse_table(r, common_sample=False)
        for m in t.index:
            for h in t.columns:
                rob.append(dict(variant=f.stem.replace("robust_", ""), model=m, h=h, rmse=t.loc[m, h],
                                main_rmse=main_rmse.loc[m, h] if m in main_rmse.index else np.nan))
    pd.DataFrame(rob).to_csv(TABLES / "robustness.csv", index=False)

    # ---------------------------------------------------------------- look-ahead flattery
    # Honest real-time result vs what a conventional backtest on today's data would report.
    la = {"Real time, judged against first estimate (this project)": E.rmse_table(p, common_sample=False),
          "Real time, judged against latest estimate": E.rmse_table(pl, common_sample=False)}
    f1 = OUTPUTS / "robust_lookahead_latest_vintage_output.csv"
    f2 = OUTPUTS / "robust_conventional_latest_vintage.csv"
    if f1.exists():
        la["Revised output data, judged against first estimate"] = E.rmse_table(
            E.prepare(pd.read_csv(f1)), common_sample=False)
    if f2.exists():
        la["Conventional: revised data, trained and judged on latest estimate"] = E.rmse_table(
            E.prepare(pd.read_csv(f2), target="latest_estimate"), common_sample=False)
    rows = []
    for design, t in la.items():
        for m in ["AR", "Bridge", "Ridge", "Lasso", "Bridge + ridge"]:
            if m in t.index:
                for h in t.columns:
                    rows.append(dict(design=design, model=m, h=h, rmse=t.loc[m, h]))
    pd.DataFrame(rows).to_csv(TABLES / "lookahead.csv", index=False)

    # ---------------------------------------------------------------- revisions
    fl = pd.read_csv(PROCESSED / "gdp_first_latest.csv", index_col=0)
    fl = fl.loc["1998Q1":]
    fl.to_csv(TABLES / "gdp_revisions.csv")

    figures(p, pi, pl, sel, live)
    print("reports written to", TABLES)


def figures(p, pi, pl, sel, live):
    # RMSE by horizon
    t = E.rmse_table(p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in KEY_MODELS:
        ax.plot([HORIZON_LABELS[h] for h in t.columns], t.loc[m], "o-", label=m,
                lw=2.5 if m in ("Bridge", "Combination") else 1.3)
    ax.set_ylabel("RMSE vs first estimate (pp)")
    ax.set_title("Nowcast error falls as the quarter's data arrive")
    ax.legend(fontsize=8, ncol=2)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "rmse_by_horizon.png")
    plt.close(fig)

    # nowcasts vs first estimate, h=2
    d = pi[(pi.h == 2) & pi.model.isin(["AR", "Bridge", "Combination"])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    act = d[d.model == "AR"].set_index("q").actual.sort_index()
    ax.plot(act.index.to_timestamp(), act.values, "k-", lw=2, label="ONS first estimate")
    for m in ["AR", "Bridge", "Combination"]:
        s = d[d.model == m].set_index("q").nowcast.sort_index()
        ax.plot(s.index.to_timestamp(), s.values, lw=1.2, label=m)
    ax.set_ylim(-3, 3)
    ax.set_ylabel("q/q GDP growth (%)")
    ax.set_title("Nowcasts at the end of the quarter vs ONS first estimate (COVID quarters off-scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "nowcast_vs_first_h2.png")
    plt.close(fig)

    # revisions
    fl = pd.read_csv(PROCESSED / "gdp_first_latest.csv", index_col=0).loc["1998Q1":]
    fl.index = pd.PeriodIndex(fl.index, freq="Q")
    fl = fl.drop(pd.period_range("2020Q1", "2021Q2", freq="Q"), errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fl.index.to_timestamp(), fl.revision, width=60, color="tab:blue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("latest minus first estimate (pp)")
    ax.set_title("How much ONS has revised each quarter's GDP growth since its first estimate")
    fig.tight_layout()
    fig.savefig(FIGURES / "gdp_revisions.png")
    plt.close(fig)

    # selection by group
    s = sel[(sel.h == 2)]
    g = s.groupby(["model", "group"]).frequency.mean().unstack("model")
    g = g[g.sum(axis=1) > 0]          # latest-vintage groups are not offered to the main models
    fig, ax = plt.subplots(figsize=(9, 4.5))
    g.plot.barh(ax=ax)
    ax.set_xlabel("average share of origins in which an indicator in the group is selected")
    ax.set_title("What the sparse models select (end-of-quarter nowcasts)\n"
                 "Bridge models keep services, production and GDP lags unpenalised, so always in",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "selection_by_group_h2.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
