"""Build and execute the analysis notebooks in notebooks/.

    python scripts/build_notebooks.py            # all
    python scripts/build_notebooks.py 04 06      # selected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = code("""
import sys, warnings, pickle
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
plt.rcParams.update({"figure.figsize": (9, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
T = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
from nowcast.config import HORIZON_LABELS
""")

# ----------------------------------------------------------------------------- 01
N01 = [
    md("""
# 01. Data, revisions and the real-time problem

**Question for the project**: how accurately can UK quarterly GDP growth be estimated *before* the
Office for National Statistics (ONS) publishes its first estimate, and does regularised regression
on a wide panel of indicators help?

This notebook covers the data and the feature of macroeconomic data that drives every design choice:
**GDP figures are revised after publication**, and so are the indicators used to predict them.

| Source | What | Real-time history? |
|---|---|---|
| ONS real-time database (ABMI) | every published estimate of quarterly GDP since 1961 | yes, 746 vintages |
| ONS revisions triangles | monthly growth of the Index of Services, Production, Manufacturing and monthly GDP, as published each month | yes, from 1998 (IoP), 2001 (IoS), 2018 (monthly GDP) |
| ONS time series (mgdp, diop, drsi, mret, lms, mm23) | monthly GDP by sector, manufacturing detail, retail, trade, labour market, prices | latest vintage only |
| Bank of England database | Bank Rate, gilt yields, sterling, mortgage rates and approvals, M4 | latest vintage (rarely revised) |
"""),
    SETUP,
    code("""
from nowcast import data
from nowcast.config import INDICATORS
cat = pd.DataFrame([(i.code, i.name, i.group, i.transform, i.lag) for i in INDICATORS],
                   columns=["code", "name", "group", "transform", "publication lag (months)"])
print(cat.groupby("group").size().to_string())
cat.head(12)
"""),
    md("## GDP revisions: the target itself moves"),
    code("""
fl = pd.read_csv(T / "gdp_revisions.csv", index_col=0)
fl.index = pd.PeriodIndex(fl.index, freq="Q")
ev = fl.loc["2006Q1":"2026Q2"].drop(pd.period_range("2020Q1", "2021Q2", freq="Q"))
print(f"Quarters 2006Q1-2026Q2 excluding COVID: {len(ev)}")
print(f"mean revision (latest - first): {ev.revision.mean():+.3f} pp")
print(f"root mean square revision:      {np.sqrt((ev.revision**2).mean()):.3f} pp")
print(f"sd of first estimates:          {ev.first_estimate.std():.3f} pp")
print(f"corr(first, latest):            {ev.first_estimate.corr(ev.latest_estimate):.3f}")
display(Image(FIG / "gdp_revisions.png"))
"""),
    md("""
Revisions are of the same order as the quarter-to-quarter variation in the first estimates
themselves. Any nowcast judged against the first estimate is partly judged against noise that ONS
later removes, and any model trained on *today's* data has seen those later revisions.

## The publication calendar changed in 2018
Until mid-2018 ONS published a preliminary GDP estimate about 25 days after the quarter; since then
the first estimate comes about 40 days after, informed by three months of monthly GDP data.
"""),
    code("""
from nowcast.features import GDPVintages
gdp = GDPVintages(data.load_gdp_vintages())
lag = pd.Series({q: (gdp.first_vintage[q] - q.asfreq("M", "end")).n
                 for q in gdp.first.index if q >= pd.Period("1998Q1", "Q")})
lag.groupby(lag.index.year).agg(lambda s: s.mode()[0]).rename("months after quarter end").to_frame().T
"""),
    md("## Monthly output data are revised too\nThe ONS revisions triangles show each month's growth as first published and as it stands now."),
    code("""
rt = data.load_realtime()
rows = []
for sid, g in rt.groupby("series"):
    first = g.sort_values("vintage").groupby("month").value.first()
    last = g.sort_values("vintage").groupby("month").value.last()
    d = (last - first).loc["2006-01":"2025-12"]
    d = d[~((d.index >= pd.Period("2020-01", "M")) & (d.index <= pd.Period("2021-06", "M")))]
    rows.append(dict(series=sid, months=len(d), mean_abs_revision=d.abs().mean(),
                     rms_revision=np.sqrt((d**2).mean()), sd_first=first.loc[d.index].std()))
pd.DataFrame(rows).set_index("series").round(3)
"""),
    md("""
**Design consequences**
1. The target is the **first estimate** (what markets, the Bank and the Treasury react to on
   publication day); accuracy against the latest estimate is reported alongside.
2. Output indicators come from the **revisions triangles**: each month's figure as published at
   the time. Today's revised monthly GDP series is used only in an experiment that measures how much
   look-ahead bias it would introduce (notebook 06).
3. GDP lags used as predictors are taken from the vintage available at each nowcast date.
"""),
]

# ----------------------------------------------------------------------------- 02
N02 = [
    md("""
# 02. Features and the ragged edge

A nowcast for quarter Q at horizon h is made at the end of month T = first month of Q + h:

| h | origin | typical information |
|---|---|---|
| -1 | end of the month before the quarter | nothing about Q itself |
| 0 | end of month 1 | financial data for month 1; output data for the previous quarter |
| 1 | end of month 2 | retail and prices for month 1 |
| 2 | end of month 3 | output data for month 1 of Q |
| 3 | end of the month after Q | output data for months 1 and 2 of Q |

Each indicator is published with a different lag, so the most recent month available differs across
series: the **ragged edge**. Features are the six latest available monthly values of each indicator
(U-MIDAS: unrestricted mixed-data sampling), so one model per horizon has consistently aligned
features.
"""),
    SETUP,
    code("""
from nowcast.features import build_all, main_spec, feature_names, origin_month
from nowcast.config import INDICATORS
B = build_all([-1, 0, 1, 2, 3])
names = feature_names()
main = [n for n in names if main_spec(n)]
print(f"features in the main specification: {len(main)} "
      f"({len(main) - 2} indicator lags + 2 GDP lags); training samples 31-107 quarters")
"""),
    md("## Which months are known at each horizon? (Q3 2025 as an example)"),
    code("""
q = pd.Period("2025Q3", "Q")
rows = []
for ind in [i for i in INDICATORS if main_spec(f"{i.code}_l0")]:
    rows.append({"indicator": ind.name, "group": ind.group,
                 **{HORIZON_LABELS[h]: str(origin_month(q, h) - ind.lag) for h in [-1, 0, 1, 2, 3]}})
tab = pd.DataFrame(rows).drop_duplicates("group").set_index("indicator")
print("Latest month available for one indicator per group:")
tab
"""),
    md("## The live information set: data published by the end of October 2026"),
    code("""
live = {"2026Q3": 3, "2026Q4": 0}
for qs, h in live.items():
    x = B["X"][h].loc[qs, [c for c in main if c.endswith("_l0") or c.startswith("GDP_")]]
    print(f"{qs} at horizon {h} ({HORIZON_LABELS[h]}): missing latest values ->",
          [c for c in x.index if np.isnan(x[c])] or "none")
B["X"][3].loc["2026Q3", ["RT_IOS_l0", "RT_IOS_l1", "RT_IOP_l0", "RT_IOP_l1", "J5EK_l0", "GDP_q1"]].round(2)
"""),
    md("""
`RT_IOS_l0` at horizon 3 is August 2026 services growth as published in October 2026; `GDP_q1` is
Q2 2026 growth as known in the latest GDP vintage before the nowcast.

## Correlation structure
Is this a highly collinear panel? Check the pairwise correlations and the conditioning of the
standardised design (1998Q1-2019Q4, end-of-quarter horizon).
"""),
    code("""
from nowcast.models import Standardiser
X = B["X"][2][main].loc["1998Q1":"2019Q4"]
Z = Standardiser().fit(X.values).transform(X.values)
C = np.corrcoef(Z.T)
off = np.abs(C[np.triu_indices_from(C, 1)])
off = off[np.isfinite(off)]
print(f"pairs of features with |corr| > 0.5: {(off > 0.5).mean():.1%}; > 0.8: {(off > 0.8).mean():.1%}")
print(f"condition number of the standardised design: {np.linalg.cond(Z):.2e}")
print(f"rows (quarters) x columns (features): {Z.shape}")
"""),
    md("""
Pairwise correlation is modest: monthly growth rates are noisy, and successive lags of the same
indicator are only weakly correlated. The design is nonetheless singular, because there are
more features than quarters. OLS then has infinitely many exact-fit solutions. That, not
collinearity, is the main reason this problem needs regularisation (learn/01 and learn/02).
"""),
]

# ----------------------------------------------------------------------------- 03
N03 = [
    md("""
# 03. Inside one nowcast

Walk through every model at a single origin: the end-of-quarter nowcast for 2024Q2 (made with data
published by the end of June 2024), so each later backtest number has a concrete meaning.
"""),
    SETUP,
    code("""
from nowcast.features import build_all, main_spec, feature_names, feature_groups, origin_month, training_mask
from nowcast.models import Standardiser, time_folds, all_models, _group_lasso_path, group_lambda_max
B = build_all([2])
q, h = pd.Period("2024Q2", "Q"), 2
T = origin_month(q, h)
names = [n for n in feature_names() if main_spec(n)]
X = B["X"][h][names]
mask = training_mask(X.index, B["gdp"], T)
tr = X.index[mask]
y = B["first"].reindex(tr).values
st = Standardiser().fit(X.loc[tr].values)
Z, z = st.transform(X.loc[tr].values), st.transform(X.loc[[q]].values)
groups = feature_groups()[[feature_names().index(n) for n in names]]
folds = time_folds(len(y))
print(f"origin {T}: {len(y)} training quarters ({tr[0]}-{tr[-1]}), {Z.shape[1]} features, {len(folds)} CV folds")
"""),
    code("""
rows = []
for m in all_models():
    m.fit(Z, y, folds, groups=groups, names=names)
    rows.append(dict(model=m.name, nowcast=float(m.predict(z)[0]),
                     non_zero=int((np.abs(m.coef_) > 1e-10).sum()),
                     penalty=getattr(m, "alpha_", np.nan)))
res = pd.DataFrame(rows).set_index("model")
print(f"ONS first estimate for {q}: {B['first'][q]:.2f}%")
res.round(3)
"""),
    md("## Lasso path and cross-validation curve at this origin"),
    code("""
from sklearn.linear_model import LassoCV, lasso_path
m = LassoCV(cv=folds, alphas=60, max_iter=20000).fit(Z, y)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
alphas, coefs, _ = lasso_path(Z, y, alphas=m.alphas_)
ax[0].semilogx(alphas, coefs.T, lw=0.8); ax[0].axvline(m.alpha_, c="k", ls="--")
ax[0].invert_xaxis(); ax[0].set(xlabel="lambda", ylabel="coefficient", title="Lasso paths (236 features)")
mean, se = m.mse_path_.mean(1), m.mse_path_.std(1) / np.sqrt(m.mse_path_.shape[1])
ax[1].semilogx(m.alphas_, mean); ax[1].fill_between(m.alphas_, mean - se, mean + se, alpha=0.2)
ax[1].axvline(m.alpha_, c="k", ls="--"); ax[1].invert_xaxis()
ax[1].set(xlabel="lambda", ylabel="CV MSE", title="Blocked time-series CV");
"""),
    md("""
The cross-validation curve is flat and noisy, with wide error bands from only five validation blocks.
That is typical with 31-107 quarterly observations, and it is one reason data-driven selection
struggles in this problem (see notebook 04).

## Group lasso: indicators enter as blocks of six lags
"""),
    code("""
ym = y.mean(); zm = Z.mean(0)
lmax = group_lambda_max(Z, y - ym, groups)
lams = lmax * np.logspace(0, -2.5, 30)
path = _group_lasso_path(Z - zm, y - ym, groups, lams)
gnames = {g: n.rsplit("_l", 1)[0] for g, n in zip(groups, names)}
norms = pd.DataFrame([[np.linalg.norm(b[groups == g]) for g in np.unique(groups)] for b in path],
                     index=lams, columns=[gnames[g] for g in np.unique(groups)])
first_in = norms.gt(0).idxmax().where(norms.gt(0).any()).sort_values(ascending=False)
print("first indicators to enter as lambda falls:", list(first_in.index[:8]))
norms[first_in.index[:8]].plot(logx=True, figsize=(9, 4)).invert_xaxis()
plt.ylabel("group norm"); plt.xlabel("lambda");
"""),
    md("""
## Bridge-anchored (partially penalised) models
The bridge equation uses the latest three months of real-time services and production output plus
two GDP lags. The anchored models leave those eight coefficients **unpenalised** and regularise only
the other indicators. By the Frisch-Waugh-Lovell theorem this equals residualising `y` and the other
indicators on the bridge regressors, then running the penalised model. `tests/test_models.py`
checks the equivalence against the direct solution.
"""),
    code("""
from nowcast.models import BridgeAnchored, RidgeModel, SubsetOLS, BRIDGE_PREFIXES
br = SubsetOLS("Bridge", BRIDGE_PREFIXES).fit(Z, y, folds, names=names)
an = BridgeAnchored(RidgeModel()).fit(Z, y, folds, groups=groups, names=names)
core = [n for n in names if n.startswith(BRIDGE_PREFIXES)]
idx = [names.index(n) for n in core]
pd.DataFrame({"bridge OLS": br.coef_[idx], "bridge + ridge": an.coef_[idx]}, index=core).round(3)
"""),
]

# ----------------------------------------------------------------------------- 04
N04 = [
    md("""
# 04. Backtest results

Pseudo-real-time backtest: 76 target quarters from 2006Q1 to 2026Q2 (excluding 2020Q1-2021Q2) at
horizons -1 to 2, and 27 quarters from 2018Q3 at horizon 3 (before mid-2018 the first estimate was
published within a month of the quarter, so there was nothing left to nowcast). Every model is
re-tuned and re-estimated at each of the 361 origins using only data published by then.
"""),
    SETUP,
    code("""
rmse = pd.read_csv(T / "rmse_first_estimate.csv", index_col=0); rmse.columns = rmse.columns.astype(int)
rmse.rename(columns=HORIZON_LABELS).style.format("{:.3f}").highlight_min(axis=0, color="#c7e9c0")
"""),
    code("display(Image(FIG / 'rmse_by_horizon.png'))"),
    md("## Significance: Diebold-Mariano tests (Harvey-Leybourne-Newbold correction)"),
    code("""
dm = pd.read_csv(T / "dm_vs_ar.csv")
dmb = pd.read_csv(T / "dm_vs_bridge.csv")
show = lambda d: d.pivot(index="model", columns="h", values="rel_rmse").round(3).astype(str) + \\
    d.pivot(index="model", columns="h", values="p_value").map(
        lambda p: "" if pd.isna(p) else ("**" if p < 0.05 else "*" if p < 0.1 else ""))
print("Relative RMSE vs AR (below 1 = better). ** p<0.05, * p<0.1")
display(show(dm).rename(columns=HORIZON_LABELS))
print("Relative RMSE vs the bridge equation")
display(show(dmb).rename(columns=HORIZON_LABELS))
"""),
    code("""
best = rmse.idxmin()
for h in rmse.columns:
    print(f"{HORIZON_LABELS[h]:>24}: best {best[h]:<22} {rmse.loc[best[h], h]:.3f}"
          f"   bridge {rmse.loc['Bridge', h]:.3f}   AR {rmse.loc['AR', h]:.3f}   mean {rmse.loc['Mean', h]:.3f}")
"""),
    md("## By period"),
    code("""
for h in [0, 2, 3]:
    t = pd.read_csv(T / f"rmse_by_period_h{h}.csv", index_col=0)
    print(HORIZON_LABELS[h]); display(t.round(3))
"""),
    md("## Nowcasts over time (end of quarter)"),
    code("display(Image(FIG / 'nowcast_vs_first_h2.png'))"),
    md("## Against the latest estimate"),
    code("""
rl = pd.read_csv(T / "rmse_latest_estimate.csv", index_col=0); rl.columns = rl.columns.astype(int)
rl.rename(columns=HORIZON_LABELS).style.format("{:.3f}").highlight_min(axis=0, color="#c7e9c0")
"""),
    md("## Interval calibration\nEmpirical 80% intervals from each model's own past errors (last 40 quarters, at least 12)."),
    code("""
cov = pd.read_csv(T / "interval_coverage_80.csv", index_col=0); cov.columns = cov.columns.astype(int)
cov.rename(columns=HORIZON_LABELS).round(2)
"""),
    md("""
## Including COVID
2020Q2 GDP fell by about 19%. Including 2020Q1-2021Q2 in the evaluation, RMSEs are dominated by six
quarters; the table is shown for completeness.
"""),
    code("""
rc = pd.read_csv(T / "rmse_first_estimate_incl_covid.csv", index_col=0); rc.columns = rc.columns.astype(int)
rc.rename(columns=HORIZON_LABELS).round(2)
"""),
]

# ----------------------------------------------------------------------------- 05
N05 = [
    md("""
# 05. What the models select, and why

Sparse models produce a list of selected indicators at every origin. Because the list changes with the
sample (learn/08), we report **selection frequencies**: the share of the 82 backtest origins (33 at
horizon 3) in which each indicator has at least one non-zero lag. The origins include the six COVID
quarters, which are nowcast (from pre-COVID training data) but excluded from the accuracy tables.
"""),
    SETUP,
    code("""
sel = pd.read_csv(T / "selection_frequency.csv")
for m in ["Lasso", "Group lasso"]:
    s = sel[(sel.model == m) & (sel.h == 2)].head(12)
    print(f"{m}, end-of-quarter nowcasts (n origins = {s.n_origins.iloc[0]})")
    display(s[["code", "name", "group", "frequency"]].round(2).set_index("code"))
"""),
    code("display(Image(FIG / 'selection_by_group_h2.png'))"),
    code("""
s = sel[sel.model == "Lasso"]
stab = s.groupby("h").frequency.agg(always=lambda f: (f >= 0.9).sum(), sometimes=lambda f: ((f > 0.1) & (f < 0.9)).sum(),
                                     rarely=lambda f: (f <= 0.1).sum())
print("Lasso: number of indicators selected in >=90%, 10-90%, <=10% of origins")
stab.rename(index=HORIZON_LABELS)
"""),
    md("""
## Contributions to the live nowcast
For a linear model on standardised features, the nowcast is the intercept plus the sum of
coefficient x feature. Summing within indicator groups shows which data move the nowcast away from
the model's baseline.
"""),
    code("""
c = pd.read_csv(T / "live_contributions.csv")
for q in sorted(c.quarter.unique()):
    t = c[(c.quarter == q) & c.model.isin(["Bridge", "Ridge", "Group lasso", "Bridge + ridge", "Bridge + group lasso"])]
    print(q); display(t.pivot(index="group", columns="model", values="contribution").round(3))
"""),
]

# ----------------------------------------------------------------------------- 06
N06 = [
    md("""
# 06. Robustness and the cost of look-ahead bias

Six alternative designs, each re-running the full 361-origin backtest:

| Variant | Change |
|---|---|
| `early_indicators_only` | no official output or trade data: retail, labour market, prices, money and markets only |
| `covid_in_training` | 2020Q1-2021Q2 kept in the training data |
| `rolling_window_40q` | train on the latest 40 quarters only |
| `one_se_rule` | penalties chosen by the one-standard-error rule instead of minimum CV error |
| `lookahead_latest_vintage_output` | today's revised monthly GDP and sector output instead of real-time output; still trained and judged on first estimates |
| `conventional_latest_vintage` | the usual published design: today's revised data as features, training target **and** yardstick |
"""),
    SETUP,
    code("""
rob = pd.read_csv(T / "robustness.csv")
order = ["early_indicators_only", "covid_in_training", "rolling_window_40q", "one_se_rule"]
for v in order:
    d = rob[rob.variant == v]
    t = pd.concat({"variant": d.pivot(index="model", columns="h", values="rmse"),
                   "main": d.pivot(index="model", columns="h", values="main_rmse")}, axis=1)
    print(v); display(t.round(3))
"""),
    md("""
**Reading the variants** (RMSE against the first estimate):
* **Early indicators only.** Without output or trade data, ridge's error falls only from 0.55 to
  0.47 between the start and the end of the quarter, against 0.59 to 0.36 for the bridge. The
  late-quarter gains in the main design come from official output data.
* **COVID in training.** It cuts both ways. Models that lean on retail, labour-market, price
  and financial indicators get much worse early in the quarter: six extreme quarters dominate their
  fitted coefficients. The bridge-type models get *better* once output data arrive, because
  2020's huge output swings pin down the mapping from monthly output to GDP. The main design
  excludes COVID from training. That is the cautious choice for most models, and it slightly
  understates what the bridge could achieve after 2021.
* **Rolling 40-quarter window.** AR and the pure regularised models gain a little (up to 9%) from
  discarding old data. The bridge and the anchored models lose heavily, because their output
  coefficients are pinned down by the large swings of 2008-09, which a 10-year window eventually
  drops. The bridge is still the most accurate model at the end of the quarter.
* **One-standard-error rule.** Heavier penalties make the bridge-anchored models 6-17% more
  accurate, which brings them close to the plain bridge. The less weight the extra indicators get,
  the better. The pure regularised models mostly get worse. The conclusions do not depend on the
  tuning rule.

## Look-ahead bias in numbers
"""),
    code("""
la = pd.read_csv(T / "lookahead.csv")
t = la.pivot_table(index=["model", "design"], columns="h", values="rmse").rename(columns=HORIZON_LABELS)
t.loc[["AR", "Bridge", "Ridge"]].round(3)
"""),
    code("""
def get(design, model, h):
    d = la[(la.design.str.startswith(design)) & (la.model == model) & (la.h == h)]
    return float(d.rmse.iloc[0])

rt, cv = "Real time, judged against first", "Conventional"
for h in [2, 3]:
    b_rt, a_rt = get(rt, "Bridge", h), get(rt, "AR", h)
    b_cv, a_cv = get(cv, "Bridge", h), get(cv, "AR", h)
    print(f"{HORIZON_LABELS[h]:<20} bridge RMSE: real time {b_rt:.3f} | conventional backtest {b_cv:.3f}"
          f"   ->  bridge vs AR: real time {1 - b_rt / a_rt:+.0%} | conventional {1 - b_cv / a_cv:+.0%}")
"""),
    md("""
Two findings:

1. **A conventional backtest flatters the late nowcast.** In the month after the quarter, today's
   revised monthly GDP for the quarter's first two months nearly adds up to today's quarterly GDP.
   A backtest built on revised data then reports an error about half the size that a forecaster could
   actually achieve, and a far larger advantage over the AR benchmark. Published nowcasting results
   that use latest-vintage data should be read with this in mind.
2. **Revised data are the wrong input for predicting the first estimate.** Swapping in revised
   output while still targeting the first estimate makes nowcasts *worse* at the end of the quarter.
   ONS builds its first estimate from output data as first published, not as later revised, so the
   real-time output series are the right inputs for the number markets react to.

Before any output data for the quarter exist (before the quarter, end of month 1), the choice of
vintage matters little. Look-ahead bias is concentrated where the hard data are.
"""),
]

# ----------------------------------------------------------------------------- 07
N07 = [
    md("""
# 07. Live nowcast: Q3 and Q4 2026

Made with data published by the end of October 2026, before ONS's first estimate for Q3 2026
(due 12 November 2026). Intervals are empirical 80% intervals from each model's past
out-of-sample errors at the same horizon.
"""),
    SETUP,
    code("""
live = pd.read_csv(T / "live_nowcast.csv")
for q in sorted(live.quarter.unique()):
    d = live[live.quarter == q].set_index("model")[["h", "nowcast", "lower80", "upper80"]]
    print(f"{q} (horizon {d.h.iloc[0]}: {HORIZON_LABELS[int(d.h.iloc[0])]})")
    display(d.drop(columns="h").round(2))
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 5))
d = live[(live.quarter == "2026Q3")].set_index("model").sort_values("nowcast")
ax.errorbar(d.nowcast, range(len(d)), xerr=[d.nowcast - d.lower80, d.upper80 - d.nowcast],
            fmt="o", capsize=3)
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index)
ax.axvline(0, c="k", lw=0.8); ax.set_xlabel("q/q GDP growth, % (80% interval)")
ax.set_title("2026Q3 nowcasts, data to end-October 2026");
"""),
    md("""
**After 12 November 2026**: rerun `python scripts/run_pipeline.py --data` to download the new GDP
vintage; the 2026Q3 first estimate then appears in `data/processed/gdp_first_latest.csv` and can be
compared with the nowcast recorded here (the nowcast files are versioned in git).
"""),
]

NOTEBOOKS = {
    "01_data_and_revisions.ipynb": N01, "02_features_and_ragged_edge.ipynb": N02,
    "03_inside_one_nowcast.ipynb": N03, "04_backtest_results.ipynb": N04,
    "05_selection_and_drivers.ipynb": N05, "06_robustness_and_lookahead.ipynb": N06,
    "07_live_nowcast.ipynb": N07,
}

if __name__ == "__main__":
    only = sys.argv[1:]
    for fname, cells in NOTEBOOKS.items():
        if only and not any(fname.startswith(o) for o in only):
            continue
        write_and_run(cells, NB / fname)
