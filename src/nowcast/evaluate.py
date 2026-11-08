"""Forecast evaluation: accuracy, significance, intervals and variable-selection stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import HORIZON_LABELS, INDICATORS, N_MONTH_LAGS
from .features import feature_names
from .models import COMBINATION_MEMBERS

COVID_EVAL = [str(q) for q in pd.period_range("2020Q1", "2021Q2", freq="Q")]
PERIODS = {
    "2006-2009 (financial crisis)": ("2006Q1", "2009Q4"),
    "2010-2019": ("2010Q1", "2019Q4"),
    "2021Q3-2026 (post-COVID)": ("2021Q3", "2026Q4"),
}
MODEL_ORDER = ["Mean", "AR", "Bridge", "OLS (all)", "Ridge", "Lasso", "Elastic net",
               "Adaptive lasso", "Group lasso", "Factor", "Bridge + ridge", "Bridge + lasso",
               "Bridge + elastic net", "Bridge + group lasso", "Bridge + factors", "Combination"]


def add_combination(preds: pd.DataFrame) -> pd.DataFrame:
    m = preds[preds.model.isin(COMBINATION_MEMBERS)]
    if set(m.model.unique()) != set(COMBINATION_MEMBERS):   # only with every member present
        return preds
    keys = ["quarter", "h", "origin"]
    comb = m.groupby(keys, as_index=False).agg(
        nowcast=("nowcast", "mean"), n_train=("n_train", "first"),
        first_estimate=("first_estimate", "first"), latest_estimate=("latest_estimate", "first"))
    comb["model"] = "Combination"
    return pd.concat([preds, comb], ignore_index=True)


def prepare(preds: pd.DataFrame, target: str = "first_estimate") -> pd.DataFrame:
    p = add_combination(preds.copy()) if "Combination" not in set(preds.model) else preds.copy()
    p["actual"] = p[target]
    p["error"] = p["actual"] - p["nowcast"]
    p["covid"] = p["quarter"].isin(COVID_EVAL)
    p["q"] = pd.PeriodIndex(p["quarter"], freq="Q")
    return p


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2)))


def rmse_table(p: pd.DataFrame, exclude_covid=True, common_sample=True) -> pd.DataFrame:
    d = p[~p.covid] if exclude_covid else p
    if common_sample:  # each horizon evaluated on the quarters every model has
        n_models = d.groupby("h").model.nunique()
        cnt = d.groupby(["h", "quarter"]).model.nunique()
        keep = cnt[cnt.values == n_models.reindex(cnt.index.get_level_values(0)).values].index
        d = d.set_index(["h", "quarter"]).loc[keep].reset_index()
    t = d.groupby(["model", "h"]).error.apply(rmse).unstack("h")
    order = [m for m in MODEL_ORDER if m in t.index] + [m for m in t.index if m not in MODEL_ORDER]
    return t.loc[order]


def n_quarters(p: pd.DataFrame, exclude_covid=True) -> pd.Series:
    d = p[~p.covid] if exclude_covid else p
    return d[d.model == "AR"].groupby("h").quarter.nunique()


def diebold_mariano(e1, e2, h_lag: int = 0):
    """DM test of equal squared-error loss with the Harvey-Leybourne-Newbold correction.

    Each nowcast targets one quarter and is evaluated once, so these are one-step
    errors: the default uses no autocovariance terms (h_lag = forecast horizon - 1 = 0).
    If h_lag > 0, autocovariances get Bartlett (Newey-West) weights, which keeps the
    variance positive; unweighted terms understate it when loss differences are
    negatively autocorrelated.

    Returns (statistic, two-sided p-value). Positive statistic: model 2 more accurate.
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h_lag + 1)]
    weights = [1 - k / (h_lag + 1) for k in range(1, h_lag + 1)]
    var = (gamma[0] + 2 * sum(w * g for w, g in zip(weights, gamma[1:]))) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hh = h_lag + 1
    corr = np.sqrt((n + 1 - 2 * hh + hh * (hh - 1) / n) / n)
    stat = dm * corr
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def dm_table(p: pd.DataFrame, benchmark="AR", exclude_covid=True) -> pd.DataFrame:
    d = p[~p.covid] if exclude_covid else p
    rows = []
    for (m, h), g in d.groupby(["model", "h"]):
        if m == benchmark:
            continue
        b = d[(d.model == benchmark) & (d.h == h)].set_index("quarter").error
        g = g.set_index("quarter").error
        idx = g.index.intersection(b.index)
        s, pv = diebold_mariano(b.loc[idx].values, g.loc[idx].values)
        rows.append(dict(model=m, h=h, rel_rmse=rmse(g.loc[idx]) / rmse(b.loc[idx]),
                         dm_stat=s, p_value=pv, n=len(idx)))
    return pd.DataFrame(rows)


def period_table(p: pd.DataFrame, h: int) -> pd.DataFrame:
    d = p[(p.h == h) & ~p.covid]
    out = {}
    for lab, (a, b) in PERIODS.items():
        dd = d[(d.q >= pd.Period(a, "Q")) & (d.q <= pd.Period(b, "Q"))]
        out[lab] = dd.groupby("model").error.apply(rmse)
    t = pd.DataFrame(out)
    return t.loc[[m for m in MODEL_ORDER if m in t.index]]


def add_intervals(p: pd.DataFrame, coverage=0.8, window=40, min_obs=12) -> pd.DataFrame:
    """Empirical intervals from each model's own past out-of-sample errors.

    For quarter q the interval uses errors of quarters at least two quarters
    earlier (their first estimates are certainly published), excluding COVID.
    """
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    out = []
    for (m, h), g in p.groupby(["model", "h"]):
        g = g.sort_values("q").copy()
        lo, hi = [], []
        for q in g.q:
            past = g[(g.q <= q - 2) & ~g.covid].error.tail(window)
            if len(past) >= min_obs:
                lo.append(np.quantile(past, lo_q))
                hi.append(np.quantile(past, hi_q))
            else:
                lo.append(np.nan)
                hi.append(np.nan)
        g["lower"] = g.nowcast + np.array(lo)
        g["upper"] = g.nowcast + np.array(hi)
        out.append(g)
    r = pd.concat(out)
    r["covered"] = np.where(r.lower.notna(), (r.actual >= r.lower) & (r.actual <= r.upper), np.nan)
    return r


def coverage_table(pi: pd.DataFrame) -> pd.DataFrame:
    d = pi[~pi.covid & pi.lower.notna()]
    t = d.groupby(["model", "h"]).covered.mean().unstack("h")
    return t.loc[[m for m in MODEL_ORDER if m in t.index]]


def selection_frequency(coefs, model="Lasso", h=2) -> pd.DataFrame:
    """Share of origins in which each indicator (any of its lags) has a non-zero coefficient."""
    names = feature_names()
    n_ind = len(INDICATORS)
    rows = []
    for q, hh, c in coefs:
        if hh != h or model not in c:
            continue
        b = np.abs(c[model]) > 1e-10
        sel = b[: n_ind * N_MONTH_LAGS].reshape(n_ind, N_MONTH_LAGS).any(axis=1)
        rows.append(np.concatenate([sel, b[-2:]]))
    if not rows:
        return pd.DataFrame()
    freq = np.mean(rows, axis=0)
    labels = [(i.code, i.name, i.group) for i in INDICATORS] + \
             [("GDP_q2", "GDP growth, quarter t-2", "GDP"), ("GDP_q1", "GDP growth, quarter t-1", "GDP")]
    df = pd.DataFrame(labels, columns=["code", "name", "group"])
    df["frequency"] = freq
    df["n_origins"] = len(rows)
    return df.sort_values("frequency", ascending=False).reset_index(drop=True)


def horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=HORIZON_LABELS)
