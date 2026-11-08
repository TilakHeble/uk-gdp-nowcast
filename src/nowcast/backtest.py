"""Pseudo-real-time backtest and live nowcast.

For every target quarter Q in the evaluation window and every horizon h:

1. origin T = first month of Q + h;
2. skip if ONS had already published Q's first estimate by T (happens at h=3
   before mid-2018, when first estimates came out one month after the quarter);
3. training rows = quarters whose first estimate was published by T (COVID
   quarters 2020Q1-2021Q2 excluded from training by default);
4. standardise on the training rows, tune each model on blocked time-series
   folds, fit, and nowcast Q.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import HORIZONS, OOS_START
from .features import (build_all, feature_groups, feature_names, main_spec, origin_month,
                       training_mask)
from .models import Standardiser, all_models, time_folds


def _target_series(B, target: str, T: pd.Period, quarters):
    """Training target values for rows in ``quarters`` as a forecaster at T would use them."""
    if target == "first":
        return B["first"].reindex(quarters).values
    if target == "realtime":  # latest estimate available at T
        return B["gdp"].known_at(T).reindex(quarters).values
    if target == "latest":    # today's vintage: only for the conventional (look-ahead) backtest
        return B["latest"].reindex(quarters).values
    raise ValueError(target)


def run_origin(B, q, h, models=None, target="first", exclude_covid=True,
               window=None, feature_filter=main_spec):
    X = B["X"][h]
    T = origin_month(q, h)
    names = feature_names()
    groups = feature_groups()
    cols = np.arange(len(names))
    if feature_filter is not None:
        cols = np.array([j for j, n in enumerate(names) if feature_filter(n)])
    names_s = [names[j] for j in cols]
    groups_s = groups[cols]

    mask = training_mask(X.index, B["gdp"], T, exclude_covid=exclude_covid)
    tr_q = X.index[mask]
    if window is not None:
        tr_q = tr_q[-window:]
    y = _target_series(B, target, T, tr_q)
    ok = np.isfinite(y)
    tr_q, y = tr_q[ok], y[ok]
    Xtr = X.loc[tr_q].values[:, cols]
    xte = X.loc[[q]].values[:, cols]
    st = Standardiser().fit(Xtr)
    Ztr, zte = st.transform(Xtr), st.transform(xte)
    folds = time_folds(len(y))

    out, coefs = [], {}
    for m in (models or all_models()):
        m.fit(Ztr, y, folds, groups=groups_s, names=names_s)
        pred = float(m.predict(zte)[0])
        rec = dict(quarter=str(q), h=h, origin=str(T), model=m.name, nowcast=pred,
                   n_train=len(y), alpha=getattr(m, "alpha_", np.nan),
                   n_selected=int(np.sum(np.abs(m.coef_) > 1e-10)) if m.sparse else np.nan)
        if hasattr(m, "l1_ratio_"):
            rec["l1_ratio"] = m.l1_ratio_
        if hasattr(m, "k_"):
            rec["n_factors"] = m.k_
        out.append(rec)
        if m.name not in ("Mean", "OLS (all)"):
            full = np.zeros(len(names))
            full[cols] = m.coef_
            coefs[m.name] = full
    return out, (str(q), h, coefs)


def evaluation_tasks(B, horizons=HORIZONS, start=OOS_START, end=None):
    first = B["first"]
    fv = B["first_vintage"]
    end = end or first.index.max()
    tasks = []
    for q in pd.period_range(start, end, freq="Q"):
        if q not in first.index:
            continue
        for h in horizons:
            T = origin_month(q, h)
            if fv[q] <= T:   # first estimate already published: nothing to nowcast
                continue
            tasks.append((q, h))
    return tasks


def run_backtest(B=None, tasks=None, n_jobs=2, models_factory=None, **kw):
    B = B or build_all(HORIZONS)
    tasks = tasks or evaluation_tasks(B)
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_origin)(B, q, h, models_factory() if models_factory else None, **kw)
        for q, h in tasks)
    rows = [r for out, _ in res for r in out]
    preds = pd.DataFrame(rows)
    preds["first_estimate"] = preds["quarter"].map(lambda s: B["first"][pd.Period(s, "Q")])
    preds["latest_estimate"] = preds["quarter"].map(lambda s: B["latest"][pd.Period(s, "Q")])
    coefs = [c for _, c in res]
    return preds, coefs


def live_tasks(B, live_origin: pd.Period, horizons=HORIZONS):
    """(quarter, h) pairs whose origin is exactly ``live_origin`` and not yet published."""
    out = []
    for h in horizons:
        q = (live_origin - h).asfreq("Q")
        if origin_month(q, h) == live_origin and not B["gdp"].published_by(q, live_origin):
            out.append((q, h))
    return out
