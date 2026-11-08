"""Independent validation checks (results in outputs/tables/validation_checks.csv).

    python scripts/validate.py

1. No look-ahead: for every backtest origin, rebuild the design row from data
   truncated at the origin month and confirm it equals the row used in the backtest.
2. First estimates: the real-time database reproduces ONS's published headline
   first estimates (rounded to 1 dp) for selected quarters.
3. Group lasso optimality: the from-scratch FISTA solution satisfies the
   Karush-Kuhn-Tucker conditions on the real design.
4. Training targets: no training quarter's first estimate was published after the origin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nowcast.backtest import evaluation_tasks  # noqa: E402
from nowcast.config import HORIZONS, OUTPUTS  # noqa: E402
from nowcast.features import (build_all, build_design, feature_groups, feature_names,  # noqa: E402
                              main_spec, origin_month, training_mask, transform_panel)
from nowcast.models import Standardiser, _group_lasso_path, group_lambda_max  # noqa: E402

# ONS headline first estimates of quarterly GDP growth, % (GDP first/preliminary estimate bulletins)
ONS_HEADLINES = {"2008Q4": -1.5, "2009Q1": -1.9, "2016Q3": 0.5, "2022Q3": -0.2,
                 "2024Q4": 0.1, "2025Q1": 0.7, "2025Q2": 0.3, "2025Q3": 0.1}


def check_truncation(B, tasks):
    names = feature_names()
    cols = [j for j, n in enumerate(names) if main_spec(n)]
    worst, n_cells = 0.0, 0
    for q, h in tasks:
        T = origin_month(q, h)
        panel_trunc = B["panel"].loc[:T]
        row = build_design(transform_panel(panel_trunc), B["gdp"], h,
                           pd.PeriodIndex([q]), B["rt"]).values[0, cols]
        ref = B["X"][h].loc[q].values[cols]
        both = np.isfinite(row) & np.isfinite(ref)
        if not np.array_equal(np.isfinite(row), np.isfinite(ref)):
            return dict(check="no look-ahead (truncated rebuild)", passed=False,
                        detail=f"missing-value pattern differs at {q} h={h}")
        worst = max(worst, float(np.max(np.abs(row[both] - ref[both]), initial=0)))
        n_cells += int(both.sum())
    return dict(check="no look-ahead (truncated rebuild)", passed=worst < 1e-9,
                detail=f"{len(tasks)} origins, {n_cells:,} feature values, max abs diff {worst:.1e}")


def check_training_targets(B, tasks):
    bad = 0
    for q, h in tasks:
        T = origin_month(q, h)
        m = training_mask(B["X"][h].index, B["gdp"], T)
        fv = B["first_vintage"].reindex(B["X"][h].index[m])
        bad += int((fv > T).sum()) + int(q in B["X"][h].index[m])
    return dict(check="training uses only published first estimates", passed=bad == 0,
                detail=f"{len(tasks)} origins, violations: {bad}")


def check_headlines(B):
    rows, ok = [], True
    for q, v in ONS_HEADLINES.items():
        mine = round(float(B["first"][pd.Period(q, "Q")]), 1)
        ok &= abs(mine - v) < 1e-9
        rows.append(f"{q} {mine:+.1f} vs {v:+.1f}")
    return dict(check="first estimates match ONS headlines", passed=ok, detail="; ".join(rows))


def check_group_lasso_kkt(B, q="2019Q4", h=2):
    names = feature_names()
    cols = np.array([j for j, n in enumerate(names) if main_spec(n)])
    groups = feature_groups()[cols]
    T = origin_month(pd.Period(q, "Q"), h)
    X = B["X"][h]
    m = training_mask(X.index, B["gdp"], T)
    Z = Standardiser().fit(X.values[m][:, cols]).transform(X.values[m][:, cols])
    y = B["first"].reindex(X.index[m]).values
    Z, y = Z - Z.mean(0), y - y.mean()
    n = len(y)
    lmax = group_lambda_max(Z, y, groups)
    worst_active, worst_zero = 0.0, 0.0
    for frac in (0.3, 0.1, 0.03):
        lam = lmax * frac
        b = _group_lasso_path(Z, y, groups, lmax * np.logspace(0, np.log10(frac), 15),
                              max_iter=20000, tol=1e-10)[-1]
        grad = Z.T @ (y - Z @ b) / n
        for g in np.unique(groups):
            idx = groups == g
            w = np.sqrt(idx.sum())
            nb = np.linalg.norm(b[idx])
            if nb > 0:   # stationarity: grad_g = lam * w * b_g / ||b_g||
                worst_active = max(worst_active, np.linalg.norm(grad[idx] - lam * w * b[idx] / nb) / (lam * w))
            else:        # ||grad_g|| <= lam * w
                worst_zero = max(worst_zero, np.linalg.norm(grad[idx]) / (lam * w) - 1)
    return dict(check="group lasso KKT conditions", passed=worst_active < 1e-3 and worst_zero < 1e-3,
                detail=f"{q} h={h}, 3 penalties: max relative stationarity gap {worst_active:.1e}, "
                       f"max zero-group excess {max(worst_zero, 0):.1e}")


def main():
    B = build_all(HORIZONS)
    tasks = evaluation_tasks(B)
    res = [check_truncation(B, tasks), check_training_targets(B, tasks), check_headlines(B),
           check_group_lasso_kkt(B)]
    out = pd.DataFrame(res)
    (OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / "tables" / "validation_checks.csv", index=False)
    pd.set_option("display.max_colwidth", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
