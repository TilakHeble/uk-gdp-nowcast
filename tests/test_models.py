import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, Ridge

from nowcast.evaluate import diebold_mariano
from nowcast.models import (BridgeAnchored, GroupLassoModel, Model, Standardiser,
                            _group_lasso_path, time_folds)

rng = np.random.default_rng(0)


def test_group_lasso_singletons_equals_lasso():
    n, p = 80, 10
    X = rng.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)
    y = X[:, 0] * 2 - X[:, 3] + rng.normal(size=n)
    y = y - y.mean()
    lam = 0.1
    gl = _group_lasso_path(X, y, np.arange(p), [lam], max_iter=20000, tol=1e-10)[-1]
    la = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
    assert np.allclose(gl, la, atol=1e-4)


def test_group_lasso_zeroes_whole_groups():
    n = 120
    groups = np.repeat(np.arange(6), 4)
    X = rng.normal(size=(n, 24))
    y = X[:, :4] @ np.array([1.0, 0.5, -0.5, 0.2]) + 0.5 * rng.normal(size=n)
    m = GroupLassoModel().fit(X, y, time_folds(n), groups=groups)
    norms = [np.linalg.norm(m.coef_[groups == g]) for g in range(6)]
    assert norms[0] > 0.3
    for g in range(6):  # each group is either entirely zero or entirely non-zero
        b = m.coef_[groups == g]
        assert np.all(b == 0) or np.all(b != 0)


class FixedRidge(Model):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, Z, y, folds, groups=None, names=None):
        r = Ridge(alpha=self.alpha).fit(Z, y)
        self.intercept_, self.coef_ = r.intercept_, r.coef_
        return self


def test_bridge_anchored_matches_direct_partial_penalty():
    """FWL: residualise-then-ridge equals ridge with an unpenalised core block."""
    n, alpha = 60, 5.0
    names = ["GDP_q1", "RT_IOS_l0", "x1", "x2", "x3", "x4"]
    Z = rng.normal(size=(n, 6))
    y = Z @ np.array([0.5, 1.0, 0.3, 0.0, -0.2, 0.1]) + rng.normal(size=n)
    m = BridgeAnchored(FixedRidge(alpha), core=("GDP_", "RT_IOS_l0")).fit(Z, y, None, names=names)
    # direct solution: minimise ||y - a - Zb||^2 + alpha * ||b_extra||^2
    A = np.column_stack([np.ones(n), Z])
    pen = np.diag([0, 0, 0, alpha, alpha, alpha, alpha])
    beta = np.linalg.solve(A.T @ A + pen, A.T @ y)
    assert np.isclose(m.intercept_, beta[0], atol=1e-8)
    assert np.allclose(m.coef_, beta[1:], atol=1e-8)


def test_bridge_anchored_infinite_penalty_is_bridge_ols():
    n = 50
    names = ["GDP_q1", "RT_IOS_l0", "x1", "x2"]
    Z = rng.normal(size=(n, 4))
    y = Z[:, 1] + rng.normal(size=n)
    m = BridgeAnchored(FixedRidge(1e12), core=("GDP_", "RT_IOS_l0")).fit(Z, y, None, names=names)
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), Z[:, :2]]), y, rcond=None)
    assert np.allclose(m.coef_[:2], b[1:], atol=1e-6)
    assert np.allclose(m.coef_[2:], 0, atol=1e-6)


def test_standardiser_uses_training_stats_and_imputes_zero():
    Xtr = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
    st = Standardiser().fit(Xtr)
    z = st.transform(np.array([[7.0, 4.0], [np.nan, np.nan]]))
    assert np.isclose(z[0, 0], (7 - 3) / np.std([1, 3, 5]))
    assert z[0, 1] == 0 and z[1, 0] == 0  # all-missing column and missing value -> 0


def test_time_folds_never_validate_on_the_past():
    for n in [25, 60, 110]:
        for tr, va in time_folds(n):
            assert tr.max() < va.min()
            assert len(tr) >= 10


def test_dm_sign_and_symmetry():
    e1 = rng.normal(size=100) * 2
    e2 = rng.normal(size=100)
    s12, p12 = diebold_mariano(e1, e2)
    s21, p21 = diebold_mariano(e2, e1)
    assert s12 > 0 and np.isclose(s12, -s21) and np.isclose(p12, p21)
