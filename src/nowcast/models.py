"""Nowcasting models.

Every model sees the same standardised design matrix (training-sample means and
standard deviations; missing values set to the training mean, i.e. 0 after
standardising) and tunes its penalty on the same blocked time-series folds, so
differences in accuracy come from the estimator, not the plumbing.

Benchmarks
  Mean          training-sample mean growth
  AR            OLS on the two real-time GDP lags
  Bridge        OLS on the latest three months of real-time Index of Services
                and Index of Production growth + GDP lags; the traditional
                "bridge equation"
  OLS (all)     minimum-norm least squares on all 428 features: what happens
                without regularisation when p > n
Regularised
  Ridge, Lasso, Elastic net, Adaptive lasso, Group lasso
Dimension reduction
  Factor        principal components of the indicators + GDP lags (Stock-Watson
                diffusion index), number of factors chosen by CV
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RIDGE_ALPHAS = np.logspace(-2, 6, 60)
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


# --------------------------------------------------------------------------- utilities
class Standardiser:
    def fit(self, X):
        with warnings.catch_warnings():   # all-missing columns (series not yet started)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        self.empty = ~np.isfinite(self.mu)          # no training data at all
        self.mu = np.where(self.empty, 0.0, self.mu)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        Z[:, self.empty] = 0.0
        return np.where(np.isfinite(Z), Z, 0.0)


def time_folds(n: int, n_splits: int = 5):
    """Blocked, expanding-window folds: train on the past, validate on the next block."""
    test = max(3, n // 8)
    folds = []
    for k in range(n_splits, 0, -1):
        end = n - (k - 1) * test
        start = end - test
        if start >= 10:
            folds.append((np.arange(0, start), np.arange(start, end)))
    if not folds:  # very short samples: a single split
        cut = max(5, int(n * 0.7))
        folds = [(np.arange(cut), np.arange(cut, n))]
    return folds


def pick_alpha(fold_mse: np.ndarray, alphas: np.ndarray, one_se: bool) -> int:
    """Index of the chosen penalty. fold_mse: (n_alphas, n_folds); alphas in any order.

    one_se=False: minimum mean CV error. one_se=True: the largest penalty whose
    mean CV error is within one standard error of the minimum (Breiman et al.).
    """
    mean = fold_mse.mean(1)
    k = int(np.argmin(mean))
    if not one_se or fold_mse.shape[1] < 2:
        return k
    se = fold_mse.std(1, ddof=1)[k] / np.sqrt(fold_mse.shape[1])
    ok = np.where(mean <= mean[k] + se)[0]
    return int(ok[np.argmax(alphas[ok])])


def ols(Z, y):
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[0], beta[1:]


# --------------------------------------------------------------------------- base
class Model:
    name = "base"
    sparse = False

    def fit(self, Z, y, folds, groups=None, names=None):
        raise NotImplementedError

    def predict(self, Z):
        return self.intercept_ + Z @ self.coef_


class MeanModel(Model):
    name = "Mean"

    def fit(self, Z, y, folds, groups=None, names=None):
        self.intercept_, self.coef_ = float(np.mean(y)), np.zeros(Z.shape[1])
        return self


class SubsetOLS(Model):
    """OLS on a named subset of columns (AR and bridge benchmarks)."""

    def __init__(self, name, prefixes):
        self.name, self.prefixes = name, prefixes

    def fit(self, Z, y, folds, groups=None, names=None):
        idx = [j for j, n in enumerate(names) if n.startswith(self.prefixes)]
        b0, b = ols(Z[:, idx], y)
        self.intercept_, self.coef_ = b0, np.zeros(Z.shape[1])
        self.coef_[idx] = b
        return self


class MinNormOLS(Model):
    name = "OLS (all)"

    def fit(self, Z, y, folds, groups=None, names=None):
        self.intercept_, self.coef_ = ols(Z, y)
        return self


class RidgeModel(Model):
    name = "Ridge"

    def __init__(self, one_se=False):
        self.one_se = one_se

    def fit(self, Z, y, folds, groups=None, names=None):
        errs = np.zeros((len(RIDGE_ALPHAS), len(folds)))
        for f, (tr, va) in enumerate(folds):
            for k, a in enumerate(RIDGE_ALPHAS):
                m = Ridge(alpha=a).fit(Z[tr], y[tr])
                errs[k, f] = np.mean((y[va] - m.predict(Z[va])) ** 2)
        if not self.one_se:   # pooled squared error, as in the main specification
            sizes = np.array([len(va) for _, va in folds])
            self.alpha_ = RIDGE_ALPHAS[int(np.argmin(errs @ sizes))]
        else:
            self.alpha_ = RIDGE_ALPHAS[pick_alpha(errs, RIDGE_ALPHAS, True)]
        m = Ridge(alpha=self.alpha_).fit(Z, y)
        self.intercept_, self.coef_ = m.intercept_, m.coef_
        return self


class LassoModel(Model):
    name = "Lasso"
    sparse = True

    def __init__(self, one_se=False):
        self.one_se = one_se

    def fit(self, Z, y, folds, groups=None, names=None):
        m = LassoCV(cv=folds, alphas=60, max_iter=20000, tol=1e-4).fit(Z, y)
        if self.one_se:
            a = m.alphas_[pick_alpha(m.mse_path_, m.alphas_, True)]
            m = Lasso(alpha=a, max_iter=20000, tol=1e-4).fit(Z, y)
            m.alpha_ = a
        self.alpha_, self.intercept_, self.coef_ = m.alpha_, m.intercept_, m.coef_
        return self


class ElasticNetModel(Model):
    name = "Elastic net"
    sparse = True

    def __init__(self, one_se=False):
        self.one_se = one_se

    def fit(self, Z, y, folds, groups=None, names=None):
        m = ElasticNetCV(l1_ratio=EN_L1_RATIOS, cv=folds, alphas=40,
                         max_iter=20000, tol=1e-4).fit(Z, y)
        if self.one_se:
            r = int(np.argmin(m.mse_path_.mean(2).min(1)))
            a = m.alphas_[r][pick_alpha(m.mse_path_[r], m.alphas_[r], True)]
            l1 = m.l1_ratio[r]
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000, tol=1e-4).fit(Z, y)
            m.alpha_, m.l1_ratio_ = a, l1
        self.alpha_, self.l1_ratio_ = m.alpha_, m.l1_ratio_
        self.intercept_, self.coef_ = m.intercept_, m.coef_
        return self


class AdaptiveLassoModel(Model):
    """Zou (2006): lasso with weights 1/|b_ridge|, via rescaling the columns."""
    name = "Adaptive lasso"
    sparse = True

    def fit(self, Z, y, folds, groups=None, names=None):
        r = RidgeModel().fit(Z, y, folds)
        w = np.abs(r.coef_) + 1e-8
        m = LassoCV(cv=folds, alphas=60, max_iter=20000, tol=1e-4).fit(Z * w, y)
        self.alpha_, self.intercept_, self.coef_ = m.alpha_, m.intercept_, m.coef_ * w
        return self


# --------------------------------------------------------------------------- group lasso
def _group_lasso_path(Z, y, groups, lambdas, b_init=None, max_iter=800, tol=1e-5):
    """FISTA for (1/2n)||y - Zb||^2 + lam * sum_g sqrt(p_g) ||b_g||_2, y and Z centred."""
    n, p = Z.shape
    ug = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ug]
    wts = np.array([np.sqrt(len(m)) for m in members])
    L = np.linalg.norm(Z, 2) ** 2 / n
    step = 1.0 / L
    b = np.zeros(p) if b_init is None else b_init.copy()
    path = []
    for lam in lambdas:
        x_old, v, t = b.copy(), b.copy(), 1.0
        for _ in range(max_iter):
            grad = Z.T @ (Z @ v - y) / n
            u = v - step * grad
            x = np.zeros(p)
            for m, w in zip(members, wts):
                nrm = np.linalg.norm(u[m])
                thr = step * lam * w
                if nrm > thr:
                    x[m] = (1 - thr / nrm) * u[m]
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            v = x + ((t - 1) / t_new) * (x - x_old)
            if np.linalg.norm(x - x_old) <= tol * max(1.0, np.linalg.norm(x_old)):
                x_old = x
                break
            x_old, t = x, t_new
        b = x_old
        path.append(b.copy())
    return np.array(path)


def group_lambda_max(Z, y, groups):
    n = len(y)
    return max(np.linalg.norm(Z[:, groups == g].T @ y) / (n * np.sqrt(np.sum(groups == g)))
               for g in np.unique(groups))


class GroupLassoModel(Model):
    """Yuan & Lin (2006): all six lags of an indicator enter or leave together."""
    name = "Group lasso"
    sparse = True
    n_lambdas = 25

    def __init__(self, one_se=False):
        self.one_se = one_se

    def fit(self, Z, y, folds, groups=None, names=None):
        ym = y.mean()
        lmax = group_lambda_max(Z, y - ym, groups)
        lambdas = lmax * np.logspace(0, -2.5, self.n_lambdas)
        errs = np.zeros((len(lambdas), len(folds)))
        for f, (tr, va) in enumerate(folds):
            zm, ymt = Z[tr].mean(0), y[tr].mean()
            path = _group_lasso_path(Z[tr] - zm, y[tr] - ymt, groups, lambdas)
            pred = ymt + (Z[va] - zm) @ path.T
            errs[:, f] = ((y[va][:, None] - pred) ** 2).mean(0)
        if not self.one_se:
            sizes = np.array([len(va) for _, va in folds])
            k = int(np.argmin(errs @ sizes))
        else:
            k = pick_alpha(errs, lambdas, True)
        self.alpha_ = lambdas[k]
        zm = Z.mean(0)
        path = _group_lasso_path(Z - zm, y - ym, groups, lambdas[: k + 1])
        self.coef_ = path[-1]
        self.intercept_ = ym - zm @ self.coef_
        return self


# --------------------------------------------------------------------------- factor model
class FactorModel(Model):
    """Diffusion index: y on the first k principal components of the indicators + GDP lags."""
    name = "Factor"
    k_max = 8

    def _design(self, pca, Z, ind, gdp, k):
        return np.column_stack([pca.transform(Z[:, ind])[:, :k], Z[:, gdp]])

    def fit(self, Z, y, folds, groups=None, names=None):
        ind = np.array([not n.startswith("GDP_") for n in names])
        gdp = ~ind
        errs = np.zeros(self.k_max)
        for tr, va in folds:
            kk = min(self.k_max, len(tr) - 3)
            pca = PCA(n_components=kk).fit(Z[tr][:, ind])
            for k in range(1, self.k_max + 1):
                kq = min(k, kk)
                b0, b = ols(self._design(pca, Z[tr], ind, gdp, kq), y[tr])
                errs[k - 1] += np.sum((y[va] - b0 - self._design(pca, Z[va], ind, gdp, kq) @ b) ** 2)
        self.k_ = int(np.argmin(errs)) + 1
        self.pca_ = PCA(n_components=self.k_).fit(Z[:, ind])
        self.ind_, self.gdp_ = ind, gdp
        self.b0_, self.b_ = ols(self._design(self.pca_, Z, ind, gdp, self.k_), y)
        # implied coefficients on the standardised features (for reporting)
        load = self.pca_.components_.T  # p_ind x k
        coef = np.zeros(Z.shape[1])
        coef[ind] = load @ self.b_[: self.k_]
        coef[gdp] = self.b_[self.k_:]
        self.coef_ = coef
        self.intercept_ = self.b0_ - self.pca_.mean_ @ coef[ind]
        return self

    def predict(self, Z):
        return self.b0_ + self._design(self.pca_, Z, self.ind_, self.gdp_, self.k_) @ self.b_


# --------------------------------------------------------------------------- partial penalisation
BRIDGE_PREFIXES = ("GDP_",) + tuple(f"RT_{s}_l{j}" for s in ("IOS", "IOP") for j in range(3))


class BridgeAnchored(Model):
    """Leave the bridge regressors unpenalised; regularise only the other indicators.

    Minimising ||y - a - Z1 b1 - Z2 b2||^2 + lambda * P(b2) over (a, b1) has a closed
    form, so by the Frisch-Waugh-Lovell theorem the problem is equivalent to
    running the penalised model on y and Z2 *residualised* on [1, Z1]. b1 is then
    recovered by OLS of y - Z2 b2 on [1, Z1].
    """

    def __init__(self, inner: Model, name: str | None = None, core=BRIDGE_PREFIXES):
        self.inner, self.core = inner, core
        self.name = name or f"Bridge + {inner.name.lower()}"
        self.sparse = inner.sparse

    def fit(self, Z, y, folds, groups=None, names=None):
        c = np.array([n.startswith(self.core) for n in names])
        A = np.column_stack([np.ones(len(Z)), Z[:, c]])
        P = A @ np.linalg.pinv(A)                      # projection onto [1, Z1]
        yr, Z2r = y - P @ y, Z[:, ~c] - P @ Z[:, ~c]
        names2 = [n for n, k in zip(names, c) if not k]
        g2 = groups[~c] if groups is not None else None
        self.inner.fit(Z2r, yr, folds, groups=g2, names=names2)
        b2 = self.inner.coef_
        b0, b1 = ols(Z[:, c], y - Z[:, ~c] @ b2)
        self.coef_ = np.zeros(Z.shape[1])
        self.coef_[c], self.coef_[~c] = b1, b2
        self.intercept_ = b0
        self.alpha_ = getattr(self.inner, "alpha_", np.nan)
        self.core_mask_ = c
        return self


# --------------------------------------------------------------------------- registry

def all_models():
    return [
        MeanModel(),
        SubsetOLS("AR", ("GDP_",)),
        SubsetOLS("Bridge", BRIDGE_PREFIXES),
        MinNormOLS(),
        RidgeModel(),
        LassoModel(),
        ElasticNetModel(),
        AdaptiveLassoModel(),
        GroupLassoModel(),
        FactorModel(),
        BridgeAnchored(RidgeModel()),
        BridgeAnchored(LassoModel()),
        BridgeAnchored(ElasticNetModel()),
        BridgeAnchored(GroupLassoModel()),
        BridgeAnchored(FactorModel(), name="Bridge + factors"),
    ]


REGULARISED = ["Ridge", "Lasso", "Elastic net", "Adaptive lasso", "Group lasso"]
ANCHORED = ["Bridge + ridge", "Bridge + lasso", "Bridge + elastic net",
            "Bridge + group lasso", "Bridge + factors"]
# Fixed before the anchored models were evaluated: equal weights on the anchored set.
COMBINATION_MEMBERS = ANCHORED
