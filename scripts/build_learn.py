"""Build the learn/ notebooks: regularised regression from basics to advanced.

    python scripts/build_learn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

LEARN = Path(__file__).resolve().parents[1] / "learn"

SETUP = code("""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (8, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(42)
""")

GDP_DATA = code("""
# The project's real data: the horizon-2 design (nowcast made at the end of the quarter)
from nowcast.features import build_all, main_spec
from nowcast.models import Standardiser
B = build_all([2])
X = B["X"][2]
X = X[[c for c in X.columns if main_spec(c)]]       # real-time output + other indicators
y = B["first"].reindex(X.index)                      # target: ONS's first published estimate
keep = y.notna() & ~X.index.isin(pd.period_range("2020Q1", "2021Q2", freq="Q"))
X, y = X[keep], y[keep]
names = list(X.columns)
print(X.shape, "quarters x features")
""")

# ----------------------------------------------------------------------------- 01
L01 = [
    md("""
# 1. Why regularise? When ordinary least squares breaks

**Learning goals**
* See why OLS overfits when there are many predictors relative to observations.
* Understand the bias-variance trade-off that regularisation exploits.
* Meet the nowcasting problem in this project: 236 candidate predictors, 31 to 107 quarters of training data.

## The problem
OLS chooses coefficients $\\hat\\beta$ to minimise the residual sum of squares
$$\\text{RSS}(\\beta) = \\sum_{i=1}^n (y_i - x_i^\\top \\beta)^2 .$$
It is unbiased, and among unbiased linear estimators it has the smallest variance (Gauss-Markov).
But unbiased is not the same as accurate. Its expected prediction error is roughly
$$\\sigma^2\\left(1 + \\frac{p}{n}\\right)$$
so every extra predictor adds variance. When $p \\ge n$, $X^\\top X$ is singular, infinitely many
coefficient vectors fit the training data perfectly, and the fitted model is pure memorisation.
"""),
    SETUP,
    md("## A simulation: train vs test error as predictors are added\n"
       "Only the first 5 predictors matter; the rest are noise. n = 60 training rows."),
    code("""
n, n_test, p_max, sigma = 60, 2000, 120, 1.0
beta = np.zeros(p_max); beta[:5] = [1.0, -0.8, 0.6, 0.5, -0.4]
Xall = rng.normal(size=(n + n_test, p_max))
yall = Xall @ beta + sigma * rng.normal(size=n + n_test)
Xtr, Xte, ytr, yte = Xall[:n], Xall[n:], yall[:n], yall[n:]

ps, tr_err, te_err = range(1, p_max + 1, 3), [], []
for p in ps:
    b, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xtr[:, :p]], ytr, rcond=None)
    tr_err.append(np.mean((ytr - np.c_[np.ones(n), Xtr[:, :p]] @ b) ** 2))
    te_err.append(np.mean((yte - np.c_[np.ones(n_test), Xte[:, :p]] @ b) ** 2))

plt.plot(ps, tr_err, label="training MSE"); plt.plot(ps, te_err, label="test MSE")
plt.axvline(n, color="grey", ls="--"); plt.text(n + 1, 4, "p = n")
plt.yscale("log"); plt.xlabel("number of predictors p"); plt.ylabel("MSE"); plt.legend()
plt.title("OLS: training error falls to zero, test error explodes near p = n");
"""),
    md("""
Training error falls monotonically; test error is lowest around the 5 true predictors, explodes as
$p \\to n$ and, beyond $p = n$, the minimum-norm solution gets slightly better again (the "double
descent" curve). None of these OLS fits is as good as knowing which 5 matter.

## Bias-variance trade-off
For a new point, expected squared error = irreducible noise + bias² + variance.
Regularisation **deliberately adds a little bias** (shrinking coefficients towards zero) in exchange for
a large cut in variance. All the methods in this folder minimise
$$\\text{RSS}(\\beta) + \\lambda \\, \\text{Penalty}(\\beta)$$
and differ only in the penalty:

| Method | Penalty | Effect |
|---|---|---|
| Ridge | $\\sum_j \\beta_j^2$ | shrinks all coefficients, keeps all |
| Lasso | $\\sum_j \\lvert\\beta_j\\rvert$ | shrinks and sets some exactly to zero |
| Elastic net | $\\alpha \\sum \\lvert\\beta_j\\rvert + \\frac{1-\\alpha}{2}\\sum \\beta_j^2$ | sparse, but keeps correlated groups together |
| Adaptive lasso | $\\sum_j w_j \\lvert\\beta_j\\rvert$ | less shrinkage on big effects |
| Group lasso | $\\sum_g \\sqrt{p_g}\\lVert\\beta_g\\rVert_2$ | selects whole groups of variables |

$\\lambda$ controls the trade-off: $\\lambda=0$ is OLS, $\\lambda\\to\\infty$ sets every coefficient to zero.
"""),
    md("## The same problem in this project\n"
       "Each nowcast uses 39 indicators x 6 monthly lags + 2 GDP lags = 236 features. Training samples run from "
       "31 quarters (2006) to 107 (2026)."),
    GDP_DATA,
    code("""
from nowcast.models import RidgeModel, time_folds
r = lambda a, b: np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
rows = []
for train_end, test in [("2007Q4", ("2008Q1", "2012Q4")), ("2015Q4", ("2016Q1", "2026Q2"))]:
    tr = X.index <= pd.Period(train_end, "Q")
    te = (X.index >= pd.Period(test[0], "Q")) & (X.index <= pd.Period(test[1], "Q"))
    st = Standardiser().fit(X[tr].values)
    Ztr, Zte = st.transform(X[tr].values), st.transform(X[te].values)
    A = lambda Z: np.c_[np.ones(len(Z)), Z]
    b, *_ = np.linalg.lstsq(A(Ztr), y[tr].values, rcond=None)
    rid = RidgeModel().fit(Ztr, y[tr].values, time_folds(tr.sum()))
    rows.append({"test period": f"{test[0]}-{test[1]}", "train quarters": int(tr.sum()),
                 "OLS in-sample": r(y[tr], A(Ztr) @ b), "OLS test": r(y[te], A(Zte) @ b),
                 "ridge test": r(y[te], rid.predict(Zte)),
                 "training mean test": r(y[te], np.full(te.sum(), y[tr].mean())),
                 "sd of target in test": y[te].std()})
pd.DataFrame(rows).set_index("test period").round(3)
"""),
    md("""
OLS fits the training quarters perfectly (in-sample RMSE ≈ 0) because there are more features than
quarters, and that perfect fit says nothing about out-of-sample accuracy. Out of sample, ridge
(penalty chosen by time-series cross-validation) is not better in every split. In 2008-12 it is
slightly worse than OLS, and both beat the average. In 2016-26 it beats OLS, but neither beats the
average. One split proves little. Across the main project's 361 real-time origins, ridge beats
minimum-norm OLS at every horizon (notebooks/04).

Notice the last two columns. Whether *any* model beats the simple average depends on how much GDP
moves in the test period. In a calm period the first estimate barely varies, and part of what does
vary is noise that ONS later revises away, so there is little left for indicators to explain. The
main project measures this properly with a full real-time backtest (notebook 04).

**Key takeaways**
* With $p$ close to or above $n$, OLS memorises the training data.
* Regularisation trades a little bias for much lower variance.
* The penalty's shape determines whether you get shrinkage, selection or group selection.
"""),
]

# ----------------------------------------------------------------------------- 02
L02 = [
    md("""
# 2. Ridge regression

**Learning goals**: closed form, the shrinkage view through the SVD, effective degrees of freedom,
why you must standardise, and how ridge treats correlated predictors.

$$\\hat\\beta^{\\text{ridge}} = \\arg\\min_\\beta \\; \\lVert y - X\\beta \\rVert^2 + \\lambda \\lVert\\beta\\rVert_2^2
= (X^\\top X + \\lambda I)^{-1} X^\\top y$$

Adding $\\lambda I$ makes the matrix invertible for any $\\lambda > 0$, even when $p > n$.
The intercept is not penalised: centre $y$ and the columns of $X$ first.
"""),
    SETUP,
    md("## Ridge from scratch vs scikit-learn"),
    code("""
from sklearn.linear_model import Ridge
n, p = 50, 8
X = rng.normal(size=(n, p)); beta = np.array([3, -2, 0, 0, 1.5, 0, 0, 0.5])
y = X @ beta + rng.normal(size=n)
Xc, yc = X - X.mean(0), y - y.mean()
lam = 10.0
b_scratch = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yc)
b_sklearn = Ridge(alpha=lam).fit(X, y).coef_
print(np.round(b_scratch, 4)); print(np.round(b_sklearn, 4))
print("max difference:", np.abs(b_scratch - b_sklearn).max())
"""),
    md("""
## The SVD view: ridge shrinks along principal directions
Write $X = U D V^\\top$ (singular values $d_j$). Then the fitted values are
$$X\\hat\\beta^{\\text{ridge}} = \\sum_j u_j \\, \\frac{d_j^2}{d_j^2 + \\lambda} \\, u_j^\\top y .$$
Each principal direction is shrunk by $d_j^2/(d_j^2+\\lambda)$: directions with little variance in
$X$ (small $d_j$, where OLS is noisiest) are shrunk most. The **effective degrees of freedom**
$\\text{df}(\\lambda) = \\sum_j d_j^2/(d_j^2+\\lambda)$ fall smoothly from $p$ to 0.
"""),
    code("""
d = np.linalg.svd(Xc, compute_uv=False)
lams = np.logspace(-2, 4, 200)
df = [(d**2 / (d**2 + l)).sum() for l in lams]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for l in [1, 30, 300]:
    ax[0].plot(d**2 / (d**2 + l), "o-", label=f"lambda={l}")
ax[0].set(xlabel="principal direction (largest variance first)", ylabel="shrinkage factor", title="Shrinkage by direction"); ax[0].legend()
ax[1].semilogx(lams, df); ax[1].set(xlabel="lambda", ylabel="effective df", title="Degrees of freedom");
"""),
    md("## Coefficient paths\nAs $\\lambda$ grows every coefficient shrinks towards zero, but none reaches exactly zero."),
    code("""
paths = np.array([Ridge(alpha=l).fit(X, y).coef_ for l in lams])
plt.semilogx(lams, paths); plt.axhline(0, color="k", lw=0.8)
plt.xlabel("lambda"); plt.ylabel("coefficient"); plt.title("Ridge coefficient paths");
"""),
    md("""
## Always standardise
The penalty $\\sum\\beta_j^2$ depends on units. A predictor measured in pence instead of pounds has a
coefficient 100 times smaller and is barely penalised. Standardise each column to mean 0, standard
deviation 1, **using the training sample only**, so the penalty treats predictors equally.
"""),
    code("""
X2 = X.copy(); X2[:, 0] *= 100          # same information, different units
b_raw = Ridge(alpha=lam).fit(X2, y).coef_
print("coef on predictor 0, original units  :", round(Ridge(alpha=lam).fit(X, y).coef_[0], 3))
print("coef on predictor 0, x100 (times 100):", round(b_raw[0] * 100, 3), " <- penalised far less")
"""),
    md("""
## Correlated predictors: ridge shares the weight
With two near-identical predictors OLS coefficients are unstable (they can be +10 and -8).
Ridge splits the effect roughly equally. That is why ridge suits a panel like ours, where
manufacturing output, production output and goods exports move together.
"""),
    code("""
z = rng.normal(size=200)
Xcor = np.c_[z + 0.05 * rng.normal(size=200), z + 0.05 * rng.normal(size=200)]
ycor = 2 * z + rng.normal(size=200)
for i in range(3):
    idx = rng.choice(200, 40, replace=False)
    ols = np.linalg.lstsq(np.c_[np.ones(40), Xcor[idx]], ycor[idx], rcond=None)[0][1:]
    rid = Ridge(alpha=5).fit(Xcor[idx], ycor[idx]).coef_
    print(f"sample {i}: OLS {np.round(ols, 2)}   ridge {np.round(rid, 2)}")
"""),
    md("""
## Bayesian interpretation
Ridge is the posterior mode (and mean) under a Gaussian prior $\\beta_j \\sim N(0, \\sigma^2/\\lambda)$:
a belief that effects are small and spread across many predictors.

**Key takeaways**
* Closed form; works when $p > n$.
* Shrinks hardest along low-variance directions; never selects.
* Standardise on the training sample; leave the intercept unpenalised.
"""),
]

# ----------------------------------------------------------------------------- 03
L03 = [
    md("""
# 3. The lasso

$$\\hat\\beta^{\\text{lasso}} = \\arg\\min_\\beta \\; \\frac{1}{2n}\\lVert y - X\\beta \\rVert^2 + \\lambda \\lVert\\beta\\rVert_1$$

**Learning goals**: why the L1 penalty produces exact zeros, soft-thresholding, coordinate descent
from scratch, paths, and the lasso's known weaknesses.
"""),
    SETUP,
    md("""
## Soft-thresholding
With orthonormal predictors the lasso solution has a closed form:
$$\\hat\\beta_j^{\\text{lasso}} = S(\\hat\\beta_j^{\\text{OLS}}, \\lambda) = \\text{sign}(\\hat\\beta_j^{\\text{OLS}}) \\max(|\\hat\\beta_j^{\\text{OLS}}| - \\lambda, 0)$$
Ridge instead scales: $\\hat\\beta_j^{\\text{OLS}}/(1+\\lambda)$. Small OLS coefficients become exactly
zero under the lasso, and large ones are shifted towards zero by a constant amount.
"""),
    code("""
b = np.linspace(-3, 3, 300); lam = 1.0
plt.plot(b, b, "k:", label="OLS")
plt.plot(b, np.sign(b) * np.maximum(np.abs(b) - lam, 0), label="lasso (soft-threshold)")
plt.plot(b, b / (1 + lam), label="ridge (proportional)")
plt.xlabel("OLS coefficient"); plt.ylabel("regularised coefficient"); plt.legend();
"""),
    md("""
## Why corners create zeros
Equivalently, minimise RSS subject to $\\sum|\\beta_j| \\le t$. The L1 constraint region is a diamond
with corners on the axes; the RSS contours usually first touch it at a corner, where some
coefficients are exactly zero. The ridge constraint is a circle with no corners.

## Coordinate descent from scratch
Cycle through coefficients, each time solving the one-dimensional problem exactly with
soft-thresholding. With standardised columns ($\\frac1n\\sum x_{ij}^2=1$):
$$\\beta_j \\leftarrow S\\Big(\\tfrac{1}{n} x_j^\\top r_{(j)}, \\lambda\\Big), \\qquad r_{(j)} = y - \\sum_{k\\ne j} x_k\\beta_k$$
"""),
    code("""
def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso_cd(X, y, lam, n_iter=500, tol=1e-8):
    n, p = X.shape
    b = np.zeros(p); r = y.copy()
    for _ in range(n_iter):
        b_old = b.copy()
        for j in range(p):
            r += X[:, j] * b[j]                 # remove j's contribution
            b[j] = soft(X[:, j] @ r / n, lam)   # 1-D lasso solution
            r -= X[:, j] * b[j]
        if np.max(np.abs(b - b_old)) < tol:
            break
    return b

from sklearn.linear_model import Lasso
n, p = 100, 20
X = rng.normal(size=(n, p)); X = (X - X.mean(0)) / X.std(0)
beta = np.r_[[2, -1.5, 1, 0.5], np.zeros(p - 4)]
y = X @ beta + rng.normal(size=n); y = y - y.mean()
b_ours = lasso_cd(X, y, lam=0.2)
b_skl = Lasso(alpha=0.2, fit_intercept=False, tol=1e-10, max_iter=100000).fit(X, y).coef_
print("ours   :", np.round(b_ours, 3))
print("sklearn:", np.round(b_skl, 3))
print("non-zero:", (np.abs(b_ours) > 0).sum(), "of", p)
"""),
    md("## Lasso paths: variables enter one at a time"),
    code("""
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(0.5, -3, 100))
plt.semilogx(alphas, coefs.T); plt.gca().invert_xaxis(); plt.axhline(0, c="k", lw=0.8)
plt.xlabel("lambda (decreasing)"); plt.ylabel("coefficient"); plt.title("Lasso paths: true signals enter first");
"""),
    md("## On the GDP data: which indicators does the lasso pick?"),
    GDP_DATA,
    code("""
from sklearn.linear_model import LassoCV
from nowcast.models import time_folds
st = Standardiser().fit(X.values); Z = st.transform(X.values)
m = LassoCV(cv=time_folds(len(y)), alphas=60, max_iter=50000).fit(Z, y.values)
sel = pd.Series(m.coef_, index=names)
sel = sel[sel.abs() > 1e-10].sort_values(key=np.abs, ascending=False)
print(f"lambda = {m.alpha_:.4f}: {len(sel)} of {len(names)} features selected")
sel.head(12).round(3)
"""),
    md("""
Feature names are `CODE_lag`: `RT_IOS_l0` is the real-time Index of Services for the latest month
available when the nowcast is made, `AP2Y` is vacancies, `IUMABEDR` is Bank Rate (see the indicator
catalogue in `src/nowcast/config.py`). With 108 quarters the lasso keeps a few dozen features
from several groups. Treat the list with suspicion: notebook 8 shows how much it changes when the
sample changes slightly.

## Weaknesses of the lasso
1. **Correlated predictors**: it tends to pick one from a correlated group, somewhat arbitrarily,
   and the choice can flip between samples (see notebook 8).
2. **At most n non-zero coefficients** when $p > n$.
3. **Biased large coefficients**: every selected coefficient is shrunk by $\\lambda$.
   Remedies: the relaxed lasso (refit OLS on the selected set) or the adaptive lasso (notebook 6).
4. **Selection is not inference**: a selected variable is not "significant" (notebook 8).

**Bayesian view**: the lasso is the posterior mode under a Laplace (double-exponential) prior,
a belief that most effects are zero and a few are large.
"""),
]

# ----------------------------------------------------------------------------- 04
L04 = [
    md("""
# 4. Elastic net

$$\\min_\\beta \\frac{1}{2n}\\lVert y - X\\beta\\rVert^2 + \\lambda\\Big[\\alpha\\lVert\\beta\\rVert_1 + \\tfrac{1-\\alpha}{2}\\lVert\\beta\\rVert_2^2\\Big]$$

$\\alpha$ (`l1_ratio` in scikit-learn) moves between ridge ($\\alpha=0$) and the lasso ($\\alpha=1$).
Zou and Hastie (2005) showed the ridge part produces a **grouping effect**: highly correlated
predictors get similar coefficients and enter or leave together, while the L1 part still gives
sparsity. It can also select more than $n$ variables.
"""),
    SETUP,
    md("## The grouping effect\nThree near-copies of one signal, plus 20 noise variables."),
    code("""
from sklearn.linear_model import Lasso, ElasticNet
n = 80
z = rng.normal(size=n)
group = np.c_[[z + 0.1 * rng.normal(size=n) for _ in range(3)]].T
noise = rng.normal(size=(n, 20))
X = np.c_[group, noise]; X = (X - X.mean(0)) / X.std(0)
y = 3 * z + rng.normal(size=n)
print("lasso      :", np.round(Lasso(alpha=0.1).fit(X, y).coef_[:5], 2))
print("elastic net:", np.round(ElasticNet(alpha=0.1, l1_ratio=0.3).fit(X, y).coef_[:5], 2))
"""),
    md("""
The lasso puts most weight on one copy; the elastic net shares it across all three. In nowcasting,
the "copies" are things like total production, manufacturing and goods exports. Sharing weight
makes the model less dependent on any single series being revised or published late.

## Tuning two parameters
Cross-validate over a grid of $\\alpha$ values and, for each, a path of $\\lambda$ values.
`ElasticNetCV` does this; the project uses `l1_ratio` in {0.1, 0.3, 0.5, 0.7, 0.9, 0.99}.
"""),
    code("""
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99], cv=KFold(5, shuffle=True, random_state=0)).fit(X, y)
print("chosen l1_ratio:", m.l1_ratio_, " lambda:", round(m.alpha_, 4))
mse = m.mse_path_.mean(axis=2)   # l1_ratio x alphas
for r, row in zip(m.l1_ratio, mse):
    print(f"l1_ratio={r:<5} best CV MSE={row.min():.3f}")
"""),
    md("""
(Shuffled K-fold is fine here only because these toy observations are independent. With time
series it leaks the future: see notebook 5.)

## On the project's data: is the selection stable?
The real design has 236 features: six monthly lags of each indicator, so every indicator is
itself a block of correlated columns. Refit the lasso and an elastic net ($\\alpha = 0.5$) on 40
random subsamples of 80% of the quarters, with the project's blocked time-series folds, and
record which **indicators** (any lag) are selected each time.
"""),
    GDP_DATA,
    code("""
from sklearn.linear_model import LassoCV
from nowcast.models import time_folds
Z = ((X - X.mean()) / X.std()).fillna(0).values
yv = y.values
ind = np.array([c.rsplit("_l", 1)[0] for c in names])
codes = pd.unique(ind)

def selected_indicators(coef):
    return set(ind[np.abs(coef) > 1e-10])

sel = {"Lasso": [], "Elastic net (0.5)": []}
for b in range(40):
    rows = np.sort(rng.choice(len(yv), int(0.8 * len(yv)), replace=False))   # keep time order
    Zb, yb = Z[rows], yv[rows]
    folds = time_folds(len(yb))
    sel["Lasso"].append(selected_indicators(LassoCV(cv=folds, max_iter=20000).fit(Zb, yb).coef_))
    sel["Elastic net (0.5)"].append(selected_indicators(
        ElasticNetCV(l1_ratio=0.5, cv=folds, max_iter=20000).fit(Zb, yb).coef_))

def mean_jaccard(sets):
    s = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return np.mean(s)

for k, sets in sel.items():
    print(f"{k:<18} indicators selected: median {np.median([len(s) for s in sets]):.0f}, "
          f"mean pairwise Jaccard similarity {mean_jaccard(sets):.2f}")
"""),
    code("""
freq = pd.DataFrame({k: pd.Series([c for s in sets for c in s]).value_counts() / len(sets)
                     for k, sets in sel.items()}).fillna(0)
top = freq.max(axis=1).sort_values(ascending=False).head(15).index
from nowcast.config import INDICATOR_BY_CODE
labels = [INDICATOR_BY_CODE[c].name if c in INDICATOR_BY_CODE else c for c in top]
ax = freq.loc[top].set_axis(labels).iloc[::-1].plot.barh(figsize=(8, 5.5))
ax.set_xlabel("share of the 40 subsamples in which the indicator is selected")
ax.set_title("Selection frequency: lasso vs elastic net");
"""),
    md("""
Read the chart as a stability check. Indicators near 1.0 are found in almost every subsample;
indicators near 0.2 are found only sometimes, usually because a correlated neighbour took their
place. The elastic net's ridge part tends to keep correlated indicators in together, and the lasso
tends to switch between them. That instability is why the project reports *groups* of indicators,
not individual lasso picks (see notebook 8 and `outputs/tables/selection_frequency.csv`).

**Key takeaways**
* Use the elastic net when predictors come in correlated blocks and you still want sparsity.
* Measure selection stability by refitting on subsamples. A single lasso fit's list of selected
  variables is not a finding.
* The extra tuning parameter costs computation, not much accuracy.
"""),
]

# ----------------------------------------------------------------------------- 05
L05 = [
    md("""
# 5. Choosing λ: cross-validation for time series

The penalty must be chosen from the data, without using the data you will be judged on.

**Learning goals**: k-fold CV, why shuffled CV leaks with time series, blocked (forward-chaining) CV,
the one-standard-error rule, and information criteria.
"""),
    SETUP,
    md("""
## Shuffled k-fold leaks information in time series
If observations are autocorrelated, a randomly held-out quarter has its neighbours in the training
folds, so validation error is too optimistic and favours small penalties (overfitting).
Below: an AR(1)-style target where an irrelevant trending predictor looks useful under shuffled CV.
"""),
    code("""
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
n, p = 120, 60
e = np.zeros(n)
for t in range(1, n): e[t] = 0.9 * e[t-1] + rng.normal()
X = np.cumsum(rng.normal(size=(n, p)), axis=0) * 0.1 + rng.normal(size=(n, p))  # persistent, irrelevant
y = e
res = {}
for name, cv in {"shuffled 5-fold": KFold(5, shuffle=True, random_state=0),
                 "forward-chaining 5-fold": TimeSeriesSplit(5)}.items():
    res[name] = [-cross_val_score(Ridge(alpha=a), X, y, cv=cv, scoring="neg_mean_squared_error").mean()
                 for a in np.logspace(-2, 4, 25)]
    print(f"{name:25s} chooses lambda = {np.logspace(-2, 4, 25)[int(np.argmin(res[name]))]:.2f},"
          f" min CV MSE = {min(res[name]):.2f}")
print("true out-of-sample MSE of the mean forecast is about", round(np.var(y), 2))
"""),
    md("""
Shuffled CV reports an error well below what is achievable from irrelevant predictors and chooses a
small penalty. Forward-chaining CV, where each fold trains on the past and validates on the next
block, gives an honest estimate.

## What this project does
`nowcast.models.time_folds(n)` creates up to 5 expanding-window folds, each validating on the next
block of about n/8 quarters, so tuning never uses the future. The same folds are used for every model.
"""),
    code("""
from nowcast.models import time_folds
for tr, va in time_folds(60):
    print(f"train rows {tr[0]:>2}-{tr[-1]:>2}   validate rows {va[0]}-{va[-1]}")
"""),
    md("""
## The one-standard-error rule
CV error curves are flat near the minimum. Choosing the largest $\\lambda$ whose CV error is within one
standard error of the minimum gives a simpler model with almost the same expected accuracy.
"""),
    code("""
from sklearn.linear_model import LassoCV
n, p = 100, 50
X = rng.normal(size=(n, p)); y = X[:, :3] @ np.array([1, -1, 0.5]) + rng.normal(size=n)
m = LassoCV(cv=5, alphas=80).fit(X, y)
mean, se = m.mse_path_.mean(1), m.mse_path_.std(1) / np.sqrt(m.mse_path_.shape[1])
k = int(np.argmin(mean)); thr = mean[k] + se[k]
lam_1se = m.alphas_[np.where(mean <= thr)[0].min()]   # alphas_ sorted large -> small
from sklearn.linear_model import Lasso
for lab, lam in [("min CV", m.alpha_), ("1-SE", lam_1se)]:
    print(f"{lab:7s} lambda={lam:.3f}  non-zero={np.sum(Lasso(alpha=lam).fit(X, y).coef_ != 0)}")
plt.semilogx(m.alphas_, mean); plt.fill_between(m.alphas_, mean - se, mean + se, alpha=0.2)
plt.axvline(m.alpha_, ls="--", label="min"); plt.axvline(lam_1se, ls=":", c="r", label="1-SE")
plt.xlabel("lambda"); plt.ylabel("CV MSE"); plt.legend();
"""),
    md("""
## Information criteria
For the lasso, the number of non-zero coefficients is an unbiased estimate of its degrees of freedom
(Zou, Hastie and Tibshirani, 2007), so AIC/BIC can choose $\\lambda$ without refitting
(`LassoLarsIC`). BIC is fast and often picks sparser models, but it relies on an estimate of the noise
variance, which is unreliable when $p > n$. This project uses CV.

**Key takeaways**
* Never shuffle time series in CV.
* Tune inside every backtest origin, using only data available at that origin.
* The 1-SE rule trades a sliver of fit for parsimony.
"""),
]

# ----------------------------------------------------------------------------- 06
L06 = [
    md("""
# 6. Adaptive lasso and group lasso

## Adaptive lasso (Zou, 2006)
$$\\min_\\beta \\frac{1}{2n}\\lVert y-X\\beta\\rVert^2 + \\lambda\\sum_j w_j|\\beta_j|, \\qquad w_j = 1/|\\tilde\\beta_j|^{\\gamma}$$
$\\tilde\\beta$ is a first-stage estimate (this project uses ridge). Big effects get small weights, so
are shrunk less; small ones get big weights and are removed. Under conditions, the adaptive lasso has
the **oracle property**: it selects the true variables with probability tending to one and estimates
them as efficiently as if the truth were known. The plain lasso does not.

**Trick**: rescale column $j$ by $1/w_j = |\\tilde\\beta_j|$, run an ordinary lasso, then multiply the
coefficients back by $|\\tilde\\beta_j|$.
"""),
    SETUP,
    code("""
from sklearn.linear_model import Lasso, Ridge
n, p = 200, 40
X = rng.normal(size=(n, p)); X = (X - X.mean(0)) / X.std(0)
beta = np.r_[[3, 2, -1.5], np.zeros(p - 3)]
y = X @ beta + rng.normal(size=n)
las = Lasso(alpha=0.15).fit(X, y).coef_
w = np.abs(Ridge(alpha=1).fit(X, y).coef_)
ada = Lasso(alpha=0.15).fit(X * w, y).coef_ * w
print("true      :", beta[:5])
print("lasso     :", np.round(las[:5], 2), " non-zero:", (las != 0).sum())
print("adaptive  :", np.round(ada[:5], 2), " non-zero:", (ada != 0).sum())
"""),
    md("""
The adaptive lasso keeps the large coefficients closer to their true size (less shrinkage bias). In this example both methods select the same three variables; the adaptive lasso's selection advantage shows up in larger samples with more correlated noise variables.

## Group lasso (Yuan and Lin, 2006)
In this project each indicator enters as six monthly lags. Asking "does retail sales matter?" is a
question about **all six together**. The group lasso penalises the Euclidean norm of each group:
$$\\min_\\beta \\frac{1}{2n}\\lVert y-X\\beta\\rVert^2 + \\lambda\\sum_{g}\\sqrt{p_g}\\,\\lVert\\beta_g\\rVert_2$$
The norm is not differentiable at $\\beta_g = 0$, so whole groups are set to zero, but within a
selected group coefficients are shrunk like ridge. $\\sqrt{p_g}$ stops big groups being penalised less.

### Solving it: proximal gradient (FISTA)
Take a gradient step on the smooth loss, then apply the group soft-threshold:
$$\\beta_g \\leftarrow \\Big(1 - \\frac{t\\lambda\\sqrt{p_g}}{\\lVert u_g \\rVert}\\Big)_+ u_g$$
with Nesterov momentum for speed. scikit-learn has no group lasso, so the project implements it in
`nowcast.models.GroupLassoModel`.
"""),
    code("""
from nowcast.models import _group_lasso_path, group_lambda_max
n, n_groups, size = 150, 15, 6
groups = np.repeat(np.arange(n_groups), size)
X = rng.normal(size=(n, n_groups * size)); X = (X - X.mean(0)) / X.std(0)
beta = np.zeros(n_groups * size); beta[:6] = [1, .8, .5, .3, .2, .1]; beta[6:12] = [-.6, -.6, -.4, 0, 0, 0]
y = X @ beta + rng.normal(size=n); y -= y.mean()
lmax = group_lambda_max(X, y, groups)
lams = lmax * np.logspace(0, -2, 40)
path = _group_lasso_path(X, y, groups, lams)
norms = np.array([[np.linalg.norm(b[groups == g]) for g in range(n_groups)] for b in path])
plt.semilogx(lams, norms[:, :2], lw=2.5, label=["group 0 (true)", "group 1 (true)"])
plt.semilogx(lams, norms[:, 2:], c="grey", alpha=0.5)
plt.gca().invert_xaxis(); plt.xlabel("lambda (decreasing)"); plt.ylabel("group norm ||b_g||")
plt.legend(); plt.title("Group lasso: whole groups enter the model");
"""),
    md("""
**Variants worth knowing**
* *Sparse group lasso* adds an L1 term to drop individual lags inside a selected group.
* *Overlapping group lasso* lets variables belong to several groups.

**Key takeaways**
* Adaptive lasso: less bias on big effects, cleaner selection; needs a decent first stage.
* Group lasso: selection at the level of a natural unit (an indicator, a dummy set, a spline basis).
"""),
]

# ----------------------------------------------------------------------------- 07
L07 = [
    md("""
# 7. Regularisation vs factor models

Central banks have nowcast with **dynamic factor models** for two decades: summarise the
indicator panel with a few common factors and regress GDP on them. Principal components regression
(PCR) is the static version (the Stock-Watson "diffusion index").

PCR and ridge are close relatives. With $X = UDV^\\top$:

| | weight on principal direction $j$ |
|---|---|
| OLS | 1 |
| PCR with $k$ components | 1 for $j \\le k$, 0 otherwise |
| Ridge | $d_j^2/(d_j^2+\\lambda)$ |

PCR is a hard cut; ridge is a smooth taper. Both assume the signal sits in the high-variance
directions of $X$. The lasso assumes something different: that a few **original** variables matter.
"""),
    SETUP,
    GDP_DATA,
    code("""
from sklearn.decomposition import PCA
ind_cols = [j for j, n in enumerate(names) if not n.startswith("GDP_")]
Z = Standardiser().fit(X.values).transform(X.values)[:, ind_cols]
pca = PCA().fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.arange(1, 31), cum[:30], "o-")
plt.xlabel("number of components"); plt.ylabel("share of variance explained")
plt.title(f"{len(ind_cols)} indicator features: first 5 components explain {cum[4]:.0%}");
"""),
    code("""
d = np.linalg.svd(Z, compute_uv=False)
j = np.arange(1, 41)
plt.plot(j, np.where(j <= 5, 1, 0), drawstyle="steps-mid", label="PCR, k = 5")
for lam in [10, 100, 1000]:
    plt.plot(j, d[:40]**2 / (d[:40]**2 + lam), label=f"ridge, lambda = {lam}")
plt.xlabel("principal direction"); plt.ylabel("weight"); plt.legend()
plt.title("PCR truncates, ridge tapers");
"""),
    md("""
## Which wins?
It depends on the structure of the truth:

* **Dense, diffuse signal** (many small effects driven by common shocks): factors and ridge do well.
* **Sparse signal** (a few series carry the information): the lasso family does well.

GDP nowcasting sits in between: a common business cycle (dense) plus a few hard measures of
output, such as the Index of Services and Index of Production, which are close to the target itself
(sparse). The main project's backtest (notebooks/04) finds both. Early in the quarter, before any
output data exist, the factor model and ridge are the most accurate. Once output data arrive, a
two-indicator bridge equation beats every penalised model.

**Key takeaways**
* Ridge and PCR both shrink low-variance directions; ridge does it smoothly.
* Factor models are interpretable as common cycles; the lasso as a short list of indicators.
* Combining the two views often beats either (see the forecast combination in the main results).
"""),
]

# ----------------------------------------------------------------------------- 08
L08 = [
    md("""
# 8. Pitfalls: instability, inference and leakage

Regularised regression is easy to fit and easy to misread. This notebook covers the mistakes that
come up in practice and in interviews.
"""),
    SETUP,
    md("""
## 1. Lasso selection is unstable with correlated predictors
Refit on bootstrap samples and count how often each variable is selected.
"""),
    GDP_DATA,
    code("""
from sklearn.linear_model import Lasso
st = Standardiser().fit(X.values); Z = st.transform(X.values); yy = y.values
counts = np.zeros(Z.shape[1]); B_ = 100
for b in range(B_):
    idx = rng.choice(len(yy), len(yy), replace=True)
    counts += Lasso(alpha=0.03, max_iter=20000).fit(Z[idx], yy[idx]).coef_ != 0
freq = pd.Series(counts / B_, index=names).sort_values(ascending=False)
print("Bootstrap selection frequency (lambda fixed at 0.03):")
print(freq.head(15).round(2).to_string())
print(f"\\nfeatures selected in >90% of resamples: {(freq > 0.9).sum()};"
      f" in 10-90%: {((freq > 0.1) & (freq <= 0.9)).sum()}")
"""),
    md("""
No single feature is selected in more than 90% of resamples. The real-time output lags come
closest (70-90%), and dozens of features flicker in and out. **Stability selection**
(Meinshausen and Bühlmann, 2010) formalises this: keep variables selected in a high share of
subsamples. Report selection frequencies, not a single selected set.

## 2. A selected variable is not "significant"
After the lasso picks variables, refitting OLS on them and reading the usual p-values is invalid:
the same data chose the variables, so the p-values are far too small. Demonstration with pure noise:
"""),
    code("""
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
pvals = []
for rep in range(200):
    Xn = rng.normal(size=(100, 50)); yn = rng.normal(size=100)          # no true signal at all
    sel = np.where(LassoCV(cv=5, alphas=30).fit(Xn, yn).coef_ != 0)[0]
    if len(sel):
        pvals += list(sm.OLS(yn, sm.add_constant(Xn[:, sel])).fit().pvalues[1:])
pvals = np.array(pvals)
print(f"share of naive post-selection p-values < 0.05 under pure noise: {np.mean(pvals < 0.05):.0%} (should be 5%)")
"""),
    md("""
Valid alternatives: sample splitting (select on one half, test on the other), selective inference
(Lee et al., 2016), or the debiased/desparsified lasso. For forecasting, the honest test is
out-of-sample accuracy, which is what the main project reports.

## 3. Leakage through preprocessing
Standardising (or imputing, or choosing variables) using the whole sample, then backtesting, lets
future information in. In this project the `Standardiser` is fitted inside every backtest origin
on that origin's training rows only.

## 4. Reading coefficients under correlation
A coefficient is the effect of a feature *holding the others fixed*. With 236 correlated features
that is rarely meaningful. Interpret groups (selection frequency by indicator group) and
contributions to the forecast, not single coefficients.

## 5. Look-ahead bias in the data themselves
Macroeconomic data are revised. Evaluating against today's GDP figures, or using today's vintage of
the indicators, flatters a model. Today's monthly GDP series, for example, aggregates almost
exactly to today's quarterly GDP; using it to "nowcast" past quarters borrows years of later
revisions. This project evaluates against the **first published** estimate from the ONS real-time
database, reads the output indices from the ONS **revisions triangles** (each month's figure as it
was published at the time), and measures the look-ahead bias directly by re-running the backtest
with latest-vintage output data.

**Key takeaways**
* Report selection frequencies, not a single selected list.
* No naive p-values after selection.
* Fit every preprocessing step inside the backtest loop.
* Interpret groups and contributions, not individual coefficients.
"""),
]

README = """# Learn: regularised regression, from basics to advanced

Eight notebooks that build the ideas used in this project. Each mixes theory, from-scratch code and
the project's real UK data.

| # | Notebook | What you learn |
|---|---|---|
| 1 | [Why regularise](01_why_regularise.ipynb) | OLS overfitting when p ≈ n, bias-variance, the nowcasting problem |
| 2 | [Ridge](02_ridge.ipynb) | Closed form, SVD shrinkage, effective df, standardisation, correlated predictors |
| 3 | [Lasso](03_lasso.ipynb) | Soft-thresholding, coordinate descent from scratch, paths, weaknesses |
| 4 | [Elastic net](04_elastic_net.ipynb) | The grouping effect, tuning two parameters, selection stability on the real data |
| 5 | [Choosing lambda](05_choosing_lambda.ipynb) | Time-series CV vs shuffled CV, the 1-SE rule, information criteria |
| 6 | [Adaptive and group lasso](06_adaptive_and_group_lasso.ipynb) | Oracle property, group selection, FISTA |
| 7 | [Regularisation vs factor models](07_regularisation_vs_factors.ipynb) | PCR vs ridge, dense vs sparse signals |
| 8 | [Pitfalls](08_pitfalls.ipynb) | Selection instability, post-selection inference, leakage, revisions |

Run them from this folder after `python scripts/run_pipeline.py --data` has built `data/processed/`.

**Further reading**
* Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, ch. 3 (free online).
* Hastie, Tibshirani and Wainwright, *Statistical Learning with Sparsity* (free online).
* Zou (2006) adaptive lasso; Yuan and Lin (2006) group lasso; Zou and Hastie (2005) elastic net.
* Bańbura, Giannone, Modugno and Reichlin (2013), "Now-casting and the real-time data flow".
"""

NOTEBOOKS = {
    "01_why_regularise.ipynb": L01, "02_ridge.ipynb": L02, "03_lasso.ipynb": L03,
    "04_elastic_net.ipynb": L04, "05_choosing_lambda.ipynb": L05,
    "06_adaptive_and_group_lasso.ipynb": L06, "07_regularisation_vs_factors.ipynb": L07,
    "08_pitfalls.ipynb": L08,
}

if __name__ == "__main__":
    only = sys.argv[1:]
    LEARN.mkdir(exist_ok=True)
    (LEARN / "README.md").write_text(README)
    for fname, cells in NOTEBOOKS.items():
        if only and not any(o in fname for o in only):
            continue
        write_and_run(cells, LEARN / fname)
