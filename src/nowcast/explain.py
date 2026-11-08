"""Decompose a nowcast into contributions from indicator groups.

For a linear model on standardised features, nowcast = intercept + sum_j b_j z_j.
The contribution of group G is sum_{j in G} b_j z_j: how far that group's latest
data move the nowcast away from the model's baseline (the intercept, which is
close to average growth). Contributions are exact for every linear model here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_names, group_of, main_spec, origin_month, training_mask
from .models import Standardiser


def standardised_row(B, q: pd.Period, h: int, feature_filter=main_spec, exclude_covid=True):
    X = B["X"][h]
    T = origin_month(q, h)
    names = feature_names()
    cols = [j for j, n in enumerate(names) if feature_filter(n)]
    mask = training_mask(X.index, B["gdp"], T, exclude_covid=exclude_covid)
    st = Standardiser().fit(X.loc[X.index[mask]].values[:, cols])
    z = np.zeros(len(names))
    z[cols] = st.transform(X.loc[[q]].values[:, cols])[0]
    return z


def contributions(B, q: pd.Period, h: int, coef: np.ndarray) -> pd.Series:
    z = standardised_row(B, q, h)
    names = feature_names()
    c = pd.Series(coef * z, index=names)
    return c.groupby([group_of(n) for n in names]).sum()
