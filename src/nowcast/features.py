"""Turn the raw monthly panel and the GDP vintages into nowcasting design matrices.

The information set
-------------------
A nowcast for target quarter Q at horizon h is made at the end of origin month

    T(Q, h) = first month of Q + h        (h = -1, 0, 1, 2, 3)

At that moment:

* indicator i is known up to month T - lag_i (its publication lag), and
* GDP is known as it stood in the latest ONS vintage published in or before T.

Three kinds of data enter:

* **real-time output** (Index of Services, Production, Manufacturing): monthly
  growth exactly as published in the latest vintage available at T, read from
  the ONS revisions triangles;
* **latest-vintage indicators** (retail, trade, labour market, prices, money and
  markets): today's vintage, with the publication calendar respected. Prices
  and financial-market data are never or barely revised; retail, trade and
  labour data are revised modestly;
* **latest-vintage output** (monthly GDP and its sectors, manufacturing detail):
  embed years of later revisions, so they are used only in the look-ahead
  experiment that measures how much revised data flatter a backtest.

Features (U-MIDAS)
------------------
For each indicator the six most recent monthly observations available at T,
transformed to stationarity, are separate features: ``CODE_l0`` is the latest
available month, ``CODE_l5`` the one five months earlier. Because publication
lags are fixed, ``CODE_l0`` refers to the same month relative to the quarter for
every row of a horizon-h model, so one model per horizon is estimated.

Two real-time GDP features are added: growth in Q-2 (always published by T) and
growth in Q-1 (missing when not yet published; imputed by the model pipeline).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import (COVID_EXCLUDE, INDICATOR_BY_CODE, INDICATORS, LATEST_VINTAGE_GROUPS,
                     N_MONTH_LAGS, REALTIME_GROUPS, SAMPLE_START)
from . import data


def transform_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply each indicator's stationarity transform (monthly frequency)."""
    out = {}
    for ind in INDICATORS:
        if ind.source == "rt":
            continue
        s = panel[ind.code].astype(float)
        if ind.transform == "dlog":
            out[ind.code] = 100 * np.log(s).diff()
        elif ind.transform == "diff":
            out[ind.code] = s.diff()
        else:
            out[ind.code] = s
    return pd.DataFrame(out, index=panel.index)


def origin_month(quarter: pd.Period, h: int) -> pd.Period:
    return quarter.asfreq("M", "start") + h


def feature_names() -> list[str]:
    names = [f"{i.code}_l{j}" for i in INDICATORS for j in range(N_MONTH_LAGS)]
    return names + ["GDP_q2", "GDP_q1"]


def feature_groups() -> np.ndarray:
    """Integer group id per feature: one group per indicator (all its lags), one per GDP lag."""
    g = [k for k, _ in enumerate(INDICATORS) for _ in range(N_MONTH_LAGS)]
    return np.array(g + [len(INDICATORS), len(INDICATORS) + 1])


class RealTimeIndicators:
    """Monthly growth of the real-time output series as published at any month."""

    def __init__(self, long: pd.DataFrame):
        self.wide = {}
        for sid, g in long.groupby("series"):
            w = g.pivot(index="vintage", columns="month", values="value").sort_index()
            self.wide[sid] = w

    def value(self, sid: str, T: pd.Period, month: pd.Period) -> float:
        w = self.wide[sid]
        pos = w.index.searchsorted(T, side="right") - 1
        if pos < 0 or month not in w.columns:
            return np.nan
        return w.iat[pos, w.columns.get_loc(month)]


class GDPVintages:
    """Real-time access to GDP growth as published at any month."""

    def __init__(self, long: pd.DataFrame):
        self.long = long
        self.wide = long.pivot(index="quarter", columns="vintage", values="level").sort_index()
        self.vintages = self.wide.columns  # PeriodIndex[M], sorted
        g = 100 * (self.wide / self.wide.shift(1) - 1)
        self.growth = g
        first = {}
        first_v = {}
        for q, row in g.iterrows():
            r = row.dropna()
            if len(r):
                first[q] = r.iloc[0]
                first_v[q] = r.index[0]
        self.first = pd.Series(first)
        self.first_vintage = pd.Series(first_v)
        self.latest = g.iloc[:, -1]

    @lru_cache(maxsize=None)
    def known_at(self, month: pd.Period) -> pd.Series:
        """GDP growth by quarter as it stood in the latest vintage published by ``month``."""
        ok = self.vintages[self.vintages <= month]
        if len(ok) == 0:
            return pd.Series(dtype=float)
        return self.growth[ok[-1]].dropna()

    def published_by(self, quarter: pd.Period, month: pd.Period) -> bool:
        fv = self.first_vintage.get(quarter)
        return fv is not None and fv <= month


def build_design(panel_t: pd.DataFrame, gdp: GDPVintages, h: int,
                 quarters: pd.PeriodIndex, rt: RealTimeIndicators) -> pd.DataFrame:
    """Design matrix for horizon h: one row per target quarter, columns = feature_names()."""
    rows = []
    for q in quarters:
        T = origin_month(q, h)
        vals = []
        for ind in INDICATORS:
            last = T - ind.lag
            months = [last - j for j in range(N_MONTH_LAGS)]
            if ind.source == "rt":
                vals.extend(rt.value(ind.code, T, m) for m in months)
            else:
                s = panel_t[ind.code]
                vals.extend(s.get(m, np.nan) for m in months)
        known = gdp.known_at(T)
        vals.append(known.get(q - 2, np.nan))
        vals.append(known.get(q - 1, np.nan))
        rows.append(vals)
    return pd.DataFrame(rows, index=quarters, columns=feature_names(), dtype=float)


def build_all(horizons, end_quarter: pd.Period | None = None):
    """Precompute design matrices for every horizon and the target series.

    Returns dict with:
      X[h]          DataFrame quarters x features
      first         first-estimate growth by quarter
      latest        latest-vintage growth by quarter
      first_vintage publication month of the first estimate
      gdp           the GDPVintages object
    """
    panel = data.load_panel()
    panel_t = transform_panel(panel)
    gdp = GDPVintages(data.load_gdp_vintages())
    rt = RealTimeIndicators(data.load_realtime())
    if end_quarter is None:
        end_quarter = (panel.index.max() + 3).asfreq("Q")
    quarters = pd.period_range(SAMPLE_START, end_quarter, freq="Q")
    X = {h: build_design(panel_t, gdp, h, quarters, rt) for h in horizons}
    return dict(X=X, first=gdp.first, latest=gdp.latest, rt=rt,
                first_vintage=gdp.first_vintage, gdp=gdp, panel=panel, panel_t=panel_t)


def group_of(name: str) -> str:
    """Indicator group of a feature name (GDP lags -> "GDP")."""
    if name.startswith("GDP_"):
        return "GDP"
    return INDICATOR_BY_CODE[name.rsplit("_l", 1)[0]].group


def main_spec(name: str) -> bool:
    """Main specification: everything except latest-vintage output data."""
    return group_of(name) not in LATEST_VINTAGE_GROUPS


def lookahead_spec(name: str) -> bool:
    """Look-ahead experiment: latest-vintage output data instead of real-time output."""
    return group_of(name) not in REALTIME_GROUPS


def soft_spec(name: str) -> bool:
    """No official output or trade data at all: surveys-free 'early' information only."""
    return group_of(name) not in LATEST_VINTAGE_GROUPS | REALTIME_GROUPS | {"Trade"}


def training_mask(quarters: pd.PeriodIndex, gdp: GDPVintages, T: pd.Period,
                  exclude_covid: bool = True) -> np.ndarray:
    """Quarters usable for training at origin T: first estimate published by T."""
    covid = {pd.Period(c, freq="Q") for c in COVID_EXCLUDE}
    return np.array([gdp.published_by(q, T) and not (exclude_covid and q in covid)
                     for q in quarters])
