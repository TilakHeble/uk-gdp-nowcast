"""Download and parse the raw data.

* ONS monthly indicator datasets (full time-series CSVs)
* Bank of England database series (monthly)
* ONS real-time database for GDP (ABMI): every published estimate of
  quarterly GDP since 1961, one column per publication ("vintage")
* ONS revisions triangles for the Index of Production, Index of Manufacturing,
  Index of Services and monthly GDP: every published estimate of monthly
  growth, one row per publication month
"""
from __future__ import annotations

import zipfile

import io
import re
import time

import numpy as np
import pandas as pd
import requests

from .config import (GDP_RTD_PAGE, INDICATORS, ONS_DATASET_URLS, PROCESSED, RAW)

ONS = "https://www.ons.gov.uk"
BOE = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
HEADERS = {"User-Agent": "Mozilla/5.0 (uk-gdp-nowcast research project)"}
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# --------------------------------------------------------------------------- download
def _get(url, params=None, retries=6, timeout=120):
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"download failed: {url}")


def download_ons_datasets():
    out = RAW / "ons"
    out.mkdir(parents=True, exist_ok=True)
    for ds, uri in ONS_DATASET_URLS.items():
        r = _get(f"{ONS}/file", params={"uri": uri})
        (out / f"{ds}.csv").write_bytes(r.content)
        print(f"ONS {ds}: {len(r.content)/1e6:.1f} MB")


def download_gdp_realtime_database():
    """Fetch the latest edition of the ONS real-time database for GDP (ABMI)."""
    page = _get(ONS + GDP_RTD_PAGE).text
    links = re.findall(r'href="(/file\?uri=[^"]+realtimedatabaseabmi\.xlsx)"', page)
    if not links:
        raise RuntimeError("no real-time database link found")
    r = _get(ONS + links[0].replace("&amp;", "&"))
    path = RAW / "ons" / "gdp_realtime_database_abmi.xlsx"
    path.write_bytes(r.content)
    print("GDP real-time database:", links[0].split("/")[-2])


def _boe_series(code, start=1990, end=None):
    """The BoE endpoint intermittently refuses long ranges, so fetch 5-year chunks."""
    end = end or pd.Timestamp.today().year
    parts = []
    for a in range(start, end + 1, 5):
        b = min(a + 4, end)
        params = {"csv.x": "yes", "Datefrom": f"01/Jan/{a}",
                  "Dateto": f"31/Dec/{b}" if b < end else "now",
                  "SeriesCodes": code, "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N"}
        for k in range(6):
            try:
                r = requests.get(BOE, params=params, headers=HEADERS, timeout=60)
                if r.text.startswith("DATE"):
                    parts.append(pd.read_csv(io.StringIO(r.text)))
                    break
            except requests.RequestException:
                pass
            time.sleep(2 + 3 * k)
        else:
            # a silently missing chunk would leave a hole in the series
            raise RuntimeError(f"BoE series {code}: chunk {a}-{b} unavailable, rerun")
    if not parts:
        raise RuntimeError(f"BoE series {code} unavailable")
    d = pd.concat(parts).drop_duplicates()
    d["DATE"] = pd.to_datetime(d["DATE"], format="%d %b %Y")
    return d.set_index("DATE")[code].sort_index()


def download_boe():
    out = RAW / "boe"
    out.mkdir(parents=True, exist_ok=True)
    codes = [i.code for i in INDICATORS if i.source == "boe"]
    frames = {c: _boe_series(c) for c in codes}
    df = pd.DataFrame(frames)
    df.index = df.index.to_period("M")
    df.to_csv(out / "boe_monthly.csv", index_label="month")
    print("BoE:", df.shape, df.index.min(), "-", df.index.max())


REALTIME_PAGES = {
    "mgdp": "/economy/grossdomesticproductgdp/datasets/revisionstrianglesformonthlygdp",
    "ios": "/economy/economicoutputandproductivity/output/datasets/indexofservicesrevisionstriangles",
    "iop": "/economy/economicoutputandproductivity/output/datasets/revisionstotheindexofproduction",
}
# (series id, file-name pattern inside the zip, sheet, label column)
REALTIME_FILES = [
    ("RT_IOP", "Revision to the index of Production - 1 Month Growth", "triangle", 1),
    ("RT_IOM", "Revision to the index of Manufacturing - 1 Month Growth", "triangle", 1),
    ("RT_IOS", "Revisions Triangles IOS (M on M)", "triangle", 1),
    ("RT_MGDP", "MGDP Revision triangle (M on M)", "Estimate", 0),
]


def download_realtime_triangles():
    out = RAW / "ons" / "realtime"
    out.mkdir(parents=True, exist_ok=True)
    for key, page in REALTIME_PAGES.items():
        html = _get(ONS + page).text
        link = re.findall(r'href="(/file\?uri=[^"]+\.zip)"', html)[0].replace("&amp;", "&")
        r = _get(ONS + link)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(out / key)
        print("revisions triangle:", key, link.split("/")[-1])


def _find_file(pattern):
    hits = [p for p in (RAW / "ons" / "realtime").rglob("*") if pattern in p.name]
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def _to_month(x):
    if isinstance(x, (pd.Timestamp,)) or hasattr(x, "year"):
        return pd.Period(pd.Timestamp(x), freq="M")
    s = str(x).strip()
    if re.match(r"^\d{4} [A-Z][a-z]{2}$", s):
        return pd.Period(pd.to_datetime(s, format="%Y %b"), freq="M")
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return pd.Period(pd.Timestamp(s[:10]), freq="M")
    return None


def parse_triangle(path, sheet, labcol) -> pd.DataFrame:
    """Long table (vintage, month, value): monthly growth as published in each vintage.

    Rows of the sheet are publication months; columns are reference months.
    """
    d = pd.read_excel(path, sheet, header=None)
    lab = d.iloc[:, labcol]
    r = lab.astype(str).str.contains("Relating").idxmax()
    months = [_to_month(x) for x in d.iloc[r, labcol + 1:]]
    recs = []
    for i in range(r + 1, len(d)):
        v = _to_month(d.iloc[i, labcol])
        if v is None:
            continue
        vals = pd.to_numeric(d.iloc[i, labcol + 1:], errors="coerce").values
        for m, x in zip(months, vals):
            if m is not None and np.isfinite(x):
                recs.append((v, m, float(x)))
    long = pd.DataFrame(recs, columns=["vintage", "month", "value"])
    # duplicated vintage labels / month columns occur in the IoS file: keep the last
    return long.drop_duplicates(["vintage", "month"], keep="last")


def build_realtime_triangles() -> pd.DataFrame:
    frames = []
    for sid, pattern, sheet, labcol in REALTIME_FILES:
        t = parse_triangle(_find_file(pattern), sheet, labcol)
        t.insert(0, "series", sid)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def load_realtime() -> pd.DataFrame:
    t = pd.read_csv(PROCESSED / "realtime_triangles_long.csv")
    t["vintage"] = pd.PeriodIndex(t["vintage"], freq="M")
    t["month"] = pd.PeriodIndex(t["month"], freq="M")
    return t


def download_all():
    download_ons_datasets()
    download_gdp_realtime_database()
    download_realtime_triangles()
    download_boe()


# --------------------------------------------------------------------------- parse
def read_ons_monthly(dataset: str) -> pd.DataFrame:
    """Return the monthly part of an ONS time-series CSV: index = Period[M], columns = CDID."""
    d = pd.read_csv(RAW / "ons" / f"{dataset}.csv", header=None, low_memory=False, dtype=str)
    cdid = d.iloc[1, 1:].tolist()
    lab = d[0].astype(str).str.strip()
    m = lab.str.match(r"^\d{4} [A-Z]{3}$")
    vals = d.loc[m, 1:]
    vals.columns = cdid
    vals.index = pd.PeriodIndex(pd.to_datetime(lab[m], format="%Y %b"), freq="M")
    vals = vals.apply(pd.to_numeric, errors="coerce")
    return vals.loc[:, ~vals.columns.duplicated()]


def build_indicator_panel() -> pd.DataFrame:
    """Raw (untransformed) monthly panel of every latest-vintage catalogue indicator.

    Real-time indicators (source "rt") are not in the panel: they are read
    vintage by vintage from the revisions triangles.
    """
    cols = {}
    for ds in sorted({i.source for i in INDICATORS if i.source not in ("boe", "derived", "rt")}):
        frame = read_ons_monthly(ds)
        for ind in INDICATORS:
            if ind.source == ds:
                cols[ind.code] = frame[ind.code]
    boe = pd.read_csv(RAW / "boe" / "boe_monthly.csv", index_col=0)
    boe.index = pd.PeriodIndex(boe.index, freq="M")
    for c in boe.columns:
        cols[c] = boe[c]
    panel = pd.DataFrame(cols).sort_index()
    panel["SPREAD"] = panel["IUMAMNPY"] - panel["IUMABEDR"]
    panel = panel.loc["1996-01":]
    panel = panel[[i.code for i in INDICATORS if i.source != "rt"]]
    return panel


def _parse_vintage(label: str) -> pd.Period | None:
    s = str(label).lower().replace("\n", " ")
    s = s.replace("feb-772", "feb-78")   # typo in the ONS file (between Jan-78 and Mar-78)
    m = re.match(r"\s*([a-z]{3})[a-z]*\s*-\s*(\d{2,4})", s)
    if not m:
        return None
    mon = MONTHS[m.group(1)]
    yr = int(m.group(2))
    if yr < 100:
        yr += 1900 if yr > 50 else 2000
    return pd.Period(year=yr, month=mon, freq="M")


def build_gdp_vintages() -> pd.DataFrame:
    """Long table of GDP levels: quarter, vintage (publication month), level (CVM, £m)."""
    path = RAW / "ons" / "gdp_realtime_database_abmi.xlsx"
    xl = pd.ExcelFile(path)
    recs = []
    for sheet in xl.sheet_names:
        if not re.match(r"^\d{4}", sheet):
            continue
        d = pd.read_excel(xl, sheet, header=None)
        hdr_row = d.index[d[0].astype(str).str.contains("Publication date", na=False)][0]
        headers = d.iloc[hdr_row]
        body = d.iloc[hdr_row + 1:]
        q = body[0].astype(str).str.strip()
        okq = q.str.match(r"^Q[1-4] \d{4}$")
        quarters = [pd.Period(f"{s[3:]}{s[:2]}", freq="Q") for s in q[okq]]
        for j in range(1, d.shape[1]):
            v = _parse_vintage(headers[j])
            if v is None:
                continue
            col = pd.to_numeric(body.loc[okq, j], errors="coerce").values
            for qq, val in zip(quarters, col):
                if np.isfinite(val):
                    recs.append((qq, v, float(val)))
    long = pd.DataFrame(recs, columns=["quarter", "vintage", "level"])
    # a vintage label can appear in two sheets (overlap) - identical values; keep last
    long = long.drop_duplicates(["quarter", "vintage"], keep="last")
    return long.sort_values(["vintage", "quarter"]).reset_index(drop=True)


def gdp_growth_by_vintage(long: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = quarter, columns = vintage, values = q/q growth (%)."""
    wide = long.pivot(index="quarter", columns="vintage", values="level").sort_index()
    return 100 * (wide / wide.shift(1) - 1)


def first_and_latest(growth: pd.DataFrame) -> pd.DataFrame:
    first = growth.apply(lambda r: r.dropna().iloc[0] if r.notna().any() else np.nan, axis=1)
    first_v = growth.apply(lambda r: r.dropna().index[0] if r.notna().any() else pd.NaT, axis=1)
    latest = growth.iloc[:, -1]
    out = pd.DataFrame({"first_estimate": first, "first_vintage": first_v, "latest_estimate": latest})
    out["revision"] = out["latest_estimate"] - out["first_estimate"]
    return out


def process_all():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = build_indicator_panel()
    panel.to_csv(PROCESSED / "indicators_raw.csv", index_label="month")
    long = build_gdp_vintages()
    long.to_csv(PROCESSED / "gdp_vintages_long.csv", index=False)
    growth = gdp_growth_by_vintage(long)
    fl = first_and_latest(growth)
    fl.to_csv(PROCESSED / "gdp_first_latest.csv", index_label="quarter")
    rt = build_realtime_triangles()
    rt.to_csv(PROCESSED / "realtime_triangles_long.csv", index=False)
    print("revisions triangles:", rt.groupby("series").vintage.agg(["min", "max", "nunique"]).to_dict("index"))
    print("panel", panel.shape, panel.index.min(), panel.index.max())
    print("vintages", long["vintage"].nunique(), long["vintage"].min(), long["vintage"].max())
    return panel, long, fl


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(PROCESSED / "indicators_raw.csv", index_col=0)
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p


def load_gdp_vintages() -> pd.DataFrame:
    long = pd.read_csv(PROCESSED / "gdp_vintages_long.csv")
    long["quarter"] = pd.PeriodIndex(long["quarter"], freq="Q")
    long["vintage"] = pd.PeriodIndex(long["vintage"], freq="M")
    return long
