"""Project configuration: paths, the indicator catalogue and the nowcast calendar."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

# Sample design
SAMPLE_START = "1998Q1"      # first quarter used for training (monthly data start Jan 1997)
OOS_START = "2006Q1"         # first quarter nowcast out of sample
# COVID quarters are excluded from *training* (never from evaluation); see docs.
COVID_EXCLUDE = ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"]

# Nowcast origins, as month offsets relative to the target quarter.
# Origin month T = third month of the quarter - 2 + h
#   h=-1 end of the month before the quarter starts (a one-quarter-ahead forecast)
#   h= 0 end of month 1, h=1 end of month 2, h=2 end of month 3 (quarter end)
#   h= 3 end of the month after the quarter (about 2 weeks before ONS's first estimate)
HORIZONS = [-1, 0, 1, 2, 3]
HORIZON_LABELS = {
    -1: "Before quarter",
    0: "End of month 1",
    1: "End of month 2",
    2: "End of month 3",
    3: "Month after quarter",
}
# GDP availability is read from the real-time database itself: a quarter's
# first estimate is known from the end of its publication month (one month
# after the quarter until mid-2018, two months since).
N_MONTH_LAGS = 6             # monthly observations per indicator used as features (two quarters)


@dataclass(frozen=True)
class Indicator:
    code: str        # ONS CDID or Bank of England series code
    source: str      # ONS dataset id (mgdp, diop, ...) or "boe"
    name: str
    group: str
    transform: str   # "dlog" (100 x monthly log change), "diff" (monthly change), "level"
    lag: int         # publication lag in months: last month known at the end of month T is T - lag


def _I(code, source, name, group, transform, lag):
    return Indicator(code, source, name, group, transform, lag)


INDICATORS: list[Indicator] = [
    # Real-time output: monthly growth exactly as first and subsequently published
    # (ONS revisions triangles). Vintage lookup replaces the publication-lag rule;
    # lag 2 is the observed lag in every vintage since 2003.
    _I("RT_IOS", "rt", "Index of Services (real time)", "Output (real time)", "rt", 2),
    _I("RT_IOP", "rt", "Index of Production (real time)", "Output (real time)", "rt", 2),
    _I("RT_IOM", "rt", "Index of Manufacturing (real time)", "Output (real time)", "rt", 2),
    # Monthly GDP and its sectors, LATEST vintage (ONS, ~6 weeks after the month).
    # Used only in the look-ahead experiment: these embed later revisions.
    _I("ECY2", "mgdp", "Monthly GDP (GVA)", "Output (latest vintage)", "dlog", 2),
    _I("ECY3", "mgdp", "Agriculture", "Output (latest vintage)", "dlog", 2),
    _I("ECY4", "mgdp", "Production", "Output (latest vintage)", "dlog", 2),
    _I("ECY5", "mgdp", "Mining and quarrying", "Output (latest vintage)", "dlog", 2),
    _I("ECY6", "mgdp", "Manufacturing", "Output (latest vintage)", "dlog", 2),
    _I("ECY7", "mgdp", "Electricity and gas", "Output (latest vintage)", "dlog", 2),
    _I("ECY8", "mgdp", "Water and waste", "Output (latest vintage)", "dlog", 2),
    _I("ECY9", "mgdp", "Construction", "Output (latest vintage)", "dlog", 2),
    _I("ECYC", "mgdp", "Services", "Output (latest vintage)", "dlog", 2),
    _I("ECYD", "mgdp", "Wholesale and retail", "Output (latest vintage)", "dlog", 2),
    _I("ECYG", "mgdp", "Transport and storage", "Output (latest vintage)", "dlog", 2),
    _I("ECYH", "mgdp", "Accommodation and food", "Output (latest vintage)", "dlog", 2),
    _I("ECYI", "mgdp", "Information and communication", "Output (latest vintage)", "dlog", 2),
    _I("ECYJ", "mgdp", "Finance and insurance", "Output (latest vintage)", "dlog", 2),
    _I("ECYK", "mgdp", "Real estate", "Output (latest vintage)", "dlog", 2),
    _I("ECYL", "mgdp", "Professional and scientific", "Output (latest vintage)", "dlog", 2),
    _I("ECYP", "mgdp", "Administrative services", "Output (latest vintage)", "dlog", 2),
    _I("ECYQ", "mgdp", "Public administration", "Output (latest vintage)", "dlog", 2),
    _I("ECYR", "mgdp", "Education", "Output (latest vintage)", "dlog", 2),
    _I("ECYS", "mgdp", "Health and social work", "Output (latest vintage)", "dlog", 2),
    _I("ECYT", "mgdp", "Arts and recreation", "Output (latest vintage)", "dlog", 2),
    _I("ECYU", "mgdp", "Other services", "Output (latest vintage)", "dlog", 2),
    # Manufacturing detail (Index of Production)
    _I("K22B", "diop", "Mfg: food, drink, tobacco", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K22P", "diop", "Mfg: textiles and clothing", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K22T", "diop", "Mfg: wood, paper, printing", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K22X", "diop", "Mfg: coke and refined petroleum", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K22Z", "diop", "Mfg: chemicals", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K239", "diop", "Mfg: pharmaceuticals", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23B", "diop", "Mfg: rubber, plastics, minerals", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23G", "diop", "Mfg: basic metals", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23N", "diop", "Mfg: computers and electronics", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23P", "diop", "Mfg: electrical equipment", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23R", "diop", "Mfg: machinery", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23T", "diop", "Mfg: transport equipment", "Manufacturing detail (latest vintage)", "dlog", 2),
    _I("K23Z", "diop", "Mfg: other and repair", "Manufacturing detail (latest vintage)", "dlog", 2),
    # Retail sales (published ~3 weeks after the month)
    _I("J5EK", "drsi", "Retail sales volume", "Retail", "dlog", 1),
    _I("J467", "drsi", "Retail sales ex fuel", "Retail", "dlog", 1),
    _I("J5DZ", "drsi", "Online (non-store) retail", "Retail", "dlog", 1),
    _I("JO5A", "drsi", "Automotive fuel sales", "Retail", "dlog", 1),
    # Trade (published with monthly GDP)
    _I("BQKQ", "mret", "Goods exports volume", "Trade", "dlog", 2),
    _I("BQKO", "mret", "Goods imports volume", "Trade", "dlog", 2),
    _I("IKBE", "mret", "Services exports volume", "Trade", "dlog", 2),
    _I("IKBF", "mret", "Services imports volume", "Trade", "dlog", 2),
    _I("BOXV", "mret", "Goods exports ex oil and erratics", "Trade", "dlog", 2),
    _I("BPIV", "mret", "Goods imports ex oil and erratics", "Trade", "dlog", 2),
    _I("ENDZ", "mret", "Car exports", "Trade", "dlog", 2),
    _I("ENGT", "mret", "Car imports", "Trade", "dlog", 2),
    _I("ENEI", "mret", "Capital goods exports", "Trade", "dlog", 2),
    _I("ENGW", "mret", "Capital goods imports", "Trade", "dlog", 2),
    # Labour market
    _I("MGSX", "lms", "Unemployment rate", "Labour market", "diff", 3),
    _I("LF24", "lms", "Employment rate (16-64)", "Labour market", "diff", 3),
    _I("MGRZ", "lms", "Employment level", "Labour market", "dlog", 3),
    _I("YBUS", "lms", "Total weekly hours worked", "Labour market", "dlog", 3),
    _I("LF2S", "lms", "Inactivity rate (16-64)", "Labour market", "diff", 3),
    _I("YCBW", "lms", "Workers with second jobs", "Labour market", "dlog", 3),
    _I("AP2Y", "lms", "Vacancies", "Labour market", "dlog", 2),
    _I("KAB9", "lms", "Average weekly earnings", "Labour market", "dlog", 2),
    # Prices
    _I("D7BT", "mm23", "CPI index", "Prices", "dlog", 1),
    _I("D7F5", "mm23", "CPI services index", "Prices", "dlog", 1),
    _I("D7G7", "mm23", "CPI inflation (annual)", "Prices", "diff", 1),
    _I("DKO8", "mm23", "Core CPI inflation (annual)", "Prices", "diff", 1),
    _I("CZBH", "mm23", "RPI inflation (annual)", "Prices", "diff", 1),
    # Money, credit and financial markets (Bank of England)
    _I("IUMABEDR", "boe", "Bank Rate", "Money and markets", "diff", 0),
    _I("IUMSNPY", "boe", "5-year gilt yield", "Money and markets", "diff", 0),
    _I("IUMAMNPY", "boe", "10-year gilt yield", "Money and markets", "diff", 0),
    _I("SPREAD", "derived", "Term spread (10y gilt - Bank Rate)", "Money and markets", "level", 0),
    _I("XUMABK67", "boe", "Sterling effective exchange rate", "Money and markets", "dlog", 0),
    _I("XUMAUSS", "boe", "Sterling-dollar rate", "Money and markets", "dlog", 0),
    _I("IUMBV34", "boe", "2-year fixed mortgage rate (75% LTV)", "Money and markets", "diff", 1),
    _I("LPMVTVX", "boe", "Mortgage approvals for house purchase", "Money and markets", "dlog", 1),
    _I("LPMAUYN", "boe", "M4 money supply", "Money and markets", "dlog", 1),
]

INDICATOR_BY_CODE = {i.code: i for i in INDICATORS}

ONS_DATASET_URLS = {
    "mgdp": "/economy/grossdomesticproductgdp/datasets/gdpmonthlyestimateuktimeseriesdataset/current/mgdp.csv",
    "diop": "/economy/economicoutputandproductivity/output/datasets/indexofproduction/current/diop.csv",
    "drsi": "/businessindustryandtrade/retailindustry/datasets/retailsales/current/drsi.csv",
    "mret": "/economy/nationalaccounts/balanceofpayments/datasets/tradeingoodsmretsallbopeu2013timeseriesspreadsheet/current/mret.csv",
    "lms": "/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/labourmarketstatistics/current/lms.csv",
    "mm23": "/economy/inflationandpriceindices/datasets/consumerpriceindices/current/mm23.csv",
}
GDP_RTD_PAGE = "/economy/grossdomesticproductgdp/datasets/realtimedatabaseforukgdpabmi"

# The live nowcast is made with data published by the end of this month.
LIVE_ORIGIN = "2026-10"

LATEST_VINTAGE_GROUPS = {"Output (latest vintage)", "Manufacturing detail (latest vintage)"}
REALTIME_GROUPS = {"Output (real time)"}
