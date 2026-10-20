import streamlit as st
import pandas as pd
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
import yfinance as yf
import plotly.graph_objects as go
import datetime
import concurrent.futures

# ----------------------------------------------------
# API Keys / Config & Session Pooling (v5 Optimization)
# ----------------------------------------------------
# For Streamlit Cloud deployment, set API keys in Secrets.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    FRED_API_KEY = "086fe9dc602ab52e1382533a7e6890c9"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# e-Stat (Official Statistics of Japan) Application ID
try:
    ESTAT_APP_ID = st.secrets["ESTAT_APP_ID"]
except Exception:
    ESTAT_APP_ID = "ddedf4a8b47ec53d8f7ee614fb844a1d2e7e1d17"
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

REQUEST_TIMEOUT = 10  # seconds
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Global-Macro-Dashboard/5.0)"}

# HTTP session reuse and connection pool optimization (Speed up)
http_session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)
http_session.headers.update(REQUEST_HEADERS)


def fred_fetch_series(series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch time-series observations from official FRED API."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    try:
        resp = http_session.get(FRED_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        if not data:
            return pd.DataFrame(columns=["date", "value"])
        df = pd.DataFrame(data)[["date", "value"]]
        df = df[df["value"] != "."]
        df["value"] = df["value"].astype(float)
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


def fred_latest_value(series_id: str, lookback_days: int = 400):
    """Returns the latest observation tuple (date, value) or None."""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=lookback_days)
    df = fred_fetch_series(series_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df.empty:
        return None
    last_row = df.iloc[-1]
    return last_row["date"], float(last_row["value"])


# ----------------------------------------------------
# Official Central Bank & Ministry of Finance Fetchers
# ----------------------------------------------------

def fetch_us_treasury_yields():
    yields = {}
    fred_map = {"USA_2Y": "DGS2", "USA_10Y": "DGS10", "USA_30Y": "DGS30"}
    for key, sid in fred_map.items():
        res = fred_latest_value(sid, lookback_days=30)
        if res is not None:
            yields[key] = round(res[1], 2)

    yf_map = {"USA_10Y": "^TNX", "USA_30Y": "^TYX"}
    for key, ticker in yf_map.items():
        try:
            hist = yf.Ticker(ticker).history(period="3d")
            if not hist.empty:
                yields[key] = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
    return yields


def fetch_japan_mof_yields():
    url = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
    try:
        resp = http_session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        df = pd.read_csv(io.BytesIO(resp.content), encoding="shift_jis", skiprows=1)
        df = df.dropna(how="all")
        if df.empty:
            return {}

        col_2 = next((c for c in df.columns if re.search(r'(?<!\d)2(?!\d)', str(c))), None)
        col_10 = next((c for c in df.columns if re.search(r'(?<!\d)10(?!\d)', str(c))), None)
        col_30 = next((c for c in df.columns if re.search(r'(?<!\d)30(?!\d)', str(c))), None)

        res = {}
        if col_2:
            s = pd.to_numeric(df[col_2], errors="coerce").dropna()
            if not s.empty:
                res["JPN_2Y"] = round(float(s.iloc[-1]), 2)
        if col_10:
            s = pd.to_numeric(df[col_10], errors="coerce").dropna()
            if not s.empty:
                res["JPN_10Y"] = round(float(s.iloc[-1]), 2)
        if col_30:
            s = pd.to_numeric(df[col_30], errors="coerce").dropna()
            if not s.empty:
                res["JPN_30Y"] = round(float(s.iloc[-1]), 2)
        return res
    except Exception:
        return {}


def fetch_ecb_eurozone_yields():
    base_url = "https://data-api.ecb.europa.eu/service/data/YC"
    maturities = {"EURO_2Y": "SR_2Y", "EURO_10Y": "SR_10Y", "EURO_30Y": "SR_30Y"}
    results = {}

    for key, mat in maturities.items():
        try:
            series_key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.{mat}"
            url = f"{base_url}/{series_key}?format=csvdata&lastNObservations=3"
            resp = http_session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if "OBS_VALUE" in df.columns and not df["OBS_VALUE"].dropna().empty:
                latest_val = float(df["OBS_VALUE"].dropna().iloc[-1])
                results[key] = round(latest_val, 2)
        except Exception:
            pass

    if "EURO_10Y" not in results:
        res = fred_latest_value("IRLTLT01EZM156N", lookback_days=120)
        if res is not None:
            results["EURO_10Y"] = round(res[1], 2)

    return results


def fetch_boe_uk_yields():
    results = {}
    try:
        url = (
            "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
            "?symbols=TMBMKGB-02Y,TMBMKGB-10Y,TMBMKGB-30Y"
            "&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json"
        )
        resp = http_session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("QuickQuoteResult", {}).get("QuickQuote", [])
            if isinstance(quotes, dict):
                quotes = [quotes]
            mapping = {
                "TMBMKGB-02Y": "UK_2Y",
                "TMBMKGB-10Y": "UK_10Y",
                "TMBMKGB-30Y": "UK_30Y"
            }
            for q in quotes:
                sym = q.get("symbol")
                val_str = q.get("last")
                if sym in mapping and val_str:
                    try:
                        results[mapping[sym]] = round(float(val_str), 2)
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    if "UK_10Y" not in results:
        try:
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=90)
            url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
            params = {
                "csv.x": "yes",
                "SeriesCodes": "IUDMNPY",
                "UsingCodes": "Y",
                "CSVF": "TN",
                "Datefrom": start.strftime("%d/%b/%Y"),
                "Dateto": end.strftime("%d/%b/%Y"),
            }
            resp = http_session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            if not df.empty:
                results["UK_10Y"] = round(float(df.iloc[-1]["value"]), 2)
        except Exception:
            pass

    return results


def fetch_boe_bank_rate():
    # Determine what was decided at the latest MPC meeting
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=800)
        url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
        params = {
            "csv.x": "yes",
            "SeriesCodes": "IUDBEDR",
            "UsingCodes": "Y",
            "CSVF": "TN",
            "Datefrom": start.strftime("%d/%b/%Y"),
            "Dateto": end.strftime("%d/%b/%Y"),
        }
        resp = http_session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if df.empty:
            return None

        cur = float(df.iloc[-1]["value"])
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        past_meetings = [d for d in boe_schedules if d <= today_str]
        if not past_meetings:
            return cur, None

        latest_meeting_date = pd.to_datetime(past_meetings[-1])
        before_meeting = df[df["date"] < latest_meeting_date]
        if before_meeting.empty:
            return cur, None
        pre_meeting_val = float(before_meeting.iloc[-1]["value"])
        return cur, pre_meeting_val
    except Exception:
        return None


# ----------------------------------------------------
# Central Bank Meeting Schedules
# ----------------------------------------------------
fomc_schedules = ["2026-09-16", "2026-10-28", "2026-12-09", "2027-01-27", "2027-03-17"]
ecb_schedules  = ["2026-10-29", "2026-12-17", "2027-02-04", "2027-03-18"]
boe_schedules  = ["2026-09-17", "2026-11-05", "2026-12-17", "2027-02-04"]
boj_schedules  = [
    "2026-01-23", "2026-03-19", "2026-04-28", "2026-06-16", 
    "2026-07-31", "2026-09-18", "2026-10-30", "2026-12-18", 
    "2027-01-22", "2027-03-19"
]

def get_next_meeting_date(schedule_dates):
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    for date_str in schedule_dates:
        if date_str >= today_str:
            return date_str
    return "TBD"


def fetch_estat_cpi_yoy(debug_log: list = None):
    # Fetch CPI (All items, All Japan, YoY) from e-Stat API
    if debug_log is None:
        debug_log = []

    candidate_tables = [
        ("0004052037", "2025-base CPI"),
        ("0003427113", "2020-base CPI"),
    ]

    for stats_id, label in candidate_tables:
        try:
            params = {
                "appId": ESTAT_APP_ID,
                "statsDataId": stats_id,
                "metaGetFlg": "Y",
                "cntGetFlg": "N",
                "lang": "E"  # Request English metadata
            }
            resp = http_session.get(ESTAT_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()

            root = payload.get("GET_STATS_DATA", {})
            result_status = root.get("RESULT", {})
            if str(result_status.get("STATUS", "0")) not in ("0",):
                debug_log.append(f"[e-Stat/{label}] API error: {result_status.get('ERROR_MSG')}")
                continue

            stat_data = root.get("STATISTICAL_DATA", {})
            class_objs = stat_data.get("CLASS_INF", {}).get("CLASS_OBJ", [])
            if isinstance(class_objs, dict):
                class_objs = [class_objs]

            def find_code(class_id, name_candidates):
                for obj in class_objs:
                    if obj.get("@id") != class_id:
                        continue
                    classes = obj.get("CLASS", [])
                    if isinstance(classes, dict):
                        classes = [classes]
                    for c in classes:
                        cname = str(c.get("@name", "")).lower()
                        if any(nc.lower() in cname for nc in name_candidates):
                            return c.get("@code")
                return None

            cat_code = find_code("cat01", ["all items", "general"])
            area_code = find_code("area", ["all japan", "nationwide", "japan"])
            tab_code = find_code("tab", ["year-on-year", "change from the previous year", "change over the year", "yoy"])

            values = stat_data.get("DATA_INF", {}).get("VALUE", [])
            if isinstance(values, dict):
                values = [values]

            filtered = []
            for v in values:
                if cat_code and v.get("@cat01") != cat_code:
                    continue
                if area_code and v.get("@area") != area_code:
                    continue
                if tab_code and v.get("@tab") != tab_code:
                    continue
                filtered.append(v)

            if not filtered:
                debug_log.append(f"[e-Stat/{label}] no matching data found")
                continue

            filtered.sort(key=lambda v: str(v.get("@time", "")))
            latest = filtered[-1]
            val = float(latest.get("$"))
            debug_log.append(f"[e-Stat/{label}] fetched successfully: {val}% (time={latest.get('@time')})")
            return round(val, 2)

        except Exception as e:
            debug_log.append(f"[e-Stat/{label}] exception: {e}")
            continue

    return None


def _extract_rate_from_boj_text(text: str):
    patterns = [
        r'uncollateralized overnight call rate to remain at around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
        r'uncollateralized overnight call rate.*?around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
        r'rate to remain at around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def fetch_boj_policy_action():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    past_meetings = [d for d in boj_schedules if d <= today_str]
    if not past_meetings:
        return None

    past_meetings.sort(reverse=True)
    extracted = []

    for date_str in past_meetings[:4]:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        url_options = [
            f"https://www.boj.or.jp/en/mopo/mpmdeci/state_{dt.year}/k{dt.strftime('%y%m%d')}a.htm",
            f"https://www.boj.or.jp/en/mopo/mpmdeci/m_{dt.year}/k{dt.strftime('%y%m%d')}a.htm"
        ]
        rate = None
        hit_url = None
        for u in url_options:
            try:
                resp = http_session.get(u, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    val = _extract_rate_from_boj_text(resp.text)
                    if val is not None:
                        rate = val
                        hit_url = u
                        break
            except Exception:
                continue

        if rate is not None:
            extracted.append((rate, hit_url))
            if len(extracted) >= 2:
                break

    if extracted:
        cur_target, cur_url = extracted[0]
        if len(extracted) >= 2:
            prev_target, _ = extracted[1]
            diff = int(round((cur_target - prev_target) * 100))
            action = f"{diff:+} bps" if diff != 0 else "Unchanged"
        else:
            # If the previous meeting's value could not be retrieved
            action = "See official statement"
        return cur_target, action, cur_url

    return None


# Page Configuration
st.set_page_config(
    page_title="Global Macro & Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# WSJ Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .wsj-header {
        font-family: 'Merriweather', serif;
        border-bottom: 2px solid #111;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    
    .wsj-title {
        font-family: 'Merriweather', serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #111111;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .wsj-subtitle {
        font-family: 'Merriweather', serif;
        font-style: italic;
        font-size: 1.05rem;
        color: #555555;
        text-align: center;
        margin-bottom: 15px;
    }

    .report-date {
        text-align: right;
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 15px;
    }

    .rate-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 12px 14px;
        background-color: #fdfdfd;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rate-card-title {
        font-size: 0.80rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 3px;
    }
    .official-link-btn {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: #1a5276;
        background-color: #eaf2f8;
        border: 1px solid #b8d4e8;
        border-radius: 14px;
        padding: 3px 12px;
        text-decoration: none;
        transition: background-color 0.15s ease;
    }
    .official-link-btn:hover {
        background-color: #d6e9f7;
        text-decoration: none;
    }
    .new-badge {
        display: inline-block;
        background-color: #d93025;
        color: #ffffff;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 1px 7px;
        border-radius: 10px;
        vertical-align: middle;
        margin-left: 6px;
        animation: pulse-badge 1.6s infinite;
    }
    @keyframes pulse-badge {
        0% { opacity: 1; }
        50% { opacity: 0.55; }
        100% { opacity: 1; }
    }
    .rate-card-value {
        font-family: 'Merriweather', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #111;
        line-height: 1.2;
    }
    .rate-card-sub {
        font-size: 0.72rem;
        color: #888;
        margin-top: 3px;
        margin-bottom: 8px;
    }
    .rate-card-details {
        border-top: 1px dashed #e0e0e0;
        padding-top: 6px;
        font-size: 0.78rem;
        color: #444;
        text-align: left;
    }
    .rate-detail-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }

    .yield-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 12px;
        background-color: #f8f9fa;
        color: #333 !important;
        margin-top: 8px;
        font-size: 0.80rem;
    }
    .yield-card span, .yield-card b {
        color: #333 !important;
    }
    .yield-card-title {
        font-weight: 700;
        color: #333;
        border-bottom: 1px solid #ddd;
        padding-bottom: 3px;
        margin-bottom: 5px;
    }
    .yield-warning {
        color: #c62828;
        font-weight: 700;
    }

    .market-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 12px 15px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    .market-name {
        font-family: 'Merriweather', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #111;
        border-bottom: 1px solid #eee;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }
    .market-rate {
        font-size: 1.35rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 8px;
    }
    .market-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        padding: 3px 0;
        color: #333;
        border-bottom: 1px dashed #f0f0f0;
    }
    .pos-dev {
        color: #2e7d32;
        font-weight: 600;
    }
    .neg-dev {
        color: #c62828;
        font-weight: 600;
    }

    .report-box {
        border: 1px solid #111;
        background-color: #fcfcf9;
        color: #111 !important;
        padding: 20px;
        margin-top: 20px;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .report-box p, .report-box b, .report-box span {
        color: #111 !important;
    }
    .report-header {
        font-family: 'Merriweather', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #111 !important;
        border-bottom: 1px solid #111;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="wsj-header"><div class="wsj-title">GLOBAL MACRO & MARKET DASHBOARD</div><div class="wsj-subtitle">Central Bank Policy Trackers, Yield Curves & Rates at a glance</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="report-date">Report Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 1. Government Bond Yield Fetcher (Parallelized v5)
# ----------------------------------------------------
fallback_defaults = {
    "USA_2Y": 3.85, "USA_10Y": 3.95, "USA_30Y": 4.25,
    "EURO_2Y": 2.15, "EURO_10Y": 2.30, "EURO_30Y": 2.65,
    "UK_2Y": 3.75, "UK_10Y": 3.90, "UK_30Y": 4.40,
    "JPN_2Y": 0.35, "JPN_10Y": 0.95, "JPN_30Y": 2.15
}

@st.cache_data(ttl=1800)
def fetch_all_bond_yields():
    data = fallback_defaults.copy()
    
    # Execute 4 API requests in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_us = executor.submit(fetch_us_treasury_yields)
        future_jpn = executor.submit(fetch_japan_mof_yields)
        future_ecb = executor.submit(fetch_ecb_eurozone_yields)
        future_boe = executor.submit(fetch_boe_uk_yields)
        
        # Update dictionary with returned results
        data.update(future_us.result())
        data.update(future_jpn.result())
        data.update(future_ecb.result())
        data.update(future_boe.result())
        
    return data

bond_data = fetch_all_bond_yields()

# ----------------------------------------------------
# Manual Bond Yield Override
# ----------------------------------------------------
# Manual override for bond yields in case automated fetching fails.
TRUSTED_YIELD_LINKS = {
    "EURO": {
        "label": "🇪🇺 Euro Area (German Bund)",
        "2Y": "https://www.investing.com/rates-bonds/germany-2-year-bond-yield",
        "30Y": "https://www.investing.com/rates-bonds/germany-30-year-bond-yield",
    },
    "UK": {
        "label": "🇬🇧 United Kingdom",
        "2Y": "https://www.investing.com/rates-bonds/uk-2-year-bond-yield",
        "30Y": "https://www.investing.com/rates-bonds/uk-30-year-bond-yield",
    },
    "JPN": {
        "label": "🇯🇵 Japan",
        "2Y": "https://www.investing.com/rates-bonds/japan-2-year-bond-yield",
        "30Y": "https://www.investing.com/rates-bonds/japan-30-year-bond-yield",
    },
}
GENERAL_YIELD_LINKS = {
    "WSJ Bond Market Data": "https://www.wsj.com/market-data/bonds",
    "FT Government Bonds": "https://markets.ft.com/data/bonds",
}

with st.sidebar:
    st.markdown("### 🖊️ Manual Bond Yield Override")
    st.caption("EURO / UK / JPN 2Y & 30Y are auto-fetched where possible, but these sources can be fragile. Check a trusted site below and correct if needed.")

    st.markdown("**General references:**")
    gen_cols = st.columns(len(GENERAL_YIELD_LINKS))
    for col, (label, url) in zip(gen_cols, GENERAL_YIELD_LINKS.items()):
        with col:
            st.markdown(f'<a href="{url}" target="_blank" class="official-link-btn">{label} ↗</a>', unsafe_allow_html=True)

    st.divider()

    for country in ["EURO", "UK", "JPN"]:
        links = TRUSTED_YIELD_LINKS[country]
        st.markdown(f"**{links['label']}**")
        c2, c30 = st.columns(2)
        with c2:
            st.markdown(f'<a href="{links["2Y"]}" target="_blank" class="official-link-btn">2Y quote ↗</a>', unsafe_allow_html=True)
            new_2y = st.number_input(f"{country} 2Y (%)", value=float(bond_data[f"{country}_2Y"]), step=0.01, format="%.2f", key=f"{country}_2Y_input")
        with c30:
            st.markdown(f'<a href="{links["30Y"]}" target="_blank" class="official-link-btn">30Y quote ↗</a>', unsafe_allow_html=True)
            new_30y = st.number_input(f"{country} 30Y (%)", value=float(bond_data[f"{country}_30Y"]), step=0.01, format="%.2f", key=f"{country}_30Y_input")
        bond_data[f"{country}_2Y"] = new_2y
        bond_data[f"{country}_30Y"] = new_30y
        st.markdown("---")

def get_bond_metrics(country):
    y2 = bond_data.get(f"{country}_2Y")
    y10 = bond_data.get(f"{country}_10Y")
    y30 = bond_data.get(f"{country}_30Y")

    s_10_2 = (y10 - y2) if (y10 is not None and y2 is not None) else 0.0
    s_30_10 = (y30 - y10) if (y30 is not None and y10 is not None) else 0.0

    return {
        "2Y": f"{y2:.2f}%" if y2 is not None else "N/A",
        "10Y": f"{y10:.2f}%" if y10 is not None else "N/A",
        "30Y": f"{y30:.2f}%" if y30 is not None else "N/A",
        "s_10_2": f"{s_10_2:+.2f}%",
        "s_30_10": f"{s_30_10:+.2f}%",
        "is_inverted": (s_10_2 < 0)
    }

# ----------------------------------------------------
# 2. Central Bank Policy & Inflation (Parallelized v5)
# ----------------------------------------------------
# --- Extract fetch logic for each country into functions ---
def _get_us_cb_data(defaults):
    debug_log = []
    usa_cpi, usa_rate, usa_change = defaults["USA"]["cpi"], defaults["USA"]["rate"], defaults["USA"]["change"]
    try:
        cpi_df = fred_fetch_series("PCEPI", (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d"))
        if len(cpi_df) >= 13:
            yoy = cpi_df["value"].pct_change(12) * 100
            usa_cpi = round(float(yoy.iloc[-1]), 2)
            debug_log.append(f"USA CPI (PCE YoY): {usa_cpi}% (FRED: PCEPI)")
    except Exception as e:
        debug_log.append(f"USA CPI error: {e}")

    try:
        lower_df = fred_fetch_series("DFEDTARL", (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"))
        upper_df = fred_fetch_series("DFEDTARU", (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"))
        if not lower_df.empty and not upper_df.empty:
            cur_lower = float(lower_df.iloc[-1]["value"])
            cur_upper = float(upper_df.iloc[-1]["value"])
            usa_rate = f"{cur_lower:.2f}% to {cur_upper:.2f}%"
            distinct_vals = []
            for v in upper_df["value"].dropna().tolist():
                if not distinct_vals or distinct_vals[-1] != v:
                    distinct_vals.append(v)
            if len(distinct_vals) >= 2:
                diff_bps = int(round((distinct_vals[-1] - distinct_vals[-2]) * 100))
                usa_change = f"{diff_bps:+} bps" if diff_bps != 0 else "Unchanged"
            debug_log.append(f"USA Fed Funds: {usa_rate} | Action: {usa_change}")
    except Exception as e:
        debug_log.append(f"USA Rate error: {e}")
    return {"cpi": usa_cpi, "rate": usa_rate, "change": usa_change}, debug_log

def _get_euro_cb_data(defaults):
    debug_log = []
    euro_cpi, euro_rate, euro_change = defaults["EURO"]["cpi"], defaults["EURO"]["rate"], defaults["EURO"]["change"]
    try:
        # CP0000EZ19M086NEST is provided as an index value. Calculate YoY % change manually.
        cpi_df = fred_fetch_series("CP0000EZ19M086NEST", (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d"))
        if len(cpi_df) >= 13:
            yoy = cpi_df["value"].pct_change(12) * 100
            euro_cpi = round(float(yoy.iloc[-1]), 2)
            debug_log.append(f"EURO CPI (HICP YoY): {euro_cpi}% (calculated from index CP0000EZ19M086NEST)")
    except Exception as e:
        debug_log.append(f"EURO CPI error: {e}")

    try:
        rate_df = fred_fetch_series("ECBDFR", (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"))
        if not rate_df.empty:
            cur = float(rate_df.iloc[-1]["value"])
            euro_rate = f"{cur:.2f}%"
            distinct_vals = []
            for v in rate_df["value"].dropna().tolist():
                if not distinct_vals or distinct_vals[-1] != v:
                    distinct_vals.append(v)
            if len(distinct_vals) >= 2:
                diff_bps = int(round((distinct_vals[-1] - distinct_vals[-2]) * 100))
                euro_change = f"{diff_bps:+} bps" if diff_bps != 0 else "Unchanged"
            debug_log.append(f"ECB Deposit Rate: {euro_rate} | Action: {euro_change}")
    except Exception as e:
        debug_log.append(f"ECB Rate error: {e}")
    return {"cpi": euro_cpi, "rate": euro_rate, "change": euro_change}, debug_log

def _get_uk_cb_data(defaults):
    debug_log = []
    uk_cpi, uk_rate, uk_change = defaults["UK"]["cpi"], defaults["UK"]["rate"], defaults["UK"]["change"]
    try:
        # The old v0 API via api.ons.gov.uk is being deprecated. Use the new endpoint as primary.
        candidate_urls = [
            "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g7/mm23/data",
            "https://api.ons.gov.uk/timeseries/D7G7/dataset/mm23/data",
        ]
        months = []
        for url in candidate_urls:
            try:
                resp = http_session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                months = resp.json().get("months", [])
                if months:
                    debug_log.append(f"UK CPI source: {url}")
                    break
            except Exception:
                continue
        if months:
            uk_cpi = round(float(months[-1]["value"]), 2)
            debug_log.append(f"UK CPI (YoY): {uk_cpi}% (ONS: D7G7)")
    except Exception as e:
        debug_log.append(f"UK CPI error: {e}")

    try:
        boe = fetch_boe_bank_rate()
        if boe is not None:
            cur, prev = boe
            uk_rate = f"{cur:.2f}%"
            if prev is not None and prev != cur:
                diff_bps = int(round((cur - prev) * 100))
                uk_change = f"{diff_bps:+} bps"
            else:
                uk_change = "Unchanged"
            debug_log.append(f"BOE Bank Rate: {uk_rate} | Action: {uk_change}")
    except Exception as e:
        debug_log.append(f"BOE Rate error: {e}")
    return {"cpi": uk_cpi, "rate": uk_rate, "change": uk_change}, debug_log

def _get_jpn_cb_data(defaults):
    debug_log = []
    jpn_cpi, jpn_rate, jpn_change = defaults["JPN"]["cpi"], defaults["JPN"]["rate"], defaults["JPN"]["change"]
    try:
        estat_val = fetch_estat_cpi_yoy(debug_log)
        if estat_val is not None:
            jpn_cpi = estat_val
        else:
            # Fallback: FRED (OECD derived, may be stale)
            cpi_df = fred_fetch_series("CPALTT01JPM659N", (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"))
            if not cpi_df.empty:
                jpn_cpi = round(float(cpi_df.iloc[-1]["value"]), 2)
                debug_log.append(f"JPN CPI (YoY, fallback): {jpn_cpi}% (FRED: CPALTT01JPM659N)")
    except Exception as e:
        debug_log.append(f"JPN CPI error: {e}")

    boj_policy = fetch_boj_policy_action()
    if boj_policy is not None:
        target_rate, action, stmt_url = boj_policy
        jpn_rate = f"{target_rate:.2f}%"
        jpn_change = action
        debug_log.append(f"BOJ Policy Rate Target: {jpn_rate} | Last Action: {jpn_change} (Source: {stmt_url})")
    return {"cpi": jpn_cpi, "rate": jpn_rate, "change": jpn_change}, debug_log

@st.cache_data(ttl=43200)
def fetch_cb_data():
    defaults = {
        "USA": {"cpi": 2.50, "rate": "3.75% to 4.00%", "change": "+25 bps"},
        "EURO": {"cpi": 2.20, "rate": "2.50%", "change": "+25 bps"},
        "UK": {"cpi": 3.40, "rate": "3.75%", "change": "Unchanged"},
        "JPN": {"cpi": 2.80, "rate": "1.25%", "change": "+25 bps"}
    }
    results = {}
    debug_log = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_us = executor.submit(_get_us_cb_data, defaults)
        f_euro = executor.submit(_get_euro_cb_data, defaults)
        f_uk = executor.submit(_get_uk_cb_data, defaults)
        f_jpn = executor.submit(_get_jpn_cb_data, defaults)

        res_us, log_us = f_us.result()
        res_euro, log_euro = f_euro.result()
        res_uk, log_uk = f_uk.result()
        res_jpn, log_jpn = f_jpn.result()

        results["USA"] = res_us
        results["EURO"] = res_euro
        results["UK"] = res_uk
        results["JPN"] = res_jpn
        
        debug_log.extend(log_us + log_euro + log_uk + log_jpn)

    results["_debug"] = debug_log
    return results

def get_greeting(name: str = "Antonio") -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return f"Good morning, {name}! ☀️"
    elif hour < 18:
        return f"Good afternoon, {name}! 🌤️"
    else:
        return f"Good evening, {name}! 🌙"

MANUAL_RATE_OVERRIDES = {}
OVERRIDE_BADGE_DAYS = 7

def apply_manual_overrides(results: dict) -> dict:
    today = datetime.datetime.now().date()
    new_badges = {}
    for cb, override in MANUAL_RATE_OVERRIDES.items():
        if cb not in results:
            continue
        try:
            eff_date = datetime.datetime.strptime(override["effective_date"], "%Y-%m-%d").date()
        except Exception:
            continue
        days_since = (today - eff_date).days
        if days_since < 0:
            continue

        results[cb]["rate"] = override["rate"]
        results[cb]["change"] = override.get("change", results[cb]["change"])
        if days_since <= OVERRIDE_BADGE_DAYS:
            new_badges[cb] = override.get("source", "Manual update")

    results["_new_badges"] = new_badges
    return results

st.markdown(f"#### {get_greeting()}")

with st.spinner("Loading Central Bank, Yield & Market Data..."):
    cb_live = fetch_cb_data()
    cb_live = apply_manual_overrides(cb_live)

with st.expander("🔍 Data Fetch Status (Diagnostics)", expanded=False):
    st.caption("Shows whether each metric was retrieved live from its source, or fell back to a default value.")
    for line in cb_live.get("_debug", []):
        st.text(line)

cb_configs = {
    "FRB": {
        "code": "USA",
        "title": "FRB (Federal Reserve Board)",
        "metric": "U.S. PCE Inflation (YoY)",
        "target_min": 1.5, "target_max": 2.5,
        "rate": cb_live["USA"]["cpi"],
        "policy_rate": cb_live["USA"]["rate"],
        "policy_rate_label": "Federal Funds Target Range",
        "last_change": cb_live["USA"]["change"],
        "next_meeting": get_next_meeting_date(fomc_schedules),
        "new_badge": cb_live.get("_new_badges", {}).get("USA"),
        "official_url": "https://www.federalreserve.gov/monetarypolicy.htm"
    },
    "ECB": {
        "code": "EURO",
        "title": "ECB (European Central Bank)",
        "metric": "Eurozone HICP Inflation (YoY)",
        "target_min": 1.5, "target_max": 2.5,
        "rate": cb_live["EURO"]["cpi"],
        "policy_rate": cb_live["EURO"]["rate"],
        "policy_rate_label": "ECB Deposit Facility Rate",
        "last_change": cb_live["EURO"]["change"],
        "next_meeting": get_next_meeting_date(ecb_schedules),
        "new_badge": cb_live.get("_new_badges", {}).get("EURO"),
        "official_url": "https://www.ecb.europa.eu/mopo/html/index.en.html"
    },
    "BOE": {
        "code": "UK",
        "title": "BOE (Bank of England)",
        "metric": "U.K. CPI Inflation (YoY)",
        "target_min": 1.0, "target_max": 3.0,
        "rate": cb_live["UK"]["cpi"],
        "policy_rate": cb_live["UK"]["rate"],
        "policy_rate_label": "Bank Rate Target",
        "last_change": cb_live["UK"]["change"],
        "next_meeting": get_next_meeting_date(boe_schedules),
        "new_badge": cb_live.get("_new_badges", {}).get("UK"),
        "official_url": "https://www.bankofengland.co.uk/monetary-policy"
    },
    "BOJ": {
        "code": "JPN",
        "title": "BOJ (Bank of Japan)",
        "metric": "Japan CPI Inflation (YoY)",
        "target_min": 1.0, "target_max": 3.0,
        "rate": cb_live["JPN"]["cpi"],
        "policy_rate": cb_live["JPN"]["rate"],
        "policy_rate_label": "Uncollateralized Overnight Call Rate",
        "last_change": cb_live["JPN"]["change"],
        "next_meeting": get_next_meeting_date(boj_schedules),
        "new_badge": cb_live.get("_new_badges", {}).get("JPN"),
        "official_url": "https://www.boj.or.jp/en/mopo/index.htm"
    }
}

def build_gauge(config):
    current = config["rate"]
    t_min = config["target_min"]
    t_max = config["target_max"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current,
        number={'suffix': "%", 'font': {'size': 24, 'color': "#111111"}},
        title={
            'text': f"<b>{config['title']}</b><br><span style='font-size:0.75em;color:#666;'>{config['metric']}</span>",
            'font': {'size': 14}
        },
        gauge={
            'axis': {'range': [0, 6.0], 'tickwidth': 1, 'tickcolor': "#333333"},
            'bar': {'color': "#111111", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#dddddd",
            'steps': [
                {'range': [0, t_min], 'color': '#FFF2CC'},
                {'range': [t_min, t_max], 'color': '#D9EAD3'},
                {'range': [t_max, 4.0], 'color': '#FFE599'},
                {'range': [4.0, 6.0], 'color': '#F4CCCC'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': current
            }
        }
    ))
    
    fig.update_layout(
        height=240,
        margin=dict(l=15, r=15, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans Pro, sans-serif")
    )
    return fig

def render_cb_row(config):
    g_col, r_col = st.columns([3, 1.5])
    bond = get_bond_metrics(config["code"])
    
    inv_class = 'class="yield-warning"' if bond["is_inverted"] else ''
    inv_label = ' ⚠️ Inverted' if bond["is_inverted"] else ''

    with g_col:
        st.plotly_chart(build_gauge(config), use_container_width=True)
        if config.get("official_url"):
            st.markdown(
                f'<div style="text-align:center; margin-top:-10px;">'
                f'<a href="{config["official_url"]}" target="_blank" class="official-link-btn">'
                f'🔗 {config["title"].split(" (")[0]} Official Site ↗</a></div>',
                unsafe_allow_html=True
            )
    with r_col:
        badge_html = ""
        if config.get("new_badge"):
            badge_html = f'<span class="new-badge" title="Source: {config["new_badge"]}">NEW</span>'
        st.markdown(f"""
            <div class="rate-card">
                <div class="rate-card-title">Policy Rate{badge_html}</div>
                <div class="rate-card-value">{config['policy_rate']}</div>
                <div class="rate-card-sub">{config['policy_rate_label']}</div>
                <div class="rate-card-details">
                    <div class="rate-detail-item">
                        <span><b>Last Action:</b></span>
                        <span>{config['last_change']}</span>
                    </div>
                    <div class="rate-detail-item">
                        <span><b>Next Meeting:</b></span>
                        <span>{config['next_meeting']}</span>
                    </div>
                    <div class="rate-detail-item">
                        <span><b>{config['metric']}:</b></span>
                        <span>{config['rate']:.2f}%</span>
                    </div>
                </div>
            </div>
            <div class="yield-card">
                <div class="yield-card-title">Govt Bond Yields & Spreads</div>
                <div class="rate-detail-item"><span><b>2Y / 10Y / 30Y:</b></span> <span>{bond['2Y']} / {bond['10Y']} / {bond['30Y']}</span></div>
                <div class="rate-detail-item"><span><b>10Y - 2Y Spread:</b></span> <span {inv_class}>{bond['s_10_2']}{inv_label}</span></div>
                <div class="rate-detail-item"><span><b>30Y - 10Y Spread:</b></span> <span>{bond['s_30_10']}</span></div>
            </div>
        """, unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    render_cb_row(cb_configs["FRB"])
    st.divider()
    render_cb_row(cb_configs["BOE"])

with col_right:
    render_cb_row(cb_configs["ECB"])
    st.divider()
    render_cb_row(cb_configs["BOJ"])

st.divider()

# ----------------------------------------------------
# 3. Benchmark Market Data (Bulk Download v5)
# ----------------------------------------------------
market_tickers = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X",
    "DJIA": "^DJI", "S&P 500": "^GSPC", "DAX": "^GDAXI", "FTSE 100": "^FTSE", "Nikkei 225": "^N225",
    "Gold": "GC=F", "WTI Crude": "CL=F"
}

@st.cache_data(ttl=300)
def fetch_market_ema_data():
    results = {}
    tickers_list = list(market_tickers.values())
    
    try:
        # Drastically speed up by using yfinance bulk download (multi-threaded)
        bulk_df = yf.download(tickers_list, period="1y", group_by="ticker", threads=True, progress=False)
        
        for name, ticker in market_tickers.items():
            try:
                # Extract data for the specific ticker from MultiIndex
                if isinstance(bulk_df.columns, pd.MultiIndex):
                    if ticker in bulk_df.columns.get_level_values(0):
                        df = bulk_df[ticker].copy()
                    else:
                        continue
                else:
                    df = bulk_df.copy()
                    
                df = df.dropna(subset=['Close'])
                
                if not df.empty and len(df) >= 144:
                    df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()
                    df['EMA_144'] = df['Close'].ewm(span=144, adjust=False).mean()

                    latest_price = float(df['Close'].iloc[-1])
                    ema_55 = float(df['EMA_55'].iloc[-1])
                    ema_144 = float(df['EMA_144'].iloc[-1])
                    
                    dev_55 = ((latest_price - ema_55) / ema_55) * 100
                    dev_144 = ((latest_price - ema_144) / ema_144) * 100
                    
                    results[name] = {
                        "latest": latest_price, "ema_55": ema_55, "dev_55": dev_55,
                        "ema_144": ema_144, "dev_144": dev_144,
                        "df": df
                    }
            except Exception:
                pass
    except Exception as e:
        print(f"yfinance bulk download error: {e}")
        
    return results

market_data = fetch_market_ema_data()

def build_candlestick_chart(name: str, df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price",
        increasing_line_color='#2e7d32',
        decreasing_line_color='#c62828',
        increasing_fillcolor='#2e7d32',
        decreasing_fillcolor='#c62828'
    ))

    if 'EMA_55' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_55'],
            mode='lines',
            line=dict(color='#f57c00', width=1.3),
            name="55 EMA"
        ))

    if 'EMA_144' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_144'],
            mode='lines',
            line=dict(color='#1976d2', width=1.3),
            name="144 EMA"
        ))

    fig.update_layout(
        height=220,
        margin=dict(l=5, r=5, t=10, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans Pro, sans-serif", size=10),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#eeeeee', nticks=6),
        yaxis=dict(showgrid=True, gridcolor='#eeeeee', side='right')
    )
    return fig

def render_market_card(name, values):
    precision = 4 if ("USD" in name or "EUR" in name or "GBP" in name) and "JPY" not in name else 2
    rate_fmt = f"{values['latest']:,.{precision}f}"
    ema55_fmt = f"{values['ema_55']:,.{precision}f}"
    ema144_fmt = f"{values['ema_144']:,.{precision}f}"
    
    dev55_class = "pos-dev" if values['dev_55'] >= 0 else "neg-dev"
    dev144_class = "pos-dev" if values['dev_144'] >= 0 else "neg-dev"
    
    st.markdown(f"""
        <div class="market-card">
            <div class="market-name">{name}</div>
            <div class="market-rate">{rate_fmt}</div>
            <div class="market-metric-row">
                <span><b>55 EMA:</b> {ema55_fmt}</span>
                <span class="{dev55_class}">Dev: {values['dev_55']:+.2f}%</span>
            </div>
            <div class="market-metric-row">
                <span><b>144 EMA:</b> {ema144_fmt}</span>
                <span class="{dev144_class}">Dev: {values['dev_144']:+.2f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


st.subheader("💱 Foreign Exchange")
f_cols = st.columns(3)
for i, name in enumerate(["EUR/USD", "USD/JPY", "GBP/USD"]):
    if name in market_data:
        with f_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

st.subheader("📈 Major Equity Indices")
i_cols = st.columns(5)
for i, name in enumerate(["DJIA", "S&P 500", "DAX", "FTSE 100", "Nikkei 225"]):
    if name in market_data:
        with i_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

st.subheader("🛢️ Commodities")
c_cols = st.columns(2)
for i, name in enumerate(["Gold", "WTI Crude"]):
    if name in market_data:
        with c_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

st.subheader("📝 Dynamic Macro Scenario & Market Report")

scenario = st.radio(
    "Select Central Bank Policy Outlook Scenario:",
    ["Base Scenario (Soft Landing)", "Hawkish Scenario (Persistent Inflation)", "Dovish Scenario (Growth Slowdown)"],
    horizontal=True
)

us_pce = cb_configs["FRB"]["rate"]
us_10_2 = get_bond_metrics("USA")["s_10_2"]
sp500_dev = market_data.get("S&P 500", {}).get("dev_55", 0.0)
usdjpy_dev = market_data.get("USD/JPY", {}).get("dev_55", 0.0)

if "Hawkish" in scenario:
    narrative_cb = f"Sticky inflation pressures (U.S. PCE: **{us_pce}%**) trigger concerns over prolonged monetary tightening, driving short-term yields higher."
    narrative_fx = f"Elevated U.S. yields keep interest rate differentials wide, supporting USD/JPY (55-day EMA dev: **{usdjpy_dev:+.2f}%**)."
    narrative_eq = f"Higher-for-longer policy rates and potential curve flattening weigh on risk assets like S&P 500 (55-day EMA dev: **{sp500_dev:+.2f}%**)."
    narrative_comm = f"Rising real yields create headwinds for Gold, while WTI Crude remains sensitive to supply policy dynamics."
elif "Dovish" in scenario:
    narrative_cb = f"Slowing global growth accelerates interest rate cuts, leading to a bull-steepening of the yield curve (U.S. 10Y-2Y Spread: **{us_10_2}**)."
    narrative_fx = f"Falling U.S. treasury yields weigh on the USD, pushing USD/JPY lower (55-day EMA dev: **{usdjpy_dev:+.2f}%**)."
    narrative_eq = f"Monetary easing fuels a risk-on rally across stock markets, pulling S&P 500 (55-day EMA dev: **{sp500_dev:+.2f}%**) upwards."
    narrative_comm = f"A weaker USD and falling interest rates support Gold demand as a safe-haven and inflation hedge."
else:
    narrative_cb = f"Inflation moderates near targets (U.S. PCE: **{us_pce}%**), allowing central banks to proceed with gradual normalization while yield curves normalize."
    narrative_fx = f"Foreign exchange markets trade in stable ranges, with USD/JPY (55-day EMA dev: **{usdjpy_dev:+.2f}%**) mirroring real rate parities."
    narrative_eq = f"Equities advance steadily, backed by earnings growth and stable financial conditions (S&P 500 55-day EMA dev: **{sp500_dev:+.2f}%**)."
    narrative_comm = f"Commodities move in line with fundamental supply and demand dynamics in a soft-landing environment."

st.markdown(f"""
<div class="report-box">
    <div class="report-header">Executive Summary & Strategic Outlook ({scenario.split()[0]})</div>
    <p><b>1. Central Bank & Yield Curve Outlook:</b><br>{narrative_cb}</p>
    <p><b>2. Foreign Exchange Impact:</b><br>{narrative_fx}</p>
    <p><b>3. Equity Market Outlook:</b><br>{narrative_eq}</p>
    <p><b>4. Commodities & Safe-Haven Assets:</b><br>{narrative_comm}</p>
</div>
""", unsafe_allow_html=True)