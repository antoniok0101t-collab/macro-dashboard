import streamlit as st
import pandas as pd
import re
import requests
import io
import yfinance as yf
import plotly.graph_objects as go
import datetime
import zlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------
# API Keys / Config
# ----------------------------------------------------
# When deploying to Streamlit Cloud, put the API keys in "Secrets"
# (Settings > Secrets, in the format below).
#   FRED_API_KEY = "..."
#   ESTAT_APP_ID = "..."
# When running on a local PC, the hard-coded fallback values below are used so
# the app still starts without st.secrets. Do not share this file publicly
# while real keys are in it.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    FRED_API_KEY = "086fe9dc602ab52e1382533a7e6890c9"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# e-Stat (Japan's official statistics portal) application ID
try:
    ESTAT_APP_ID = st.secrets["ESTAT_APP_ID"]
except Exception:
    ESTAT_APP_ID = "ddedf4a8b47ec53d8f7ee614fb844a1d2e7e1d17"
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
ESTAT_LIST_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsList"
ESTAT_META_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getMetaInfo"

REQUEST_TIMEOUT = 10  # seconds
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Global-Macro-Dashboard/4.0)"}

# ----------------------------------------------------
# Chart settings
# ----------------------------------------------------
HISTORY_YEARS = 3      # length of the inflation / policy-rate / unemployment charts
CHART_HEIGHT = 300

# Estimated "full employment" (NAIRU-like) zone for each unemployment chart.
# The zone is painted green from the bottom of the chart up to `solid_to`, then fades
# out towards `fade_to` (no hard line). These are rough estimates, not official
# figures: edit the numbers freely.
FULL_EMPLOYMENT_ZONES = {
    "USA":  {"solid_to": 4.20, "fade_to": 4.50},
    "EURO": {"solid_to": 6.00, "fade_to": 6.50},
    "UK":   {"solid_to": 4.25, "fade_to": 4.75},
    "JPN":  {"solid_to": 2.30, "fade_to": 2.70},
}


# FRED limits request bursts (HTTP 429). Keep at most 3 requests in flight, retry on
# temporary errors, and remember the last error per series so Diagnostics can show it.
_FRED_SEM = threading.Semaphore(3)
FRED_LAST_ERROR: dict = {}
_FRED_DOWN_UNTIL = [0.0]   # circuit breaker: skip FRED for a while after it fails completely


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

    if time.time() < _FRED_DOWN_UNTIL[0]:
        FRED_LAST_ERROR[series_id] = "skipped: FRED was unreachable a moment ago"
        return pd.DataFrame(columns=["date", "value"])
    err = "unknown error"
    for attempt in range(3):
        try:
            with _FRED_SEM:
                resp = requests.get(FRED_BASE_URL, params=params, headers=REQUEST_HEADERS,
                                    timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                err = f"HTTP {resp.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            try:
                data = resp.json().get("observations", [])
            except ValueError:
                raise ValueError(f"not JSON (HTTP {resp.status_code}, "
                                 f"{len(resp.content)} bytes: {resp.text[:60]!r})")
            FRED_LAST_ERROR.pop(series_id, None)
            if not data:
                return pd.DataFrame(columns=["date", "value"])
            df = pd.DataFrame(data)[["date", "value"]]
            df = df[df["value"] != "."]
            df["value"] = df["value"].astype(float)
            df["date"] = pd.to_datetime(df["date"])
            return df.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:90]}"
            time.sleep(1.5 * (attempt + 1))
    # The keyed API failed (e.g. an empty reply). Try FRED's public CSV download instead,
    # which needs no key and is a different endpoint.
    try:
        with _FRED_SEM:
            r2 = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                              params={"id": series_id, **({"cosd": start_date} if start_date else {}),
                                      **({"coed": end_date} if end_date else {})},
                              headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        r2.raise_for_status()
        raw = pd.read_csv(io.StringIO(r2.text))
        out = pd.DataFrame({"date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
                            "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce")})
        out = out.dropna().reset_index(drop=True)
        if not out.empty:
            FRED_LAST_ERROR.pop(series_id, None)
            return out
        err += " | CSV fallback: no rows"
    except Exception as e:  # noqa: BLE001
        err += f" | CSV fallback: {type(e).__name__}: {str(e)[:70]}"
    _FRED_DOWN_UNTIL[0] = time.time() + 120   # stop hammering a site that is down
    FRED_LAST_ERROR[series_id] = err
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
# Generic helpers (parallel fetching, series handling)
# ----------------------------------------------------
def run_parallel(tasks: dict, max_workers: int = 8) -> dict:
    """Run independent fetch callables concurrently.

    Returns {name: (result, error_text_or_None)}. An exception in one task never
    breaks the others.
    """
    out = {}
    if not tasks:
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {name: pool.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                out[name] = (fut.result(), None)
            except Exception as e:  # noqa: BLE001
                out[name] = (None, f"{type(e).__name__}: {e}")
    return out


def history_start(years: int = HISTORY_YEARS + 1) -> str:
    """Start date for history downloads (one extra year so YoY can be computed)."""
    d = datetime.datetime.now() - datetime.timedelta(days=365 * years + 10)
    return d.strftime("%Y-%m-%d")


def to_series(df) -> pd.Series:
    """FRED-style DataFrame (date, value) -> pd.Series indexed by date."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"].astype(float).sort_index()


def yoy_from_index(df) -> pd.Series:
    """Year-over-year % change of a monthly price-index DataFrame (date, value)."""
    s = to_series(df)
    if len(s) < 13:
        return pd.Series(dtype=float)
    return (s.pct_change(12) * 100).dropna()


def trim_history(s: pd.Series, years: int = HISTORY_YEARS) -> pd.Series:
    """Keep only the last `years` years of a series."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    return s[s.index >= cutoff]


def ons_months_series(urls: list, label: str, debug_log: list) -> pd.Series:
    """Monthly values of an ONS time series ("months" array), as a pd.Series."""
    last_error = None
    for url in urls:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            rows = {}
            for m in resp.json().get("months", []):
                try:
                    rows[pd.to_datetime(m["date"], format="%Y %b")] = float(m["value"])
                except (KeyError, ValueError, TypeError):
                    continue
            if rows:
                debug_log.append(f"{label} source: {url}")
                return pd.Series(rows).sort_index()
        except Exception as e:  # noqa: BLE001
            last_error = e
    if last_error is not None:
        raise last_error
    raise ValueError(f"{label}: no monthly data returned")


# ----------------------------------------------------
# Official Central Bank & Ministry of Finance Fetchers
# ----------------------------------------------------

def fetch_us_treasury_yields():
    """
    Fetches US Treasury yields (2Y, 10Y, 30Y) directly from FRED (fetched in parallel).
    Note: ^IRX is 13-week T-bill and NOT 2-year yield; DGS2 is used for 2Y.
    """
    yields = {}
    fred_map = {"USA_2Y": "DGS2", "USA_10Y": "DGS10", "USA_30Y": "DGS30"}
    yf_map = {"USA_10Y": "^TNX", "USA_30Y": "^TYX"}

    def yf_last(ticker):
        hist = yf.Ticker(ticker).history(period="3d")
        return None if hist.empty else round(float(hist["Close"].iloc[-1]), 2)

    tasks = {f"fred:{k}": (lambda sid=sid: fred_latest_value(sid, lookback_days=30))
             for k, sid in fred_map.items()}
    tasks.update({f"yf:{k}": (lambda t=t: yf_last(t)) for k, t in yf_map.items()})
    got = run_parallel(tasks)

    for key in fred_map:
        res, _ = got[f"fred:{key}"]
        if res is not None:
            yields[key] = round(res[1], 2)
    # Yahoo Finance (more current intraday) overrides FRED for 10Y / 30Y when available
    for key in yf_map:
        res, _ = got[f"yf:{key}"]
        if res is not None:
            yields[key] = res
    return yields


def fetch_japan_mof_yields():
    """Fetches Japanese Government Bond (JGB) yields from MOF Japan benchmark CSV."""
    url = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
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
    """Fetches Euro Area AAA Sovereign Yield Curve spot rates from ECB Data Portal."""
    base_url = "https://data-api.ecb.europa.eu/service/data/YC"
    maturities = {"EURO_2Y": "SR_2Y", "EURO_10Y": "SR_10Y", "EURO_30Y": "SR_30Y"}

    def one(mat):
        series_key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.{mat}"
        url = f"{base_url}/{series_key}?format=csvdata&lastNObservations=3"
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "OBS_VALUE" in df.columns and not df["OBS_VALUE"].dropna().empty:
            return round(float(df["OBS_VALUE"].dropna().iloc[-1]), 2)
        return None

    got = run_parallel({k: (lambda m=m: one(m)) for k, m in maturities.items()})
    results = {k: v for k, (v, _) in got.items() if v is not None}

    if "EURO_10Y" not in results:
        res = fred_latest_value("IRLTLT01EZM156N", lookback_days=120)
        if res is not None:
            results["EURO_10Y"] = round(res[1], 2)

    return results


def fetch_boe_uk_yields():
    """Fetches UK Government Gilt yields (2Y, 10Y, 30Y) automatically."""
    results = {}
    try:
        url = (
            "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
            "?symbols=TMBMKGB-02Y,TMBMKGB-10Y,TMBMKGB-30Y"
            "&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json"
        )
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
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
            resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
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
    """
    Bank of England Statistical Database (IADB) Bank Rate history (4 years).

    Returns (current, value_before_latest_meeting, history_dataframe) or None.

    The "last action" is decided per meeting. Searching the history for "the latest
    different value" would mistake a change made many meetings ago for the latest
    decision (e.g. after six consecutive holds it would still report the cut made
    seven meetings back). So the known meeting dates (boe_schedules) are used: the
    value just before the latest past meeting is compared with the current value.
    """
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365 * (HISTORY_YEARS + 1))
        url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
        params = {
            "csv.x": "yes",
            "SeriesCodes": "IUDBEDR",
            "UsingCodes": "Y",
            "CSVF": "TN",
            "Datefrom": start.strftime("%d/%b/%Y"),
            "Dateto": end.strftime("%d/%b/%Y"),
        }
        resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if df.empty:
            return None

        cur = float(df.iloc[-1]["value"])

        # Find the latest past meeting and compare the value just before it with now
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        past_meetings = [d for d in boe_schedules if d <= today_str]
        if not past_meetings:
            return cur, None, df

        latest_meeting_date = pd.to_datetime(past_meetings[-1])
        before_meeting = df[df["date"] < latest_meeting_date]
        if before_meeting.empty:
            return cur, None, df
        pre_meeting_val = float(before_meeting.iloc[-1]["value"])
        return cur, pre_meeting_val, df
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

# Past BoJ decision dates (last day of each meeting) used to rebuild the policy-rate
# history from the statements. Meetings from 2026 on come from boj_schedules.
BOJ_HISTORY_MEETINGS = [
    "2023-10-31", "2023-12-19",
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14", "2024-07-31",
    "2024-09-20", "2024-10-31", "2024-12-19",
    "2025-01-24", "2025-03-19", "2025-05-01", "2025-06-17", "2025-07-31",
    "2025-09-19", "2025-10-30", "2025-12-19",
]
# Used only when the statements cannot be read: (effective date, rate in %).
# Upper end of the range is used for the 0-0.1% period of 2024. Add new changes here.
BOJ_RATE_FALLBACK = [
    ("2023-10-31", -0.10), ("2024-03-19", 0.10), ("2024-07-31", 0.25),
    ("2025-01-24", 0.50), ("2025-12-19", 0.75),
    ("2026-06-17", 1.00), ("2026-09-24", 1.25),     # effective dates (decisions on Jun 16 / Sep 18)
]


def get_next_meeting_date(schedule_dates):
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    for date_str in schedule_dates:
        if date_str >= today_str:
            return date_str
    return "TBD"


def _estat_request(params: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """GET e-Stat getStatsData; returns the GET_STATS_DATA dict or raises on an API error."""
    resp = requests.get(ESTAT_BASE_URL, params=params, headers=REQUEST_HEADERS, timeout=timeout)
    resp.raise_for_status()
    root = resp.json().get("GET_STATS_DATA", {})
    status = root.get("RESULT", {})
    if str(status.get("STATUS", "0")) != "0":
        raise ValueError(f"e-Stat API error: {status.get('ERROR_MSG')}")
    return root


def _estat_class_objs(stats_id: str, timeout: int = 30) -> list:
    """All dimensions and items of a table, from getMetaInfo.

    (A one-row getStatsData call is only a fallback: it can list just the items that
    occur in that single row, which hides most of the table.)
    """
    objs = []
    try:
        resp = requests.get(ESTAT_META_URL, params={"appId": ESTAT_APP_ID, "statsDataId": stats_id},
                            headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        objs = (resp.json().get("GET_META_INFO", {}).get("METADATA_INF", {})
                .get("CLASS_INF", {}).get("CLASS_OBJ", []))
    except Exception:  # noqa: BLE001
        objs = []
    if not objs:
        root = _estat_request({"appId": ESTAT_APP_ID, "statsDataId": stats_id, "cntGetFlg": "N",
                               "metaGetFlg": "Y", "limit": 1}, timeout=timeout)
        objs = root.get("STATISTICAL_DATA", {}).get("CLASS_INF", {}).get("CLASS_OBJ", [])
    return [objs] if isinstance(objs, dict) else objs


def _estat_filter_params(dim_codes: dict) -> dict:
    """{'cat01': '0001', 'area': '00000'} -> {'cdCat01': '0001', 'cdArea': '00000'}."""
    return {"cd" + dim[0].upper() + dim[1:]: code for dim, code in dim_codes.items() if code}


def _estat_time_to_month(code):
    """e-Stat monthly time code (e.g. '2026000808') -> first day of that month, else None."""
    t = str(code)
    if len(t) == 10 and t[4:6] == "00" and t[6:8] == t[8:10]:
        mo = int(t[8:10])
        if 1 <= mo <= 12:
            return pd.Timestamp(int(t[:4]), mo, 1)
    return None


def _estat_class_list(class_objs):
    """CLASS_OBJ list -> {dimension id: [(code, name), ...]}."""
    maps = {}
    for obj in class_objs:
        classes = obj.get("CLASS", [])
        if isinstance(classes, dict):
            classes = [classes]
        maps[obj.get("@id")] = [(c.get("@code"), c.get("@name", "")) for c in classes]
    return maps


def fetch_estat_cpi_series(debug_log: list = None) -> pd.Series:
    """
    Monthly Japan CPI (nationwide, all items, YoY %) from the e-Stat API, as a pd.Series.

    The base year changed from 2020 to 2025 on 21 Aug 2026, so the new table is tried
    first; if it has less than three years of history the old table is used to fill
    the earlier months. Category codes are not hard-coded: they are looked up by name
    in the response (the Japanese names below must stay in Japanese because they are
    matched against the e-Stat data), so a future base-year change should not break it.
    """
    if debug_log is None:
        debug_log = []

    candidate_tables = [
        ("0004052037", "2025-base CPI"),
        ("0003427113", "2020-base CPI"),
    ]
    combined = pd.Series(dtype=float)
    start_year = datetime.datetime.now().year - HISTORY_YEARS - 1

    for stats_id, label in candidate_tables:
        try:
            base = {"appId": ESTAT_APP_ID, "statsDataId": stats_id, "cntGetFlg": "N"}
            # Step 1 (fast): table metadata only, to resolve the category codes by name
            class_objs = _estat_class_objs(stats_id)

            def find_code(class_id, name_candidates):
                for obj in class_objs:
                    if obj.get("@id") != class_id:
                        continue
                    classes = obj.get("CLASS", [])
                    if isinstance(classes, dict):
                        classes = [classes]
                    for c in classes:
                        cname = c.get("@name", "")
                        if any(nc in cname for nc in name_candidates):
                            return c.get("@code")
                return None

            cat_code = find_code("cat01", ["総合"])                      # "All items"
            area_code = find_code("area", ["全国"])                      # "Nationwide"
            tab_code = find_code("tab", ["前年同月比", "対前年同月比"])  # "YoY change"

            # Step 2: download only the needed rows (all items, nationwide, YoY, recent years).
            # If the filtered request fails, fall back to downloading the whole table.
            try:
                data_root = _estat_request({
                    **base, "metaGetFlg": "N", "cdTimeFrom": f"{start_year}000101",
                    **_estat_filter_params({"cat01": cat_code, "area": area_code, "tab": tab_code})},
                    timeout=30)
            except Exception as e:  # noqa: BLE001
                debug_log.append(f"[e-Stat/{label}] filtered request failed ({e}); downloading full table")
                data_root = _estat_request({**base, "metaGetFlg": "N"}, timeout=60)
            values = data_root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
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
                debug_log.append(f"[e-Stat/{label}] no matching data found (category code resolution may have failed)")
                continue

            rows = {}
            for v in filtered:
                ts = _estat_time_to_month(v.get("@time"))
                if ts is None:
                    continue
                try:
                    rows[ts] = float(v.get("$"))
                except (TypeError, ValueError):
                    continue
            if not rows:
                # Unknown time-code format: keep at least the latest value (string sort)
                filtered.sort(key=lambda v: str(v.get("@time", "")))
                try:
                    rows[pd.Timestamp.now().normalize().replace(day=1)] = float(filtered[-1].get("$"))
                    debug_log.append(f"[e-Stat/{label}] time code format not recognised; latest value only")
                except (TypeError, ValueError):
                    continue

            ser = pd.Series(rows).sort_index()
            debug_log.append(
                f"[e-Stat/{label}] fetched successfully: {ser.iloc[-1]}% "
                f"({ser.index[-1]:%Y-%m}, {len(ser)} months)")
            combined = combined.combine_first(ser) if not combined.empty else ser   # earlier table wins
            if len(combined) >= 12 * HISTORY_YEARS + 2:
                break

        except Exception as e:  # noqa: BLE001
            debug_log.append(f"[e-Stat/{label}] exception: {e}")
            continue

    return combined.sort_index()


def fetch_estat_cpi_yoy(debug_log: list = None):
    """Latest Japan CPI YoY (%) from e-Stat, or None."""
    ser = fetch_estat_cpi_series(debug_log)
    return None if ser.empty else round(float(ser.iloc[-1]), 2)


def fetch_estat_unemployment_series(debug_log: list):
    """
    Japan's monthly complete unemployment rate (seasonally adjusted, both sexes) from
    the e-Stat Labour Force Survey.

    The table ID is not hard-coded: the survey's tables are searched by name, the most
    plausible monthly tables are tried in turn, and the first one that yields at least a
    year of sensible values (0.5-8 %) is used. A seasonally adjusted table is preferred;
    a table that is not seasonally adjusted is kept only as a second choice.

    In e-Stat's long time-series tables the rate is not an item called "unemployment
    rate": it is the "unemployed persons" item combined with the "rate (%)" measure. Both
    forms are handled. The Japanese keywords below must stay in Japanese (they match
    e-Stat's data).

    Returns (series, is_seasonally_adjusted).
    """
    # Several broad searches (a single multi-word search can easily return nothing),
    # merged and de-duplicated by table ID
    searches = [{"searchWord": "季節調整値"}, {"searchWord": "完全失業率"},
                {"searchWord": "月次"}, {}]
    tables, seen, errors = [], set(), []
    for extra in searches:
        try:
            params = {"appId": ESTAT_APP_ID, "statsCode": "00200531", "limit": 100, **extra}
            resp = requests.get(ESTAT_LIST_URL, params=params, headers=REQUEST_HEADERS,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            lst = resp.json().get("GET_STATS_LIST", {})
            found = lst.get("DATALIST_INF", {}).get("TABLE_INF", [])
            if isinstance(found, dict):
                found = [found]
            for t in found:
                if t.get("@id") not in seen:
                    seen.add(t.get("@id"))
                    tables.append(t)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:80])
    debug_log.append(f"[e-Stat/unemployment] {len(tables)} Labour Force Survey tables found"
                     + (f" (search errors: {errors[:2]})" if errors else ""))
    if not tables:
        raise ValueError("e-Stat: no Labour Force Survey tables found by search")

    def text_of(x):
        return x.get("$", "") if isinstance(x, dict) else str(x or "")

    scored = []
    for t in tables:
        title = text_of(t.get("TITLE"))
        cycle = text_of(t.get("CYCLE"))
        score = 0
        score += 3 if "月" in cycle else 0                 # monthly
        score += 2 if "季節調整" in title else 0           # seasonally adjusted
        score += 2 if "完全失業率" in title else 0         # unemployment rate
        score += 1 if "全国" in title else 0               # nationwide
        score -= 2 if "都道府県" in title else 0           # prefectural tables
        scored.append((score, t.get("@id"), title))
    scored.sort(key=lambda x: -x[0])

    start_year = datetime.datetime.now().year - HISTORY_YEARS - 1
    keyword_priority = ("完全失業率", "男女計", "季節調整値", "全国", "総数", "15歳以上")

    best_original = None          # (series, table_id, title) of the first non-adjusted table
    for score, table_id, title in scored[:12]:
        if best_original is not None and "季節調整" not in title:
            continue                          # only look for an adjusted table from now on
        try:
            base = {"appId": ESTAT_APP_ID, "statsDataId": table_id, "cntGetFlg": "N"}
            # Step 1 (fast): table metadata only, to pick the right item in every dimension
            class_objs = _estat_class_objs(table_id)
            maps = _estat_class_list(class_objs)

            # "Rate mode": the table has a rate (%) measure and an "unemployed persons" item
            # but no item literally named "unemployment rate"
            all_names = [n for d, it in maps.items() if d != "time" for _, n in it]
            rate_tab = next((c for c, n in maps.get("tab", []) if n.strip().endswith("率")), None)
            rate_mode = (rate_tab is not None and any("完全失業者" in n for n in all_names)
                         and not any("完全失業率" in n for n in all_names))
            priority = (("完全失業者",) if rate_mode else ()) + keyword_priority

            # One code per dimension: the first keyword (in priority order) it contains
            selection, found_rate = {}, False
            for dim, items in maps.items():
                if dim == "time":
                    continue
                if dim == "tab" and rate_mode:
                    selection[dim] = rate_tab
                    continue
                for kw in priority:
                    hit = next((code for code, name in items if kw in name), None)
                    if hit is not None:
                        selection[dim] = hit
                        found_rate = found_rate or kw in ("完全失業率", "完全失業者")
                        break
            if not found_rate:
                sample = {d: [n for _, n in it[:4]] for d, it in maps.items() if d != "time"}
                debug_log.append(f"[e-Stat/unemployment {table_id}] no unemployment-rate item in "
                                 f"'{title[:40]}' (items: {str(sample)[:160]})")
                continue

            # Step 2: download only the selected rows (falls back to the whole table)
            try:
                data_root = _estat_request({
                    **base, "metaGetFlg": "N", "cdTimeFrom": f"{start_year}000101",
                    **_estat_filter_params(selection)}, timeout=30)
            except Exception as e:  # noqa: BLE001
                debug_log.append(f"[e-Stat/unemployment {table_id}] filtered request failed ({e}); full table")
                data_root = _estat_request({**base, "metaGetFlg": "N"}, timeout=60)
            values = data_root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
            if isinstance(values, dict):
                values = [values]
            rows, dupes = {}, 0
            for v in values:
                if any(v.get(f"@{dim}") != code for dim, code in selection.items()):
                    continue
                ts = _estat_time_to_month(v.get("@time"))
                if ts is None:
                    continue
                try:
                    val = float(v.get("$"))
                except (TypeError, ValueError):
                    continue
                if ts in rows:
                    dupes += 1
                    continue
                rows[ts] = val
            ser = pd.Series(rows).sort_index()
            if len(ser) >= 12 and ser.between(0.5, 8.0).all():
                chosen_names = [n for d, c in selection.items() for cc, n in maps.get(d, []) if cc == c]
                adjusted = "季節調整" in title or any("季節調整" in n for n in chosen_names)
                debug_log.append(
                    f"[e-Stat/unemployment {table_id}] '{title[:40]}': {ser.iloc[-1]}% "
                    f"({ser.index[-1]:%Y-%m}, {len(ser)} months, "
                    f"{'seasonally adjusted' if adjusted else 'NOT seasonally adjusted'}, "
                    f"ignored duplicates: {dupes})")
                if adjusted:
                    return ser, True
                if best_original is None:
                    best_original = (ser, table_id, title)
                continue
            debug_log.append(
                f"[e-Stat/unemployment {table_id}] rejected ({len(ser)} months, duplicates {dupes})")
        except Exception as e:  # noqa: BLE001
            debug_log.append(f"[e-Stat/unemployment {table_id}] exception: {e}")
    if best_original is not None:
        return best_original[0], False
    raise ValueError("e-Stat: no usable unemployment-rate table found")


# ----------------------------------------------------
# Bank of Japan: Direct Statement Scraper, Rate History & Action Calculator
# ----------------------------------------------------
def _extract_rate_from_boj_text(text: str):
    """Policy rate (%) stated in a BoJ monetary policy statement, or None."""
    text = re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    # Range form used in 2024 ("around 0 to 0.1 percent"): use the upper end
    m = re.search(r'around\s*-?[0-9]+(?:\.[0-9]+)?\s*to\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
                  text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    patterns = [
        r'uncollateralized overnight call rate to remain at around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
        r'uncollateralized overnight call rate.*?around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
        r'rate to remain at around\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    # Negative-rate era (before March 2024)
    m = re.search(r'policy interest rate[^.]{0,60}?minus\s*([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)',
                  text, re.IGNORECASE)
    if m:
        return -float(m.group(1))
    return None


def _pdf_to_text(content: bytes) -> str:
    """Text of a PDF. Uses pypdf / PyPDF2 when installed (recommended: pip install pypdf);
    otherwise a crude fallback that reads plain-text strings from the compressed streams
    (works only for PDFs that do not use embedded-font glyph codes)."""
    for mod in ("pypdf", "PyPDF2"):
        try:
            lib = __import__(mod)
            reader = lib.PdfReader(io.BytesIO(content))
            return " ".join((pg.extract_text() or "") for pg in reader.pages)
        except ImportError:
            continue
        except Exception:  # noqa: BLE001
            break
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", content, re.DOTALL):
        raw = m.group(1)
        try:
            raw = zlib.decompress(raw)
        except Exception:  # noqa: BLE001
            pass
        for sm in re.finditer(rb"\(((?:\\.|[^\\()])*)\)", raw):
            piece = re.sub(rb"\\([()\\])", rb"\1", sm.group(1))
            chunks.append(piece.decode("latin-1"))
    return "".join(chunks)


def _fetch_boj_statement_rate(date_str: str, notes: list = None):
    """(rate, url) from the statement of the meeting that ended on date_str, or None.

    BoJ publishes some statements as HTML (state_YYYY/) and, from 2026, most as PDF
    (mpr_YYYY/), so several URL forms are tried.
    """
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    stem = f"k{dt.strftime('%y%m%d')}a"
    base = "https://www.boj.or.jp/en/mopo/mpmdeci"
    url_options = [
        f"{base}/state_{dt.year}/{stem}.htm",
        f"{base}/mpr_{dt.year}/{stem}.pdf",
        f"{base}/m_{dt.year}/{stem}.htm",
        f"{base}/mpr_{dt.year}/{stem}.htm",
        f"{base}/state_{dt.year}/{stem}.pdf",
    ]
    for u in url_options:
        try:
            resp = requests.get(u, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            if u.endswith(".pdf"):
                text = _pdf_to_text(resp.content)
                val = _extract_rate_from_boj_text(text) if text else None
                if val is None and notes is not None:
                    notes.append(f"{date_str}: PDF found but its text could not be read "
                                 "(run: pip install pypdf)")
            else:
                val = _extract_rate_from_boj_text(resp.text)
            if val is not None:
                return val, u
        except Exception:
            continue
    return None


def fetch_boj_rate_data(debug_log: list = None) -> dict:
    """
    Policy-rate history and latest decision of the Bank of Japan, read from the
    meeting statements (all meetings are fetched in parallel).

    Returns {"series": step series (date -> %), "current", "action", "url", "source"}.
    If the statements cannot be read, or the latest meeting's statement is missing
    (showing an older level as "current" would be wrong), everything comes from the
    hand-kept table BOJ_RATE_FALLBACK instead.
    """
    if debug_log is None:
        debug_log = []
    today = datetime.datetime.now().date()
    window_start = pd.Timestamp(today) - pd.DateOffset(years=HISTORY_YEARS) - pd.Timedelta(days=100)
    meetings = sorted({d for d in BOJ_HISTORY_MEETINGS + boj_schedules
                       if d <= today.strftime("%Y-%m-%d") and pd.Timestamp(d) >= window_start})

    notes = []
    got = run_parallel({d: (lambda d=d: _fetch_boj_statement_rate(d, notes)) for d in meetings},
                       max_workers=8)
    parsed = {d: res for d, (res, _) in got.items() if res is not None}
    unread = [d for d in meetings if d not in parsed]
    for n in sorted(set(notes))[:3]:
        debug_log.append(f"BOJ statement note: {n}")
    # The latest decision must be known, otherwise an old level would be shown as current
    latest_known = bool(parsed) and max(parsed) == max(meetings)

    if len(parsed) >= 3 and latest_known:
        ordered = sorted(parsed.items())                      # oldest -> newest
        series = pd.Series({pd.Timestamp(d): v[0] for d, v in ordered})
        cur_rate, cur_url = ordered[-1][1]
        if len(ordered) >= 2:
            diff = int(round((cur_rate - ordered[-2][1][0]) * 100))
            action = f"{diff:+} bps" if diff != 0 else "Unchanged"
        else:
            # Do not guess the size of the change when the previous meeting is unknown
            action = "See official statement"
        debug_log.append(f"BOJ rate history: {len(parsed)}/{len(meetings)} statements read")
        return {"series": series, "current": cur_rate, "action": action,
                "url": cur_url, "source": "BoJ policy statements"}

    debug_log.append(
        f"BOJ rate history: {len(parsed)}/{len(meetings)} statements readable, latest decision "
        f"{'known' if latest_known else 'NOT readable'} (unread: {', '.join(unread[-5:])}); "
        "using the built-in table BOJ_RATE_FALLBACK")
    series = pd.Series({pd.Timestamp(d): v for d, v in BOJ_RATE_FALLBACK})
    # Current level and last action from the table (only as good as the table: keep it updated)
    last_d, last_v = BOJ_RATE_FALLBACK[-1]
    prev_v = BOJ_RATE_FALLBACK[-2][1]
    changed = bool(meetings) and pd.Timestamp(last_d) >= pd.Timestamp(max(meetings))
    action = f"{int(round((last_v - prev_v) * 100)):+} bps" if changed else "Unchanged"
    debug_log.append(f"BOJ: current rate {last_v:.2f}% / action '{action}' taken from the built-in "
                     "table BOJ_RATE_FALLBACK (statements unreadable) - verify on the BoJ site")
    return {"series": series, "current": last_v, "action": action, "url": None,
            "source": "built-in table (BoJ statements not readable - check it is up to date)"}


def fetch_boj_policy_action():
    """Latest BoJ decision as (rate, action, statement_url), or None (kept for compatibility)."""
    data = fetch_boj_rate_data()
    if data["current"] is None:
        return None
    return data["current"], data["action"], data["url"]


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

    /* Extended Policy Rate Card Styling */
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

    /* Yield Curve Card Styling */
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

    /* Market Card Styling */
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

    /* Report Box Styling */
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

# Header Section (Restored exactly to original)
st.markdown('<div class="wsj-header"><div class="wsj-title">GLOBAL MACRO & MARKET DASHBOARD</div><div class="wsj-subtitle">Central Bank Policy Trackers, Yield Curves & Rates at a glance</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="report-date">Report Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 1. Government Bond Yield Fetcher (All Automated)
# ----------------------------------------------------
fallback_defaults = {
    "USA_2Y": 3.85, "USA_10Y": 3.95, "USA_30Y": 4.25,
    "EURO_2Y": 2.15, "EURO_10Y": 2.30, "EURO_30Y": 2.65,
    "UK_2Y": 3.75, "UK_10Y": 3.90, "UK_30Y": 4.40,
    "JPN_2Y": 0.35, "JPN_10Y": 0.95, "JPN_30Y": 2.15
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_bond_yields():
    data = fallback_defaults.copy()
    got = run_parallel({
        "US": fetch_us_treasury_yields,
        "JP": fetch_japan_mof_yields,
        "EU": fetch_ecb_eurozone_yields,
        "UK": fetch_boe_uk_yields,
    })
    for key in ("US", "JP", "EU", "UK"):
        res, _ = got[key]
        if res:
            data.update(res)
    return data

bond_data = fetch_all_bond_yields()

# ----------------------------------------------------
# Manual Bond Yield Override (EURO / UK / JPN — 2Y & 30Y)
# ----------------------------------------------------
# Most of these yields are now fetched automatically by fetch_all_bond_yields(), but
# the UK (via CNBC) and Japan (column parsing of the MOF CSV) sources break easily when
# their structure changes, so a manual override is kept as a safety net. The inputs
# are pre-filled with the automatically fetched values, so normally you only check them.
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
    st.caption(
        "EURO / UK / JPN 2Y & 30Y are auto-fetched where possible, but these "
        "sources can be fragile. Check a trusted site below and correct if needed."
    )

    st.markdown("**General references:**")
    gen_cols = st.columns(len(GENERAL_YIELD_LINKS))
    for col, (label, url) in zip(gen_cols, GENERAL_YIELD_LINKS.items()):
        with col:
            st.markdown(
                f'<a href="{url}" target="_blank" class="official-link-btn">{label} ↗</a>',
                unsafe_allow_html=True
            )

    st.divider()

    for country in ["EURO", "UK", "JPN"]:
        links = TRUSTED_YIELD_LINKS[country]
        st.markdown(f"**{links['label']}**")
        c2, c30 = st.columns(2)
        with c2:
            st.markdown(
                f'<a href="{links["2Y"]}" target="_blank" class="official-link-btn">2Y quote ↗</a>',
                unsafe_allow_html=True
            )
            new_2y = st.number_input(
                f"{country} 2Y (%)", value=float(bond_data[f"{country}_2Y"]),
                step=0.01, format="%.2f", key=f"{country}_2Y_input"
            )
        with c30:
            st.markdown(
                f'<a href="{links["30Y"]}" target="_blank" class="official-link-btn">30Y quote ↗</a>',
                unsafe_allow_html=True
            )
            new_30y = st.number_input(
                f"{country} 30Y (%)", value=float(bond_data[f"{country}_30Y"]),
                step=0.01, format="%.2f", key=f"{country}_30Y_input"
            )
        bond_data[f"{country}_2Y"] = new_2y
        bond_data[f"{country}_30Y"] = new_30y
        st.markdown("---")

with st.sidebar:
    st.markdown("### 🔄 Data refresh")
    st.caption("Data is cached (bond yields 30 min, central-bank & history data 12 h, "
               "markets 5 min). Press to reload everything now.")
    if st.button("Refresh all data", key="refresh_all_data"):
        st.cache_data.clear()
        st.rerun()


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
# 2. Central Bank Policy, Inflation & Unemployment Automated Fetcher
# ----------------------------------------------------
def _hist_dict(infl=None, rate=None, unemp=None, src=None):
    """Container for the three history series of one country (3 years each)."""
    empty = pd.Series(dtype=float)
    return {
        "infl": trim_history(infl) if infl is not None else empty,
        "rate": trim_history(rate) if rate is not None else empty,
        "unemp": trim_history(unemp) if unemp is not None else empty,
        "src": src or {},
    }


def _change_from_distinct(values):
    """'+25 bps' / 'Unchanged' from the last two distinct values of a rate history."""
    distinct_vals = []
    for v in values:
        if not distinct_vals or distinct_vals[-1] != v:
            distinct_vals.append(v)
    if len(distinct_vals) >= 2:
        diff_bps = int(round((distinct_vals[-1] - distinct_vals[-2]) * 100))
        return f"{diff_bps:+} bps" if diff_bps != 0 else "Unchanged"
    return None


def _cb_usa(defaults):
    log = []
    start = history_start()
    got = run_parallel({
        "pce": lambda: fred_fetch_series("PCEPI", start),
        "lower": lambda: fred_fetch_series("DFEDTARL", start),
        "upper": lambda: fred_fetch_series("DFEDTARU", start),
        "unemp": lambda: fred_fetch_series("UNRATE", start),
    })
    pce_df, lower_df, upper_df, unemp_df = (got[k][0] for k in ("pce", "lower", "upper", "unemp"))
    usa_cpi, usa_rate, usa_change = defaults["cpi"], defaults["rate"], defaults["change"]

    infl = yoy_from_index(pce_df)
    if not infl.empty:
        usa_cpi = round(float(infl.iloc[-1]), 2)
        log.append(f"USA CPI (PCE YoY): {usa_cpi}% (FRED: PCEPI)")
    else:
        log.append("USA CPI: no data, using default value")

    if lower_df is not None and upper_df is not None and not lower_df.empty and not upper_df.empty:
        usa_rate = f"{float(lower_df.iloc[-1]['value']):.2f}% to {float(upper_df.iloc[-1]['value']):.2f}%"
        chg = _change_from_distinct(upper_df["value"].dropna().tolist())
        usa_change = chg or usa_change
        log.append(f"USA Fed Funds: {usa_rate} | Action: {usa_change}")
    else:
        log.append("USA Rate: no data, using default value")

    unemp = to_series(unemp_df)
    log.append(f"USA unemployment (FRED UNRATE): {unemp.iloc[-1]:.1f}% ({unemp.index[-1]:%Y-%m})"
               if not unemp.empty else "USA unemployment: no data")
    return {"cpi": usa_cpi, "rate": usa_rate, "change": usa_change,
            "hist": _hist_dict(infl, to_series(upper_df), unemp, {
                "infl": "FRED PCEPI (PCE price index), YoY",
                "rate": "FRED DFEDTARU (target range, upper bound)",
                "unemp": "FRED UNRATE"})}, log


def _ecb_unemployment_series():
    """Euro-area harmonised unemployment rate (monthly) from the ECB Data Portal."""
    start = (datetime.datetime.now() - datetime.timedelta(days=365 * (HISTORY_YEARS + 1))).strftime("%Y-%m")
    url = ("https://data-api.ecb.europa.eu/service/data/LFSI/"
           f"M.I9.S.UNEHRT.TOTAL0.15_74.T?format=csvdata&startPeriod={start}")
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["OBS_VALUE"])
    idx = pd.to_datetime(df["TIME_PERIOD"].astype(str) + "-01")
    return pd.Series(df["OBS_VALUE"].astype(float).values, index=idx).sort_index()


def _cb_euro(defaults):
    log = []
    start = history_start()
    got = run_parallel({
        "hicp": lambda: fred_fetch_series("CP0000EZ19M086NEST", start),
        "rate": lambda: fred_fetch_series("ECBDFR", start),
        "unemp_ecb": _ecb_unemployment_series,
        "unemp_fred": lambda: fred_fetch_series("LRHUTTTTEZM156S", start),
    })
    hicp_df, rate_df = got["hicp"][0], got["rate"][0]
    euro_cpi, euro_rate, euro_change = defaults["cpi"], defaults["rate"], defaults["change"]

    # CP0000EZ19M086NEST is a price INDEX (not a percentage), so the YoY % is computed here.
    infl = yoy_from_index(hicp_df)
    if not infl.empty:
        euro_cpi = round(float(infl.iloc[-1]), 2)
        log.append(f"EURO CPI (HICP YoY): {euro_cpi}% (calculated from index CP0000EZ19M086NEST)")
    else:
        log.append("EURO CPI: insufficient data, using default value")

    if rate_df is not None and not rate_df.empty:
        euro_rate = f"{float(rate_df.iloc[-1]['value']):.2f}%"
        euro_change = _change_from_distinct(rate_df["value"].dropna().tolist()) or euro_change
        log.append(f"ECB Deposit Rate: {euro_rate} | Action: {euro_change}")
    else:
        log.append("ECB Rate: no data, using default value")

    unemp, unemp_src = pd.Series(dtype=float), "n/a"
    ecb_u, ecb_err = got["unemp_ecb"]
    if ecb_u is not None and not ecb_u.empty:
        unemp, unemp_src = ecb_u, "ECB Data Portal (LFSI, harmonised unemployment rate)"
    else:
        fred_u = to_series(got["unemp_fred"][0])
        if not fred_u.empty:
            unemp, unemp_src = fred_u, "FRED LRHUTTTTEZM156S (OECD)"
        log.append(f"EURO unemployment: ECB source failed ({ecb_err}); FRED fallback "
                   f"{'used' if not fred_u.empty else 'also failed'}")
    if not unemp.empty:
        log.append(f"EURO unemployment: {unemp.iloc[-1]:.1f}% ({unemp.index[-1]:%Y-%m}) [{unemp_src}]")
    return {"cpi": euro_cpi, "rate": euro_rate, "change": euro_change,
            "hist": _hist_dict(infl, to_series(rate_df), unemp, {
                "infl": "FRED CP0000EZ19M086NEST (HICP index), YoY computed",
                "rate": "FRED ECBDFR (deposit facility rate)",
                "unemp": unemp_src})}, log


def _cb_uk(defaults):
    log = []
    got = run_parallel({
        "cpi": lambda: ons_months_series(
            ["https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g7/mm23/data",
             "https://api.ons.gov.uk/timeseries/D7G7/dataset/mm23/data"], "UK CPI", log),
        "unemp": lambda: ons_months_series(
            ["https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms/data",
             "https://api.ons.gov.uk/timeseries/MGSX/dataset/lms/data"], "UK unemployment", log),
        "boe": fetch_boe_bank_rate,
    })
    uk_cpi, uk_rate, uk_change = defaults["cpi"], defaults["rate"], defaults["change"]

    # The old v0 api.ons.gov.uk endpoint is being retired (404s were seen), so the
    # www.ons.gov.uk "/data" endpoint is tried first and the old URL kept as a fallback.
    infl, err = got["cpi"]
    if infl is not None and not infl.empty:
        uk_cpi = round(float(infl.iloc[-1]), 2)
        log.append(f"UK CPI (YoY): {uk_cpi}% (ONS: D7G7)")
    else:
        infl = None
        log.append(f"UK CPI error: {err}")

    boe, rate_hist = got["boe"][0], None
    if boe is not None:
        cur, prev, boe_df = boe
        uk_rate = f"{cur:.2f}%"
        if prev is not None and prev != cur:
            uk_change = f"{int(round((cur - prev) * 100)):+} bps"
        else:
            uk_change = "Unchanged"
        rate_hist = to_series(boe_df)
        log.append(f"BOE Bank Rate: {uk_rate} | Action: {uk_change}")
    else:
        log.append(f"BOE Rate error: {got['boe'][1] or 'no data'}")

    unemp, err = got["unemp"]
    if unemp is None or unemp.empty:
        log.append(f"UK unemployment error: {err}")
        unemp = None
    else:
        log.append(f"UK unemployment (ONS MGSX, 3-month average): {unemp.iloc[-1]:.1f}% ({unemp.index[-1]:%Y-%m})")
    return {"cpi": uk_cpi, "rate": uk_rate, "change": uk_change,
            "hist": _hist_dict(infl, rate_hist, unemp, {
                "infl": "ONS D7G7 (CPI YoY)",
                "rate": "Bank of England IADB IUDBEDR (Bank Rate)",
                "unemp": "ONS MGSX (3-month rolling average, labelled by its last month)"})}, log


def _cb_jpn(defaults):
    log = []
    start = history_start()
    got = run_parallel({
        "cpi": lambda: fetch_estat_cpi_series(log),
        "unemp": lambda: fetch_estat_unemployment_series(log),
        "boj": lambda: fetch_boj_rate_data(log),
        "cpi_fred": lambda: fred_fetch_series("CPALTT01JPM659N", start),
        "unemp_fred": lambda: fred_fetch_series("LRUNTTTTJPM156S", start),
    })
    jpn_cpi, jpn_rate, jpn_change = defaults["cpi"], defaults["rate"], defaults["change"]
    src = {"infl": "e-Stat CPI (nationwide, all items, YoY)", "rate": "n/a", "unemp": "n/a"}

    infl = got["cpi"][0]
    if infl is not None and not infl.empty:
        jpn_cpi = round(float(infl.iloc[-1]), 2)
    else:
        # Fallback: FRED (OECD-derived, may have stopped updating after 2021)
        infl = to_series(got["cpi_fred"][0])
        if not infl.empty:
            jpn_cpi = round(float(infl.iloc[-1]), 2)
            src["infl"] = "FRED CPALTT01JPM659N (OECD, may be stale)"
            log.append(f"JPN CPI (YoY, fallback): {jpn_cpi}% (FRED: CPALTT01JPM659N - may be stale)")
        else:
            infl = None
            log.append("JPN CPI: all sources failed, using default value")

    boj, boj_err = got["boj"]
    rate_hist = None
    if boj is not None:
        rate_hist = boj["series"]
        src["rate"] = boj["source"]
        if boj["current"] is not None:
            jpn_rate = f"{boj['current']:.2f}%"
            jpn_change = boj["action"]
            log.append(f"BOJ Policy Rate Target: {jpn_rate} | Last Action: {jpn_change} (Source: {boj['url']})")
    else:
        log.append(f"BOJ rate error: {boj_err}")

    # Preference: e-Stat seasonally adjusted > FRED/OECD (seasonally adjusted, if recent)
    # > e-Stat not seasonally adjusted > stale FRED
    est, err = got["unemp"]
    e_ser, e_adj = (est if est is not None else (None, False))
    fred_u = to_series(got["unemp_fred"][0])
    fred_fresh = (not fred_u.empty
                  and (pd.Timestamp.now() - fred_u.index[-1]).days <= 150)
    unemp = None
    if e_ser is not None and not e_ser.empty and e_adj:
        unemp, src["unemp"] = e_ser, "e-Stat Labour Force Survey (seasonally adjusted)"
    elif fred_fresh:
        unemp, src["unemp"] = fred_u, "FRED LRUNTTTTJPM156S (OECD, seasonally adjusted)"
    elif e_ser is not None and not e_ser.empty:
        unemp, src["unemp"] = e_ser, "e-Stat Labour Force Survey (NOT seasonally adjusted)"
    elif not fred_u.empty:
        unemp, src["unemp"] = fred_u, "FRED LRUNTTTTJPM156S (OECD, may be stale)"
    if unemp is None:
        log.append(f"JPN unemployment: all sources failed (e-Stat: {err})")
    else:
        log.append(f"JPN unemployment: {unemp.iloc[-1]:.1f}% ({unemp.index[-1]:%Y-%m}) [{src['unemp']}]")
    return {"cpi": jpn_cpi, "rate": jpn_rate, "change": jpn_change,
            "hist": _hist_dict(infl, rate_hist, unemp, src)}, log


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_cb_data():
    debug_log = []
    FRED_LAST_ERROR.clear()
    # Fully verified baseline defaults
    defaults = {
        "USA": {"cpi": 2.50, "rate": "3.75% to 4.00%", "change": "+25 bps"},
        "EURO": {"cpi": 2.20, "rate": "2.50%", "change": "+25 bps"},
        "UK": {"cpi": 3.40, "rate": "3.75%", "change": "Unchanged"},
        "JPN": {"cpi": 2.80, "rate": "1.25%", "change": "+25 bps"}
    }
    workers = {"USA": _cb_usa, "EURO": _cb_euro, "UK": _cb_uk, "JPN": _cb_jpn}
    got = run_parallel({code: (lambda f=f, c=code: f(defaults[c])) for code, f in workers.items()},
                       max_workers=4)

    results = {}
    for code in workers:
        res, err = got[code]
        if err is not None:
            debug_log.append(f"{code}: fetch failed entirely ({err}); defaults used")
            results[code] = {**defaults[code], "hist": _hist_dict()}
        else:
            results[code], log = res
            debug_log.extend(log)
    for sid, err in sorted(FRED_LAST_ERROR.items()):
        debug_log.append(f"FRED {sid}: failed after retries ({err})")
    results["_debug"] = debug_log
    # Remember whether anything failed, so a bad result is not kept for the full 12 hours.
    results["_complete"] = not FRED_LAST_ERROR
    results["_ts"] = time.time()
    return results


def get_cb_data():
    """fetch_cb_data(), but a result with failed FRED series is refetched after 3 minutes
    instead of being cached for 12 hours."""
    d = fetch_cb_data()
    if not d.get("_complete", True) and time.time() - d.get("_ts", 0) > 180:
        fetch_cb_data.clear()
        d = fetch_cb_data()
    return d

# Original personalized greeting to Antonio
def get_greeting(name: str = "Antonio") -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return f"Good morning, {name}! ☀️"
    elif hour < 18:
        return f"Good afternoon, {name}! 🌤️"
    else:
        return f"Good evening, {name}! 🌙"

# Emergency Fast-Path Rate Overrides (Optional)
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
    cb_live = get_cb_data()
    cb_live = apply_manual_overrides(cb_live)

with st.expander("🔍 Data Fetch Status (Diagnostics)", expanded=False):
    st.caption("Shows whether each metric was retrieved live from its source, or fell back to a default value.")
    for line in cb_live.get("_debug", []):
        st.text(line)

# Central Bank Configurations Mapping
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


# ----------------------------------------------------
# History charts (3 years): inflation vs policy rate, and unemployment
# ----------------------------------------------------
COUNTRY_SHORT = {"USA": "U.S.", "EURO": "Euro area", "UK": "U.K.", "JPN": "Japan"}


def _chart_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.0, xanchor="left", y=0.97, yanchor="top"),
        height=CHART_HEIGHT,
        margin=dict(l=5, r=15, t=45, b=75),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, sans-serif", size=11),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5,
                    font=dict(size=11)),
        hovermode="x unified",
    )
    # One label every 6 months, on two horizontal lines ("Jan" / "2024")
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", tickformat="%b<br>%Y",
                     dtick="M6", tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", ticksuffix="%")
    return fig


def _empty_chart(title, message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="#888888"))
    fig = _chart_layout(fig, title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _window():
    end = pd.Timestamp.now().normalize()
    # a little space on the right so the last value label is not cut off
    return end - pd.DateOffset(years=HISTORY_YEARS), end + pd.Timedelta(days=30)


def build_trend_chart(code, hist):
    """Inflation (YoY) and policy rate over the last 3 years, one chart."""
    title = f"<b>{COUNTRY_SHORT[code]}</b>: inflation vs policy rate ({HISTORY_YEARS}Y)"
    infl, rate = hist.get("infl"), hist.get("rate")
    has_infl = infl is not None and not infl.empty
    has_rate = rate is not None and not rate.empty
    if not has_infl and not has_rate:
        return _empty_chart(title, "Data unavailable (see Data Fetch Status)")

    fig = go.Figure()
    if has_infl:
        fig.add_trace(go.Scatter(
            x=infl.index, y=infl.values, mode="lines", name="Inflation (YoY)",
            line=dict(color="#c62828", width=2),
            hovertemplate="%{y:.2f}%"))
    if has_rate:
        r = rate.copy()
        today = pd.Timestamp.now().normalize()
        if r.index[-1] < today:                      # draw the last level up to today
            r.loc[today] = r.iloc[-1]
        fig.add_trace(go.Scatter(
            x=r.index, y=r.values, mode="lines", name="Policy rate",
            line=dict(color="#1a5276", width=2, shape="hv"),
            hovertemplate="%{y:.2f}%"))
    fig = _chart_layout(fig, title)
    x0, x1 = _window()
    fig.update_xaxes(range=[x0, x1])
    return fig


def build_unemployment_chart(code, hist):
    """Unemployment rate over the last 3 years with a green full-employment zone.

    The zone is solid from the bottom of the chart up to `solid_to`, then fades out
    towards `fade_to` (stacked translucent bands), so there is no hard edge.
    """
    title = f"<b>{COUNTRY_SHORT[code]}</b>: unemployment rate ({HISTORY_YEARS}Y)"
    s_ = hist.get("unemp")
    if s_ is None or s_.empty:
        return _empty_chart(title, "Data unavailable (see Data Fetch Status)")

    zone = FULL_EMPLOYMENT_ZONES[code]
    solid_to, fade_to = zone["solid_to"], zone["fade_to"]
    lo = max(0.0, min(float(s_.min()), solid_to) - 0.6)
    hi = max(float(s_.max()), fade_to) + 0.5

    fig = go.Figure()
    max_alpha, bands = 0.32, 16

    def band(y0, y1, alpha):
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=f"rgba(46,125,50,{alpha:.3f})", line_width=0, layer="below")

    band(lo, solid_to, max_alpha)
    for i in range(bands):
        y0 = solid_to + (fade_to - solid_to) * i / bands
        y1 = solid_to + (fade_to - solid_to) * (i + 1) / bands
        band(y0, y1, max_alpha * (1 - (i + 0.5) / bands))

    fig.add_trace(go.Scatter(
        x=s_.index, y=s_.values, mode="lines+markers", name="Unemployment rate",
        line=dict(color="#222222", width=2), marker=dict(size=4),
        hovertemplate="%{y:.1f}%"))
    fig.add_annotation(x=s_.index[-1], y=float(s_.iloc[-1]), text=f"<b>{s_.iloc[-1]:.1f}%</b>",
                       showarrow=False, yshift=12, font=dict(size=11, color="#222222"))
    fig.add_annotation(x=0.01, y=(lo + solid_to) / 2, xref="paper", yref="y", xanchor="left",
                       text=f"Full-employment zone (est. {solid_to:.1f}-{fade_to:.1f}%)",
                       showarrow=False, font=dict(size=10, color="#2e7d32"))
    fig = _chart_layout(fig, title)
    x0, x1 = _window()
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[lo, hi])
    fig.update_layout(showlegend=False)
    return fig


def render_history_charts(config):
    """Two charts under a country's gauge row: inflation vs policy rate, and unemployment."""
    code, hist = config["code"], config.get("hist") or _hist_dict()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_trend_chart(code, hist), use_container_width=True,
                        key=f"trend_{code}", config={"displayModeBar": False})
    with c2:
        st.plotly_chart(build_unemployment_chart(code, hist), use_container_width=True,
                        key=f"unemp_{code}", config={"displayModeBar": False})
    src = hist.get("src", {})
    u = hist.get("unemp")
    latest = (f" Latest unemployment: {u.iloc[-1]:.1f}% ({u.index[-1]:%b %Y})."
              if u is not None and not u.empty else "")
    zone = FULL_EMPLOYMENT_ZONES[code]
    st.caption(
        f"Sources: inflation - {src.get('infl', 'n/a')}; policy rate - {src.get('rate', 'n/a')}; "
        f"unemployment - {src.get('unemp', 'n/a')}.{latest} Green band = rough estimate of the "
        f"full-employment range ({zone['solid_to']:.1f}-{zone['fade_to']:.1f}%), not an official figure.")


# Render Central Bank & Yield Row (Restored original style and link labels)
for _cfg in cb_configs.values():
    _cfg["hist"] = cb_live[_cfg["code"]].get("hist")


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

    render_history_charts(config)

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
# 3. Benchmark Market Data & Candlestick Charts (yfinance)
# ----------------------------------------------------
market_tickers = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X",
    "DJIA": "^DJI", "S&P 500": "^GSPC", "DAX": "^GDAXI", "FTSE 100": "^FTSE", "Nikkei 225": "^N225",
    "Gold": "GC=F", "WTI Crude": "CL=F"
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_ema_data():
    def one(ticker):
        df = yf.Ticker(ticker).history(period="1y")
        if df.empty or len(df) < 144:
            return None
        df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()
        df['EMA_144'] = df['Close'].ewm(span=144, adjust=False).mean()

        latest_price = float(df['Close'].iloc[-1])
        ema_55 = float(df['EMA_55'].iloc[-1])
        ema_144 = float(df['EMA_144'].iloc[-1])
        return {
            "latest": latest_price, "ema_55": ema_55,
            "dev_55": ((latest_price - ema_55) / ema_55) * 100,
            "ema_144": ema_144,
            "dev_144": ((latest_price - ema_144) / ema_144) * 100,
            "df": df
        }

    # The 10 tickers are independent, so they are downloaded concurrently
    got = run_parallel({name: (lambda t=t: one(t)) for name, t in market_tickers.items()},
                       max_workers=10)
    return {name: res for name, (res, _) in got.items() if res is not None}

market_data = fetch_market_ema_data()

def build_candlestick_chart(name: str, df: pd.DataFrame):
    """Builds a 1-year daily candlestick chart with 55 EMA and 144 EMA overlays."""
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
        xaxis=dict(
            showgrid=True,
            gridcolor='#eeeeee',
            nticks=6
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eeeeee',
            side='right'
        )
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


# Foreign Exchange Section (Restored exact original header + Candles added below)
st.subheader("💱 Foreign Exchange")
f_cols = st.columns(3)
for i, name in enumerate(["EUR/USD", "USD/JPY", "GBP/USD"]):
    if name in market_data:
        with f_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

# Equity Indices Section (Restored exact original header + Candles added below)
st.subheader("📈 Major Equity Indices")
i_cols = st.columns(5)
for i, name in enumerate(["DJIA", "S&P 500", "DAX", "FTSE 100", "Nikkei 225"]):
    if name in market_data:
        with i_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

# Commodities Section (Restored exact original header + Candles added below)
st.subheader("🛢️ Commodities")
c_cols = st.columns(2)
for i, name in enumerate(["Gold", "WTI Crude"]):
    if name in market_data:
        with c_cols[i]:
            render_market_card(name, market_data[name])
            st.plotly_chart(build_candlestick_chart(name, market_data[name]["df"]), use_container_width=True)

st.divider()

# Scenario Analysis Section (Restored exact original header)
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