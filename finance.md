# Finance Bundle
> Skills: Alpha Vantage · FRED · EDGARTools · Hedge Fund Monitor · US Fiscal Data
> Usage: Paste this URL at session start for all financial data tasks.

---

## 1. Alpha Vantage — Market Data
**Trigger:** "stock price", "forex", "crypto", "technical indicators", "company fundamentals"
**Requires:** Free API key → https://alphavantage.co (25 req/day free)

```python
import requests, os
API_KEY = os.environ["ALPHAVANTAGE_API_KEY"]
BASE = "https://www.alphavantage.co/query"

def av(function, **params):
    r = requests.get(BASE, params={"function": function, "apikey": API_KEY, **params})
    return r.json()

# Stock quote
price = av("GLOBAL_QUOTE", symbol="AAPL")["Global Quote"]["05. price"]

# Daily OHLCV (compact=last 100, full=20yr)
daily = av("TIME_SERIES_DAILY", symbol="AAPL", outputsize="compact")

# Fundamentals
overview = av("OVERVIEW", symbol="AAPL")        # P/E, market cap, etc.
income   = av("INCOME_STATEMENT", symbol="AAPL")["annualReports"][0]
balance  = av("BALANCE_SHEET", symbol="AAPL")["annualReports"][0]

# Technical indicators
rsi  = av("RSI",   symbol="AAPL", interval="daily", time_period=14, series_type="close")
macd = av("MACD",  symbol="AAPL", interval="daily", series_type="close")
bb   = av("BBANDS", symbol="AAPL", interval="daily", time_period=20)

# Crypto / Forex
btc = av("DIGITAL_CURRENCY_DAILY", symbol="BTC", market="USD")
fx  = av("CURRENCY_EXCHANGE_RATE", from_currency="USD", to_currency="EUR")

# Economic
gdp  = av("REAL_GDP", interval="annual")
cpi  = av("CPI", interval="monthly")
```

**Error handling:** check for `"Error Message"`, `"Note"` (rate limit), `"Information"` keys.

---

## 2. FRED — Federal Reserve Economic Data
**Trigger:** "GDP", "unemployment", "inflation", "interest rates", "economic indicators"
**Requires:** Free API key → https://fredaccount.stlouisfed.org (800K+ series)

```python
import requests, os
API_KEY = os.environ["FRED_API_KEY"]
BASE = "https://api.stlouisfed.org/fred"

def fred(endpoint, **params):
    r = requests.get(f"{BASE}/{endpoint}",
                     params={"api_key": API_KEY, "file_type": "json", **params})
    return r.json()

# Key series IDs
# GDP · GDPC1 · UNRATE · CPIAUCSL · FEDFUNDS · DGS10
# HOUST · PAYEMS · M2SL · SP500 · UMCSENT

# Get observations
obs = fred("series/observations", series_id="UNRATE", limit=24)
for o in obs["observations"]: print(o["date"], o["value"])

# With transformation (pch=% change, pc1=% change yr ago, log=log)
gdp_growth = fred("series/observations", series_id="GDP", units="pc1")

# Search
results = fred("series/search", search_text="consumer price index",
               filter_variable="frequency", filter_value="Monthly")

# Regional data (state unemployment map)
regional = fred("geofred/regional/data",
                series_group="1220", region_type="state",
                date="2024-01-01", units="Percent", season="NSA")
```

**Aggregation:** add `frequency="m"` + `aggregation_method="avg"` to downsample.
**Vintage data:** add `realtime_start` + `realtime_end` to get historical snapshots.

---

## 3. EDGARTools — SEC Filings
**Trigger:** "SEC filing", "10-K", "10-Q", "insider trades", "13F", "financial statements"
**Requires:** `uv pip install edgartools` · No API key (identity required)

```python
from edgar import set_identity, Company, get_filings
set_identity("Your Name your@email.com")  # Required by SEC

# Find company
co = Company("AAPL")          # by ticker
co = Company(320193)           # by CIK (fastest)

# Get filings
tenk   = co.get_filings(form="10-K").latest()
tenq   = co.get_filings(form="10-Q").latest()
form4  = co.get_filings(form="4").latest()   # insider trades
f13    = co.get_filings(form="13F-HR").latest()  # institutional holdings

# Extract structured data
obj = tenk.obj()                    # → TenK object
financials = obj.financials
income  = financials.income_statement()
balance = financials.balance_sheet()
cashflow = financials.cashflow_statement()

# Or use convenience methods
financials = co.get_financials()           # annual
financials = co.get_quarterly_financials() # quarterly

# Filing content
text = tenk.text()       # plain text
md   = tenk.markdown()   # markdown (best for LLM)
```

**Key pitfall:** `filing.financials` → AttributeError · use `filing.obj().financials`
**Form → object map:** 10-K→TenK · 10-Q→TenQ · 8-K→EightK · Form4→Form4 · 13F→ThirteenF

---

## 4. Hedge Fund Monitor — OFR API
**Trigger:** "hedge fund data", "leverage ratio", "AUM", "repo volumes", "systemic risk"
**Requires:** No API key · Free · `https://data.financialresearch.gov/hf/v1`

```python
import requests, pandas as pd
BASE = "https://data.financialresearch.gov/hf/v1"

# Search available series
results = requests.get(f"{BASE}/metadata/search",
                       params={"query": "*leverage*"}).json()

# Fetch time series
r = requests.get(f"{BASE}/series/timeseries", params={
    "mnemonic": "FPF-ALLQHF_LEVERAGERATIO_GAVWMEAN",
    "start_date": "2015-01-01"
}).json()
df = pd.DataFrame(r, columns=["date", "value"])

# Key mnemonics (FPF = Form PF, quarterly)
# FPF-ALLQHF_LEVERAGERATIO_GAVWMEAN  — all funds leverage (gross asset-weighted)
# FPF-ALLQHF_GAV_SUM                 — gross assets total
# FPF-ALLQHF_NAV_SUM                 — net assets total
# FPF-ALLQHF_GNE_SUM                 — gross notional exposure
# FICC-SPONSORED_REPO_VOL            — sponsored repo volume (monthly)
```

**Datasets:** `fpf` (Form PF, quarterly) · `tff` (CFTC futures, monthly) · `scoos` (FRB survey) · `ficc` (repo, monthly)
**Params:** `start_date`, `end_date`, `periodicity` (Q/M/A), `how` (last/mean/sum)

---

## 5. US Fiscal Data — Treasury API
**Trigger:** "national debt", "treasury", "government spending", "federal revenue", "savings bonds"
**Requires:** No API key · Free · `https://api.fiscaldata.treasury.gov/services/api/fiscal_service`

```python
import requests, pandas as pd
BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

def treasury(endpoint, **params):
    r = requests.get(f"{BASE}/{endpoint}", params=params)
    return r.json()

# National debt (daily)
debt = treasury("v2/accounting/od/debt_to_penny",
                sort="-record_date", **{"page[size]": 1})
print(f"Total debt: ${float(debt['data'][0]['tot_pub_debt_out_amt']):,.0f}")

# Exchange rates (quarterly)
fx = treasury("v1/accounting/od/rates_of_exchange",
              fields="country_currency_desc,exchange_rate,record_date",
              filter="record_date:gte:2024-01-01",
              sort="-record_date", **{"page[size]": 100})
df = pd.DataFrame(fx["data"])

# Average interest rates on Treasury securities (monthly)
rates = treasury("v2/accounting/od/avg_interest_rates",
                 sort="-record_date", **{"page[size]": 12})

# I-Bond interest rates
ibonds = treasury("v2/accounting/od/i_bond_interest_rates")
```

**Key endpoints:**
| Data | Endpoint |
|------|----------|
| Debt to the Penny | `v2/accounting/od/debt_to_penny` |
| Exchange Rates | `v1/accounting/od/rates_of_exchange` |
| Avg Interest Rates | `v2/accounting/od/avg_interest_rates` |
| Daily Cash Balance | `v1/accounting/dts/operating_cash_balance` |
| Treasury Auctions | `v1/accounting/od/auctions_query` |

**Filter operators:** `lt` `lte` `gt` `gte` `eq` `in` · All values returned as strings → cast with `float()`.
