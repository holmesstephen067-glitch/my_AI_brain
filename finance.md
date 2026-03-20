# Finance Bundle
> Skills: Alpha Vantage · FRED · EDGARTools · Hedge Fund Monitor · US Fiscal Data
> Trading: Polygon · Finnhub · Intermarket Flow · Carry Unwind · XGBoost ML · Options
> Usage: Paste this URL at session start for all financial data tasks.

---

## TRADING SYSTEM QUICK REFERENCE

### Portfolio
| Ticker | Shares | Avg Cost | CC Contracts |
|--------|--------|----------|--------------|

**Portfolio value:** $xxx | **Buying power:** $xxx

### API Keys
| API | Key | Use |
|-----|-----|-----|
| Polygon.io | `POLYGON_KEY` | Bars, options chain, VWAP, snapshot |
| Alpha Vantage | `AV_KEY` | RSI, MACD, CCI, ADX, ATR, EMA, STOCH, OVERVIEW |
| Finnhub | `FINNHUB_KEY` | Quotes, news, insider sentiment, earnings |
| FRED | `FRED_KEY` | VIX, yield curve, CPI, oil, GDP, Wilshire |

### Bot Files (GitHub)
```
trading_bot/signal_engine.py   — full 10-indicator stack + XGBoost + ARIMA
trading_bot/backtester.py      — VectorBT RSI optimizer + covered call backtest
trading_bot/alerts.py          — Telegram morning brief scheduler
trading_bot/README.md          — full documentation + roadmap
```

---

## 1. Alpha Vantage — Market Data + All 10 Indicators
**Trigger:** "stock price", "forex", "crypto", "technical indicators", "company fundamentals"
**Requires:** Free API key → https://alphavantage.co (25 req/day free)
**Rate limit:** 25 calls/day on free tier — prioritize top 3–5 tickers per scan

```python
import requests
AV_KEY = "YOUR_AV_KEY"  # loaded from Claude Project instructions
BASE   = "https://www.alphavantage.co/query"

def av(function, **params):
    r = requests.get(BASE, params={"function": function, "apikey": API_KEY, **params})
    return r.json()

# ── INDICATORS (all 10) ───────────────────────────────────────────────────────

# 1. RSI — normalized: <30=90pts, 30-40=70, 40-50=55, 50-60=45, 60-70=30, >70=10
rsi  = av("RSI",   symbol="NVDA", interval="daily", time_period=14, series_type="close")

# 2. MACD — bullish crossover = +signal, histogram expanding = confirmed
macd = av("MACD",  symbol="NVDA", interval="daily", series_type="close")

# 3. Bollinger Bands — near lower = long bias, near upper = overbought
bb   = av("BBANDS", symbol="NVDA", interval="daily", time_period=20, series_type="close")

# 4. ADX — >25 = strong trend (trade), <20 = ranging (use spreads/condors)
adx  = av("ADX",   symbol="NVDA", interval="daily", time_period=14)

# 5. Stochastic (NEW) — %K crosses %D below 20 = buy | above 80 = sell
stoch = av("STOCH", symbol="NVDA", interval="daily",
           fastkperiod=5, slowkperiod=3, slowdperiod=3)

# 6. CCI(50) + CCI(5) — below -100 + CCI(5) crossing up = strong buy
cci50 = av("CCI", symbol="NVDA", interval="daily", time_period=50)
cci5  = av("CCI", symbol="NVDA", interval="daily", time_period=5)

# 7. ATR — stop = entry - (1.5 × ATR). Replaces flat 7% stop
atr  = av("ATR",   symbol="NVDA", interval="daily", time_period=14)

# 8. EMA 20/50/200 — price above all 3 = uptrend. Golden cross (50>200) = buy
ema20  = av("EMA", symbol="NVDA", interval="daily", time_period=20,  series_type="close")
ema50  = av("EMA", symbol="NVDA", interval="daily", time_period=50,  series_type="close")
ema200 = av("EMA", symbol="NVDA", interval="daily", time_period=200, series_type="close")

# 9. Volume — via Polygon bars (see Section 3)

# 10. Relative Strength vs SPY — via Polygon 30-day returns (see Section 3)

# ── FUNDAMENTALS (Buffett lens — now with actual numbers) ────────────────────

# P/E, PEG, EPS growth, profit margins, analyst targets
overview = av("OVERVIEW", symbol="NVDA")
# Key fields: PERatio, PEGRatio, EPS, QuarterlyEarningsGrowthYOY,
#             ProfitMargin, AnalystTargetPrice, Beta, 52WeekHigh/Low

# Income statement — EPS trend, revenue growth
income   = av("INCOME_STATEMENT", symbol="NVDA")["annualReports"][0]

# Earnings calendar — confirms or disputes Finnhub
earnings_cal = av("EARNINGS_CALENDAR", symbol="NVDA", horizon="3month")

# ARIMA / forecasting inputs — daily close series
daily = av("TIME_SERIES_DAILY", symbol="NVDA", outputsize="full")

# Forex pairs — for USD/JPY, EUR/USD, AUD/USD monitoring
fx_daily = av("FX_DAILY", from_symbol="USD", to_symbol="JPY")
fx_rate  = av("CURRENCY_EXCHANGE_RATE", from_currency="USD", to_currency="JPY")

# Stochastic RSI — more sensitive, better for crypto swing trades
stochrsi = av("STOCHRSI", symbol="BTCUSD", interval="daily",
              time_period=14, series_type="close", fastkperiod=5, fastdperiod=3)
```

**Scoring rules:**
| Stochastic | Signal |
|------------|--------|
| %K crosses above %D below 20 | 🔥 Strong buy — oversold reversal |
| %K crosses below %D above 80 | 🔥 Strong sell |
| Both below 20, turning up | Confirming buy |
| Divergence from price | Reversal warning |

**Error handling:** check for `"Error Message"`, `"Note"` (rate limit), `"Information"` keys.

---

## 2. FRED — Federal Reserve Economic Data + Macro Snapshot
**Trigger:** "GDP", "unemployment", "inflation", "interest rates", "macro snapshot"
**Requires:** Free API key → https://fredaccount.stlouisfed.org

```python
import requests
FRED_KEY = "YOUR_FRED_KEY"  # loaded from Claude Project instructions
BASE     = "https://api.stlouisfed.org/fred"

def fred(series_id, limit=1):
    r = requests.get(f"{BASE}/series/observations", params={
        "series_id": series_id, "sort_order": "desc",
        "limit": limit, "api_key": FRED_KEY, "file_type": "json"
    })
    obs = r.json()["observations"]
    return float(obs[0]["value"]) if obs else None

# ── FULL MACRO SNAPSHOT (run every scan) ─────────────────────────────────────
macro = {
    "vix":           fred("VIXCLS"),          # VIX — regime classifier
    "yield_curve":   fred("T10Y2Y"),           # Inverted = recession risk 🟣
    "fed_rate":      fred("FEDFUNDS"),         # Current Fed funds rate
    "cpi":           fred("CPIAUCSL"),         # Inflation — rising = bearish growth
    "core_cpi":      fred("CPILFESL"),         # Core CPI
    "pce":           fred("PCEPI"),            # PCE — Fed's preferred inflation gauge
    "unemployment":  fred("UNRATE"),           # Jobs market
    "oil_wti":       fred("DCOILWTICO"),       # Crude — LEADS stocks by 1-3 sessions
    "gold":          fred("GOLDAMGBD228NLBM"), # Safe haven signal
    "mortgage":      fred("MORTGAGE30US"),     # Relevant for SOFI, HOOD
    "nat_gas":       fred("DHHNGSP"),          # Iran war — LNG export risk
    # Buffett Indicator components (NEW)
    "wilshire5000":  fred("WILL5000PR"),       # Total market cap proxy
    "gdp":           fred("GDP"),              # Latest GDP reading
}

# Buffett Indicator = market cap / GDP × 100
# <80%  = undervalued | 80-100% = fair | 100-120% = slightly over
# >120% = overvalued  | >200%   = "playing with fire" (Buffett)
buffett = (macro["wilshire5000"] / macro["gdp"]) * 100

# ── REGIME CLASSIFICATION ─────────────────────────────────────────────────────
# SPY vs 200 EMA | VIX     | Yield Curve | Regime          | Strategy
# Above           | <20     | Positive    | 🟢 Bull Trend   | Buy calls / Sell puts
# Above           | 20-30   | Any         | 🟡 Choppy Bull  | Spreads / Covered calls
# Below           | 20-30   | Any         | 🟠 Bear/Sideways | Buy puts / Credit spreads
# Any             | >30     | Any         | 🔴 Vol Spike    | Iron condors / Cash
# Any             | Any     | Negative    | 🟣 Recession    | Defensive

# ── YIELD CURVE DEEP DIVE ─────────────────────────────────────────────────────
t10y2y  = fred("T10Y2Y")    # 10yr minus 2yr — key inversion signal
t10yie  = fred("T10YIE")    # 10yr inflation expectations

# Aggregation helpers
def fred_series(series_id, limit=30):
    r = requests.get(f"{BASE}/series/observations", params={
        "series_id": series_id, "sort_order": "desc",
        "limit": limit, "api_key": FRED_KEY, "file_type": "json"
    })
    return r.json()["observations"]
```

**Macro regime output format:**
```
🌡️ MACRO SNAPSHOT
Fed rate:      X.XX%
CPI:           X.X% YoY
Yield curve:   +/-X.X bps [normal/flat/inverted 🔴]
VIX:           XX.X [low/elevated/high/extreme]
Oil WTI:       $XX.XX [war premium / normal]
Gold:          $X,XXX [safe haven bid / neutral]
Buffett:       XXX% [undervalued / fair / overvalued / extreme 🔴]
Macro regime:  [Risk-on / Neutral / Risk-off / Stagflation]
```

---

## 3. Polygon.io — Bars, Options, VWAP, Snapshot
**Trigger:** "live price", "options chain", "historical bars", "VWAP", "volume"
**Key:** zgI7pxcaymBCYt8NsPEp35FCJvhAksXz

```python
import requests, pandas as pd
POLY_KEY = "YOUR_POLYGON_KEY"  # loaded from Claude Project instructions

# ── HISTORICAL BARS ────────────────────────────────────────────────────────────
def polygon_bars(ticker, days=400):
    import pandas as pd
    end   = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    url   = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
             f"{start}/{end}?adjusted=true&sort=asc&limit=500&apiKey={POLY_KEY}")
    r = requests.get(url, timeout=10)
    return r.json().get("results", [])

# ── LIVE SNAPSHOT ──────────────────────────────────────────────────────────────
def polygon_snapshot(ticker):
    url = (f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/"
           f"tickers/{ticker}?apiKey={POLY_KEY}")
    return requests.get(url).json()

# ── OPTIONS CHAIN (full) ──────────────────────────────────────────────────────
def options_chain(ticker, limit=50):
    url = (f"https://api.polygon.io/v3/snapshot/options/{ticker}"
           f"?limit={limit}&apiKey={POLY_KEY}")
    return requests.get(url).json()

# ── RELATIVE STRENGTH vs SPY (30-day) ─────────────────────────────────────────
# RS > 1.0 = outperforming = long bias
# RS < 1.0 = underperforming = avoid long / short candidate
def calc_rs_spy(ticker_bars, spy_bars):
    ticker_ret = (ticker_bars[-1]["c"] - ticker_bars[-30]["c"]) / ticker_bars[-30]["c"]
    spy_ret    = (spy_bars[-1]["c"]    - spy_bars[-30]["c"])    / spy_bars[-30]["c"]
    return ticker_ret / spy_ret if spy_ret != 0 else 1.0

# ── EARNINGS CALENDAR (free, no key) ──────────────────────────────────────────
def nasdaq_earnings(date_str):
    # date_str format: "2026-03-27"
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={date_str}"
    return requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
```

**Options liquidity filter (mandatory — skip illiquid):**
```
Bid/ask spread < 5% of mid price ✅
Open interest > 500 ✅
Volume today > 100 ✅
Volume/OI > 1.0 = new positions opening = smart money signal
```

---

## 4. Finnhub — Quotes, News, Insider, Earnings
**Trigger:** "live quote", "company news", "insider buying", "earnings date"
**Key:** d6r2q5pr01qgdhqcut90d6r2q5pr01qgdhqcut9g

```python
import requests
FH_KEY = "YOUR_FINNHUB_KEY"  # loaded from Claude Project instructions

def finnhub(endpoint, **params):
    r = requests.get(f"https://finnhub.io/api/v1/{endpoint}",
                     params={"token": FH_KEY, **params})
    return r.json()

# Live quote
quote    = finnhub("quote", symbol="TSLA")
# Returns: c=current, h=high, l=low, o=open, pc=prev close, dp=% change

# Company news (scan for earnings/buyback/CEO change)
news     = finnhub("company-news", symbol="NVDA",
                   **{"from": "2026-03-12", "to": "2026-03-19"})
# Flag: "earnings","guidance","buyback","CEO","acquisition","dividend"

# Insider sentiment — net buying = bullish Buffett signal
insider  = finnhub("stock/insider-sentiment", symbol="TSLA")

# Earnings calendar
earnings = finnhub("calendar/earnings",
                   **{"from": "2026-03-19", "to": "2026-03-27"})

# Recommendation trends — analyst consensus
recs     = finnhub("stock/recommendation", symbol="NVDA")
# strongBuy, buy, hold, sell, strongSell counts

# ── EVENT DETECTION (override normal analysis) ────────────────────────────────
# Earnings within 5 days → flag, avoid new entries
# Guidance change → strong directional signal
# CEO/CFO change  → bearish until confirmed stable
# Buyback         → bullish signal
# M&A target      → immediate spike play
```

---

## 5. Intermarket Flow Analysis
**Trigger:** "market direction", "macro bias", "crude oil", "dollar", "flow"

### Flow Sequence (always run in this order)
```
WTI Crude → US Dollar (DXY) → USD/JPY 160 gauge → Stocks
```

```python
# Panel 1 — Crude Oil proxy
uso_bars = polygon_bars("USO", days=90)
wti_spot = fred("DCOILWTICO")

# Panel 2 — Dollar Index proxy
uup_bars = polygon_bars("UUP", days=90)

# Panel 3 — USD/JPY (forex — Alpha Vantage)
usdjpy = av("FX_DAILY", from_symbol="USD", to_symbol="JPY")

# Intermarket Signal Matrix
# Crude ↑ + DXY ↓ = Risk-on  ✅ → long equities
# Crude ↑ + DXY ↑ = Stagflation 🟠 → defensive
# Crude ↓ + DXY ↑ = Deflationary ❌ → bearish
# Crude ↓ + DXY ↓ = Recession 🔴 → cash
```

### USD/JPY 160 — Risk Premium Gauge
```
Rate approaching 160, stalling, heading south WITHOUT retest
= Risk premium LEAVING the market = carry unwind signal

Rate breaks cleanly through 160 (close above + volume)
= Risk premium still IN the market = risk-on confirmed

Fetch: av("FX_DAILY", from_symbol="USD", to_symbol="JPY")
Or search: "USD/JPY rate today"
```

### Echo Lag Timing
| Crude Signal Age | Action |
|-----------------|--------|
| 0–1 days | Echo incoming — BEST entry window |
| 2–3 days | Echo in progress — enter if technicals confirm |
| 4–5 days | Likely priced in — reduce size |
| 5+ days | Wait for next crude inflection |

---

## 6. Carry Unwind Early Warning System
**Trigger:** "risk assessment", "market stress", "yen carry", "unwind risk"

```python
# 7-signal scoring system — run every scan
# Score 0-3:  🟢 No risk — trade normally
# Score 4-7:  🟡 Early warning — reduce size 25%
# Score 8-12: 🟠 Elevated — no new longs, tighten stops
# Score 13+:  🔴 Unwind in progress — defensive mode

def carry_unwind_score(macro):
    score = 0

    # Signal 1: USD/JPY at 160 wall (search or AV FX)
    # Stalling at 160 no retest = +3

    # Signal 2: BOJ hawkish signals
    # Search: "BOJ rate decision today"
    # Hawkish statement = +3

    # Signal 3: Nikkei -2%+ session
    # Search: "Nikkei 225 today"
    # Single session -2% = +2

    # Signal 4: VIX velocity
    vix = macro.get("vix", 20)
    if vix > 30:   score += 3
    elif vix > 25: score += 3
    elif vix > 20: score += 2

    # Signal 5: BTC leading down
    # BTC -5% while SPY flat = +3
    # Crypto F&G falling rapidly = +1

    # Signal 6: Cross-asset correlation break
    # BTC + Nikkei + SPY + crude all dropping = +3

    # Signal 7: BOJ meeting within 7 days = +1 (pre-emptive)
    # Search: "Bank of Japan meeting schedule"

    return score

# August 2024 template (reference pattern):
# Day -7: BOJ JGB yields drift up. USD/JPY stalls ~155.
# Day -4: Nikkei underperforms SPY. BTC -8% in 2 sessions.
# Day -2: USD/JPY breaks 152. VIX 19. BTC -6% more.
# Day -1: BOJ surprise hike. USD/JPY collapses through 150.
# Day 0:  VIX 65. Nikkei -12.4%. BTC -20%. SPY -4.25%.
```

---

## 7. XGBoost ML Signal Engine
**Trigger:** "ML prediction", "probability", "signal", "train model"
**Full code:** trading_bot/signal_engine.py

```python
# Quick reference — key functions from signal_engine.py

from signal_engine import (
    fetch_polygon_bars,      # Polygon OHLCV data
    calculate_indicators,    # All 10 indicators via pandas-ta
    build_features,          # Feature matrix + target variable
    train_xgboost,           # XGBoost with TimeSeriesSplit CV
    predict_proba,           # P(price higher in 5 days)
    arima_forecast,          # ARIMA(2,1,2) 5-day price forecast
    calculate_tps,           # ML-enhanced TPS score
    analyze_covered_call,    # Covered call income calculator
    calculate_position_size, # ATR-based position sizing
    carry_unwind_score,      # 0-21 carry unwind risk score
    detect_regime,           # 5-class market regime classifier
    run_full_scan,           # Master scan — runs everything
    print_report,            # Formatted trade block output
)

# Full scan — run at session start
scan = run_full_scan(tickers=["TSLA","AMD","NVDA","SOFI","AMZN","HOOD"])
print_report(scan)

# Single ticker quick check
df     = fetch_polygon_bars("TSLA", days=400)
df     = calculate_indicators(df)
df_ml, features = build_features(df, forward_days=5)
model  = train_xgboost(df_ml, features, "TSLA")
prob   = predict_proba(model, df_ml, features)
arima  = arima_forecast(df, "TSLA", steps=5)
print(f"TSLA — ML prob up: {prob:.1%} | ARIMA 5d: {arima['forecast_5d']}")
```

**ML weights (data-learned, not hand-coded):**
```
XGBoost (all 10 indicators)  × 35%   ← replaces manual technical scoring
Stochastic confirmation       × 10%
ARIMA directional forecast    × 15%   ← new
RSI normalized score          × 10%
FRED macro regime             × 20%
ADX trend bonus               × 10%
= TPS score | Edge = TPS - 50 (market implied) | Trade if edge ≥ 10%
```

---

## 8. Options Strategy Reference
**Trigger:** "options trade", "covered call", "put", "spread", "condor"

### Strategy Labels
| Label | Regime | Capital |
|-------|--------|---------|
| BUY CALL | Bull trend | Premium paid |
| BUY PUT | Bear trend | Premium paid |
| SELL CALL (covered) | Any — income | $0 |
| SELL PUT (cash-secured) | Bull — want to own lower | Strike × 100 |
| DEBIT SPREAD (call) | Choppy bull | Net debit |
| DEBIT SPREAD (put) | Choppy bear | Net debit |
| CREDIT SPREAD (call) | Bear/sideways | Margin |
| IRON CONDOR | Sideways / vol spike | Margin |

### Covered Call Checklist
```
✅ Own 100+ shares
✅ Strike at or above avg cost basis — NEVER sell below cost
✅ Strike OTM 5-10% above current price
✅ Expiry 2-6 weeks out (optimal theta decay)
✅ Premium ≥ 1% of stock price (= 12%+ annualized)
✅ NOT within 5 days of earnings (check Finnhub + Nasdaq calendar)
✅ IVP > 40% (elevated IV = richer premiums — VIX > 20 qualifies)
```

### Trade Block Format (required on every recommendation)
```
Strategy:          [exact label]
Entry:             $XX.XX
Target:            $XX.XX (+X%)
Stop:              Entry − (1.5 × ATR) = $XX.XX
Position size:     $X,XXX (X% of $117,125)
Buying power used: $XXX of $24,514
Risk on trade:     $XXX (X% of portfolio)
R/R ratio:         X.X : 1
Timeframe:         X days / weeks
Correlation check: [safe / warning]
Carry unwind:      [score X/21 — impact on trade]
Breakout filter:   [volume ≥1.5x + ADX>25 + close beyond level]
```

### Breakout Authenticity Filter
**Genuine breakout requires 2 of 3:**
1. Volume ≥ 1.5× 30-day average
2. ADX > 25 (real momentum, not noise)
3. Candle CLOSES beyond the level (not just a wick)

Only 1 of 3 = noise. Watch list only. Do not enter.

---

## 9. Free Replacement Data Sources
**No new APIs needed — all free**

```python
# Max Pain / GEX pin zone
# Search: "market chameleon {TICKER} max pain {expiry}"

# Short interest (critical for SOFI)
# Search: "iborrowdesk {TICKER}" — borrow rate + squeeze fuel
# FINRA short volume: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data

# Congress trades
# Search: "capitol trades {TICKER} this week"
# SEC Form 4: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4

# CBOE Put/Call ratio (market tide replacement)
# Search: "CBOE equity put call ratio today"
# <0.7 = bullish | >1.3 = bearish

# COT Report (futures positioning — free from CFTC)
# https://www.cftc.gov/dea/newcot/deahistfo.txt
# Commercials net short crude = crude likely to fall despite headlines

# Crypto whale alerts
# Search: "whale alert BTC today"
# Exchange inflows: search "glassnode BTC exchange inflows today"

# Seasonality
# Search: "equity clock {TICKER} seasonality"

# Earnings (3 free sources — use all 3 to cross-check)
nasdaq_url = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
av_earnings = av("EARNINGS_CALENDAR", horizon="3month")
finnhub_cal = finnhub("calendar/earnings", **{"from": start, "to": end})
```

---

## 10. EDGARTools — SEC Filings
**Trigger:** "SEC filing", "10-K", "10-Q", "insider trades", "13F", "financial statements"
**Requires:** `pip install edgartools`

```python
from edgar import set_identity, Company
set_identity("Your Name your@email.com")

co      = Company("NVDA")
tenk    = co.get_filings(form="10-K").latest()
tenq    = co.get_filings(form="10-Q").latest()
form4   = co.get_filings(form="4").latest()    # insider trades
f13     = co.get_filings(form="13F-HR").latest()  # institutional

obj          = tenk.obj()
income       = obj.financials.income_statement()
balance      = obj.financials.balance_sheet()
cashflow     = obj.financials.cashflow_statement()
financials_q = co.get_quarterly_financials()
```

**Buffett lens via EDGAR:**
```python
overview = av("OVERVIEW", symbol="NVDA")
peg  = float(overview.get("PEGRatio", 99))   # < 1.0 = undervalued ✅
pe   = float(overview.get("PERatio", 99))    # vs sector avg
eps_growth = overview.get("QuarterlyEarningsGrowthYOY") # > 10% = ✅
margin = float(overview.get("ProfitMargin", 0)) # expanding = ✅
```

---

## 11. Hedge Fund Monitor — OFR API
**Trigger:** "hedge fund data", "leverage ratio", "systemic risk", "repo volumes"
**Requires:** No key · Free · https://data.financialresearch.gov/hf/v1

```python
import requests, pandas as pd
BASE = "https://data.financialresearch.gov/hf/v1"

r = requests.get(f"{BASE}/series/timeseries", params={
    "mnemonic": "FPF-ALLQHF_LEVERAGERATIO_GAVWMEAN",
    "start_date": "2020-01-01"
}).json()
df = pd.DataFrame(r, columns=["date", "value"])

# Key: rising hedge fund leverage + VIX spike = forced unwind risk
# Pairs with carry unwind checklist — if both signal, defensive mode
```

---

## 12. US Fiscal Data — Treasury API
**Trigger:** "national debt", "treasury", "government spending"
**Requires:** No key · Free

```python
BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

debt = requests.get(f"{BASE}/v2/accounting/od/debt_to_penny",
    params={"sort": "-record_date", "page[size]": 1}).json()
print(f"Total debt: ${float(debt['data'][0]['tot_pub_debt_out_amt']):,.0f}")

rates = requests.get(f"{BASE}/v2/accounting/od/avg_interest_rates",
    params={"sort": "-record_date", "page[size]": 12}).json()
```

---

## Session Start Checklist

At the start of every trading session, fetch in this order:

```python
# 1. Macro snapshot (FRED)
macro = fetch_macro_snapshot()

# 2. Carry unwind score
carry = carry_unwind_score(macro)
if carry["score"] >= 8:
    print("⚠️ CARRY UNWIND ALERT — defensive mode")

# 3. Buffett Indicator
buffett = (macro["wilshire5000"] / macro["gdp"]) * 100
if buffett > 200:
    print("🔴 BUFFETT 200%+ — no new speculative longs")

# 4. USD/JPY check
usdjpy = av("FX_DAILY", from_symbol="USD", to_symbol="JPY")
# Check proximity to 160 — scenario A or B?

# 5. Intermarket flow: crude direction → DXY → regime

# 6. Regime classification

# 7. Per-ticker XGBoost + ARIMA scan

# 8. Covered call income opportunities

# 9. Risk check: correlation clusters + buying power remaining
```

---

## Disclaimer
> Research and signal generation only — not financial advice.
> Data from Polygon.io, Finnhub, Alpha Vantage, FRED.
> Verify all prices and premiums before executing.
> Options involve substantial risk.
> Past performance does not guarantee future results.
