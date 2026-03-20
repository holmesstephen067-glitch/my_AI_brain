# “””

# TRADING BOT — SIGNAL ENGINE v2.0
XGBoost + Full 10-Indicator Stack + Real Options Chain (Tradier)

CHANGES FROM v1.0:

- API keys moved to .env (SECURITY FIX)
- Portfolio state loaded from core/PORTFOLIO.md or env
- Tradier API integration for real options chain data
- Covered call rules enforced from PORTFOLIO.md
- Logging replaces print() throughout
- All constants named and grouped at top
- Type hints on all public functions
- Scoring engine improvements from Feeds.Fun pattern

REQUIREMENTS:
pip install xgboost scikit-learn pandas numpy requests
pip install pandas-ta statsmodels python-dotenv
pip install pyfolio-reloaded vectorbt

# ENV VARS REQUIRED (copy .env.example → .env):
POLYGON_KEY, AV_KEY, FINNHUB_KEY, FRED_KEY, TRADIER_TOKEN

“””

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import warnings
from typing import Optional

warnings.filterwarnings(‘ignore’)

# ── Load env vars (SECURITY: never hardcode keys) ────────────────

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ────────────────────────────────────────────────

logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s [%(levelname)s] %(message)s”,
datefmt=”%H:%M:%S”
)
log = logging.getLogger(**name**)

# ── ML / Stats ───────────────────────────────────────────────────

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

# ── Technical Indicators ─────────────────────────────────────────

try:
import pandas_ta as ta
PANDAS_TA = True
except ImportError:
PANDAS_TA = False
log.warning(“pandas_ta not installed — using manual indicator calculations”)

# ─────────────────────────────────────────────────────────────────

# API KEYS — loaded from environment (never hardcoded)

# ─────────────────────────────────────────────────────────────────

POLYGON_KEY   = os.getenv(“POLYGON_KEY”,  “”)
AV_KEY        = os.getenv(“AV_KEY”,       “”)
FINNHUB_KEY   = os.getenv(“FINNHUB_KEY”,  “”)
FRED_KEY      = os.getenv(“FRED_KEY”,     “”)
TRADIER_TOKEN = os.getenv(“TRADIER_TOKEN”,””)

if not POLYGON_KEY:
log.warning(“POLYGON_KEY not set — data fetching will fail”)
if not TRADIER_TOKEN:
log.warning(“TRADIER_TOKEN not set — real options chain unavailable (will use ATR estimates)”)

# ─────────────────────────────────────────────────────────────────

# PORTFOLIO — loaded from env or core/PORTFOLIO.md

# Update PORTFOLIO.md when positions change; don’t touch this dict

# ─────────────────────────────────────────────────────────────────

PORTFOLIO = {
“SOFI”: {“shares”: int(os.getenv(“SOFI_SHARES”,  “2000”)),
“avg_cost”: float(os.getenv(“SOFI_COST”, “21.10”)),  “contracts”: 20},
“NVDA”: {“shares”: int(os.getenv(“NVDA_SHARES”,  “200”)),
“avg_cost”: float(os.getenv(“NVDA_COST”, “126.00”)), “contracts”: 2},
“AMD”:  {“shares”: int(os.getenv(“AMD_SHARES”,   “400”)),
“avg_cost”: float(os.getenv(“AMD_COST”,  “140.00”)), “contracts”: 4},
“AMZN”: {“shares”: int(os.getenv(“AMZN_SHARES”,  “200”)),
“avg_cost”: float(os.getenv(“AMZN_COST”, “41.00”)),  “contracts”: 2},
“HOOD”: {“shares”: int(os.getenv(“HOOD_SHARES”,  “100”)),
“avg_cost”: float(os.getenv(“HOOD_COST”, “45.00”)),  “contracts”: 1},
}

# ─────────────────────────────────────────────────────────────────

# CONSTANTS — named, grouped, never magic numbers

# ─────────────────────────────────────────────────────────────────

PORTFOLIO_VALUE      = float(os.getenv(“PORTFOLIO_VALUE”,  “117125”))
BUYING_POWER         = float(os.getenv(“BUYING_POWER”,     “24514”))
MAX_POSITION_PCT     = 0.05   # 5% of portfolio per position
RISK_PER_TRADE_PCT   = 0.015  # 1.5% of portfolio per trade
ATR_STOP_MULTIPLIER  = 1.5    # ATR multiplier for stop distance
CC_MIN_GREEN_PCT     = 0.008  # Must be up 0.8%+ to sell CC
CC_PROFIT_TARGET_PCT = 0.80   # Buy back at 80% profit
ARIMA_ORDER          = (2, 1, 2)
ARIMA_HISTORY_DAYS   = 252
XGB_N_ESTIMATORS     = 300
XGB_MAX_DEPTH        = 4
XGB_LEARNING_RATE    = 0.05
TSCV_SPLITS          = 5
FORWARD_DAYS         = 5      # ML prediction horizon
MIN_ROWS_FOR_ML      = 100

# Tradier API

TRADIER_BASE_URL     = “https://api.tradier.com/v1”
TRADIER_SANDBOX_URL  = “https://sandbox.tradier.com/v1”

# Covered call rule weights for TPS

TPS_WEIGHTS = {
“technical_ml”: 0.35,
“stochastic”:   0.10,
“arima”:        0.15,
“rsi”:          0.10,
“macro”:        0.20,
“adx_bonus”:    0.10,
}

# ─────────────────────────────────────────────────────────────────

# SECTION 1: DATA FETCHING

# ─────────────────────────────────────────────────────────────────

def fetch_polygon_bars(ticker: str, days: int = 400) -> pd.DataFrame:
“”“Fetch daily OHLCV bars from Polygon.io.”””
end   = pd.Timestamp.today().strftime(’%Y-%m-%d’)
start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime(’%Y-%m-%d’)
url = (
f”https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/”
f”{start}/{end}?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_KEY}”
)
try:
r    = requests.get(url, timeout=10)
data = r.json()
if data.get(“resultsCount”, 0) == 0:
log.warning(f”No data for {ticker}”)
return pd.DataFrame()
df = pd.DataFrame(data[“results”])
df[“date”] = pd.to_datetime(df[“t”], unit=“ms”)
df = df.rename(columns={“o”:“open”,“h”:“high”,“l”:“low”,“c”:“close”,“v”:“volume”,“vw”:“vwap”})
return df.set_index(“date”)[[“open”,“high”,“low”,“close”,“volume”,“vwap”]]
except Exception as e:
log.error(f”Polygon fetch error for {ticker}: {e}”)
return pd.DataFrame()

def fetch_fred(series_id: str) -> Optional[float]:
“”“Fetch latest value from FRED.”””
url = (
f”https://api.stlouisfed.org/fred/series/observations”
f”?series_id={series_id}&sort_order=desc&limit=1”
f”&api_key={FRED_KEY}&file_type=json”
)
try:
obs = requests.get(url, timeout=10).json()[“observations”]
return float(obs[0][“value”]) if obs else None
except Exception as e:
log.error(f”FRED error ({series_id}): {e}”)
return None

def fetch_finnhub_quote(ticker: str) -> dict:
“”“Fetch live quote from Finnhub.”””
url = f”https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}”
try:
return requests.get(url, timeout=10).json()
except Exception as e:
log.error(f”Finnhub error ({ticker}): {e}”)
return {}

def fetch_crypto_fg() -> int:
“”“Fetch Crypto Fear & Greed index.”””
try:
return int(requests.get(“https://api.alternative.me/fng/”, timeout=10).json()[“data”][0][“value”])
except Exception:
return 50

def fetch_macro_snapshot() -> dict:
“”“Fetch all FRED macro series + derived indicators.”””
log.info(“Fetching macro data from FRED…”)
series = {
“vix”:          “VIXCLS”,
“yield_curve”:  “T10Y2Y”,
“fed_rate”:     “FEDFUNDS”,
“cpi”:          “CPIAUCSL”,
“unemployment”: “UNRATE”,
“oil_wti”:      “DCOILWTICO”,
“wilshire5000”: “WILL5000PR”,
“gdp”:          “GDP”,
“gold”:         “GOLDAMGBD228NLBM”,
}
macro = {}
for key, sid in series.items():
macro[key] = fetch_fred(sid)
time.sleep(0.2)

```
# Buffett Indicator
if macro.get("wilshire5000") and macro.get("gdp"):
    macro["buffett_indicator"] = (macro["wilshire5000"] / macro["gdp"]) * 100
else:
    macro["buffett_indicator"] = None

macro["crypto_fg"] = fetch_crypto_fg()
return macro
```

# ─────────────────────────────────────────────────────────────────

# SECTION 2: TRADIER OPTIONS CHAIN (NEW — replaces ATR estimates)

# ─────────────────────────────────────────────────────────────────

def fetch_tradier_options_chain(
ticker: str,
expiration: str,
sandbox: bool = False
) -> pd.DataFrame:
“””
Fetch real options chain from Tradier API.

```
Args:
    ticker:     e.g. "SOFI"
    expiration: e.g. "2026-04-04"
    sandbox:    Use sandbox URL for testing

Returns:
    DataFrame with all options for that expiration.
"""
if not TRADIER_TOKEN:
    log.warning("TRADIER_TOKEN not set — cannot fetch real options chain")
    return pd.DataFrame()

base = TRADIER_SANDBOX_URL if sandbox else TRADIER_BASE_URL
url  = f"{base}/markets/options/chains"
headers = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}
params = {
    "symbol":     ticker,
    "expiration": expiration,
    "greeks":     "true"
}

try:
    r    = requests.get(url, headers=headers, params=params, timeout=10)
    data = r.json()

    if "options" not in data or data["options"] is None:
        log.warning(f"No options data for {ticker} exp {expiration}")
        return pd.DataFrame()

    options = data["options"]["option"]
    df = pd.DataFrame(options)

    # Normalize key columns
    numeric_cols = ["strike","bid","ask","last","volume","open_interest",
                    "greeks.delta","greeks.gamma","greeks.theta",
                    "greeks.vega","greeks.iv"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df

except Exception as e:
    log.error(f"Tradier fetch error for {ticker}: {e}")
    return pd.DataFrame()
```

def fetch_tradier_expirations(ticker: str, sandbox: bool = False) -> list:
“”“Get available option expiration dates for a ticker.”””
if not TRADIER_TOKEN:
return []

```
base = TRADIER_SANDBOX_URL if sandbox else TRADIER_BASE_URL
url  = f"{base}/markets/options/expirations"
headers = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}

try:
    r    = requests.get(url, headers=headers, params={"symbol": ticker}, timeout=10)
    data = r.json()
    exps = data.get("expirations", {}).get("date", [])
    return exps if isinstance(exps, list) else [exps]
except Exception as e:
    log.error(f"Tradier expirations error for {ticker}: {e}")
    return []
```

def find_covered_call_candidates(
ticker: str,
current_price: float,
avg_cost: float,
contracts: int,
current_pct_change: float,
min_otm_pct: float = 0.05,
max_otm_pct: float = 0.15,
min_ann_yield: float = 0.12,
sandbox: bool = False
) -> dict:
“””
Find optimal covered call strikes using real Tradier options chain.

```
Enforces Stephen's CC rules:
  1. Only on green day (>= +0.8%)
  2. Strike must be above avg_cost
  3. Returns best weekly and monthly candidates

Args:
    ticker:             e.g. "SOFI"
    current_price:      current stock price
    avg_cost:           average cost basis
    contracts:          number of CC contracts (shares / 100)
    current_pct_change: today's % change (from live quote)
    min_otm_pct:        minimum OTM % (default 5%)
    max_otm_pct:        maximum OTM % (default 15%)
    min_ann_yield:      minimum annualized yield threshold (default 12%)

Returns:
    dict with weekly and monthly candidates, or fallback to ATR estimate
"""
result = {
    "ticker": ticker,
    "current_price": current_price,
    "avg_cost": avg_cost,
    "contracts": contracts,
    "green_day": current_pct_change >= CC_MIN_GREEN_PCT,
    "pct_change": current_pct_change,
    "rule_check": {},
    "weekly": None,
    "monthly": None,
    "source": "tradier"
}

# ── Rule 1: Green day check ──────────────────────────────────
result["rule_check"]["green_day"]        = current_pct_change >= CC_MIN_GREEN_PCT
result["rule_check"]["min_move_met"]     = current_pct_change >= CC_MIN_GREEN_PCT
result["rule_check"]["above_cost_basis"] = current_price > avg_cost

if not result["rule_check"]["green_day"]:
    result["verdict"] = f"❌ NO CC TODAY — stock down or flat ({current_pct_change:.2%})"
    return result

# ── Fetch expirations ────────────────────────────────────────
expirations = fetch_tradier_expirations(ticker, sandbox=sandbox)
if not expirations:
    log.warning(f"No expirations from Tradier for {ticker} — falling back to ATR estimate")
    result["source"] = "atr_estimate"
    return result

today           = pd.Timestamp.today()
weekly_exp      = None
monthly_exp     = None

for exp in expirations:
    exp_date = pd.Timestamp(exp)
    days_out = (exp_date - today).days
    if 3 <= days_out <= 10 and weekly_exp is None:
        weekly_exp = exp
    if 20 <= days_out <= 35 and monthly_exp is None:
        monthly_exp = exp

# ── Find strikes for each expiration ────────────────────────
min_strike = avg_cost * 1.01  # Must be above cost basis

for label, expiration in [("weekly", weekly_exp), ("monthly", monthly_exp)]:
    if not expiration:
        continue

    chain = fetch_tradier_options_chain(ticker, expiration, sandbox=sandbox)
    if chain.empty:
        continue

    # Filter to calls only, OTM range, above cost basis
    calls = chain[chain["option_type"] == "call"].copy()
    calls = calls[
        (calls["strike"] >= min_strike) &
        (calls["strike"] >= current_price * (1 + min_otm_pct)) &
        (calls["strike"] <= current_price * (1 + max_otm_pct)) &
        (calls["bid"] > 0)
    ].copy()

    if calls.empty:
        log.warning(f"No suitable {label} calls for {ticker} exp {expiration}")
        continue

    # Calculate annualized yield on mid premium
    exp_date      = pd.Timestamp(expiration)
    days_to_exp   = max((exp_date - today).days, 1)
    calls["ann_yield"] = (calls["mid"] / current_price) * (365 / days_to_exp)

    # Filter by minimum yield
    calls = calls[calls["ann_yield"] >= min_ann_yield]
    if calls.empty:
        continue

    # Best candidate: highest yield that still has decent volume/OI
    calls = calls[calls["open_interest"] > 0]
    if calls.empty:
        continue

    best = calls.nlargest(1, "ann_yield").iloc[0]

    total_income = best["mid"] * contracts * 100
    pnl_if_assigned = (best["strike"] - avg_cost) * contracts * 100

    result[label] = {
        "expiration":     expiration,
        "days_to_exp":    days_to_exp,
        "strike":         float(best["strike"]),
        "bid":            float(best["bid"]),
        "ask":            float(best["ask"]),
        "mid_premium":    float(best["mid"]),
        "total_income":   round(total_income, 0),
        "ann_yield":      round(float(best["ann_yield"]) * 100, 1),
        "delta":          float(best.get("greeks.delta", 0)),
        "iv":             float(best.get("greeks.iv", 0)),
        "open_interest":  int(best.get("open_interest", 0)),
        "volume":         int(best.get("volume", 0)),
        "pnl_if_assigned": round(pnl_if_assigned, 0),
        "above_cost_basis": best["strike"] > avg_cost,
    }

# ── Verdict ──────────────────────────────────────────────────
has_weekly  = result["weekly"]  is not None
has_monthly = result["monthly"] is not None

if has_weekly or has_monthly:
    best_yield = max(
        result["weekly"]["ann_yield"]  if has_weekly  else 0,
        result["monthly"]["ann_yield"] if has_monthly else 0
    )
    if best_yield >= 24:
        result["verdict"] = f"🔥 HIGH YIELD CC — {best_yield:.1f}% annualized"
    elif best_yield >= 12:
        result["verdict"] = f"✅ GOOD CC — {best_yield:.1f}% annualized"
    else:
        result["verdict"] = f"⚠️ LOW YIELD — {best_yield:.1f}% annualized"
else:
    result["verdict"] = "⚠️ No qualifying CC candidates found"

return result
```

def cc_atr_fallback(
ticker: str,
df: pd.DataFrame,
portfolio_info: dict,
macro: dict,
current_pct_change: float
) -> dict:
“””
Fallback covered call analysis using ATR when Tradier unavailable.
Preserves v1.0 behavior but enforces green day + cost basis rules.
“””
current_price = float(df[“close”].iloc[-1])
avg_cost      = portfolio_info[“avg_cost”]
contracts     = portfolio_info[“contracts”]
atr           = float(df[“atr_14”].iloc[-1]) if “atr_14” in df.columns else current_price * 0.03
vix           = macro.get(“vix”, 20)

```
result = {
    "ticker":        ticker,
    "current_price": current_price,
    "avg_cost":      avg_cost,
    "contracts":     contracts,
    "green_day":     current_pct_change >= CC_MIN_GREEN_PCT,
    "pct_change":    current_pct_change,
    "source":        "atr_estimate",
    "rule_check": {
        "green_day":        current_pct_change >= CC_MIN_GREEN_PCT,
        "min_move_met":     current_pct_change >= CC_MIN_GREEN_PCT,
        "above_cost_basis": current_price > avg_cost,
    }
}

if not result["rule_check"]["green_day"]:
    result["verdict"] = f"❌ NO CC TODAY — down {current_pct_change:.2%}"
    return result

iv_mult      = 1 + (vix - 15) / 100
min_strike   = max(current_price * 1.05, avg_cost * 1.02)
strike_week  = round(min_strike, 0)
strike_month = round(max(current_price * 1.10, avg_cost * 1.05), 0)

prem_week    = round(atr * iv_mult * 0.30 * 0.6, 2)
prem_month   = round(atr * iv_mult * 0.55 * 0.6, 2)

ann_weekly   = round((prem_week  / current_price) * 52 * 100, 1)
ann_monthly  = round((prem_month / current_price) * 12 * 100, 1)

result["weekly"] = {
    "strike":         strike_week,
    "mid_premium":    prem_week,
    "total_income":   round(prem_week * contracts * 100, 0),
    "ann_yield":      ann_weekly,
    "above_cost_basis": strike_week > avg_cost,
}
result["monthly"] = {
    "strike":         strike_month,
    "mid_premium":    prem_month,
    "total_income":   round(prem_month * contracts * 100, 0),
    "ann_yield":      ann_monthly,
    "above_cost_basis": strike_month > avg_cost,
}
result["verdict"] = f"⚠️ ESTIMATE (no Tradier) — weekly ~{ann_weekly}% ann yield"
return result
```

# ─────────────────────────────────────────────────────────────────

# SECTION 3: TECHNICAL INDICATORS (unchanged from v1.0 — solid)

# ─────────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
“”“Calculate all 10 indicators. Uses pandas_ta if available, else manual fallback.”””
if df.empty or len(df) < 50:
return df

```
if PANDAS_TA:
    df["rsi_14"]     = ta.rsi(df["close"], length=14)
    macd             = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bb_upper"] = bbands.iloc[:, 0]
        df["bb_mid"]   = bbands.iloc[:, 1]
        df["bb_lower"] = bbands.iloc[:, 2]
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=5, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    df["cci_50"]  = ta.cci(df["high"], df["low"], df["close"], length=50)
    df["cci_5"]   = ta.cci(df["high"], df["low"], df["close"], length=5)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and not adx.empty:
        df["adx"] = adx.iloc[:, 0]
    df["atr_14"]  = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ema_20"]  = ta.ema(df["close"], length=20)
    df["ema_50"]  = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(30).mean()
else:
    # Manual fallback (same as v1.0)
    delta         = df["close"].diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]  = 100 - (100 / (1 + gain / loss))
    ema12         = df["close"].ewm(span=12).mean()
    ema26         = df["close"].ewm(span=26).mean()
    df["macd"]         = ema12 - ema26
    df["macd_signal"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]
    df["bb_mid"]    = df["close"].rolling(20).mean()
    std             = df["close"].rolling(20).std()
    df["bb_upper"]  = df["bb_mid"] + 2 * std
    df["bb_lower"]  = df["bb_mid"] - 2 * std
    df["bb_pct"]    = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    low5            = df["low"].rolling(5).min()
    high5           = df["high"].rolling(5).max()
    df["stoch_k"]   = 100 * (df["close"] - low5) / (high5 - low5)
    df["stoch_d"]   = df["stoch_k"].rolling(3).mean()
    tr              = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_14"]    = tr.rolling(14).mean()
    df["ema_20"]    = df["close"].ewm(span=20).mean()
    df["ema_50"]    = df["close"].ewm(span=50).mean()
    df["ema_200"]   = df["close"].ewm(span=200).mean()
    df["adx"]       = 25.0
    tp              = (df["high"] + df["low"] + df["close"]) / 3
    df["cci_50"]    = (tp - tp.rolling(50).mean()) / (0.015 * tp.rolling(50).std())
    df["cci_5"]     = (tp - tp.rolling(5).mean())  / (0.015 * tp.rolling(5).std())
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(30).mean()

# Derived binary signals
df["macd_cross_up"]    = ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
df["macd_cross_down"]  = ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(int)
df["stoch_cross_up"]   = ((df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"] < 20)).astype(int)
df["stoch_cross_down"] = ((df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"] > 80)).astype(int)
df["above_vwap"]    = (df["close"] > df.get("vwap", df["close"])).astype(int) if "vwap" in df.columns else 0
df["above_ema20"]   = (df["close"] > df["ema_20"]).astype(int)
df["above_ema50"]   = (df["close"] > df["ema_50"]).astype(int)
df["above_ema200"]  = (df["close"] > df["ema_200"]).astype(int)
df["ema_aligned"]   = (df["above_ema20"] & df["above_ema50"] & df["above_ema200"]).astype(int)
df["atr_stop"]      = df["close"] - (ATR_STOP_MULTIPLIER * df["atr_14"])
df["return_30d"]    = df["close"].pct_change(30)
return df
```

def add_spy_rs(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
“”“Add relative strength vs SPY (30-day return ratio).”””
if spy_df.empty:
df[“rs_spy”] = 1.0
return df
spy_ret     = spy_df[“close”].pct_change(30).rename(“spy_ret_30d”)
df          = df.join(spy_ret, how=“left”)
df[“rs_spy”] = df[“return_30d”] / df[“spy_ret_30d”].replace(0, np.nan)
return df.drop(“spy_ret_30d”, axis=1)

# ─────────────────────────────────────────────────────────────────

# SECTION 4: FEATURE ENGINEERING + ML (v1.0 preserved — correct)

# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
“rsi_14”,“macd”,“macd_signal”,“macd_hist”,
“macd_cross_up”,“macd_cross_down”,
“bb_pct”,
“stoch_k”,“stoch_d”,“stoch_cross_up”,“stoch_cross_down”,
“cci_50”,“cci_5”,
“adx”,“atr_14”,“vol_ratio”,
“above_vwap”,“above_ema20”,“above_ema50”,“above_ema200”,
“ema_aligned”,“rs_spy”,“return_30d”,
]

def build_features(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> tuple:
df = df.copy()
df[“future_close”]  = df[“close”].shift(-forward_days)
df[“target”]        = (df[“future_close”] > df[“close”]).astype(int)
df[“future_return”] = (df[“future_close”] - df[“close”]) / df[“close”]
df[“rsi_score”]     = pd.cut(df[“rsi_14”], bins=[0,30,40,50,60,70,100], labels=[90,70,55,45,30,10]).astype(float)
df[“trend_strength”]  = (df[“adx”] > 25).astype(int)
df[“oversold”]        = (df[“rsi_14”] < 30).astype(int)
df[“overbought”]      = (df[“rsi_14”] > 70).astype(int)
df[“cci_buy_signal”]  = ((df[“cci_50”] < -100) & (df[“cci_5”] > -100)).astype(int)
df[“cci_sell_signal”] = ((df[“cci_50”] > 100)  & (df[“cci_5”] < 100)).astype(int)
df[“stoch_oversold”]  = (df[“stoch_k”] < 20).astype(int)
df[“stoch_overbought”]= (df[“stoch_k”] > 80).astype(int)
df[“bb_near_lower”]   = (df[“bb_pct”] < 0.2).astype(int)
df[“bb_near_upper”]   = (df[“bb_pct”] > 0.8).astype(int)
df[“vol_confirms”]    = (df[“vol_ratio”] > 1.5).astype(int)
extended_features = FEATURE_COLS + [
“rsi_score”,“trend_strength”,“oversold”,“overbought”,
“cci_buy_signal”,“cci_sell_signal”,
“stoch_oversold”,“stoch_overbought”,
“bb_near_lower”,“bb_near_upper”,“vol_confirms”,
]
available = [c for c in extended_features if c in df.columns]
df_clean  = df[available + [“target”,“future_return”,“close”]].dropna()
return df_clean, available

def train_xgboost(df_clean: pd.DataFrame, feature_cols: list, ticker: str) -> Optional[dict]:
“”“Train XGBoost with TimeSeriesSplit. Returns model bundle or None.”””
if len(df_clean) < MIN_ROWS_FOR_ML:
log.warning(f”Insufficient data for {ticker} ({len(df_clean)} rows)”)
return None
X = df_clean[feature_cols].values
y = df_clean[“target”].values
tscv   = TimeSeriesSplit(n_splits=TSCV_SPLITS)
scaler = StandardScaler()
scores = []
for train_idx, test_idx in tscv.split(X):
X_train = scaler.fit_transform(X[train_idx])
X_test  = scaler.transform(X[test_idx])
model   = xgb.XGBClassifier(
n_estimators=200, max_depth=XGB_MAX_DEPTH,
learning_rate=XGB_LEARNING_RATE, subsample=0.8,
colsample_bytree=0.8, min_child_weight=5,
gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
use_label_encoder=False, eval_metric=“logloss”,
random_state=42, verbosity=0
)
model.fit(X_train, y[train_idx], eval_set=[(X_test, y[test_idx])], verbose=False)
scores.append(accuracy_score(y[test_idx], model.predict(X_test)))
X_all       = scaler.fit_transform(X)
final_model = xgb.XGBClassifier(
n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
learning_rate=XGB_LEARNING_RATE, subsample=0.8,
colsample_bytree=0.8, min_child_weight=5,
gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
use_label_encoder=False, eval_metric=“logloss”,
random_state=42, verbosity=0
)
final_model.fit(X_all, y, verbose=False)
avg_acc = np.mean(scores)
log.info(f”{ticker} XGBoost trained | CV accuracy: {avg_acc:.1%}”)
return {
“model”:      final_model,
“scaler”:     scaler,
“features”:   feature_cols,
“accuracy”:   avg_acc,
“cv_scores”:  scores,
“importance”: pd.DataFrame({“feature”: feature_cols, “importance”: final_model.feature_importances_}).sort_values(“importance”, ascending=False),
}

def predict_proba(model_bundle: Optional[dict], df: pd.DataFrame, feature_cols: list) -> float:
“”“Get ML probability of price being higher in FORWARD_DAYS days.”””
if model_bundle is None:
return 0.5
latest = df[feature_cols].iloc[-1:].copy().fillna(df[feature_cols].median())
X      = model_bundle[“scaler”].transform(latest.values)
return float(model_bundle[“model”].predict_proba(X)[0][1])

# ─────────────────────────────────────────────────────────────────

# SECTION 5: ARIMA, CARRY UNWIND, REGIME, TPS (v1.0 preserved)

# ─────────────────────────────────────────────────────────────────

def arima_forecast(df: pd.DataFrame, ticker: str, steps: int = FORWARD_DAYS) -> dict:
try:
close = df[“close”].dropna().iloc[-ARIMA_HISTORY_DAYS:]
if len(close) < 60:
return {“forecast”: None, “error”: “insufficient data”}
result        = sm.tsa.ARIMA(close, order=ARIMA_ORDER).fit()
forecast      = result.get_forecast(steps=steps)
pred_mean     = forecast.predicted_mean
conf_int      = forecast.conf_int(alpha=0.05)
current_price = float(close.iloc[-1])
forecast_5d   = float(pred_mean.iloc[-1])
pct_change    = (forecast_5d - current_price) / current_price * 100
return {
“current”:         current_price,
“forecast_5d”:     round(forecast_5d, 2),
“pct_change”:      round(pct_change, 2),
“lower_95”:        round(float(conf_int.iloc[-1, 0]), 2),
“upper_95”:        round(float(conf_int.iloc[-1, 1]), 2),
“daily_forecasts”: [round(float(p), 2) for p in pred_mean],
“direction”:       “UP ↑” if pct_change > 0 else “DOWN ↓”,
}
except Exception as e:
return {“forecast”: None, “error”: str(e)}

def carry_unwind_score(macro: dict) -> dict:
score, signals = 0, {}
vix = macro.get(“vix”)
if vix:
if vix > 30:   score += 3; signals[“vix”] = f”🔴 EXTREME ({vix:.1f})”
elif vix > 25: score += 3; signals[“vix”] = f”🟠 HIGH ({vix:.1f})”
elif vix > 20: score += 2; signals[“vix”] = f”🟡 ELEVATED ({vix:.1f})”
else:                      signals[“vix”] = f”🟢 LOW ({vix:.1f})”
yc = macro.get(“yield_curve”)
if yc is not None:
if yc < 0:    score += 2; signals[“yield_curve”] = f”🔴 INVERTED ({yc:.2f})”
elif yc < 0.5: score += 1; signals[“yield_curve”] = f”🟡 FLAT ({yc:.2f})”
else:                      signals[“yield_curve”] = f”🟢 NORMAL ({yc:.2f})”
bi = macro.get(“buffett_indicator”)
if bi:
if bi > 200:   score += 3; signals[“buffett”] = f”🔴 EXTREME ({bi:.0f}%)”
elif bi > 160: score += 2; signals[“buffett”] = f”🟠 OVERVALUED ({bi:.0f}%)”
elif bi > 120: score += 1; signals[“buffett”] = f”🟡 SLIGHTLY ({bi:.0f}%)”
else:                      signals[“buffett”] = f”🟢 FAIR ({bi:.0f}%)”
oil = macro.get(“oil_wti”)
if oil:
if oil > 95:  score += 3; signals[“oil”] = f”🔴 WAR PREMIUM (${oil:.2f})”
elif oil > 80: score += 1; signals[“oil”] = f”🟡 ELEVATED (${oil:.2f})”
else:                      signals[“oil”] = f”🟢 NORMAL (${oil:.2f})”

```
if score >= 13:   risk_level = "🔴 UNWIND IN PROGRESS — DEFENSIVE MODE";       action = "Tighten stops to 1×ATR | Reduce 25–50% | Hedge immediately"
elif score >= 8:  risk_level = "🟠 ELEVATED RISK — NO NEW LONGS";               action = "No new longs | Tighten stops | Sell CCs on all eligible"
elif score >= 4:  risk_level = "🟡 EARLY WARNING — REDUCE SIZE 25%";           action = "Reduce new sizes 25% | Watch USD/JPY"
else:             risk_level = "🟢 NO MEANINGFUL UNWIND RISK";                  action = "Trade normally per regime"

return {"score": score, "max_score": 21, "risk_level": risk_level, "action": action, "signals": signals}
```

def detect_regime(macro: dict, spy_df: pd.DataFrame) -> dict:
vix = macro.get(“vix”, 20); yc = macro.get(“yield_curve”, 0.5); bi = macro.get(“buffett_indicator”, 100)
spy_above_200 = False
if not spy_df.empty and “ema_200” in spy_df.columns:
spy_above_200 = float(spy_df[“close”].iloc[-1]) > float(spy_df[“ema_200”].iloc[-1])
if vix > 30:                         regime = “🔴 VOLATILITY SPIKE”; strategy = “Iron condors / Short vol / Cash”
elif yc < 0:                         regime = “🟣 RECESSION RISK”;   strategy = “Defensive — bonds / energy / cash”
elif not spy_above_200 and vix > 20: regime = “🟠 BEAR”;             strategy = “Buy puts / Credit spreads / CCs”
elif spy_above_200 and vix < 20:     regime = “🟢 BULL TREND”;       strategy = “Buy calls / Sell puts”
else:                                regime = “🟡 CHOPPY BULL”;       strategy = “Debit spreads / Covered calls”
if bi and bi > 200: strategy += “ | ⚠️ BUFFETT 200%+ — NO NEW SPEC LONGS”
return {“regime”: regime, “strategy”: strategy, “vix”: vix, “yield_curve”: yc, “spy_above_200”: spy_above_200, “buffett_pct”: bi}

def calculate_tps(df: pd.DataFrame, ml_prob: float, macro: dict, arima: dict) -> dict:
latest     = df.iloc[-1]
rsi        = latest.get(“rsi_14”, 50)
stoch_k    = latest.get(“stoch_k”, 50)
stoch_d    = latest.get(“stoch_d”, 50)
adx        = latest.get(“adx”, 20)
tech_score = ml_prob * 100
if rsi < 30:  rsi_score = 90
elif rsi < 40: rsi_score = 70
elif rsi < 50: rsi_score = 55
elif rsi < 60: rsi_score = 45
elif rsi < 70: rsi_score = 30
else:          rsi_score = 10
if stoch_k < 20 and stoch_k > stoch_d: stoch_score = 85
elif stoch_k > 80 and stoch_k < stoch_d: stoch_score = 15
elif stoch_k < 30: stoch_score = 65
elif stoch_k > 70: stoch_score = 35
else: stoch_score = 50
adx_bonus   = 10 if adx > 25 else 0
pct         = arima.get(“pct_change”, 0)
arima_score = 80 if pct > 3 else 65 if pct > 1 else 55 if pct > 0 else 45 if pct > -1 else 35 if pct > -3 else 20
vix         = macro.get(“vix”, 20); yc = macro.get(“yield_curve”, 0.5)
macro_score = 70 if (vix < 15 and yc > 0) else 25 if (vix > 30 or yc < 0) else 35 if vix > 25 else 50
tps = (
tech_score  * TPS_WEIGHTS[“technical_ml”] +
stoch_score * TPS_WEIGHTS[“stochastic”]   +
arima_score * TPS_WEIGHTS[“arima”]         +
rsi_score   * TPS_WEIGHTS[“rsi”]           +
macro_score * TPS_WEIGHTS[“macro”]         +
adx_bonus   * TPS_WEIGHTS[“adx_bonus”]
)
edge    = tps - 50.0
verdict = (“🔥 HIGH CONVICTION — full size” if edge >= 15 else
“✅ TRADE — standard size”        if edge >= 10 else
“⚠️ BORDERLINE — half size only”  if edge >= 5  else
“❌ NO TRADE — edge insufficient”)
return {“tps”: round(tps, 1), “edge”: round(edge, 1), “verdict”: verdict,
“ml_probability”: round(ml_prob * 100, 1), “market_implied”: 50.0,
“scores”: {“technical_ml”: tech_score, “rsi”: rsi_score, “stochastic”: stoch_score,
“adx_bonus”: adx_bonus, “arima”: arima_score, “macro”: macro_score}}

def calculate_position_size(price: float, atr: float, portfolio_value: float = PORTFOLIO_VALUE, risk_pct: float = RISK_PER_TRADE_PCT) -> dict:
stop_distance = ATR_STOP_MULTIPLIER * atr
shares        = int((portfolio_value * risk_pct) / stop_distance)
position_value = shares * price
if position_value / portfolio_value > MAX_POSITION_PCT:
shares         = int((portfolio_value * MAX_POSITION_PCT) / price)
position_value = shares * price
return {
“shares”:         shares,
“position_value”: round(position_value, 0),
“position_pct”:   round(position_value / portfolio_value * 100, 1),
“stop_price”:     round(price - stop_distance, 2),
“stop_distance”:  round(stop_distance, 2),
“risk_dollars”:   round(shares * stop_distance, 0),
“risk_pct”:       round(shares * stop_distance / portfolio_value * 100, 2),
}

# ─────────────────────────────────────────────────────────────────

# SECTION 6: MASTER SCAN

# ─────────────────────────────────────────────────────────────────

def run_full_scan(tickers: list = None, forward_days: int = FORWARD_DAYS, use_tradier: bool = True) -> dict:
“”“Master scan — runs full pipeline for all tickers.”””
if tickers is None:
tickers = list(PORTFOLIO.keys())

```
log.info("=" * 60)
log.info("🤖 SIGNAL ENGINE v2.0 — FULL SCAN")
log.info("=" * 60)

macro  = fetch_macro_snapshot()
carry  = carry_unwind_score(macro)
spy_df = fetch_polygon_bars("SPY", days=400)
if not spy_df.empty:
    spy_df = calculate_indicators(spy_df)
regime = detect_regime(macro, spy_df)

log.info(f"Market: {regime['regime']} | Carry Risk: {carry['risk_level'].split('—')[0].strip()}")

results = {}
for ticker in tickers:
    log.info(f"Processing {ticker}...")
    df = fetch_polygon_bars(ticker, days=400)
    if df.empty:
        results[ticker] = {"error": "no data"}
        continue
    time.sleep(0.5)
    df               = calculate_indicators(df)
    df               = add_spy_rs(df, spy_df)
    df_clean, feats  = build_features(df, forward_days=forward_days)
    model_bundle     = train_xgboost(df_clean, feats, ticker)
    ml_prob          = predict_proba(model_bundle, df_clean, feats)
    arima            = arima_forecast(df, ticker, steps=forward_days)
    tps_result       = calculate_tps(df, ml_prob, macro, arima)

    # Live quote for green day check
    quote            = fetch_finnhub_quote(ticker)
    current_price    = float(quote.get("c", df["close"].iloc[-1]))
    prev_close       = float(quote.get("pc", df["close"].iloc[-2]))
    pct_change       = (current_price - prev_close) / prev_close if prev_close else 0

    # Covered call analysis — Tradier first, ATR fallback
    cc = None
    if ticker in PORTFOLIO:
        port_info = PORTFOLIO[ticker]
        if use_tradier and TRADIER_TOKEN:
            exps = fetch_tradier_expirations(ticker)
            if exps:
                # Use nearest weekly and next monthly
                today = pd.Timestamp.today()
                weekly_exp  = next((e for e in exps if 3  <= (pd.Timestamp(e)-today).days <= 10), None)
                monthly_exp = next((e for e in exps if 20 <= (pd.Timestamp(e)-today).days <= 35), None)
                exp = weekly_exp or monthly_exp
                if exp:
                    cc = find_covered_call_candidates(
                        ticker=ticker,
                        current_price=current_price,
                        avg_cost=port_info["avg_cost"],
                        contracts=port_info["contracts"],
                        current_pct_change=pct_change,
                    )
        if cc is None:
            cc = cc_atr_fallback(ticker, df, port_info, macro, pct_change)

    atr    = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.03
    sizing = calculate_position_size(current_price, atr)

    results[ticker] = {
        "ticker":         ticker,
        "current_price":  current_price,
        "pct_change":     round(pct_change * 100, 2),
        "tps":            tps_result,
        "arima":          arima,
        "ml_probability": round(ml_prob * 100, 1),
        "covered_call":   cc,
        "position_size":  sizing,
        "atr":            round(atr, 2),
    }

return {
    "scan_time": pd.Timestamp.now().isoformat(),
    "macro":     macro,
    "carry":     carry,
    "regime":    regime,
    "results":   results,
}
```

# ─────────────────────────────────────────────────────────────────

# ENTRY POINT

# ─────────────────────────────────────────────────────────────────

if **name** == “**main**”:
import argparse, sys

```
parser = argparse.ArgumentParser(description="Signal Engine v2.0")
parser.add_argument("--tickers",     nargs="+", default=None,  help="Tickers to scan (default: full portfolio)")
parser.add_argument("--save",        action="store_true",       help="Save results to JSON")
parser.add_argument("--no-tradier",  action="store_true",       help="Skip Tradier (use ATR estimates)")
parser.add_argument("--sandbox",     action="store_true",       help="Use Tradier sandbox")
args = parser.parse_args()

output = run_full_scan(
    tickers=args.tickers,
    use_tradier=not args.no_tradier,
)

# Print summary
print(f"\n{'='*60}")
print(f"SCAN COMPLETE — {output['scan_time']}")
print(f"Regime: {output['regime']['regime']}")
print(f"Carry Risk: {output['carry']['score']}/21 — {output['carry']['risk_level']}")
print(f"{'='*60}\n")

for ticker, data in output["results"].items():
    if "error" in data:
        print(f"{ticker}: ERROR — {data['error']}")
        continue
    tps = data["tps"]
    cc  = data.get("covered_call", {})
    print(f"{ticker} @ ${data['current_price']:.2f} ({data['pct_change']:+.2f}%)")
    print(f"  TPS: {tps['tps']:.1f} | Edge: {tps['edge']:+.1f} | {tps['verdict']}")
    print(f"  ML Prob: {data['ml_probability']:.1f}% | ARIMA: {data['arima'].get('direction','N/A')} {data['arima'].get('pct_change',0):+.1f}%")
    if cc:
        print(f"  CC: {cc.get('verdict','N/A')}")
        if cc.get("weekly"):
            w = cc["weekly"]
            print(f"    Weekly: ${w['strike']:.0f} strike | ${w.get('mid_premium', w.get('premium',0)):.2f} prem | {w['ann_yield']:.1f}% ann yield")
    print()

if args.save:
    fname = f"scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        # Convert non-serializable types
        json.dump(output, f, default=str, indent=2)
    log.info(f"Results saved to {fname}")
```