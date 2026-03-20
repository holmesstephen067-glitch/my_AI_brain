"""
=============================================================
TRADING BOT — SIGNAL ENGINE v1.0
XGBoost + Full 10-Indicator Stack
Author: Built for Demik Wells trading system
=============================================================
Libraries required:
  pip install xgboost scikit-learn pandas numpy requests
  pip install pandas-ta statsmodels quantlib-python
  pip install pyfolio-reloaded vectorbt backtrader
=============================================================
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── ML / Stats ────────────────────────────────────────────
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

# ── Technical Indicators ──────────────────────────────────
try:
    import pandas_ta as ta
    PANDAS_TA = True
except ImportError:
    PANDAS_TA = False
    print("⚠️  pandas_ta not installed — using manual indicator calculations")

# ─────────────────────────────────────────────────────────
# API KEYS — matches your existing project instructions
# ─────────────────────────────────────────────────────────
POLYGON_KEY    = "zgI7pxcaymBCYt8NsPEp35FCJvhAksXz"
AV_KEY         = "PXFEPVSDRGKWVSYU"
FINNHUB_KEY    = "d6r2q5pr01qgdhqcut90d6r2q5pr01qgdhqcut9g"
FRED_KEY       = "1897439c3462a95e33dfa3e739f69ced"

# ─────────────────────────────────────────────────────────
# YOUR PORTFOLIO — from project instructions
# ─────────────────────────────────────────────────────────
PORTFOLIO = {
    "TSLA": {"shares": 700,  "avg_cost": 204.68, "contracts": 7},
    "AMD":  {"shares": 400,  "avg_cost": 129.86, "contracts": 4},
    "NVDA": {"shares": 200,  "avg_cost": 125.94, "contracts": 2},
    "SOFI": {"shares": 2000, "avg_cost": 21.09,  "contracts": 20},
    "AMZN": {"shares": 200,  "avg_cost": 40.44,  "contracts": 2},
    "HOOD": {"shares": 100,  "avg_cost": 45.00,  "contracts": 1},
}

PORTFOLIO_VALUE  = 117125
BUYING_POWER     = 24514
MAX_POSITION_PCT = 0.05
MAX_PORTFOLIO_RISK = 0.20

# ─────────────────────────────────────────────────────────
# SECTION 1: DATA FETCHING
# ─────────────────────────────────────────────────────────

def fetch_polygon_bars(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Polygon."""
    end   = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    url   = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=500"
        f"&apiKey={POLYGON_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("resultsCount", 0) == 0:
            print(f"  ⚠️  No data for {ticker}")
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap"
        })
        df = df.set_index("date")[["open","high","low","close","volume","vwap"]]
        return df
    except Exception as e:
        print(f"  ❌ Polygon fetch error for {ticker}: {e}")
        return pd.DataFrame()


def fetch_fred(series_id: str) -> float:
    """Fetch latest value from FRED."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&sort_order=desc&limit=1"
        f"&api_key={FRED_KEY}&file_type=json"
    )
    try:
        r = requests.get(url, timeout=10)
        obs = r.json()["observations"]
        return float(obs[0]["value"]) if obs else None
    except Exception as e:
        print(f"  ❌ FRED error ({series_id}): {e}")
        return None


def fetch_finnhub_quote(ticker: str) -> dict:
    """Fetch live quote from Finnhub."""
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  ❌ Finnhub error ({ticker}): {e}")
        return {}


def fetch_crypto_fg() -> int:
    """Fetch Crypto Fear & Greed index."""
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        return int(r.json()["data"][0]["value"])
    except:
        return 50  # neutral default


def fetch_macro_snapshot() -> dict:
    """Fetch all FRED macro series."""
    print("  📡 Fetching macro data from FRED...")
    series = {
        "vix":          "VIXCLS",
        "yield_curve":  "T10Y2Y",
        "fed_rate":     "FEDFUNDS",
        "cpi":          "CPIAUCSL",
        "unemployment": "UNRATE",
        "oil_wti":      "DCOILWTICO",
        "wilshire5000": "WILL5000PR",
        "gdp":          "GDP",
        "gold":         "GOLDAMGBD228NLBM",
    }
    macro = {}
    for key, sid in series.items():
        val = fetch_fred(sid)
        macro[key] = val
        time.sleep(0.2)  # respect rate limits

    # Buffett Indicator
    if macro.get("wilshire5000") and macro.get("gdp"):
        macro["buffett_indicator"] = (macro["wilshire5000"] / macro["gdp"]) * 100
    else:
        macro["buffett_indicator"] = None

    return macro


# ─────────────────────────────────────────────────────────
# SECTION 2: TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 10 indicators on OHLCV dataframe.
    Uses pandas_ta if available, otherwise manual calculation.
    """
    if df.empty or len(df) < 50:
        return df

    if PANDAS_TA:
        # ── RSI ───────────────────────────────────────────
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # ── MACD ──────────────────────────────────────────
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd"]        = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 1]
            df["macd_hist"]   = macd.iloc[:, 2]

        # ── Bollinger Bands ───────────────────────────────
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df["bb_upper"] = bbands.iloc[:, 0]
            df["bb_mid"]   = bbands.iloc[:, 1]
            df["bb_lower"] = bbands.iloc[:, 2]
            df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (
                              df["bb_upper"] - df["bb_lower"])

        # ── Stochastic ────────────────────────────────────
        stoch = ta.stoch(df["high"], df["low"], df["close"],
                         k=5, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # ── CCI ───────────────────────────────────────────
        df["cci_50"] = ta.cci(df["high"], df["low"], df["close"], length=50)
        df["cci_5"]  = ta.cci(df["high"], df["low"], df["close"], length=5)

        # ── ADX ───────────────────────────────────────────
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            df["adx"] = adx.iloc[:, 0]

        # ── ATR ───────────────────────────────────────────
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # ── EMAs ──────────────────────────────────────────
        df["ema_20"]  = ta.ema(df["close"], length=20)
        df["ema_50"]  = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)

        # ── Volume ratio ──────────────────────────────────
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(30).mean()

    else:
        # Manual fallback calculations
        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_mid"]   = df["close"].rolling(20).mean()
        std            = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * std
        df["bb_lower"] = df["bb_mid"] - 2 * std
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (
                          df["bb_upper"] - df["bb_lower"])

        # Stochastic
        low5  = df["low"].rolling(5).min()
        high5 = df["high"].rolling(5).max()
        df["stoch_k"] = 100 * (df["close"] - low5) / (high5 - low5)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # EMAs
        df["ema_20"]  = df["close"].ewm(span=20).mean()
        df["ema_50"]  = df["close"].ewm(span=50).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()

        # ADX (simplified)
        df["adx"] = 25.0  # placeholder if pandas_ta not available

        # CCI
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["cci_50"] = (tp - tp.rolling(50).mean()) / (0.015 * tp.rolling(50).std())
        df["cci_5"]  = (tp - tp.rolling(5).mean())  / (0.015 * tp.rolling(5).std())

        # Volume ratio
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(30).mean()

    # ── Derived signals ───────────────────────────────────
    # MACD crossover signal
    df["macd_cross_up"]   = (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_cross_down"] = (
        (df["macd"] < df["macd_signal"]) &
        (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)

    # Stochastic crossover
    df["stoch_cross_up"]   = (
        (df["stoch_k"] > df["stoch_d"]) &
        (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) &
        (df["stoch_k"] < 20)
    ).astype(int)
    df["stoch_cross_down"] = (
        (df["stoch_k"] < df["stoch_d"]) &
        (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) &
        (df["stoch_k"] > 80)
    ).astype(int)

    # VWAP position
    if "vwap" in df.columns:
        df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    else:
        df["above_vwap"] = 0

    # EMA alignment
    df["above_ema20"]  = (df["close"] > df["ema_20"]).astype(int)
    df["above_ema50"]  = (df["close"] > df["ema_50"]).astype(int)
    df["above_ema200"] = (df["close"] > df["ema_200"]).astype(int)
    df["ema_aligned"]  = (
        df["above_ema20"] & df["above_ema50"] & df["above_ema200"]
    ).astype(int)

    # ATR-based stop distance
    df["atr_stop"] = df["close"] - (1.5 * df["atr_14"])

    # Relative strength vs SPY (30-day return)
    df["return_30d"] = df["close"].pct_change(30)

    return df


def add_spy_rs(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Add relative strength vs SPY."""
    if spy_df.empty:
        df["rs_spy"] = 1.0
        return df
    spy_ret = spy_df["close"].pct_change(30).rename("spy_ret_30d")
    df = df.join(spy_ret, how="left")
    df["rs_spy"] = df["return_30d"] / df["spy_ret_30d"].replace(0, np.nan)
    df = df.drop("spy_ret_30d", axis=1)
    return df


# ─────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING FOR ML
# ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "macd_cross_up", "macd_cross_down",
    "bb_pct",
    "stoch_k", "stoch_d", "stoch_cross_up", "stoch_cross_down",
    "cci_50", "cci_5",
    "adx", "atr_14", "vol_ratio",
    "above_vwap", "above_ema20", "above_ema50", "above_ema200",
    "ema_aligned", "rs_spy",
    "return_30d",
]


def build_features(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Build ML feature matrix with target variable.
    Target: 1 if price is higher in `forward_days` trading days, else 0.
    Strict — no lookahead bias. Target uses FUTURE close.
    """
    df = df.copy()

    # Target variable — future return
    df["future_close"]  = df["close"].shift(-forward_days)
    df["target"]        = (df["future_close"] > df["close"]).astype(int)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # RSI normalized score (matches your TPS table)
    df["rsi_score"] = pd.cut(
        df["rsi_14"],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=[90, 70, 55, 45, 30, 10]
    ).astype(float)

    # Regime features
    df["trend_strength"]  = (df["adx"] > 25).astype(int)
    df["oversold"]        = (df["rsi_14"] < 30).astype(int)
    df["overbought"]      = (df["rsi_14"] > 70).astype(int)
    df["cci_buy_signal"]  = (
        (df["cci_50"] < -100) & (df["cci_5"] > -100)
    ).astype(int)
    df["cci_sell_signal"] = (
        (df["cci_50"] > 100) & (df["cci_5"] < 100)
    ).astype(int)
    df["stoch_oversold"]  = (df["stoch_k"] < 20).astype(int)
    df["stoch_overbought"]= (df["stoch_k"] > 80).astype(int)
    df["bb_near_lower"]   = (df["bb_pct"] < 0.2).astype(int)
    df["bb_near_upper"]   = (df["bb_pct"] > 0.8).astype(int)
    df["vol_confirms"]    = (df["vol_ratio"] > 1.5).astype(int)

    extended_features = FEATURE_COLS + [
        "rsi_score", "trend_strength", "oversold", "overbought",
        "cci_buy_signal", "cci_sell_signal",
        "stoch_oversold", "stoch_overbought",
        "bb_near_lower", "bb_near_upper", "vol_confirms",
    ]

    available = [c for c in extended_features if c in df.columns]
    df_clean = df[available + ["target", "future_return", "close"]].dropna()

    return df_clean, available


# ─────────────────────────────────────────────────────────
# SECTION 4: XGBOOST MODEL
# ─────────────────────────────────────────────────────────

def train_xgboost(df_clean: pd.DataFrame,
                  feature_cols: list,
                  ticker: str) -> dict:
    """
    Train XGBoost classifier on historical data.
    Uses TimeSeriesSplit to prevent lookahead bias.
    Returns model, scaler, feature importance, and metrics.
    """
    if len(df_clean) < 100:
        print(f"  ⚠️  Insufficient data for {ticker} ({len(df_clean)} rows)")
        return None

    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    # Time-series split — NEVER shuffle financial data
    tscv     = TimeSeriesSplit(n_splits=5)
    scores   = []
    last_model = None

    # Scale features
    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test  = y[test_idx]

        model = xgb.XGBClassifier(
            n_estimators    = 200,
            max_depth       = 4,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            min_child_weight= 5,
            gamma           = 0.1,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            use_label_encoder=False,
            eval_metric     = "logloss",
            random_state    = 42,
            verbosity       = 0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds  = model.predict(X_test)
        acc    = accuracy_score(y_test, preds)
        scores.append(acc)
        last_model = model

    # Final model on all data
    X_all = scaler.fit_transform(X)
    final_model = xgb.XGBClassifier(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 5,
        gamma           = 0.1,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        use_label_encoder=False,
        eval_metric     = "logloss",
        random_state    = 42,
        verbosity       = 0,
    )
    final_model.fit(X_all, y, verbose=False)

    # Feature importance
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)

    avg_accuracy = np.mean(scores)
    print(f"  ✅ {ticker} XGBoost trained | "
          f"Avg CV accuracy: {avg_accuracy:.1%} | "
          f"Folds: {len(scores)}")

    return {
        "model":      final_model,
        "scaler":     scaler,
        "features":   feature_cols,
        "accuracy":   avg_accuracy,
        "cv_scores":  scores,
        "importance": importance,
    }


def predict_proba(model_bundle: dict,
                  df: pd.DataFrame,
                  feature_cols: list) -> float:
    """
    Get probability of price being higher in 5 days.
    Returns float 0.0–1.0
    """
    if model_bundle is None:
        return 0.5

    latest = df[feature_cols].iloc[-1:].copy()
    if latest.isnull().any().any():
        latest = latest.fillna(latest.median())

    X = model_bundle["scaler"].transform(latest.values)
    prob = model_bundle["model"].predict_proba(X)[0][1]
    return float(prob)


# ─────────────────────────────────────────────────────────
# SECTION 5: ARIMA PRICE FORECAST (statsmodels)
# ─────────────────────────────────────────────────────────

def arima_forecast(df: pd.DataFrame,
                   ticker: str,
                   steps: int = 5) -> dict:
    """
    ARIMA(2,1,2) price forecast for next `steps` trading days.
    Returns forecast prices and confidence intervals.
    """
    try:
        close = df["close"].dropna()
        if len(close) < 60:
            return {"forecast": None, "error": "insufficient data"}

        # Use last 252 trading days (1 year) for speed
        close = close.iloc[-252:]

        model = sm.tsa.ARIMA(close, order=(2, 1, 2))
        result = model.fit()

        forecast = result.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean
        conf_int  = forecast.conf_int(alpha=0.05)

        current_price = float(close.iloc[-1])
        forecast_price = float(pred_mean.iloc[-1])
        pct_change = (forecast_price - current_price) / current_price * 100

        return {
            "current":       current_price,
            "forecast_5d":   round(forecast_price, 2),
            "pct_change":    round(pct_change, 2),
            "lower_95":      round(float(conf_int.iloc[-1, 0]), 2),
            "upper_95":      round(float(conf_int.iloc[-1, 1]), 2),
            "daily_forecasts": [round(float(p), 2) for p in pred_mean],
            "direction":     "UP ↑" if pct_change > 0 else "DOWN ↓",
        }
    except Exception as e:
        return {"forecast": None, "error": str(e)}


# ─────────────────────────────────────────────────────────
# SECTION 6: CARRY UNWIND EARLY WARNING
# ─────────────────────────────────────────────────────────

def carry_unwind_score(macro: dict) -> dict:
    """
    Score carry unwind risk 0–21 based on macro data.
    Matches your project instructions carry unwind checklist.
    """
    score = 0
    signals = {}

    # Signal 4 — VIX velocity (using level as proxy without history)
    vix = macro.get("vix")
    if vix:
        if vix > 30:
            score += 3
            signals["vix"] = f"🔴 EXTREME ({vix:.1f})"
        elif vix > 25:
            score += 3
            signals["vix"] = f"🟠 HIGH ({vix:.1f}) — velocity spike zone"
        elif vix > 20:
            score += 2
            signals["vix"] = f"🟡 ELEVATED ({vix:.1f})"
        else:
            signals["vix"] = f"🟢 LOW ({vix:.1f})"

    # Yield curve
    yc = macro.get("yield_curve")
    if yc is not None:
        if yc < 0:
            score += 2
            signals["yield_curve"] = f"🔴 INVERTED ({yc:.2f}bps)"
        elif yc < 0.5:
            score += 1
            signals["yield_curve"] = f"🟡 FLAT ({yc:.2f}bps)"
        else:
            signals["yield_curve"] = f"🟢 NORMAL ({yc:.2f}bps)"

    # Buffett indicator
    bi = macro.get("buffett_indicator")
    if bi:
        if bi > 200:
            score += 3
            signals["buffett"] = f"🔴 EXTREME OVERVALUED ({bi:.0f}%)"
        elif bi > 160:
            score += 2
            signals["buffett"] = f"🟠 OVERVALUED ({bi:.0f}%)"
        elif bi > 120:
            score += 1
            signals["buffett"] = f"🟡 SLIGHTLY OVERVALUED ({bi:.0f}%)"
        else:
            signals["buffett"] = f"🟢 FAIR VALUE ({bi:.0f}%)"

    # Oil (war premium proxy)
    oil = macro.get("oil_wti")
    if oil:
        if oil > 95:
            score += 3
            signals["oil"] = f"🔴 WAR PREMIUM (${oil:.2f})"
        elif oil > 80:
            score += 1
            signals["oil"] = f"🟡 ELEVATED (${oil:.2f})"
        else:
            signals["oil"] = f"🟢 NORMAL (${oil:.2f})"

    # Classify
    if score >= 13:
        risk_level = "🔴 UNWIND IN PROGRESS — DEFENSIVE MODE"
        action = "Tighten all stops to 1×ATR | Reduce positions 25–50% | Hedge immediately"
    elif score >= 8:
        risk_level = "🟠 ELEVATED RISK — NO NEW LONGS"
        action = "No new longs | Tighten stops | Sell covered calls on all eligible"
    elif score >= 4:
        risk_level = "🟡 EARLY WARNING — REDUCE SIZE 25%"
        action = "Reduce new position sizes 25% | Watch USD/JPY closely"
    else:
        risk_level = "🟢 NO MEANINGFUL UNWIND RISK"
        action = "Trade normally per regime"

    return {
        "score":      score,
        "max_score":  21,
        "risk_level": risk_level,
        "action":     action,
        "signals":    signals,
    }


# ─────────────────────────────────────────────────────────
# SECTION 7: REGIME DETECTOR
# ─────────────────────────────────────────────────────────

def detect_regime(macro: dict, spy_df: pd.DataFrame) -> dict:
    """
    Classify market regime using SPY vs EMA200, VIX,
    yield curve, and carry unwind score.
    """
    regime = "🟡 CHOPPY"
    strategy = "Debit spreads / Covered calls"

    vix = macro.get("vix", 20)
    yc  = macro.get("yield_curve", 0.5)
    bi  = macro.get("buffett_indicator", 100)

    # SPY vs 200 EMA
    spy_above_200 = False
    if not spy_df.empty and "ema_200" in spy_df.columns:
        latest_spy   = float(spy_df["close"].iloc[-1])
        ema200_spy   = float(spy_df["ema_200"].iloc[-1])
        spy_above_200 = latest_spy > ema200_spy

    if vix > 30:
        regime   = "🔴 VOLATILITY SPIKE"
        strategy = "Iron condors / Short vol / Cash"
    elif yc < 0:
        regime   = "🟣 RECESSION RISK"
        strategy = "Defensive — bonds / energy / cash"
    elif not spy_above_200 and vix > 20:
        regime   = "🟠 BEAR / STAGFLATION"
        strategy = "Buy puts / Credit spreads / Covered calls"
    elif spy_above_200 and vix < 20:
        regime   = "🟢 BULL TREND"
        strategy = "Buy calls / Sell puts"
    elif spy_above_200 and vix <= 30:
        regime   = "🟡 CHOPPY BULL"
        strategy = "Debit spreads / Covered calls"

    # Buffett override
    if bi and bi > 200:
        strategy += " | ⚠️ BUFFETT 200%+ — NO NEW SPECULATIVE LONGS"

    return {
        "regime":        regime,
        "strategy":      strategy,
        "vix":           vix,
        "yield_curve":   yc,
        "spy_above_200": spy_above_200,
        "buffett_pct":   bi,
    }


# ─────────────────────────────────────────────────────────
# SECTION 8: TPS SCORE (ML-ENHANCED)
# ─────────────────────────────────────────────────────────

def calculate_tps(df: pd.DataFrame,
                  ml_prob: float,
                  macro: dict,
                  arima: dict) -> dict:
    """
    Calculate Trade Probability Score combining:
    - XGBoost ML probability (replaces manual technical scoring)
    - Options flow signals (PCR proxy)
    - Macro / FRED data
    - ARIMA directional forecast
    """
    latest = df.iloc[-1]
    scores = {}

    # ── Technical (ML-powered) ────────────────────────────
    # XGBoost probability already incorporates all 10 indicators
    tech_score = ml_prob * 100  # 0–100
    scores["technical_ml"] = tech_score

    # ── Normalize RSI ─────────────────────────────────────
    rsi = latest.get("rsi_14", 50)
    if rsi < 30:   rsi_score = 90
    elif rsi < 40: rsi_score = 70
    elif rsi < 50: rsi_score = 55
    elif rsi < 60: rsi_score = 45
    elif rsi < 70: rsi_score = 30
    else:          rsi_score = 10
    scores["rsi"] = rsi_score

    # ── Stochastic ────────────────────────────────────────
    stoch_k = latest.get("stoch_k", 50)
    stoch_d = latest.get("stoch_d", 50)
    if stoch_k < 20 and stoch_k > stoch_d:
        stoch_score = 85  # oversold + crossing up = strong buy
    elif stoch_k > 80 and stoch_k < stoch_d:
        stoch_score = 15  # overbought + crossing down = sell
    elif stoch_k < 30:
        stoch_score = 65
    elif stoch_k > 70:
        stoch_score = 35
    else:
        stoch_score = 50
    scores["stochastic"] = stoch_score

    # ── ADX trend confirmation ────────────────────────────
    adx = latest.get("adx", 20)
    adx_bonus = 10 if adx > 25 else 0
    scores["adx_bonus"] = adx_bonus

    # ── ARIMA directional ─────────────────────────────────
    arima_score = 55  # neutral default
    if arima.get("pct_change"):
        pct = arima["pct_change"]
        if pct > 3:    arima_score = 80
        elif pct > 1:  arima_score = 65
        elif pct > 0:  arima_score = 55
        elif pct > -1: arima_score = 45
        elif pct > -3: arima_score = 35
        else:          arima_score = 20
    scores["arima"] = arima_score

    # ── Macro score ───────────────────────────────────────
    vix = macro.get("vix", 20)
    yc  = macro.get("yield_curve", 0.5)
    macro_score = 50
    if vix < 15 and yc > 0:   macro_score = 70
    elif vix > 30 or yc < 0:  macro_score = 25
    elif vix > 25:             macro_score = 35
    scores["macro"] = macro_score

    # ── Weighted TPS ──────────────────────────────────────
    # ML probability replaces manual technical weight
    tps = (
        tech_score  * 0.35 +   # XGBoost (upgraded weight — data-learned)
        stoch_score * 0.10 +   # New stochastic
        arima_score * 0.15 +   # New ARIMA forecast
        rsi_score   * 0.10 +   # RSI confirmation
        macro_score * 0.20 +   # Macro regime
        adx_bonus   * 0.10     # ADX trend confirmation
    )

    # Market-implied probability proxy (using ATM delta ~0.50)
    market_implied = 50.0
    edge = tps - market_implied

    verdict = "❌ NO TRADE — edge insufficient"
    if edge >= 15:
        verdict = "🔥 HIGH CONVICTION — full size"
    elif edge >= 10:
        verdict = "✅ TRADE — standard size"
    elif edge >= 5:
        verdict = "⚠️ BORDERLINE — half size only"

    return {
        "tps":            round(tps, 1),
        "edge":           round(edge, 1),
        "verdict":        verdict,
        "ml_probability": round(ml_prob * 100, 1),
        "scores":         scores,
        "market_implied": market_implied,
    }


# ─────────────────────────────────────────────────────────
# SECTION 9: COVERED CALL ANALYZER
# ─────────────────────────────────────────────────────────

def analyze_covered_call(ticker: str,
                          df: pd.DataFrame,
                          portfolio_info: dict,
                          macro: dict) -> dict:
    """
    Analyze covered call opportunity for a given ticker.
    Returns optimal strike, premium estimate, and yield.
    """
    current_price = float(df["close"].iloc[-1])
    avg_cost      = portfolio_info["avg_cost"]
    contracts     = portfolio_info["contracts"]
    atr           = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.03
    vix           = macro.get("vix", 20)

    # IV proxy: higher VIX = higher IV = richer premiums
    iv_multiplier = 1 + (vix - 15) / 100

    # OTM strikes 5% and 10% above current
    strike_5pct  = round(current_price * 1.05, 0)
    strike_10pct = round(current_price * 1.10, 0)

    # Ensure strike is above avg cost
    strike_5pct  = max(strike_5pct,  avg_cost * 1.02)
    strike_10pct = max(strike_10pct, avg_cost * 1.05)

    # Premium estimate using simplified Black-Scholes proxy
    # Premium ≈ ATR × IV_multiplier × time_factor
    time_factor_weekly    = 0.30  # 1 week theta
    time_factor_monthly   = 0.55  # end of month

    prem_weekly_5pct  = round(atr * iv_multiplier * time_factor_weekly  * 0.6, 2)
    prem_weekly_10pct = round(atr * iv_multiplier * time_factor_weekly  * 0.3, 2)
    prem_month_5pct   = round(atr * iv_multiplier * time_factor_monthly * 0.6, 2)
    prem_month_10pct  = round(atr * iv_multiplier * time_factor_monthly * 0.3, 2)

    # Total income
    total_weekly  = round(prem_weekly_5pct * contracts * 100, 0)
    total_monthly = round(prem_month_5pct  * contracts * 100, 0)

    # Annualized yield on premium vs stock price
    ann_yield_weekly  = round((prem_weekly_5pct / current_price) * 52 * 100, 1)
    ann_yield_monthly = round((prem_month_5pct  / current_price) * 12 * 100, 1)

    # P&L if called away
    pnl_if_assigned = round((strike_5pct - avg_cost) * contracts * 100, 0)

    above_cost = strike_5pct > avg_cost

    return {
        "ticker":            ticker,
        "current_price":     current_price,
        "avg_cost":          avg_cost,
        "contracts":         contracts,
        "atr":               round(atr, 2),
        "vix":               vix,
        "above_cost_basis":  above_cost,
        "weekly": {
            "strike":        strike_5pct,
            "premium":       prem_weekly_5pct,
            "total_income":  total_weekly,
            "ann_yield":     ann_yield_weekly,
        },
        "monthly": {
            "strike":        strike_10pct,
            "premium":       prem_month_5pct,
            "total_income":  total_monthly,
            "ann_yield":     ann_yield_monthly,
        },
        "pnl_if_assigned":   pnl_if_assigned,
    }


# ─────────────────────────────────────────────────────────
# SECTION 10: RISK MANAGEMENT
# ─────────────────────────────────────────────────────────

def calculate_position_size(price: float,
                             atr: float,
                             portfolio_value: float = PORTFOLIO_VALUE,
                             risk_pct: float = 0.015) -> dict:
    """
    Calculate position size based on ATR stop and portfolio risk.
    risk_pct = max risk as fraction of portfolio (default 1.5%)
    """
    stop_distance = 1.5 * atr
    stop_price    = price - stop_distance
    risk_per_share = stop_distance

    max_risk_dollars = portfolio_value * risk_pct
    shares = int(max_risk_dollars / risk_per_share)
    position_value = shares * price
    position_pct   = position_value / portfolio_value

    # Cap at 5% of portfolio
    if position_pct > 0.05:
        shares = int((portfolio_value * 0.05) / price)
        position_value = shares * price
        position_pct   = position_value / portfolio_value

    return {
        "shares":          shares,
        "position_value":  round(position_value, 0),
        "position_pct":    round(position_pct * 100, 1),
        "stop_price":      round(stop_price, 2),
        "stop_distance":   round(stop_distance, 2),
        "risk_dollars":    round(shares * risk_per_share, 0),
        "risk_pct":        round((shares * risk_per_share / portfolio_value) * 100, 2),
    }


# ─────────────────────────────────────────────────────────
# SECTION 11: MASTER SCAN
# ─────────────────────────────────────────────────────────

def run_full_scan(tickers: list = None,
                  forward_days: int = 5) -> dict:
    """
    Master scan — runs full pipeline for all tickers.
    Fetches data, calculates indicators, trains XGBoost,
    runs ARIMA, calculates TPS, generates trade blocks.
    """
    if tickers is None:
        tickers = list(PORTFOLIO.keys())

    print("\n" + "="*60)
    print("🤖 TRADING BOT — FULL SCAN INITIATED")
    print("="*60)

    # ── 1. Macro data ─────────────────────────────────────
    print("\n📡 STEP 1: Fetching macro data...")
    macro = fetch_macro_snapshot()
    carry = carry_unwind_score(macro)

    # ── 2. SPY baseline ───────────────────────────────────
    print("\n📊 STEP 2: Fetching SPY data...")
    spy_df = fetch_polygon_bars("SPY", days=400)
    if not spy_df.empty:
        spy_df = calculate_indicators(spy_df)

    # ── 3. Regime detection ───────────────────────────────
    regime = detect_regime(macro, spy_df)

    # ── 4. Per-ticker analysis ────────────────────────────
    print(f"\n📈 STEP 3: Analyzing {len(tickers)} tickers...")
    results = {}

    for ticker in tickers:
        print(f"\n  → Processing {ticker}...")

        # Fetch data
        df = fetch_polygon_bars(ticker, days=400)
        if df.empty:
            results[ticker] = {"error": "no data"}
            continue
        time.sleep(0.5)  # Polygon rate limit

        # Calculate indicators
        df = calculate_indicators(df)
        df = add_spy_rs(df, spy_df)

        # Build features
        df_clean, feature_cols = build_features(df, forward_days=forward_days)

        # Train XGBoost
        model_bundle = train_xgboost(df_clean, feature_cols, ticker)

        # Get prediction probability
        ml_prob = predict_proba(model_bundle, df_clean, feature_cols)

        # ARIMA forecast
        arima = arima_forecast(df, ticker, steps=forward_days)

        # TPS score
        tps_result = calculate_tps(df, ml_prob, macro, arima)

        # Covered call analysis
        if ticker in PORTFOLIO:
            cc = analyze_covered_call(
                ticker, df, PORTFOLIO[ticker], macro
            )
        else:
            cc = None

        # Risk/position sizing
        atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else 0
        price = float(df["close"].iloc[-1])
        sizing = calculate_position_size(price, atr)

        # Live quote
        quote = fetch_finnhub_quote(ticker)
        time.sleep(0.3)

        results[ticker] = {
            "price":        price,
            "quote":        quote,
            "indicators":   df.iloc[-1].to_dict(),
            "ml_prob":      ml_prob,
            "arima":        arima,
            "tps":          tps_result,
            "covered_call": cc,
            "sizing":       sizing,
            "model":        model_bundle,
            "feature_importance": (
                model_bundle["importance"].head(10).to_dict("records")
                if model_bundle else None
            ),
        }

    # ── 5. Compile report ─────────────────────────────────
    return {
        "macro":    macro,
        "carry":    carry,
        "regime":   regime,
        "results":  results,
        "spy":      {
            "price":    float(spy_df["close"].iloc[-1]) if not spy_df.empty else None,
            "ema_200":  float(spy_df["ema_200"].iloc[-1]) if not spy_df.empty and "ema_200" in spy_df.columns else None,
        }
    }


# ─────────────────────────────────────────────────────────
# SECTION 12: REPORT GENERATOR
# ─────────────────────────────────────────────────────────

def print_report(scan: dict):
    """Print formatted trading report matching your trade block format."""

    macro  = scan["macro"]
    carry  = scan["carry"]
    regime = scan["regime"]
    results= scan["results"]
    spy    = scan["spy"]

    print("\n" + "="*60)
    print("🧭 MARKET REGIME DETECTOR")
    print("="*60)
    print(f"SPY price:         ${spy.get('price', 'N/A'):.2f}")
    print(f"SPY 200 EMA:       ${spy.get('ema_200', 0):.2f} "
          f"[{'above ✅' if regime['spy_above_200'] else 'below ❌'}]")
    print(f"VIX:               {regime['vix']:.1f}")
    print(f"Yield curve:       {regime['yield_curve']:.3f}bps")
    print(f"Current regime:    {regime['regime']}")
    print(f"Optimal strategy:  {regime['strategy']}")

    print("\n" + "="*60)
    print("🏦 BUFFETT INDICATOR")
    print("="*60)
    bi = macro.get("buffett_indicator")
    if bi:
        print(f"Buffett Indicator: {bi:.0f}%")
        if bi > 200:
            print("Status:            🔴 PLAYING WITH FIRE — market strongly overvalued")
        elif bi > 160:
            print("Status:            🟠 OVERVALUED")
        elif bi > 120:
            print("Status:            🟡 SLIGHTLY OVERVALUED")
        else:
            print("Status:            🟢 FAIR VALUE")

    print("\n" + "="*60)
    print(f"⚠️  CARRY UNWIND SCORE: {carry['score']}/{carry['max_score']}")
    print("="*60)
    print(f"Risk level:        {carry['risk_level']}")
    print(f"Action:            {carry['action']}")
    for k, v in carry["signals"].items():
        print(f"  {k:15s}:  {v}")

    print("\n" + "="*60)
    print("📊 MACRO SNAPSHOT")
    print("="*60)
    print(f"Fed rate:          {macro.get('fed_rate', 'N/A')}%")
    print(f"CPI:               {macro.get('cpi', 'N/A')}")
    print(f"Unemployment:      {macro.get('unemployment', 'N/A')}%")
    print(f"Oil WTI:           ${macro.get('oil_wti', 'N/A')}")
    print(f"Gold:              ${macro.get('gold', 'N/A')}")
    print(f"VIX:               {macro.get('vix', 'N/A')}")

    print("\n" + "="*60)
    print("🎯 PER-TICKER ANALYSIS")
    print("="*60)

    for ticker, data in results.items():
        if "error" in data:
            print(f"\n{ticker}: ❌ {data['error']}")
            continue

        ind    = data["indicators"]
        tps    = data["tps"]
        arima  = data["arima"]
        cc     = data["covered_call"]
        sizing = data["sizing"]

        print(f"\n{'─'*50}")
        print(f"📌 {ticker} @ ${data['price']:.2f}")
        print(f"{'─'*50}")

        # Core indicators
        print(f"RSI(14):           {ind.get('rsi_14', 0):.1f}")
        print(f"Stochastic %K/%D:  {ind.get('stoch_k', 0):.1f} / {ind.get('stoch_d', 0):.1f}")
        print(f"CCI(50):           {ind.get('cci_50', 0):.1f}")
        print(f"CCI(5):            {ind.get('cci_5', 0):.1f}")
        print(f"ADX(14):           {ind.get('adx', 0):.1f} "
              f"[{'strong ✅' if ind.get('adx', 0) > 25 else 'ranging ⚠️'}]")
        print(f"ATR(14):           ${ind.get('atr_14', 0):.2f}")
        print(f"BB Position:       {ind.get('bb_pct', 0.5):.2f} "
              f"[{'near lower ✅' if ind.get('bb_pct', 0.5) < 0.2 else 'near upper ⚠️' if ind.get('bb_pct', 0.5) > 0.8 else 'middle'}]")
        print(f"EMA aligned:       {'✅ All above' if ind.get('ema_aligned') else '❌ Mixed'}")
        print(f"RS vs SPY:         {ind.get('rs_spy', 1.0):.2f} "
              f"[{'outperforming ✅' if ind.get('rs_spy', 1.0) > 1 else 'underperforming ❌'}]")
        print(f"Volume ratio:      {ind.get('vol_ratio', 1.0):.2f}x avg")

        # ML prediction
        print(f"\n🤖 ML PREDICTION (XGBoost)")
        print(f"Probability up 5d: {data['ml_prob']*100:.1f}%")
        print(f"TPS Score:         {tps['tps']:.1f}")
        print(f"Market implied:    {tps['market_implied']:.1f}%")
        print(f"Edge:              {tps['edge']:+.1f}%")
        print(f"Verdict:           {tps['verdict']}")

        # ARIMA forecast
        if arima.get("forecast_5d"):
            print(f"\n📈 ARIMA FORECAST (5-day)")
            print(f"Current:           ${arima['current']:.2f}")
            print(f"Forecast:          ${arima['forecast_5d']:.2f} "
                  f"({arima['pct_change']:+.1f}%) {arima['direction']}")
            print(f"95% CI:            ${arima['lower_95']:.2f} – ${arima['upper_95']:.2f}")

        # Feature importance top 5
        fi = data.get("feature_importance")
        if fi:
            print(f"\n🔍 TOP 5 SIGNAL WEIGHTS (XGBoost learned)")
            for i, f in enumerate(fi[:5]):
                bar = "█" * int(f["importance"] * 100)
                print(f"  {i+1}. {f['feature']:20s} {f['importance']:.3f}  {bar}")

        # Covered call
        if cc:
            print(f"\n💰 COVERED CALL OPPORTUNITY")
            print(f"Above cost basis:  {'✅' if cc['above_cost_basis'] else '❌'} "
                  f"(cost: ${cc['avg_cost']:.2f})")
            print(f"\n  NEXT WEEK:")
            print(f"  Strike:          ${cc['weekly']['strike']:.0f}")
            print(f"  Premium est:     ${cc['weekly']['premium']:.2f}/share")
            print(f"  Total income:    ${cc['weekly']['total_income']:,.0f} "
                  f"({cc['contracts']} contracts)")
            print(f"  Annualized yield:{cc['weekly']['ann_yield']}%")
            print(f"\n  END OF MONTH:")
            print(f"  Strike:          ${cc['monthly']['strike']:.0f}")
            print(f"  Premium est:     ${cc['monthly']['premium']:.2f}/share")
            print(f"  Total income:    ${cc['monthly']['total_income']:,.0f}")
            print(f"  Annualized yield:{cc['monthly']['ann_yield']}%")
            print(f"  P&L if assigned: ${cc['pnl_if_assigned']:,.0f}")

        # Position sizing
        if tps["edge"] >= 10:
            print(f"\n📐 POSITION SIZING (new trade)")
            print(f"Shares:            {sizing['shares']}")
            print(f"Position value:    ${sizing['position_value']:,.0f} "
                  f"({sizing['position_pct']}% of portfolio)")
            print(f"ATR Stop:          ${sizing['stop_price']:.2f}")
            print(f"Risk on trade:     ${sizing['risk_dollars']:,.0f} "
                  f"({sizing['risk_pct']}%)")

    # Income summary
    print("\n" + "="*60)
    print("💵 COVERED CALL INCOME SUMMARY")
    print("="*60)
    total_weekly  = 0
    total_monthly = 0
    for ticker, data in results.items():
        cc = data.get("covered_call")
        if cc:
            total_weekly  += cc["weekly"]["total_income"]
            total_monthly += cc["monthly"]["total_income"]
            print(f"  {ticker:6s}: weekly=${cc['weekly']['total_income']:>8,.0f}  "
                  f"monthly=${cc['monthly']['total_income']:>8,.0f}")
    print(f"{'─'*50}")
    print(f"  TOTAL: weekly=${total_weekly:>8,.0f}  monthly=${total_monthly:>8,.0f}")
    print(f"  Capital required: $0 (all covered)")

    print("\n" + "="*60)
    print("⚠️  DISCLAIMER")
    print("="*60)
    print("Research and signal generation only — not financial advice.")
    print("All premiums are estimates. Verify live bid/ask before executing.")
    print("Options involve substantial risk. Past performance ≠ future results.")
    print("="*60)


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot Signal Engine")
    parser.add_argument("--tickers", nargs="+",
                        default=list(PORTFOLIO.keys()),
                        help="Tickers to scan")
    parser.add_argument("--days", type=int, default=5,
                        help="Forward days for ML target (default: 5)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    scan = run_full_scan(tickers=args.tickers, forward_days=args.days)
    print_report(scan)

    if args.save:
        # Save results (excluding non-serializable model objects)
        save_data = {
            "macro":   scan["macro"],
            "carry":   scan["carry"],
            "regime":  scan["regime"],
            "spy":     scan["spy"],
            "results": {
                t: {k: v for k, v in d.items()
                    if k not in ["model", "indicators"]}
                for t, d in scan["results"].items()
                if "error" not in d
            }
        }
        with open("scan_results.json", "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print("\n✅ Results saved to scan_results.json")
