# Data Science Bundle
> Skills: EDA · Polars · SHAP · Statistical Analysis · Time Series Forecasting · XGBoost · Stock Prediction
> Usage: Paste this URL at session start for all data science tasks.

---

## 1. Exploratory Data Analysis (EDA)
**Trigger:** "explore/analyze/summarize a data file"
**Supports:** 200+ formats — CSV, FASTQ, PDB, TIFF, HDF5, NPY, DICOM, mzML, etc.

**Workflow:**
1. Detect file type by extension
2. Load format-specific analysis approach
3. Perform structure + quality + stats analysis
4. Generate markdown report → `{filename}_eda_report.md`

**Report sections:** Basic Info · File Type Details · Data Analysis · Key Findings · Recommendations

**By data type:**
```python
# Tabular (CSV/Excel)
import pandas as pd
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes, df.isnull().sum(), df.describe())

# Sequence (FASTA/FASTQ)
from Bio import SeqIO
seqs = list(SeqIO.parse('reads.fastq', 'fastq'))
# → count, length dist, GC content, quality scores

# Arrays (NPY/HDF5)
import numpy as np, h5py
arr = np.load('data.npy')
print(arr.shape, arr.dtype, arr.min(), arr.max(), np.isnan(arr).sum())
```

**Common libs by category:**
- Bioinformatics: `biopython`, `pysam`
- Chemistry: `rdkit`, `mdanalysis`
- Microscopy: `tifffile`, `nd2reader`, `pydicom`
- General: `pandas`, `numpy`, `h5py`

---

## 2. Polars
**Trigger:** "fast dataframe", "pandas too slow", "1-100GB dataset", "ETL pipeline"

**Core pattern:**
```python
import polars as pl

# Lazy (preferred for large data)
result = (
    pl.scan_csv("large.csv")
    .filter(pl.col("age") > 25)
    .with_columns(pl.col("value") * 10)
    .group_by("city").agg(pl.col("age").mean())
    .collect()
)

# Window functions
df.with_columns(
    avg_by_city=pl.col("age").mean().over("city"),
    rank=pl.col("salary").rank().over("city")
)

# Joins / concat
df1.join(df2, on="id", how="left")
pl.concat([df1, df2], how="vertical")
```

**Pandas → Polars quick map:**
| Pandas | Polars |
|--------|--------|
| `df[df["x"] > 10]` | `df.filter(pl.col("x") > 10)` |
| `df.assign(x=...)` | `df.with_columns(x=...)` |
| `df.groupby().transform()` | `df.with_columns(...).over()` |

**Performance rules:** use lazy, select early, avoid `map_elements`, use `streaming=True` for huge data.

---

## 3. SHAP (Model Explainability)
**Trigger:** "explain model", "feature importance", "why did model predict", "SHAP plots"

**Explainer selection:**
```python
import shap

# Tree models (XGBoost, LightGBM, RF) — fastest
explainer = shap.TreeExplainer(model)

# Neural nets (TF/PyTorch)
explainer = shap.DeepExplainer(model, background_data)

# Linear models
explainer = shap.LinearExplainer(model, X_train)

# Any black-box (slow)
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

# Auto-select
explainer = shap.Explainer(model)
```

**Core workflow:**
```python
shap_values = explainer(X_test)

# Global importance
shap.plots.beeswarm(shap_values)     # distribution + magnitude
shap.plots.bar(shap_values)          # mean absolute SHAP

# Single prediction
shap.plots.waterfall(shap_values[0]) # feature contributions
shap.plots.force(shap_values[0])     # additive force view

# Feature relationships
shap.plots.scatter(shap_values[:, "feature_name"])
```

**Interpretation:** positive SHAP = pushes prediction up · magnitude = strength · values sum to prediction − baseline

---

## 4. Statistical Analysis
**Trigger:** "hypothesis test", "t-test", "ANOVA", "regression", "p-value", "APA report"

**Test selection quick ref:**
| Scenario | Test |
|----------|------|
| 2 groups, continuous, normal | Independent t-test |
| 2 groups, non-normal | Mann-Whitney U |
| 2 groups, paired | Paired t-test / Wilcoxon |
| 3+ groups, normal | One-way ANOVA → Tukey HSD |
| 3+ groups, non-normal | Kruskal-Wallis |
| 2 continuous vars | Pearson / Spearman correlation |
| Binary outcome | Logistic regression |

**Core pattern (pingouin):**
```python
import pingouin as pg

# T-test with effect size
result = pg.ttest(group_a, group_b, correction='auto')
# → T, dof, p-val, cohen-d, CI95%

# ANOVA + post-hoc
aov = pg.anova(dv='score', between='group', data=df, detailed=True)
if aov['p-unc'].values[0] < 0.05:
    posthoc = pg.pairwise_tukey(dv='score', between='group', data=df)
```

**APA report format:**
```
t(98) = 3.82, p < .001, d = 0.77, 95% CI [0.36, 1.18]
F(2, 147) = 8.45, p < .001, η²_p = .10
```

**Effect size benchmarks:**
| Test | Small | Medium | Large |
|------|-------|--------|-------|
| Cohen's d | 0.20 | 0.50 | 0.80 |
| η²_p | 0.01 | 0.06 | 0.14 |
| r | 0.10 | 0.30 | 0.50 |

**Always:** check assumptions first (`pg.normality`, `pg.homoscedasticity`) · report effect sizes + CIs · distinguish statistical from practical significance.

---

## 5. Time Series Forecasting (ARIMA / GARCH)
**Trigger:** "forecast prices", "time series model", "ARIMA", "GARCH", "volatility modeling", "stock prediction"
**Requires:** `pip install statsmodels arch pmdarima`

**ARIMA — price/returns forecasting:**
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Step 1: Check stationarity
result = adfuller(df['close'])
print(f"ADF p-value: {result[1]:.4f}")  # p < 0.05 → stationary
# If non-stationary → difference: df['returns'] = df['close'].pct_change().dropna()

# Step 2: Fit ARIMA(p, d, q)
model = ARIMA(df['close'], order=(2, 1, 2))
fit = model.fit()
print(fit.summary())

# Step 3: Forecast
forecast = fit.get_forecast(steps=10)
pred_mean = forecast.predicted_mean
pred_ci   = forecast.conf_int()  # 95% confidence interval

# Auto-select p,d,q with pmdarima
from pmdarima import auto_arima
auto_model = auto_arima(df['close'], seasonal=False, stepwise=True, trace=True)
```

**GARCH — volatility forecasting:**
```python
from arch import arch_model

returns = df['close'].pct_change().dropna() * 100  # scale for numerical stability

# Fit GARCH(1,1) — most common
garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
res = garch.fit(disp='off')
print(res.summary())

# Forecast volatility
forecasts = res.forecast(horizon=5)
vol_forecast = forecasts.variance.iloc[-1] ** 0.5  # annualize: * sqrt(252)

# Variants
# GJR-GARCH (asymmetric — captures bad news effect)
gjr = arch_model(returns, vol='Garch', p=1, o=1, q=1)

# EGARCH (log variance — no non-negativity constraint)
egarch = arch_model(returns, vol='EGarch', p=1, q=1)
```

**Model selection:**
| Use case | Model |
|----------|-------|
| Price level forecasting | ARIMA |
| Return forecasting | ARIMA on log-returns |
| Volatility / options | GARCH(1,1) |
| Leverage effect | GJR-GARCH |
| Risk metrics (VaR) | GARCH + normal/t dist |

**Pitfalls:** never forecast raw prices with ARIMA long-term · always check residuals (`plot_diagnostics`) · GARCH on returns not prices.

---

## 6. XGBoost for Tabular Finance Features
**Trigger:** "predict price movement", "classify market regime", "feature-based stock model", "XGBoost"
**Requires:** `pip install xgboost scikit-learn pandas-ta`

**Full pipeline — price direction classification:**
```python
import xgboost as xgb
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import shap

# Step 1: Build features from OHLCV
def build_features(df):
    df = df.copy()
    # Technical indicators via pandas-ta
    df['rsi']     = ta.rsi(df['close'], length=14)
    df['macd']    = ta.macd(df['close'])['MACD_12_26_9']
    df['bb_pct']  = ta.bbands(df['close'])['BBP_5_2.0']   # % inside bands
    df['atr']     = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['obv']     = ta.obv(df['close'], df['volume'])
    # Lag features
    for lag in [1, 3, 5, 10]:
        df[f'return_{lag}d'] = df['close'].pct_change(lag)
    # Target: next-day direction (1=up, 0=down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

df = build_features(ohlcv_df)
features = [c for c in df.columns if c not in ['open','high','low','close','volume','target']]
X, y = df[features], df['target']

# Step 2: Time-series cross-validation (NO random shuffle)
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=20,
              verbose=False)
    preds = model.predict(X_val)
    print(f"Fold {fold+1}:\n", classification_report(y_val, preds))

# Step 3: SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_val)
shap.plots.beeswarm(shap_values)
```

**Regression variant (predict return magnitude):**
```python
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
df['target'] = df['close'].pct_change().shift(-1)  # next-day return
```

**Golden rules:**
- Always use `TimeSeriesSplit` — never `train_test_split` with shuffle on time series
- Features must use only past data (no lookahead) — shift targets by -1, not features forward
- Normalize volume/price features or use returns/ratios instead of raw prices
- Combine with SHAP to audit which indicators actually drive predictions

---

## 7. Technical Indicators (pandas-ta)
**Trigger:** "RSI", "MACD", "Bollinger Bands", "technical analysis", "indicators", "TA"
**Requires:** `pip install pandas-ta`

**Core indicators:**
```python
import pandas_ta as ta
import pandas as pd

# df must have columns: open, high, low, close, volume (lowercase)

# Momentum
rsi  = ta.rsi(df['close'], length=14)             # 0-100, overbought >70
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
# → MACD_12_26_9, MACDh_12_26_9 (histogram), MACDs_12_26_9 (signal)
stoch = ta.stoch(df['high'], df['low'], df['close'])

# Volatility
bb   = ta.bbands(df['close'], length=20, std=2)
# → BBL (lower), BBM (mid), BBU (upper), BBB (bandwidth), BBP (percent)
atr  = ta.atr(df['high'], df['low'], df['close'], length=14)

# Trend
ema20  = ta.ema(df['close'], length=20)
sma50  = ta.sma(df['close'], length=50)
adx    = ta.adx(df['high'], df['low'], df['close'], length=14)
# → ADX_14, DMP_14 (+DI), DMN_14 (-DI)

# Volume
obv   = ta.obv(df['close'], df['volume'])
vwap  = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

# Add all indicators at once (appends to df)
df.ta.strategy("all")  # 150+ indicators — use selectively

# Custom strategy (selective)
MyStrategy = ta.Strategy(
    name="quant_signals",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20},
        {"kind": "atr", "length": 14},
    ]
)
df.ta.strategy(MyStrategy)
```

**Signal interpretation quick ref:**
| Indicator | Bullish | Bearish |
|-----------|---------|---------|
| RSI | <30 (oversold) | >70 (overbought) |
| MACD | MACD crosses above signal | MACD crosses below signal |
| BBands | Price touches lower band | Price touches upper band |
| ADX | >25 = strong trend | <20 = weak/ranging |
| VWAP | Price above VWAP | Price below VWAP |

**Pitfall:** `df.ta.strategy("all")` is slow and adds noise — always select only what your model needs.
