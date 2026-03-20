# 🤖 Trading Bot — Signal Engine

Personal AI trading research bot built on top of a full 10-indicator stack,
XGBoost ML predictions, ARIMA forecasting, carry unwind detection, and
intermarket flow analysis (Crude → DXY → USD/JPY → Stocks).

---

## Repository Structure

```
trading_bot/
├── signal_engine.py    # Core — data fetch, indicators, XGBoost, TPS, report
├── backtester.py       # VectorBT strategy optimization + covered call backtest
├── alerts.py           # Telegram morning brief scheduler
├── requirements.txt    # All Python dependencies
└── README.md           # This file
```

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/holmesstephen067-glitch/my_AI_brain.git
cd my_AI_brain/trading_bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full scan (uses API keys already in signal_engine.py)
python signal_engine.py

# 4. Save results to JSON
python signal_engine.py --save

# 5. Scan specific tickers
python signal_engine.py --tickers TSLA NVDA AMD

# 6. Run backtest optimizer for one ticker
python backtester.py TSLA

# 7. Test Telegram alerts
python alerts.py test

# 8. Start scheduled morning briefs (7:30 AM daily)
python alerts.py
```

---

## What It Does

### Signal Engine (`signal_engine.py`)

**Data Layer**
- Fetches 400 days of daily OHLCV from Polygon.io
- Live quotes from Finnhub
- Full FRED macro suite (VIX, yield curve, CPI, oil, gold, unemployment)
- Buffett Indicator (Wilshire 5000 / GDP)
- Crypto Fear & Greed

**All 10 Indicators (calculated fresh every run)**
1. RSI(14) — normalized scoring table
2. MACD — crossover + histogram expansion
3. Bollinger Bands — %B position
4. ADX(14) — trend strength filter
5. Stochastic %K/%D — oversold/overbought crossover signals *(new)*
6. CCI(50) + CCI(5) — divergence and signal confirmation
7. ATR(14) — dynamic stop placement (1.5×ATR)
8. EMA 20/50/200 — trend structure + golden cross
9. Volume ratio — 30-day average comparison
10. Relative Strength vs SPY — 30-day momentum

**ML Layer**
- XGBoost classifier trained on all 10 indicators
- TimeSeriesSplit cross-validation (no lookahead bias)
- Feature importance: tells you which indicators matter most per ticker
- ARIMA(2,1,2) price forecast: 5-day direction with 95% confidence interval
- ML probability replaces hand-coded TPS — data-learned weights

**Macro Intelligence**
- Buffett Indicator: market cap / GDP — overvaluation gauge *(new)*
- Carry unwind early warning: 7-signal scoring system (0–21)
- USD/JPY 160 framework: risk premium gauge
- Market regime: 5-class classification (Bull/Choppy/Bear/Vol Spike/Recession)

**Trade Output**
- Covered call analysis for all eligible positions (zero capital required)
- Position sizing via ATR-based risk management
- TPS score with explicit edge calculation
- Full trade block matching project instruction format

---

### Backtester (`backtester.py`)

- RSI threshold optimization: tests all combinations of entry/exit RSI
  to find the statistically optimal levels for each ticker
- Covered call backtest: simulates monthly call selling over 1 year,
  shows total income, assignment frequency, combined return
- VectorBT: fully vectorized, runs thousands of combinations fast

---

### Alerts (`alerts.py`)

- Telegram bot sends morning brief at 7:30 AM
- Carry unwind monitor runs every 2 hours — alerts immediately if score > 8
- Only surfaces trades where edge >= 10% or CC yield >= 12% annualized

**Setup Telegram:**
1. Message @BotFather on Telegram → /newbot → copy token
2. Message @userinfobot → copy your chat ID
3. Set env vars: `export TELEGRAM_TOKEN=xxx` and `export TELEGRAM_CHAT_ID=yyy`

---

## API Keys (already configured in signal_engine.py)

| API | Use |
|-----|-----|
| Polygon.io | OHLCV bars, options chain, VWAP |
| Alpha Vantage | RSI, MACD, CCI, ADX, ATR, EMA, BBANDS, STOCH, OVERVIEW |
| Finnhub | Live quotes, earnings calendar, insider sentiment, news |
| FRED | VIX, yield curve, CPI, oil, gold, Buffett Indicator components |
| alternative.me | Crypto Fear & Greed (free, no key) |

---

## Portfolio (configured in signal_engine.py)

| Ticker | Shares | Avg Cost | CC Contracts |
|--------|--------|----------|--------------|
| TSLA | 700 | $204.68 | 7 |
| AMD | 400 | $129.86 | 4 |
| NVDA | 200 | $125.94 | 2 |
| SOFI | 2,000 | $21.09 | 20 |
| AMZN | 200 | $40.44 | 2 |
| HOOD | 100 | $45.00 | 1 |

---

## How Claude Pulls This Each Session

At the start of each conversation, Claude fetches:

```
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/trading_bot/signal_engine.py
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/trading_bot/README.md
```

This keeps the full system logic available in context without manually
pasting thousands of lines every session. Saves ~2,000–4,000 tokens per conversation.

---

## ML Library Stack

| Library | Purpose | Status |
|---------|---------|--------|
| xgboost | Primary ML model — direction prediction | ✅ Core |
| scikit-learn | Cross-validation, scaling, metrics | ✅ Core |
| statsmodels | ARIMA price forecasting | ✅ Core |
| pandas-ta | All technical indicators | ✅ Core |
| vectorbt | Backtesting + optimization | ✅ Included |
| backtrader | Live trading integration | 🔜 Next phase |
| pyfolio-reloaded | Portfolio analytics | 🔜 Next phase |
| tensorflow/keras | LSTM sequence models | 🔜 Phase 3 |
| QuantLib | Precise options pricing | 🔜 Phase 3 |

---

## Roadmap

**Phase 1 (Current) — Signal Engine**
- [x] All 10 indicators
- [x] XGBoost ML predictions
- [x] ARIMA forecasting
- [x] Carry unwind checklist
- [x] Covered call analyzer
- [x] Telegram morning briefs

**Phase 2 — Live Execution**
- [ ] Alpaca API integration (options/stocks)
- [ ] Semi-auto mode (Telegram confirm before execute)
- [ ] Paper trading 30-day validation
- [ ] Trade journal auto-update

**Phase 3 — Advanced ML**
- [ ] LSTM sequence model for crypto/forex
- [ ] QuantLib precise options pricing
- [ ] XGBoost regime classifier (replaces manual regime table)
- [ ] VaR dynamic risk management

**Phase 4 — Multi-Asset**
- [ ] OANDA forex integration (USD/JPY live trades)
- [ ] Binance crypto bot
- [ ] IBKR futures (COT-based crude signals)

---

## Disclaimer

> Research and signal generation only — not financial advice.
> All data from Polygon.io, Finnhub, Alpha Vantage, and FRED.
> Verify all prices and premiums before executing.
> Options involve substantial risk.
> Past performance does not guarantee future results.
