"""
=============================================================
TRADING BOT — VECTORBT BACKTESTER v1.0
Optimizes indicator parameters using historical data
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("⚠️  vectorbt not installed — run: pip install vectorbt")

from signal_engine import fetch_polygon_bars, calculate_indicators


def backtest_rsi_strategy(ticker: str,
                           rsi_low:  float = 30,
                           rsi_high: float = 70,
                           days: int = 730) -> dict:
    """
    Vectorized backtest of RSI strategy.
    Tests whether your RSI thresholds are actually optimal.
    """
    if not VBT_AVAILABLE:
        return {"error": "vectorbt not installed"}

    df = fetch_polygon_bars(ticker, days=days)
    if df.empty:
        return {"error": f"No data for {ticker}"}

    df = calculate_indicators(df)
    close = df["close"]
    rsi   = df["rsi_14"]

    # Entry: RSI crosses above rsi_low from below (oversold bounce)
    entries = (rsi > rsi_low) & (rsi.shift(1) <= rsi_low)
    # Exit:  RSI crosses below rsi_high from above (overbought)
    exits   = (rsi < rsi_high) & (rsi.shift(1) >= rsi_high)

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=10000,
        fees=0.001,       # 0.1% commission
        slippage=0.001,   # 0.1% slippage
    )

    stats = portfolio.stats()

    return {
        "ticker":         ticker,
        "total_return":   round(stats.get("Total Return [%]", 0), 2),
        "sharpe":         round(stats.get("Sharpe Ratio", 0), 3),
        "max_drawdown":   round(stats.get("Max Drawdown [%]", 0), 2),
        "win_rate":       round(stats.get("Win Rate [%]", 0), 2),
        "total_trades":   stats.get("Total Trades", 0),
        "avg_return":     round(stats.get("Avg Winning Trade [%]", 0), 2),
        "expectancy":     round(stats.get("Expectancy", 0), 4),
        "rsi_low":        rsi_low,
        "rsi_high":       rsi_high,
    }


def optimize_rsi_thresholds(ticker: str,
                              days: int = 730) -> pd.DataFrame:
    """
    Test all RSI threshold combinations to find optimal levels.
    Answers: "Is 30/70 actually the best threshold for THIS ticker?"
    """
    if not VBT_AVAILABLE:
        print("⚠️  vectorbt not installed")
        return pd.DataFrame()

    df = fetch_polygon_bars(ticker, days=days)
    if df.empty:
        return pd.DataFrame()

    df   = calculate_indicators(df)
    close = df["close"]
    rsi   = df["rsi_14"]

    # Test entry thresholds 20–45, exit thresholds 55–80
    entry_thresholds = np.arange(20, 45, 5)
    exit_thresholds  = np.arange(55, 85, 5)

    results = []
    for entry in entry_thresholds:
        for exit_ in exit_thresholds:
            entries = (rsi > entry) & (rsi.shift(1) <= entry)
            exits   = (rsi < exit_)  & (rsi.shift(1) >= exit_)

            try:
                pf = vbt.Portfolio.from_signals(
                    close, entries, exits,
                    init_cash=10000, fees=0.001, slippage=0.001
                )
                stats = pf.stats()
                results.append({
                    "entry_rsi":    entry,
                    "exit_rsi":     exit_,
                    "total_return": round(stats.get("Total Return [%]", 0), 2),
                    "sharpe":       round(stats.get("Sharpe Ratio", 0), 3),
                    "win_rate":     round(stats.get("Win Rate [%]", 0), 2),
                    "max_drawdown": round(stats.get("Max Drawdown [%]", 0), 2),
                    "trades":       stats.get("Total Trades", 0),
                })
            except:
                pass

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values("sharpe", ascending=False)
        print(f"\n📊 RSI OPTIMIZATION — {ticker}")
        print(f"{'─'*60}")
        print(f"Best by Sharpe ratio (top 5):")
        print(df_results.head(5).to_string(index=False))
        best = df_results.iloc[0]
        print(f"\n✅ OPTIMAL for {ticker}: "
              f"Entry RSI={best['entry_rsi']}, "
              f"Exit RSI={best['exit_rsi']}, "
              f"Sharpe={best['sharpe']:.3f}, "
              f"Win rate={best['win_rate']:.1f}%")

    return df_results


def backtest_covered_call(ticker: str,
                           avg_cost: float,
                           contracts: int,
                           days: int = 365) -> dict:
    """
    Backtest covered call income strategy.
    Simulates selling 5% OTM calls every month and rolling.
    """
    df = fetch_polygon_bars(ticker, days=days)
    if df.empty:
        return {"error": f"No data for {ticker}"}

    df = calculate_indicators(df)

    # Monthly resampling — sell calls at start of each month
    monthly = df["close"].resample("ME").last()

    # Premium estimate: ATR-based
    monthly_atr = df["atr_14"].resample("ME").last()

    # IV proxy from VIX (simplified)
    iv_proxy = 0.35  # ~35% IV assumption for elevated VIX environment

    premiums = []
    for i in range(len(monthly) - 1):
        price  = monthly.iloc[i]
        atr    = monthly_atr.iloc[i] if not pd.isna(monthly_atr.iloc[i]) else price * 0.03
        strike = price * 1.05

        # Black-Scholes simplified premium estimate
        # P ≈ ATR × IV_factor for 30-day options
        premium = atr * iv_proxy * 0.8
        total   = premium * contracts * 100
        premiums.append({
            "month":          monthly.index[i].strftime("%Y-%m"),
            "stock_price":    round(price, 2),
            "strike":         round(strike, 2),
            "premium":        round(premium, 2),
            "total_income":   round(total, 2),
            "called_away":    monthly.iloc[i+1] > strike,
        })

    df_premiums = pd.DataFrame(premiums)
    if df_premiums.empty:
        return {"error": "No monthly data"}

    total_income    = df_premiums["total_income"].sum()
    months_traded   = len(df_premiums)
    called_away_pct = df_premiums["called_away"].mean() * 100
    avg_monthly     = total_income / months_traded if months_traded > 0 else 0
    stock_return    = ((monthly.iloc[-1] - avg_cost) / avg_cost) * 100

    print(f"\n💰 COVERED CALL BACKTEST — {ticker}")
    print(f"{'─'*50}")
    print(f"Period:            {months_traded} months")
    print(f"Total CC income:   ${total_income:,.0f}")
    print(f"Avg monthly:       ${avg_monthly:,.0f}")
    print(f"Called away:       {called_away_pct:.1f}% of months")
    print(f"Stock return:      {stock_return:+.1f}% vs avg cost ${avg_cost}")
    print(f"Combined return:   ${(total_income + (monthly.iloc[-1] - avg_cost) * contracts * 100):,.0f}")

    return {
        "ticker":           ticker,
        "total_cc_income":  round(total_income, 0),
        "avg_monthly":      round(avg_monthly, 0),
        "months_traded":    months_traded,
        "called_away_pct":  round(called_away_pct, 1),
        "stock_return_pct": round(stock_return, 2),
        "monthly_detail":   df_premiums.to_dict("records"),
    }


def run_strategy_validation(portfolio: dict = None) -> None:
    """
    Run full strategy validation across all portfolio tickers.
    Optimizes RSI thresholds and validates covered call history.
    """
    from signal_engine import PORTFOLIO
    if portfolio is None:
        portfolio = PORTFOLIO

    print("\n" + "="*60)
    print("🔬 STRATEGY VALIDATION — VECTORBT")
    print("="*60)

    for ticker, info in portfolio.items():
        print(f"\n{'='*40}")
        print(f"🔍 Validating {ticker}")
        print(f"{'='*40}")

        # RSI optimization
        optimize_rsi_thresholds(ticker, days=730)

        # Covered call backtest
        backtest_covered_call(
            ticker,
            avg_cost  = info["avg_cost"],
            contracts = info["contracts"],
            days      = 365
        )


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"

    print(f"\n🔬 Running backtest optimization for {ticker}...")
    optimize_rsi_thresholds(ticker, days=730)

    from signal_engine import PORTFOLIO
    if ticker in PORTFOLIO:
        backtest_covered_call(
            ticker,
            PORTFOLIO[ticker]["avg_cost"],
            PORTFOLIO[ticker]["contracts"],
            days=365
        )
