"""
=============================================================
TRADING BOT — ALERT SYSTEM v1.0
Sends morning briefs via Telegram + email
=============================================================
Setup:
  1. Create a Telegram bot: message @BotFather on Telegram
     Type /newbot, follow steps, copy the API token
  2. Get your chat ID: message @userinfobot
  3. Add both to config below

Optional email: set EMAIL_* variables below
=============================================================
"""

import os
import json
import schedule
import time
from datetime import datetime

# ── Telegram Config ───────────────────────────────────────
# Fill these in after creating your bot
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ── Alert threshold ───────────────────────────────────────
MIN_EDGE_TO_ALERT = 10   # Only alert if TPS edge >= 10%
MIN_CC_YIELD      = 12   # Only alert covered calls with annualized yield >= 12%
MAX_CARRY_SCORE   = 8    # Alert if carry unwind score exceeds this


def send_telegram(message: str) -> bool:
    """Send message via Telegram bot."""
    try:
        import requests
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def format_morning_brief(scan: dict) -> str:
    """Format scan results as a concise Telegram message."""
    macro  = scan["macro"]
    carry  = scan["carry"]
    regime = scan["regime"]
    results= scan["results"]
    now    = datetime.now().strftime("%b %d, %Y %I:%M %p")

    lines = [
        f"🤖 <b>MORNING BRIEF — {now}</b>",
        f"",
        f"🧭 <b>REGIME: {regime['regime']}</b>",
        f"Strategy: {regime['strategy']}",
        f"VIX: {regime['vix']:.1f} | "
        f"Yield curve: {regime['yield_curve']:.3f}",
        f"Buffett Indicator: {macro.get('buffett_indicator', 0):.0f}%",
        f"",
        f"⚠️ <b>CARRY UNWIND: {carry['score']}/{carry['max_score']}</b>",
        f"{carry['risk_level']}",
        f"",
    ]

    # Top trade opportunities
    top_trades = []
    total_cc_income = 0

    for ticker, data in results.items():
        if "error" in data:
            continue

        tps = data.get("tps", {})
        cc  = data.get("covered_call")
        edge = tps.get("edge", 0)

        # Flag high-conviction trades
        if edge >= MIN_EDGE_TO_ALERT:
            top_trades.append(
                f"🎯 {ticker}: TPS={tps.get('tps', 0):.0f} "
                f"Edge={edge:+.0f}% {tps.get('verdict', '')}"
            )

        # Covered call income
        if cc and cc.get("above_cost_basis"):
            weekly_yield = cc["weekly"].get("ann_yield", 0)
            if weekly_yield >= MIN_CC_YIELD:
                income = cc["weekly"]["total_income"]
                total_cc_income += income
                top_trades.append(
                    f"💰 {ticker} CC: ${cc['weekly']['strike']:.0f} call "
                    f"= ${income:,.0f} ({weekly_yield:.0f}% ann)"
                )

    if top_trades:
        lines.append("📋 <b>TOP TRADES TODAY:</b>")
        lines.extend(top_trades)
    else:
        lines.append("📋 No high-conviction setups today — hold and wait.")

    if total_cc_income > 0:
        lines.append(f"")
        lines.append(f"💵 <b>Total CC income available: ${total_cc_income:,.0f}</b>")

    # Carry unwind alert
    if carry["score"] >= MAX_CARRY_SCORE:
        lines.append(f"")
        lines.append(
            f"🚨 <b>CARRY UNWIND ALERT</b>\n"
            f"Score {carry['score']}/21 — {carry['action']}"
        )

    lines.append(f"")
    lines.append(f"⚠️ Research only — not financial advice.")

    return "\n".join(lines)


def format_trade_alert(ticker: str, data: dict) -> str:
    """Format a single trade alert for Telegram."""
    tps    = data.get("tps", {})
    arima  = data.get("arima", {})
    cc     = data.get("covered_call")
    sizing = data.get("sizing", {})
    price  = data.get("price", 0)

    lines = [
        f"🎯 <b>TRADE ALERT: {ticker}</b>",
        f"Price: ${price:.2f}",
        f"",
        f"ML Probability: {data.get('ml_prob', 0.5)*100:.1f}%",
        f"TPS Score: {tps.get('tps', 0):.1f}",
        f"Edge: {tps.get('edge', 0):+.1f}%",
        f"Verdict: {tps.get('verdict', '')}",
    ]

    if arima.get("forecast_5d"):
        lines.extend([
            f"",
            f"📈 ARIMA 5d: ${arima['forecast_5d']:.2f} "
            f"({arima['pct_change']:+.1f}%) {arima['direction']}",
        ])

    if cc and cc.get("above_cost_basis"):
        lines.extend([
            f"",
            f"💰 COVERED CALL:",
            f"  Strike: ${cc['weekly']['strike']:.0f}",
            f"  Premium: ${cc['weekly']['premium']:.2f}/share",
            f"  Income: ${cc['weekly']['total_income']:,.0f} "
            f"({cc['contracts']} contracts)",
            f"  Ann yield: {cc['weekly']['ann_yield']:.0f}%",
            f"  Capital required: $0",
        ])
    elif tps.get("edge", 0) >= 10:
        lines.extend([
            f"",
            f"📐 SIZING:",
            f"  Shares: {sizing.get('shares', 0)}",
            f"  Value: ${sizing.get('position_value', 0):,.0f}",
            f"  Stop: ${sizing.get('stop_price', 0):.2f}",
            f"  Risk: ${sizing.get('risk_dollars', 0):,.0f}",
        ])

    lines.append(f"")
    lines.append(f"⚠️ Research only — verify before executing.")

    return "\n".join(lines)


def morning_brief_job():
    """Scheduled job: run full scan and send Telegram brief."""
    print(f"\n⏰ Running morning brief at {datetime.now()}")
    try:
        from signal_engine import run_full_scan
        scan = run_full_scan()

        # Send main brief
        message = format_morning_brief(scan)
        success = send_telegram(message)

        if success:
            print("✅ Morning brief sent via Telegram")
        else:
            print("❌ Telegram send failed — check token and chat ID")
            print("\n--- BRIEF CONTENT ---")
            print(message)

        # Send individual high-conviction alerts
        for ticker, data in scan["results"].items():
            if "error" in data:
                continue
            tps  = data.get("tps", {})
            edge = tps.get("edge", 0)
            if edge >= 15:  # Only send individual alerts for high conviction
                alert = format_trade_alert(ticker, data)
                send_telegram(alert)
                time.sleep(1)

    except Exception as e:
        error_msg = f"🔴 Bot error at {datetime.now()}: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)


def carry_unwind_monitor():
    """
    Run every 2 hours during market hours.
    Alert immediately if carry unwind score spikes.
    """
    try:
        from signal_engine import fetch_macro_snapshot, carry_unwind_score
        macro = fetch_macro_snapshot()
        carry = carry_unwind_score(macro)

        if carry["score"] >= MAX_CARRY_SCORE:
            message = (
                f"🚨 <b>CARRY UNWIND ALERT — {datetime.now().strftime('%I:%M %p')}</b>\n"
                f"Score: {carry['score']}/{carry['max_score']}\n"
                f"{carry['risk_level']}\n"
                f"Action: {carry['action']}\n"
                f"\nVIX: {macro.get('vix', 'N/A')}\n"
                f"Oil: ${macro.get('oil_wti', 'N/A')}\n"
                f"Yield curve: {macro.get('yield_curve', 'N/A')}"
            )
            send_telegram(message)
            print(f"🚨 Carry unwind alert sent — score {carry['score']}")

    except Exception as e:
        print(f"❌ Carry monitor error: {e}")


def start_scheduler():
    """
    Start the scheduled bot.
    Runs morning brief at 7:30 AM and carry monitor every 2 hours.
    """
    print("🤖 Trading Bot Scheduler Started")
    print("="*40)
    print("Schedule:")
    print("  • 7:30 AM daily   — Morning brief")
    print("  • Every 2 hours   — Carry unwind monitor")
    print("  • Ctrl+C to stop")
    print("="*40)

    # Schedule jobs
    schedule.every().day.at("07:30").do(morning_brief_job)
    schedule.every(2).hours.do(carry_unwind_monitor)

    # Run immediately on start
    morning_brief_job()

    while True:
        schedule.run_pending()
        time.sleep(60)


def test_telegram():
    """Test Telegram connection."""
    success = send_telegram(
        "🤖 <b>Trading Bot Connected</b>\n"
        "Telegram alerts are working correctly.\n"
        "Morning briefs will arrive at 7:30 AM."
    )
    if success:
        print("✅ Telegram test successful — check your messages")
    else:
        print("❌ Telegram test failed")
        print("Make sure TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set correctly")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_telegram()
    elif len(sys.argv) > 1 and sys.argv[1] == "brief":
        morning_brief_job()
    else:
        start_scheduler()
