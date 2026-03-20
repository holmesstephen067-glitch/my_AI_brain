# “””
brain_mcp_server.py — Master MCP Server

Exposes the full AI Brain as MCP tools consumable by Claude Desktop,
Cursor, or any MCP-compatible client.

Pattern: fastapi-mcp (from dev.md) auto-converts FastAPI routes → MCP tools.

Usage:
pip install fastapi fastapi-mcp uvicorn python-dotenv
python brain_mcp_server.py

Claude Desktop config:
{
“mcpServers”: {
“brain”: {
“url”: “http://localhost:8000/mcp”
}
}
}
“””

import os
import json
import logging
import requests
from typing import Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(**name**)
logging.basicConfig(level=logging.INFO, format=”%(asctime)s [%(levelname)s] %(message)s”)

# ── Constants ────────────────────────────────────────────────────

BRAIN_ROOT = Path(os.getenv(“BRAIN_ROOT”, Path(**file**).parent.parent))
PORTFOLIO_PATH = BRAIN_ROOT / “core” / “PORTFOLIO.md”
CONTEXT_PATH   = BRAIN_ROOT / “core” / “CONTEXT.md”
RULES_PATH     = BRAIN_ROOT / “core” / “RULES.md”
STACK_PATH     = BRAIN_ROOT / “core” / “STACK.md”

# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(
title=“AI Brain MCP Server”,
description=“Exposes Stephen’s AI Brain as MCP tools for Claude”,
version=“2.0.0”
)

# ─────────────────────────────────────────────────────────────────

# CONTEXT TOOLS — fetch brain files as AI context

# ─────────────────────────────────────────────────────────────────

@app.get(
“/brain/context”,
operation_id=“get_context”,
summary=“Get master project context (who I am, what I’m building)”,
tags=[“context”]
)
def get_context():
“”“Returns CONTEXT.md — identity, active projects, session protocol.”””
try:
return {“content”: CONTEXT_PATH.read_text(), “file”: “core/CONTEXT.md”}
except FileNotFoundError:
raise HTTPException(404, “CONTEXT.md not found”)

@app.get(
“/brain/portfolio”,
operation_id=“get_portfolio”,
summary=“Get current portfolio positions, cost basis, and trading rules”,
tags=[“context”]
)
def get_portfolio():
“”“Returns PORTFOLIO.md — all positions, CC rules, sizing formula.”””
try:
return {“content”: PORTFOLIO_PATH.read_text(), “file”: “core/PORTFOLIO.md”}
except FileNotFoundError:
raise HTTPException(404, “PORTFOLIO.md not found”)

@app.get(
“/brain/rules”,
operation_id=“get_rules”,
summary=“Get non-negotiable rules across all projects”,
tags=[“context”]
)
def get_rules():
“”“Returns RULES.md — security, code quality, trading, architecture rules.”””
try:
return {“content”: RULES_PATH.read_text(), “file”: “core/RULES.md”}
except FileNotFoundError:
raise HTTPException(404, “RULES.md not found”)

@app.get(
“/brain/stack”,
operation_id=“get_stack”,
summary=“Get preferred tech stack decisions”,
tags=[“context”]
)
def get_stack():
“”“Returns STACK.md — all technology choices with rationale.”””
try:
return {“content”: STACK_PATH.read_text(), “file”: “core/STACK.md”}
except FileNotFoundError:
raise HTTPException(404, “STACK.md not found”)

@app.get(
“/brain/file”,
operation_id=“get_brain_file”,
summary=“Get any file from the brain repo by relative path”,
tags=[“context”]
)
def get_brain_file(path: str = Query(…, description=“Relative path from repo root e.g. ‘trading/signal_engine.py’”)):
“”“Fetch any file from the brain repo. Use for source code inspection.”””
target = BRAIN_ROOT / path
if not target.exists():
raise HTTPException(404, f”File not found: {path}”)
# Security: only allow files within brain root
try:
target.resolve().relative_to(BRAIN_ROOT.resolve())
except ValueError:
raise HTTPException(403, “Path traversal not allowed”)
return {“content”: target.read_text(), “file”: path, “size_bytes”: target.stat().st_size}

# ─────────────────────────────────────────────────────────────────

# TRADING TOOLS — live market data and signals

# ─────────────────────────────────────────────────────────────────

@app.get(
“/trading/scan”,
operation_id=“run_trading_scan”,
summary=“Run full signal scan on portfolio tickers”,
tags=[“trading”]
)
def run_trading_scan(
tickers: str = Query(None, description=“Comma-separated tickers e.g. ‘SOFI,NVDA’. Default: full portfolio”),
use_tradier: bool = Query(True, description=“Use Tradier for real options chain”)
):
“””
Run the full XGBoost + ARIMA + TPS signal scan.
Returns signals, covered call candidates, and position sizing for each ticker.
Takes 2-3 minutes for full portfolio scan.
“””
try:
# Import here to avoid circular deps
from trading.signal_engine import run_full_scan
ticker_list = tickers.split(”,”) if tickers else None
result = run_full_scan(ticker_list, use_tradier=use_tradier)
return result
except Exception as e:
raise HTTPException(500, f”Scan error: {e}”)

@app.get(
“/trading/covered_calls”,
operation_id=“get_covered_call_candidates”,
summary=“Get covered call candidates for a ticker using real options chain”,
tags=[“trading”]
)
def get_covered_call_candidates(
ticker: str = Query(…, description=“Ticker symbol e.g. ‘SOFI’”),
current_price: float = Query(…, description=“Current stock price”),
pct_change: float = Query(…, description=“Today’s % change as decimal e.g. 0.012 for +1.2%”)
):
“””
Find optimal covered call strikes using Tradier real options chain.
Enforces: green day rule, cost basis rule, min yield threshold.
Falls back to ATR estimates if Tradier unavailable.
“””
try:
from trading.signal_engine import PORTFOLIO, find_covered_call_candidates, cc_atr_fallback, fetch_polygon_bars, calculate_indicators
if ticker not in PORTFOLIO:
raise HTTPException(400, f”{ticker} not in portfolio. Portfolio tickers: {list(PORTFOLIO.keys())}”)
port_info = PORTFOLIO[ticker]
result = find_covered_call_candidates(
ticker=ticker,
current_price=current_price,
avg_cost=port_info[“avg_cost”],
contracts=port_info[“contracts”],
current_pct_change=pct_change,
)
return result
except Exception as e:
raise HTTPException(500, f”CC analysis error: {e}”)

@app.get(
“/trading/portfolio_state”,
operation_id=“get_portfolio_state”,
summary=“Get current portfolio positions from PORTFOLIO.md”,
tags=[“trading”]
)
def get_portfolio_state():
“”“Returns the configured portfolio positions with avg costs and contracts.”””
from trading.signal_engine import PORTFOLIO, PORTFOLIO_VALUE, BUYING_POWER
return {
“positions”: PORTFOLIO,
“portfolio_value”: PORTFOLIO_VALUE,
“buying_power”: BUYING_POWER,
“source”: “core/PORTFOLIO.md + env”
}

@app.get(
“/trading/cc_rules”,
operation_id=“get_cc_rules”,
summary=“Get Stephen’s covered call rules”,
tags=[“trading”]
)
def get_cc_rules():
“”“Returns the non-negotiable covered call rules enforced by the signal engine.”””
from trading.signal_engine import CC_MIN_GREEN_PCT, CC_PROFIT_TARGET_PCT, ATR_STOP_MULTIPLIER
return {
“rules”: [
f”1. Green day only — stock must be up ≥ {CC_MIN_GREEN_PCT*100:.1f}% on the day”,
“2. Strike must be ABOVE average cost basis — always”,
f”3. Buy back at {CC_PROFIT_TARGET_PCT*100:.0f}% profit (20% of premium remaining)”,
“4. Prefer 5% OTM weekly, 10% OTM monthly”,
“5. Check VIX before selling — prefer VIX > 18”,
f”6. Stop loss: {ATR_STOP_MULTIPLIER}× ATR below entry”,
],
“source”: “core/RULES.md + core/PORTFOLIO.md”
}

# ─────────────────────────────────────────────────────────────────

# PROJECT TOOLS — architecture docs and project scaffolds

# ─────────────────────────────────────────────────────────────────

@app.get(
“/projects/list”,
operation_id=“list_projects”,
summary=“List all projects and their status”,
tags=[“projects”]
)
def list_projects():
“”“Returns all projects in the brain repo with their status and key files.”””
projects_dir = BRAIN_ROOT / “projects”
projects = []
if projects_dir.exists():
for p in projects_dir.iterdir():
if p.is_dir():
readme = p / “README.md”
fetch  = p / “FETCH.md”
projects.append({
“name”:         p.name,
“has_readme”:   readme.exists(),
“has_fetch”:    fetch.exists(),
“files”:        [f.name for f in p.iterdir() if f.is_file()],
})
return {“projects”: projects, “count”: len(projects)}

@app.get(
“/projects/{project_name}”,
operation_id=“get_project”,
summary=“Get project README and fetch manifest”,
tags=[“projects”]
)
def get_project(project_name: str):
“”“Get all files for a specific project.”””
project_dir = BRAIN_ROOT / “projects” / project_name
if not project_dir.exists():
raise HTTPException(404, f”Project ‘{project_name}’ not found”)
result = {“name”: project_name, “files”: {}}
for f in project_dir.iterdir():
if f.is_file() and f.suffix in [”.md”, “.py”, “.ts”, “.json”, “.yml”, “.yaml”]:
result[“files”][f.name] = f.read_text()
return result

# ─────────────────────────────────────────────────────────────────

# AGENT TOOLS — skill bundles from agents.md

# ─────────────────────────────────────────────────────────────────

@app.get(
“/agents/skills”,
operation_id=“get_agent_skills”,
summary=“Get full agent skill bundle (MCP builder, agentic loop, n8n, swarm)”,
tags=[“agents”]
)
def get_agent_skills():
“”“Returns agents.md — all agentic patterns and skill bundles.”””
agents_path = BRAIN_ROOT / “agents” / “agents.md”
if not agents_path.exists():
agents_path = BRAIN_ROOT / “agents.md”
try:
return {“content”: agents_path.read_text(), “file”: “agents/agents.md”}
except FileNotFoundError:
raise HTTPException(404, “agents.md not found”)

@app.get(
“/agents/dev”,
operation_id=“get_dev_bundle”,
summary=“Get dev skill bundle (Modal, Lightning, Transformers, FastAPI-MCP)”,
tags=[“agents”]
)
def get_dev_bundle():
“”“Returns dev.md — all development patterns and library references.”””
dev_path = BRAIN_ROOT / “dev.md”
try:
return {“content”: dev_path.read_text(), “file”: “dev.md”}
except FileNotFoundError:
raise HTTPException(404, “dev.md not found”)

# ─────────────────────────────────────────────────────────────────

# UTILITY TOOLS

# ─────────────────────────────────────────────────────────────────

@app.get(
“/brain/health”,
operation_id=“health_check”,
summary=“Health check — confirms brain server is running”,
tags=[“utility”]
)
def health_check():
return {
“status”: “healthy”,
“brain_root”: str(BRAIN_ROOT),
“timestamp”: datetime.now().isoformat(),
“tools_available”: [
“get_context”, “get_portfolio”, “get_rules”, “get_stack”,
“get_brain_file”, “run_trading_scan”, “get_covered_call_candidates”,
“get_portfolio_state”, “get_cc_rules”, “list_projects”,
“get_project”, “get_agent_skills”, “get_dev_bundle”
]
}

# ── Mount MCP ────────────────────────────────────────────────────

mcp = FastApiMCP(app)
mcp.mount_http()  # Mounts at /mcp

if **name** == “**main**”:
import uvicorn
port = int(os.getenv(“MCP_PORT”, “8000”))
log.info(f”🧠 Brain MCP Server starting on port {port}”)
log.info(f”MCP endpoint: http://localhost:{port}/mcp”)
log.info(f”API docs:     http://localhost:{port}/docs”)
uvicorn.run(app, host=“0.0.0.0”, port=port, reload=False)