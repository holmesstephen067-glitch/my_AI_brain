# FETCH.md — Trading Dashboard

> Paste these URLs at the start of a trading dashboard session.
> Or just start the MCP server and Claude fetches automatically.

-----

## Manual Fetch URLs (Option B)

```
# Core context (always)
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/main/core/CONTEXT.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/main/core/PORTFOLIO.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/main/core/RULES.md

# Trading engine source
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/main/trading/signal_engine.py

# This project context
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/main/projects/trading_dashboard/README.md
```

-----

## MCP Auto-Fetch (Option A — preferred)

With the brain MCP server running, Claude automatically calls:

- `get_context`
- `get_portfolio`
- `get_cc_rules`
- `get_brain_file?path=trading/signal_engine.py`

No manual fetching needed.

-----

## Session Prompt Template

```
I'm working on my trading dashboard. My portfolio and rules are in PORTFOLIO.md.
Key trading rules: never sell CCs below cost basis, green day + 0.8% rule, 80% profit target.

Today's focus: [describe task]
```