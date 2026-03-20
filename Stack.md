# STACK.md — Preferred Tech Stack

> Authoritative stack decisions. New projects default to these unless there’s a strong reason not to.
> “Strong reason” = documented in the project’s own README, not just preference.

-----

## Backend

|Layer                 |Choice                                               |Why                                                         |
|----------------------|-----------------------------------------------------|------------------------------------------------------------|
|Primary language      |Python 3.12 (data/ML) / TypeScript (web APIs)        |Python owns the ML/trading stack; TS owns the web/SaaS stack|
|Web framework (Python)|FastAPI                                              |Async-native, auto OpenAPI, FastAPI-MCP auto-converts to MCP|
|Web framework (TS)    |Fastify                                              |2-3x faster than Express, schema-first, TypeScript-native   |
|MCP exposure          |fastapi-mcp (Python) / @modelcontextprotocol/sdk (TS)|Every tool gets an MCP interface                            |
|Job queues            |BullMQ (TS) / Celery (Python)                        |Redis-backed, retry, priority lanes, Bull Board UI          |
|ORM                   |Prisma (TS) / SQLAlchemy (Python)                    |Type-safe, migration-aware                                  |

## Frontend

|Layer            |Choice                           |Why                                             |
|-----------------|---------------------------------|------------------------------------------------|
|Framework        |Next.js 15 (App Router)          |Server components, streaming, ecosystem         |
|Styling          |Tailwind CSS + shadcn/ui         |Zero-bloat, fully customizable                  |
|Mobile           |React Native (future) / PWA first|PWA ships faster; RN when native features needed|
|Browser extension|WXT (WebExtension Tools)         |React-based, cross-browser                      |

## Data

|Layer         |Choice                                 |Why                                      |
|--------------|---------------------------------------|-----------------------------------------|
|Primary DB    |PostgreSQL 16 + pgvector               |Relational + vector in one service       |
|Search        |Meilisearch                            |BM25 + vector hybrid, instant search UI  |
|Cache / Queue |Redis 7                                |BullMQ + caching + pub/sub               |
|Object storage|MinIO (self-hosted) / S3 (cloud)       |S3-compatible API, zero migration cost   |
|Time-series   |PostgreSQL (adequate at personal scale)|Avoid TimescaleDB unless proven necessary|

## ML / AI

|Layer          |Choice                                                   |Why                                                 |
|---------------|---------------------------------------------------------|----------------------------------------------------|
|Classic ML     |XGBoost + scikit-learn                                   |Battle-tested, fast, interpretable                  |
|Deep learning  |PyTorch + Lightning                                      |Clean training loop, multi-GPU, DDP/FSDP            |
|LLM integration|Vercel AI SDK (TS) / LangChain (Python)                  |Provider-agnostic, streaming, structured output     |
|Embeddings     |text-embedding-3-small (cloud) / nomic-embed-text (local)|Cost vs. quality tradeoff                           |
|Local models   |Ollama                                                   |Zero-cost inference, privacy-preserving             |
|Cloud GPU      |Modal                                                    |Serverless, L40S best cost/perf, free $30/mo credits|

## Infrastructure

|Layer             |Choice                       |Why                                   |
|------------------|-----------------------------|--------------------------------------|
|Containers        |Docker + Docker Compose      |Single-command deploy, no K8s overhead|
|Reverse proxy     |Traefik                      |Docker-native, auto-SSL, zero config  |
|VPS               |Hetzner CX32 (~$11/mo)       |Best bang/buck in EU/US               |
|Deployment tool   |Sidekick (for VPS)           |Go CLI, zero-downtime, encrypted env  |
|Secrets management|SOPS + age (encrypted `.env`)|Git-safe secret storage               |
|Monitoring        |OpenTelemetry + Prometheus   |Standard, provider-agnostic           |
|CI/CD             |GitHub Actions               |Already in GitHub ecosystem           |

## Trading-Specific

|Layer               |Choice          |Why                                          |
|--------------------|----------------|---------------------------------------------|
|OHLCV data          |Polygon.io      |Best coverage, reliable, options data        |
|Live quotes         |Finnhub         |Fast, free tier adequate                     |
|Macro data          |FRED            |Official source, free, comprehensive         |
|Options chains      |Tradier         |Real options data, reasonable pricing        |
|Technical indicators|pandas-ta       |Comprehensive, fast, pandas-native           |
|Backtesting         |VectorBT        |Vectorized, fast, handles thousands of combos|
|Notifications       |Telegram Bot API|Free, reliable, rich formatting              |

-----

## What We Don’t Use (and Why)

|Rejected                   |Why                                             |Use instead                |
|---------------------------|------------------------------------------------|---------------------------|
|Kubernetes                 |Overkill for personal/small team scale          |Docker Compose             |
|LangChain                  |Heavy abstractions, breaking changes            |Vercel AI SDK or direct API|
|MongoDB                    |Relational data fits SQL; pgvector covers vector|PostgreSQL                 |
|Pinecone/Weaviate          |External dependency, cost                       |pgvector                   |
|Serverless (Vercel/Netlify)|Can’t run workers/queues                        |VPS + Docker               |
|Express.js                 |Outdated, no schema validation                  |Fastify                    |
|Create React App           |Dead project                                    |Next.js or Vite            |
|Jupyter for production     |Not reproducible, hard to version               |Python scripts + Modal     |