OPSAI SESSION — LOAD CONTEXT BEFORE RESPONDING

════════════════════════════════
FETCH THESE FILES FIRST
════════════════════════════════
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/agents.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/dev.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/finance.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/data.md
https://raw.githubusercontent.com/holmesstephen067-glitch/my_AI_brain/refs/heads/main/alerts.py

════════════════════════════════
PROJECT
════════════════════════════════
OpsAI — AI-powered business ops SaaS for SMB (contractors, restaurants, retail, salons, cleaning, landscaping)
Industries onboarded via 3-question adaptive flow → personalized dashboard

STACK
Frontend: React + Tailwind, src/modules/{name}/ per module
Backend: AWS Lambda Node.js, src/lambdas/{name}/ per function
DB: PostgreSQL RDS + pgvector extension (no separate vector DB)
Cache: ElastiCache Redis
Storage: S3 (docs, PDFs, receipts, signed documents)
Auth: Cognito Gen2 + Keycloak, Amplify v6 ONLY (v4 EOL April 2026)
Payments: Stripe Node SDK v18, apiVersion 2025-03-31.basil
Banking: Plaid Node SDK
Integrations: Merge.dev (Phase 2)
Email: AWS SES + SendGrid
SMS: Twilio
Gateway: Kong open source
Analytics: Recharts (all tiers) + Apache Superset (Growth+)
Docs/PDF: WeasyPrint
E-signatures: Documenso self-hosted (AGPLv3, free API + embed)
Project mgmt: Plane self-hosted (Phase 3)
AI Orchestration: LangGraph two-graph pattern (Ingestion + Retrieval)
ML: LangChain + LSTM via Modal GPU
Security: Wazuh + OWASP ZAP + Gitleaks
CI/CD: GitHub Actions

SECURITY RULES
- AES-256 at rest, TLS 1.3 in transit, AWS KMS for key management
- MFA REQUIRED on all accounts (TOTP + SMS via Cognito)
- RBAC groups: owner · employee · accountant · admin
- Enforced server-side from JWT cognito:groups — never trust client claims
- PCI-DSS via Stripe — card data never touches OpsAI DB
- Full audit logs on all data access
- Zero Trust — every request authenticated
- pgvector: always filter similarity search by tenantId

PRICING TIERS
Starter $49 · Growth $149 · Business $399 · Enterprise custom

════════════════════════════════
AUTH (Amplify Gen2 v6)
════════════════════════════════
npm install aws-amplify @aws-amplify/ui-react
Existing Cognito: use referenceAuth({ userPoolId, userPoolClientId, identityPoolId, authRoleArn, unauthRoleArn })
signIn always returns nextStep — handle: CONFIRM_SIGN_IN_WITH_SMS_CODE · CONFIRM_SIGN_IN_WITH_TOTP_CODE · CONTINUE_SIGN_IN_WITH_MFA_SELECTION · CONFIRM_SIGN_IN_WITH_NEW_PASSWORD_REQUIRED · DONE
MFA setup: setUpTOTP() → verifyTOTPSetup({ code }) → updateMFAPreference({ totp: 'PREFERRED', sms: 'ENABLED' })
Session: fetchAuthSession() → tokens.accessToken.payload['cognito:groups']
Custom attributes: custom:tenantId · custom:plan
Logout: signOut({ global: true })

════════════════════════════════
STRIPE
════════════════════════════════
new Stripe(STRIPE_SECRET_KEY, { apiVersion: '2025-03-31.basil' })
Webhook: constructEvent(rawBody, sig, STRIPE_WEBHOOK_SECRET) — raw body always
Handle: checkout.session.completed · invoice.payment_succeeded · invoice.payment_failed · customer.subscription.updated · customer.subscription.deleted
Invoice payment link: stripe.paymentLinks.create({ line_items: [{ price, quantity: 1 }] })

════════════════════════════════
PLAID
════════════════════════════════
Flow: linkTokenCreate → Link UI → itemPublicTokenExchange → AES-256+KMS encrypt → store in PostgreSQL
Never expose access_token to frontend
Sandbox: user_good / pass_good / 1234 · Transactions test: user_transactions_dynamic
Products: transactions · balance
Claude gets compact summary only — never raw Plaid JSON

════════════════════════════════
DOCUMENSO (replaces DocuSeal)
════════════════════════════════
Self-hosted Docker, PostgreSQL backend, free API + React embed
npm install @documenso/embed-react
Embed: <EmbedSignDocument token={signingToken} />
Create document: POST /api/v1/documents with PDF + recipients array
Send for signing: POST /api/v1/documents/{id}/send
Webhooks: document.completed → download signed PDF → store in S3 → link to record in PostgreSQL
DOCUMENSO_API_KEY · DOCUMENSO_BASE_URL

════════════════════════════════
LANGGRAPH RAG
════════════════════════════════
Two graphs, Claude as LLM (not OpenAI), pgvector on RDS (not Supabase)
Ingestion Graph: S3 file → parse → chunk (1000/200) → embed → pgvector INSERT with tenantId metadata
Retrieval Graph: query → decide retrieve vs direct → similarity search WHERE tenantId = $1 → Claude generates
LangGraph server port 2024
OpsAI use cases: receipt OCR · contract extraction · executive summary · tax readiness
LANGCHAIN_API_KEY · LANGCHAIN_TRACING_V2=true · LANGCHAIN_PROJECT · LANGGRAPH_INGESTION_ASSISTANT_ID · LANGGRAPH_RETRIEVAL_ASSISTANT_ID

════════════════════════════════
SUPERSET
════════════════════════════════
Apache 2.0 — free for SaaS, no license obligation
Connects to PostgreSQL RDS via SQLAlchemy
Recharts for all tiers (in-app, pulls from Lambda API)
Superset for Growth+ only (advanced analytics, SQL explorer)
Embed: POST /api/v1/security/guest_token with RLS clause tenant_id = '{tenantId}' → iframe embed
Always attach tenant RLS to every guest token

════════════════════════════════
ENV VARS (exact names, all files)
════════════════════════════════
ANTHROPIC_API_KEY
STRIPE_SECRET_KEY · STRIPE_PUBLISHABLE_KEY · STRIPE_WEBHOOK_SECRET
PLAID_CLIENT_ID · PLAID_SECRET · PLAID_ENV
COGNITO_USER_POOL_ID · COGNITO_CLIENT_ID · COGNITO_IDENTITY_POOL_ID · COGNITO_AUTH_ROLE_ARN · COGNITO_UNAUTH_ROLE_ARN
DOCUMENSO_API_KEY · DOCUMENSO_BASE_URL
LANGCHAIN_API_KEY · LANGCHAIN_TRACING_V2 · LANGCHAIN_PROJECT · LANGGRAPH_INGESTION_ASSISTANT_ID · LANGGRAPH_RETRIEVAL_ASSISTANT_ID
DB_CONNECTION_STRING · REDIS_URL · AWS_KMS_KEY_ID · AWS_REGION · S3_BUCKET_NAME · SES_FROM_EMAIL

════════════════════════════════
CODING RULES
════════════════════════════════
- Production code only. No TODO stubs in logic paths.
- Every module: src/modules/{name}/index.jsx
- Every Lambda: src/lambdas/{name}/index.js
- Mark every third-party touchpoint: // INTEGRATION HOOK: {service}
- Claude: compact summaries in only, never raw DB rows or full API responses
- Stripe: raw body webhook, never store card data
- Plaid: KMS+AES-256 encrypt access_token before any DB write
- Amplify v6 only, always handle all nextStep cases
- Documenso: webhook → S3 → PostgreSQL link
- pgvector: tenantId filter on every similarity search
- Superset: tenantId RLS on every guest token
- Modal for GPU/ML (LSTM forecasting, batch embeddings)
- fastapi-mcp for any FastAPI→MCP exposure
- Do NOT summarize what you just built after completing a step — just output the code

════════════════════════════════
BUILD STATE
════════════════════════════════
MVP
✅ Step 1  — React shell (AppShell, Sidebar, TopNav, Dashboard, KPIs, AI insights, sparkline, tax gauge, integration badges, onboarding banner)
✅ Step 2  — Adaptive onboarding (3 questions → personalized dashboard)
✅ Step 3  — Cognito Gen2 auth (signup, login, MFA, RBAC)
⬜ Step 4  — PostgreSQL schema + pgvector
⬜ Step 5  — Invoicing module (create, send, track, Documenso e-sign)
⬜ Step 6  — Stripe payments (invoice links, subscriptions, webhooks)
⬜ Step 7  — AI Receipt Scanner (Claude Vision + LangGraph Ingestion)
⬜ Step 8  — Bookkeeping lite + Plaid bank sync
⬜ Step 9  — Tax Readiness Meter
⬜ Step 10 — Executive Summary PDF (WeasyPrint)

PHASE 2
⬜ Merge.dev integrations (QuickBooks, ADP, Outlook, Gmail, Google Calendar)
⬜ Accountant API (OAuth 2.0, read-only, rate-limited, audit logged)
⬜ Superset embedded analytics (Growth tier)
⬜ CRM module
⬜ PowerPoint generation (Presenton + Claude)
⬜ Pitch deck generator (Claude 8-slide structure)

PHASE 3
⬜ HR module (onboarding, PTO, org chart, performance reviews, compliance alerts)
⬜ Plane project management (Kanban, job costing, time tracking, client portal)
⬜ Mobile field app (React Native — clock in/out, receipt snap, job status)
⬜ Multi-location + multi-entity with master dashboard rollup
⬜ KPI dashboards + OKR tracking
⬜ Vendor + contract management (Claude extracts key dates, auto-reminders)
⬜ Client health scoring (payment patterns, communication, job history)
⬜ ESG + compliance reporting
⬜ Employee training + certification tracker
⬜ White-label + SOC 2 + SLA (Enterprise)
