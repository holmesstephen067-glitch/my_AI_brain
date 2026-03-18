# Agents Bundle
> Skills: mcp-apps-builder · Agentic Loop · n8n Validator · Swarm Orchestration
> Usage: Paste this URL at session start for agentic/automation tasks.

---

## 1. mcp-apps-builder (mcp-use framework)
**Trigger:** "build MCP server", "MCP tool/resource/prompt", "MCP widget", "deploy MCP"
**Requires:** `npx create-mcp-use-app my-server` · TypeScript

**Detection first:**
```bash
# Existing project? Check for mcp-use in package.json or imports from "mcp-use/server"
# YES → skip scaffold, add features directly
# NO  → npx create-mcp-use-app my-server && cd my-server && npm run dev
```

**Core server pattern:**
```typescript
import { MCPServer, text, object, markdown, widget, error } from "mcp-use/server";
import { z } from "zod";

const server = new MCPServer({ name: "my-server", title: "My Server", version: "1.0.0" });

// Tool (action the AI calls)
server.tool(
  { name: "fetch-data", description: "Fetch data by ID",
    schema: z.object({ id: z.string().describe("Record ID to fetch") }) },
  async ({ id }) => object({ id, data: "..." })
);

// Resource (read-only data)
server.resource(
  { uri: "config://settings", name: "Settings", mimeType: "application/json" },
  async () => object({ theme: "dark", lang: "en" })
);

// Prompt template
server.prompt(
  { name: "summarize", description: "Summarize text",
    schema: z.object({ text: z.string().describe("Text to summarize") }) },
  async ({ text }) => text(`Summarize this concisely: ${text}`)
);

// Widget (React UI)
server.tool(
  { name: "show-dashboard", schema: z.object({ userId: z.string().describe("User ID") }),
    widget: { name: "dashboard" } },
  async ({ userId }) => widget({ props: { userId }, output: text("Dashboard loaded") })
);

server.listen();
```

**Widget file (`resources/dashboard.tsx`):**
```typescript
import { McpUseProvider, useWidget, type WidgetMetadata } from "mcp-use/react";
const propsSchema = z.object({ userId: z.string() });
type Props = z.infer<typeof propsSchema>;
export const widgetMetadata: WidgetMetadata = { description: "User dashboard", props: propsSchema };
export default function Dashboard() {
  const { props, isPending, theme } = useWidget<Props>();
  if (isPending) return <McpUseProvider autoSize><div>Loading...</div></McpUseProvider>;
  return <McpUseProvider autoSize><div>User: {props.userId}</div></McpUseProvider>;
}
```

**Response helpers:** `text()` `object()` `markdown()` `widget()` `mix()` `error()` `resource()`

**Golden rules:**
- One tool = one capability (split broad actions)
- Return complete data upfront (avoid lazy-loading chains)
- Widgets own UI state via `useState` (not separate tools)
- Always check `isPending` before accessing `props`
- Use `process.env.API_KEY` — never hardcode secrets

**Auth options:** WorkOS · Supabase · Custom provider → all via `oauth` config + `ctx.auth`
**Deploy:** `modal deploy` (Manufact Cloud) · Docker · self-host

---

## 2. Agentic Loop (dylan/Wirasm pattern)
**Trigger:** "auto code review", "self-healing loop", "review→develop→validate→PR", "agentic pipeline"

**Architecture:** Review → Develop → Validate → PR · auto-retry on failure · max N iterations

```python
from enum import Enum
from pathlib import Path
import subprocess, time

class ComparisonMode(Enum):
    LATEST_COMMIT = "latest_commit"   # HEAD~1...HEAD
    BRANCH = "branch"                  # base...head

class AgenticLoop:
    def __init__(self, base="main", head="dev", mode=ComparisonMode.BRANCH,
                 max_iterations=3, output_dir=None, verbose=False):
        self.base, self.head = base, head
        self.mode = mode
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir or f"tmp/loop_{int(time.time())}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compare_cmd = ("HEAD~1...HEAD" if mode == ComparisonMode.LATEST_COMMIT
                            else f"{base}...{head}")

    def run_agent(self, script, *args):
        try:
            subprocess.run(["python", f"scripts/{script}", *args], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def run(self):
        for i in range(1, self.max_iterations + 1):
            review_ok   = self.run_agent("simple_review.py",    self.head, "--output", str(self.output_dir/"review.md"))
            dev_ok      = self.run_agent("simple_dev.py",       str(self.output_dir/"review.md"), "--branch", self.head)
            val_result  = subprocess.run(["python", "scripts/simple_validator.py",
                                          str(self.output_dir/"review.md"),
                                          str(self.output_dir/"dev_report.md"),
                                          "--branch", self.head])
            if val_result.returncode == 0:  # validation passed
                self.run_agent("simple_pr.py", str(self.output_dir/"validation.md"),
                               "--branch", self.head, "--base", self.base)
                return True
        return False
```

**CLI usage:**
```bash
# Latest commit review
uv run python scripts/agentic_loop.py --latest-commit --verbose

# Branch comparison, 5 retries
uv run python scripts/agentic_loop.py --branches main feature-x --max-iterations 5

# Custom output + PR title
uv run python scripts/agentic_loop.py --latest-commit --output-dir tmp/run1 --pr-title "Fix auth"
```

**Outputs to:** `review.md` · `dev_report.md` · `validation.md` · `pr.md` (all in timestamped dir)

---

## 3. n8n Workflow Validator
**Trigger:** "validate n8n workflow", "n8n best practices", "check n8n JSON"
**Requires:** `const { validateWorkflow } = require('./n8n-workflow-validator')`

```javascript
const { validateWorkflow } = require('./n8n-workflow-validator');

const result = validateWorkflow(workflowJson, {
  validators: ['naming', 'errorHandling', 'security', 'performance', 'documentation'],
  strictness: 'medium'  // 'low' | 'medium' | 'high'
});

console.log(result.passed, result.totalIssues);
// result.results.naming.issues → array of issue strings
// result.results.security.suggestions → fix suggestions
```

**What each validator checks:**

| Validator | Checks |
|-----------|--------|
| `naming` | Workflow name length · default node names · duplicate names |
| `errorHandling` | Error Trigger node · error workflow setting · HTTP nodes without error branches |
| `security` | Hard-coded credentials · unsecured webhooks · missing credential objects |
| `performance` | Node count >50 · loop batch size >100 · excessive HTTP nodes |
| `documentation` | Missing description · no sticky notes on complex flows · missing tags |

**Strictness:** `low` = skip warnings · `medium` = standard · `high` = enforce all including descriptions

---

## 4. Swarm Orchestration (pheromone-based)
**Trigger:** "multi-agent coordination", "swarm signals", "pheromone orchestration", "agent recruitment"

**Signal categories & evaporation rates:**
| Category | Rate | Use |
|----------|------|-----|
| `state` | 0.02 (slow) | Completed milestones |
| `need` | 0.08 | Work required |
| `problem` | 0.05 | Issues/bugs |
| `priority` | 0.04 | Escalations |
| `dependency` | 0.01 (persist) | Blockers |
| `anticipatory` | 0.15 (fast) | Look-ahead hints |

**Key signal patterns:**
```json
{ "type": "coding_needed_for_feature_X",    "category": "need",    "strength": 1.0 }
{ "type": "critical_bug_in_feature_X",       "category": "problem", "strength": 2.5 }
{ "type": "security_vulnerability_found_in_M","category": "problem", "strength": 2.7 }
{ "type": "coding_complete_for_feature_X",   "category": "state",   "strength": 1.0 }
```

**Auto-recruitment thresholds:**
```json
"Debugger_Targeted":    { "critical_bug_in_feature_X": 6.0, "system_level_bug_detected": 8.0 }
"SecurityReviewer_Module": { "security_vulnerability_found_in_M": 4.0 }
"Optimizer_Module":     { "performance_bottleneck_in_N": 5.0 }
```

**Emergency escalation:** `system_level_bug_detected` at strength ≥9.0 triggers immediate halt + all-hands
**Conflict resolution:** highest priority → signal strength → minimal context switching
**Anticipatory signals:** look 2 steps ahead · threshold 0.7 · evaporate fast (0.15) to stay fresh

**Keyword → signal mapping (for LLM interpretation):**
- "coding complete" / "tests pass" → `coding_complete_for_feature_X`
- "critical bug" / "environment error" → `critical_bug_in_feature_X`
- "fix proposed" → `debug_fix_proposed_for_feature_X`
- "documentation created" → `documentation_file_registered`
