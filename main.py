from flask import Flask, request, jsonify
import requests
import os
import sqlite3
import json
import re
from datetime import datetime

app = Flask(__name__)

# 🔐 API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 🧠 Database Setup
conn = sqlite3.connect("memory.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT,
    response TEXT
)
""")
conn.commit()

# =========================
# 💾 MEMORY SYSTEM
# =========================
def save_memory(goal, response):
    c.execute(
        "INSERT INTO memory (goal, response) VALUES (?, ?)",
        (goal, response)
    )
    conn.commit()


def get_memory():
    c.execute("SELECT goal, response FROM memory ORDER BY id DESC LIMIT 5")
    return c.fetchall()


# =========================
# 🔍 LIGHT MEMORY MATCH (IMPROVED)
# =========================
def get_relevant_memory(goal):
    memory = get_memory()

    relevant = []
    goal_words = set(goal.lower().split())

    for g, r in memory:
        memory_words = set(g.lower().split())
        overlap = goal_words.intersection(memory_words)

        if len(overlap) > 0:
            relevant.append(f"{g} -> {r}")

    return "\n".join(relevant)


# =========================
# 🧰 SAFE TOOL SYSTEM
# =========================
def safe_calculate(expression):
    try:
        # Allow only safe characters
        if not re.match(r"^[0-9+\-*/(). ]+$", expression):
            return "Invalid characters in math expression"

        return str(eval(expression, {"__builtins__": {}}))
    except:
        return "Calculation error"


def get_time():
    return datetime.now().isoformat()


# 🧰 TOOL REGISTRY
TOOLS = {
    "calculate": safe_calculate,
    "get_time": get_time
}


# =========================
# 🧠 TOOL EXECUTION (REAL)
# =========================
def run_tool(tool_name, tool_input):
    if tool_name in TOOLS:
        return TOOLS[tool_name](tool_input)
    return f"Unknown tool: {tool_name}"


# =========================
# 🔥 OPENAI CALL (CLEAN + SAFE)
# =========================
def call_openai(messages):
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages
            },
            timeout=20
        )

        if response.status_code != 200:
            return f"HTTP ERROR: {response.status_code} - {response.text}"

        data = response.json()

        if "error" in data:
            return f"OpenAI Error: {data['error']['message']}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"


# =========================
# 🧠 AI BRAIN (YOUR LOOP - IMPROVED)
# =========================
def think(goal):
    if not OPENAI_API_KEY:
        return "ERROR: Missing OPENAI_API_KEY"

    memory_text = get_relevant_memory(goal)

    # 🧠 PLAN STEP
    plan_prompt = f"""
Break this goal into step-by-step actions.

Goal:
{goal}
"""

    plan = call_openai([
        {"role": "system", "content": "You are a planning AI."},
        {"role": "user", "content": plan_prompt}
    ])

    # 🧠 EXECUTION STEP
    execute_prompt = f"""
Use this plan and memory to complete the goal.

Relevant Memory:
{memory_text}

Plan:
{plan}

Goal:
{goal}
"""

    result = call_openai([
        {"role": "system", "content": "You execute plans and produce results."},
        {"role": "user", "content": execute_prompt}
    ])

    # 🧰 TOOL DETECTION (SAFE VERSION)
    tool_result = None

    if "calculate" in goal.lower():
        expression = goal.lower().split("calculate")[-1].strip()
        tool_result = run_tool("calculate", expression)

    if "time" in goal.lower():
        tool_result = run_tool("get_time", None)

    if tool_result:
        result += f"\n\n[Tool Used]: {tool_result}"

    return f"PLAN:\n{plan}\n\nRESULT:\n{result}"


# =========================
# 🌐 API ROUTES
# =========================
@app.route("/brain", methods=["POST"])
def brain():
    data = request.get_json()

    if not data or "goal" not in data:
        return jsonify({"error": "Missing 'goal'"}), 400

    goal = data["goal"]

    result = think(goal)

    save_memory(goal, result)

    return jsonify({"result": result})


@app.route("/test")
def test():
    return think("Give me a business idea I can start with $100")


@app.route("/")
def home():
    return "AI Brain (Agent + Memory + Tools) is running"


# =========================
# 🚀 RUN (RENDER SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
