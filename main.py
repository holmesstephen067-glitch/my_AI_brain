from flask import Flask, request, jsonify
import requests
import os
import sqlite3
import json
import re
from datetime import datetime

app = Flask(__name__)

# =========================
# 🔐 API KEYS
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # add later

# =========================
# 🧠 DATABASE
# =========================
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
    c.execute("INSERT INTO memory (goal, response) VALUES (?, ?)", (goal, response))
    conn.commit()


def get_memory():
    c.execute("SELECT goal, response FROM memory ORDER BY id DESC LIMIT 10")
    return c.fetchall()


def get_relevant_memory(goal):
    memory = get_memory()

    goal_words = set(goal.lower().split())
    scored = []

    for g, r in memory:
        memory_words = set(g.lower().split())
        overlap = goal_words.intersection(memory_words)

        score = len(overlap)
        if score > 0:
            scored.append((score, g, r))

    scored.sort(reverse=True, key=lambda x: x[0])

    return "\n".join([f"{g} -> {r}" for _, g, r in scored[:5]])


# =========================
# 🧰 SAFE TOOLS
# =========================
def safe_calculate(expression):
    try:
        if not re.match(r"^[0-9+\-*/(). ]+$", expression):
            return "Invalid math expression"

        return str(eval(expression, {"__builtins__": {}}))
    except:
        return "Calculation error"


def get_time():
    return datetime.now().isoformat()


def echo(text):
    return text


# =========================
# 🧰 TOOL REGISTRY
# =========================
TOOLS = {
    "calculate": safe_calculate,
    "time": get_time,
    "echo": echo
}


# =========================
# 🧠 TOOL EXECUTION
# =========================
def run_tool(tool_name, tool_input):
    if tool_name in TOOLS:
        return TOOLS[tool_name](tool_input)
    return f"Unknown tool: {tool_name}"


# =========================
# 🔥 LLM CALLS
# =========================

# --- OpenAI ---
def call_openai(messages):
    if not OPENAI_API_KEY:
        return None

    try:
        res = requests.post(
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

        if res.status_code != 200:
            return None

        data = res.json()
        return data["choices"][0]["message"]["content"]

    except:
        return None


# --- Gemini (READY BUT OPTIONAL) ---
def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

        res = requests.post(
            url,
            json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            },
            timeout=20
        )

        if res.status_code != 200:
            return None

        data = res.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except:
        return None


# =========================
# 🧠 LLM ROUTER
# =========================
def call_llm(messages, prompt_text):
    """
    Smart routing system:
    1. Try OpenAI
    2. Fallback to Gemini
    3. Fallback to simple response
    """

    # Try OpenAI
    response = call_openai(messages)
    if response:
        return response

    # Try Gemini
    response = call_gemini(prompt_text)
    if response:
        return response

    return "No LLM available. Please check API keys."


# =========================
# 🧠 AI BRAIN (YOUR LOOP — IMPROVED)
# =========================
def think(goal):
    memory_text = get_relevant_memory(goal)

    # 🧠 PLAN
    plan_prompt = f"""
Break this goal into steps.

Goal:
{goal}
"""

    plan = call_llm(
        [{"role": "user", "content": plan_prompt}],
        plan_prompt
    )

    # 🧠 EXECUTE
    execute_prompt = f"""
Use this plan and memory.

Memory:
{memory_text}

Plan:
{plan}

Goal:
{goal}
"""

    result = call_llm(
        [{"role": "user", "content": execute_prompt}],
        execute_prompt
    )

    # 🧰 TOOL DETECTION
    tool_result = None

    if "calculate" in goal.lower():
        expression = goal.lower().split("calculate")[-1].strip()
        tool_result = run_tool("calculate", expression)

    elif "time" in goal.lower():
        tool_result = run_tool("time", None)

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
        return jsonify({"error": "Missing goal"}), 400

    goal = data["goal"]

    result = think(goal)

    save_memory(goal, result)

    return jsonify({"result": result})


@app.route("/test")
def test():
    return think("Give me a business idea with $100")


@app.route("/")
def home():
    return "AI Brain (Multi-LLM + Tools + Memory) is running"


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
