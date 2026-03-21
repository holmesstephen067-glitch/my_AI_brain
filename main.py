from flask import Flask, request, jsonify
import requests
import os
import sqlite3
import json
import re
from datetime import datetime
import hashlib

app = Flask(__name__)

# =========================
# 🔐 API KEYS
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# =========================
# 🧠 DATABASE
# =========================
conn = sqlite3.connect("app.db", check_same_thread=False)
c = conn.cursor()

# Users table (basic auth system)
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT
)
""")

# Memory tied to users
c.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    goal TEXT,
    response TEXT
)
""")

conn.commit()

# =========================
# 🔐 SIMPLE AUTH (MINIMAL)
# =========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(email, password):
    try:
        c.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False


def authenticate_user(email, password):
    c.execute("SELECT id, password FROM users WHERE email = ?", (email,))
    user = c.fetchone()

    if not user:
        return None

    user_id, stored_password = user

    if hash_password(password) == stored_password:
        return user_id

    return None


# =========================
# 💾 MEMORY (USER-BASED)
# =========================
def save_memory(user_id, goal, response):
    c.execute(
        "INSERT INTO memory (user_id, goal, response) VALUES (?, ?, ?)",
        (user_id, goal, response)
    )
    conn.commit()


def get_memory(user_id):
    c.execute(
        "SELECT goal, response FROM memory WHERE user_id = ? ORDER BY id DESC LIMIT 10",
        (user_id,)
    )
    return c.fetchall()


# =========================
# 🧠 MEMORY RELEVANCE
# =========================
def get_relevant_memory(goal, user_id):
    memory = get_memory(user_id)
    goal_words = set(goal.lower().split())

    scored = []

    for g, r in memory:
        memory_words = set(g.lower().split())
        score = len(goal_words.intersection(memory_words))

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
            return "Invalid expression"

        return str(eval(expression, {"__builtins__": {}}))
    except:
        return "Calculation error"


def get_time():
    return datetime.now().isoformat()


TOOLS = {
    "calculate": safe_calculate,
    "time": get_time
}


# =========================
# 🧰 TOOL EXECUTION
# =========================
def run_tool(tool_name, tool_input):
    if tool_name in TOOLS:
        return TOOLS[tool_name](tool_input)
    return f"Unknown tool: {tool_name}"


# =========================
# 🔥 LLM CALLS
# =========================
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


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

        res = requests.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}]
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
    response = call_openai(messages)
    if response:
        return response

    response = call_gemini(prompt_text)
    if response:
        return response

    return "No LLM available."


# =========================
# 🧠 AGENT CORE
# =========================
def think(goal, user_id):
    memory_text = get_relevant_memory(goal, user_id)

    # PLAN
    plan_prompt = f"""
Break this goal into steps.

Goal:
{goal}
"""

    plan = call_llm(
        [{"role": "user", "content": plan_prompt}],
        plan_prompt
    )

    # EXECUTE
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

    # TOOL TRIGGER (simple + reliable)
    if "calculate" in goal.lower():
        expr = goal.lower().split("calculate")[-1].strip()
        tool_result = run_tool("calculate", expr)
        result += f"\n\n[Tool Used]: {tool_result}"

    if "time" in goal.lower():
        tool_result = run_tool("time", None)
        result += f"\n\n[Tool Used]: {tool_result}"

    return f"PLAN:\n{plan}\n\nRESULT:\n{result}"


# =========================
# 🌐 API ROUTES
# =========================

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    success = create_user(email, password)

    return jsonify({"success": success})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user_id = authenticate_user(email, password)

    if user_id:
        return jsonify({"user_id": user_id})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/brain", methods=["POST"])
def brain():
    data = request.json

    goal = data.get("goal")
    user_id = data.get("user_id")

    if not goal or not user_id:
        return jsonify({"error": "Missing goal or user_id"}), 400

    result = think(goal, user_id)

    save_memory(user_id, goal, result)

    return jsonify({"result": result})


@app.route("/")
def home():
    return "🚀 SaaS AI Brain Running"


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
