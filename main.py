from flask import Flask, request, jsonify
import requests
import os
import sqlite3

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

# 💾 Save Memory
def save_memory(goal, response):
    c.execute("INSERT INTO memory (goal, response) VALUES (?, ?)", (goal, response))
    conn.commit()

# 🔍 Get Memory
def get_memory():
    c.execute("SELECT goal, response FROM memory ORDER BY id DESC LIMIT 5")
    return c.fetchall()

# 🧰 Simple Tools (expand later)
def use_tools(text):
    if "calculate" in text.lower():
        try:
            expression = text.lower().split("calculate")[-1].strip()
            return f"Calculation result: {eval(expression)}"
        except:
            return "Could not calculate"
    return None

# 🔥 SAFE OpenAI Call (FIXES YOUR ERROR)
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
            timeout=15
        )

        data = response.json()

        if "choices" not in data:
            return f"ERROR: {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"

# 🧠 AI Brain (Agent Loop)
def think(goal):
    memory = get_memory()

    memory_text = "\n".join(
        [f"Goal: {m[0]} | Result: {m[1]}" for m in memory]
    )

    if not OPENAI_API_KEY:
        return "ERROR: Missing OPENAI_API_KEY"

    # 🧠 PLAN
    plan_prompt = f"""
Break this goal into step-by-step actions.

Goal:
{goal}
"""

    plan = call_openai([
        {"role": "system", "content": "You are a planning AI."},
        {"role": "user", "content": plan_prompt}
    ])

    # 🧠 EXECUTE
    execute_prompt = f"""
Use this plan and memory to complete the goal.

Previous memory:
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

    # 🧰 TOOL CHECK
    tool_result = use_tools(goal)
    if tool_result:
        result += f"\n\n[Tool Used]: {tool_result}"

    return f"PLAN:\n{plan}\n\nRESULT:\n{result}"

# 🚀 Main API
@app.route("/brain", methods=["POST"])
def brain():
    data = request.json
    goal = data.get("goal")

    result = think(goal)

    save_memory(goal, result)

    return jsonify({"result": result})

# 🌐 Easy Phone Test
@app.route("/test")
def test():
    result = think("Give me a business idea I can start with $100")
    return result

# 🟢 Health Check
@app.route("/")
def home():
    return "AI Brain (Agent + Memory + Tools) is running"