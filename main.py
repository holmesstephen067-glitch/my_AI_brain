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

# 🔥 SAFE OpenAI Call
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

        # 🔴 HTTP ERROR CHECK
        if response.status_code != 200:
            return f"HTTP ERROR: {response.status_code} - {response.text}"

        data = response.json()

        # 🔍 DEBUG LOG
        print("OPENAI RESPONSE:", data)

        # 🔴 OPENAI ERROR CHECK
        if "error" in data:
            return f"OpenAI Error: {data['error']['message']}"

        if "choices" not in data:
            return f"Unexpected response format: {data}"

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
    data = request.get_json()

    if not data or "goal" not in data:
        return jsonify({"error": "Missing 'goal'"}), 400

    goal = data["goal"]

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

# 🚀 RUN (RENDER SAFE)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
