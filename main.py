from flask import Flask, request, jsonify
import requests
import os
import sqlite3

app = Flask(__name__)

# 🔐 Load API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 🧠 Setup SQLite Memory
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

# 🔍 Get Recent Memory
def get_memory():
    c.execute("SELECT goal, response FROM memory ORDER BY id DESC LIMIT 5")
    return c.fetchall()

# 🧠 AI Thinking Function
def think(prompt):
    memory = get_memory()

    memory_text = "\n".join(
        [f"Goal: {m[0]} | Result: {m[1]}" for m in memory]
    )

    full_prompt = f"""
You are an AI brain that remembers past tasks and improves over time.

Previous memory:
{memory_text}

Current goal:
{prompt}

Respond clearly and helpfully.
"""

    if not OPENAI_API_KEY:
        return "ERROR: Missing OPENAI_API_KEY"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an intelligent AI brain."},
                    {"role": "user", "content": full_prompt}
                ]
            },
            timeout=15
        )

        data = response.json()

        if "choices" not in data:
            return f"ERROR: {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"

# 🚀 Main Brain Endpoint
@app.route("/brain", methods=["POST"])
def brain():
    data = request.json
    goal = data.get("goal")

    result = think(goal)

    save_memory(goal, result)

    return jsonify({"result": result})

# 🌐 Health Check
@app.route("/")
def home():
    return "AI Brain with Memory is running"