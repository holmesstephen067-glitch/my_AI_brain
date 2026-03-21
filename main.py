from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def think(prompt):
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
                    {"role": "system", "content": "You are an AI brain that thinks step-by-step and helps complete tasks."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"

@app.route("/brain", methods=["POST"])
def brain():
    data = request.json
    goal = data.get("goal")

    result = think(goal)

    return jsonify({"result": result})

@app.route("/")
def home():
    return "AI Brain is running"