import os
import json
import requests
import random
import io
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from textblob import TextBlob
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# 🔐 Load Firebase credentials from environment variable (escaped JSON string for Render)
firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Hugging Face setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app, origins="*")

# Role-based context
ROLE_CONTEXT = {
    "parent": "You are helping a parent who is worried about their neurodiverse child. Answer with clarity and kindness.",
    "teacher": "You are supporting a teacher who wants to create a kind, inclusive classroom. Be practical and empathetic.",
    "mentor": "You are assisting a mentor who guides neurodiverse youth emotionally. Be thoughtful and calm.",
    "individual": "You are talking to someone who is neurodiverse and looking for support. Be validating and gentle.",
    "general": "You are a helpful and supportive assistant. Keep responses friendly, empathetic, and informative."
}

# Firestore logging
def log_to_firestore(sender, message, role, flag_score=None, flag_label=None):
    entry = {
        "sender": sender,
        "message": message,
        "timestamp": datetime.utcnow(),
        "role": role,
    }
    if flag_score is not None:
        entry["flag_score"] = flag_score
    if flag_label:
        entry["flag_label"] = flag_label
    db.collection("chatLogs").add(entry)

# Behavior flag detection
def flag_behavior(user_input):
    classification_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    categories = ["overwhelm", "confusion", "focus issue", "neutral"]

    payload = {
        "inputs": user_input,
        "parameters": {"candidate_labels": categories}
    }

    response = requests.post(classification_url, headers=HEADERS, json=payload)
    result = response.json()

    top_label = result["labels"][0]
    score = result["scores"][0] if top_label != "neutral" else 0

    session["last_detected_flag"] = top_label if top_label != "neutral" else None

    previous_inputs = session.get("inputs", [])
    if user_input in previous_inputs:
        score += 0.5
    previous_inputs.append(user_input)
    session["inputs"] = previous_inputs[-10:]

    total_score = session.get("flag_score", 0) + score
    session["flag_score"] = total_score

    if total_score >= 8 and not session.get("neuro_flagged", False):
        session["neuro_flagged"] = True

    return total_score

# Sentiment analysis
def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    return blob.sentiment.polarity

# Chat logic
def get_gpt_response(user_input, user_role):
    if user_input.lower().strip() in ["hi", "hello", "hey"]:
        return "👋 Hello! I'm here to listen and support you. What's on your mind today?"

    sentiment = analyze_sentiment(user_input)
    total_flags = flag_behavior(user_input)

    tone = "You are a kind, emotionally intelligent assistant. Respond empathetically, understand the user's feelings, and provide supportive, helpful answers that feel natural and thoughtful."
    if sentiment < -0.3:
        tone += " The user might be feeling overwhelmed. Validate their emotions gently and be calming."
    elif sentiment > 0.3:
        tone += " The user seems encouraged. Reinforce that optimism and confidence."

    role_intro = ROLE_CONTEXT.get(user_role.lower(), ROLE_CONTEXT["general"])
    history = session.get("history", [])
    history.append({"role": "user", "content": user_input})

    formatted_history = "\n".join([
        f"User: {entry['content']}" if entry["role"] == "user" else f"Assistant: {entry['content']}"
        for entry in history
    ])

    prompt = f"{role_intro}\n{tone}\n{formatted_history}\nAssistant:"

    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            response_text = result[0]["generated_text"].split("Assistant:")[-1].strip()

            if total_flags >= 8 and not session.get("neuro_flagged", False):
                response_text += "\n\n🧠 It seems like you might be facing some challenges that could be related to neurodiverse traits. Would you like to explore this gently together?"

            if len(response_text) > 700:
                response_text = response_text[:680].rsplit('.', 1)[0] + "..."

            history.append({"role": "assistant", "content": response_text})
            session["history"] = history[-6:]
            return response_text

        elif "error" in result:
            return f"Error from Hugging Face: {result['error']}"
        else:
            return "Sorry, I couldn't understand the response."

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main chat route
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    user_role = request.json.get("role", "general")

    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    log_to_firestore("user", user_input, user_role)
    response = get_gpt_response(user_input, user_role)
    log_to_firestore("bot", response, user_role,
                     flag_score=session.get("flag_score"),
                     flag_label=session.get("last_detected_flag"))

    return jsonify({"response": response})

# Analyze logs and trigger alerts
@app.route("/analyze", methods=["POST"])
def analyze_logs():
    logs = db.collection("chatLogs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()

    flag_counter = {}
    alerts = []

    for doc in logs:
        data = doc.to_dict()
        if data["sender"] == "user" and "flag_label" in data:
            user_id = data.get("role", "general")
            flag = data["flag_label"]
            key = f"{user_id}:{flag}"
            flag_counter[key] = flag_counter.get(key, 0) + 1

            if flag_counter[key] == 3:
                alert = {
                    "user": user_id,
                    "flag": flag,
                    "timestamp": datetime.utcnow(),
                    "message": f"⚠️ Frequent signs of {flag} detected.",
                    "type": "behavioral"
                }
                alerts.append(alert)
                db.collection("user_alerts").add(alert)

    return jsonify({"message": "Analysis completed", "alerts_triggered": alerts})

# Reset session
@app.route("/reset", methods=["POST"])
def reset_convo():
    session["history"] = []
    session["flag_score"] = 0
    session["inputs"] = []
    session["neuro_flagged"] = False
    session["last_detected_flag"] = None
    return jsonify({"message": "Session reset"})

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
