import os
import requests
import random
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from textblob import TextBlob
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session tracking
CORS(app, origins="*")

ROLE_CONTEXT = {
    "parent": "You are helping a parent who is worried about their neurodiverse child. Answer with clarity and kindness.",
    "teacher": "You are supporting a teacher who wants to create a kind, inclusive classroom. Be practical and empathetic.",
    "mentor": "You are assisting a mentor who guides neurodiverse youth emotionally. Be thoughtful and calm.",
    "individual": "You are talking to someone who is neurodiverse and looking for support. Be validating and gentle.",
    "general": "You are a helpful and supportive assistant. Keep responses friendly, empathetic, and informative."
}

# Smarter behavior detection using zero-shot classification
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

    previous_inputs = session.get("inputs", [])
    similar = [msg for msg in previous_inputs if msg == user_input]
    if len(similar) >= 1:
        score += 0.5

    previous_inputs.append(user_input)
    session["inputs"] = previous_inputs[-10:]

    total_score = session.get("flag_score", 0) + score
    session["flag_score"] = total_score

    if total_score >= 8 and not session.get("neuro_flagged", False):
        session["neuro_flagged"] = True

    return total_score

def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    return blob.sentiment.polarity

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

    prompt = (
        f"{role_intro}\n"
        f"{tone}\n"
        f"{formatted_history}\n"
        f"Assistant:"
    )

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

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    user_role = request.json.get("role", "general")

    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    response = get_gpt_response(user_input, user_role)
    return jsonify({"response": response})

@app.route("/reset", methods=["POST"])
def reset_convo():
    session["history"] = []
    session["flag_score"] = 0
    session["inputs"] = []
    session["neuro_flagged"] = False
    return jsonify({"message": "Session reset"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
