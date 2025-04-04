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
    "parent": "You are helping a parent who is worried about their neurodiverse child.",
    "teacher": "You are supporting a teacher who wants to create a kind, inclusive classroom.",
    "mentor": "You are assisting a mentor who guides neurodiverse youth emotionally.",
    "individual": "You are talking to someone who is neurodiverse and looking for support.",
    "general": "You are helping someone who wants to better understand neurodiversity."
}

# Emotional starters
empathetic_starters = [
    "I hear you. 💙",
    "That sounds really tough.",
    "You're not alone in this.",
    "It’s okay to feel this way.",
    "Thank you for sharing that with me.",
    "I'm here for you."
]

# Detect behavior flags from message
def flag_behavior(user_input):
    flags = {
        "overwhelm": ["i can't", "i feel", "i'm tired", "i'm stressed", "i'm overwhelmed", "too much", "exhausted"],
        "confusion": ["i don't understand", "i'm confused", "what do you mean", "explain again"],
        "focus": ["can't focus", "distracted", "hard to pay attention", "forget", "mind wanders"],
        "repetition": [],  # handled separately
    }

    score = 0
    lowered = user_input.lower()

    for category, phrases in flags.items():
        for phrase in phrases:
            if phrase in lowered:
                score += 2 if category != "repetition" else 1

    # Repetition flag: check against previous inputs
    previous_inputs = session.get("inputs", [])
    similar = [msg for msg in previous_inputs if msg == user_input]
    if len(similar) >= 1:
        score += 2

    previous_inputs.append(user_input)
    session["inputs"] = previous_inputs[-10:]  # Keep last 10 inputs

    total_score = session.get("flag_score", 0) + score
    session["flag_score"] = total_score

    return total_score

# Sentiment scoring
def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    return blob.sentiment.polarity

# Chat logic
def get_gpt_response(user_input, user_role):
    sentiment = analyze_sentiment(user_input)
    total_flags = flag_behavior(user_input)

    tone = "Speak gently and supportively, like a kind therapist. Use short, caring sentences when needed."
    if sentiment < -0.3:
        tone += " The user might be feeling overwhelmed or upset. Focus on calming and reassuring them."
    elif sentiment > 0.3:
        tone += " The user seems hopeful. Encourage and nurture that feeling."

    role_intro = ROLE_CONTEXT.get(user_role.lower(), ROLE_CONTEXT["general"])

    prompt = (
        f"{role_intro}\n"
        f"{tone}\n"
        f"User: {user_input}\n"
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

            # Check for emotional triggers
            emotional_trigger_phrases = [
                "i feel", "i’m feeling", "i'm feeling", "i am struggling", "i'm struggling",
                "i need help", "i'm overwhelmed", "i'm tired", "i can't", "i don't know how",
                "i’m worried", "i'm scared", "i'm anxious"
            ]
            is_emotional = any(phrase in user_input.lower() for phrase in emotional_trigger_phrases)

            if is_emotional and not any(response_text.lower().startswith(starter.lower()) for starter in empathetic_starters):
                response_text = f"{random.choice(empathetic_starters)} {response_text}"

            # Append soft suggestion if flags are high
            if total_flags >= 8:
                response_text += "\n\n🧠 It seems like you might be facing some challenges that could be related to neurodiverse traits. Would you like to explore this gently together?"

            if len(response_text) > 700:
                response_text = response_text[:680].rsplit('.', 1)[0] + "..."

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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

