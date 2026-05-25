import os
import time
import json
import re
import sqlite3
import hashlib
import secrets
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from dotenv import load_dotenv
import openai

app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

DB_PATH = "chatbot.db"
CSV_KB_PATH = "../knowledgebase/data/training_data.csv"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'incident_request_model.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'label_encoders.pkl')

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    password TEXT NOT NULL
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    type TEXT,
    category TEXT,
    subcategory TEXT,
    opened_by TEXT,
    affected_user TEXT,
    description TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT
)""")
conn.commit()


def is_valid_email(email):
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email.strip()))


def hash_password(password):
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt + hashed.hex()


def verify_password(password, hashed_password):
    if len(hashed_password) < 32:
        return False
    salt = hashed_password[:32]
    stored_hash = hashed_password[32:]
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return password_hash.hex() == stored_hash


def generate_session_token():
    return secrets.token_urlsafe(32)


def get_user_by_token(session_token):
    cursor.execute("SELECT user_id FROM sessions WHERE token = ?", (session_token,))
    row = cursor.fetchone()
    if not row:
        return None
    cursor.execute("SELECT full_name, id, email FROM users WHERE id = ?", (row[0],))
    user = cursor.fetchone()
    if not user:
        return None
    return {"full_name": user[0], "id": user[1], "email": user[2]}


def classify_ml(text):
    try:
        data = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        X = data["vectorizer"].transform([text])
        return {
            col: label_encoders[col].inverse_transform(
                [data["classifiers"][col].predict(X)[0]])[0]
            for col in ["category", "subcategory", "type"]
        }
    except Exception as e:
        print(f"ML classification failed: {e}")
        return None


def classify_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an IT support classifier. Return JSON with category, subcategory, type fields only."},
                {"role": "user", "content": f"Classify: {text}"}
            ]
        )
        return json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"GPT classification failed: {e}")
        return {"category": "General", "subcategory": "Other", "type": "Incident"}


def hybrid_classify(text):
    result = classify_ml(text)
    if result and result.get("category") != "General":
        return result
    return classify_gpt(text)


def get_rag_context(issue_text):
    try:
        df = pd.read_csv(CSV_KB_PATH)
        keywords = issue_text.lower().split()
        relevant = df[df.apply(
            lambda row: any(kw in str(row.values).lower() for kw in keywords), axis=1
        )].head(3)
        return relevant.to_json(orient='records') if not relevant.empty else "[]"
    except Exception:
        return "[]"


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()
    user_id = data.get('user_id', '').strip()
    password = data.get('password', '')

    if not full_name or not email or not user_id:
        return jsonify({"error": "All fields required"}), 400
    if not is_valid_email(email):
        return jsonify({"error": "Invalid email"}), 400

    cursor.execute("SELECT id FROM users WHERE id = ? OR email = ?", (user_id, email))
    if cursor.fetchone():
        return jsonify({"error": "User already exists"}), 409

    cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                   (user_id, full_name, email, hash_password(password)))
    conn.commit()
    return jsonify({"message": "User created successfully"})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('user_id', '').strip()
    password = data.get('password', '')

    cursor.execute("SELECT password, full_name, email FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row or not verify_password(password, row[0]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_session_token()
    cursor.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
    conn.commit()
    return jsonify({"token": token, "full_name": row[1], "email": row[2]})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_token = data.get('session_token')
    message = data.get('message', '').strip()

    user = get_user_by_token(session_token)
    if not user:
        return jsonify({"error": "Invalid session"}), 401

    classification = hybrid_classify(message)
    session_id = f"{user['full_name']}_{int(time.time())}"
    app.config[session_id] = {
        "classification": classification,
        "opened_by": user['full_name'],
        "affected_user": user['full_name'],
        "description": message
    }

    return jsonify({
        "reply": f"📝 Here's what I captured:\n"
                 f"• Category: {classification['category']}\n"
                 f"• Subcategory: {classification['subcategory']}\n"
                 f"• Description: {message}\n\n"
                 "Does this look correct? (yes / no / cancel)",
        "session_id": session_id,
        "fields": {
            "Opened By": user['full_name'],
            "Category": classification["category"],
            "Subcategory": classification["subcategory"],
            "Details": message
        }
    })


@app.route('/api/submit', methods=['POST'])
def submit():
    session_id = request.json.get('session_id')
    session_data = app.config.get(session_id)

    if not session_data:
        return jsonify({"error": "Invalid or expired session"}), 400

    try:
        ticket_id = f"TKT-{int(time.time())}"
        cursor.execute("""
            INSERT INTO tickets 
            (ticket_id, type, category, subcategory, opened_by, affected_user, description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            session_data["classification"].get("type", "Incident"),
            session_data["classification"]["category"],
            session_data["classification"]["subcategory"],
            session_data["opened_by"],
            session_data["affected_user"],
            session_data["description"],
            "Open"
        ))
        conn.commit()
        app.config.pop(session_id, None)
        return jsonify({"reply": f"✅ Ticket {ticket_id} created!", "ticket_id": ticket_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/troubleshoot', methods=['POST'])
def troubleshoot():
    issue_text = request.json.get('issue')
    context = get_rag_context(issue_text)

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful IT support assistant."},
                {"role": "user", "content": f"Issue: {issue_text}\nContext: {context}\nSuggest a quick fix."}
            ]
        )
        return jsonify({"suggestion": response['choices'][0]['message']['content'].strip()})
    except Exception:
        return jsonify({"suggestion": "Unable to suggest a fix. Please create a ticket."})


@app.route('/api/tickets', methods=['POST'])
def get_tickets():
    user = get_user_by_token(request.json.get('session_token'))
    if not user:
        return jsonify({"error": "Invalid session"}), 401

    cursor.execute("""
        SELECT ticket_id, type, category, subcategory, opened_by, 
               affected_user, description, status, timestamp
        FROM tickets WHERE opened_by = ? ORDER BY ticket_id DESC
    """, (user['full_name'],))

    tickets = [{
        "ticket_id": r[0], "type": r[1], "category": r[2],
        "subcategory": r[3], "opened_by": r[4], "affected_user": r[5],
        "description": r[6], "status": r[7], "created": r[8]
    } for r in cursor.fetchall()]

    return jsonify({"tickets": tickets})


@app.route('/api/logout', methods=['POST'])
def logout():
    cursor.execute("DELETE FROM sessions WHERE token = ?",
                   (request.json.get('session_token'),))
    conn.commit()
    return jsonify({"message": "Logged out successfully."})


@app.route('/api/verify-session', methods=['POST'])
def verify_session():
    user = get_user_by_token(request.json.get('session_token'))
    if not user:
        return jsonify({"error": "Invalid session"}), 401
    return jsonify({"user": user})


if __name__ == '__main__':
    app.run(debug=True)