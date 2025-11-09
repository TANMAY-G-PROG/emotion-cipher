from flask import Flask, render_template, request, jsonify
from emotion_cipher import encrypt_message, decrypt_message

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/encrypt')
def api_encrypt():
    data = request.get_json(force=True, silent=True) or {}
    msg = (data.get("message") or "").strip()
    pw  = data.get("passphrase") or "default-key"   # or enforce required, see below
    if not msg:
        return jsonify({"error": "message is required"}), 400
    try:
        result = encrypt_message(msg, pw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post('/decrypt')
def api_decrypt():
    data = request.get_json(force=True, silent=True) or {}
    token = (data.get("encrypted_text") or "").strip().replace("\n", "")
    pw    = data.get("passphrase") or "default-key"
    if not token:
        return jsonify({"error": "encrypted_text is required"}), 400
    if not token.startswith("EC"):
        return jsonify({"error": "Malformed token: missing EC prefix"}), 400
    try:
        result = decrypt_message(token, pw)
        return jsonify(result)
    except Exception as e:
        # Wrong passphrase or corrupted token commonly raises InvalidTag
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

