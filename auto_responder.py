import os
import json
import time
import httpx
from pathlib import Path
from flask import Flask, request, jsonify
import dotenv

dotenv.load_dotenv(override=True)

app = Flask(__name__)

VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "antigravity_secret_123")
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID = os.getenv("INSTAGRAM_USER_ID", "")
META_GRAPH_BASE_URL = os.getenv("META_GRAPH_BASE_URL", "https://graph.facebook.com").rstrip("/")
TRIGGER_KEYWORDS = [kw.strip().upper() for kw in os.getenv("TRIGGER_KEYWORDS", "LOOK,OFERTA,QUIERO,VER").split(",") if kw.strip()]

def get_product_for_media(media_id: str) -> dict:
    history_file = Path("published_history.json")
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
            if str(media_id) in history:
                return history[str(media_id)]
        except Exception:
            pass
    return {"product_url": os.getenv("DEFAULT_PRODUCT_URL", "https://knasta.cl"), "title": "la oferta"}

def send_instagram_dm(comment_id: str, message_text: str) -> bool:
    """Send a private DM reply via Meta Graph API to a commenter."""
    url = f"{META_GRAPH_BASE_URL}/v19.0/{INSTAGRAM_USER_ID}/messages"
    payload = {
        "recipient": {"comment_id": comment_id},
        "message": {"text": message_text},
        "access_token": ACCESS_TOKEN
    }
    try:
        with httpx.Client(timeout=15) as client:
            r = client.post(url, json=payload)
            print(f"[Auto-Responder] DM Response Status: {r.status_code}, Body: {r.text}")
            return r.status_code == 200
    except Exception as e:
        print(f"[Auto-Responder] Error enviando DM por Instagram: {e}")
        return False

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("[Auto-Responder] Webhook verificado correctamente por Meta!")
        return challenge, 200
    return "Verification token mismatch", 403

@app.route("/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json(force=True, silent=True) or {}
    print(f"[Auto-Responder] Webhook event recibido: {json.dumps(data)}")
    
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            val = change.get("value", {})
            comment_text = val.get("text", "").strip()
            comment_id = val.get("id") or val.get("comment_id")
            media_id = val.get("media", {}).get("id") or val.get("media_id")
            from_user = val.get("from", {}).get("username", "usuario")
            
            if comment_text and comment_id:
                if any(kw in comment_text.upper() for kw in TRIGGER_KEYWORDS):
                    print(f"[Auto-Responder] 🎯 Coincidencia de palabra clave! Usuario: @{from_user}, Texto: '{comment_text}'")
                    prod_info = get_product_for_media(str(media_id))
                    link = prod_info.get("product_url")
                    title = prod_info.get("title", "la prenda")
                    
                    dm_text = (
                        f"¡Hola @{from_user}! 👋 Gracias por tu comentario.\n\n"
                        f"🛍️ Aquí tienes el enlace directo a la oferta de {title}:\n"
                        f"👉 {link}\n\n"
                        f"¡Aprovecha el descuento antes de que se agote stock! ✨"
                    )
                    send_instagram_dm(comment_id, dm_text)
                    
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 Servidor de Respuesta Automática iniciado en puerto {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
