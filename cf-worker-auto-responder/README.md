# ⚡ Cloudflare Worker Instagram Auto-Responder

Este proyecto es un **Cloudflare Worker** serverless diseñado para responder automáticamente a comentarios en Instagram Reels enviando un **Instagram Direct Message (DM)** con el enlace de la oferta de Knasta/Falabella.

---

## 🚀 Despliegue en 2 Minutos con Wrangler

### 1. Crear el Namespace KV en Cloudflare
```bash
npx wrangler kv namespace create PUBLISHED_REELS
```
Copia el `id` devuelto y reemplázalo en `wrangler.jsonc`:
```json
"kv_namespaces": [
  {
    "binding": "PUBLISHED_REELS",
    "id": "TU_KV_NAMESPACE_ID"
  }
]
```

### 2. Guardar Secretos en Cloudflare
```bash
npx wrangler secret put INSTAGRAM_ACCESS_TOKEN
npx wrangler secret put INSTAGRAM_USER_ID
npx wrangler secret put WEBHOOK_VERIFY_TOKEN
```

### 3. Publicar el Worker
```bash
npx wrangler deploy
```
Obtendrás tu URL permanente en vivo:  
`https://instagram-auto-responder.<tu-subdominio>.workers.dev`

---

## 🔗 Vinculación con Meta Graph API Webhook

1. Ingresa a **Meta Developer Dashboard** -> **Instagram Graph API** -> **Webhooks**.
2. **Callback URL:** `https://instagram-auto-responder.<tu-subdominio>.workers.dev/webhook`
3. **Verify Token:** `antigravity_secret_123` (o tu `WEBHOOK_VERIFY_TOKEN`).
4. Suscríbete a los eventos de: **`comments`**.

---

## 🐍 Sincronización Automática con `pipeline.py`

En tu archivo `.env` local, agrega:
```env
CLOUDFLARE_WORKER_URL=https://instagram-auto-responder.<tu-subdominio>.workers.dev
API_SECRET_KEY=antigravity_api_secret
```

Cada vez que ejecutes `python pipeline.py --publish`, el script enviará automáticamente el `media_id` y la URL de la prenda a tu **Cloudflare KV**, dejando la respuesta automática activa de forma inmediata.
