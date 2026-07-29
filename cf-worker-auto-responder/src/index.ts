interface Env {
  PUBLISHED_REELS: KVNamespace;
  WEBHOOK_VERIFY_TOKEN: string;
  INSTAGRAM_ACCESS_TOKEN: string;
  INSTAGRAM_USER_ID: string;
  API_SECRET_KEY?: string;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    // 1. GET /webhook -> Meta Webhook Handshake Verification
    if (request.method === "GET" && url.pathname === "/webhook") {
      const mode = url.searchParams.get("hub.mode");
      const token = url.searchParams.get("hub.verify_token");
      const challenge = url.searchParams.get("hub.challenge");

      const verifyToken = env.WEBHOOK_VERIFY_TOKEN || "antigravity_secret_123";

      if (mode === "subscribe" && token === verifyToken) {
        console.log("[CF Worker] Meta Webhook verificado con éxito!");
        return new Response(challenge, { status: 200 });
      }
      return new Response("Forbidden", { status: 403 });
    }

    // 2. POST /api/record-reel -> Internal Sync API for Python pipeline.py
    if (request.method === "POST" && url.pathname === "/api/record-reel") {
      const authHeader = request.headers.get("Authorization");
      const secret = env.API_SECRET_KEY || "antigravity_api_secret";
      if (authHeader !== `Bearer ${secret}`) {
        return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
      }

      try {
        const body: { media_id: string; product_url: string; title?: string } = await request.json();
        if (!body.media_id || !body.product_url) {
          return new Response(JSON.stringify({ error: "Missing media_id or product_url" }), { status: 400 });
        }

        await env.PUBLISHED_REELS.put(
          String(body.media_id),
          JSON.stringify({
            product_url: body.product_url,
            title: body.title || "la prenda",
            published_at: new Date().toISOString()
          })
        );

        console.log(`[CF Worker] Reel ${body.media_id} guardado en Cloudflare KV!`);
        return new Response(JSON.stringify({ status: "success", media_id: body.media_id }), { status: 200 });
      } catch (err: any) {
        return new Response(JSON.stringify({ error: err.message }), { status: 500 });
      }
    }

    // 3. POST /webhook -> Incoming Meta Instagram Comment Events
    if (request.method === "POST" && url.pathname === "/webhook") {
      try {
        const body: any = await request.json();
        console.log("[CF Worker] Webhook Event recibido:", JSON.stringify(body));

        ctx.waitUntil(handleMetaWebhookEvent(body, env));
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      } catch (err: any) {
        return new Response(JSON.stringify({ error: err.message }), { status: 500 });
      }
    }

    return new Response("Instagram Auto-Responder Worker is Active", { status: 200 });
  }
};

async function handleMetaWebhookEvent(body: any, env: Env) {
  const triggerKeywords = ["LOOK", "OFERTA", "QUIERO", "VER", "LINK"];

  for (const entry of body.entry || []) {
    for (const change of entry.changes || []) {
      const val = change.value || {};
      const commentText = (val.text || "").trim();
      const commentId = val.id || val.comment_id;
      const mediaId = val.media?.id || val.media_id;
      const fromUsername = val.from?.username || "usuario";

      if (commentText && commentId) {
        const upperText = commentText.toUpperCase();
        if (triggerKeywords.some((kw) => upperText.includes(kw))) {
          console.log(`[CF Worker] 🎯 Coincidencia de palabra clave! Usuario: @${fromUsername}, Comentario: '${commentText}'`);

          let productUrl = "https://knasta.cl";
          let title = "la oferta";

          if (mediaId && env.PUBLISHED_REELS) {
            const rawData = await env.PUBLISHED_REELS.get(String(mediaId));
            if (rawData) {
              try {
                const parsed = JSON.parse(rawData);
                productUrl = parsed.product_url || productUrl;
                title = parsed.title || title;
              } catch (_) {}
            }
          }

          const dmMessage = `¡Hola @${fromUsername}! 👋 Gracias por tu comentario en nuestro Reel.\n\n` +
                            `🛍️ Aquí tienes el enlace directo a la oferta de ${title}:\n` +
                            `👉 ${productUrl}\n\n` +
                            `¡Aprovecha el descuento antes de que se agote stock! ✨`;

          await sendInstagramPrivateReply(commentId, dmMessage, env);
        }
      }
    }
  }
}

async function sendInstagramPrivateReply(commentId: string, message: string, env: Env) {
  const url = `https://graph.facebook.com/v19.0/${env.INSTAGRAM_USER_ID}/messages`;
  const payload = {
    recipient: { comment_id: commentId },
    message: { text: message },
    access_token: env.INSTAGRAM_ACCESS_TOKEN
  };

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const resBody = await res.text();
    console.log(`[CF Worker] DM Response Status: ${res.status}, Body: ${resBody}`);
  } catch (err) {
    console.error("[CF Worker] Error enviando DM por Instagram:", err);
  }
}
