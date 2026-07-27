const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

function bearer(request) {
  const value = request.headers.get("Authorization") || "";
  return value.toLowerCase().startsWith("bearer ") ? value.slice(7).trim() : "";
}

function authorized(request, env) {
  const expected = env.WORKER_TRIGGER_TOKEN || "";
  return expected && bearer(request) === expected;
}

async function invokePipeline(env, source) {
  if (source === "cron" && env.ENABLE_AUTOMATION !== "true") {
    return { accepted: false, skipped: true, reason: "automation_disabled" };
  }
  if (!env.PIPELINE_URL || !env.PIPELINE_TRIGGER_TOKEN) {
    throw new Error("PIPELINE_URL or PIPELINE_TRIGGER_TOKEN is not configured");
  }

  const response = await fetch(`${env.PIPELINE_URL.replace(/\/$/, "")}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.PIPELINE_TRIGGER_TOKEN}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      product_url: env.PRODUCT_URL || undefined,
      narration_text: env.NARRATION_TEXT || undefined,
      caption: env.INSTAGRAM_CAPTION || undefined,
      model_image_url: env.MODEL_IMAGE_URL || undefined,
      base_video_url: env.BASE_VIDEO_URL || undefined,
      publish: env.PUBLISH_ENABLED === "true",
      organic_test: env.ORGANIC_TEST !== "false",
    }),
  });
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = { raw: text.slice(0, 1000) };
  }
  if (!response.ok) {
    throw new Error(`Cloud Run returned ${response.status}: ${JSON.stringify(payload)}`);
  }
  return { accepted: true, source, response: payload };
}

function isPeakSlot(date) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Santiago",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const hour = parts.find((part) => part.type === "hour")?.value;
  const minute = parts.find((part) => part.type === "minute")?.value;
  return new Set(["08:30", "14:00", "20:30"]).has(`${hour}:${minute}`);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health" && request.method === "GET") {
      return json({
        ok: true,
        automation_enabled: env.ENABLE_AUTOMATION === "true",
        pipeline_configured: Boolean(env.PIPELINE_URL && env.PIPELINE_TRIGGER_TOKEN),
      });
    }
    if (url.pathname === "/trigger" && request.method === "POST") {
      if (!authorized(request, env)) return json({ error: "Unauthorized" }, 401);
      try {
        return json(await invokePipeline(env, "manual"));
      } catch (error) {
        return json({ error: String(error.message || error) }, 502);
      }
    }
    return json({ error: "Not found" }, 404);
  },

  async scheduled(controller, env, ctx) {
    if (!isPeakSlot(new Date(controller.scheduledTime))) return;
    ctx.waitUntil(invokePipeline(env, "cron"));
  },
};
