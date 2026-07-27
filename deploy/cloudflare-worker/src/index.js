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

function parsePrice(value) {
  if (typeof value === "number" && Number.isFinite(value)) return Math.round(value);
  const digits = String(value ?? "").replace(/[^0-9]/g, "");
  const parsed = Number(digits);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function parseNextData(html) {
  const match = html.match(/<script[^>]+id=["']__NEXT_DATA__["'][^>]*>([\s\S]*?)<\/script>/i);
  if (!match) throw new Error("Knasta no entregó __NEXT_DATA__");
  const decoded = match[1]
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
  return JSON.parse(decoded);
}

function findKnastaDetailUrl(html, productId) {
  const marker = `"sku":"falabella#${productId}"`;
  const start = html.indexOf(marker);
  if (start < 0) return "";
  const window = html.slice(start, start + 1200);
  const direct = window.match(/"url":"(https:\/\/knasta\.cl\/detail\/[^"\\]+)"/);
  return direct ? direct[1] : "";
}

function searchTerms(env, override) {
  const raw = override || env.KNASTA_SEARCH_TERMS || "poleron,polera,jeans,chaqueta,vestido";
  return raw.split(",").map((term) => term.trim()).filter(Boolean);
}

async function scrapeKnastaOffers(env, options = {}) {
  const terms = searchTerms(env, options.terms);
  const minDiscount = Number(options.minDiscount ?? env.KNASTA_MIN_DISCOUNT_PERCENT ?? "30");
  const limit = Number(options.limit ?? 5);
  if (!Number.isFinite(minDiscount) || minDiscount < 0 || minDiscount > 100) {
    throw new Error("KNASTA_MIN_DISCOUNT_PERCENT debe estar entre 0 y 100");
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) {
    throw new Error("El límite de ofertas debe estar entre 1 y 20");
  }

  const offers = new Map();
  const delayMs = Math.max(0, Number(env.KNASTA_REQUEST_DELAY_MS || "750"));
  for (let index = 0; index < terms.length; index += 1) {
    if (index && delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    const sourceSearchUrl = `https://knasta.cl/results?q=${encodeURIComponent(terms[index])}`;
    const response = await fetch(sourceSearchUrl, {
      headers: {
        Accept: "text/html,application/xhtml+xml",
        "Accept-Language": "es-CL,es;q=0.9,en;q=0.5",
        "User-Agent": "Mozilla/5.0 (compatible; InstagramOfferPipeline/1.0)",
      },
    });
    if (response.status === 403) {
      throw new Error("Knasta rechazó la consulta pública desde Cloudflare; no se intenta evadir el bloqueo");
    }
    if (!response.ok) throw new Error(`Knasta devolvió HTTP ${response.status}`);
    const html = await response.text();
    const data = parseNextData(html);
    const products = data?.props?.pageProps?.initialData?.products;
    if (!Array.isArray(products)) continue;

    for (const product of products) {
      if (String(product?.retail || "").toLowerCase() !== "falabella") continue;
      const productId = String(product?.product_id || "").trim();
      const productUrl = String(product?.url || "").trim();
      const current = parsePrice(product?.current_price);
      const previous = parsePrice(product?.last_variation_price);
      if (!productId || !productUrl.startsWith("https://www.falabella.com/falabella-cl/")) continue;
      if (!current || !previous || previous <= current) continue;
      const declared = Number(product?.percent);
      const calculated = ((previous - current) / previous) * 100;
      const discount = Math.round(Math.max(calculated, declared < 0 ? -declared : 0) * 10) / 10;
      if (discount < minDiscount) continue;
      const offer = {
        product_name: String(product?.title || product?.brand_title || "").trim(),
        brand: String(product?.brand || "").trim(),
        store: "Falabella",
        seller: "Falabella",
        category: String(product?.category || "ropa").trim(),
        price_before_clp: previous,
        price_after_clp: current,
        discount_percent: discount,
        keyword: String(env.KNASTA_COMMENT_KEYWORD || "LOOK").trim().toUpperCase() || "LOOK",
        knasta_url: findKnastaDetailUrl(html, productId) || sourceSearchUrl,
        product_url: productUrl,
        image_url: String(product?.image || "").trim(),
        current_day: String(product?.current_day || "").trim(),
        previous_price_day: String(product?.last_variation_day || "").trim(),
        days_since_previous_price: Number.isFinite(Number(product?.ndays)) ? Number(product.ndays) : null,
        availability_note: "Verificar stock y tallas directamente en Falabella antes de publicar.",
        source_search_url: sourceSearchUrl,
      };
      const existing = offers.get(productUrl);
      if (!existing || offer.discount_percent > existing.discount_percent) offers.set(productUrl, offer);
    }
  }
  return [...offers.values()]
    .sort((a, b) => b.discount_percent - a.discount_percent || a.price_after_clp - b.price_after_clp)
    .slice(0, limit);
}

async function invokePipeline(env, source) {
  if (source === "cron" && env.ENABLE_AUTOMATION !== "true") {
    return { accepted: false, skipped: true, reason: "automation_disabled" };
  }
  if (!env.PIPELINE_URL || !env.PIPELINE_TRIGGER_TOKEN) {
    throw new Error("PIPELINE_URL or PIPELINE_TRIGGER_TOKEN is not configured");
  }

  const knastaEnabled = env.KNASTA_ENABLED === "true";
  const minDiscount = Number(env.KNASTA_MIN_DISCOUNT_PERCENT || "30");
  const searchTerms = (env.KNASTA_SEARCH_TERMS || "")
    .split(",")
    .map((term) => term.trim())
    .filter(Boolean);
  const selectedOffer = knastaEnabled
    ? (await scrapeKnastaOffers(env, { limit: 1 }))[0]
    : undefined;
  if (knastaEnabled && !selectedOffer) {
    throw new Error("Knasta no encontró una oferta de Falabella con el descuento mínimo");
  }

  const response = await fetch(`${env.PIPELINE_URL.replace(/\/$/, "")}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.PIPELINE_TRIGGER_TOKEN}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      product_url: selectedOffer ? undefined : env.PRODUCT_URL || undefined,
      narration_text: selectedOffer ? undefined : env.NARRATION_TEXT || undefined,
      caption: env.INSTAGRAM_CAPTION || undefined,
      model_image_url: env.MODEL_IMAGE_URL || undefined,
      base_video_url: env.BASE_VIDEO_URL || undefined,
      offer: selectedOffer,
      knasta_enabled: Boolean(knastaEnabled && !selectedOffer),
      knasta_search_terms: searchTerms.length ? searchTerms : undefined,
      min_discount_percent: Number.isFinite(minDiscount) ? minDiscount : 30,
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
    if (url.pathname === "/offers" && request.method === "GET") {
      if (!authorized(request, env)) return json({ error: "Unauthorized" }, 401);
      try {
        const terms = url.searchParams.get("terms") || undefined;
        const minDiscount = url.searchParams.get("min_discount");
        const limit = url.searchParams.get("limit");
        return json({
          offers: await scrapeKnastaOffers(env, {
            terms,
            minDiscount: minDiscount === null ? undefined : Number(minDiscount),
            limit: limit === null ? 5 : Number(limit),
          }),
        });
      } catch (error) {
        return json({ error: String(error.message || error) }, 502);
      }
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
