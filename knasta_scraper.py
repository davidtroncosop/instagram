"""Polite scraper for public Knasta search result pages.

Knasta renders the public search results in the HTML as ``__NEXT_DATA__``.
This module reads only that public page, filters direct Falabella products and
does not call Knasta's private API or redirect endpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from html import unescape
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv


load_dotenv()

KNASTA_BASE_URL = "https://knasta.cl"
DEFAULT_SEARCH_TERMS = ("poleron", "polera", "jeans", "chaqueta", "vestido")
DEFAULT_USER_AGENT = (
    "InstagramOfferPipeline/1.0 "
    "(public search reader; contact project owner before increasing request rate)"
)


class KnastaScraperError(RuntimeError):
    """An expected, user-actionable scraper failure."""


@dataclass(frozen=True)
class KnastaOffer:
    product_name: str
    brand: str
    store: str
    seller: str
    category: str
    price_before_clp: int
    price_after_clp: int
    discount_percent: float
    keyword: str
    knasta_url: str
    product_url: str
    image_url: str
    current_day: str
    previous_price_day: str
    days_since_previous_price: int | None
    availability_note: str
    source_search_url: str


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise KnastaScraperError(f"{name} debe ser true o false")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise KnastaScraperError(f"{name} debe ser numérico") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise KnastaScraperError(f"{name} debe ser entero") from exc


def configured_search_terms(value: str | Iterable[str] | None = None) -> list[str]:
    if value is None:
        raw_terms: Iterable[str] = os.getenv(
            "KNASTA_SEARCH_TERMS", ",".join(DEFAULT_SEARCH_TERMS)
        ).split(",")
    elif isinstance(value, str):
        raw_terms = value.split(",")
    else:
        raw_terms = value

    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        term = str(raw_term).strip()
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        terms.append(term)
    if not terms:
        raise KnastaScraperError("KNASTA_SEARCH_TERMS no contiene términos")
    return terms


def _parse_clp(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        amount = int(round(float(value)))
        return amount if amount > 0 else None
    digits = re.sub(r"[^0-9]", "", str(value))
    if not digits:
        return None
    amount = int(digits)
    return amount if amount > 0 else None


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_next_data(document: str) -> dict[str, Any]:
    match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise KnastaScraperError("Knasta no entregó __NEXT_DATA__ en la página pública")
    try:
        parsed = json.loads(unescape(match.group(1)))
    except json.JSONDecodeError as exc:
        raise KnastaScraperError("No se pudo interpretar el JSON público de Knasta") from exc
    if not isinstance(parsed, dict):
        raise KnastaScraperError("El JSON público de Knasta no tiene el formato esperado")
    return parsed


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _extract_knasta_urls(document: str) -> dict[str, str]:
    """Map retailer SKU to its public Knasta detail URL from JSON-LD."""

    urls: dict[str, str] = {}
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_script in scripts:
        try:
            data = json.loads(unescape(raw_script))
        except json.JSONDecodeError:
            continue
        for item in _walk(data):
            sku = str(item.get("sku") or "").strip()
            detail_url = str(item.get("url") or "").strip()
            if "#" not in sku or not detail_url.startswith("https://knasta.cl/detail/"):
                continue
            retailer, product_id = sku.split("#", 1)
            if retailer.lower() == "falabella" and product_id:
                urls[product_id] = detail_url
    return urls


def _robots_allows_results(client: httpx.Client) -> None:
    response = client.get(f"{KNASTA_BASE_URL}/robots.txt")
    response.raise_for_status()
    disallowed: list[str] = []
    applies_to_all = False
    for raw_line in response.text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            applies_to_all = value == "*"
        elif key == "disallow" and applies_to_all and value:
            disallowed.append(value)
    result_path = urlparse(f"{KNASTA_BASE_URL}/results").path
    if any(result_path.startswith(rule.rstrip("*")) for rule in disallowed):
        raise KnastaScraperError("robots.txt no permite consultar los resultados de Knasta")


def _search_url(term: str) -> str:
    from urllib.parse import quote_plus

    return f"{KNASTA_BASE_URL}/results?q={quote_plus(term)}"


def _product_to_offer(
    product: dict[str, Any],
    source_search_url: str,
    knasta_urls: dict[str, str],
    minimum_discount_percent: float,
) -> KnastaOffer | None:
    retailer = str(product.get("retail") or "").strip().lower()
    if retailer != "falabella":
        return None

    product_id = str(product.get("product_id") or "").strip()
    product_url = str(product.get("url") or "").strip()
    if not product_id or not product_url.startswith("https://www.falabella.com/falabella-cl/"):
        return None

    current_price = _parse_clp(product.get("current_price"))
    previous_price = _parse_clp(product.get("last_variation_price"))
    if current_price is None or previous_price is None or previous_price <= current_price:
        return None

    calculated_discount = (previous_price - current_price) / previous_price * 100
    declared_percent = product.get("percent")
    try:
        declared_discount = max(0.0, -float(declared_percent))
    except (TypeError, ValueError):
        declared_discount = 0.0
    discount = round(max(calculated_discount, declared_discount), 1)
    if discount < minimum_discount_percent:
        return None

    knasta_url = knasta_urls.get(product_id, source_search_url)
    return KnastaOffer(
        product_name=str(product.get("title") or product.get("brand_title") or "").strip(),
        brand=str(product.get("brand") or "").strip(),
        store="Falabella",
        seller="Falabella",
        category=str(product.get("category") or "ropa").strip(),
        price_before_clp=previous_price,
        price_after_clp=current_price,
        discount_percent=discount,
        keyword=os.getenv("KNASTA_COMMENT_KEYWORD", "LOOK").strip().upper() or "LOOK",
        knasta_url=knasta_url,
        product_url=product_url,
        image_url=str(product.get("image") or "").strip(),
        current_day=str(product.get("current_day") or "").strip(),
        previous_price_day=str(product.get("last_variation_day") or "").strip(),
        days_since_previous_price=_parse_optional_int(product.get("ndays")),
        availability_note="Verificar stock y tallas directamente en Falabella antes de publicar.",
        source_search_url=source_search_url,
    )


def scrape_knasta_offers(
    search_terms: str | Iterable[str] | None = None,
    minimum_discount_percent: float | None = None,
    limit: int | None = None,
    max_results_per_term: int | None = None,
) -> list[dict[str, Any]]:
    """Return verified Falabella clothing deals from public Knasta results."""

    minimum = (
        _env_float("KNASTA_MIN_DISCOUNT_PERCENT", 30.0)
        if minimum_discount_percent is None
        else float(minimum_discount_percent)
    )
    if not 0 <= minimum <= 100:
        raise KnastaScraperError("El descuento mínimo debe estar entre 0 y 100")

    max_offers = (
        _env_int("KNASTA_MAX_OFFERS", 5) if limit is None else int(limit)
    )
    per_term = (
        _env_int("KNASTA_MAX_RESULTS_PER_TERM", 40)
        if max_results_per_term is None
        else int(max_results_per_term)
    )
    if max_offers <= 0 or per_term <= 0:
        raise KnastaScraperError("KNASTA_MAX_OFFERS y KNASTA_MAX_RESULTS_PER_TERM deben ser mayores que cero")

    terms = configured_search_terms(search_terms)
    headers = {
        "User-Agent": os.getenv("KNASTA_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "es-CL,es;q=0.9,en;q=0.5",
    }
    delay = max(0.0, _env_float("KNASTA_REQUEST_DELAY_SECONDS", 2.0))
    offers_by_product: dict[str, KnastaOffer] = {}

    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=45) as client:
            if _env_bool("KNASTA_RESPECT_ROBOTS", True):
                _robots_allows_results(client)

            for index, term in enumerate(terms):
                if index:
                    time.sleep(delay)
                source_search_url = _search_url(term)
                response = client.get(source_search_url)
                if response.status_code == 403:
                    raise KnastaScraperError(
                        "Knasta rechazó la consulta pública (403); no se intenta evadir el bloqueo."
                    )
                response.raise_for_status()
                document = response.text
                data = _extract_next_data(document)
                initial_data = data.get("props", {}).get("pageProps", {}).get("initialData", {})
                products = initial_data.get("products", [])
                if not isinstance(products, list):
                    continue
                knasta_urls = _extract_knasta_urls(document)
                for raw_product in products[:per_term]:
                    if not isinstance(raw_product, dict):
                        continue
                    offer = _product_to_offer(
                        raw_product,
                        source_search_url,
                        knasta_urls,
                        minimum,
                    )
                    if offer is None:
                        continue
                    existing = offers_by_product.get(offer.product_url)
                    if existing is None or offer.discount_percent > existing.discount_percent:
                        offers_by_product[offer.product_url] = offer
    except KnastaScraperError:
        raise
    except httpx.HTTPError as exc:
        raise KnastaScraperError(f"No se pudo consultar Knasta: {exc}") from exc

    ordered = sorted(
        offers_by_product.values(),
        key=lambda item: (-item.discount_percent, item.price_after_clp, item.product_name),
    )
    return [asdict(offer) for offer in ordered[:max_offers]]


def format_clp_for_voice(amount: int) -> str:
    """Spell common CLP amounts so Fish Audio does not read symbols poorly."""

    units = (
        "cero",
        "uno",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
        "diez",
        "once",
        "doce",
        "trece",
        "catorce",
        "quince",
        "dieciséis",
        "diecisiete",
        "dieciocho",
        "diecinueve",
        "veinte",
        "veintiuno",
        "veintidós",
        "veintitrés",
        "veinticuatro",
        "veinticinco",
        "veintiséis",
        "veintisiete",
        "veintiocho",
        "veintinueve",
    )
    tens = ("", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa")

    def under_thousand(value: int) -> str:
        if value < 30:
            return units[value]
        if value < 100:
            base, remainder = divmod(value, 10)
            return tens[base] if remainder == 0 else f"{tens[base]} y {units[remainder]}"
        hundreds = {
            1: "cien",
            2: "doscientos",
            3: "trescientos",
            4: "cuatrocientos",
            5: "quinientos",
            6: "seiscientos",
            7: "setecientos",
            8: "ochocientos",
            9: "novecientos",
        }
        base, remainder = divmod(value, 100)
        if remainder == 0:
            return hundreds[base]
        prefix = "ciento" if base == 1 else hundreds[base]
        return f"{prefix} {under_thousand(remainder)}"

    if amount < 0:
        return f"menos {format_clp_for_voice(-amount)}"
    if amount < 1000:
        return under_thousand(amount)
    thousands, remainder = divmod(amount, 1000)
    thousand_word = "mil" if thousands == 1 else f"{under_thousand(thousands)} mil"
    return thousand_word if remainder == 0 else f"{thousand_word} {under_thousand(remainder)}"


def _speakable_title(title: str) -> str:
    replacements = {
        "oversize": "oversais",
        "oversized": "oversais",
        "boxy": "boxi",
        "hoodie": "judi",
        "hood": "jod",
        "zip": "sip",
        "fit": "fit",
    }
    result = title
    for source, target in replacements.items():
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
    return result


def build_narration(offer: dict[str, Any]) -> str:
    """Build the short, fact-bound Chilean narration used by Fish Audio."""

    before = int(offer["price_before_clp"])
    after = int(offer["price_after_clp"])
    keyword = str(offer.get("keyword") or "LOOK").upper()
    title = _speakable_title(str(offer.get("product_name") or "esta prenda").strip())
    brand = _speakable_title(str(offer.get("brand") or "").strip())
    if brand and brand.casefold() not in title.casefold():
        title = f"{title} {brand}"
    return (
        f"Mira lo que encontré... {title}, de {format_clp_for_voice(before)} "
        f"bajó a {format_clp_for_voice(after)}... Comenta {keyword} y te mando el link..."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", help="Términos separados por coma")
    parser.add_argument("--min-discount", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-results-per-term", type=int, default=None)
    parser.add_argument("--output", type=Path, help="Guardar la lista JSON en un archivo")
    parser.add_argument("--narration", action="store_true", help="Imprimir también una narración para la primera oferta")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        offers = scrape_knasta_offers(
            search_terms=args.terms,
            minimum_discount_percent=args.min_discount,
            limit=args.limit,
            max_results_per_term=args.max_results_per_term,
        )
        payload: dict[str, Any] = {"offers": offers, "count": len(offers)}
        if args.narration and offers:
            payload["narration"] = build_narration(offers[0])
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    except KnastaScraperError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
