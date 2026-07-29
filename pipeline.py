"""Generate an affiliate-fashion Reel with GPT Image 2 and Gemini Omni Flash.

The script deliberately stops at a local MP4 unless a publication flag is
passed. Instagram and TikTok require the final MP4 to be available at a public
HTTPS URL (or the corresponding URL variable in .env).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlparse

import io
from PIL import Image
import httpx
from dotenv import load_dotenv
from google import genai
import google.auth
from openai import OpenAI


load_dotenv()

DEFAULT_OUTFIT_PROMPT = """
La Imagen 1 es la chica de referencia (rostro, cabello, cuerpo, identidad).
La Imagen 2 es la habitación de fondo.
La Imagen 3 en adelante son diferentes vistas de la prenda.
Colócale a la chica de la Imagen 1 esta prenda mostrada en las imágenes de referencia siendo fiel a la prenda.
Mantén exactamente su rostro, cuerpo e identidad. Ubícala en la habitación de fondo de la Imagen 2 en formato portrait vertical 9:16.
""".strip()

DEFAULT_VIDEO_PROMPT = """
Recreate the 9:16 vertical video using the base video's motion guidance.
MANDATORY: Use the exact face, hair, body, identity, and outfit of the model girl from Image 1 (front view) and Image 2 (back view).
MANDATORY: Use the exact background room environment (walls, furniture, floor, decor, lighting) from Image 1 and Image 2.
Replace the original person and original environment in the base video with the girl and background from Image 1 and Image 2.
Keep the background room 100% static, frozen, and still with zero camera motion or distortion.
""".strip()

DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_MODEL_IMAGE = DOWNLOADS_DIR / "ChatGPT_Image_22_jul_2026,_202607252032.jpeg"
DEFAULT_BASE_VIDEO = DOWNLOADS_DIR / "0718_202607252032.mp4"
DEFAULT_GARMENT_DIR = DOWNLOADS_DIR / "prendas"


class PipelineError(RuntimeError):
    """An expected, user-actionable pipeline failure."""


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PipelineError(f"Falta la variable {name} en el archivo .env")
    return value


def gemini_backend() -> str:
    backend = os.getenv("GEMINI_BACKEND", "vertex").strip().lower()
    aliases = {
        "gcp": "vertex",
        "agent_platform": "vertex",
        "ai-studio": "ai_studio",
        "gemini_api": "ai_studio",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"vertex", "ai_studio"}:
        raise PipelineError(
            "GEMINI_BACKEND debe ser 'vertex' (GCP Agent Platform) o 'ai_studio'."
        )
    return backend


def create_gemini_client(backend: str) -> Any:
    if backend == "vertex":
        project = required_env("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)

    return genai.Client(api_key=required_env("GEMINI_API_KEY"))


def positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise PipelineError(f"{name} debe ser un número entero") from exc
    if value <= 0:
        raise PipelineError(f"{name} debe ser mayor que cero")
    return value


def boolean_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise PipelineError(f"{name} debe ser booleano: true/false")


def duration_env(name: str, default: str, minimum: int = 3, maximum: int = 10) -> str:
    raw = os.getenv(name, default).strip().lower()
    if not re.fullmatch(r"\d+s", raw):
        raise PipelineError(f"{name} debe tener el formato Ns, por ejemplo 10s")
    seconds = int(raw[:-1])
    if seconds < minimum or seconds > maximum:
        raise PipelineError(
            f"{name} debe estar entre {minimum}s y {maximum}s para Gemini Omni Flash"
        )
    return raw


def validate_input(path: Path, label: str) -> Path:
    if not path.is_file():
        raise PipelineError(f"No existe el archivo de {label}: {path}")
    return path


def image_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise PipelineError(
            f"Formato de imagen no soportado para {path}. Usa PNG, JPG o WEBP."
        )
    return mime


def encode_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def default_download_asset(env_name: str, fallback: Path) -> Path:
    configured = os.getenv(env_name, "").strip()
    return Path(configured).expanduser() if configured else fallback


def collect_garment_images(
    explicit_paths: Iterable[Path] | None,
    garment_dir: Path | None,
) -> list[Path]:
    paths: list[Path] = list(explicit_paths or [])
    if garment_dir:
        if not garment_dir.is_dir():
            raise PipelineError(f"No existe la carpeta de prendas: {garment_dir}")
        paths.extend(
            sorted(
                path
                for path in garment_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))

    if not unique:
        raise PipelineError(
            "Faltan las imágenes de la prenda. Usa --garment-dir con una carpeta "
            "que contenga las cuatro vistas o repite --garment-image."
        )
    if len(unique) > 9:
        raise PipelineError(
            "Puedes entregar como máximo 9 imágenes de prenda: la décima imagen "
            "del prompt sería la referencia de la modelo."
        )
    return [validate_input(path, "prenda") for path in unique]


class ProductImageParser(HTMLParser):
    """Collect image candidates from common product-page HTML metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self._json_script: list[str] | None = None

    def add_url(self, value: str | None) -> None:
        if not value:
            return
        for candidate in value.split(",") if "," in value else [value]:
            candidate = candidate.strip().split(" ", 1)[0]
            if candidate and not candidate.startswith("data:"):
                self.urls.append(unescape(candidate))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value for key, value in attrs}
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key in {"og:image", "og:image:url", "twitter:image"}:
                self.add_url(attributes.get("content"))
        elif tag == "img":
            for key in ("src", "data-src", "data-original", "data-lazy-src", "srcset"):
                self.add_url(attributes.get(key))
        elif tag == "link" and "image_src" in (attributes.get("rel") or "").lower():
            self.add_url(attributes.get("href"))
        elif tag == "script" and "ld+json" in (attributes.get("type") or "").lower():
            self._json_script = []

    def handle_data(self, data: str) -> None:
        if self._json_script is not None:
            self._json_script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "script" or self._json_script is None:
            return
        raw = "".join(self._json_script)
        self._json_script = None
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            return

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower() in {"image", "images", "contenturl", "thumbnailurl"}:
                        if isinstance(child, str):
                            self.add_url(child)
                        elif isinstance(child, list):
                            for item in child:
                                if isinstance(item, str):
                                    self.add_url(item)
                    visit(child)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(document)


def product_image_candidates(page_html: str, page_url: str) -> list[str]:
    parser = ProductImageParser()
    try:
        parser.feed(page_html)
    except Exception as exc:
        raise PipelineError(f"No se pudo interpretar la ficha del producto: {exc}") from exc

    candidates: list[str] = []
    seen: set[str] = set()
    for raw_url in parser.urls:
        absolute = urljoin(page_url, raw_url)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized = absolute.split("#", 1)[0]
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    return candidates


def scrape_knasta_product(knasta_url: str, output_dir: Path, count: int = 4) -> list[Path]:
    """Scrape garment images and offer data directly from a Knasta deal URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    }
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
            resp = client.get(knasta_url)
            resp.raise_for_status()
            html = resp.text

        json_ld_matches = re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL)
        product_data = {}
        for j_text in json_ld_matches:
            try:
                data = json.loads(j_text.strip())
                if isinstance(data, dict) and data.get("@type") == "Product":
                    product_data = data
                    break
            except Exception:
                continue

        if not product_data:
            raise PipelineError(f"No se pudo extraer la información del producto de Knasta: {knasta_url}")

        product_name = product_data.get("name", "Producto sin nombre")
        brand = product_data.get("brand", {}).get("name", "Marca") if isinstance(product_data.get("brand"), dict) else str(product_data.get("brand") or "Marca")
        offers = product_data.get("offers", {})
        if isinstance(offers, list):
            offers = offers[0] if offers else {}

        offer_price = str(offers.get("price", ""))
        seller = offers.get("seller", {}).get("name", "Retailer") if isinstance(offers.get("seller"), dict) else "Retailer"
        orig_price_match = re.search(r'Anterior:</span><span[^>]*>\$\s*([\d\.]+)', html)
        normal_price = orig_price_match.group(1).replace(".", "") if orig_price_match else offer_price
        disc_match = re.search(r'<span[^>]*>(\d+%)</span>', html)
        discount_percentage = disc_match.group(1) if disc_match else "0%"

        partner_match = re.search(r'partner_url=([^"&\s]+)', html)
        if partner_match:
            retailer_url = unquote(partner_match.group(1))
            dl_match = re.search(r'dl=([^"&\s]+)', retailer_url)
            if dl_match:
                retailer_url = unquote(dl_match.group(1))
        else:
            retailer_url = knasta_url

        raw_images = product_data.get("image", [])
        if isinstance(raw_images, str):
            raw_images = [raw_images]

        image_urls = []
        for img_url in raw_images:
            upgraded_url = re.sub(r'w=\d+,h=\d+', 'w=1000,h=1000', img_url)
            image_urls.append(upgraded_url)

        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        downloaded_urls: list[str] = []
        with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
            for idx, img_url in enumerate(image_urls[:count], 1):
                try:
                    img_resp = client.get(img_url)
                    img_resp.raise_for_status()
                    img_path = output_dir / f"{idx:02d}-prenda.jpg"
                    img_path.write_bytes(img_resp.content)
                    downloaded.append(img_path)
                    downloaded_urls.append(img_url)
                except Exception as exc:
                    print(f"Advertencia: no se pudo descargar la imagen {img_url}: {exc}")

        if not downloaded:
            raise PipelineError("No se pudieron descargar imágenes del producto desde Knasta")

        offer_json = {
            "product_name": product_name,
            "brand": brand,
            "retailer": seller,
            "normal_price": normal_price,
            "offer_price": offer_price,
            "discount_percentage": discount_percentage,
            "url": retailer_url,
            "knasta_url": knasta_url
        }
        (output_dir / "offer.json").write_text(json.dumps(offer_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "source.json").write_text(json.dumps({"product_url": knasta_url, "image_urls": downloaded_urls}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return downloaded
    except Exception as exc:
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(f"Error al procesar la ficha de Knasta: {exc}") from exc


def download_product_images(product_url: str, output_dir: Path, count: int = 4) -> list[Path]:
    """Download garment images from Knasta or Falabella product pages."""
    parsed = urlparse(product_url)
    hostname = (parsed.hostname or "").lower()
    if "knasta.cl" in hostname or "knasta.com" in hostname:
        return scrape_knasta_product(product_url, output_dir, count)

    allowed_host = (
        hostname in {"falabella.com", "falabella.cl"}
        or hostname.endswith(".falabella.com")
        or hostname.endswith(".falabella.cl")
    )
    if parsed.scheme != "https" or not allowed_host:
        raise PipelineError(
            "--product-url debe ser una ficha HTTPS de Knasta (knasta.cl) o Falabella."
        )
    if count <= 0:
        raise PipelineError("La cantidad de imágenes a descargar debe ser mayor que cero")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; InstagramOfferPipeline/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=45) as client:
            page_response = client.get(product_url)
            page_response.raise_for_status()
            candidates = product_image_candidates(page_response.text, str(page_response.url))
            if len(candidates) < count:
                raise PipelineError(
                    f"La ficha solo expuso {len(candidates)} imagen(es); se necesitan {count}."
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            downloaded: list[Path] = []
            downloaded_urls: list[str] = []
            for image_url in candidates:
                if len(downloaded) >= count:
                    break
                try:
                    image_response = client.get(
                        image_url,
                        headers={"Accept": "image/avif,image/webp,image/jpeg,image/png,image/*"},
                    )
                    image_response.raise_for_status()
                except httpx.HTTPError:
                    continue
                content_type = image_response.headers.get("content-type", "").split(";", 1)[0].lower()
                extension = mimetypes.guess_extension(content_type or "") or Path(urlparse(image_url).path).suffix.lower()
                if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
                    extension = ".jpg"
                if not image_response.content or len(image_response.content) > 20 * 1024 * 1024:
                    continue
                image_path = output_dir / f"{len(downloaded) + 1:02d}-prenda{extension}"
                image_path.write_bytes(image_response.content)
                downloaded.append(image_path)
                downloaded_urls.append(image_url)

            if len(downloaded) < count:
                raise PipelineError(
                    f"Solo se pudieron descargar {len(downloaded)} de {count} imágenes desde la ficha de Falabella."
                )
            (output_dir / "source.json").write_text(
                json.dumps(
                    {"product_url": product_url, "image_urls": downloaded_urls},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return downloaded
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            raise PipelineError(
                "Falabella rechazó la consulta automática (403). No se intenta evadir el bloqueo: "
                "usa cuatro imágenes locales en --garment-dir."
            ) from exc
        raise PipelineError(f"No se pudo consultar la ficha del producto: {exc}") from exc
    except httpx.HTTPError as exc:
        raise PipelineError(f"No se pudo consultar la ficha del producto: {exc}") from exc


def gcs_bucket() -> str:
    bucket = os.getenv("GEMINI_GCS_BUCKET", "").strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    return bucket.rstrip("/")


def cloudinary_credentials() -> tuple[str, str, str]:
    """Parse CLOUDINARY_URL without exposing its API secret."""

    raw_url = os.getenv("CLOUDINARY_URL", "").strip()
    if not raw_url:
        raise PipelineError("Falta CLOUDINARY_URL en el archivo .env")

    parsed = urlparse(raw_url)
    cloud_name = unquote(parsed.hostname or "")
    api_key = unquote(parsed.username or "")
    api_secret = unquote(parsed.password or "")
    if parsed.scheme != "cloudinary" or not cloud_name or not api_key or not api_secret:
        raise PipelineError(
            "CLOUDINARY_URL debe tener el formato cloudinary://API_KEY:API_SECRET@CLOUD_NAME"
        )
    return cloud_name, api_key, api_secret


def upload_video_to_cloudinary(video_path: Path) -> str:
    """Upload a local MP4 and return its public HTTPS delivery URL."""

    cloud_name, api_key, api_secret = cloudinary_credentials()
    timestamp = str(int(time.time()))
    public_id = f"instagram-reels/{video_path.stem}-{timestamp}"
    signed_params = {"public_id": public_id, "timestamp": timestamp}
    signature_base = "&".join(
        f"{key}={signed_params[key]}" for key in sorted(signed_params)
    )
    signature = hashlib.sha1(
        f"{signature_base}{api_secret}".encode("utf-8")
    ).hexdigest()
    endpoint = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
    form_data = {
        "api_key": api_key,
        "timestamp": timestamp,
        "public_id": public_id,
        "signature": signature,
    }

    try:
        with video_path.open("rb") as video_file:
            response = httpx.post(
                endpoint,
                data=form_data,
                files={"file": (video_path.name, video_file, "video/mp4")},
                timeout=positive_int_env("CLOUDINARY_TIMEOUT_SECONDS", 600),
            )
    except (OSError, httpx.HTTPError) as exc:
        raise PipelineError(f"Cloudinary no respondió al subir el video: {exc}") from exc
    if response.is_error:
        raise PipelineError(f"Cloudinary falló al subir el video: {response.text}")

    try:
        result = response.json()
    except ValueError as exc:
        raise PipelineError("Cloudinary devolvió una respuesta inválida") from exc
    public_url = str(result.get("secure_url") or "").strip()
    if not public_url.startswith("https://"):
        raise PipelineError("Cloudinary no devolvió una URL HTTPS pública")
    return public_url


def gcloud_binary() -> str:
    configured = os.getenv("GCLOUD_BIN", "").strip()
    if configured:
        return configured
    return (
        shutil.which("gcloud")
        or shutil.which("gcloud.cmd")
        or str(Path.home() / "AppData" / "Local" / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd")
        or str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud.cmd")
        or "gcloud"
    )


def upload_to_gcs(path: Path, bucket: str) -> str:
    uri = f"gs://{bucket}/inputs/{path.name}"
    try:
        subprocess.run(
            [gcloud_binary(), "storage", "cp", "--quiet", str(path), uri],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PipelineError("No se encontró gcloud para subir el archivo a Cloud Storage") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise PipelineError(f"No se pudo subir {path.name} a Cloud Storage: {detail}") from exc
    return uri


def gcs_bucket() -> str:
    bucket = os.getenv("GEMINI_GCS_BUCKET", "").strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    return bucket.rstrip("/")


def cloudinary_credentials() -> tuple[str, str, str]:
    """Parse CLOUDINARY_URL without exposing its API secret."""

    raw_url = os.getenv("CLOUDINARY_URL", "").strip()
    if not raw_url:
        raise PipelineError("Falta CLOUDINARY_URL en el archivo .env")

    parsed = urlparse(raw_url)
    cloud_name = unquote(parsed.hostname or "")
    api_key = unquote(parsed.username or "")
    api_secret = unquote(parsed.password or "")
    if parsed.scheme != "cloudinary" or not cloud_name or not api_key or not api_secret:
        raise PipelineError(
            "CLOUDINARY_URL debe tener el formato cloudinary://API_KEY:API_SECRET@CLOUD_NAME"
        )
    return cloud_name, api_key, api_secret


def upload_video_to_cloudinary(video_path: Path) -> str:
    """Upload a local MP4 and return its public HTTPS delivery URL."""

    cloud_name, api_key, api_secret = cloudinary_credentials()
    timestamp = str(int(time.time()))
    public_id = f"instagram-reels/{video_path.stem}-{timestamp}"
    signed_params = {"public_id": public_id, "timestamp": timestamp}
    signature_base = "&".join(
        f"{key}={signed_params[key]}" for key in sorted(signed_params)
    )
    signature = hashlib.sha1(
        f"{signature_base}{api_secret}".encode("utf-8")
    ).hexdigest()
    endpoint = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
    form_data = {
        "api_key": api_key,
        "timestamp": timestamp,
        "public_id": public_id,
        "signature": signature,
    }

    try:
        with video_path.open("rb") as video_file:
            response = httpx.post(
                endpoint,
                data=form_data,
                files={"file": (video_path.name, video_file, "video/mp4")},
                timeout=positive_int_env("CLOUDINARY_TIMEOUT_SECONDS", 600),
            )
    except (OSError, httpx.HTTPError) as exc:
        raise PipelineError(f"Cloudinary no respondió al subir el video: {exc}") from exc
    if response.is_error:
        raise PipelineError(f"Cloudinary falló al subir el video: {response.text}")

    try:
        result = response.json()
    except ValueError as exc:
        raise PipelineError("Cloudinary devolvió una respuesta inválida") from exc
    public_url = str(result.get("secure_url") or "").strip()
    if not public_url.startswith("https://"):
        raise PipelineError("Cloudinary no devolvió una URL HTTPS pública")
    return public_url


def gcloud_binary() -> str:
    configured = os.getenv("GCLOUD_BIN", "").strip()
    if configured:
        return configured
    return (
        shutil.which("gcloud")
        or shutil.which("gcloud.cmd")
        or str(Path.home() / "AppData" / "Local" / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd")
        or str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud.cmd")
        or "gcloud"
    )


def upload_to_gcs(path: Path, bucket: str) -> str:
    uri = f"gs://{bucket}/inputs/{path.name}"
    try:
        subprocess.run(
            [gcloud_binary(), "storage", "cp", "--quiet", str(path), uri],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PipelineError("No se encontró gcloud para subir el archivo a Cloud Storage") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise PipelineError(f"No se pudo subir {path.name} a Cloud Storage: {detail}") from exc
    return uri


def to_png_file_tuple(image_path: Path, filename: str) -> tuple[str, io.BytesIO, str]:
    """Open an image, convert to RGB, and return a tuple suitable for OpenAI API file uploads."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        buf.name = f"{filename}.png"
        return (f"{filename}.png", buf, "image/png")


def crop_head_from_image(image_path: Path, output_dir: Path, crop_ratio: float = 0.22) -> Path:
    """Crop out the top portion (head/face) of an image to keep only the garment and body."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cropped_ref_{image_path.stem}.png"
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        top_offset = int(h * crop_ratio)
        cropped_img = img.crop((0, top_offset, w, h))
        cropped_img.save(out_path, format="PNG")
    print(f"Imagen de referencia recortada sin rostro ({int(crop_ratio*100)}% superior eliminado): {out_path}")
    return out_path


def generate_outfit_image(
    model_image: Path,
    garment_images: list[Path],
    mask: Path | None,
    output_path: Path,
    prompt: str,
    garment_description: str | None = None,
    background_image: Path | None = None,
) -> None:
    """Create the virtual try-on image with GPT Image 2."""

    client = OpenAI(api_key=required_env("OPENAI_API_KEY"))
    if not garment_images:
        raise PipelineError("Debes entregar al menos una imagen de la prenda")

    image_tuples = [to_png_file_tuple(model_image, "model")]
    if background_image and background_image.is_file():
        image_tuples.append(to_png_file_tuple(background_image, "background"))
    for i, path in enumerate(garment_images, 1):
        image_tuples.append(to_png_file_tuple(path, f"garment_{i}"))

    mask_tuple = to_png_file_tuple(mask, "mask") if mask else None

    final_prompt = prompt
    if garment_description:
        final_prompt = f"{prompt}\nDetalles adicionales de la prenda: {garment_description}"

    try:
        request: dict[str, Any] = {
            "model": "gpt-image-2",
            "image": image_tuples,
            "prompt": final_prompt,
            "size": "1024x1536",
            "quality": os.getenv("OPENAI_IMAGE_QUALITY", "low").strip() or "low",
        }
        if mask_tuple:
            request["mask"] = mask_tuple

        try:
            result = client.images.edit(**request)
        except Exception as exc:
            raise PipelineError(f"OpenAI no pudo generar la imagen de outfit: {exc}") from exc
    finally:
        pass

    if not result.data or not result.data[0].b64_json:
        raise PipelineError("OpenAI no devolvió una imagen de outfit")

    output_path.write_bytes(base64.b64decode(result.data[0].b64_json))


def resource_state(resource: Any) -> str:
    state = getattr(resource, "state", None)
    name = getattr(state, "name", state)
    value = getattr(name, "value", name)
    return str(value or "").upper()


def wait_for_gemini_file(client: Any, file_name: str) -> Any:
    timeout = positive_int_env("GEMINI_TIMEOUT_SECONDS", 1200)
    poll_seconds = positive_int_env("GEMINI_POLL_SECONDS", 5)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        resource = client.files.get(name=file_name)
        state = resource_state(resource)
        if state in {"ACTIVE", "SUCCEEDED", "COMPLETED"}:
            return resource
        if state in {"FAILED", "ERROR", "CANCELLED"}:
            raise PipelineError(f"Gemini no pudo procesar {file_name}: {state}")
        print(f"Gemini: {file_name} está {state or 'PROCESSING'}...")
        time.sleep(poll_seconds)

    raise PipelineError(f"Tiempo agotado esperando a Gemini: {file_name}")


def wait_for_gemini_interaction(client: Any, interaction: Any) -> Any:
    timeout = positive_int_env("GEMINI_TIMEOUT_SECONDS", 1200)
    poll_seconds = positive_int_env("GEMINI_POLL_SECONDS", 10)
    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        raise PipelineError("Gemini no devolvió un ID de interacción")

    deadline = time.monotonic() + timeout
    current = interaction
    while time.monotonic() < deadline:
        status = str(getattr(current, "status", "")).lower()
        print(f"Gemini: interacción {status or 'in_progress'}...")
        if status == "completed":
            return current
        if status in {"failed", "cancelled", "incomplete", "budget_exceeded"}:
            detail = (
                getattr(current, "error", None)
                or getattr(current, "failure_reason", None)
                or getattr(current, "reason", None)
            )
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("detail") or detail
            suffix = f" Detalle: {detail}" if detail else ""
            raise PipelineError(f"Gemini terminó la interacción con estado: {status}.{suffix}")
        time.sleep(poll_seconds)
        current = client.interactions.get(id=interaction_id)

    raise PipelineError("Tiempo agotado esperando la interacción de Gemini")


def file_name_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    last_segment = parsed.path.rstrip("/").split("/")[-1]
    # URI delivery may contain :download or query parameters.
    file_id = last_segment.split(":", 1)[0]
    if file_id.startswith("files/"):
        return file_id
    return f"files/{file_id}"


def save_gemini_video(client: Any, interaction: Any, output_path: Path) -> None:
    steps = getattr(interaction, "steps", None) or []
    for step in reversed(steps):
        step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
        if step_type == "model_output":
            content = getattr(step, "content", None) or (step.get("content") if isinstance(step, dict) else [])
            for item in content:
                item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                if item_type == "video":
                    data = getattr(item, "data", None) or (item.get("data") if isinstance(item, dict) else None)
                    if data:
                        if isinstance(data, str):
                            output_path.write_bytes(base64.b64decode(data))
                        else:
                            output_path.write_bytes(bytes(data))
                        return
                    uri = getattr(item, "uri", None) or (item.get("uri") if isinstance(item, dict) else None)
                    if uri:
                        if str(uri).startswith("gs://"):
                            subprocess.run([gcloud_binary(), "storage", "cp", "--quiet", str(uri), str(output_path)], check=True, capture_output=True, text=True)
                            return
                        wait_for_gemini_file(client, file_name_from_uri(uri))
                        video_bytes = client.files.download(file=uri)
                        if hasattr(video_bytes, "read"):
                            video_bytes = video_bytes.read()
                        output_path.write_bytes(bytes(video_bytes))
                        return

    output_video = getattr(interaction, "output_video", None)
    if output_video is None:
        raise PipelineError("Gemini no devolvió un objeto de video")

    inline_data = getattr(output_video, "data", None)
    if inline_data:
        if isinstance(inline_data, str):
            output_path.write_bytes(base64.b64decode(inline_data))
        else:
            output_path.write_bytes(bytes(inline_data))
        return

    uri = getattr(output_video, "uri", None)
    if not uri:
        raise PipelineError("Gemini no devolvió ni data ni uri para el video")

    if str(uri).startswith("gs://"):
        try:
            subprocess.run(
                [gcloud_binary(), "storage", "cp", "--quiet", str(uri), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise PipelineError("No se encontró gcloud para descargar el video generado") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise PipelineError(f"No se pudo descargar el video generado: {detail}") from exc
        return
    output_path.write_bytes(base64.b64decode(result.data[0].b64_json))


def resource_state(resource: Any) -> str:
    state = getattr(resource, "state", None)
    name = getattr(state, "name", state)
    value = getattr(name, "value", name)
    return str(value or "").upper()


def wait_for_gemini_file(client: Any, file_name: str) -> Any:
    timeout = positive_int_env("GEMINI_TIMEOUT_SECONDS", 1200)
    poll_seconds = positive_int_env("GEMINI_POLL_SECONDS", 5)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        resource = client.files.get(name=file_name)
        state = resource_state(resource)
        if state in {"ACTIVE", "SUCCEEDED", "COMPLETED"}:
            return resource
        if state in {"FAILED", "ERROR", "CANCELLED"}:
            raise PipelineError(f"Gemini no pudo procesar {file_name}: {state}")
        print(f"Gemini: {file_name} está {state or 'PROCESSING'}...")
        time.sleep(poll_seconds)

    raise PipelineError(f"Tiempo agotado esperando a Gemini: {file_name}")


def wait_for_gemini_interaction(client: Any, interaction: Any) -> Any:
    timeout = positive_int_env("GEMINI_TIMEOUT_SECONDS", 1200)
    poll_seconds = positive_int_env("GEMINI_POLL_SECONDS", 10)
    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        raise PipelineError("Gemini no devolvió un ID de interacción")

    deadline = time.monotonic() + timeout
    current = interaction
    while time.monotonic() < deadline:
        status = str(getattr(current, "status", "")).lower()
        print(f"Gemini: interacción {status or 'in_progress'}...")
        if status == "completed":
            return current
        if status in {"failed", "cancelled", "incomplete", "budget_exceeded"}:
            detail = (
                getattr(current, "error", None)
                or getattr(current, "failure_reason", None)
                or getattr(current, "reason", None)
                or str(current)
            )
            print(f"Detalle completo de error Gemini: {detail}")
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("detail") or detail
            suffix = f" Detalle: {detail}" if detail else ""
            raise PipelineError(f"Gemini terminó la interacción con estado: {status}.{suffix}")
        time.sleep(poll_seconds)
        current = client.interactions.get(id=interaction_id)

    raise PipelineError("Tiempo agotado esperando la interacción de Gemini")


def file_name_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    last_segment = parsed.path.rstrip("/").split("/")[-1]
    # URI delivery may contain :download or query parameters.
    file_id = last_segment.split(":", 1)[0]
    if file_id.startswith("files/"):
        return file_id
    return f"files/{file_id}"


def save_gemini_video(client: Any, interaction: Any, output_path: Path) -> None:
    steps = getattr(interaction, "steps", None) or []
    for step in reversed(steps):
        step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
        if step_type == "model_output":
            content = getattr(step, "content", None) or (step.get("content") if isinstance(step, dict) else [])
            for item in content:
                item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                if item_type == "video":
                    data = getattr(item, "data", None) or (item.get("data") if isinstance(item, dict) else None)
                    if data:
                        if isinstance(data, str):
                            output_path.write_bytes(base64.b64decode(data))
                        else:
                            output_path.write_bytes(bytes(data))
                        return
                    uri = getattr(item, "uri", None) or (item.get("uri") if isinstance(item, dict) else None)
                    if uri:
                        if str(uri).startswith("gs://"):
                            subprocess.run([gcloud_binary(), "storage", "cp", "--quiet", str(uri), str(output_path)], check=True, capture_output=True, text=True)
                            return
                        wait_for_gemini_file(client, file_name_from_uri(uri))
                        video_bytes = client.files.download(file=uri)
                        if hasattr(video_bytes, "read"):
                            video_bytes = video_bytes.read()
                        output_path.write_bytes(bytes(video_bytes))
                        return

    output_video = getattr(interaction, "output_video", None)
    if output_video is None:
        raise PipelineError("Gemini no devolvió un objeto de video")

    inline_data = getattr(output_video, "data", None)
    if inline_data:
        if isinstance(inline_data, str):
            output_path.write_bytes(base64.b64decode(inline_data))
        else:
            output_path.write_bytes(bytes(inline_data))
        return

    uri = getattr(output_video, "uri", None)
    if not uri:
        raise PipelineError("Gemini no devolvió ni data ni uri para el video")

    if str(uri).startswith("gs://"):
        try:
            subprocess.run(
                [gcloud_binary(), "storage", "cp", "--quiet", str(uri), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise PipelineError("No se encontró gcloud para descargar el video generado") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise PipelineError(f"No se pudo descargar el video generado: {detail}") from exc
        return

    wait_for_gemini_file(client, file_name_from_uri(uri))
    video_bytes = client.files.download(file=uri)
    if hasattr(video_bytes, "read"):
        video_bytes = video_bytes.read()
    output_path.write_bytes(bytes(video_bytes))


def generate_video(
    references: Path | list[Path],
    base_video: Path | None,
    output_path: Path,
    prompt: str,
    garment_images: list[Path] | None = None,
) -> None:
    """Animate the outfit image/references, optionally preserving motion from a source video."""

    backend = gemini_backend()
    if backend == "vertex":
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

    client = create_gemini_client(backend)
    aspect_ratio = os.getenv("GEMINI_ASPECT_RATIO", "9:16").strip() or "9:16"
    if aspect_ratio not in {"9:16", "16:9"}:
        raise PipelineError("GEMINI_ASPECT_RATIO debe ser '9:16' o '16:9'")
    duration = duration_env("GEMINI_VIDEO_DURATION", "10s")
    
    ref_list = [references] if isinstance(references, Path) else references
    inputs: list[dict[str, Any]] = []
    for ref in ref_list:
        inputs.append(
            {
                "type": "image",
                "data": encode_base64(ref),
                "mime_type": image_mime_type(ref),
            }
        )
    if garment_images:
        for g_img in garment_images:
            inputs.append(
                {
                    "type": "image",
                    "data": encode_base64(g_img),
                    "mime_type": image_mime_type(g_img),
                }
            )

    task_mode = os.getenv("GEMINI_VIDEO_TASK", "image_to_video").strip() or "image_to_video"

    if base_video and task_mode == "edit":
        inputs.append(
            {
                "type": "video",
                "data": encode_base64(base_video),
                "mime_type": "video/mp4",
            }
        )

    inputs.append({"type": "text", "text": prompt})

    request: dict[str, Any] = {
        "model": "gemini-omni-flash-preview",
        "input": inputs,
        "generation_config": {
            "video_config": {
                "task": task_mode
            }
        },
    }

    video_response_format: dict[str, Any] = {"type": "video"}
    if not base_video:
        video_response_format.update(
            {
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }
        )

    if backend == "vertex":
        request["response_format"] = [video_response_format]
        request["background"] = True
    else:
        request["response_format"] = video_response_format

    try:
        if backend == "vertex":
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
        interaction = client.interactions.create(**request)
    except Exception as exc:
        if backend == "vertex":
            raise PipelineError(
                "Agent Platform no pudo crear la interacción. Verifica que "
                "GOOGLE_CLOUD_PROJECT esté configurado, que Agent Platform API "
                "esté habilitada y que exista ADC (`gcloud auth "
                "application-default login`). Detalle: "
                f"{exc}"
            ) from exc
        raise PipelineError(f"Gemini API no pudo crear la interacción: {exc}") from exc
    if backend == "vertex":
        interaction = wait_for_gemini_interaction(client, interaction)
    save_gemini_video(client, interaction, output_path)
    remove_gemini_watermark(output_path)


def remove_gemini_watermark(video_path: Path) -> None:
    """Micro-crop bottom 2.5% of frame to eliminate Gemini's SynthID watermark."""
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(str(video_path))
        w, h = clip.size
        crop_h = int(h * 0.975)
        cropped = clip.cropped(x1=0, y1=0, width=w, height=crop_h)
        scaled = cropped.resized((w, h))
        temp_out = video_path.parent / f"clean_{video_path.name}"
        scaled.write_videofile(
            str(temp_out),
            codec="libx264",
            audio_codec="aac",
            fps=clip.fps or 24,
            logger=None
        )
        clip.close()
        cropped.close()
        scaled.close()
        if temp_out.exists() and temp_out.stat().st_size > 0:
            temp_out.replace(video_path)
            print("[Pipeline] Marca de agua de Gemini eliminada mediante micro-recorte limpio.")
    except Exception as exc:
        print(f"Aviso al limpiar marca de agua: {exc}")


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                fragments.append(str(value))
    return "\n".join(fragments).strip()


def normalize_narration(text: str) -> str:
    """Make common English fashion terms speakable and prevent hard audio cuts."""

    normalized = text.strip()
    normalized = re.sub(r"\b(?:over\s*size|oversized|oversize)\b", "oversais", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?:\s*[.!?…])+\s*$", "", normalized).rstrip()
    return f"{normalized}..." if normalized else ""


def generate_script(offer_path: Path, output_path: Path, affiliate: bool = True) -> str:
    """Generate a short, fact-bound Chilean Spanish script from an offer JSON."""

    try:
        offer = json.loads(offer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"No se pudo leer la oferta JSON: {offer_path}") from exc
    if not isinstance(offer, dict):
        raise PipelineError("La oferta JSON debe ser un objeto")

    client = OpenAI(api_key=required_env("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o").strip() or "gpt-4o"
    commercial_line = (
        "Termina exactamente con una invitación a comentar la palabra clave y decir que enviarás el link. No menciones publicidad ni afiliación dentro del audio; eso se informa en el caption."
        if affiliate
        else "No menciones enlaces, afiliación ni comisiones. Termina invitando a comentar la palabra clave para pedir una revisión o seguir la cuenta."
    )
    instructions = f"""
Eres guionista de Reels de ofertas de ropa para Chile. Escribe un guion hablado
de máximo 28 palabras, con ritmo rápido y español chileno natural. Debe comenzar
con "Mira lo que encontré..." y seguir esta estructura:
"Mira lo que encontré... [prenda], de [precio anterior] bajó a [precio actual].
[urgencia o tallas solo si está confirmada]. Comenta [palabra clave] y [cierre]..."
Usa precios hablados como "35 mil" o "12 mil dos cincuenta", manteniendo el
monto exacto del JSON. No inventes precio, descuento, stock, tallas, marca,
beneficios ni disponibilidad. Escribe los términos ingleses de moda como suenan
en español para la voz: "oversize" debe ser "oversais". Conserva la palabra
clave exacta en mayúsculas, por ejemplo LOOK, porque activa los comentarios.
Si no existe palabra clave, usa LOOK. {commercial_line} Termina siempre con
tres puntos. Devuelve solo el texto hablado, sin comillas, hashtags ni acotaciones.
""".strip()
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(offer, ensure_ascii=False),
        )
    except Exception as exc:
        raise PipelineError(f"OpenAI no pudo generar el guion: {exc}") from exc

    script = normalize_narration(response_text(response))
    if not script:
        raise PipelineError("OpenAI no devolvió texto para el guion")
    output_path.write_text(script + "\n", encoding="utf-8")
    return script


def fish_api_key() -> str:
    return (
        os.getenv("FISH_API_KEY", "").strip()
        or os.getenv("FISH_AUDIO_API_KEY", "").strip()
        or required_env("FISH_API_KEY")
    )


def generate_voiceover(script: str, output_path: Path) -> None:
    """Generate MP3 speech with Fish Audio's HTTP API."""

    payload: dict[str, Any] = {
        "text": script,
        "format": "mp3",
        "normalize": True,
        "sample_rate": 44100,
        "mp3_bitrate": 128,
    }
    reference_id = (
        os.getenv("FISH_AUDIO_VOICE_ID", "").strip()
        or os.getenv("FISH_REFERENCE_ID", "").strip()
    )
    if reference_id:
        payload["reference_id"] = reference_id

    fish_model = (
        os.getenv("FISH_AUDIO_MODEL", "").strip()
        or os.getenv("FISH_MODEL", "").strip()
        or "s2.1-pro-free"
    )
    fish_url = (
        os.getenv("FISH_AUDIO_API_URL", "").strip()
        or os.getenv("FISH_TTS_URL", "").strip()
        or "https://api.fish.audio/v1/tts"
    )
    headers = {
        "Authorization": f"Bearer {fish_api_key()}",
        "Content-Type": "application/json",
        "model": fish_model,
    }
    try:
        response = httpx.post(
            fish_url,
            headers=headers,
            json=payload,
            timeout=positive_int_env(
                "FISH_AUDIO_TIMEOUT_SECONDS",
                positive_int_env("FISH_TIMEOUT_SECONDS", 900),
            ),
        )
    except httpx.HTTPError as exc:
        raise PipelineError(f"Fish Audio no respondió: {exc}") from exc
    if response.is_error:
        raise PipelineError(f"Fish Audio falló: {response.text}")
    if not response.content:
        raise PipelineError("Fish Audio devolvió un archivo de audio vacío")
    output_path.write_bytes(response.content)


def words_from_narration(text: str, audio_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Generate word-level timestamps directly from narration text and audio duration without Groq."""
    try:
        from moviepy import AudioFileClip
    except ImportError as exc:
        raise PipelineError("Falta moviepy; ejecuta pip install -r requirements.txt") from exc

    raw_words = [w.strip() for w in re.split(r"\s+", text) if w.strip()]
    if not raw_words:
        raise PipelineError("El texto de narración está vacío")

    audio = AudioFileClip(str(audio_path))
    duration = float(audio.duration)
    audio.close()

    time_per_word = duration / len(raw_words)
    words: list[dict[str, Any]] = []
    for idx, word in enumerate(raw_words):
        start = round(idx * time_per_word, 2)
        end = round((idx + 1) * time_per_word, 2)
        words.append({"word": word, "start": start, "end": end})

    output_path.write_text(json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return words


def transcription_words(media_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Transcribe one audio track and persist word-level timestamps."""

    try:
        from groq import Groq
    except ImportError as exc:
        raise PipelineError("Falta la dependencia groq; ejecuta pip install -r requirements.txt") from exc

    client = Groq(api_key=required_env("GROQ_API_KEY"))
    try:
        with media_path.open("rb") as media_file:
            transcription = client.audio.transcriptions.create(
                file=(media_path.name, media_file.read()),
                model=os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"),
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language=os.getenv("GROQ_LANGUAGE", "es"),
                temperature=0.0,
            )
    except Exception as exc:
        raise PipelineError(f"Groq no pudo transcribir {media_path.name}: {exc}") from exc

    raw_words = getattr(transcription, "words", None) or []
    words: list[dict[str, Any]] = []
    for raw_word in raw_words:
        if isinstance(raw_word, dict):
            word = raw_word.get("word")
            start = raw_word.get("start")
            end = raw_word.get("end")
        else:
            word = getattr(raw_word, "word", None)
            start = getattr(raw_word, "start", None)
            end = getattr(raw_word, "end", None)
        if word and start is not None and end is not None:
            words.append({"word": str(word), "start": float(start), "end": float(end)})

    if not words:
        raise PipelineError("Groq no devolvió marcas de tiempo por palabra")
    output_path.write_text(json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return words


def caption_chunks(words: list[dict[str, Any]], max_words: int = 4) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        if len(current) >= max_words or (float(word["end"]) - float(current[0]["start"])) >= 1.6:
            chunks.append(
                {
                    "text": " ".join(item["word"] for item in current),
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                }
            )
            current = []
    if current:
        chunks.append(
            {
                "text": " ".join(item["word"] for item in current),
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
            }
        )
    return chunks


def caption_font() -> str:
    configured = os.getenv("CAPTION_FONT", "").strip()
    candidates = [
        configured,
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise PipelineError(
        "No se encontró una fuente para subtítulos. Define CAPTION_FONT con la ruta a un archivo .ttf."
    )


def attach_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Replace the video's audio track with the generated narration."""

    try:
        from moviepy import AudioFileClip, VideoFileClip
    except ImportError as exc:
        raise PipelineError("Falta moviepy; ejecuta pip install -r requirements.txt") from exc

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))
    combined = None
    trimmed_audio = None
    try:
        duration = min(float(video.duration), float(audio.duration))
        trimmed_video = video.subclipped(0, duration)
        trimmed_audio = audio.subclipped(0, duration)
        combined = trimmed_video.with_audio(trimmed_audio)
        combined.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=video.fps or 30,
            logger=None,
        )
    finally:
        if combined:
            combined.close()
        if trimmed_audio:
            trimmed_audio.close()
        audio.close()
        video.close()


def render_hyperframes_subtitles(video_path: Path, words: list[dict[str, Any]], output_path: Path) -> None:
    """Render aesthetic captions using HyperFrames with Google Font 'Outfit'."""
    try:
        html_dir = output_path.parent / f"hyperframes-{int(time.time())}"
        html_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy video locally to avoid file:// security blocks in Puppeteer
        local_bg = html_dir / "bg.mp4"
        shutil.copy(video_path, local_bg)
        
        html_file = html_dir / "index.html"
        words_json = json.dumps(words, ensure_ascii=False)

        html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>HyperFrames Captions - Outfit</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;800;900&display=swap');

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body, html {{
            width: 1080px;
            height: 1920px;
            overflow: hidden;
            background: #000;
            font-family: 'Outfit', sans-serif;
        }}

        #bg-video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 1080px;
            height: 1920px;
            object-fit: cover;
        }}

        .caption-container {{
            position: absolute;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            width: 920px;
            text-align: center;
            z-index: 10;
        }}

        .caption-box {{
            display: inline-block;
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.18);
            border-radius: 26px;
            padding: 22px 38px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.65);
        }}

        .word {{
            display: inline-block;
            font-size: 52px;
            font-weight: 900;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin: 0 7px;
            transition: all 0.08s ease-in-out;
            text-shadow: 0 4px 12px rgba(0, 0, 0, 0.85);
        }}

        .word.active {{
            color: #FFE600;
            transform: scale(1.18);
            text-shadow: 0 0 24px rgba(255, 230, 0, 0.95), 0 4px 14px rgba(0, 0, 0, 0.9);
        }}
    </style>
</head>
<body>
    <video id="bg-video" src="./bg.mp4" data-start="0" autoplay muted></video>
    <div class="caption-container">
        <div class="caption-box" id="caption-box"></div>
    </div>

    <script>
        const words = {words_json};
        const container = document.getElementById('caption-box');
        const video = document.getElementById('bg-video');

        function updateCaptions() {{
            const curTime = video.currentTime;
            const activeWords = words.filter(w => curTime >= w.start - 0.05 && curTime <= w.end + 0.15);
            if (activeWords.length > 0) {{
                container.innerHTML = activeWords.map(w => {{
                    const isActive = curTime >= w.start && curTime <= w.end;
                    return `<span class="word ${{isActive ? 'active' : ''}}">${{w.word}}</span>`;
                }}).join(' ');
            }} else {{
                container.innerHTML = '';
            }}
            requestAnimationFrame(updateCaptions);
        }}
        video.addEventListener('timeupdate', updateCaptions);
        requestAnimationFrame(updateCaptions);
    </script>
</body>
</html>
"""
        html_file.write_text(html_content, encoding="utf-8")
        
        npx_bin = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
        cmd = [npx_bin, "-y", "hyperframes", "render", str(html_dir), "-o", str(output_path), "--resolution=portrait"]
        print(f"Renderizando subtítulos aesthetic (Outfit) con HyperFrames...")
        res = subprocess.run(cmd, capture_output=True, text=True, shell=(sys.platform == "win32"), timeout=60)
        if res.returncode != 0:
            print(f"HyperFrames aviso: {res.stderr or res.stdout}. Usando MoviePy fallback...")
            render_subtitles(video_path, words, output_path)
    except Exception as exc:
        print(f"Aviso al renderizar con HyperFrames: {exc}. Usando fallback MoviePy...")
        render_subtitles(video_path, words, output_path)


def render_subtitles(video_path: Path, words: list[dict[str, Any]], output_path: Path) -> None:
    """Burn readable captions into a vertical video using MoviePy v2."""

    try:
        from moviepy import CompositeVideoClip, TextClip, VideoFileClip
    except ImportError as exc:
        raise PipelineError("Falta moviepy; ejecuta pip install -r requirements.txt") from exc

    video = VideoFileClip(str(video_path))
    caption_clips: list[Any] = []
    final = None
    try:
        max_words = positive_int_env("CAPTION_MAX_WORDS", 4)
        size = (int(video.w * 0.88), int(video.h * 0.18))
        font_size = max(42, int(video.h * 0.045))
        caption_position = os.getenv("CAPTION_POSITION", "top").strip().lower()
        if caption_position == "center":
            position: tuple[str, str | int] = ("center", "center")
        elif caption_position == "bottom":
            position = ("center", int(video.h * 0.74))
        else:
            # Positioned higher up near top edge of the 9:16 frame
            position = ("center", int(video.h * 0.04))
        for chunk in caption_chunks(words, max_words=max_words):
            start = max(0.0, float(chunk["start"]))
            end = min(float(video.duration), float(chunk["end"]))
            if end <= start:
                continue
            caption = TextClip(
                font=caption_font(),
                text=chunk["text"],
                font_size=font_size,
                color="white",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=size,
                text_align="center",
            ).with_start(start).with_duration(end - start).with_position(position)
            caption_clips.append(caption)

        final = CompositeVideoClip([video, *caption_clips], size=video.size)
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=video.fps or 30,
            logger=None,
        )
    finally:
        if final:
            final.close()
        for caption in caption_clips:
            caption.close()
        video.close()


def normalize_base_video(video_path: Path, out_dir: Path, stamp: str) -> Path:
    """Trim source footage to Gemini Omni Flash's 10-second input limit."""

    try:
        from moviepy import VideoFileClip
    except ImportError as exc:
        raise PipelineError("Falta moviepy para validar el video base") from exc

    clip = VideoFileClip(str(video_path))
    try:
        duration = float(clip.duration or 0)
        if duration <= 10.0:
            return video_path

        trimmed_path = out_dir / f"source-trimmed-{stamp}.mp4"
        trimmed = clip.subclipped(0, 10.0)
        try:
            trimmed.write_videofile(
                str(trimmed_path),
                codec="libx264",
                audio=clip.audio is not None,
                audio_codec="aac" if clip.audio is not None else None,
                fps=clip.fps or 30,
                logger=None,
            )
        finally:
            trimmed.close()
        print(
            f"Video base de {duration:.2f}s recortado a 10.00s para cumplir el límite de Gemini: {trimmed_path}"
        )
        return trimmed_path
    finally:
        clip.close()


def meta_endpoint(path: str) -> str:
    host = os.getenv("META_GRAPH_BASE_URL", "https://graph.instagram.com").rstrip("/")
    version = required_env("META_API_VERSION")
    return f"{host}/{version}/{path.lstrip('/')}"


def meta_json(response: httpx.Response, action: str) -> dict[str, Any]:
    if response.is_error:
        raise PipelineError(f"Instagram falló al {action}: {response.text}")
    try:
        data = response.json()
    except ValueError as exc:
        raise PipelineError(f"Instagram devolvió una respuesta inválida al {action}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise PipelineError(f"Instagram falló al {action}: {data['error']}")
    return data


def record_published_reel(media_id: str, product_url: str, title: str = "") -> None:
    """Save media_id to product mapping in published_history.json & Cloudflare KV."""
    history_file = Path("published_history.json")
    history = {}
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = {}
    history[str(media_id)] = {
        "product_url": product_url,
        "title": title,
        "published_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cloudflare KV Sync
    worker_url = os.getenv("CLOUDFLARE_WORKER_URL", "").strip()
    secret = os.getenv("API_SECRET_KEY", "antigravity_api_secret").strip()
    if worker_url:
        try:
            endpoint = f"{worker_url.rstrip('/')}/api/record-reel"
            headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
            payload = {"media_id": str(media_id), "product_url": product_url, "title": title}
            with httpx.Client(timeout=10) as client:
                r = client.post(endpoint, json=payload, headers=headers)
                print(f"[Cloudflare Sync] Status: {r.status_code}")
        except Exception as exc:
            print(f"[Cloudflare Sync] Aviso al sincronizar con Cloudflare Worker: {exc}")


def publish_reel(public_video_url: str, caption: str) -> str:
    """Create, poll, and publish an Instagram Reel container."""

    access_token = required_env("INSTAGRAM_ACCESS_TOKEN")
    instagram_user_id = required_env("INSTAGRAM_USER_ID")
    timeout = positive_int_env("INSTAGRAM_TIMEOUT_SECONDS", 600)
    poll_seconds = positive_int_env("INSTAGRAM_POLL_SECONDS", 10)

    if not public_video_url.startswith("https://"):
        raise PipelineError("Instagram necesita una URL HTTPS pública para el video")

    with httpx.Client(timeout=60) as client:
        create_response = client.post(
            meta_endpoint(f"{instagram_user_id}/media"),
            data={
                "media_type": "REELS",
                "video_url": public_video_url,
                "caption": caption,
                "share_to_feed": "true",
                "access_token": access_token,
            },
        )
        container = meta_json(create_response, "crear el contenedor del Reel")
        container_id = container.get("id")
        if not container_id:
            raise PipelineError(f"Instagram no devolvió container_id: {container}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status_response = client.get(
                meta_endpoint(container_id),
                params={
                    "fields": "status_code,status",
                    "access_token": access_token,
                },
            )
            status = meta_json(status_response, "consultar el estado del Reel")
            status_code = str(status.get("status_code", "")).upper()
            print(f"Instagram: {status.get('status', status_code)}")
            if status_code == "FINISHED":
                break
            if status_code in {"ERROR", "EXPIRED"}:
                raise PipelineError(f"Instagram no pudo procesar el Reel: {status}")
            time.sleep(poll_seconds)
        else:
            raise PipelineError("Tiempo agotado esperando a Instagram")

        publish_response = client.post(
            meta_endpoint(f"{instagram_user_id}/media_publish"),
            data={
                "creation_id": container_id,
                "access_token": access_token,
            },
        )
        published = meta_json(publish_response, "publicar el Reel")
        media_id = published.get("id")
        if not media_id:
            raise PipelineError(f"Instagram no devolvió media_id: {published}")
        return str(media_id)


def tiktok_endpoint(path: str) -> str:
    host = os.getenv("TIKTOK_API_BASE_URL", "https://open.tiktokapis.com").rstrip("/")
    return f"{host}/v2/{path.lstrip('/')}"


def tiktok_json(response: httpx.Response, action: str) -> dict[str, Any]:
    if response.is_error:
        raise PipelineError(f"TikTok falló al {action}: {response.text}")
    try:
        data = response.json()
    except ValueError as exc:
        raise PipelineError(f"TikTok devolvió una respuesta inválida al {action}") from exc
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and str(error.get("code", "ok")).lower() not in {"", "ok"}:
        raise PipelineError(f"TikTok falló al {action}: {error}")
    if not isinstance(data, dict):
        raise PipelineError(f"TikTok devolvió un formato inesperado al {action}")
    return data


def publish_tiktok(public_video_url: str, title: str) -> str:
    """Direct-post a public MP4 URL to TikTok and poll its publish status."""

    access_token = required_env("TIKTOK_ACCESS_TOKEN")
    if not public_video_url.startswith("https://"):
        raise PipelineError("TikTok necesita una URL HTTPS pública del video")
    title = title.strip()
    if not title:
        raise PipelineError("TikTok necesita un título/caption no vacío")

    timeout = positive_int_env("TIKTOK_TIMEOUT_SECONDS", 600)
    poll_seconds = positive_int_env("TIKTOK_POLL_SECONDS", 10)
    privacy_level = os.getenv("TIKTOK_PRIVACY_LEVEL", "PUBLIC_TO_EVERYONE").strip()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    try:
        with httpx.Client(timeout=60) as client:
            creator_response = client.post(
                tiktok_endpoint("post/publish/creator_info/query/"),
                headers=headers,
            )
            creator_data = tiktok_json(creator_response, "consultar el creador")
            creator = creator_data.get("data", {})
            privacy_options = creator.get("privacy_level_options", [])
            if privacy_options and privacy_level not in privacy_options:
                raise PipelineError(
                    f"TIKTOK_PRIVACY_LEVEL={privacy_level} no está disponible para la cuenta. "
                    f"Opciones: {', '.join(privacy_options)}"
                )

            init_response = client.post(
                tiktok_endpoint("post/publish/video/init/"),
                headers=headers,
                json={
                    "post_info": {
                        "title": title[:2200],
                        "privacy_level": privacy_level,
                        "disable_duet": boolean_env("TIKTOK_DISABLE_DUET"),
                        "disable_comment": boolean_env("TIKTOK_DISABLE_COMMENT"),
                        "disable_stitch": boolean_env("TIKTOK_DISABLE_STITCH"),
                        "video_cover_timestamp_ms": positive_int_env(
                            "TIKTOK_COVER_TIMESTAMP_MS", 1000
                        ),
                    },
                    "source_info": {
                        "source": "PULL_FROM_URL",
                        "video_url": public_video_url,
                    },
                },
            )
            init_data = tiktok_json(init_response, "iniciar la publicación")
            publish_id = str(init_data.get("data", {}).get("publish_id", ""))
            if not publish_id:
                raise PipelineError(f"TikTok no devolvió publish_id: {init_data}")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                status_response = client.post(
                    tiktok_endpoint("post/publish/status/fetch/"),
                    headers=headers,
                    json={"publish_id": publish_id},
                )
                status_data = tiktok_json(status_response, "consultar el estado")
                status_info = status_data.get("data", {})
                status = str(status_info.get("status", "")).upper()
                print(f"TikTok: {status or 'PROCESSING'}")
                if status == "PUBLISH_COMPLETE":
                    post_ids = status_info.get("publicaly_available_post_id") or []
                    return str(post_ids[0]) if post_ids else publish_id
                if status == "FAILED":
                    reason = status_info.get("fail_reason", "desconocido")
                    raise PipelineError(f"TikTok no pudo publicar el video: {reason}")
                time.sleep(poll_seconds)
    except httpx.HTTPError as exc:
        raise PipelineError(f"TikTok no respondió: {exc}") from exc

    raise PipelineError("Tiempo agotado esperando a TikTok")


def auto_discover_knasta_url(history_file: Path = Path("published_history.json"), query: str = "polera") -> str:
    """Automatically find the top unpublished deal on Knasta for autonomous execution."""
    history = set()
    if history_file.is_file():
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Add both media_id mapping values and published_urls list
                for item in data.values():
                    if isinstance(item, dict) and item.get("product_url"):
                        history.add(item.get("product_url").strip())
                for u in data.get("published_urls", []):
                    history.add(u.strip())
        except Exception:
            pass

    # Priority category: Falabella Ropa Mujer with >= 30% discount
    category_urls = [
        "https://knasta.cl/results/mujer/ropa-mujer?d=-30&partners=falabella",
        f"https://knasta.cl/results?q={query}",
        "https://knasta.cl/results?q=chaqueta",
        "https://knasta.cl/results?q=vestido"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    for cat_url in category_urls:
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
                resp = client.get(cat_url)
                resp.raise_for_status()
                html = resp.text

            detail_urls = re.findall(r'href="(/detail/[^"]+)"', html)
            unique_links = list(dict.fromkeys(detail_urls))
            for rel_path in unique_links:
                full_url = f"https://knasta.cl{rel_path}"
                if full_url not in history:
                    print(f"Auto-descubierta oferta destacada de Falabella en Knasta: {full_url}")
                    return full_url
        except Exception as exc:
            print(f"Advertencia al consultar Knasta URL '{cat_url}': {exc}")

    raise PipelineError("No se encontraron ofertas nuevas en Knasta para publicar.")


def save_published_history(history_file: Path, url: str) -> None:
    history = []
    if history_file.is_file():
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            history = data.get("published_urls", [])
        except Exception:
            pass
    if url not in history:
        history.append(url)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(json.dumps({"published_urls": history}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera y publica automáticamente Reels de ofertas de ropa en Instagram y TikTok."
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Descubrir y procesar automáticamente la mejor oferta nueva disponible en Knasta sin intervención humana",
    )
    parser.add_argument("--model-image", type=Path, help="Foto del modelo/persona")
    parser.add_argument(
        "--garment-image",
        type=Path,
        action="append",
        help="Foto de la prenda; repite la opción para varias vistas (hasta 9)",
    )
    parser.add_argument(
        "--garment-dir",
        type=Path,
        help="Carpeta con las imágenes de la prenda; se leen en orden alfabético",
    )
    parser.add_argument(
        "--garment-description",
        help="Descripción opcional de la prenda para enriquecer el prompt de generación",
    )
    parser.add_argument(
        "--product-url",
        help="Ficha HTTPS oficial de Falabella; descarga cuatro fotos de la prenda antes de generar",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Imagen existente que Gemini usará como reemplazo en el video",
    )
    parser.add_argument(
        "--crop-garment-head",
        action="store_true",
        default=True,
        help="Recortar el rostro/cabeza superior de las imágenes extraídas de la prenda/tienda",
    )
    parser.add_argument(
        "--crop-reference-head",
        action="store_true",
        default=False,
        help="Recortar el rostro/cabeza superior de las imágenes de referencia antes de enviar a Gemini",
    )
    parser.add_argument(
        "--crop-head-ratio",
        type=float,
        default=0.22,
        help="Porcentaje superior a recortar de la imagen (por defecto 0.22)",
    )
    parser.add_argument("--mask", type=Path, help="Máscara PNG opcional para la zona de ropa")
    parser.add_argument("--base-video", type=Path, help="Video de movimiento para video-to-video")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--outfit-prompt", default=DEFAULT_OUTFIT_PROMPT)
    parser.add_argument(
        "--video-prompt",
        default=os.getenv("GEMINI_VIDEO_PROMPT", "").strip() or DEFAULT_VIDEO_PROMPT,
    )
    parser.add_argument("--offer-json", type=Path, help="Oferta JSON para generar un guion comercial")
    parser.add_argument("--generate-script", action="store_true", help="Generar guion con OpenAI desde --offer-json")
    parser.add_argument("--script-file", type=Path, help="Archivo de texto con la narración")
    parser.add_argument("--narration-text", help="Texto de narración para Fish Audio")
    parser.add_argument("--voiceover", action="store_true", help="Generar y reemplazar el audio con Fish Audio")
    parser.add_argument("--subtitles", action="store_true", help="Transcribir y quemar subtítulos con Groq/MoviePy")
    parser.add_argument("--publish", action="store_true", help="Publicar en Instagram después de generar")
    parser.add_argument("--publish-tiktok", action="store_true", help="Publicar en TikTok después de generar")
    parser.add_argument(
        "--publish-both",
        action="store_true",
        help="Publicar el mismo MP4 en Instagram y TikTok después de generar",
    )
    parser.add_argument("--public-url", help="URL HTTPS pública del MP4 para Instagram/TikTok")
    parser.add_argument("--caption", help="Caption del Reel")
    parser.add_argument(
        "--organic-test",
        action="store_true",
        help="Modo de prueba sin enlace afiliado ni declaración de comisión",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.script_file and args.narration_text:
            raise PipelineError("Usa --script-file o --narration-text, no ambos")
        if args.generate_script and not args.offer_json and not args.product_url:
            raise PipelineError("--generate-script necesita --offer-json")

        if args.reference_image and (
            args.model_image or args.garment_image or args.garment_dir or args.product_url
        ):
            raise PipelineError(
                "Usa --reference-image solo, o usa --model-image junto con las imágenes de la prenda; no mezcles los dos modos."
            )
        if args.product_url and (args.garment_image or args.garment_dir):
            raise PipelineError("Usa --product-url o las imágenes locales de la prenda, no ambos.")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if args.auto_discover and not args.product_url:
            args.product_url = auto_discover_knasta_url(args.out_dir / "published_history.json")

        if args.reference_image:
            video_references = [
                validate_input(path, "imagen de referencia")
                for path in args.reference_image
            ]
        else:
            model_path = args.model_image or default_download_asset(
                "MODEL_IMAGE_PATH", DEFAULT_MODEL_IMAGE
            )
            model_image = validate_input(model_path, "modelo")
            if args.product_url:
                scraped_garment_dir = args.out_dir / f"scraped-garments-{stamp}"
                print(f"Descargando fotos del producto desde: {args.product_url}")
                garment_images = download_product_images(args.product_url, scraped_garment_dir)
                if not args.offer_json and (scraped_garment_dir / "offer.json").is_file():
                    args.offer_json = scraped_garment_dir / "offer.json"
                if args.voiceover and not args.narration_text and not args.script_file and args.offer_json:
                    args.generate_script = True
            else:
                garment_dir = args.garment_dir
                if garment_dir is None and not args.garment_image:
                    garment_dir = default_download_asset("GARMENT_DIR", DEFAULT_GARMENT_DIR)
                garment_images = collect_garment_images(args.garment_image, garment_dir)

            if args.crop_garment_head and garment_images:
                print("Recortando rostro/cabeza de las imágenes de la prenda...")
                garment_images = [
                    crop_head_from_image(img, args.out_dir, crop_ratio=args.crop_head_ratio)
                    for img in garment_images
                ]

        if args.generate_script and not args.offer_json:
            raise PipelineError(
                "--generate-script no encontró offer.json. Usa --offer-json o una ficha "
                "de producto que exponga datos de oferta."
            )

        mask = validate_input(args.mask, "máscara") if args.mask else None
        if args.base_video:
            base_video = validate_input(args.base_video, "video base")
        if args.base_video:
            base_video = validate_input(args.base_video, "video base")
        else:
            default_video = default_download_asset("BASE_VIDEO_PATH", DEFAULT_BASE_VIDEO)
            base_video = validate_input(default_video, "video base") if default_video.is_file() else None

        if base_video:
            base_video = normalize_base_video(base_video, args.out_dir, stamp)
        video_path = args.out_dir / f"reel-{stamp}.mp4"

        if args.reference_image:
            print(
                "1/2 Usando imágenes de referencia existentes: "
                + ", ".join(str(path) for path in video_references)
            )
        else:
            outfit_path = args.out_dir / f"outfit-{stamp}.png"
            print(
                f"1/3 Generando outfit con GPT Image 2 usando {len(garment_images)} vista(s) de la prenda..."
            )
            bg_path_env = os.getenv("BACKGROUND_IMAGE_PATH", "").strip()
            bg_image = Path(bg_path_env) if bg_path_env and Path(bg_path_env).is_file() else None
            generate_outfit_image(
                model_image,
                garment_images,
                mask,
                outfit_path,
                args.outfit_prompt,
                garment_description=args.garment_description,
                background_image=bg_image,
            )
            print(f"Imagen guardada en: {outfit_path}")
            video_references = [outfit_path]

        if args.crop_reference_head:
            video_references = [
                crop_head_from_image(ref, args.out_dir, crop_ratio=args.crop_head_ratio)
                for ref in video_references
            ]

        print(
            "2/2 Generando video con Gemini Omni Flash..."
            if args.reference_image
            else "2/3 Generando video con Gemini Omni Flash..."
        )
        generate_video(
            video_references,
            base_video,
            video_path,
            args.video_prompt,
            garment_images=garment_images if not args.reference_image else None,
        )
        final_video_path = video_path
        print(f"Video guardado en: {final_video_path}")

        narration = args.narration_text.strip() if args.narration_text else ""
        if args.script_file:
            try:
                narration = args.script_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise PipelineError(f"No se pudo leer el guion: {args.script_file}") from exc
        if args.generate_script:
            script_path = args.out_dir / f"script-{stamp}.txt"
            narration = generate_script(
                args.offer_json,
                script_path,
                affiliate=not args.organic_test,
            )
            print(f"Guion guardado en: {script_path}")

        narration = normalize_narration(narration)

        if args.voiceover:
            if not narration:
                raise PipelineError(
                    "--voiceover necesita --narration-text, --script-file o --generate-script"
                )
            voice_path = args.out_dir / f"voice-{stamp}.mp3"
            print("Generando voz con Fish Audio...")
            generate_voiceover(narration, voice_path)
            voiced_video_path = args.out_dir / f"reel-voice-{stamp}.mp4"
            attach_audio(final_video_path, voice_path, voiced_video_path)
            final_video_path = voiced_video_path
            print(f"Video con voz guardado en: {final_video_path}")

        if args.subtitles:
            transcription_source = voice_path if args.voiceover else final_video_path
            transcript_path = args.out_dir / f"transcript-{stamp}.json"
            if narration:
                print("Generando marcas de tiempo desde el guion (sin usar Groq)...")
                words = words_from_narration(narration, transcription_source, transcript_path)
            else:
                print(f"Transcribiendo {transcription_source.name} con Groq...")
                words = transcription_words(transcription_source, transcript_path)
            subtitled_video_path = args.out_dir / f"reel-captioned-{stamp}.mp4"
            render_hyperframes_subtitles(final_video_path, words, subtitled_video_path)
            final_video_path = subtitled_video_path
            print(f"Video con subtítulos guardado en: {final_video_path}")

        publish_instagram = args.publish or args.publish_both
        publish_tiktok_requested = args.publish_tiktok or args.publish_both
        if publish_instagram or publish_tiktok_requested:
            public_url = (args.public_url or "").strip()
            if publish_instagram and not public_url:
                public_url = os.getenv("INSTAGRAM_VIDEO_URL", "").strip()
            if publish_tiktok_requested and not public_url:
                public_url = os.getenv("TIKTOK_VIDEO_URL", "").strip()
            if not public_url and os.getenv("CLOUDINARY_URL", "").strip():
                print(f"Subiendo {final_video_path.name} a Cloudinary...")
                public_url = upload_video_to_cloudinary(final_video_path)
                print("Video subido a Cloudinary y listo para publicación.")
            if not public_url:
                raise PipelineError(
                    "Para publicar usa --public-url o completa INSTAGRAM_VIDEO_URL/TIKTOK_VIDEO_URL en .env. "
                    "La URL debe ser pública y HTTPS."
                )
            disclosure = (
                os.getenv(
                    "ORGANIC_DISCLOSURE",
                    "Contenido de prueba generado con IA. Verifica precio y stock en la tienda.",
                ).strip()
                if args.organic_test
                else os.getenv(
                    "AFFILIATE_DISCLOSURE",
                    "Enlace de afiliado: puedo recibir una comisión si compras.",
                ).strip()
            )
            caption = args.caption or os.getenv("INSTAGRAM_CAPTION", "").strip()
            caption = f"{caption}\n\n{disclosure}" if caption else disclosure
            if publish_instagram:
                print(f"Publicando {final_video_path.name} en Instagram...")
                media_id = publish_reel(public_url, caption)
                print(f"Reel publicado. Media ID: {media_id}")
                prod_url = args.product_url or ""
                prod_title = "la prenda"
                offer_json_file = args.out_dir / "offer.json"
                if offer_json_file.is_file():
                    try:
                        off_data = json.loads(offer_json_file.read_text(encoding="utf-8"))
                        prod_url = off_data.get("url") or prod_url
                        prod_title = off_data.get("product_name") or prod_title
                    except Exception:
                        pass
                record_published_reel(media_id, prod_url, prod_title)
            if publish_tiktok_requested:
                tiktok_title = (
                    os.getenv("TIKTOK_TITLE", "").strip()
                    or caption.replace("\n", " ").strip()
                )
                print(f"Publicando {final_video_path.name} en TikTok...")
                tiktok_id = publish_tiktok(public_url, tiktok_title)
                print(f"Video publicado en TikTok. ID: {tiktok_id}")
        else:
            print(
                "Publicación omitida. Usa --publish, --publish-tiktok o --publish-both "
                "cuando el MP4 tenga una URL pública."
            )

        return 0
    except (PipelineError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
