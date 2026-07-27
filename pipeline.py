"""Generate an affiliate-fashion Reel with Gemini Omni Flash.

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

import httpx
from dotenv import load_dotenv
from google import genai
from openai import OpenAI


load_dotenv()

DEFAULT_VIDEO_PROMPT = """
Create one continuous realistic vertical 9:16 Reel.
The source video, when provided, is authority only for timing, movement,
gestures, camera motion, and framing. The still photo of the adult woman is
authority for her identity, room, background, lighting, and camera feeling.
The following four still images are the only authority for the exact retailer
garment: front, back, collar, sleeves, fabric, seams, fit, color, print,
labels, and logos. Replace the original performer, clothing, and entire
original environment with the woman wearing that exact garment in her room.
Do not preserve the source video's background or original performer. Do not
invent text, patterns, accessories, or product claims. Keep the result
photorealistic and suitable for an Instagram Reel.
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
        # Agent Platform video Interactions use Cloud OAuth/ADC. Do not pass
        # GEMINI_API_KEY here: the Developer API key is a different backend.
        return genai.Client(vertexai=True, project=project, location=location)

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


def download_product_images(product_url: str, output_dir: Path, count: int = 4) -> list[Path]:
    """Download garment photos from a public Falabella product page.

    Knasta is used to discover the deal; this function intentionally receives
    the official retailer product URL instead of scraping Knasta's database.
    """

    parsed = urlparse(product_url)
    hostname = (parsed.hostname or "").lower()
    allowed_host = (
        hostname in {"falabella.com", "falabella.cl"}
        or hostname.endswith(".falabella.com")
        or hostname.endswith(".falabella.cl")
    )
    if parsed.scheme != "https" or not allowed_host:
        raise PipelineError(
            "--product-url debe ser una ficha HTTPS de Falabella obtenida desde Knasta; "
            "no se descarga la base de imágenes de Knasta."
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
    return shutil.which("gcloud") or str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud")


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
    reference_images: list[Path],
    base_video: Path | None,
    output_path: Path,
    prompt: str,
) -> None:
    """Generate video directly from the movement video and all still references."""

    if not reference_images:
        raise PipelineError("Debes entregar la foto de la modelo y las vistas de la prenda")
    backend = gemini_backend()
    client = create_gemini_client(backend)
    aspect_ratio = os.getenv("GEMINI_ASPECT_RATIO", "9:16").strip() or "9:16"
    if aspect_ratio not in {"9:16", "16:9"}:
        raise PipelineError("GEMINI_ASPECT_RATIO debe ser '9:16' o '16:9'")
    duration = duration_env("GEMINI_VIDEO_DURATION", "10s")
    inputs: list[dict[str, Any]] = []
    bucket = gcs_bucket() if backend == "vertex" else ""

    if base_video:
        if backend == "vertex":
            if bucket:
                inputs.append(
                    {
                        "type": "video",
                        "uri": upload_to_gcs(base_video, bucket),
                        "mime_type": "video/mp4",
                    }
                )
            else:
                inputs.append(
                    {
                        "type": "video",
                        "data": encode_base64(base_video),
                        "mime_type": "video/mp4",
                    }
                )
        else:
            # Use inline video data for AI Studio. This avoids FileService
            # restrictions on API keys while keeping short Reels self-contained.
            inputs.append(
                {
                    "type": "video",
                    "data": encode_base64(base_video),
                    "mime_type": "video/mp4",
                }
            )

    for reference_image in reference_images:
        if bucket:
            inputs.append(
                {
                    "type": "image",
                    "uri": upload_to_gcs(reference_image, bucket),
                    "mime_type": image_mime_type(reference_image),
                }
            )
        else:
            inputs.append(
                {
                    "type": "image",
                    "data": encode_base64(reference_image),
                    "mime_type": image_mime_type(reference_image),
                }
            )
    inputs.append({"type": "text", "text": prompt})

    request: dict[str, Any] = {
        "model": "gemini-omni-flash-preview",
        "input": inputs,
        "generation_config": {
            "video_config": {
                "task": (
                    "edit"
                    if base_video
                    else ("reference_to_video" if backend == "vertex" else "image_to_video")
                )
            }
        },
    }
    # Omni's uploaded-video edit schema derives the output geometry and
    # duration from the source video. The API rejects aspect_ratio (and other
    # generation controls) inside response_format for that task.
    video_response_format: dict[str, Any] = {"type": "video"}
    if backend == "vertex" and bucket:
        video_response_format.update(
            {
                "delivery": "uri",
                "gcs_uri": f"gs://{bucket}/outputs/",
            }
        )
    elif backend != "vertex":
        video_response_format["delivery"] = "uri"
    if not base_video:
        video_response_format.update(
            {
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }
        )
    if backend == "vertex":
        # Agent Platform uses a list of response formats for video requests.
        request["response_format"] = [video_response_format]
        request["background"] = True
    else:
        request["response_format"] = video_response_format

    try:
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
            # Leave a safe margin below Instagram's top controls.
            position = ("center", int(video.h * 0.08))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--product-url",
        help="Ficha HTTPS oficial de Falabella; descarga cuatro fotos de la prenda antes de generar",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Imagen existente que Gemini usará como reemplazo en el video",
    )
    parser.add_argument("--base-video", type=Path, help="Video de movimiento para video-to-video")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
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
        if args.generate_script and not args.offer_json:
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

        if args.reference_image:
            reference_images = [validate_input(args.reference_image, "imagen de referencia")]
        else:
            model_path = args.model_image or default_download_asset(
                "MODEL_IMAGE_PATH", DEFAULT_MODEL_IMAGE
            )
            model_image = validate_input(model_path, "modelo")
            if args.product_url:
                scraped_garment_dir = args.out_dir / f"scraped-garments-{stamp}"
                print(f"Descargando fotos de la prenda desde Falabella: {args.product_url}")
                garment_images = download_product_images(args.product_url, scraped_garment_dir)
            else:
                garment_dir = args.garment_dir
                if garment_dir is None and not args.garment_image:
                    garment_dir = default_download_asset("GARMENT_DIR", DEFAULT_GARMENT_DIR)
                garment_images = collect_garment_images(args.garment_image, garment_dir)
            reference_images = [model_image, *garment_images]

        if args.base_video:
            base_video = validate_input(args.base_video, "video base")
        else:
            default_video = default_download_asset("BASE_VIDEO_PATH", DEFAULT_BASE_VIDEO)
            base_video = validate_input(default_video, "video base") if default_video.is_file() else None

        if base_video:
            base_video = normalize_base_video(base_video, args.out_dir, stamp)
        video_path = args.out_dir / f"reel-{stamp}.mp4"

        print(
            "Generando video con Gemini Omni Flash usando "
            f"{len(reference_images)} referencia(s) visual(es)..."
        )
        generate_video(reference_images, base_video, video_path, args.video_prompt)
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
            print(f"Transcribiendo {transcription_source.name} con Groq...")
            words = transcription_words(transcription_source, transcript_path)
            subtitled_video_path = args.out_dir / f"reel-captioned-{stamp}.mp4"
            render_subtitles(final_video_path, words, subtitled_video_path)
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
