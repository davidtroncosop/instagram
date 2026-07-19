"""Generate an affiliate-fashion Reel with GPT Image 2 and Gemini Omni Flash.

The script deliberately stops at a local MP4 unless --publish is passed. To
publish through Instagram, the MP4 must first be available at a public HTTPS
URL (INSTAGRAM_VIDEO_URL or --public-url).
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from google import genai
from openai import OpenAI


load_dotenv()

DEFAULT_OUTFIT_PROMPT = """
Use the first image as the model and the second image as the exact garment
reference. Dress the model in that garment. Preserve the model's face, body,
pose, hands, background, lighting, camera angle, and framing. Preserve the
garment's exact color, fabric, cut, seams, print, labels, and logos. Do not
invent text, patterns, accessories, or brand details. Create a photorealistic
vertical fashion image in 9:16 format.
""".strip()

DEFAULT_VIDEO_PROMPT = """
Use the outfit image as the exact clothing reference. Keep the garment's color,
cut, texture, print, seams, and logo unchanged. Preserve the source video's
movement, pose, framing, and lighting. Generate a realistic vertical 9:16
Instagram Reel. Use one continuous shot, no scene cuts, no fake text, and no
invented product claims.
""".strip()


class PipelineError(RuntimeError):
    """An expected, user-actionable pipeline failure."""


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PipelineError(f"Falta la variable {name} en el archivo .env")
    return value


def positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise PipelineError(f"{name} debe ser un número entero") from exc
    if value <= 0:
        raise PipelineError(f"{name} debe ser mayor que cero")
    return value


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


def generate_outfit_image(
    model_image: Path,
    garment_image: Path,
    mask: Path | None,
    output_path: Path,
    prompt: str,
) -> None:
    """Create the virtual try-on image with GPT Image 2."""

    client = OpenAI(api_key=required_env("OPENAI_API_KEY"))
    model_file = model_image.open("rb")
    garment_file = garment_image.open("rb")
    mask_file = mask.open("rb") if mask else None

    try:
        request: dict[str, Any] = {
            "model": "gpt-image-2",
            # If a mask is supplied, OpenAI applies it to the first image.
            "image": [model_file, garment_file],
            "prompt": prompt,
        }
        if mask_file:
            request["mask"] = mask_file

        result = client.images.edit(**request)
    finally:
        model_file.close()
        garment_file.close()
        if mask_file:
            mask_file.close()

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

    wait_for_gemini_file(client, file_name_from_uri(uri))
    video_bytes = client.files.download(file=uri)
    if hasattr(video_bytes, "read"):
        video_bytes = video_bytes.read()
    output_path.write_bytes(bytes(video_bytes))


def generate_video(outfit_image: Path, base_video: Path | None, output_path: Path, prompt: str) -> None:
    """Animate the outfit image, optionally preserving motion from a source video."""

    client = genai.Client(api_key=required_env("GEMINI_API_KEY"))
    inputs: list[dict[str, Any]] = []

    if base_video:
        uploaded_video = client.files.upload(file=str(base_video))
        uploaded_name = getattr(uploaded_video, "name", None)
        if not uploaded_name:
            raise PipelineError("Gemini no devolvió el nombre del video subido")
        uploaded_video = wait_for_gemini_file(client, uploaded_name)
        inputs.append({"type": "document", "uri": uploaded_video.uri})

    inputs.append(
        {
            "type": "image",
            "data": encode_base64(outfit_image),
            "mime_type": image_mime_type(outfit_image),
        }
    )
    inputs.append({"type": "text", "text": prompt})

    task = "edit" if base_video else "image_to_video"
    interaction = client.interactions.create(
        model="gemini-omni-flash-preview",
        input=inputs,
        generation_config={"video_config": {"task": task}},
        response_format={"type": "video", "delivery": "uri"},
    )
    save_gemini_video(client, interaction, output_path)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-image", required=True, type=Path, help="Foto del modelo/persona")
    parser.add_argument("--garment-image", required=True, type=Path, help="Foto o PNG de la prenda")
    parser.add_argument("--mask", type=Path, help="Máscara PNG opcional para la zona de ropa")
    parser.add_argument("--base-video", type=Path, help="Video de movimiento para video-to-video")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--outfit-prompt", default=DEFAULT_OUTFIT_PROMPT)
    parser.add_argument("--video-prompt", default=DEFAULT_VIDEO_PROMPT)
    parser.add_argument("--publish", action="store_true", help="Publicar en Instagram después de generar")
    parser.add_argument("--public-url", help="URL HTTPS pública del MP4 para Instagram")
    parser.add_argument("--caption", help="Caption del Reel")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        model_image = validate_input(args.model_image, "modelo")
        garment_image = validate_input(args.garment_image, "prenda")
        mask = validate_input(args.mask, "máscara") if args.mask else None
        base_video = validate_input(args.base_video, "video base") if args.base_video else None

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outfit_path = args.out_dir / f"outfit-{stamp}.png"
        video_path = args.out_dir / f"reel-{stamp}.mp4"

        print("1/3 Generando outfit con GPT Image 2...")
        generate_outfit_image(model_image, garment_image, mask, outfit_path, args.outfit_prompt)
        print(f"Imagen guardada en: {outfit_path}")

        print("2/3 Generando video con Gemini Omni Flash...")
        generate_video(outfit_path, base_video, video_path, args.video_prompt)
        print(f"Video guardado en: {video_path}")

        if args.publish:
            public_url = (args.public_url or os.getenv("INSTAGRAM_VIDEO_URL", "")).strip()
            if not public_url:
                raise PipelineError(
                    "Para publicar usa --public-url o completa INSTAGRAM_VIDEO_URL en .env. "
                    "La URL debe ser pública y HTTPS."
                )
            disclosure = os.getenv(
                "AFFILIATE_DISCLOSURE",
                "Enlace de afiliado: puedo recibir una comisión si compras.",
            ).strip()
            caption = args.caption or os.getenv("INSTAGRAM_CAPTION", "").strip()
            caption = f"{caption}\n\n{disclosure}" if caption else disclosure
            print("3/3 Publicando Reel en Instagram...")
            media_id = publish_reel(public_url, caption)
            print(f"Reel publicado. Media ID: {media_id}")
        else:
            print("3/3 Publicación omitida. Usa --publish cuando el MP4 tenga una URL pública.")

        return 0
    except (PipelineError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
