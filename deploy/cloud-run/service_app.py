"""Authenticated HTTP wrapper for the Instagram offer pipeline on Cloud Run."""

from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from knasta_scraper import KnastaScraperError, build_narration, scrape_knasta_offers


APP_ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(os.getenv("RUN_ROOT", "/tmp/instagram-runs"))
RUN_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Instagram Offers Pipeline")
STATE_LOCK = threading.Lock()
STATE: dict[str, dict[str, Any]] = {}
ACTIVE_RUN: str | None = None


class RunRequest(BaseModel):
    product_url: str | None = Field(default=None, description="Official Falabella product URL")
    narration_text: str | None = None
    caption: str | None = None
    publish: bool = False
    organic_test: bool = True
    model_image_url: str | None = None
    base_video_url: str | None = None
    upload_output: bool = False
    offer: dict[str, Any] | None = None
    knasta_enabled: bool | None = None
    knasta_search_terms: list[str] | None = None
    min_discount_percent: float | None = None


def require_trigger_token(authorization: str | None) -> None:
    expected = os.getenv("PIPELINE_TRIGGER_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="PIPELINE_TRIGGER_TOKEN is not configured")
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid trigger token")


def required_value(request_value: str | None, env_name: str) -> str:
    value = (request_value or os.getenv(env_name, "")).strip()
    if not value:
        raise ValueError(f"Missing {env_name}")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def resolve_offer_inputs(request: RunRequest, workdir: Path) -> tuple[str, str, str]:
    """Resolve either a static test offer or the best live Knasta offer."""

    use_knasta = (
        request.knasta_enabled
        if request.knasta_enabled is not None
        else env_bool("KNASTA_ENABLED", False)
    )
    offer = request.offer
    if offer is None and use_knasta:
        offers = scrape_knasta_offers(
            search_terms=request.knasta_search_terms,
            minimum_discount_percent=request.min_discount_percent,
            limit=1,
        )
        if not offers:
            raise ValueError("Knasta no encontró una oferta de Falabella con el descuento mínimo")
        offer = offers[0]
    if offer is not None:
        if not isinstance(offer, dict):
            raise ValueError("La oferta recibida no tiene formato JSON válido")
        if not str(offer.get("product_url") or "").startswith("https://www.falabella.com/falabella-cl/"):
            raise ValueError("La oferta recibida no contiene una URL oficial de Falabella")
        (workdir / "offer.json").write_text(
            json.dumps(offer, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        product_url = str(offer["product_url"])
        narration = (request.narration_text or build_narration(offer)).strip()
        caption = (
            request.caption
            or os.getenv("INSTAGRAM_CAPTION", "").strip()
            or f"{offer['product_name']} en Falabella... #CreadoConIA #Publicidad"
        ).strip()
        return product_url, narration, caption

    product_url = required_value(request.product_url, "PRODUCT_URL")
    narration = required_value(request.narration_text, "NARRATION_TEXT")
    caption = (request.caption or os.getenv("INSTAGRAM_CAPTION", "")).strip()
    return product_url, narration, caption


def download_asset(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError("Asset URL must use HTTPS")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
    destination.write_bytes(response.content)


def update_state(run_id: str, **values: Any) -> None:
    with STATE_LOCK:
        STATE.setdefault(run_id, {}).update(values)


def execute_pipeline(run_id: str, request: RunRequest) -> None:
    global ACTIVE_RUN
    workdir = RUN_ROOT / run_id
    output_dir = workdir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = workdir / "model-reference.jpg"
    base_video_path = workdir / "base-video.mp4"

    try:
        product_url, narration, caption = resolve_offer_inputs(request, workdir)
        if not product_url.startswith("https://www.falabella.com/"):
            raise ValueError("product_url must be an official Falabella HTTPS URL")
        model_url = required_value(request.model_image_url, "MODEL_IMAGE_URL")
        base_video_url = required_value(request.base_video_url, "BASE_VIDEO_URL")

        update_state(run_id, status="downloading_assets", updated_at=time.time())
        download_asset(model_url, model_path)
        download_asset(base_video_url, base_video_path)

        command = [
            sys.executable,
            str(APP_ROOT / "pipeline.py"),
            "--model-image",
            str(model_path),
            "--product-url",
            product_url,
            "--base-video",
            str(base_video_path),
            "--out-dir",
            str(output_dir),
            "--narration-text",
            narration,
            "--voiceover",
            "--subtitles",
        ]
        if caption:
            command.extend(["--caption", caption])
        if request.organic_test:
            command.append("--organic-test")
        if request.publish:
            command.append("--publish")
        if request.upload_output:
            command.append("--upload-cloudinary")

        update_state(run_id, status="running", updated_at=time.time())
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(
            command,
            cwd=APP_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("PIPELINE_MAX_SECONDS", "1800")),
        )
        (workdir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (workdir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
        update_state(
            run_id,
            status="completed" if result.returncode == 0 else "failed",
            return_code=result.returncode,
            output_tail=(result.stdout or "")[-4000:],
            error_tail=(result.stderr or "")[-4000:],
            updated_at=time.time(),
        )
    except Exception as exc:
        update_state(run_id, status="failed", error=str(exc), updated_at=time.time())
    finally:
        with STATE_LOCK:
            ACTIVE_RUN = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline_present": (APP_ROOT / "pipeline.py").is_file(),
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_URL", "").strip()),
        "gcp_project_configured": bool(os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()),
        "knasta_scraper_present": True,
        "active_run": ACTIVE_RUN,
    }


@app.get("/offers")
def offers(
    authorization: str | None = Header(default=None),
    terms: str | None = Query(default=None, description="Términos separados por coma"),
    min_discount: float | None = Query(default=None, ge=0, le=100),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    """Read live offers without starting the expensive media pipeline."""

    require_trigger_token(authorization)
    try:
        return {
            "offers": scrape_knasta_offers(
                search_terms=terms,
                minimum_discount_percent=min_discount,
                limit=limit,
            )
        }
    except KnastaScraperError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/run", status_code=202)
def run_pipeline(request: RunRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    global ACTIVE_RUN
    require_trigger_token(authorization)
    with STATE_LOCK:
        if ACTIVE_RUN is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A pipeline run is already active", "run_id": ACTIVE_RUN},
            )
        run_id = uuid.uuid4().hex[:12]
        ACTIVE_RUN = run_id
        STATE[run_id] = {"status": "queued", "created_at": time.time(), "updated_at": time.time()}
    threading.Thread(target=execute_pipeline, args=(run_id, request), daemon=True).start()
    return {"accepted": True, "run_id": run_id, "status_url": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
def run_status(run_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_trigger_token(authorization)
    with STATE_LOCK:
        state = STATE.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, **state}
