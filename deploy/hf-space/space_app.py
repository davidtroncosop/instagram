"""Small authenticated HTTP wrapper for the Instagram pipeline."""

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
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


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


def download_asset(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError("Asset URL must use HTTPS")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
    destination.write_bytes(response.content)


def prepare_gcp_credentials() -> None:
    """Make a JSON service-account secret available to Google ADC."""

    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    if not raw or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    credentials_path = Path("/tmp/gcp-application-credentials.json")
    credentials_path.write_text(raw, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)


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
        prepare_gcp_credentials()
        product_url = required_value(request.product_url, "PRODUCT_URL")
        if not product_url.startswith("https://www.falabella.com/"):
            raise ValueError("product_url must be an official Falabella HTTPS URL")
        narration = required_value(request.narration_text, "NARRATION_TEXT")
        model_url = required_value(request.model_image_url, "MODEL_IMAGE_URL")
        base_video_url = required_value(request.base_video_url, "BASE_VIDEO_URL")
        caption = (request.caption or os.getenv("INSTAGRAM_CAPTION", "")).strip()

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

        update_state(run_id, status="running", command=command, updated_at=time.time())
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
        status = "completed" if result.returncode == 0 else "failed"
        update_state(
            run_id,
            status=status,
            return_code=result.returncode,
            output_tail=(result.stdout or "")[-4000:],
            error_tail=(result.stderr or "")[-4000:],
            updated_at=time.time(),
        )
    except Exception as exc:  # The status endpoint must expose failures to the caller.
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
        "active_run": ACTIVE_RUN,
    }


@app.post("/run", status_code=202)
def run_pipeline(request: RunRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    global ACTIVE_RUN
    require_trigger_token(authorization)
    with STATE_LOCK:
        if ACTIVE_RUN is not None:
            raise HTTPException(status_code=409, detail={"message": "A pipeline run is already active", "run_id": ACTIVE_RUN})
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
