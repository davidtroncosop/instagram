import modal
import subprocess
import os
import dotenv
import json
from pathlib import Path

app = modal.App("instagram-reel-pipeline")

# Cargar secretos desde el archivo .env local
env_vars = dotenv.dotenv_values(".env")
clean_secrets = {k: str(v) for k, v in env_vars.items() if k and v is not None}
app_secrets = modal.Secret.from_dict(clean_secrets)

# Definir contenedor Debian con FFmpeg y dependencias necesarias
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git", "curl", "unzip")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs")
    .pip_install(
        "httpx",
        "pillow",
        "moviepy",
        "python-dotenv",
        "openai",
        "google-genai",
        "cloudinary",
        "primp"
    )
    .run_commands("echo 'rebuild-124'")
    .add_local_file(r"C:\Users\david.troncoso\AppData\Roaming\gcloud\application_default_credentials.json", remote_path="/root/gcloud_adc.json", copy=True)
    .add_local_file("pipeline.py", remote_path="/root/project/pipeline.py", copy=True)
    .add_local_file("published_history.json", remote_path="/root/project/published_history.json", copy=True)
    .add_local_dir("assets", remote_path="/root/project/assets", copy=True)
)

history_vol = modal.Volume.from_name("instagram-published-history", create_if_missing=True)

@app.function(
    image=image,
    secrets=[app_secrets],
    volumes={"/root/project/history_vol": history_vol},
    timeout=1200,
    schedule=modal.Cron("0 14,19,0 * * *")  # Cron UTC: 14:00, 19:00, 00:00 UTC -> 10:00, 15:00, 20:00 Chile (UTC-4)
)
def run_automated_pipeline(product_url: str = None, garment_files: dict[str, bytes] = None, offer_data: dict = None):
    """Ejecuta el pipeline de creación y publicación autónoma de Reels en Modal."""
    os.chdir("/root/project")
    
    vol_history = Path("/root/project/history_vol/published_history.json")
    local_history = Path("/root/project/published_history.json")
    if vol_history.is_file():
        local_history.write_bytes(vol_history.read_bytes())
        print("[Modal Container] 📚 Historial persistente cargado desde Modal Volume.")

    cmd = [
        "python", "-u", "pipeline.py",
        "--generate-script",
        "--voiceover",
        "--subtitles",
        "--organic-test",
        "--publish"
    ]

    if garment_files:
        garment_dir = Path("/tmp/garments")
        garment_dir.mkdir(parents=True, exist_ok=True)
        for fname, fbytes in garment_files.items():
            (garment_dir / fname).write_bytes(fbytes)
        cmd.extend(["--garment-dir", str(garment_dir)])
        if offer_data:
            offer_file = garment_dir / "offer.json"
            offer_file.write_text(json.dumps(offer_data, ensure_ascii=False, indent=2), encoding="utf-8")
            cmd.extend(["--offer-json", str(offer_file)])
        print(f"[Modal Container] 🚀 Iniciando flujo con {len(garment_files)} imágenes pre-extraídas...")
    elif product_url:
        cmd.extend(["--product-url", product_url])
        print(f"[Modal Container] 🚀 Iniciando flujo con URL ({product_url})...")
    else:
        cmd.append("--auto-discover")
        print("[Modal Container] 🚀 Iniciando flujo con auto-descubrimiento en la nube...")

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)

    if local_history.is_file():
        vol_history.write_bytes(local_history.read_bytes())
        history_vol.commit()
        print("[Modal Container] 💾 Historial actualizado guardado en Modal Volume.")

    if res.returncode != 0:
        print("[Modal Error]:", res.stderr)
        raise RuntimeError(f"El pipeline falló en Modal: {res.stderr}")
    return res.stdout

@app.local_entrypoint()
def main():
    print("Iniciando auto-descubrimiento y preparación local de la oferta...")
    from pipeline import auto_discover_knasta_url, download_product_images
    import tempfile

    garment_files = None
    offer_data = None
    url = None

    try:
        url = auto_discover_knasta_url()
        print(f"Oferta descubierta con éxito: {url}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            downloaded = download_product_images(url, tmp_path, count=4)
            print(f"Descargadas {len(downloaded)} imágenes de la prenda localmente.")
            
            garment_files = {}
            for p in downloaded:
                garment_files[p.name] = p.read_bytes()
            
            offer_file = tmp_path / "offer.json"
            if offer_file.exists():
                offer_data = json.loads(offer_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Advertencia al preparar oferta localmente ({e}), la nube intentará auto-descubrimiento.")

    result = run_automated_pipeline.remote(product_url=url, garment_files=garment_files, offer_data=offer_data)
    print("Resultado:", result)
