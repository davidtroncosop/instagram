import modal
import subprocess
import os

app = modal.App("instagram-reel-pipeline")

# Definir contenedor Debian con FFmpeg e dependencias necesarias
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "httpx",
        "pillow",
        "moviepy",
        "python-dotenv",
        "openai",
        "google-genai",
        "cloudinary"
    )
    .add_local_file(".env", remote_path="/root/.env")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image,
    timeout=1200,
    schedule=modal.Cron("0 13,18,23 * * *")  # Cron: Ejecución automática diaria a las 10:00, 15:00 y 20:00 Chile
)
def run_automated_pipeline():
    """Ejecuta el pipeline de creación y publicación autónoma de Reels en Modal."""
    os.chdir("/root/project")
    cmd = [
        "python", "-u", "pipeline.py",
        "--auto-discover",
        "--voiceover",
        "--subtitles",
        "--organic-test",
        "--publish"
    ]
    print("[Modal Container] 🚀 Iniciando flujo automatizado de creación y publicación en la nube...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[Modal Error]:", res.stderr)
        raise RuntimeError(f"El pipeline falló en Modal: {res.stderr}")
    return res.stdout

@app.local_entrypoint()
def main():
    print("Iniciando prueba manual en Modal Cloud...")
    result = run_automated_pipeline.remote()
    print("Resultado:", result)
