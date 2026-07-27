# Servicio de pipeline en Cloud Run

Este contenedor ejecuta `pipeline.py` con Gemini en Vertex AI, Fish Audio,
Groq, MoviePy/FFmpeg, Cloudinary e Instagram. El endpoint `/run` requiere un
Bearer token y responde `202`; el estado de la ejecución se consulta en
`/runs/{id}`.

El Worker de Cloudflare es el único componente que debe activar el cron. La
publicación permanece desactivada hasta verificar una ejecución manual.
