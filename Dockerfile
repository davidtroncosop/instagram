FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        imagemagick \
        fonts-dejavu \
        fonts-liberation \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY pipeline.py /app/pipeline.py
COPY deploy/cloud-run/service_app.py /app/service_app.py

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn service_app:app --host 0.0.0.0 --port ${PORT:-8080}"]
