FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY transcription_proxy.py .

RUN uv run --script transcription_proxy.py -- --help || true

EXPOSE 8091

CMD ["uv", "run", "--script", "transcription_proxy.py", "--", "--host", "0.0.0.0", "--port", "8091", "--lfm-url", "http://lfm-audio:8090"]
