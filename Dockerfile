FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY transcription_proxy.py .

EXPOSE 8091

CMD ["uv", "run", "--frozen", "python", "transcription_proxy.py", "--host", "0.0.0.0", "--port", "8091", "--lfm-url", "http://lfm-audio:8090"]
