FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv

# Install only runtime dependencies first for better Docker layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project

# Copy only application files needed at runtime.
COPY src ./src
COPY docs ./docs
COPY models ./models
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
