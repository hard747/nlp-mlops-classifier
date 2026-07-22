# --- Builder stage: install CPU-only torch + deps into a venv ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
# Model weights are baked into the image: Render's free tier has no persistent
# bind-mount equivalent, so the served checkpoint must ship inside the build.
COPY src/infrastructure/ml_model/weights/ ./src/infrastructure/ml_model/weights/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.infrastructure.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
