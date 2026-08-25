# Multi-stage Dockerfile for ev-flex-ml

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime Application
FROM python:3.11-slim AS runner

WORKDIR /app

COPY --from=builder /install /usr/local
COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8000

# Default command launches FastAPI REST API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
