# Builder - train the model
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .
RUN python train_model.py

# Runtime - lean production image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY tests/ tests/

# Copy trained artefacts from builder stage
COPY --from=builder /build/model.pkl .
COPY --from=builder /build/model_metadata.pkl .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
