# HSTTB - Healthcare Streaming STT Benchmarking
# Docker image for running the web application

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]"

# Copy remaining files
COPY docs/ docs/
COPY tests/ tests/
COPY Makefile .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hsttb && \
    chown -R hsttb:hsttb /app
USER hsttb

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the web application
CMD ["uvicorn", "hsttb.webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
