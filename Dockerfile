# HSTTB - Healthcare Streaming STT Benchmarking
# Docker image for running the web application

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HSTTB_DATA_DIR=/app/data

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies including httpx for API fetching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]" && \
    pip install --no-cache-dir httpx

# Copy scripts and other files
COPY scripts/ scripts/
COPY docs/ docs/
COPY tests/ tests/
COPY Makefile .

# Create directories for data persistence
RUN mkdir -p /app/data /app/reports /root/.hsttb

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hsttb && \
    chown -R hsttb:hsttb /app && \
    mkdir -p /home/hsttb/.hsttb && \
    chown -R hsttb:hsttb /home/hsttb/.hsttb

USER hsttb

# Set home directory for lexicon database
ENV HOME=/home/hsttb

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the startup script which initializes lexicon and starts server
CMD ["python", "scripts/startup.py", "--host", "0.0.0.0", "--port", "8000"]
