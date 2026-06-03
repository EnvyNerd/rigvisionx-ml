# Use a small, secure Python base image (override with PYTHON_BASE_IMAGE build arg)
ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for better security
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# System deps (kept minimal). Add build-essential only if you add libs that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Make sure Python can import from src/
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER appuser

# Expose API and dashboard ports
EXPOSE 8080 8050

# Start the FastAPI server (production-friendly settings)
CMD ["sh", "-c", "uvicorn rigvisionx.serving.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
