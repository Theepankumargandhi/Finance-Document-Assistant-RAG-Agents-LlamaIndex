# Stable base with manylinux wheels available
FROM python:3.10-slim

# Saner defaults + Streamlit settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Minimal build deps and libgomp (needed by torch/faiss wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Keep pip toolchain predictable and pre-pin numpy<2 to avoid source builds
RUN python -m pip install --upgrade pip "setuptools<70" wheel && \
    python -m pip install "numpy<2"

# Install Python deps (use -vvv so CI logs show the failing package if any)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -vvv --no-cache-dir -r /app/requirements.txt

# Copy the app
COPY . /app

# Expose Streamlit
EXPOSE 8501

# Run from the Streamlit app directory
WORKDIR /app/llm-app/streamlit
CMD ["streamlit", "run", "app.py"]