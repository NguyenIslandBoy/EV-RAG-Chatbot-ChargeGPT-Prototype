FROM python:3.13-slim

WORKDIR /app

# System deps required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs as a separate service — this container is the app layer only.
# Mount /app/data as a volume to persist DuckDB and FAISS index across restarts.
VOLUME ["/app/data", "/app/logs"]

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]