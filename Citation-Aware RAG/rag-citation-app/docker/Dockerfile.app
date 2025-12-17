# 1. Base Image: Use a lightweight, official Python runtime
FROM python:3.11-slim

# 2. System Dependencies
# 'build-essential' and 'curl' are often needed for compiling libraries like ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory: Where our code lives inside the container
WORKDIR /app

# 4. Install Dependencies
# We copy requirements FIRST to leverage Docker's "Layer Caching"
# (This makes rebuilds faster if you only change code, not libs)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY src/ src/

# 6. Create Persistence Directory (So ChromaDB doesn't crash)
RUN mkdir -p data/chroma_db

# 7. Environment Setup
# Add /app to python path so 'import src.xxx' works
ENV PYTHONPATH=/app
# Disable ChromaDB telemetry to speed up boot
ENV ANONYMIZED_TELEMETRY=False

# 8. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Expose Port
EXPOSE 8501

# 10. Run Command
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]