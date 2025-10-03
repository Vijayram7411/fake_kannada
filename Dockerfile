# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for some libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Download NLTK data needed at runtime
RUN python - <<PY
import nltk
for p in ['punkt','stopwords','wordnet']:
    try:
        nltk.data.find(p)
    except LookupError:
        nltk.download(p, quiet=True)
print('NLTK data ready')
PY

# Copy project
COPY . /app

EXPOSE 5000

# Default: run API
CMD ["python", "main.py", "--mode", "api", "--port", "5000"]
