# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy model and NLTK resources during build to avoid runtime delays
RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet punkt omw-1.4

# Set NLTK data path environment variable so it finds it
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy the rest of the application files
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY README.md .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# The default command will be overridden in docker-compose.yml for different services
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
