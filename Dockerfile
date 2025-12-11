# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies using pip (faster than Poetry)
# Break into smaller chunks to avoid timeout issues
RUN pip install --no-cache-dir --timeout 300 --upgrade pip && \
    pip install --no-cache-dir --timeout 300 torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --timeout 300 -r requirements.txt

# Note: Docling/EasyOCR models are downloaded at runtime on first use.
# To persist models between container restarts, mount volumes:
#   - /root/.EasyOCR (EasyOCR models)
#   - /root/.cache/docling (Docling models)
#   - /root/.cache/huggingface (HuggingFace models for embeddings)

# Pre-download the embedding model used by SemanticCache to avoid download on startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
