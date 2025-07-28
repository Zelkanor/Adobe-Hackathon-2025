# Dockerfile for Adobe Hackathon - Round 1B

# --- Stage 1: The Builder ---
# This stage downloads models so they don't need to be in the local context.
FROM python:3.11-slim as builder

WORKDIR /app

# Install only the downloader utility
RUN pip install huggingface-hub

# Copy and run the download script to fetch all models
COPY download_models.py .
RUN python download_models.py


# --- Stage 2: The Final Application Image ---
# Start fresh with a clean base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies for PDF processing
# poppler-utils is required by the 'unstructured' library for PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code
COPY . .

# Copy the pre-downloaded models from the builder stage
# This is the key step for a fully offline image
COPY --from=builder /app/models ./models

# Define the command to run the application.
# It expects the input JSON to be in the /app/input directory as per hackathon rules.
# You might need to adjust the filename 'in.json' if it's different.
CMD ["python", "main.py", "/app/input/in.json"]