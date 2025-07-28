# Dockerfile for Adobe Hackathon - Round 1B 

# Start with a clean base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project (including the local 'models' folder)
# The .dockerignore file will prevent copying unwanted folders like .venv
COPY . .

# Define the command to run the application
CMD ["python", "main.py"]