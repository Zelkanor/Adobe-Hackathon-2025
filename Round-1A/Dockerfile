# Dockerfile for Adobe Hackathon - Round 1A

FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# FIX: Install the required libgomp1 system dependency for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the minimal requirements file
COPY requirements.txt .

# Install Python packages, ensuring no cache is stored
RUN pip install --no-cache-dir -r requirements.txt

# Copy your trained model artifacts and the prediction script
COPY final_heading_model/ ./final_heading_model/
COPY predict.py .

# Define the command to run your application
CMD ["python", "predict.py"]