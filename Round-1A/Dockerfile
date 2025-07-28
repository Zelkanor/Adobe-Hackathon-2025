FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model artifacts into the container
COPY final_heading_model/ ./final_heading_model/

# Copy the prediction script into the container
COPY predict.py .

# Define the command to run your application
# This will execute the main() function in your predict.py script
CMD ["python", "predict.py"]