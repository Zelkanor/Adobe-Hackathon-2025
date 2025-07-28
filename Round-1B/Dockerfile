# ------ Stage 1: Builder ------
FROM python:3.10-slim as builder
WORKDIR /tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt --wheel-dir=/wheels --no-cache-dir

RUN pip install --no-index --find-links=/wheels nltk

# Download only needed NLTK data resources
RUN python -m nltk.downloader -d /tmp/nltk_data \
    punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng punkt_tab

# Unzip any zipped resources, just to be safe
RUN find /tmp/nltk_data -name '*.zip' -exec unzip -o {} -d /tmp/nltk_data \; || true

# ------ Stage 2: Runtime ------
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgomp1 libgl1-mesa-glx libglib2.0-0 tesseract-ocr unzip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && pip install --no-index --find-links=/wheels /wheels/*.whl --no-cache-dir && rm -rf /wheels /root/.cache ~/.cache

# Copy only minimal NLTK data needed
COPY --from=builder /tmp/nltk_data /usr/local/share/nltk_data

# Symlink for all standard lookup paths
RUN ln -s /usr/local/share/nltk_data /root/nltk_data || true \
 && ln -s /usr/local/share/nltk_data /usr/local/nltk_data || true \
 && ln -s /usr/local/share/nltk_data /usr/share/nltk_data || true \
 && ln -s /usr/local/share/nltk_data /usr/lib/nltk_data || true

COPY . .

RUN find . -type d -name '__pycache__' | xargs rm -rf || true

CMD ["python", "main.py"]
