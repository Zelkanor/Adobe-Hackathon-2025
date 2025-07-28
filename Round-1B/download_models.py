# download_models.py
from huggingface_hub import snapshot_download, hf_hub_download
import os

MODELS_DIR = "Round-1B/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. NEW LLM (TinyLlama 1.1B)
print("Downloading Final LLM: Qwen2 1.5B...")
hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q2_k.gguf",
    local_dir=MODELS_DIR
)


# 2. Embedding Model (Stays the same)
print("Downloading Embedding Model (PyTorch version only)...")
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=os.path.join(MODELS_DIR, "all-MiniLM-L6-v2"),
    # FIX: Only download the necessary files for sentence-transformers
    allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", ".gitattributes", "README.md"]
)

# 3. NEW Reranker Model (mxbai-rerank-base)
print("Downloading new Reranker Model (PyTorch version only)...")
snapshot_download(
    repo_id="mixedbread-ai/mxbai-rerank-base-v1",
    local_dir=os.path.join(MODELS_DIR, "mxbai-rerank-base-v1"),
    # FIX: Only download the necessary files
    allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", ".gitattributes", "README.md"]
)

# 4. DLA Model (Stays the same)
print("Downloading DLA Model...")
hf_hub_download(
    repo_id="omoured/YOLOv10-Document-Layout-Analysis",
    filename="yolov10n_best.pt",
    local_dir=MODELS_DIR
)

print("\nAll final quality models downloaded successfully!")