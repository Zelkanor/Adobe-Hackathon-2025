import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger
import numpy as np

def build_indices(processed_docs_path: str, embedding_model_name: str, faiss_path: str, bm25_path: str):
    """
    Builds and saves FAISS (dense) and BM25 (sparse) indices from processed documents.
    """
    logger.info("Loading processed documents...")
    with open(processed_docs_path, "rb") as f:
        documents = pickle.load(f)

    all_chunks_text = [chunk.text for doc in documents for chunk in doc.chunks]
    
    # --- Build BM25 Index (Sparse) ---
    logger.info("Building BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in all_chunks_text]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.success(f"BM25 index saved to {bm25_path}")

    # --- Build FAISS Index (Dense) ---
    logger.info(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name, device="cpu")
    
    logger.info("Encoding all chunks... (This may take a while)")
    embeddings = model.encode(all_chunks_text, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype='float32')

    logger.info("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    logger.success(f"FAISS index saved to {faiss_path}")