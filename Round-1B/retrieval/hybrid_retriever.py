from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import List, Any

class HybridRetriever:
    def __init__(self, config: dict):
        logger.info("Initializing Hybrid Retriever...")
        self.embed_model = SentenceTransformer(config['embedding_model'])
        self.top_k = config['retrieval']['top_k']
        self.corpus_elements = None
        self.bm25_index = None
        self.faiss_index = None

    def build_indices(self, all_elements: List[Any]):
        """Builds BM25 and Faiss indices from a flat list of unstructured Elements."""
        logger.info(f"Building in-memory indices for {len(all_elements)} semantic chunks...")
        self.corpus_elements = all_elements
        
        tokenized_corpus = [el.text.split(" ") for el in self.corpus_elements]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Set show_progress_bar to False for cleaner logs in Docker
        embeddings = self.embed_model.encode([el.text for el in self.corpus_elements], show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, all_elements: List[Any]) -> List[tuple]:
        """Performs hybrid search over the corpus of unstructured Elements."""
        if self.corpus_elements is None or len(self.corpus_elements) != len(all_elements):
            self.build_indices(all_elements)

        # BM25 search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Faiss search
        query_embedding = self.embed_model.encode([query])
        distances, faiss_indices = self.faiss_index.search(query_embedding, len(self.corpus_elements))
        
        # --- Robust Score Normalization ---
        # Normalize BM25 scores safely
        min_bm25, max_bm25 = np.min(bm25_scores), np.max(bm25_scores)
        if max_bm25 == min_bm25:
            norm_bm25 = np.zeros_like(bm25_scores)
        else:
            norm_bm25 = (bm25_scores - min_bm25) / (max_bm25 - min_bm25)

        # Normalize Faiss scores safely
        min_faiss, max_faiss = np.min(distances[0]), np.max(distances[0])
        if max_faiss == min_faiss:
            norm_faiss = np.zeros_like(distances[0])
        else:
            # Invert distance to score
            norm_faiss = 1 - (distances[0] / max_faiss)
        
        faiss_score_map = {idx: score for idx, score in zip(faiss_indices[0], norm_faiss)}
        
        # Combine scores with a 60% weight on semantic search
        combined_scores = [
            (0.4 * norm_bm25[i] + 0.6 * faiss_score_map.get(i, 0))
            for i in range(len(self.corpus_elements))
        ]

        scored_candidates = sorted(
            [(score, (el.metadata.filename, el)) for score, el in zip(combined_scores, self.corpus_elements)],
            key=lambda x: x[0],
            reverse=True
        )
        
        final_candidates = [(cand, score) for score, cand in scored_candidates[:self.top_k]]
        
        logger.success(f"Retrieved {len(final_candidates)} candidates via hybrid search.")
        return final_candidates