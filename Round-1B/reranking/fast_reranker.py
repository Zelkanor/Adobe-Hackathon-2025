from sentence_transformers.cross_encoder import CrossEncoder
from loguru import logger
from typing import List, Tuple, Any

class FastReranker:
    def __init__(self, config: dict):
        model_path = config['reranker_model']
        logger.info(f"Initializing CrossEncoder Reranker from path: {model_path}...")
        # Use CrossEncoder, which correctly loads the model from the local path.
        self.model = CrossEncoder(model_path)
        self.top_n = config['reranking']['fast_rerank_top_n']

    def rerank(self, query: str, candidates: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Reranks candidates using the sentence-transformers CrossEncoder."""
        
        if not candidates:
            return []

        # The CrossEncoder expects a list of [query, passage] pairs.
        query_passage_pairs = [(query, cand[0][1].text) for cand in candidates]
        
        # Predict scores
        scores = self.model.predict(query_passage_pairs, show_progress_bar=False)
        
        # Combine candidates with their new scores
        scored_candidates = list(zip(scores, candidates))
        
        # Sort in descending order of score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Prune to top_n and extract the original candidate items
        reranked_candidates = [cand for score, cand in scored_candidates[:self.top_n]]
        
        logger.success(f"Reranked and pruned to {len(reranked_candidates)} candidates.")
        return reranked_candidates