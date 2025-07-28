from ultralytics import YOLO
from loguru import logger
from typing import List

class JITLayoutAnalyzer:
    def __init__(self, config):
        logger.info("Initializing JIT Layout Analyzer...")
        self.model = YOLO(config['dla_model_path'])
        self.config = config

    def analyze(self, candidates: list, parsed_docs: list):
        """
        Performs Just-in-Time layout analysis on pages of top candidates.
        Receives parsed_docs to access page images for the current request.
        """
        logger.info("Starting JIT Layout Analysis...")
        
        # Create a temporary mapping of page images for this request
        page_images_map = {}
        for doc in parsed_docs:
            page_images_map[doc.doc_id] = doc.page_images

        total_score = sum(score for _, score in candidates)
        cumulative_score = 0
        pages_to_analyze = set()
        
        for candidate, score in candidates:
            doc_id, chunk = candidate
            pages_to_analyze.add((doc_id, chunk.page_number))
            cumulative_score += score
            if cumulative_score / total_score > self.config['analysis']['adaptive_dla_score_threshold']:
                break
        
        logger.info(f"Adaptively selected {len(pages_to_analyze)} unique pages for DLA.")

        # Run DLA on selected pages
        layout_results = {}
        for doc_id, page_num in pages_to_analyze:
            # Check if the doc_id and page_num exist in our map
            if doc_id in page_images_map and page_num in page_images_map[doc_id]:
                img = page_images_map[doc_id][page_num]
                results = self.model(img)[0] # Run YOLO inference
                layout_results[(doc_id, page_num)] = results.boxes.data
            else:
                logger.warning(f"Could not find page image for {doc_id}, page {page_num}.")


        # Enrich candidates with layout info
        enriched_candidates = []
        for candidate, score in candidates:
            doc_id, chunk = candidate
            
            if (doc_id, chunk.page_number) in layout_results:
                page_layouts = layout_results[(doc_id, chunk.page_number)]
                for layout_box in page_layouts:
                    _, _, _, _, _, class_id = layout_box
                    layout_name = self.model.names[int(class_id)]
                    # Logic to check if chunk_bbox is inside a layout element
                    # For simplicity, we just add the detected layouts as metadata
                    chunk.text += f" [LAYOUT_CONTEXT: {layout_name}]"

            enriched_candidates.append((candidate, score))
            
        logger.success("Finished enriching candidates with layout data.")
        return enriched_candidates