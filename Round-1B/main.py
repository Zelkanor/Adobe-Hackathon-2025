import yaml
import json
import sys
import gc
from loguru import logger
from llama_cpp import Llama
from datetime import datetime
from typing import List, Tuple, Any
import concurrent.futures
from pathlib import Path
import argparse


# Import all pipeline components
from ingestion.document_parser import parse_document
from retrieval.hybrid_retriever import HybridRetriever
from reranking.fast_reranker import FastReranker
from reranking.slm_reranker import SLMTitleGenerator
from generation.answer_generator import AnswerGenerator

class JITAR_OnTheFly_Pipeline:
    def __init__(self, config_path="config.yaml"):
        logger.info("🚀 Initializing JITAR Pipeline for a new request...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Lazy loading for models
        self.retriever = None
        self.layout_analyzer = None
        self.fast_reranker = None
        self.slm_reranker = None
        self.answer_generator = None
        self.shared_llm = None
        
    def _load_models(self):
        """Loads all models on demand."""
        if self.shared_llm is None:
            self.retriever = HybridRetriever(self.config)
            self.fast_reranker = FastReranker(self.config)
            
            logger.info(f"Loading SLM from {self.config['llm']['model_path']}...")
            self.shared_llm = Llama(
                model_path=self.config['llm']['model_path'],
                n_ctx=self.config['llm']['n_ctx'],
                n_gpu_layers=self.config['llm']['n_gpu_layers'],
                verbose=False
            )
            self.title_generator  = SLMTitleGenerator(self.shared_llm, self.config)
            self.answer_generator = AnswerGenerator(self.shared_llm)

    def _determine_strategy(self, job: str) -> str:
        """Determines the strategy using a keyword-based heuristic."""
        logger.info("Determining optimal strategy via keyword heuristic...")
        
        EXPLORATORY_KEYWORDS = [
            'plan', 'prepare', 'review', 'analyze', 'compare', 
            'discover', 'explore', 'summarize', 'brainstorm', 'itinerary', 'trip'
        ]
        
        job_lower = job.lower()
        if any(keyword in job_lower for keyword in EXPLORATORY_KEYWORDS):
            logger.success("Exploratory keywords found. Strategy selected: EXPLORATORY")
            return "EXPLORATORY"
        
        logger.success("No exploratory keywords found. Strategy selected: PRECISION")
        return "PRECISION"

    def _diversify_candidates(self, candidates: List[Tuple[Any, float]], num_final: int = 5) -> List[Tuple[Any, float]]:
        """
        Enforces document diversity in a ranked list of candidates.
        
        It ensures the top results come from different source documents,
        preventing the final output from being dominated by a single file.
        """
        logger.info("Enforcing document diversity in ranked list...")
        
        diversified_candidates = []
        used_documents = set()
        
        # Add the absolute best candidate first, regardless of source
        if candidates:
            best_candidate = candidates.pop(0)
            diversified_candidates.append(best_candidate)
            doc_id = best_candidate[0][0] # Get the document ID
            used_documents.add(doc_id)
            
        # Now, iterate through the rest to find candidates from new documents
        remaining_candidates = []
        for cand in candidates:
            doc_id = cand[0][0]
            if doc_id not in used_documents and len(diversified_candidates) < num_final:
                diversified_candidates.append(cand)
                used_documents.add(doc_id)
            else:
                remaining_candidates.append(cand) # Keep the rest for fallback
        
        # If we still don't have enough candidates, fill with the best of the rest
        if len(diversified_candidates) < num_final:
            needed = num_final - len(diversified_candidates)
            diversified_candidates.extend(remaining_candidates[:needed])
            
        logger.success(f"Diversified list to {len(diversified_candidates)} candidates from {len(used_documents)} unique documents.")
        return diversified_candidates


    def process_request(self, input_data: dict,input_dir: Path):
        """Runs the full on-the-fly RAG pipeline for a given JSON request."""
        self._load_models()

        # --- Extract info from input JSON ---
        persona = input_data['persona']['role']
        job = input_data['job_to_be_done']['task']
        doc_infos = input_data['documents']
        pdf_paths = [input_dir / doc['filename'] for doc in doc_infos]
        query = f"As a {persona}, I need to {job}"
        # 1. Determine the Strategy
        strategy = self._determine_strategy(job)

        # --- Pipeline Steps ---
        logger.info(f"Starting parallel parsing of {len(pdf_paths)} documents...")
        parsed_doc_elements = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # map() runs parse_document on each path in parallel
            results = executor.map(parse_document, pdf_paths)
            for doc_list in results:
                parsed_doc_elements.append(doc_list)
        all_elements = [element for doc_list in parsed_doc_elements for element in doc_list]
        logger.success(f"Finished parallel parsing. Found {len(all_elements)} total chunks.")

         

        # 2. Hybrid Retrieval
        candidates = self.retriever.retrieve(query, all_elements )

        
        pruned_candidates = self.fast_reranker.rerank(query, candidates)
        # 5. ADAPTIVE STEP: Apply diversity only if the task is exploratory
        final_candidates_for_llm = []
        if strategy == "EXPLORATORY":
            final_candidates_for_llm = self._diversify_candidates(
                pruned_candidates, 
                num_final=self.config['reranking']['slm_rerank_top_n']
            )
        else: # "PRECISION" strategy
            logger.info("Using 'Precision First' strategy, skipping diversity step.")
            final_candidates_for_llm = pruned_candidates[:self.config['reranking']['slm_rerank_top_n']]
        extracted_sections = self.title_generator.generate_titles(persona, job, final_candidates_for_llm)
        subsection_analysis = self.answer_generator.generate_subsections(persona, job, extracted_sections)

        # --- Assemble Final Output ---
        output_json = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in doc_infos],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [{k: v for k, v in sec.items() if k != 'original_chunk'} for sec in extracted_sections],
            "subsection_analysis": subsection_analysis
        }
        
        return output_json
        
    def shutdown(self):
        """
        Explicitly deallocates the shared LLM to release resources cleanly.
        """
        if self.shared_llm is not None:
            logger.info("Deallocating shared LLM to free up memory...")
            del self.shared_llm
            self.shared_llm = None
            logger.info("Cleanup complete.")

def main(args):
    """Main function to find and process all JSON requests in the input directory."""
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # FIX: Find all .json files in the input directory
    json_request_files = list(args.input_dir.glob("*.json"))
    if not json_request_files:
        logger.error(f"No JSON request files found in '{args.input_dir}'")
        sys.exit(1)
        
    logger.info(f"Found {len(json_request_files)} JSON request(s) to process.")
    
    # Initialize the pipeline once to avoid reloading models
    pipeline = None
    try:
        pipeline = JITAR_OnTheFly_Pipeline(config_path=args.config)
        
        # Loop through each request file and process it
        for input_json_path in json_request_files:
            logger.info(f"--- Processing new request: {input_json_path.name} ---")
            
            # Define the output path based on the input filename
            output_filename = f"{input_json_path.stem}_output.json"
            output_file_path = args.output_dir / output_filename
            
            # Load the specific input JSON file
            try:
                with open(input_json_path, 'r') as f:
                    input_data = json.load(f)
                logger.success(f"Successfully loaded {input_json_path.name}.")
            except Exception as e:
                logger.error(f"Failed to load or parse {input_json_path.name}: {e}")
                continue # Skip to the next file

            # Run the main pipeline process
            final_output = pipeline.process_request(input_data, args.input_dir)
            
            # Write the final JSON to its corresponding output file
            logger.info(f"Writing final output to: {output_file_path}")
            with open(output_file_path, 'w') as f:
                json.dump(final_output, f, indent=4)
            logger.success(f"Finished processing {input_json_path.name}.")

    except Exception as e:
        logger.critical(f"A critical error occurred during pipeline initialization or execution: {e}")
        sys.exit(1)
    finally:
        if pipeline:
            pipeline.shutdown()
            gc.collect()
if __name__ == "__main__":
    # Using argparse for robust command-line handling, with defaults for Docker
    parser = argparse.ArgumentParser(description="JITAR Persona-Driven Document Intelligence Pipeline")
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        default=Path("/app/input"),
        help="Directory containing input PDFs and the request JSON file."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("/app/output"),
        help="Directory where the output JSON file will be saved."
    )
    parser.add_argument(
        "--request_filename", 
        type=str, 
        default="in.json",
        help="The name of the input request JSON file inside the input directory."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to the configuration YAML file."
    )
    
    args = parser.parse_args()
    main(args)