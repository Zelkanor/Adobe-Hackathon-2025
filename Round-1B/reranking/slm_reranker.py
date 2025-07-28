import re
from loguru import logger
from typing import List, Dict, Tuple
from llama_cpp import Llama


class SLMTitleGenerator:
    _TITLE_PROMPT_TEMPLATE = """[INST]
As a {persona}, your goal is to {job}. Create a concise, action-oriented title for the
following text chunk that is relevant to your goal. The title should be like a heading in a help document. 
Your response must be ONLY the title itself.

Text Chunk: "{text_chunk}"
[/INST]
"""

    def __init__(self, llm_instance: Llama, config: Dict[str, any]):
        logger.info("Initializing SLM Title Generator...")
        self.llm = llm_instance
        self.top_n = config.get('reranking', {}).get('slm_rerank_top_n', 5)

    def _clean_llm_output(self, text: str) -> str:
        """A robust function to clean final LLM text output."""
        # Remove any instruction tags or artifacts, case-insensitive
        text = re.sub(r'\[/?INST\].*', '', text, flags=re.IGNORECASE)
        # Remove common prefixes
        if text.lower().startswith("title:"):
            text = text[6:]
        # Remove quotes and extra whitespace
        return text.replace('"', '').replace('**', '').strip()

    def generate_titles(self, persona: str, job: str, candidates: List[Tuple[any, float]]) -> List[Dict[str, any]]:
        final_candidates = candidates[:self.top_n]
        logger.info(f"Generating section titles for top {len(final_candidates)} candidates...")
        titled_sections = []
        for i, (candidate, _) in enumerate(final_candidates):
            doc_id, element = candidate
            prompt = self._TITLE_PROMPT_TEMPLATE.format(persona=persona,job=job, text_chunk=element.text)
            try:
                response = self.llm(prompt, max_tokens=30, temperature=0.3, stop=["\n", "<|im_end|>"])
                title = self._clean_llm_output(response['choices'][0]['text'])
                
                titled_sections.append({
                    "document": doc_id.split('/')[-1],
                    "section_title": title if title else f"Section {i+1} Overview",
                    "importance_rank": i + 1,
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "original_chunk": element 
                })
            except Exception as e:
                logger.warning(f"Could not generate title for chunk {i}: {e}")
        
        logger.success("Finished generating section titles.")
        return titled_sections

    def __init__(self, llm_instance: Llama, config: Dict[str, any]):
        logger.info("Initializing SLM Title Generator...")
        self.llm = llm_instance
        self.top_n = config.get('reranking', {}).get('slm_rerank_top_n', 5)

    def _clean_llm_output(self, text: str) -> str:
        """A robust function to clean final LLM text output."""
        # Remove any instruction tags or artifacts, case-insensitive
        text = re.sub(r'\[/?INST\].*', '', text, flags=re.IGNORECASE)
        # Remove common prefixes
        if text.lower().startswith("title:"):
            text = text[6:]
        # Remove quotes and extra whitespace
        return text.replace('"', '').replace('**', '').strip()

    def generate_titles(self, persona: str, job: str, candidates: List[Tuple[any, float]]) -> List[Dict[str, any]]:
        final_candidates = candidates[:self.top_n]
        logger.info(f"Generating section titles for top {len(final_candidates)} candidates...")
        titled_sections = []
        for i, (candidate, _) in enumerate(final_candidates):
            doc_id, element = candidate
            prompt = self._TITLE_PROMPT_TEMPLATE.format(job=job, text_chunk=element.text)
            try:
                response = self.llm(prompt, max_tokens=30, temperature=0.3, stop=["\n", "<|im_end|>"])
                title = self._clean_llm_output(response['choices'][0]['text'])
                
                titled_sections.append({
                    "document": doc_id.split('/')[-1],
                    "section_title": title if title else f"Section {i+1} Overview",
                    "importance_rank": i + 1,
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "original_chunk": element 
                })
            except Exception as e:
                logger.warning(f"Could not generate title for chunk {i}: {e}")
        
        logger.success("Finished generating section titles.")
        return titled_sections

    def __init__(self, llm_instance: Llama, config: Dict[str, any]):
        logger.info("Initializing SLM Title Generator...")
        self.llm = llm_instance
        self.top_n = config.get('reranking', {}).get('slm_rerank_top_n', 5)

    def _clean_llm_output(self, text: str) -> str:
        """A robust function to clean final LLM text output."""
        # Remove any instruction tags or artifacts, case-insensitive
        text = re.sub(r'\[/?INST\].*', '', text, flags=re.IGNORECASE)
        # Remove common prefixes
        if text.lower().startswith("title:"):
            text = text[6:]
        # Remove quotes and extra whitespace
        return text.replace('"', '').replace('**', '').strip()

    def generate_titles(self, persona: str, job: str, candidates: List[Tuple[any, float]]) -> List[Dict[str, any]]:
        final_candidates = candidates[:self.top_n]
        logger.info(f"Generating section titles for top {len(final_candidates)} candidates...")
        titled_sections = []
        for i, (candidate, _) in enumerate(final_candidates):
            doc_id, element = candidate
            prompt = self._TITLE_PROMPT_TEMPLATE.format(persona=persona,job=job, text_chunk=element.text)
            try:
                response = self.llm(prompt, max_tokens=30, temperature=0.3, stop=["\n", "<|im_end|>"])
                title = self._clean_llm_output(response['choices'][0]['text'])
                
                titled_sections.append({
                    "document": doc_id.split('/')[-1],
                    "section_title": title if title else f"Section {i+1} Overview",
                    "importance_rank": i + 1,
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "original_chunk": element 
                })
            except Exception as e:
                logger.warning(f"Could not generate title for chunk {i}: {e}")
        
        logger.success("Finished generating section titles.")
        return titled_sections