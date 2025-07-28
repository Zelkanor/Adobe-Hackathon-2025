import re
from loguru import logger
from typing import List, Dict
from llama_cpp import Llama

class AnswerGenerator:
    """Generates prose for each subsection using a simple, direct prompt."""

    _EXTRACTOR_PROMPT_TEMPLATE = """[INST]
Act as a {persona}. Your goal is to {job}.
Extract the key sentences from the following text that are relevant to your goal.
Do not summarize or add any new information. Your output must be only the extracted text.

Text Chunk: "{text_chunk}"
[/INST]
"""

    def __init__(self, llm_instance: Llama):
        logger.info("Initializing Answer Generator...")
        self.llm = llm_instance

    def _clean_and_validate_prose(self, text: str, original_chunk: str) -> str:
        """
        A simple function to clean the LLM's output and validate it.
        """
        cleaned_text = text.strip()
        
        # 1. Remove any potential instruction tags
        cleaned_text = re.sub(r'\[/?INST\].*', '', cleaned_text, flags=re.IGNORECASE)

        # 2. Check if the model simply repeated the original chunk
        # This prevents returning the full, un-extracted text.
        if cleaned_text.strip() == original_chunk.strip():
             return "" # Invalid output, trigger fallback

        # 3. Final cleanup of whitespace and common artifacts
        cleaned_text = cleaned_text.replace('\n', ' ').replace('**', '')
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()

    def generate_subsections(self, persona: str, job: str, titled_sections: List[Dict[str, any]]) -> List[Dict[str, any]]:
        logger.info("Generating final prose for subsection analysis...")
        subsection_analysis = []
        for section in titled_sections:
            try:
                chunk = section['original_chunk']
                prompt = self._EXTRACTOR_PROMPT_TEMPLATE.format(persona=persona, job=job, text_chunk=chunk.text)
                
                response = self.llm(
                    prompt, max_tokens=512, temperature=0.1, 
                    repeat_penalty=1.1, stop=["\n\n\n", "[/INST]"]
                )
                
                # Apply the new, simpler cleaning function
                refined_text = self._clean_and_validate_prose(response['choices'][0]['text'], chunk.text)
                
                # If cleaning returned an empty string or the result is too short, fall back.
                if len(refined_text.split()) < 10:
                    logger.warning(f"Generated text for '{section['section_title']}' was invalid or too short. Using original chunk as fallback.")
                    refined_text = (chunk.text[:700] + '...') if len(chunk.text) > 700 else chunk.text
                
                subsection_analysis.append({
                    "document": section['document'],
                    "refined_text": refined_text,
                    "page_number": section['page_number']
                })
            except Exception as e:
                logger.error(f"Could not process subsection for document {section.get('document', 'N/A')}: {e}")
        
        logger.success("Finished generating subsection analysis.")
        return subsection_analysis