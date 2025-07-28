import re
from loguru import logger
from typing import List, Dict
from llama_cpp import Llama
import unicodedata
class AnswerGenerator:
    """Generates prose for each subsection using a simple, direct prompt."""
    
    _PROSE_PROMPT_TEMPLATE = """[INST]
Your role is a "{persona}" and your task is to "{job}".
From the 'Text Chunk' below, extract only the key sentences and instructions that are directly useful for your task.
Do not summarize or add any new information. Combine the extracted sentences into a clean block of text.

Text Chunk: "{text_chunk}"
[/INST]
"""

    def __init__(self, llm_instance: Llama):
        logger.info("Initializing Answer Generator...")
        self.llm = llm_instance

    def _clean_prose(self, text: str) -> str:
        """A robust function to clean generated prose and remove prompt echoes."""
        cleaned_text = text.strip()
        
        # Define keywords that indicate the model is repeating its instructions
        instruction_pattern = re.compile(
            r'^\s*("|\'|paragraph: |)*\s*(you are an|your task is|acting as a|write a concise|the goal is to)', 
            re.IGNORECASE
        )
        
        # If the pattern matches, it's an echo. Return empty.
        if instruction_pattern.match(cleaned_text):
            return ""

        # Remove any lingering instruction tags
        cleaned_text = re.sub(r'\[/?INST\].*', '', cleaned_text, flags=re.IGNORECASE)
        # Remove common prefixes
        if cleaned_text.lower().startswith("paragraph:"):
            cleaned_text = cleaned_text[10:]

        # Enhanced newline handling
        cleaned_text = self._normalize_newlines(cleaned_text)
        
        # Enhanced Unicode ligature handling
        cleaned_text = self._normalize_ligatures(cleaned_text)
        
        # Additional cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single space
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _normalize_newlines(self, text: str) -> str:
        """Normalize various newline patterns"""
        # Handle different newline representations
        text = text.replace('\\n\\n', ' ')  # Double newlines to space
        text = text.replace('\\n', ' ')     # Single newlines to space
        text = text.replace('\n\n', ' ')    # Actual double newlines
        text = text.replace('\n', ' ')      # Actual single newlines
        
        # Clean up any remaining escaped characters
        text = text.replace('\\t', ' ')     # Tabs
        text = text.replace('\\r', ' ')     # Carriage returns
        
        return text

    def _normalize_ligatures(self, text: str) -> str:
        """Comprehensive Unicode ligature normalization"""
        
        # Handle escaped Unicode ligatures (from your example)
        ligature_map_escaped = {
            '\\ufb00': 'ff',   # ﬀ
            '\\ufb01': 'fi',   # ﬁ  
            '\\ufb02': 'fl',   # ﬂ
            '\\ufb03': 'ffi',  # ﬃ 
            '\\ufb04': 'ffl',  # ﬄ
            '\\ufb05': 'ft',   # ﬅ
            '\\ufb06': 'st',   # ﬆ
        }
        
        # Handle actual Unicode ligature characters
        ligature_map_unicode = {
            '\ufb00': 'ff',    # ﬀ
            '\ufb01': 'fi',    # ﬁ
            '\ufb02': 'fl',    # ﬂ
            '\ufb03': 'ffi',   # ﬃ
            '\ufb04': 'ffl',   # ﬄ
            '\ufb05': 'ft',    # ﬅ
            '\ufb06': 'st',    # ﬆ
        }
        
        # Apply escaped ligature replacements
        for escaped, replacement in ligature_map_escaped.items():
            text = text.replace(escaped, replacement)
            
        # Apply Unicode ligature replacements
        for unicode_char, replacement in ligature_map_unicode.items():
            text = text.replace(unicode_char, replacement)
            
        # Handle additional problematic Unicode characters
        text = self._handle_special_unicode(text)
        
        return text
    
    def _handle_special_unicode(self, text: str) -> str:
        """Handle other problematic Unicode characters commonly found in PDFs"""
        
        # Common PDF extraction artifacts
        replacements = {
            '\u2013': '-',      # en dash
            '\u2014': '-',      # em dash
            '\u2018': "'",      # left single quotation mark
            '\u2019': "'",      # right single quotation mark
            '\u201c': '"',      # left double quotation mark
            '\u201d': '"',      # right double quotation mark
            '\u2026': '...',    # horizontal ellipsis
            '\u00a0': ' ',      # non-breaking space
            '\u200b': '',       # zero-width space
            '\ufeff': '',       # byte order mark
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
            
        # Normalize remaining Unicode to closest ASCII equivalent
        try:
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
        except:
            # If normalization fails, just continue with what we have
            pass
            
        return text

    def generate_subsections(self, persona: str, job: str, titled_sections: List[Dict[str, any]]) -> List[Dict[str, any]]:
        logger.info("Generating prose for subsection analysis...")
        subsection_analysis = []
        for section in titled_sections:
            try:
                chunk = section['original_chunk']
                prompt = self._PROSE_PROMPT_TEMPLATE.format(persona=persona, job=job, text_chunk=chunk.text)
                
                response = self.llm(
                    prompt, max_tokens=200, temperature=0.2, 
                    repeat_penalty=1.1, stop=["\n\n", "[/INST]"]
                )
                
                refined_text = self._clean_prose(response['choices'][0]['text'])
                
                if len(refined_text.split()) < 15:
                    logger.warning(f"Generated prose too short. Using original chunk as fallback.")
                    refined_text = (chunk.text[:600] + '...') if len(chunk.text) > 600 else chunk.text
                
                subsection_analysis.append({
                    "document": section['document'],
                    "refined_text": refined_text,
                    "page_number": section['page_number']
                })
            except Exception as e:
                logger.error(f"Could not process subsection for document {section.get('document', 'N/A')}: {e}")
        
        logger.success("Finished generating subsection analysis.")
        return subsection_analysis