import os
import json
from typing import Dict, Any
from openai import OpenAI
from src.config import get_provider_config, Provider, VLLM_MODELS

class MetadataExtractor:
    """
    Extracts metadata from document text using a local LLM (vLLM).
    """
    
    def __init__(self, api_base: str = None, api_key: str = None, model: str = None):
        vllm_config = get_provider_config(Provider.VLLM)
        self.client = OpenAI(
            base_url=api_base or vllm_config.base_url,
            api_key=api_key or vllm_config.api_key,
        )
        self.model = model or (VLLM_MODELS[0].id if VLLM_MODELS else "Qwen/Qwen2.5-0.5B-Instruct")

    def extract(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Extracts structured metadata from the first 2000 characters of the text.
        """
        # Truncate text to avoid context window issues for metadata extraction
        sample_text = text[:2000]
        
        prompt = f"""
        Analyze the following document excerpt and extract metadata in JSON format.
        
        Filename: {filename}
        Excerpt:
        {sample_text}
        
        Required Fields:
        - document_type: (e.g., Invoice, Contract, Report, Memo, Technical Manual)
        - date: (YYYY-MM-DD if found, else null)
        - department: (e.g., HR, Engineering, Sales, Legal)
        - sensitivity_level: (Public, Internal, Confidential, Restricted)
        - summary: (One sentence summary)

        Return ONLY the JSON object.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise metadata extraction assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            # Basic cleanup to ensure JSON
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {
                "document_type": "Unknown",
                "date": None,
                "department": "Unknown",
                "sensitivity_level": "Internal",
                "summary": "Metadata extraction failed."
            }
