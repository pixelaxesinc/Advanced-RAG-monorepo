import os
from openai import OpenAI
from src.config import get_provider_config, Provider, VLLM_MODELS

class QueryProcessor:
    """
    Handles Query Rewriting and HyDE (Hypothetical Document Embeddings).
    """
    
    def __init__(self):
        vllm_config = get_provider_config(Provider.VLLM)
        self.client = OpenAI(
            base_url=vllm_config.base_url,
            api_key=vllm_config.api_key,
        )
        # Use model from centralized config
        self.model = VLLM_MODELS[0].id if VLLM_MODELS else os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    def rewrite_query(self, user_query: str, chat_history: list = []) -> str:
        """
        Rewrites a conversational query into a search-optimized keyword query.
        """
        prompt = f"""
        You are a search optimization expert. Rewrite the following user query to be precise and keyword-rich for a vector database search.
        Remove conversational filler. Focus on technical terms and intent.
        
        User Query: {user_query}
        
        Rewritten Query:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return user_query

    def generate_hyde_answer(self, query: str) -> str:
        """
        Generates a hypothetical answer to the query to improve retrieval (HyDE).
        """
        prompt = f"""
        Please write a short, plausible passage that answers the question. 
        It doesn't need to be factually correct, but it should use the right vocabulary and structure.
        
        Question: {query}
        
        Passage:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 # Higher temperature for creativity
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query
