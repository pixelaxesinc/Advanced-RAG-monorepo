import os
from enum import Enum
from openai import OpenAI  # Standard OpenAI client
from langfuse import observe, get_client  # SDK v3

from src.config import (
    get_provider_config, get_models_by_provider, get_default_model,
    Provider, VLLM_MODELS, OPENROUTER_MODELS
)

class ModelTier(Enum):
    TIER_1_SIMPLE = "tier-1"
    TIER_2_RAG = "tier-2"
    TIER_3_COMPLEX = "tier-3"

class ModelRouter:
    """
    Routes queries to the appropriate model based on complexity.
    Uses centralized configuration from src/config.py.
    """
    
    def __init__(self):
        # Get provider configs from centralized config
        vllm_config = get_provider_config(Provider.VLLM)
        openrouter_config = get_provider_config(Provider.OPENROUTER)
        
        # Local vLLM Client
        self.local_client = OpenAI(
            base_url=vllm_config.base_url,
            api_key=vllm_config.api_key
        )
        
        # OpenRouter Client (for complex queries)
        self.cloud_client = OpenAI(
            base_url=openrouter_config.base_url,
            api_key=openrouter_config.api_key
        )
        
        # Model Definitions from centralized config
        # Use first vLLM model for local tiers, first OpenRouter model for cloud tier
        local_model = VLLM_MODELS[0].id if VLLM_MODELS else "Qwen/Qwen3-0.6B"
        cloud_model = OPENROUTER_MODELS[0].id if OPENROUTER_MODELS else "anthropic/claude-3.5-sonnet"
        
        self.models = {
            ModelTier.TIER_1_SIMPLE: local_model,  # Local
            ModelTier.TIER_2_RAG: local_model,     # Local (same model for simplicity)
            ModelTier.TIER_3_COMPLEX: cloud_model  # Cloud
        }

    def classify_complexity(self, query: str) -> ModelTier:
        """
        Uses a small, fast model to classify the query complexity.
        """
        prompt = f"""
        Classify the following user query into one of these tiers:
        1. TIER_1: Simple greetings, factual lookups, basic questions.
        2. TIER_2: Questions requiring context, summarization, or information retrieval (RAG).
        3. TIER_3: Complex reasoning, coding tasks, math, or creative writing.
        
        Query: {query}
        
        Return ONLY the tier name (TIER_1, TIER_2, or TIER_3).
        """
        
        try:
            response = self.local_client.chat.completions.create(
                model=self.models[ModelTier.TIER_1_SIMPLE],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            
            if "TIER_3" in result: return ModelTier.TIER_3_COMPLEX
            if "TIER_2" in result: return ModelTier.TIER_2_RAG
            return ModelTier.TIER_1_SIMPLE
            
        except Exception as e:
            print(f"Classification failed: {e}. Defaulting to TIER_2.")
            return ModelTier.TIER_2_RAG

    @observe(name="llm_generation")
    def generate(self, prompt: str, tier: ModelTier = None, system_prompt: str = None) -> str:
        """
        Generates a response using the selected model tier.
        """
        if not tier:
            # Auto-route if not specified
            # Note: In a full agent flow, the agent might decide the tool/model. 
            # This is for direct generation.
            tier = ModelTier.TIER_2_RAG 

        model_id = self.models[tier]
        client = self.cloud_client if tier == ModelTier.TIER_3_COMPLEX else self.local_client
        
        # Tag the trace with the model used for cost tracking in Langfuse
        # SDK v3: Use get_client().update_current_trace() instead of langfuse_context
        get_client().update_current_trace(
            tags=[tier.value, "production"],
            metadata={"model": model_id, "provider": "local" if client == self.local_client else "openrouter"}
        )
        
        print(f"Routing to: {tier.value} ({model_id})")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Generation failed on {tier.value}: {e}"
