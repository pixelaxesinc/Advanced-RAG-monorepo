"""
Centralized Configuration for Model Providers and Models.

This file defines all available models for each provider (vLLM and OpenRouter).
Add or remove models here to update what's available in Open WebUI.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class Provider(str, Enum):
    """Available LLM providers."""
    VLLM = "vllm"
    OPENROUTER = "openrouter"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str                          # Model identifier (used in API calls)
    name: str                         # Human-readable display name
    provider: Provider                # Which provider serves this model
    context_window: int = 4096        # Context window size
    is_chat_model: bool = True        # Whether it's a chat model
    description: str = ""             # Optional description


# =============================================================================
# VLLM MODELS - Local models served via vLLM
# =============================================================================
# Add your locally running vLLM models here
# The model id should match exactly what you configured in vLLM

VLLM_MODELS: List[ModelConfig] = [
    ModelConfig(
        id=os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        name="Local RAG Agent (Qwen 2.5 0.5B)",
        provider=Provider.VLLM,
        context_window=32768,
        description="Local model served via vLLM - fast inference, no thinking mode"
    ),
    # Add more vLLM models here as needed:
    # ModelConfig(
    #     id="mistralai/Mistral-7B-Instruct-v0.3",
    #     name="Mistral 7B Instruct",
    #     provider=Provider.VLLM,
    #     context_window=32768,
    # ),
]


# =============================================================================
# OPENROUTER MODELS - Cloud models via OpenRouter API
# =============================================================================
# Add your preferred OpenRouter models here
# Full list: https://openrouter.ai/models

OPENROUTER_MODELS: List[ModelConfig] = [
    # --- FREE MODELS ---
    ModelConfig(
        id="openai/gpt-oss-120b:free",
        name="GPT-120B OSS (Free)",
        provider=Provider.OPENROUTER,
        context_window=131072,
        description="OpenAI's efficient small model - FREE tier"
    ),

    # --- PAID MODELS (uncomment as needed) ---
    # ModelConfig(
    #     id="anthropic/claude-3.5-sonnet",
    #     name="Claude 3.5 Sonnet",
    #     provider=Provider.OPENROUTER,
    #     context_window=200000,
    #     description="Anthropic's best model for complex reasoning"
    # ),
    # ModelConfig(
    #     id="openai/gpt-4o",
    #     name="GPT-4o",
    #     provider=Provider.OPENROUTER,
    #     context_window=128000,
    #     description="OpenAI's flagship multimodal model"
    # ),
    # ModelConfig(
    #     id="anthropic/claude-3-opus",
    #     name="Claude 3 Opus",
    #     provider=Provider.OPENROUTER,
    #     context_window=200000,
    #     description="Anthropic's most powerful model"
    # ),
]


# =============================================================================
# COMBINED MODEL REGISTRY
# =============================================================================

def get_all_models() -> List[ModelConfig]:
    """Returns all available models from all providers."""
    return VLLM_MODELS + OPENROUTER_MODELS


def get_models_by_provider(provider: Provider) -> List[ModelConfig]:
    """Returns models for a specific provider."""
    all_models = get_all_models()
    return [m for m in all_models if m.provider == provider]


def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
    """Finds a model by its ID."""
    for model in get_all_models():
        if model.id == model_id:
            return model
    return None


def get_default_model() -> ModelConfig:
    """Returns the default model (first vLLM model or first available)."""
    if VLLM_MODELS:
        return VLLM_MODELS[0]
    if OPENROUTER_MODELS:
        return OPENROUTER_MODELS[0]
    raise ValueError("No models configured!")


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    base_url: str
    api_key: str


def get_provider_config(provider: Provider) -> ProviderConfig:
    """Returns the configuration for a specific provider."""
    if provider == Provider.VLLM:
        return ProviderConfig(
            base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "local-vllm-key")
        )
    elif provider == Provider.OPENROUTER:
        return ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
