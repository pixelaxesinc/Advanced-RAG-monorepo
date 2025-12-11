"""
Langfuse SDK v3 Observability Configuration

Note: LlamaIndex instrumentation via OpenInference requires llama-index >= 0.11.0
For older versions, we skip LlamaIndex instrumentation but keep Langfuse tracing
via the @observe decorator which works for all Python code.
"""
import os
from langfuse import get_client


def setup_observability():
    """
    Configures Langfuse SDK v3 observability.
    
    Note: The @observe decorator in main.py provides tracing for all endpoints.
    LlamaIndex-specific instrumentation is optional and requires compatible versions.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not (public_key and secret_key):
        print("Langfuse credentials not found. Observability disabled.")
        return None

    print(f"Initializing Langfuse SDK v3 observability (host: {host})...")
    
    # Get the Langfuse client (automatically configured from env vars)
    langfuse = get_client()
    
    # Verify connection
    try:
        auth_result = langfuse.auth_check()
        print(f"Langfuse auth check: {auth_result}")
    except Exception as e:
        print(f"Langfuse auth check failed: {e}")
        return None
    
    # Try to set up LlamaIndex instrumentation if compatible versions are installed
    # This is optional - the @observe decorator provides tracing without this
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        LlamaIndexInstrumentor().instrument()
        print("LlamaIndex instrumentation enabled via OpenInference")
    except ImportError:
        print("Note: LlamaIndex instrumentation not available (openinference not installed)")
        print("  Langfuse tracing still works via @observe decorators")
    except Exception as e:
        # Catch version compatibility errors gracefully
        print(f"Note: LlamaIndex instrumentation skipped due to version incompatibility")
        print(f"  Details: {type(e).__name__}: {str(e)[:100]}")
        print("  Langfuse tracing still works via @observe decorators")
    
    return langfuse
