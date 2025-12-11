import os
import time
import json
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI  # Use standard OpenAI client

# Initialize Langfuse v3 for tracing
# SDK v3 uses get_client() to access the global client instance
# configured via environment variables (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
from langfuse import observe, get_client, Langfuse, propagate_attributes

# Get the global Langfuse client (automatically configured from env vars)
# Set debug=True via environment variable LANGFUSE_DEBUG=true for debugging
langfuse = get_client()

# Import our modules
from src.ingestion.router import IngestionPipeline
from src.retrieval.qdrant_client import QdrantRetriever
from src.generation.semantic_cache import SemanticCache
from src.generation.agents import AgentFactory
from src.observability.config import setup_observability
from src.config import (
    get_all_models, get_model_by_id, get_default_model,
    get_provider_config, Provider, ModelConfig
)

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to flush Langfuse after each request
@app.middleware("http")
async def flush_langfuse_middleware(request, call_next):
    response = await call_next(request)
    # Flush Langfuse traces after each request to ensure they're sent
    # In SDK v3, get_client() returns the global client
    try:
        get_client().flush()
    except Exception as e:
        print(f"Error flushing Langfuse: {e}")
    return response

# Initialize other Components
setup_observability() # Initialize LlamaIndex tracing

ingestion_pipeline = IngestionPipeline()
retriever = QdrantRetriever() # Used for upserting chunks
semantic_cache = SemanticCache()
agent_factory = AgentFactory()
orchestrator = agent_factory.create_orchestrator()

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("Langfuse Configuration:")
    print(f"  LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'not set')}")
    print(f"  LANGFUSE_BASE_URL: {os.getenv('LANGFUSE_BASE_URL', 'not set')}")
    print(f"  LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY', 'not set')[:20]}...")
    print(f"  LANGFUSE_SECRET_KEY: {'set' if os.getenv('LANGFUSE_SECRET_KEY') else 'not set'}")
    print("=" * 50)
    # Verify Langfuse connection
    try:
        auth_result = langfuse.auth_check()
        print(f"Langfuse auth check: {auth_result}")
    except Exception as e:
        print(f"Langfuse auth check failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    # Flush and shutdown Langfuse client before application shutdown
    # SDK v3 recommends calling shutdown() for clean resource release
    client = get_client()
    client.shutdown()
    print("Langfuse shutdown successfully")

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/test-langfuse")
@observe(name="test_langfuse_trace")
def test_langfuse():
    """
    Simple test endpoint to verify Langfuse tracing is working.
    Visit this endpoint, then check Langfuse UI for a trace named 'test_langfuse_trace'.
    """
    import uuid
    test_id = str(uuid.uuid4())
    
    # SDK v3: Use get_client().update_current_trace() instead of langfuse_context
    client = get_client()
    client.update_current_trace(
        name="test_langfuse_trace",
        metadata={"test_id": test_id, "purpose": "verification"},
        tags=["test", "verification"]
    )
    
    print(f"Created test trace with test_id: {test_id}")
    
    return {
        "status": "success",
        "message": "Langfuse trace created! Check the Langfuse UI for a trace named 'test_langfuse_trace'",
        "test_id": test_id
    }

@app.get("/v1/models")
def list_models():
    """
    Returns available models for Open Web UI.
    Reads from centralized config (src/config.py).
    """
    models_data = []
    
    for model in get_all_models():
        models_data.append({
            "id": model.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": model.provider.value,
            "name": model.name,  # Display name for WebUI
            "description": model.description,
            "context_window": model.context_window,
        })
    
    return {
        "object": "list",
        "data": models_data
    }

@app.post("/v1/chat/completions")
@observe(name="chat_completion") # Langfuse tracing
async def chat_completions(
    request: ChatRequest,
    # Open WebUI sends these headers when ENABLE_OPENWEBUI_USER_HEADERS=true
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_name: Optional[str] = Header(None, alias="X-OpenWebUI-User-Name"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
):
    """
    OpenAI-compatible endpoint for Chat.
    Routes to appropriate provider based on selected model.
    Supports streaming responses.
    
    Reads Open WebUI headers for session tracking in Langfuse:
    - X-OpenWebUI-Chat-Id → session_id (groups conversation traces)
    - X-OpenWebUI-User-Id → user_id (identifies the user)
    """
    user_query = request.messages[-1].content
    selected_model_id = request.model
    
    # Look up the selected model in our config
    model_config = get_model_by_id(selected_model_id)
    
    # Fallback to default if model not found
    if not model_config:
        model_config = get_default_model()
        print(f"Model '{selected_model_id}' not found, using default: {model_config.id}")
    
    # SDK v3: Update Langfuse trace with model info and session/user IDs
    client = get_client()
    
    # Build metadata including Open WebUI user info if available
    trace_metadata = {
        "model_id": model_config.id,
        "provider": model_config.provider.value,
        "streaming": request.stream,
    }
    if x_openwebui_user_name:
        trace_metadata["user_name"] = x_openwebui_user_name
    if x_openwebui_user_email:
        trace_metadata["user_email"] = x_openwebui_user_email
    
    # Update trace with session_id and user_id for Langfuse grouping
    client.update_current_trace(
        name=f"chat_{model_config.provider.value}",
        session_id=x_openwebui_chat_id,  # Groups all messages in the same conversation
        user_id=x_openwebui_user_id,      # Identifies the user across sessions
        metadata=trace_metadata,
        tags=[model_config.provider.value, "chat"]
    )
    
    # Propagate session_id and user_id to all child observations
    # This ensures nested spans/generations also have these attributes
    with propagate_attributes(
        session_id=x_openwebui_chat_id,
        user_id=x_openwebui_user_id
    ):
        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                _stream_response(request, model_config, user_query),
                media_type="text/event-stream"
            )
        
        # Non-streaming path
        # 1. Check Semantic Cache
        cached_response = semantic_cache.check(user_query)
        if cached_response:
            get_client().update_current_trace(tags=["cache_hit"])
            return _format_response(cached_response, model_config.id)

        try:
            # 2. Route based on provider
            if model_config.provider == Provider.VLLM:
                # Use RAG Orchestrator for vLLM models (has retrieval & tools)
                response_obj = orchestrator.chat(user_query)
                response_text = str(response_obj)
            else:
                # Direct API call for OpenRouter models
                response_text = await _call_openrouter(request.messages, model_config, stream=False)
            
            # 3. Update Cache (only for substantial responses)
            if len(response_text) > 20:
                semantic_cache.add(user_query, response_text)
                
            return _format_response(response_text, model_config.id)
            
        except Exception as e:
            print(f"Error processing request with model {model_config.id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(request: ChatRequest, model_config: ModelConfig, user_query: str) -> AsyncGenerator[str, None]:
    """
    Streams response tokens as Server-Sent Events (SSE) in OpenAI format.
    """
    response_id = f"chatcmpl-{int(time.time())}"
    full_response = ""
    
    try:
        if model_config.provider == Provider.VLLM:
            # For vLLM with RAG orchestrator - use direct vLLM streaming
            async for chunk in _stream_vllm(request.messages, model_config):
                full_response += chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_config.id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
        else:
            # OpenRouter streaming
            async for chunk in _stream_openrouter(request.messages, model_config):
                full_response += chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_config.id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
        
        # Send final chunk with finish_reason
        final_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_config.id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
        # Cache the full response
        if len(full_response) > 20:
            semantic_cache.add(user_query, full_response)
            
    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


async def _stream_vllm(messages: List[ChatMessage], model_config: ModelConfig) -> AsyncGenerator[str, None]:
    """
    Streams response from vLLM.
    """
    provider_config = get_provider_config(Provider.VLLM)
    
    client = OpenAI(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key
    )
    
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
    
    stream = client.chat.completions.create(
        model=model_config.id,
        messages=formatted_messages,
        temperature=0.7,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def _stream_openrouter(messages: List[ChatMessage], model_config: ModelConfig) -> AsyncGenerator[str, None]:
    """
    Streams response from OpenRouter.
    """
    provider_config = get_provider_config(Provider.OPENROUTER)
    
    client = OpenAI(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key
    )
    
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
    
    stream = client.chat.completions.create(
        model=model_config.id,
        messages=formatted_messages,
        temperature=0.7,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def _call_openrouter(messages: List[ChatMessage], model_config: ModelConfig, stream: bool = False) -> str:
    """
    Makes a direct call to OpenRouter API (non-streaming) with Langfuse logging.
    SDK v3: Uses context managers for generation tracking.
    """
    provider_config = get_provider_config(Provider.OPENROUTER)
    
    openai_client = OpenAI(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key
    )
    
    # Convert our message format to OpenAI format
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
    
    # SDK v3: Use context manager for generation tracking
    client = get_client()
    with client.start_as_current_observation(
        as_type="generation",
        name="openrouter-completion",
        model=model_config.id,
        input=formatted_messages,
        metadata={"provider": "openrouter"}
    ) as generation:
        try:
            response = openai_client.chat.completions.create(
                model=model_config.id,
                messages=formatted_messages,
                temperature=0.7,
            )
            
            output = response.choices[0].message.content
            
            # Log the result to Langfuse
            generation.update(
                output=output,
                usage={
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                    "total": response.usage.total_tokens if response.usage else 0,
                }
            )
            
            return output
        except Exception as e:
            generation.update(output=str(e), level="ERROR")
            raise

def _format_response(content: str, model_name: str) -> ChatResponse:
    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=model_name,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    )

class FeedbackRequest(BaseModel):
    trace_id: str
    score: float # 1.0 for Thumbs Up, 0.0 for Thumbs Down
    comment: Optional[str] = None

@app.post("/v1/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Receives user feedback (Thumbs Up/Down) and sends it to Langfuse.
    SDK v3: Uses get_client() to access the Langfuse client.
    """
    try:
        client = get_client()
        client.score(
            trace_id=request.trace_id,
            name="user_feedback",
            value=request.score,
            comment=request.comment
        )
        return {"status": "success"}
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ingest")
async def ingest_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Endpoint to upload and ingest documents.
    """
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Run ingestion in background
    background_tasks.add_task(_process_ingestion, temp_path)
    
    return {"status": "processing", "filename": file.filename}

def _process_ingestion(file_path: str):
    try:
        print(f"Starting background ingestion for {file_path}")
        nodes = ingestion_pipeline.process_document(file_path)
        
        if nodes:
            retriever.upsert_nodes(nodes)
            print(f"Successfully ingested {len(nodes)} chunks from {file_path}")
        else:
            print(f"No content extracted from {file_path}")
            
    except Exception as e:
        print(f"Ingestion failed for {file_path}: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
