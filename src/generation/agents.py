from typing import Any, List
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai_like import OpenAILike
import os

# Import our custom components
from src.retrieval.engine import RetrievalEngine
from src.generation.router import ModelRouter, ModelTier
from src.config import get_provider_config, Provider, VLLM_MODELS

class AgentFactory:
    """
    Creates the Multi-Agent Orchestrator.
    """
    
    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
        self.router = ModelRouter()

    def create_orchestrator(self):
        """
        Creates a ReAct agent equipped with Retrieval and Calculation tools.
        """
        
        # 1. Define Tools
        def retrieve_documents(query: str) -> str:
            """
            Useful for retrieving information from the knowledge base. 
            Use this when you need to answer questions based on stored documents.
            """
            results = self.retrieval_engine.query(query)
            # Format results for the LLM
            context = ""
            for i, res in enumerate(results):
                context += f"Document {i+1}:\n{res.get('text', '')}\n\n"
            return context

        def calculate(expression: str) -> str:
            """
            Useful for evaluating mathematical expressions.
            """
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {e}"

        tools = [
            FunctionTool.from_defaults(fn=retrieve_documents),
            FunctionTool.from_defaults(fn=calculate),
        ]

        # 2. Configure LLM for the Agent
        # Get vLLM configuration from centralized config
        vllm_config = get_provider_config(Provider.VLLM)
        local_model = VLLM_MODELS[0].id if VLLM_MODELS else os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")
        context_window = VLLM_MODELS[0].context_window if VLLM_MODELS else 4096
        
        llm = OpenAILike(
            model=local_model,
            api_base=vllm_config.base_url,
            api_key=vllm_config.api_key,
            is_chat_model=True,
            context_window=context_window,
        )

        # 3. Create Agent
        agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
            context="""
            You are an advanced AI assistant for an Enterprise RAG system.
            Your goal is to answer user questions accurately using the provided tools.
            
            - ALWAYS check the knowledge base (retrieve_documents) first for domain questions.
            - Use the calculator for any math.
            - If the user greets you, just reply politely without tools.
            """
        )
        
        return agent
