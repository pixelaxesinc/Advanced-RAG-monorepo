from typing import List, Dict, Any
from langfuse import observe  # SDK v3
from .qdrant_client import QdrantRetriever
from .query_processor import QueryProcessor
from .reranker import Reranker

class RetrievalEngine:
    """
    Orchestrates the full retrieval pipeline:
    Query -> Rewrite/HyDE -> Hybrid Search -> Rerank -> Top K
    """
    
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.processor = QueryProcessor()
        self.reranker = Reranker()

    @observe(name="retrieval_pipeline")
    def query(self, user_query: str, use_hyde: bool = True) -> List[Dict[str, Any]]:
        print(f"Original Query: {user_query}")
        
        # 1. Query Processing
        rewritten_query = self.processor.rewrite_query(user_query)
        print(f"Rewritten Query: {rewritten_query}")
        
        search_query = rewritten_query
        if use_hyde:
            hyde_doc = self.processor.generate_hyde_answer(rewritten_query)
            print(f"HyDE Document generated.")
            # We might search with the HyDE doc, or mix it. 
            # Often searching with the HyDE doc directly is effective.
            search_query = hyde_doc

        # 2. Hybrid Retrieval
        # Retrieve a broad set (e.g., 50)
        initial_results = self.retriever.search(search_query, limit=50)
        print(f"Retrieved {len(initial_results)} candidates.")

        # 3. Re-ranking
        # Select top 5-10
        final_results = self.reranker.rerank(user_query, initial_results, top_k=5)
        
        return final_results
