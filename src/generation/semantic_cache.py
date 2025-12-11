import os
from typing import Optional
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class SemanticCache:
    """
    Implements Semantic Caching using Qdrant.
    Stores (Query Embedding) -> (Response).
    """
    
    def __init__(self, collection_name: str = "semantic_cache", threshold: float = 0.95):
        self.collection_name = collection_name
        self.threshold = threshold
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        
        # Use local HuggingFace embedding model (no API key required)
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding dimension
                    distance=models.Distance.COSINE,
                )
            )

    def check(self, query: str) -> Optional[str]:
        """
        Checks the cache for a semantically similar query.
        Returns the cached response if found, else None.
        """
        query_vector = self.embed_model.get_text_embedding(query)
        
        # Use query_points (new API) instead of deprecated search()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
            score_threshold=self.threshold,
            with_payload=True
        )
        
        if results.points:
            print(f"Cache Hit! Score: {results.points[0].score}")
            return results.points[0].payload.get("response")
        
        return None

    def add(self, query: str, response: str):
        """
        Adds a query-response pair to the cache.
        """
        query_vector = self.embed_model.get_text_embedding(query)
        
        # Use a hash of the query as ID to avoid duplicates
        import hashlib
        doc_id = hashlib.md5(query.encode()).hexdigest()
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=query_vector,
                    payload={"query": query, "response": response}
                )
            ]
        )
