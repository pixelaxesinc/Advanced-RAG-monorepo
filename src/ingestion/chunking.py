from typing import List, Dict, Any
from llama_index.core.node_parser import HierarchicalNodeParser, SemanticSplitterNodeParser
from llama_index.core.schema import Document, BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

class Chunker:
    """
    Implements Hierarchical and Semantic chunking strategies.
    """
    
    def __init__(self, strategy: str = "hierarchical"):
        self.strategy = strategy
        
        # Use local HuggingFace embedding model (no API key required)
        # Same model as SemanticCache for consistency
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[BaseNode]:
        """
        Chunks the text based on the selected strategy.
        """
        document = Document(text=text, metadata=metadata)
        
        if self.strategy == "hierarchical":
            return self._hierarchical_chunking([document])
        elif self.strategy == "semantic":
            return self._semantic_chunking([document])
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _hierarchical_chunking(self, documents: List[Document]) -> List[BaseNode]:
        """
        Parent-Child chunking:
        - Parent: 1024 tokens
        - Child: 256 tokens (with overlap)
        """
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[1024, 512, 256],
            chunk_overlap=20
        )
        return node_parser.get_nodes_from_documents(documents)

    def _semantic_chunking(self, documents: List[Document]) -> List[BaseNode]:
        """
        Splits text based on semantic similarity changes.
        """
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
        return splitter.get_nodes_from_documents(documents)
