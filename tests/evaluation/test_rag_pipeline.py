import pytest
import json
import os
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevanceMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# Import RAG Pipeline components
from src.retrieval.engine import RetrievalEngine
from src.generation.router import ModelRouter, ModelTier

# Initialize components
retrieval_engine = RetrievalEngine()
model_router = ModelRouter()

def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, "r") as f:
        return json.load(f)

@pytest.mark.parametrize("example", load_dataset())
def test_rag_pipeline(example):
    input_query = example["input"]
    expected_output = example["expected_output"]
    
    # 1. Run Retrieval
    # We need to extract the text from the retrieved nodes
    retrieved_results = retrieval_engine.query(input_query)
    retrieval_context = [res.get("text", "") for res in retrieved_results]
    
    # 2. Run Generation
    # Construct a prompt with context (simulating the RAG flow)
    context_str = "\n\n".join(retrieval_context)
    prompt = f"Context:\n{context_str}\n\nQuestion: {input_query}\nAnswer:"
    
    # Use Tier 2 (Local) or Tier 3 (Cloud) for the test generation
    # For evaluation, we often want to test the best possible output, or the production setting.
    actual_output = model_router.generate(prompt, tier=ModelTier.TIER_2_RAG)

    # 3. Define DeepEval Metrics
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4", # Use a strong model for evaluation (or local if configured)
        include_reason=True
    )
    
    answer_relevance = AnswerRelevanceMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )
    
    contextual_recall = ContextualRecallMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )

    # 4. Create Test Case
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )

    # 5. Assert Metrics
    # assert_test checks if all metrics pass their thresholds
    assert_test(test_case, [faithfulness, answer_relevance, contextual_recall])
