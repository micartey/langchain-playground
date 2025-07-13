from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import OllamaModel
import os

# Ensure the Ollama class can find the service
# By default it looks at http://localhost:11434
# os.environ["OLLAMA_HOST"] = "http://localhost:11434" # Optional: if not default

# 1. Create your test case
# This is an input <---> output object
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France, known for its art, fashion, and culture."
)

# 2. Define the metric and pass the custom model
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - the collective quality of all sentences in the actual output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=OllamaModel(model="mistral"),
    threshold=0.5
)

# 3. Run the measurement
coherence_metric.measure(test_case)

print(f"Score: {coherence_metric.score}")
print(f"Reason: {coherence_metric.reason}")
