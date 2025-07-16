from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import OllamaModel
import os

# Ensure the Ollama class can find the service
# By default it looks at http://localhost:11434
# os.environ["OLLAMA_HOST"] = "http://localhost:11434" # Optional: if not default

# 1. Get the LLM output
rag_output = os.popen('python rag.py "What can you tell me about france?"').read()

# 2. Create your test case
# This is an input <---> output object
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output=rag_output,
    expected_output="The capital of France is Paris"
)

# 3. Define the metric and pass the custom model
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - the collective quality of all sentences in the actual output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=OllamaModel(model="mistral"),
    threshold=0.5
)

# 4. Run the measurement
coherence_metric.measure(test_case)

print(f"Score: {coherence_metric.score}")
print(f"Reason: {coherence_metric.reason}")
