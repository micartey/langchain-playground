from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

rag_output = os.popen('python rag.py "What can you tell me about france?"').read()

# Sentences to compare
sentences = [
    "The capital of France is Paris.",
    rag_output
]

# Generate embeddings
embeddings = model.encode(sentences)

# Calculate cosine similarity
sim = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print(f"Cosine Similarity: {sim[0][0]}")
