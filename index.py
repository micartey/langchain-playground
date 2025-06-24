from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Initialize Ollama models
llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

print("Reading files from directory and splitting...")

# 2. Load and split documents
loader = DirectoryLoader('./docs/', glob="**/*.txt") # or e.g. WebBaseLoader(url)
documents = loader.load()
text_splitter = SemanticChunker(embeddings) # or e.g. RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print("Storing to db...")

# 3. Create an in-memory vector store
docsearch = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory=".chromadb"
)

# From what I can tell, as soon as we add an id, we will not add data, but override data
# So that means the database doesn't increase in size every run anymore.
# However: Using the source might be unique to the DirectoryLoader, other loaders might have other metadata
for doc in texts:
    doc.id = doc.metadata['source']

docsearch.add_documents(texts) # <-- Comment this out to use only already stored data

print("Done")
