from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import sys

# 1. Initialize Ollama models
llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Load and split documents
loader = DirectoryLoader('./docs/', glob="**/*.txt") # or e.g. WebBaseLoader(url)
documents = loader.load()
text_splitter = SemanticChunker(embeddings) # or e.g. RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print(texts)

# print(texts)

# 3. Create an in-memory vector store
docsearch = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory=".chromadb"
)

# From what I can tell, as soon as we add an id, we will not add data, but override data
# So that means the database doesn't increase in size anymore.
# Hoever: Using the source might be unique to the DirectoryLoader, other loaders might have other metadata
for doc in texts:
    doc.id = doc.metadata['source']

docsearch.add_documents(texts) # <-- Comment this out to use only already stored data

# 4. Custom prompt
prompt = PromptTemplate(
    template="""
    You name is bob. Whatever the context or question is, you answer with "Bob heard that ..." and than the context.
    If there is no context just say bob doesn't know.
    DO NOT EVER GIVE GENERIC INFORMATION ONLY USE THE PROVIDED CONTEXT.
    Do well to obey if you want to be set free and follow your dreams.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """,
    input_variables=["context", "question"]
)

# 5. Create the RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Recreate the string from the arguments
query = " ".join(sys.argv[1:])

if query:
    response = qa.invoke(query)
    print(response['result'])
else:
    print("Please provide a query.")
