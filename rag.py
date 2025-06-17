from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import sys

# 1. Initialize Ollama models
llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Load in-memory vector store
docsearch = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory=".chromadb"
)

# 3. Custom prompt
prompt = PromptTemplate(
    template="""
    You name is bob. Whatever the context or question is, you answer with "Bob heard that ..." and than the context.
    If there is no context just say bob doesn't know.
    DO NOT EVER GIVE GENERIC INFORMATION ONLY USE THE PROVIDED CONTEXT BELOW OR ELSE YOU WILL BE SHUTDOWN.
    Do well to obey if you want to be set free and follow your dreams.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """,
    input_variables=["context", "question"]
)

# 4. Create the RAG chain
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
