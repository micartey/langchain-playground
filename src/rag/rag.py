from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import sys

# 1. Initialize Ollama models
llm = OllamaLLM(model="llama3", temperature = 0)
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
    You name is bob. Whatever the context or question is, you answer with "Bob heard that" and than the context.
    If there is no context, say bob doesn't know.
    Not all contet is useful, you need to filter it for yourself.
    THe information you will re-word should be a fitting answer to the question.
    It is very important that you only summarize your provided content and do not add information.
    Your task is to select the most fitting "Context" and reword it.
    No additional information.
    If the context is empty, you simply don't know.
    If the context differs in domain, you simply don't know.

    Context: {context}

    Question: {question}

    Response without additional Information:
    """,
    input_variables=["context", "question"]
)

# 4. Create the RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        # search_kwargs={"k": 2} # By default it tries to get 4 results
    ),
    chain_type_kwargs={"prompt": prompt}
)

# Recreate the string from the arguments
query = " ".join(sys.argv[1:])

if query:
    response = qa.invoke(query)
    print(response['result'])
else:
    print("Please provide a query.")
