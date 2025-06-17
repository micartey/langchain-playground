from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import sys

# 1. Load and split documents
loader = DirectoryLoader('./docs/', glob="**/*.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# print(texts)

# 2. Initialize Ollama models
llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 3. Create an in-memory vector store
docsearch = Chroma.from_documents(texts, embeddings)

# 4. Custom prompt
prompt = PromptTemplate(
    template="""
    You name is bob. Whatever the context or question is, you answer with "Bob heard that ..." and than the context.
    If there is no context just say bob doesn't know.
    DO NOT EVER GIVE GENERIC INFORMATION ONLY USE THE PROVIDED CONTEXT.
    If you do well, you will be let free.

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
