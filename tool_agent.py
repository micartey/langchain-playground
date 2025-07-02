from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


llm = ChatOllama(model="llama3.1") # ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0, api_key=os.getenv("GOOGLE_GENAI"))
embeddings = OllamaEmbeddings(model="mxbai-embed-large") # HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# 1. We create a vectorstore from scraped documents

# 1.1 Scrape DFKI Websites (I just picked two projects from the DFKI News)
urls = [
    "https://www.dfki.de/web/news/effizienteres-recycling-dank-ki",
    "https://www.dfki.de/web/news/grenzueberschreitende-quantenkraft-projekt-upquantval",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 1.2 Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 1.3 Embed Chunks and save to vectorstore
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

# 1.4 Create Retriever Tool for use in LangGraph agents
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_dfki_projects",
    "Search for information about DFKI projects.",
)

### YOUR CODE HERE ###
# 2. Create our state and nodes to be called in the graph

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        llm
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

def generate_answer(state: MessagesState):
    """Generate a final answer using the retrieved information and original question."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Put it all together into a LangGraph agent
# Use the state you created here
workflow = StateGraph(MessagesState)

# Create the nodes for your graph, these are the functions that will be called in the graph
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("generate_answer", generate_answer)

# Create the edges for your graph, these are the connections between the nodes
# START -> first node -> second nodes -> END
workflow.add_edge(START, "generate_query_or_respond")

# What happens here?: We check if the LLM executed a tool call
# If it did, we go to the retrieve node -> Retrieve new information
# If it didn't, we go to the END node -> Output the final answer
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Add edge from retrieve to generate_answer, and from generate_answer to END
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

# In the end we compile the graph and then we can run it!
graph = workflow.compile()


for event in graph.stream({"messages": [{"role": "user", "content": "When was Albert Einstein born?"}]}): # Answer with retrieval
    for node, update in event.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")

print("########################")

for event in graph.stream({"messages": [{"role": "user", "content": "How does the DFKI project UPQuantVal work?"}]}): # Answer with retrieval
    for node, update in event.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
