from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

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

def decide_action(messages):
    """
    Decide whether to retrieve information or answer directly.
    """
    print("I shall decide the path!")
    print(messages)

    # Example logic: Check if the message contains a keyword indicating retrieval
    # print(messages['messages'][-1]["content"])
    user_message = messages['messages'][-1].content
    if "DFKI" in user_message or "project" in user_message:
        return "tools"  # Indicating retrieval

    return END

def respond(state):
    return {"messages": [llm.invoke(state["messages"])]}

# Use the state you created here
workflow = StateGraph(MessagesState)

# Create the nodes for your graph, these are the functions that will be called in the graph
workflow.add_node("generate_query_or_respond", ToolNode([decide_action]))
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("respond", respond)

# Create the edges for your graph, these are the connections between the nodes
# START -> first node -> second nodes -> END
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_edge("generate_query_or_respond", "respond")
workflow.add_edge("respond", END)

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

# Which edge is still missing?

# In the end we compile the graph and then we can run it!
graph = workflow.compile()


# Run the graph
output = graph.invoke({"messages": [{"role": "user", "content": "When was Albert Einstein born?"}]}) # Answer without retrieval
print(output)
output = graph.invoke({"messages": [{"role": "user", "content": "How does the DFKI project UPQuantVal work?"}]}) # Answer with retrieval
print(output)
