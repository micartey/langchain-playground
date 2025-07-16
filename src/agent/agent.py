from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chains import RetrievalQA

# Initialize LLM and embeddings
llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 1. URLs for "knowledge"
urls = [
    "https://www.dfki.de/web/news/effizienteres-recycling-dank-ki",
    "https://www.dfki.de/web/news/grenzueberschreitende-quantenkraft-projekt-upquantval",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 2 Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

# 3 Embed Chunks and save to vectorstore
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

def get_human_message(state):
    """
    Get the last user prompt
    """

    messages = state["messages"]
    last_message = messages[-1]

    # Get the content of the message
    if isinstance(last_message, HumanMessage):
        content = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        content = last_message["content"]
    else:
        content = str(last_message)

    return content

# Define the agent nodes and functions
def should_use_tools(state):
    """
    Determine whether to use tools or not based on the query.
    Returns a string key for routing.

    This is so goddamn stupid... I am ashamed
    """

    content = get_human_message(state)

    # Check if we need to retrieve information from the DFKI projects
    if "DFKI" in content or "project" in content:
        print("Will use retrieval tool")
        return "retrive"

    print("No retrieval needed, responding directly")
    return "default"

def generate_response(state):
    """
    Generate a response using the LLM.
    """
    # print("State in generate_response:", state)

    response = llm.invoke(state['messages']) # <-- Why do I even need to do this on my own?!

    assistant_message = {"role": "assistant", "content": response}
    return {"messages": state["messages"] + [assistant_message]}

def retrive_dfki_projects(state):
    """
    Not quite a tool
    """

    print("Getting context from RAG system")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        # chain_type_kwargs={"prompt": prompt} # Ignore the prompt atm
    )

    response = qa.invoke(get_human_message(state))
    assistant_message = {"role": "tool", "tool_call_id": "-1", "content": response}
    return {"messages": state["messages"] + [assistant_message]}


# 4. Create the graph
workflow = StateGraph(MessagesState)
workflow.add_node("retrive_dfki_projects", retrive_dfki_projects)
workflow.add_node("generate", generate_response)

# 5. Add edges
workflow.add_conditional_edges(
    START,
    should_use_tools,
    {
        # If tools are returned, use the tools node
        "retrive": "retrive_dfki_projects",
        # Otherwise, go straight to generate
        "default": "generate"
    }
)

workflow.add_edge("retrive_dfki_projects", "generate")
workflow.add_edge("generate", END)

# 6. Compile the graph
graph = workflow.compile()
with open('agent_graph.md', 'w') as file:
    file.write(graph.get_graph().draw_mermaid())

print("Finished building graph")

#################
# Test examples #
#################

output = graph.invoke({"messages": [{"role": "user", "content": "When was Albert Einstein born?"}]})
print(output['messages'][-1].content)

output = graph.invoke({"messages": [{"role": "user", "content": "Tell me about DFKI project UPQuantVal"}]})
print(output['messages'][-1].content)
