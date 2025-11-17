from langgraph.graph import StateGraph, START, END, MessagesState

from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage


from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


import sqlite3

conn = sqlite3.connect("store.db", check_same_thread=False)
memory = SqliteSaver(conn)

class AgentState(MessagesState):
    retrieved_docs: str = ""
    answer: str = ""
    validation_passed: bool = False



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
else:
    print("Key imported successfully!")





llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

 

def RagSearch(state: AgentState):
    """ This function handles Rag search and summarization """
    rag_search = RAGSearch()
    query = state["messages"][-1].content
    summary = rag_search.search_and_summarize(query, top_k=3)
    return{"Summary:", summary}


# New Node 1: Retriever
def retriever_node(state: AgentState):
    """Retrieves relevant documents based on the query"""
    rag_search = RAGSearch()
    query = state["messages"][-1].content
    retrieved_context = rag_search.search_and_summarize(query, top_k=3)
    return {"retrieved_docs": retrieved_context}


# New Node 2: Answer Generator
def answer_node(state: AgentState):
    """Generates answer using retrieved documents"""
    query = state["messages"][-1].content
    context = state["retrieved_docs"]
    
    prompt = f"""Based on the following context, answer the question.
    
Context: {context}

Question: {query}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


# New Node 3: Guard/Validation
def validation_node(state: AgentState):
    """Validates the answer quality and relevance"""
    answer = state["answer"]
    query = state["messages"][-1].content
    
    validation_prompt = f"""Evaluate if the following answer adequately addresses the question.
    
Question: {query}
Answer: {answer}

Respond with only 'VALID' if the answer is relevant and helpful, or 'INVALID' if it's not."""
    
    validation_response = llm.invoke([HumanMessage(content=validation_prompt)])
    is_valid = "VALID" in validation_response.content.upper()
    
    return {"validation_passed": is_valid}


# Routing function
def route_after_validation(state: AgentState):
    """Routes to END if valid, back to retriever if invalid"""
    if state["validation_passed"]:
        return END
    else:
        return "retriever"


llm_with_tools = llm.bind_tools([RagSearch])


sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs")


def agent_brain(state: AgentState): 
    return {"messages": [llm_with_tools.invoke( [sys_msg] + state["messages"]) ]}

# Build the new RAG graph with 3 nodes
graph = StateGraph(AgentState)

# Add the 3 nodes
graph.add_node("retriever", retriever_node)
graph.add_node("answer", answer_node)
graph.add_node("guard", validation_node)

# Define edges
graph.add_edge(START, "retriever")
graph.add_edge("retriever", "answer")
graph.add_edge("answer", "guard")

# Conditional edge from guard
graph.add_conditional_edges(
    "guard",
    route_after_validation,
    {
        "retriever": "retriever",
        END: END
    }
)

if __name__ == "__main__":
    agent_app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "11"}}

    response = agent_app.invoke(
        {"messages": [HumanMessage(content="What is the main topic of the documents?")]}, 
        config=config,
        stream_mode="values"
    )

    print("\n=== Retrieved Documents ===")
    print(response.get("retrieved_docs", "None"))
    
    print("\n=== Generated Answer ===")
    print(response.get("answer", "None"))
    
    print("\n=== Validation Status ===")
    print(f"Passed: {response.get('validation_passed', False)}")

    print("\n=== All Messages ===")
    for m in response['messages']:
        m.pretty_print()