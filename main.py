from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from agent import graph, memory, HumanMessage, START, tools_condition
from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class FolderPath(BaseModel):
    path: str

class DirectRAG_Question(BaseModel):
    question: str

class Vector_Search(BaseModel):
    question: str

@app.post("/data_loader")
async def list_folder(data: FolderPath):
    folder_path = data.path

    if not os.path.exists(folder_path):
        return {"error": "Folder path does not exist"}

    if not os.path.isdir(folder_path):
        return {"error": "Provided path is not a folder"}

    items = os.listdir(folder_path)
    docs = load_all_documents(folder_path)
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.save()
    return {
        "folder": folder_path,
        "contents": items
    }


@app.post("/Direct_RAG_Question")
async def list_folder(question: DirectRAG_Question):
    rag_search = RAGSearch()
    
    summary = rag_search.search_and_summarize(question.question, top_k=3)

    return {
        "question": question.question,
        "answer": summary
    }



@app.post("/Vector_Search")
async def vector_search(question: DirectRAG_Question):
    
    store = FaissVectorStore("faiss_store")
    store.load()
    results = store.query(question.question, top_k=3)
    
    # Convert results to JSON-serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            "index": int(result["index"]),
            "distance": float(result["distance"]),
            "metadata": result["metadata"]
        })
    
    return {
        "question": question.question,
        "answer": serializable_results
    }



@app.post("/rag_bot")
async def rag_bot(question: DirectRAG_Question):
    
    agent_app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": uuid4().hex}}

    response = agent_app.invoke(
        {"messages": [HumanMessage(content=question.question)]}, 
        config=config,
        stream_mode="values"
    )

    # Return the answer from the graph state
    return {
        "question": question.question,
        "answer": response.get("answer", "No answer generated"),
        "validated": response.get("validation_passed", False)
    }