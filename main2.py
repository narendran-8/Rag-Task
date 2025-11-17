from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


# Example usage
if __name__ == "__main__":
    
    # docs = load_all_documents("data")
    # store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # store.save()
    # store.load()
    # print(store.query("how many words 'Attacks' ?, top_k=3))
    rag_search = RAGSearch()
    query = 'how many words "Attacks" ?'
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
