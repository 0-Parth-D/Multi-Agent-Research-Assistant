from load_dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time
from src.graph.state import AgentState

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = None

    try:
        vs = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )

    except Exception as e:
        print(f"Failed to load Chroma store: {e}")

    if vs is None:
        raise RuntimeError(
            "No vector store found. Run ingestion first."
        )
    
    return vs

def get_retriever(k=5):
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever

def retriever_node(state: AgentState) -> dict:
    """
    LangGraph node: read query from state, retrieve docs from Chroma,
    and return a partial state update with retrieved_docs + timestamps.
    """
    old_timestamps = state.get("timestamps", {})
    start = time.time()

    retriever = get_retriever()
    results = retriever.get_relevant_documents(state["query"])

    end = time.time()
    diff = end - start

    new_timestamps = dict(old_timestamps)
    new_timestamps["retriever_s"] = diff

    return {
        "retrieved_docs": results,
        "timestamps": new_timestamps
    }