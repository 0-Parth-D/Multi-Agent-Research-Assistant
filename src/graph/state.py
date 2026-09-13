from typing import TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str #Retriever
    retrieved_docs: list[Document] #Retriever
    current_answer: str #Summarizer
    critic_decision: str #Critic   
    iteration_count: int #Critic   
    timestamps: dict[str, float] #Retriever
    confidence_score: float #Summarizer