from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage

class RAGState(TypedDict):
    """
    Represents the state of the RAG agent, including conversation history and source data.
    """
    messages: List[Union[HumanMessage, AIMessage]]
    source: str
    url: str
    pdf_path: str
    question: str
    vectorstore: object