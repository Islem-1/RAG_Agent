# -*- coding: utf-8 -*-
# graph.py
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from nodes import load_pdf_node, load_web_node, answer_node
from agent_schema import RAGState

# Create the graph
graph = StateGraph(RAGState)

# Add nodes to the graph
graph.add_node("load_pdf_node", load_pdf_node)
graph.add_node("load_web_node", load_web_node)
graph.add_node("answer_node", answer_node)

# Define conditional transitions
def select_source(state: RAGState):
    """Determines the graph path based on the selected source."""
    source = state.get('source')
    if source == 'local_pdf': # Correction ici
        return "load_pdf_node"
    elif source == 'web':
        return "load_web_node"
    else:
        # Fallback for an invalid source
        return "answer_node"

graph.add_conditional_edges(START, select_source, {
    "load_pdf_node": "load_pdf_node",
    "load_web_node": "load_web_node"
})

# Define edges
graph.add_edge("load_pdf_node", "answer_node")
graph.add_edge("load_web_node", "answer_node")
graph.add_edge("answer_node", END)

# Compile the agent
agent = graph.compile()
print(" LangGraph agent compiled successfully.")
