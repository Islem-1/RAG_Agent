# nodes.py
from langchain_core.messages import AIMessage, HumanMessage
from rag_pipline import load_pdf, load_web, answer_semantic_question
from agent_schema import RAGState

def load_pdf_node(state: RAGState) -> RAGState:
    """Loads a vector store from a PDF file."""
    print("Agent: Loading vector store from PDF...")
    try:
        if state.get('pdf_path'): 
            state['vectorstore'] = load_pdf(state['pdf_path'])
        
        # Add a check to ensure vectorstore was successfully created
        if not state.get('vectorstore'):
            print("Erreur: Le chargement du PDF a échoué. Aucune base de données vectorielle créée.")
            state['messages'].append(AIMessage(content="Désolé, je ne peux pas charger ce fichier PDF. Il est peut-être corrompu ou vide."))
    except Exception as e:
        print(f"Erreur lors du chargement du PDF: {e}")
        state['messages'].append(AIMessage(content="Désolé, une erreur s'est produite lors du chargement du PDF."))
        state['vectorstore'] = None
    return state

def load_web_node(state: RAGState) -> RAGState:
    """Loads a vector store from a web page URL."""
    print("Agent: Loading vector store from web page...")
    try:
        if state.get('url'):
            state['vectorstore'] = load_web(state['url'])
        
        # Add a check to ensure vectorstore was successfully created
        if not state.get('vectorstore'):
            print("Erreur: Le chargement de la page web a échoué. Aucune base de données vectorielle créée.")
            state['messages'].append(AIMessage(content="Désolé, je ne peux pas charger cette page web. L'URL est peut-être invalide ou le contenu est inaccessible."))
    except Exception as e:
        print(f"Erreur lors du chargement de la page web: {e}")
        state['messages'].append(AIMessage(content="Désolé, une erreur s'est produite lors du chargement de la page web."))
        state['vectorstore'] = None
    return state

def answer_node(state: RAGState) -> RAGState:
    """Uses the vector store to answer the user's question and updates the history."""
    print("Agent: Generating response...")
    question = state.get('question')
    vectorstore = state.get('vectorstore')
    
    # Append the user's question to the history
    if question:
        state['messages'].append(HumanMessage(content=question))
    
    if vectorstore and question:
        # Call the RAG chain to get the answer
        result = answer_semantic_question(vectorstore, question)
        answer_text = result.get('result', result)
        # Append the AI's response to the history
        state['messages'].append(AIMessage(content=answer_text))
    else:
        state['messages'].append(AIMessage(
            content="Désolé, je ne peux pas répondre à cette question. La source n'a pas pu être chargée ou la question est manquante."
        ))
    
    return state
