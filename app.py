# app.py
import os
import json
import io
import tempfile
from flask import Flask, render_template, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from werkzeug.utils import secure_filename

# Import du LangGraph agent
from graphs import agent
from agent_schema import RAGState

# --- USER_AGENT pour le loader web ---
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

app = Flask(__name__)

# --- Sauvegarde conversation dans un fichier ---
def save_conversation_to_file(messages: list, filename="conversation.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("Your Conversation Log: \n")
            for message in messages:
                if isinstance(message, HumanMessage):
                    file.write(f"You: {message.content}\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n")
            file.write("End of Conversation\n")
        print(f"Conversation saved to {filename}")
    except Exception as e:
        print(f"Error saving conversation: {e}")


# --- Génération du graphe ---
try:
    print("Generating and saving LangGraph image...")
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    png_data = agent.get_graph().draw_mermaid_png()
    out_file = os.path.join(out_dir, "graph.png")
    with open(out_file, "wb") as f:
        f.write(png_data)
        print(f"Saved graph image to {out_file}")
except Exception as e:
    print(f"Could not generate graph image. Error: {e}")


# --- Page d’accueil ---
@app.route("/")
def index():
    return render_template("index.html")


# --- Endpoint /ask ---
@app.route("/ask", methods=["POST"])
def ask():
    source = request.form.get("source")
    question = request.form.get("question")
    url = request.form.get("url")
    history_json = request.form.get("history", "[]")

    #  Récupère l’historique
    try:
        history_list = json.loads(history_json)
        messages = []
        for item in history_list:
            if item.get("type") == "human":
                messages.append(HumanMessage(content=item.get("content", "")))
            elif item.get("type") == "ai":
                messages.append(AIMessage(content=item.get("content", "")))
    except Exception as e:
        print(f"Erreur historique JSON: {e}")
        messages = []

    # Gérer le PDF uploadé
    pdf_path = None
    if source == "local_pdf" and "pdf_file" in request.files:
        pdf_file = request.files["pdf_file"]
        if pdf_file.filename != "":
            # Utilise un chemin temporaire pour le fichier
            temp_dir = tempfile.gettempdir()
            filename = secure_filename(pdf_file.filename)
            pdf_path = os.path.join(temp_dir, filename)
            pdf_file.save(pdf_path)
            print(f"PDF uploadé et sauvegardé temporairement dans {pdf_path}")
            
    #  Créer l’état initial
    state = RAGState(
        messages=messages,
        source=source,
        url=url,
        pdf_path=pdf_path,
        question=question,
        vectorstore=None,
    )

    try:
        #  Exécute l’agent
        result = agent.invoke(state)
        new_messages = result.get("messages", [])

        # Sauvegarde conversation
        save_conversation_to_file(new_messages)

        #  Formater la réponse
        response_messages = []
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                response_messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                response_messages.append({"type": "ai", "content": msg.content})

        return jsonify({"history": response_messages})

    except Exception as e:
        print(f"Erreur LangGraph: {e}")
        return jsonify(
            {"error": f"Erreur lors de la génération de la réponse: {e}"}
        ), 500
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"Fichier temporaire supprimé: {pdf_path}")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
