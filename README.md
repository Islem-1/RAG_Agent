# ia_agents
 
# 📘 RAG Chat with PDF/Web

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that can answer questions by retrieving information from uploaded PDFs or from web sources.  
It integrates **LangChain, OpenAI API, FAISS/Chroma** for semantic search, combined with a **Flask backend** and a **TailwindCSS frontend**.  
It also uses **LangGraph** for managing conversation flows, chaining, and generating graphs that represent the workflow (start, end, and tools used).


---

## 🚀 Features
- 📂 Upload PDF files from your computer.  
- 🔎 Ask questions about the uploaded document.  
- 🌍 Ask general knowledge questions (e.g., about Albert Einstein).  
- ⚡ Fast semantic search powered by vector embeddings.  
- 🎨 Clean UI with **TailwindCSS**, **Google Fonts (Inter)**, and light **custom CSS animations**.


---

## 🛠️ Tech Stack
- **Frontend**:  
  - TailwindCSS (main styling with utility classes).  
  - Google Fonts (Inter for typography).  
  - Custom CSS (for input animations).  

- **Backend**:  
  - Flask (Python web framework).  
  - LangChain (RAG logic: loaders, splitters, embeddings, vector store).
  - LangGraph for managing conversation flows and chaining and generating graphs that represent the workflow. 
  - FAISS/Chroma (vector database).  
  - PyPDF (PDF parsing).
  - Web Loader: WebBaseLoader (for retrieving and parsing online documents) 

---

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/Islem-1/RAG_Agent.git
cd RAG_Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
