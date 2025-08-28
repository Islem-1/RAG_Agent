from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# --- Load .env ---
load_dotenv()


# --- Fonction utilitaire commune ---
def process_documents(documents):
    """Découpe en chunks et construit un vectorstore"""
    if not documents:
        print("⚠️ Aucun document fourni")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"{len(documents)} documents chargés, {len(chunks)} chunks créés")
    print("✅ Vector store créé avec succès")
    return vectorstore


# --- PDF Loader ---
def load_pdf(path):
    """Charge un PDF et crée un vectorstore"""
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
    except Exception as e:
        print(f"❌ Erreur lecture PDF {path}: {e}")
        return None

    return process_documents(documents)


# --- Web Loader ---
def load_web(url):
    """Charge une page web et crée un vectorstore"""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
    except Exception as e:
        print(f"❌ Erreur lecture page web {url}: {e}")
        return None

    return process_documents(documents)


# --- LangChain QA avec Gemini ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def answer_semantic_question(vectorstore, question, k=3):
    """Réponse sémantique via FAISS"""
    if not vectorstore:
        return "⚠️ Aucun vectorstore disponible."

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return qa_chain.invoke(question)
