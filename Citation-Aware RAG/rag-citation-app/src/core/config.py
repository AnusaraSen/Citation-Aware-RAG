import os
from pathlib import Path

class Config:
    # --- 1. Project Paths ---
    # Calculates the project root automatically relative to this file
    SRC_DIR = Path(__file__).parent
    PROJECT_ROOT = SRC_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

    # --- 2. LLM & Brain Settings (Ollama) ---
    # We use os.getenv to allow Docker to override 'localhost' with 'http://ollama:11434'
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME = os.getenv("LLM_MODEL", "llama3")
    TEMPERATURE = 0.1
    
    # --- 3. Retrieval Settings ---
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "rag_citation_collection"
    
    # RAG Tuning Parameters
    RETRIEVAL_K = 30       # Broad search (High Recall)
    RERANK_K = 5           # Strict reranking (High Precision)
    USE_HYDE = False       # Default state
    
    # --- 4. Ingestion Settings ---
    # Critical for PDF parsing structure
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # --- 5. UI Settings ---
    PAGE_TITLE = "Advanced Citation-Aware RAG"
    PAGE_ICON = "🧠"

# Ensure crucial directories exist on import
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)