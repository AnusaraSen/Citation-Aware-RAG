import os
from pathlib import Path

class Config:
    # --- 1. Project Paths ---
    SRC_DIR = Path(__file__).parent
    PROJECT_ROOT = SRC_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

    # --- 2. LLM & Brain Settings (Ollama) ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Override the local host for dockers or remote setups
    MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2:3b")
    TEMPERATURE = 0.2
                                                                                                                                        
    # --- 3. Retrieval Settings ---
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "rag_citation_collection"
    
    # RAG Tuning Parameters
    RETRIEVAL_K = 30       # Broad search (High Recall)
    RERANK_K = 15           # Strict reranking (High Precision)
    USE_HYDE = False       
    
    # --- 4. Ingestion Settings ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # --- 5. UI Settings ---
    PAGE_TITLE = "Advanced Citation-Aware RAG"
    PAGE_ICON = ""

# Ensure crucial directories exist on import
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)