from pathlib import Path
from typing import List
from src.core.types import Document
from src.ingestion.loader import PDFLoader
from src.ingestion.splitter import DocumentSplitter
from src.storage.vector_store import VectorStore

class IngestionPipeline:
    """
    The Orchestrator for the RAG ETL process.
    
    Responsibilities:
    1. EXTRACT: Load raw text from PDFs.
    2. TRANSFORM: Split text semantically and preserve metadata.
    3. LOAD: Push vectors into the ChromaDB.
    
    This adheres to the 'Modular RAG' architecture described in the project guide.
    """
    
    def __init__(self):
        # We initialize the helper classes once
        self.splitter = DocumentSplitter()
        self.vector_store = VectorStore()
        
    def run(self, file_path: str) -> List[Document]:
        """
        Executes the full ingestion lifecycle for a single file.
        Returns the processed chunks for inspection.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"\n🚀 [Pipeline] Starting ingestion for: {path.name}")
        
        # --- Step 1: Extract ---
        loader = PDFLoader(path)
        raw_docs = loader.load()
        print(f"   [Extract] Loaded {len(raw_docs)} pages.")
        
        if not raw_docs:
            print("   ⚠️ Warning: No text extracted. Skipping.")
            return []
        
        # Optional: Separate TOC from regular pages for different processing
        toc_docs = [doc for doc in raw_docs if doc.metadata.get("is_toc", False)]
        page_docs = [doc for doc in raw_docs if not doc.metadata.get("is_toc", False)]
        
        print(f"   [Extract] Found {len(toc_docs)} TOC, {len(page_docs)} pages.")
        print(f"   [Extract] Total blocks: {sum(doc.metadata.get('block_count', 0) for doc in page_docs)}")
        print(f"   [Extract] Total tables: {sum(doc.metadata.get('table_count', 0) for doc in page_docs)}")


        # --- Step 2: Transform ---
        # The splitter uses the 'all-MiniLM-L6-v2' model locally
        chunks = self.splitter.split_documents(raw_docs)
        print(f"   [Transform] Created {len(chunks)} semantic chunks.")
        
        # --- Step 3: Load ---
        # Persist to local vector database
        if chunks:
            self.vector_store.add_documents(chunks)
            print(f"   [Load] Successfully indexed {len(chunks)} vectors.")
        
        print(f"✅ [Pipeline] Finished processing {path.name}")
        return chunks

# --- Smoke Test ---
if __name__ == "__main__":
    import sys
    
    # This allows us to run the pipeline from the command line:
    # poetry run python -m src.ingestion.pipeline "data/MyFile.pdf"
    if len(sys.argv) > 1:
        pipeline = IngestionPipeline()
        pipeline.run(sys.argv[1])
    else:
        print("Usage: python -m src.ingestion.pipeline <path_to_pdf>")