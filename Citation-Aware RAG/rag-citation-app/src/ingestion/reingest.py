import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import IngestionPipeline
from src.storage.vector_store import VectorStore

if __name__ == "__main__":
    print("🧹 Clearing old database...")
    db_path = Path("data/chroma_db")
    if db_path.exists():
        shutil.rmtree(db_path)
        print("   Deleted old database.")
    
    print("\n🚀 Starting fresh ingestion...")
    pdf_path = "data/RAG Project Guide Generation.pdf"
    
    try:
        pipeline = IngestionPipeline()
        chunks = pipeline.run(pdf_path)
        print(f"\n✅ Ingested {len(chunks)} chunks!")
        
        # Verify immediately
        print("\n🔍 Verifying database...")
        store = VectorStore()
        count = store.get_document_count()
        print(f"📊 Vector store contains: {count} documents")
        
        if count > 0:
            sources = store.list_sources()
            print(f"✅ Sources: {sources}")
        else:
            print("❌ Database is still empty after ingestion!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()