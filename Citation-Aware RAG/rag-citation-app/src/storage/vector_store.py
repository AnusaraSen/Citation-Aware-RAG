import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Disable Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi  
from src.core.types import Document
from src.core.config import Config

class VectorStore:
    """
    Hybrid Vector Store (Dense + Sparse).
    
    Implements:
    1. HNSW Vector Search (via ChromaDB)
    2. BM25 Keyword Search (via rank_bm25)
    3. Reciprocal Rank Fusion (RRF) to merge results
    """
    
    def __init__(self, persist_dir: Optional [str] = None, reset: bool = False):
        """
        Initialize the Vector Store with optional database reset.
        
        Args:
            persist_dir: Path to the ChromaDB persistence directory
            reset: If True, wipes the existing collection before initialization.
                   Use with caution - this is irreversible!
        """
        if persist_dir is None:
            persist_dir = str(Config.CHROMA_PERSIST_DIR)
        
        self.persist_dir = str(Path(persist_dir).resolve())  # Use absolute path
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_fn = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Graceful database reset (if requested)
        if reset:
            self._reset_database()
        
        print(f"Initializing Hybrid Store at {self.persist_dir}...")
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn,
            collection_name=self.collection_name,
            client=self.chroma_client
        )
        
        # Initialize Sparse Index (BM25)
        self.bm25 = None
        self.doc_map: Dict[str, Document] = {} # Quick lookup for BM25 results
        self._build_bm25_index()

    def _reset_database(self):
        """
        Gracefully wipes the vector database using ChromaDB's native API.
        
        This approach:
        - Avoids file system locks (no PermissionError on Windows)
        - Uses ChromaDB's internal cleanup mechanisms
        - Leaves the client in a valid state for re-initialization
        
        Implementation:
        1. Attempts to delete the collection via ChromaDB API
        2. Handles case where collection doesn't exist (first run)
        3. Logs clear warnings about data loss
        """
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: DATABASE RESET IN PROGRESS")
        print(f"{'='*60}")
        print(f"Target Collection: {self.collection_name}")
        print(f"Persist Directory: {self.persist_dir}")
        print(f"This action is IRREVERSIBLE. All indexed documents will be lost.")
        print(f"{'='*60}\n")
        
        try:
            # Attempt to delete the collection using ChromaDB's native API
            # This is the professional way - avoids file locks entirely
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"✅ Collection '{self.collection_name}' successfully deleted.\n")
            
        except ValueError as e:
            # Collection doesn't exist - this is expected on first run
            if "does not exist" in str(e).lower():
                print(f"⚠️ Collection '{self.collection_name}' does not exist yet.")
                print(f"   This is normal on first initialization.\n")
            else:
                # Unexpected ValueError - re-raise
                print(f"❌ Unexpected ValueError during reset: {str(e)}")
                raise
                
        except Exception as e:
            # Catch any other unexpected errors
            print(f"❌ Unexpected error during database reset: {str(e)}")
            print(f"   The collection may still exist. Manual cleanup may be required.")
            raise RuntimeError(
                f"Failed to reset collection '{self.collection_name}'. "
                f"Error: {str(e)}"
            ) from e

    def _build_bm25_index(self):
        """
        Hydrates the BM25 index from the existing Chroma data.
        In a massive production system, this would be an external search engine (Elasticsearch).
        For local RAG, in-memory build is professional and fast enough.
        """
        print("⚡ Building BM25 Sparse Index...")
        data = self.db.get() # Fetch all docs from Chroma
        
        documents = []
        tokenized_corpus = []
        
        if not data['ids']:
            print("   (Index empty, skipping BM25 build)")
            return

        for i, doc_id in enumerate(data['ids']):
            content = data['documents'][i]
            metadata = data['metadatas'][i]
            
            # Reconstruct Document object
            doc = Document(
                id=doc_id,
                content=content,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page", 0),
                metadata=metadata
            )
            
            self.doc_map[doc_id] = doc # When BM25 points to a result, this map instantly grab the actual content.
            documents.append(doc)
            # Simple tokenization for BM25 (split by space)
            tokenized_corpus.append(content.lower().split())

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"   BM25 Index ready with {len(documents)} documents.")

    def add_documents(self, docs: List[Document]):
        if not docs: return

        texts = [d.content for d in docs]
        ids = [d.id for d in docs]
        
        # Create mutable copies of metadata and add required fields
        metadatas = []
        for doc in docs:
            meta = dict(doc.metadata)  # Create a copy
            meta["id"] = doc.id
            meta["source"] = doc.source
            meta["page"] = doc.page
            metadatas.append(meta)

        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"   Added {len(docs)} documents to ChromaDB.")
        
        # Explicitly persist the data
        self.db.persist()
        
        # Rebuild BM25 after adding new data to keep indices in sync
        self._build_bm25_index()

    def _rrf_merge(self, vector_results: List[Document], bm25_results: List[Document], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) Algorithm.
        Score = 1 / (k + rank)
        """
        scores = defaultdict(float)
        
        # 1. Process Vector Results
        for rank, doc in enumerate(vector_results):
            scores[doc.id] += 1 / (k + rank)
            self.doc_map[doc.id] = doc # Ensure we have the object
            
        # 2. Process BM25 Results
        for rank, doc in enumerate(bm25_results):
            scores[doc.id] += 1 / (k + rank)
            self.doc_map[doc.id] = doc

        # 3. Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return unique documents
        return [self.doc_map[doc_id] for doc_id in sorted_ids]

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        The Main Retrieval Method.
        Executes Vector + Keyword search and merges via RRF.
        """
        # 1. Dense Search (Vector)
        # We fetch k*2 candidates to give RRF enough data to re-rank
        vector_candidates = self.db.similarity_search(query, k=k*2)
        vector_docs = [
            Document(
                id=d.metadata.get("id", "unknown"),
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page", 0),
                metadata=d.metadata
            ) for d in vector_candidates
        ]
        
        # 2. Sparse Search (BM25)
        bm25_docs = []
        if self.bm25:
            tokenized_query = query.lower().split()
            # Get top k*2 keyword matches
            raw_bm25 = self.bm25.get_top_n(tokenized_query, list(self.doc_map.values()), n=k*2)
            bm25_docs = raw_bm25
            
        # 3. Fusion (RRF)
        fused_results = self._rrf_merge(vector_docs, bm25_docs)
        
        # Return top K after fusion
        return fused_results[:k]

    # Keep the old interface for compatibility, but redirect to hybrid
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        # Note: RRF doesn't produce a "similarity score" like cosine similarity.
        # We return a dummy score of 1.0 because the relative order is what matters.
        docs = self.hybrid_search(query, k)
        return [(d, 1.0) for d in docs]
   
    def clear(self):
        """
        Legacy clear method. Deprecated - use reset=True in __init__ instead.
        """
        print("⚠️ Warning: clear() is deprecated. Use VectorStore(reset=True) instead.")
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print("Database collection cleared.")
        except Exception as e:
            print(f"Failed to clear collection: {e}")
    

    def get_document_count(self) -> int:
        """Returns the number of documents in the vector store."""
        data = self.db.get()
        return len(data['ids']) if data['ids'] else 0
    
    def list_sources(self) -> List[str]:
        """Returns a list of unique source files in the database."""
        data = self.db.get()
        if not data['metadatas']:
            return []
        sources = set(meta.get('source', 'unknown') for meta in data['metadatas'])
        return sorted(sources)
    
    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all document chunks from a specific source file.
        
        """
        try:
            # Get all documents from the collection
            data = self.db.get()
            
            if not data['ids']:
                print(f"⚠️ No documents found in database")
                return 0
            
            # Find IDs of chunks from this source
            ids_to_delete = []
            for i, metadata in enumerate(data['metadatas']):
                if metadata.get('source') == source_filename:
                    ids_to_delete.append(data['ids'][i])
            
            if not ids_to_delete:
                print(f"⚠️ No chunks found for source: {source_filename}")
                return 0
            
            # Delete from ChromaDB
            print(f"🗑️ Deleting {len(ids_to_delete)} chunks from '{source_filename}'...")
            self.db._collection.delete(ids=ids_to_delete)
            
            # Rebuild BM25 index to reflect changes
            self._build_bm25_index()
            
            print(f"✅ Successfully deleted {len(ids_to_delete)} chunks")
            return len(ids_to_delete)
            
        except Exception as e:
            print(f"❌ Error deleting source '{source_filename}': {str(e)}")
            raise
            
