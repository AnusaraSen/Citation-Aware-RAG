from typing import List, Optional
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from src.core.types import Document
from src.core.config import Config

class DocumentSplitter:
    """
    Splits strict Document objects into smaller chunks using Semantic Analysis.
    
    Strategy: "Semantic Chunking"

    
    """

    def __init__(self, model_name: Optional[str] = None):
        # Use Config value if not specified
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL
            
        # We use a small, fast local model for the splitting process.
        # This runs on your CPU/GPU.
        print(f"Loading embedding model '{model_name}' for semantic splitting...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize the Semantic Splitter
        # 'percentile' means we split at the distinct "dips" in similarity.
        self.splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile" 
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs: List[Document] = []

        print(f"Starting semantic split on {len(documents)} source pages...")

        for doc in documents:
            # 1. Extract plain text from our Pydantic model
            original_text = doc.content
            
            # 2. Feed text to the Semantic Splitter
            # This is the heavy computation step 
            chunks = self.splitter.split_text(original_text)

            # 3. Re-wrap chunks into strict Document objects
            for i, chunk_text in enumerate(chunks):
                # creates derived chunk IDs by appending a chunk index to the parent ID (loader.py)
                new_id = f"{doc.id}_chunk_{i}"
                
                new_doc = Document(
                    id=new_id,
                    content=chunk_text,
                    source=doc.source,
                    page=doc.page, # Crucial: The chunk inherits the page number 
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "split_strategy": "semantic"
                    }
                )
                chunked_docs.append(new_doc)
        
        print(f"Split {len(documents)} pages into {len(chunked_docs)} semantic chunks.")
        return chunked_docs