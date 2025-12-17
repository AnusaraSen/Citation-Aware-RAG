from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.types import Document

class AdvancedRAGModifiers:
    """
    1. HyDE (Hypothetical Document Embeddings) [Pre-Retrieval]
    2. Cross-Encoder Reranking [Post-Retrieval]
    """

    def __init__(self, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker.
        """
        print(f"⚖️ Loading Cross-Encoder: {rerank_model}...")
        self.reranker = CrossEncoder(rerank_model)

    def generate_hyde_doc(self, query: str, llm: BaseChatModel) -> str: #Hypothetical Document Embedding
        """
        Implements HyDE: Uses the LLM to hallucinate a theoretical answer.
        
        """
        hyde_prompt = ChatPromptTemplate.from_template("""
        You are a generic AI assistant. 
        Write a brief, hypothetical passage that answers the following question.
        Do not try to be factual. Focus on the keywords, vocabulary, and sentence structure 
        that a relevant technical document would contain.
        
        QUESTION: {question}
        HYPOTHETICAL ANSWER:
        """)
        
        chain = hyde_prompt | llm | StrOutputParser()
        
        print(f"🔮 [HyDE] Generating hypothetical answer for: '{query}'")
        hypothetical_doc = chain.invoke({"question": query})
        return hypothetical_doc

    def rerank_documents(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """
        Implements Cross-Encoder Reranking.
        
        Bi-Encoders (Vector DB) are fast but less accurate.
        Cross-Encoders are slow but highly accurate.
        
        Strategy: Retrieve 25 (Fast), Rerank to 5 (Slow & Precise).
        This boosts 'NDCG@10' metrics significantly.
        """
        if not docs:
            return []
            
        # Creates a list of pairs where every single document is paired with the user's question.
        pairs = [[query, d.content] for d in docs]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores to documents for debugging
        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])
            
        # Sort by score descending
        # We zip docs and scores, sort, and unzip (Glues the document list and score list together.)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Select Top K
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        
        print(f"🎯 [Rerank] Filtered {len(docs)} candidates down to {len(reranked_docs)}.")
        return reranked_docs