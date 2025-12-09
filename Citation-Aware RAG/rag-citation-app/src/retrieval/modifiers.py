from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.types import Document

class AdvancedRAGModifiers:
    """
    Implements 'Advanced RAG' techniques from the technical report:
    1. HyDE (Hypothetical Document Embeddings) [Pre-Retrieval]
    2. Cross-Encoder Reranking [Post-Retrieval]
    """

    def __init__(self, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker.
        We use a lightweight MS-MARCO model optimized for CPU/Edge Inference.
        It fits easily on your RTX 4050 alongside Llama 3.
        """
        print(f"⚖️ Loading Cross-Encoder: {rerank_model}...")
        self.reranker = CrossEncoder(rerank_model)

    def generate_hyde_doc(self, query: str, llm: BaseChatModel) -> str:
        """
        Implements HyDE: Uses the LLM to hallucinate a theoretical answer.
        This 'fake' answer often has better vector similarity to the real docs 
        than the raw user question does.
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
            
        # Prepare pairs for the model: [[Query, Doc1], [Query, Doc2], ...]
        pairs = [[query, d.content] for d in docs]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores to documents for debugging
        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])
            
        # Sort by score descending
        # We zip docs and scores, sort, and unzip
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Select Top K
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        
        print(f"🎯 [Rerank] Filtered {len(docs)} candidates down to {len(reranked_docs)}.")
        return reranked_docs