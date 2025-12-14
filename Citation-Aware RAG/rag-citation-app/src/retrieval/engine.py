from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import Config
from src.storage.vector_store import VectorStore
from src.retrieval.modifiers import AdvancedRAGModifiers 
from src.core.types import Document

class RAGEngine:
    """
    Professional RAG Engine with HyDE and Reranking.
    """
    
    def __init__(self, model_name: Optional [str] = None):
        # Use Config values
        if model_name is None:
            model_name = Config.MODEL_NAME
            
        self.store = VectorStore()
        
        print(f"Initializing Llama 3 Brain ({model_name})...")
        self.llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL, 
            model=model_name,
            temperature=Config.TEMPERATURE,
            keep_alive="1h"
        )
        
        # Initialize Advanced Modifiers
        self.modifiers = AdvancedRAGModifiers()
        
        # Enhanced prompt with clearer structural boundaries
        self.prompt = ChatPromptTemplate.from_template("""You are a technical research assistant. Your job is to answer questions using ONLY the information contained within the <context_data> section below.

=== INSTRUCTIONS ===
1. Read ALL documents inside the <context_data> tags carefully
2. Each document has a 'source' and 'page' attribute - use these for citations
3. Answer the question using ONLY information from these documents
4. ALWAYS cite your sources in this format: [Source: filename.pdf, Page: X]
5. If multiple documents contain relevant information, synthesize them and cite all sources
6. If the <context_data> does not contain enough information to answer, respond with: "I don't have enough information in the provided documents to answer this question."

=== CONTEXT DATA ===
<context_data>
{context}
</context_data>

=== USER QUESTION ===
{question}

=== YOUR ANSWER ===
Provide a clear, well-structured answer with citations:""")
            
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_context(self, docs: List[Document]) -> str:
        """
        Format documents with XML tags for structural clarity.
        
        Uses explicit XML structure to help Llama 3 distinguish between:
        - System instructions (outside tags)
        - Retrieved knowledge (inside tags)
        - Document boundaries (individual <doc> elements)
        """
        formatted = []
        for idx, doc in enumerate(docs, 1):
            # Include rerank score for transparency
            score_info = ""
            if 'rerank_score' in doc.metadata:
                score_info = f' relevance_score="{doc.metadata["rerank_score"]:.3f}"'
            
            # Add document index for easier reference
            entry = (
                f'<doc id="{idx}" source="{doc.source}" page="{doc.page}"{score_info}>\n'
                f"{doc.content.strip()}\n"
                f"</doc>"
            )
            formatted.append(entry)
        
        # Return with clear document separation
        return "\n\n".join(formatted)

    def query(self, user_question: str, use_hyde: bool = False) -> Dict[str, Any]:
        """
        Executes Advanced RAG Pipeline:
        Query -> (HyDE) -> Hybrid Search (Top 30) -> Rerank (Top 15) -> Generation
        """
        print(f"\n🤔 [RAG] Processing: {user_question}")
        
        search_query = user_question
        
        # --- Step 1: HyDE (Optional) ---
        if use_hyde:
            fake_doc = self.modifiers.generate_hyde_doc(user_question, self.llm)
            search_query = fake_doc
            print(f"   [HyDE] Using hypothetical document for search.")

        # --- Step 2: High-Recall Retrieval ---
        print("   [Retrieval] Fetching Top 30 candidates via Hybrid Search...")
        results = self.store.hybrid_search(search_query, k=30)
        
        if not results:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": []
            }

        # --- Step 3: High-Precision Reranking ---
        print("   [Reranking] Applying Cross-Encoder reranking...")
        top_docs = self.modifiers.rerank_documents(user_question, results, top_k=Config.RERANK_K)
        
        # Debug: Log top reranked document
        if top_docs:
            top_score = top_docs[0].metadata.get('rerank_score', 'N/A')
            print(f"   [Debug] Top doc: {top_docs[0].source} (Page {top_docs[0].page}, Score: {top_score})")
            print(f"   [Debug] Content preview: {top_docs[0].content[:150]}...")
        
        # --- Step 4: Context Formatting ---
        context_str = self._format_context(top_docs)
        
        # Debug: Check context length
        print(f"   [Debug] Context length: {len(context_str)} chars, {len(top_docs)} documents")
        
        # --- Step 5: Generation ---
        print("   [Generation] Invoking LLM with structured prompt...")
        try:
            answer = self.chain.invoke({
                "context": context_str,
                "question": user_question
            })
            
            # Post-process: Check if LLM returned a refusal
            if "don't have enough information" in answer.lower() or "i don't know" in answer.lower():
                print("   [Warning] LLM returned 'I don't know' despite having context")
            
            # Extract unique sources used
            unique_sources = {}
            for doc in top_docs:
                key = (doc.source, doc.page)
                if key not in unique_sources:
                    unique_sources[key] = doc
            
            # Append citation footer
            if unique_sources and "don't have enough information" not in answer.lower():
                citation_footer = "\n\n---\n**📚 Sources:**\n"
                for idx, (source, page) in enumerate(sorted(unique_sources.keys()), 1):
                    citation_footer += f"{idx}. {source} (Page {page})\n"
                answer += citation_footer
            
            return {
                "answer": answer,
                "sources": top_docs
            }
            
        except Exception as e:
            return {
                "answer": f"Error during generation: {str(e)}",
                "sources": []
            }

