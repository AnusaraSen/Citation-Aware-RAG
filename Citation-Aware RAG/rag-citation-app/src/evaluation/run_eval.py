# src/evaluation/run_eval.py
import sys
from pathlib import Path
import time
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ragas import evaluate
from ragas.metrics import answer_relevancy
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from src.retrieval.engine import RAGEngine
from tests.golden_dataset import GOLDEN_DATASET

# ==================== CUSTOM METRIC FUNCTIONS ====================

def extract_citations_from_ground_truth(ground_truth: str) -> List[int]:
    """
    Extract page numbers from ground truth citations.
    Example: "[cite: 11, 13]" -> [11, 13]
    """
    pattern = r'\[cite:\s*([\d,\s]+)\]'
    matches = re.findall(pattern, ground_truth)
    pages = []
    for match in matches:
        pages.extend([int(p.strip()) for p in match.split(',') if p.strip().isdigit()])
    return sorted(set(pages))  # Unique pages


def compute_retrieval_hit_rate(retrieved_docs: List[Any], expected_pages: List[int]) -> float:
    """
    Hit Rate: Did we retrieve at least ONE of the expected pages?
    Returns 1.0 if yes, 0.0 if no
    """
    if not expected_pages:
        return 1.0  # If no expected pages, consider it a hit
    
    retrieved_pages = set()
    for doc in retrieved_docs:
        # Use direct attribute access for custom Document class
        if hasattr(doc, 'page'):
            retrieved_pages.add(doc.page)
    
    # Check if ANY expected page was retrieved
    hit = 1.0 if any(page in retrieved_pages for page in expected_pages) else 0.0
    
    # Debug output
    if not hit and expected_pages:
        print(f"      [Debug] Expected pages: {expected_pages}, Retrieved: {sorted(retrieved_pages)}")
    
    return hit


def compute_mrr(retrieved_docs: List[Any], expected_pages: List[int]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant document
    If first relevant doc is at position 3, MRR = 1/3
    """
    if not expected_pages:
        return 1.0
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        if hasattr(doc, 'page'):
            if doc.page in expected_pages:
                return 1.0 / rank
    
    return 0.0  # No relevant document found


def check_citation_format_compliance(answer: str) -> Dict[str, Any]:
    """
    Check if answer follows citation format:
    - Pattern 1: [cite: X] or [cite: X, Y]
    - Pattern 2: [Source: filename.pdf, Page: X]
    - Pattern 3: Sources footer with numbered list
    Returns dict with compliance score and details
    """
    # Citation patterns
    pattern1 = r'\[cite:\s*\d+(?:\s*,\s*\d+)*\]'  # [cite: 11] or [cite: 11, 13]
    pattern2 = r'\[Source:\s*[^,]+,\s*Page:\s*\d+\]'   # [Source: filename.pdf, Page: 5]
    pattern3 = r'\d+\.\s+[^\n]+\(Page\s+\d+\)'  # Footer format: 1. filename.pdf (Page 5)
    
    citations1 = re.findall(pattern1, answer, re.IGNORECASE)
    citations2 = re.findall(pattern2, answer, re.IGNORECASE)
    citations3 = re.findall(pattern3, answer, re.IGNORECASE)
    
    total_citations = len(citations1) + len(citations2) + len(citations3)
    
    # Check if answer has ANY citations or sources footer
    has_citations = total_citations > 0 or "**📚 Sources:**" in answer
    
    return {
        "compliant": has_citations,
        "citation_count": total_citations,
        "pattern1_count": len(citations1),
        "pattern2_count": len(citations2),
        "pattern3_count": len(citations3),
        "has_footer": "**📚 Sources:**" in answer,
        "score": 1.0 if has_citations else 0.0
    }


def check_negative_constraint(answer: str, query: str) -> Dict[str, Any]:
    """
    Negative Constraint Adherence: 
    - Answer should not contain "I don't know" when sources are available
    - Answer should not make claims without citations
    """
    negative_phrases = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "not sure",
        "unclear",
        "no information",
        "don't have enough information"
    ]
    
    answer_lower = answer.lower()
    
    # Check for negative phrases
    has_negative = any(phrase in answer_lower for phrase in negative_phrases)
    
    # Check if answer is too short (likely evasive)
    is_too_short = len(answer.split()) < 10
    
    # Compliance means: NOT negative AND NOT too short
    compliant = not has_negative and not is_too_short
    
    return {
        "compliant": compliant,
        "has_negative_phrase": has_negative,
        "is_too_short": is_too_short,
        "score": 1.0 if compliant else 0.0
    }


def measure_latency(rag_engine: RAGEngine, query: str) -> Tuple[float, Dict[str, Any]]:
    """
    Measure time to first token (approximated as total query time)
    Returns: (latency_seconds, response_dict)
    """
    start_time = time.time()
    response = rag_engine.query(query)
    end_time = time.time()
    
    latency = end_time - start_time
    
    return latency, response


# ==================== MAIN EVALUATION ====================

def run_custom_evaluation():
    print("="*70)
    print("🎯 Custom RAG Evaluation Suite")
    print("="*70)
    print("\nMetrics:")
    print("  1. Retrieval Hit Rate (Did we retrieve the right pages?)")
    print("  2. Mean Reciprocal Rank (How early are relevant docs?)")
    print("  3. Citation Format Compliance (Are citations properly formatted?)")
    print("  4. Negative Constraint Adherence (No evasive answers?)")
    print("  5. Latency (Time-to-response)")
    print("  6. Answer Relevancy (RAGAS semantic similarity)")
    print("\n" + "="*70 + "\n")
    
    # Initialize RAG Engine
    rag_engine = RAGEngine()
    
    # Setup RAGAS components
    judge_llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        timeout=120
    )
    judge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Results storage
    results = []
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    # Process each question
    for i, item in enumerate(GOLDEN_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        evolution_type = item.get("evolution_type", "unknown")
        
        print(f"\n[{i}/{len(GOLDEN_DATASET)}] {question[:60]}...")
        
        # Extract expected pages from ground truth
        expected_pages = extract_citations_from_ground_truth(ground_truth)
        print(f"  📄 Expected pages: {expected_pages}")
        
        try:
            # Measure latency and get response
            latency, response = measure_latency(rag_engine, question)
            
            answer = response["answer"]
            retrieved_docs = response["sources"]
            
            print(f"  ⏱️  Latency: {latency:.2f}s")
            print(f"  📚 Retrieved: {len(retrieved_docs)} documents")
            
            # Metric 1: Retrieval Hit Rate
            hit_rate = compute_retrieval_hit_rate(retrieved_docs, expected_pages)
            
            # Metric 2: MRR
            mrr = compute_mrr(retrieved_docs, expected_pages)
            
            # Metric 3: Citation Format Compliance
            citation_check = check_citation_format_compliance(answer)
            
            # Metric 4: Negative Constraint
            constraint_check = check_negative_constraint(answer, question)
            
            # Store for RAGAS
            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append([doc.content for doc in retrieved_docs])
            
            # Store results
            result = {
                "question": question,
                "evolution_type": evolution_type,
                "expected_pages": str(expected_pages),
                "retrieval_hit_rate": hit_rate,
                "mrr": mrr,
                "citation_compliance": citation_check["score"],
                "citation_count": citation_check["citation_count"],
                "has_footer": citation_check["has_footer"],
                "negative_constraint_compliance": constraint_check["score"],
                "latency_seconds": latency,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            }
            results.append(result)
            
            # Print metrics
            print(f"  ✅ Hit Rate: {hit_rate:.2f}")
            print(f"  🎯 MRR: {mrr:.3f}")
            print(f"  📌 Citations: {citation_check['citation_count']} + Footer: {citation_check['has_footer']}")
            print(f"  ⚠️  Constraint: {'✓' if constraint_check['compliant'] else '✗'}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "evolution_type": evolution_type,
                "expected_pages": str(expected_pages),
                "retrieval_hit_rate": 0.0,
                "mrr": 0.0,
                "citation_compliance": 0.0,
                "citation_count": 0,
                "has_footer": False,
                "negative_constraint_compliance": 0.0,
                "latency_seconds": 0.0,
                "answer": f"ERROR: {str(e)}"
            })
            ragas_data["question"].append(question)
            ragas_data["answer"].append("")
            ragas_data["contexts"].append([])
    
    # Run RAGAS Answer Relevancy
    print("\n" + "="*70)
    print("🔍 Computing RAGAS Answer Relevancy...")
    print("="*70 + "\n")
    
    try:
        ragas_dataset = Dataset.from_dict(ragas_data)
        ragas_results = evaluate(
            dataset=ragas_dataset,
            metrics=[answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False
        )
        
        ragas_df = ragas_results.to_pandas()  # type: ignore
        
        # Add RAGAS scores to results
        for idx, row in enumerate(results):
            if idx < len(ragas_df):
                row["answer_relevancy"] = ragas_df.iloc[idx]["answer_relevancy"]
            else:
                row["answer_relevancy"] = 0.0
                
    except Exception as e:
        print(f"⚠️  RAGAS evaluation failed: {e}")
        for row in results:
            row["answer_relevancy"] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Compute aggregate metrics
    print("\n" + "="*70)
    print("📊 AGGREGATE RESULTS")
    print("="*70)
    
    avg_hit_rate = df["retrieval_hit_rate"].mean()
    avg_mrr = df["mrr"].mean()
    avg_citation_compliance = df["citation_compliance"].mean()
    avg_constraint_compliance = df["negative_constraint_compliance"].mean()
    avg_latency = df["latency_seconds"].mean()
    avg_relevancy = df["answer_relevancy"].mean()
    
    print(f"\n🎯 Average Retrieval Hit Rate:           {avg_hit_rate:.2%}")
    print(f"📍 Average MRR:                          {avg_mrr:.3f}")
    print(f"📌 Average Citation Compliance:          {avg_citation_compliance:.2%}")
    print(f"⚠️  Average Negative Constraint Adherence: {avg_constraint_compliance:.2%}")
    print(f"⏱️  Average Latency:                      {avg_latency:.2f}s")
    print(f"🎓 Average Answer Relevancy (RAGAS):     {avg_relevancy:.3f}")
    
    # Save to CSV
    output_path = "tests/rag_custom_evaluation_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print per-type breakdown
    print("\n" + "="*70)
    print("📈 BREAKDOWN BY QUESTION TYPE")
    print("="*70)
    
    for evo_type in df["evolution_type"].unique():
        subset = df[df["evolution_type"] == evo_type]
        print(f"\n{evo_type.upper()}:")
        print(f"  Hit Rate: {subset['retrieval_hit_rate'].mean():.2%}")
        print(f"  MRR: {subset['mrr'].mean():.3f}")
        print(f"  Citations: {subset['citation_compliance'].mean():.2%}")
        print(f"  Relevancy: {subset['answer_relevancy'].mean():.3f}")
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    results_df = run_custom_evaluation()