
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ragas import evaluate
from ragas.metrics import answer_relevancy  # Only import this one
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from src.retrieval.engine import RAGEngine
from tests.golden_dataset import GOLDEN_DATASET

# 1. Setup Local LLM
judge_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    timeout=60
)
judge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

# 2. Initialize your RAG Engine
rag_engine = RAGEngine()

# 3. Use FULL dataset 
questions = [item["question"] for item in GOLDEN_DATASET]
ground_truths = [item["ground_truth"] for item in GOLDEN_DATASET]
contexts = []
answers = []

print("🚀 Starting RAG Evaluation...")
print(f"📝 Evaluating {len(questions)} questions...\n")

for i, query in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] Processing: {query[:60]}...")
    try:
        response = rag_engine.query(query)
        answers.append(response["answer"])
        retrieved_ctx = [doc.content for doc in response["sources"]]
        contexts.append(retrieved_ctx)
        print(f"  ✓ Retrieved {len(retrieved_ctx)} context chunks")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        answers.append("")
        contexts.append([])

# 4. Construct Dataset
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

print("\n🔍 Running RAGAS Evaluation (answer_relevancy only)...\n")

# 5. Evaluate ONLY answer_relevancy (most reliable with small models)
try:
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy],  # Only this metric works reliably
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False
    )

    # 6. Save Results
    df = results.to_pandas() #type: ignore
    output_path = "tests/rag_evaluation_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n📊 Evaluation Complete! Results saved to {output_path}")
    print("\n" + "="*60)
    print(results)
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")