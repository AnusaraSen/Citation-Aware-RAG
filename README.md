# 🧠 Citation‑Aware Retrieval‑Augmented Generation (RAG) System

> **Production‑oriented RAG architecture focused on factual grounding, citation enforcement, and retrieval precision.**
>
> Built to demonstrate real‑world AI engineering practices beyond tutorial‑level RAG.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![Ollama](https://img.shields.io/badge/LLM-Llama3.2--3B%20%7C%20Local-orange)
![Status](https://img.shields.io/badge/Status-Evaluated-success)

---

## 🎯 Why This Project Exists

Most RAG demos break down in production settings. They:

* Treat PDFs as plain text
* Assume cosine similarity equals relevance
* Allow LLMs to answer without verifiable sources

This project was built to **solve those failures explicitly** and to serve as a **portfolio-grade Applied ML / AI Engineering system**, emphasizing evaluation rigor, retrieval correctness, and controllable generation.

### Core Problems Addressed

1. **Document Structure Loss**
   PDFs are visual artifacts. Naïve loaders scramble columns, headers, and footnotes.
   → Solved using **layout‑aware PDF parsing** with coordinate‑based filtering.

2. **High Recall, Low Precision Retrieval**
   Vector search retrieves *similar* chunks, not necessarily the *most relevant* ones.
   → Solved using a **two‑stage retrieval pipeline** with cross‑encoder reranking.

3. **LLM Hallucinations**
   Fluent answers without evidence are worse than no answers.
   → Solved using **strict citation constraints** enforced at generation time.

---

## ✨ Key Features

* 📄 **Layout-Aware PDF Ingestion** using block-level geometry
* 🔍 **Hybrid Retrieval Pipeline** for high recall
* 🧠 **Cross-Encoder Reranking** for ranking precision
* 📌 **Strict Citation Enforcement** (`[Source: File, Page X]`)
* 🚫 **Negative Constraint Handling** (explicit refusal on out-of-scope queries)
* 🧪 **Fully Automated Evaluation Suite** with reproducible metrics

---

## 🛠️ Tech Stack

| Layer         | Technology                                  |
| ------------- | ------------------------------------------- |
| Orchestration | LangChain + custom Python modules           |
| LLM Inference | **Llama 3.2 (3B) – Local via Ollama**       |
| Embeddings    | `all-MiniLM-L6-v2`                          |
| Reranking     | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)    |
| Vector Store  | ChromaDB (HNSW, persistent)                 |
| Ingestion     | PyMuPDF (Fitz) – layout-aware block parsing |
| Frontend      | Streamlit                                   |
| DevOps        | Docker and Docker compose                   |
| Evaluation    | RAGAS + custom deterministic tests          |



---

## 🏗️ System Architecture

```
PDFs
  ↓
Layout‑Aware Parsing
  ↓
Semantic Chunking + Metadata Injection
  ↓
Vector Index (ChromaDB)
  ↓
Hybrid Retrieval (Top‑K)
  ↓
Cross‑Encoder Reranking
  ↓
LLM Generation (Citation‑Constrained)
```

---

## 🔍 Pipeline Breakdown

### 1️⃣ Ingestion Layer (`src/ingestion`)

* **Layout-Aware Parsing:** Extracts text blocks with coordinates instead of linear text
* **Noise Removal:** Headers and footers filtered via Y-axis thresholds
* **Chunking:** **Semantic chunking** to preserve meaning boundaries, with controlled overlap to maintain cross-section context
* **Metadata Injection:** Every chunk includes:

  ```python
  metadata = {
      "total_pages": int,
      "block_count": int,
      "table_count": int,
      "extraction_method": "layout_aware_blocks",
      "is_toc": False
  }
  ```

This metadata enables **auditable retrieval**, **page-level citation**, and future structured filtering.

---

### 2️⃣ Retrieval Engine (`src/retrieval`)

**Stage 1 – Recall**

* Hybrid search retrieves top‑N candidates
* Optimized for *coverage*, not precision

**Stage 2 – Precision**

* Cross‑encoder reranks candidates using `(query, chunk)` pairs
* Eliminates false positives common in dense retrieval

**Stage 3 – Generation**

* Only top‑K reranked chunks are passed to the LLM
* Prompt enforces citation formatting and negative constraints

---

### 3️⃣ Answer Generation

* LLM is instructed to:

  * Use *only* provided context
  * Cite every factual claim
  * Refuse to answer if evidence is missing

**Example Output:**

```
The system uses a cross‑encoder reranker to improve precision
[Source: architecture.pdf, Page 12]
```

---

## 🧪 Evaluation Results

Evaluation was performed using a **custom deterministic evaluation pipeline** on a controlled dataset.

### 📊 Aggregate Results

| Metric                        | Score      |
| ----------------------------- | ---------- |
| Retrieval Hit Rate            | **100%**   |
| Mean Reciprocal Rank (MRR)    | **1.000**  |
| Citation Compliance           | **100%**   |
| Negative Constraint Adherence | **100%**   |
| Avg Latency (Local 3B Model)  | **15.05s** |
| Answer Relevancy (RAGAS)      | **0.934**  |

Results are exported to `tests/rag_custom_evaluation_report.csv` for auditability.

### 📈 Breakdown by Question Type

| Type      | Hit Rate | MRR   | Citations | Relevancy |
| --------- | -------- | ----- | --------- | --------- |
| Reasoning | 100%     | 1.000 | 100%      | 0.964     |
| Simple    | 100%     | 1.000 | 100%      | 0.918     |
| Technical | 100%     | 1.000 | 100%      | 0.907     |

---

## ⚡ Getting Started

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/AnusaraSen/Citation-Aware-Rag.git
cd Citation-Aware RAG\rag-citation-app
docker-compose up --build
```

Then open: **[http://localhost:8501](http://localhost:8501)**

---

### Option 2: Local Development

**Prerequisites:** Python 3.11+, Poetry, Ollama

```bash
poetry install

# Ingest documents
poetry run python -m src.ingestion.pipeline data/sample.pdf

# Run UI
poetry run streamlit run src/ui/app.py
```

---

## 🔐 Privacy-First Design Choice

This system intentionally uses a **fully local LLM (Llama 3.2 – 3B via Ollama)**.

### Why Local Inference?

* 📄 **Sensitive Inputs:** Designed for company policies, internal documentation, and legal content
* 🔒 **Data Residency:** No documents, embeddings, or queries leave the local machine
* 🛡️ **Compliance-Friendly:** Suitable for environments with strict privacy or regulatory constraints

This makes the system applicable to **enterprise, legal, and internal-knowledge settings** where cloud-based LLM APIs are not acceptable.

> **Deployment Note:** This architecture is suitable for **on‑prem or air‑gapped environments** in regulated industries such as **finance, healthcare, and legal services**.

---

## ⚠️ Known Limitations & Trade-offs

* **Latency:** Local inference prioritizes privacy and data control over response time
* **Single-Node Execution:** No distributed ingestion or retrieval
* **CPU/GPU Constraints:** Performance bound by local hardware

### Engineering Rationale

These trade-offs were made deliberately to emphasize **privacy, factual grounding, and evaluation rigor** over raw throughput.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 👤 Author

**Anusara Senanayake**
Applied ML / AI Engineering Portfolio Project
Focus Areas:

* Retrieval-Augmented Generation (RAG)
* LLM Reliability & Evaluation
* Citation-Aware AI Systems
* Production-Oriented ML Design

🔗 Repository: [https://github.com/AnusaraSen/Citation-Aware-RAG](https://github.com/AnusaraSen/Citation-Aware-RAG)

---

## 📌 Recruiter Note

This project demonstrates **end-to-end ownership** of a modern RAG system — from document ingestion and retrieval modeling to evaluation, failure handling, and architectural trade-offs. It is intentionally designed to reflect **real-world AI engineering constraints**, not demo-level experimentation.
