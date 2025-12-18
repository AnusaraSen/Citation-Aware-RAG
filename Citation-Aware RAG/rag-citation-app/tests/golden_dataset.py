# A list of dictionary pairs: Question + The Ideal Answer
# tests/golden_dataset.py

GOLDEN_DATASET = [
    {
        "question": "What is the primary differentiator for a high-value data science candidate in 2025?",
        "ground_truth": "The primary differentiator is the transition from a 'modeling-first' mindset to a 'product-first' engineering mindset, specifically the ability to operationalize models rather than just training them. [cite: 11, 13]",
        "evolution_type": "reasoning"
    },
    {
        "question": "Why do standard LLMs suffer from hallucinations?",
        "ground_truth": "Hallucinations occur because LLMs are probabilistic engines trained to predict the next token based on statistical correlations, not to query a database of facts. [cite_start]They lack external grounding. [cite: 23, 24]",
        "evolution_type": "simple"
    },
    {
        "question": "What is the recommended method for splitting text in a professional RAG pipeline?",
        "ground_truth": "The industry standard is 'Recursive Character Text Splitting'. [cite_start]This method preserves semantic structure by splitting on a hierarchy of separators (paragraphs, lines, words) rather than fixed character counts. [cite: 80, 81]",
        "evolution_type": "simple"
    },
    {
        "question": "Which vector database is recommended for this portfolio project and why?",
        "ground_truth": "ChromaDB is recommended because it is open-source, AI-native, easy to set up (pip installable), handles metadata filtering natively, and offers automatic persistence. [cite: 103, 104, 108]",
        "evolution_type": "reasoning"
    },
    {
        "question": "What is the main difference between LangChain and LlamaIndex?",
        "ground_truth": "LangChain is a versatile, general-purpose framework for complex workflows and agents. [cite_start]LlamaIndex is specialized specifically for data indexing and retrieval, offering advanced structures like the CitationQueryEngine. [cite: 115, 119, 121]",
        "evolution_type": "reasoning"
    },
    {
        "question": "How should citations be handled technically in the ingestion phase?",
        "ground_truth": "During ingestion, every text chunk must be tagged with metadata containing the 'source' (filename) and 'page' number. [cite_start]This metadata travels with the vector embedding throughout the pipeline. [cite: 63, 73, 132]",
        "evolution_type": "technical"
    },
    {
        "question": "What is the purpose of the 'session_state' in the Streamlit frontend?",
        "ground_truth": "Session state is used to persist variables across re-runs of the script. [cite_start]It is critical for storing the 'messages' list so the chatbot does not forget the conversation history after every user interaction. [cite: 150, 151, 152]",
        "evolution_type": "technical"
    },
    {
        "question": "Why is PyMuPDF (Fitz) recommended over PyPDF2?",
        "ground_truth": "PyMuPDF is recommended because it allows for 'block' extraction, which respects the visual structure of multi-column documents, preventing 'text soup.' [cite_start]It also enables the extraction of bounding boxes for highlighting. [cite: 56, 57, 58, 59]",
        "evolution_type": "reasoning"
    }
]