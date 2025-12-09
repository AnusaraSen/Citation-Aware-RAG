import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.storage.vector_store import VectorStore

# Page Configuration
st.set_page_config(
    page_title="RAG Architect | Citation-Aware System",
    page_icon="📘",
    layout="wide"
)

# Constants
DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

def initialize_state():
    """
    Centralized state initialization.
    Ensures all session variables exist before use.
    """
    if "rag_engine" not in st.session_state:
        with st.spinner("🧠 Initializing RAG Engine..."):
            st.session_state.rag_engine = RAGEngine()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if "use_hyde" not in st.session_state:
        st.session_state.use_hyde = False

def get_db_stats():
    """
    Retrieve current database statistics.
    Returns document count and list of sources.
    """
    try:
        store = VectorStore()
        doc_count = store.get_document_count()
        sources = store.list_sources()
        return doc_count, sources
    except Exception as e:
        st.error(f"Error retrieving database stats: {e}")
        return 0, []

def process_batch(uploaded_files):
    """
    Batch document ingestion with unified progress tracking.
    
    Handles:
    - Duplicate detection (skip already processed files)
    - Multi-step pipeline execution per file
    - Unified status container for batch visibility
    """
    if not uploaded_files:
        st.sidebar.warning("⚠️ No files selected for processing.")
        return
    
    # Get current database sources to check for duplicates
    _, db_sources = get_db_stats()
    
    # Filter out already processed files (check against database)
    new_files = [f for f in uploaded_files if f.name not in db_sources]
    skipped_files = [f for f in uploaded_files if f.name in db_sources]
    
    # Single status container for entire batch
    with st.sidebar.status(
        f"⚙️ Batch Processing: {len(new_files)} new, {len(skipped_files)} skipped",
        expanded=True
    ) as status:
        
        # Log skipped files
        if skipped_files:
            st.write("**📋 Skipped (Already Indexed):**")
            for f in skipped_files:
                st.caption(f"  ⏭️ {f.name}")
        
        if not new_files:
            status.update(label="✅ All files already processed", state="complete", expanded=False)
            return
        
        # Process new files
        st.write(f"\n**🚀 Processing {len(new_files)} New Documents:**")
        pipeline = IngestionPipeline()
        
        success_count = 0
        error_count = 0
        
        for idx, uploaded_file in enumerate(new_files, 1):
            try:
                st.write(f"\n**[{idx}/{len(new_files)}] {uploaded_file.name}**")
                
                # Step 1: Save to disk
                file_path = DATA_FOLDER / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.caption("  ✓ Saved to disk")
                
                # Step 2: Extract & Transform
                st.caption("  🔄 Running ingestion pipeline...")
                chunks = pipeline.run(str(file_path))
                
                # Step 3: Confirm indexing
                st.caption(f"  ✓ Indexed {len(chunks)} chunks")
                
                # Update state
                st.session_state.processed_files.add(uploaded_file.name)
                success_count += 1
                
            except Exception as e:
                st.error(f"  ❌ Failed: {str(e)}")
                error_count += 1
        
        # Final status update
        if error_count == 0:
            status.update(
                label=f"✅ Batch Complete: {success_count} documents indexed",
                state="complete",
                expanded=False
            )
        else:
            status.update(
                label=f"⚠️ Completed with errors: {success_count} OK, {error_count} failed",
                state="error",
                expanded=True
            )

def reset_system():
    """
    Complete system reset: clears chat history and wipes vector database.
    """
    with st.sidebar.status("🔄 Resetting System...", expanded=True) as status:
        try:
            st.write("1. Clearing chat history...")
            st.session_state.messages = []
            
            st.write("2. Wiping vector database...")
            # Reinitialize the vector store with reset=True
            VectorStore(reset=True)
            
            st.write("3. Clearing processed files tracker...")
            st.session_state.processed_files = set()
            
            st.write("4. Reinitializing RAG engine...")
            st.session_state.rag_engine = RAGEngine()
            
            status.update(label="✅ System Reset Complete", state="complete", expanded=False)
            st.success("✅ All data cleared. Ready for fresh start!")
            
        except Exception as e:
            status.update(label="❌ Reset Failed", state="error")
            st.error(f"Reset error: {str(e)}")

def delete_document(source_filename: str):
    """
    Delete a specific document from the vector database.
    """
    with st.sidebar.status(f"🗑️ Deleting {source_filename}...", expanded=True) as status:
        try:
            st.write("1. Removing chunks from vector database...")
            store = VectorStore()
            deleted_count = store.delete_by_source(source_filename)
            
            st.write(f"2. Deleted {deleted_count} chunks")
            
            st.write("3. Updating session state...")
            if source_filename in st.session_state.processed_files:
                st.session_state.processed_files.remove(source_filename)
            
            st.write("4. Reinitializing RAG engine...")
            # Reload engine to pick up changes
            if "rag_engine" in st.session_state:
                del st.session_state.rag_engine
            st.session_state.rag_engine = RAGEngine()
            
            status.update(
                label=f"✅ Deleted {source_filename}",
                state="complete",
                expanded=False
            )
            st.success(f"✅ {source_filename} removed successfully!")
            
        except Exception as e:
            status.update(label="❌ Deletion Failed", state="error")
            st.error(f"Error: {str(e)}")

def render_sidebar():
    """
    Structured Sidebar with Three Zones: Knowledge, Control, Actions
    """
    with st.sidebar:
        # ===== ZONE 1: KNOWLEDGE BASE (Top) =====
        with st.container():
            st.header("🗂️ Knowledge Base")
            
            # Multi-file uploader
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["pdf"],
                accept_multiple_files=True,
                help="Select one or multiple PDF files to add to the knowledge base"
            )
            
            # Batch process button
            if uploaded_files:
                if st.button("📥 Process All Documents", use_container_width=True, type="primary"):
                    process_batch(uploaded_files)
                    st.rerun()  # Refresh to show updated file count
            
            # Get real-time database statistics
            doc_count, indexed_sources = get_db_stats()
            
            # Active documents counter (from vector database)
            if doc_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Documents",
                        value=len(indexed_sources),
                        help="Number of unique source files"
                    )
                with col2:
                    st.metric(
                        label="Chunks",
                        value=doc_count,
                        help="Total indexed chunks"
                    )
                
                # Document Management: Show indexed files with delete buttons
                st.subheader("📚 Manage Documents")
                for source in sorted(indexed_sources):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"📄 {source}")
                    with col2:
                        # Unique key for each delete button
                        if st.button("🗑️", key=f"delete_{source}", help=f"Delete {source}"):
                            # Confirmation via session state
                            if f"confirm_delete_{source}" not in st.session_state:
                                st.session_state[f"confirm_delete_{source}"] = True
                                st.warning(f"⚠️ Click again to confirm deletion of {source}")
                                st.rerun()
                            else:
                                delete_document(source)
                                del st.session_state[f"confirm_delete_{source}"]
                                st.rerun()
            else:
                st.info("📭 No documents indexed yet. Upload PDFs to get started.")
        
        st.divider()  # Visual separator
        
        # ===== ZONE 2: QUERY CONTROLS (Middle) =====
        with st.container():
            st.subheader("⚙️ Query Settings")
            
            # HyDE Toggle
            hyde_enabled = st.toggle(
                "Enable HyDE",
                value=st.session_state.use_hyde,
                help=(
                    "**Hypothetical Document Embeddings (HyDE)**\n\n"
                    "When enabled, the system generates a hypothetical answer first, "
                    "then uses it to improve retrieval accuracy.\n\n"
                    "✅ Use when: Complex queries requiring deep reasoning\n"
                    "❌ Skip when: Simple factual lookups"
                )
            )
            st.session_state.use_hyde = hyde_enabled
            
            if hyde_enabled:
                st.success("🔬 HyDE Active: Enhanced retrieval mode")
            else:
                st.caption("🔍 Standard Mode: Direct vector search")
            
            # Future: Reranking configuration could go here
            # st.slider("Reranker Top-K", min_value=3, max_value=10, value=5)
        
        st.divider()  # Visual separator
        
        # ===== ZONE 3: ACTIONS (Bottom - Always Visible) =====
        with st.container():
            st.subheader("🛠️ Actions")
            
            # New Chat button (clears chat only)
            if st.button("💬 Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            # Cancel confirmation if user navigates away
            if "confirm_reset" in st.session_state and st.session_state.get("cancel_reset"):
                del st.session_state.confirm_reset
                st.rerun()

def render_chat():
    """
    Main chat interface with citation support
    """
    st.title("📘 Citation-Aware RAG Assistant")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Render sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        # Extract rerank score if available
                        rerank_score = doc.metadata.get('rerank_score')
                        score_badge = f" `Confidence: {rerank_score:.2%}`" if rerank_score else ""
                        
                        st.markdown(f"**Source {i}:** {doc.source} (Page {doc.page}){score_badge}")
                        st.text(doc.content[:300] + "..." if len(doc.content) > 300 else doc.content)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are loaded
        doc_count, _ = get_db_stats()
        if doc_count == 0:
            st.warning("⚠️ Please upload and process documents before querying.")
            return
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Dynamic status based on settings
            mode_label = "🔬 HyDE Mode" if st.session_state.use_hyde else "🔍 Standard Mode"
            
            with st.spinner(f"{mode_label} | Analyzing documents..."):
                try:
                    # Call RAG engine with HyDE toggle
                    response = st.session_state.rag_engine.query(
                        prompt,
                        use_hyde=st.session_state.use_hyde
                    )
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    error_msg = f"⚠️ Query failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

def main():
    """
    Application entry point
    """
    initialize_state()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()