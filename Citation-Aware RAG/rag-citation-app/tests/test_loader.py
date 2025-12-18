import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.ingestion.loader import PDFLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TEST_PDF_PATH = Config.DATA_DIR / "Docker Deep Dive by Nigel Poulton.pdf"

def test_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        PDFLoader(Path("non_existent_file.pdf"))

def test_loader_extracts_content():
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    loader = PDFLoader(TEST_PDF_PATH)
    documents = loader.load()
    assert len(documents) > 0, "No documents were extracted from the PDF."
    assert documents[0].source == TEST_PDF_PATH.name, "Source metadata does not match the PDF filename."

def test_toc_extraction():
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    loader = PDFLoader(TEST_PDF_PATH)
    documents = loader.load()
    
    toc_pages = [doc for doc in documents if "Table of Contents" in doc.content]
    assert len(toc_pages) > 0, "No Table of Contents pages were extracted."
    