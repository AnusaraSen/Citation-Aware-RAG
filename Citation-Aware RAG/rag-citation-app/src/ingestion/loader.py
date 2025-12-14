import fitz  
import hashlib
from typing import List, Dict, Any, Tuple, Union
from pathlib import Path
from src.core.types import Document

# Block format: (x0: float, y0: float, x1: float, y1: float, text: str, block_no: int, block_type: int)
BlockTuple = Tuple[float, float, float, float, str, int, int]

class PDFLoader:
    
    
    # Layout constants (in pixels)
    HEADER_THRESHOLD = 50  # Ignore blocks in top 50px
    FOOTER_THRESHOLD = 50  # Ignore blocks in bottom 50px
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.file_path}")

    def _sort_blocks_spatially(self, blocks: List[BlockTuple], page_height: float) -> List[BlockTuple]:
        """
        Sort blocks by vertical position (y0), then horizontal position (x0).
        Critical for preserving reading order in multi-column layouts.
        
        Args:
            blocks: List of block tuples (x0, y0, x1, y1, text, block_no, block_type)
            page_height: Height of the page for coordinate validation
        
        Returns:
            Sorted list of blocks in natural reading order
        """
        # Filter out non-text blocks and apply header/footer filtering
        text_blocks: List[BlockTuple] = []
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            
            # Only process text blocks (type 0)
            if block_type != 0:
                continue
            
            # Filter header/footer noise
            if y0 < self.HEADER_THRESHOLD:
                continue
            if y1 > (page_height - self.FOOTER_THRESHOLD):
                continue
            
            text_blocks.append(block)
        
        # Sort: Primary by vertical position (y0), Secondary by horizontal (x0)
        # make a grid like structure to sort the numbers properly
        sorted_blocks = sorted(text_blocks, key=lambda b: (round(b[1], -1), b[0]))
        
        return sorted_blocks

    def _extract_tables_as_markdown(self, page) -> Dict[Tuple[float, float], str]:
        """
        Detect tables and convert them to Markdown format.
        
        Returns:
            Dictionary mapping (y0, y1) coordinates to Markdown table strings
        """
        tables_map: Dict[Tuple[float, float], str] = {}
        
        try:
            tables = page.find_tables()
            
            for table in tables:
                # Extract table data
                table_data = table.extract()
                
                if not table_data:
                    continue
                
                # Convert to Markdown
                markdown_lines = []
                
                # akes the first row of table_data (assumed to be the header row) and converts it to a Markdown table header plus the required separator row of column dashes.
                if len(table_data) > 0:
                    header = table_data[0]
                    markdown_lines.append("| " + " | ".join(str(cell) if cell else "" for cell in header) + " |")
                    markdown_lines.append("|" + "|".join(["---"] * len(header)) + "|")
                
                # Data rows
                for row in table_data[1:]:
                    markdown_lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
                
                markdown_table = "\n".join(markdown_lines)
                
                # Store with vertical coordinates for positioning the table
                bbox = table.bbox
                tables_map[(bbox[1], bbox[3])] = markdown_table
                
        except Exception as e:
            # Silently continue if table detection fails
            pass
        
        return tables_map

    def get_toc_summary(self) -> str:
        """
        Extract PDF's Table of Contents (bookmarks) as a Markdown tree.
        
        Returns:
            Formatted Markdown string representing the document structure
        """
        try:
            with fitz.open(self.file_path) as pdf_doc:
                toc = pdf_doc.get_toc()
                
                if not toc:
                    return "**Table of Contents**\n\n_No bookmarks found in this document._"
                
                markdown_lines = ["**Table of Contents**\n"]
                
                for level, title, page_num in toc:
                    # Indent based on heading level
                    indent = "  " * (level - 1)
                    markdown_lines.append(f"{indent}- {title} (Page {page_num})")
                
                return "\n".join(markdown_lines)
                
        except Exception as e:
            return f"**Table of Contents**\n\n_Error extracting TOC: {str(e)}_"

    def load(self) -> List[Document]: #List of Document objects with rich metadata
        
        docs: List[Document] = []
        
        try:
            with fitz.open(self.file_path) as pdf_doc:
                
                # ---  Inject TOC as first document ---
                toc_content = self.get_toc_summary()
                toc_doc = Document(
                    id=hashlib.md5(f"{self.file_path.name}_toc".encode()).hexdigest(),
                    content=toc_content,
                    source=self.file_path.name,
                    page=0,  # Special page number for TOC
                    metadata={
                        "is_toc": True,
                        "total_pages": len(pdf_doc),
                        "document_structure": "table_of_contents"
                    }
                )
                docs.append(toc_doc)
                
                # --- Step 2: Process each page ---
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    page_rect = page.rect # Page’s bounding rectangle
                    page_height = page_rect.height # height of the page in PDF units
                    
                    # Extract tables as Markdown
                    tables_map = self._extract_tables_as_markdown(page)
                    
                    # Extract text blocks with proper type casting
                    raw_blocks = page.get_text("blocks")
                    # Cast to our typed structure

                    '''
                           x0 , y0 - Top-left corner of the text block
                           x1 , y1 - Bottom-right corner of the text block
                           text    - The actual text content of the block
                           block_no - Unique identifier for the block on the page
                           block_type - Integer indicating the type of block (0=text, 1=image, etc.)
                    '''
                    typed_blocks: List[BlockTuple] = [
                        (
                            
                            float(b[0]),  # x0
                            float(b[1]),  # y0
                            float(b[2]),  # x1
                            float(b[3]),  # y1
                            str(b[4]),    # text
                            int(b[5]),    # block_no
                            int(b[6])     # block_type
                        )

                        for b in raw_blocks
                    ]
                    
                    # Sort blocks spatially for proper reading order
                    sorted_blocks = self._sort_blocks_spatially(typed_blocks, page_height)
                    
                    # Merge text blocks and tables
                    content_parts: List[str] = []
                    table_positions = set(tables_map.keys())
                    
                    for block in sorted_blocks:
                        x0, y0, x1, y1, text, block_no, block_type = block
                        
                        # Check if this block overlaps with a table
                        is_in_table = False
                        for (table_y0, table_y1) in table_positions: 
                            if table_y0 <= y0 <= table_y1 or table_y0 <= y1 <= table_y1: # Checks top or bottom of the text block falls inside a table’s vertical area 
                                # Insert table if not already added
                                if (table_y0, table_y1) in tables_map:
                                    content_parts.append(f"\n{tables_map[(table_y0, table_y1)]}\n")
                                    del tables_map[(table_y0, table_y1)]  # Remove it from tables_map to avoid duplicates
                                is_in_table = True
                                break
                        
                        # Add text block if not part of a table
                        if not is_in_table:
                            content_parts.append(text.strip())
                    
                    # Add any remaining tables
                    for table_md in tables_map.values():
                        content_parts.append(f"\n{table_md}\n")
                    
                    # Join with double newlines to preserve paragraph structure
                    final_text = "\n\n".join(filter(None, content_parts))
                    
                    # Skip empty pages
                    if not final_text.strip():
                        continue
                    
                    # generating a deterministic (The same input will always produce the same ID), repeatable unique ID for a document chunk
                    chunk_id = hashlib.md5(
                        f"{self.file_path.name}_{page_num}".encode() # Convert string → bytes
                    ).hexdigest()
                    
                    # Create Document with enriched metadata
                    document = Document(
                        id=chunk_id,
                        content=final_text,
                        source=self.file_path.name,
                        page=page_num + 1,
                        metadata={
                            "total_pages": len(pdf_doc),
                            "block_count": len(sorted_blocks),
                            "table_count": len(self._extract_tables_as_markdown(page)),
                            "extraction_method": "layout_aware_blocks",
                            "is_toc": False
                        }
                    )
                    docs.append(document)
                    
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse PDF {self.file_path.name}: {str(e)}"
            ) from e
            
        return docs


