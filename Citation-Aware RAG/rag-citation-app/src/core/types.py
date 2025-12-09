from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Document(BaseModel):
    """
    Represents a single chunk or page of text from a source document.
    
    This adheres to the 'Metadata Injection' requirement in the project guide,
    ensuring every piece of text carries its source provenance.
    """
    id: str = Field(..., description="Unique identifier for the document chunk")
    content: str = Field(..., description="The actual text content")
    source: str = Field(..., description="Filename of the source document")
    page: int = Field(..., description="Page number (1-based index)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context (e.g., bounding boxes)")

    class Config:
        frozen = True  # Makes instances immutable (Senior pattern: prevents accidental data mutation)