"""Chunk models for RAG integration."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of document content for RAG."""

    chunk_id: str
    text: str
    token_count: int

    # Source metadata
    source_path: Path
    document_id: Optional[str] = None

    # Position metadata
    page_no: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int

    # Structural metadata
    is_table: bool = False
    is_heading: bool = False
    heading_level: Optional[int] = None

    # Bounding box for visual grounding
    bbox: Optional[Dict[str, float]] = None  # {l, t, r, b}

    # Parent context for hierarchical chunking
    parent_section: Optional[str] = None
    subsection_path: Optional[str] = None  # e.g., "Chapter 1 > Section 1.2 > Subsection 1.2.3"

    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary for vector store ingestion.

        Returns:
            Dictionary with all chunk data
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "source_path": str(self.source_path),
            "document_id": self.document_id,
            "page_no": self.page_no,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "is_table": self.is_table,
            "is_heading": self.is_heading,
            "heading_level": self.heading_level,
            "bbox": self.bbox,
            "parent_section": self.parent_section,
            "subsection_path": self.subsection_path,
            **self.metadata,
        }

    def get_context_string(self) -> str:
        """
        Get a context string that includes section information.

        Returns:
            Formatted string with section context
        """
        parts = []

        if self.subsection_path:
            parts.append(f"Context: {self.subsection_path}")

        if self.section_title:
            parts.append(f"Section: {self.section_title}")

        if self.page_no:
            parts.append(f"Page: {self.page_no}")

        parts.append(self.text)

        return "\n".join(parts)


class ChunkingResult(BaseModel):
    """Result of chunking a document."""

    source_path: Path
    chunks: list[DocumentChunk]
    total_chunks: int
    total_tokens: int
    chunking_strategy: str  # e.g., "hybrid", "hierarchical", "fixed"

    # Statistics
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int

    def __init__(self, **data):
        super().__init__(**data)
        if not self.total_chunks:
            self.total_chunks = len(self.chunks)
        if not self.total_tokens:
            self.total_tokens = sum(c.token_count for c in self.chunks)
        if not self.avg_chunk_size:
            self.avg_chunk_size = (
                self.total_tokens / self.total_chunks if self.total_chunks > 0 else 0
            )
        if not self.min_chunk_size:
            self.min_chunk_size = (
                min(c.token_count for c in self.chunks) if self.chunks else 0
            )
        if not self.max_chunk_size:
            self.max_chunk_size = (
                max(c.token_count for c in self.chunks) if self.chunks else 0
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chunking statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "avg_chunk_size": self.avg_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunking_strategy": self.chunking_strategy,
        }
