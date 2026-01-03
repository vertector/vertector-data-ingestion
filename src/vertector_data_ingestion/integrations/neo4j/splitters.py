"""Text splitters for Neo4j SimpleKGPipeline integration.

This module provides TextSplitter implementations that integrate Vertector's
HybridChunker with Neo4j's SimpleKGPipeline, preserving Docling's rich metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer

from vertector_data_ingestion import HybridChunker
from vertector_data_ingestion.models.chunk import DocumentChunk
from vertector_data_ingestion.models.config import ChunkingConfig

if TYPE_CHECKING:
    from vertector_data_ingestion.integrations.neo4j.loaders import (
        MultimodalLoader,
        VertectorAudioLoader,
        VertectorDataLoader,
    )

try:
    from neo4j_graphrag.experimental.components.text_splitters.base import (
        TextChunks,
        TextSplitter,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk
except ImportError as e:
    msg = (
        "neo4j-graphrag is required for Neo4j integration. "
        "Install with: uv pip install vertector-data-ingestion[neo4j]"
    )
    raise ImportError(msg) from e


class VertectorTextSplitter(TextSplitter):
    """Text splitter using Vertector's HybridChunker with rich Docling metadata.

    This splitter leverages Docling's HybridChunker via chunk_document() to preserve:
    - Hierarchical structure (sections, headings)
    - Page numbers and bounding boxes
    - Table detection
    - Section titles and hierarchy paths

    It accesses the DoclingDocumentWrapper from the loader's last_document attribute
    to perform structure-aware chunking, then extracts chunk.text for Neo4j.

    Args:
        loader: VertectorDataLoader instance to access last_document
        chunk_size: Maximum tokens per chunk (default: 512)
        tokenizer: HuggingFace tokenizer model name (default: Qwen/Qwen3-Embedding-0.6B)

    Attributes:
        loader: Reference to data loader for accessing documents
        chunker: HybridChunker instance
        last_chunks: List of DocumentChunk objects from last split (with rich metadata)

    Example:
        >>> from vertector_data_ingestion.integrations.neo4j import (
        ...     VertectorDataLoader,
        ...     VertectorTextSplitter
        ... )
        >>>
        >>> loader = VertectorDataLoader()
        >>> splitter = VertectorTextSplitter(loader=loader, chunk_size=512)
        >>>
        >>> # Load document (stores DoclingDocumentWrapper in loader.last_document)
        >>> doc_result = await loader.run(Path("paper.pdf"))
        >>>
        >>> # Split using rich document structure
        >>> chunks = await splitter.run(doc_result.text)
        >>>
        >>> # Access rich metadata
        >>> for chunk in chunks.chunks:
        ...     print(f"Page: {chunk.metadata.get('page_no')}")
        ...     print(f"Section: {chunk.metadata.get('subsection_path')}")
    """

    def __init__(
        self,
        loader: VertectorDataLoader | VertectorAudioLoader | MultimodalLoader,
        chunk_size: int = 512,
        tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    ) -> None:
        """Initialize text splitter with HybridChunker.

        Args:
            loader: Loader instance (VertectorDataLoader, VertectorAudioLoader, or MultimodalLoader)
            chunk_size: Maximum tokens per chunk (must be > 0)
            tokenizer: HuggingFace tokenizer model name

        Raises:
            ValueError: If chunk_size is not positive
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)

        self.loader = loader
        self.tokenizer_name = tokenizer
        config = ChunkingConfig(
            tokenizer=tokenizer,
            max_tokens=chunk_size,
        )
        self.chunker = HybridChunker(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)  # For audio chunking
        self.last_chunks: list[DocumentChunk] = []

    async def run(self, text: str) -> TextChunks:
        """Split document or audio using appropriate chunking strategy.

        For documents: Uses Docling's structure-aware chunking via chunk_document()
        For audio: Creates chunks from Whisper transcription segments

        Args:
            text: Input text (not used directly for documents; we access loader state)

        Returns:
            TextChunks containing list of Neo4j TextChunk objects with rich metadata

        Raises:
            RuntimeError: If no document or audio was loaded

        Example:
            >>> # After loading a document
            >>> chunks = await splitter.run(loaded_text)
            >>> print(chunks.chunks[0].metadata)
            {'page_no': '1', 'section_title': 'Introduction', ...}
            >>>
            >>> # After loading audio
            >>> chunks = await splitter.run(loaded_text)
            >>> print(chunks.chunks[0].metadata)
            {'start_time': '0.0', 'end_time': '5.5', 'duration': '5.5', ...}
        """
        # Check if audio was loaded
        if (
            hasattr(self.loader, "last_transcription_result")
            and self.loader.last_transcription_result
        ):
            return await self._chunk_audio()

        # Check if document was loaded
        if hasattr(self.loader, "last_document") and self.loader.last_document:
            return await self._chunk_document()

        # No content available
        msg = (
            "No document or audio available in loader. "
            "Ensure content was loaded before calling run()."
        )
        raise RuntimeError(msg)

    async def _chunk_document(self) -> TextChunks:
        """Chunk document using Docling's HybridChunker."""
        doc_wrapper = self.loader.last_document

        # Use chunk_document() to get rich metadata chunks
        chunking_result = self.chunker.chunk_document(doc_wrapper, include_metadata=True)

        # Store DocumentChunk objects for metadata access
        self.last_chunks = chunking_result.chunks

        # Convert DocumentChunk → Neo4j TextChunk
        neo4j_chunks = []
        for doc_chunk in chunking_result.chunks:
            # Extract text from chunk (preserves Docling's structure-aware text)
            chunk_text = doc_chunk.text

            # Build metadata dictionary from rich DocumentChunk
            metadata = self._extract_metadata(doc_chunk)

            # Create Neo4j TextChunk
            text_chunk = TextChunk(
                text=chunk_text,
                index=doc_chunk.chunk_index,
                metadata=metadata,
            )
            neo4j_chunks.append(text_chunk)

        return TextChunks(chunks=neo4j_chunks)

    async def _chunk_audio(self) -> TextChunks:
        """Chunk audio using Whisper transcription segments."""
        from pathlib import Path

        result = self.loader.last_transcription_result

        # Get audio filename from loader metadata
        audio_filename = self.loader.last_metadata.get("filename", "audio")
        document_id = Path(audio_filename).stem  # Use filename without extension as document_id

        # Create DocumentChunk objects from segments (mirroring notebook 04)
        doc_chunks = []
        for i, segment in enumerate(result.segments):
            # Count tokens in the segment text
            tokens = self.tokenizer.encode(segment.text, add_special_tokens=False)

            # Create DocumentChunk with audio metadata
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_{i}",
                text=segment.text,
                token_count=len(tokens),
                source_path=Path(audio_filename),
                document_id=document_id,
                chunk_index=i,
                metadata={
                    "modality": "audio",
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "duration": segment.end - segment.start,
                    "language": result.language,
                },
            )
            doc_chunks.append(chunk)

        # Store DocumentChunk objects
        self.last_chunks = doc_chunks

        # Convert to Neo4j TextChunk format
        neo4j_chunks = []
        for doc_chunk in doc_chunks:
            metadata = self._extract_metadata(doc_chunk)

            text_chunk = TextChunk(
                text=doc_chunk.text,
                index=doc_chunk.chunk_index,
                metadata=metadata,
            )
            neo4j_chunks.append(text_chunk)

        return TextChunks(chunks=neo4j_chunks)

    def _extract_metadata(self, chunk: DocumentChunk) -> dict[str, str]:
        """Extract metadata from DocumentChunk.

        Converts all metadata values to strings for Neo4j compatibility.

        Args:
            chunk: DocumentChunk with rich Docling metadata

        Returns:
            Dictionary with string-valued metadata including page_no, section_title,
            subsection_path, bbox, is_table, is_heading, etc.
        """
        metadata: dict[str, str] = {
            "chunk_id": chunk.chunk_id,
            "token_count": str(chunk.token_count),
            "document_id": chunk.document_id,
        }

        # Add optional fields if present
        if chunk.page_no is not None:
            metadata["page_no"] = str(chunk.page_no)

        if chunk.section_title:
            metadata["section_title"] = chunk.section_title

        if chunk.is_table:
            metadata["is_table"] = str(chunk.is_table)

        if chunk.is_heading:
            metadata["is_heading"] = str(chunk.is_heading)

        if chunk.heading_level is not None:
            metadata["heading_level"] = str(chunk.heading_level)

        if chunk.parent_section:
            metadata["parent_section"] = chunk.parent_section

        if chunk.subsection_path:
            metadata["subsection_path"] = chunk.subsection_path

        if chunk.bbox:
            # bbox is dict with keys: 'l', 't', 'r', 'b'
            bbox = chunk.bbox
            metadata["bbox"] = f"{bbox['l']},{bbox['t']},{bbox['r']},{bbox['b']}"

        # Merge additional metadata from chunk.metadata dict
        if chunk.metadata:
            for key, value in chunk.metadata.items():
                # Skip nested dicts
                if not isinstance(value, dict):
                    metadata[key] = str(value)

        return metadata

    def get_chunk_metadata(self, chunk_index: int) -> dict[str, str]:
        """Get metadata for a specific chunk by index.

        This allows retrieving full DocumentChunk metadata after splitting.

        Args:
            chunk_index: Index of chunk (0-based)

        Returns:
            Dictionary with chunk metadata, empty dict if index out of range

        Example:
            >>> chunks = await splitter.run(text)
            >>> meta = splitter.get_chunk_metadata(0)
            >>> print(meta["page_no"])
            '1'
            >>> print(meta["subsection_path"])
            'DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis'
        """
        if chunk_index >= len(self.last_chunks) or chunk_index < 0:
            return {}

        chunk = self.last_chunks[chunk_index]
        return self._extract_metadata(chunk)
