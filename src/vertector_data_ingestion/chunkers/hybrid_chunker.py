"""HybridChunker implementation for RAG-ready document chunking."""

from pathlib import Path

from docling_core.transforms.chunker import HybridChunker as DoclingHybridChunker
from loguru import logger
from transformers import AutoTokenizer

from vertector_data_ingestion.models.chunk import ChunkingResult, DocumentChunk
from vertector_data_ingestion.models.config import ChunkingConfig
from vertector_data_ingestion.models.document import DoclingDocumentWrapper
from vertector_data_ingestion.utils.model_utils import get_padding_side


class HybridChunker:
    """
    Wrapper around Docling's HybridChunker with metadata enrichment.

    Provides hierarchical, tokenization-aware chunking with:
    - Structural boundary splitting (sections, headers, tables)
    - Greedy merging of small chunks
    - Metadata enrichment (page numbers, sections, bboxes)
    """

    def __init__(self, config: ChunkingConfig | None = None):
        """
        Initialize hybrid chunker.

        Args:
            config: Chunking configuration (uses defaults if None)
        """
        self.config = config or ChunkingConfig()

        # Auto-detect padding side based on model type
        padding_side = get_padding_side(self.config.tokenizer)

        # Load tokenizer with appropriate padding side
        logger.info(f"Loading tokenizer: {self.config.tokenizer} (padding_side={padding_side})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer, padding_side=padding_side
        )

        # Create Docling chunker
        self.chunker = DoclingHybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.config.max_tokens,
            merge_peers=self.config.merge_peers,
        )

        logger.info(
            f"Initialized HybridChunker with max_tokens={self.config.max_tokens}, "
            f"merge_peers={self.config.merge_peers}"
        )

    def chunk_document(
        self, doc_wrapper: DoclingDocumentWrapper, include_metadata: bool = True
    ) -> ChunkingResult:
        """
        Chunk document into RAG-ready chunks.

        Args:
            doc_wrapper: Document wrapper to chunk
            include_metadata: Whether to enrich chunks with metadata

        Returns:
            ChunkingResult with list of chunks and statistics
        """
        logger.info(
            f"Chunking document: {doc_wrapper.metadata.source_path.name} "
            f"({doc_wrapper.metadata.num_pages} pages)"
        )

        chunks = []
        doc = doc_wrapper.doc

        # Chunk using Docling's HybridChunker
        for idx, chunk in enumerate(self.chunker.chunk(doc)):
            # Get token count
            token_count = len(self.tokenizer.encode(chunk.text))

            # Extract metadata if enabled
            metadata = {}
            if include_metadata:
                metadata = self._extract_metadata(chunk, doc_wrapper, idx)

            # Create DocumentChunk
            chunk_obj = DocumentChunk(
                chunk_id=f"{doc_wrapper.metadata.source_path.stem}_{idx}",
                text=chunk.text,
                token_count=token_count,
                source_path=doc_wrapper.metadata.source_path,
                document_id=doc_wrapper.metadata.source_path.stem,
                chunk_index=idx,
                **metadata,
            )

            chunks.append(chunk_obj)

        logger.info(f"Created {len(chunks)} chunks")

        # Create result
        result = ChunkingResult(
            source_path=doc_wrapper.metadata.source_path,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
            chunking_strategy="hybrid",
            avg_chunk_size=0,  # Will be computed in __init__
            min_chunk_size=0,
            max_chunk_size=0,
        )

        logger.debug(
            f"Chunking stats: avg={result.avg_chunk_size:.1f}, "
            f"min={result.min_chunk_size}, max={result.max_chunk_size} tokens"
        )

        return result

    def _extract_metadata(
        self, chunk, doc_wrapper: DoclingDocumentWrapper, chunk_index: int
    ) -> dict:
        """
        Extract metadata from chunk.

        Args:
            chunk: Docling chunk object
            doc_wrapper: Document wrapper
            chunk_index: Index of chunk

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Try to extract page number
        if hasattr(chunk, "meta") and chunk.meta:
            # Extract from chunk metadata
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                first_item = chunk.meta.doc_items[0]

                # Page number
                if hasattr(first_item, "prov") and first_item.prov:
                    metadata["page_no"] = first_item.prov[0].page_no

                    # Bounding box
                    if hasattr(first_item.prov[0], "bbox"):
                        bbox = first_item.prov[0].bbox
                        metadata["bbox"] = {
                            "l": bbox.l,
                            "t": bbox.t,
                            "r": bbox.r,
                            "b": bbox.b,
                        }

                # Section title (heading)
                if hasattr(first_item, "obj_type"):
                    if first_item.obj_type == "table":
                        metadata["is_table"] = True
                    elif first_item.obj_type in ["section-header", "heading"]:
                        metadata["is_heading"] = True
                        if hasattr(first_item, "text"):
                            metadata["section_title"] = first_item.text

                        # Heading level
                        if hasattr(first_item, "level"):
                            metadata["heading_level"] = first_item.level

        # Build hierarchical section path
        metadata["subsection_path"] = self._build_section_path(chunk, doc_wrapper)

        return metadata

    def _build_section_path(self, chunk, doc_wrapper: DoclingDocumentWrapper) -> str:
        """
        Build hierarchical section path for chunk.

        Args:
            chunk: Docling chunk object
            doc_wrapper: Document wrapper

        Returns:
            Section path string (e.g., "Chapter 1 > Section 1.2")
        """
        # This is a simplified implementation
        # A full implementation would track document structure hierarchy

        if hasattr(chunk, "meta") and chunk.meta:
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                return " > ".join(chunk.meta.headings)

        return ""

    def chunk_text(self, text: str, source_path: Path) -> list[DocumentChunk]:
        """
        Chunk plain text (without Docling document structure).

        Args:
            text: Plain text to chunk
            source_path: Source file path for metadata

        Returns:
            List of document chunks
        """
        chunks = []

        # Simple sliding window chunking for plain text
        tokens = self.tokenizer.encode(text)
        max_tokens = self.config.max_tokens

        for idx, start in enumerate(range(0, len(tokens), max_tokens)):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk = DocumentChunk(
                chunk_id=f"{source_path.stem}_{idx}",
                text=chunk_text,
                token_count=len(chunk_tokens),
                source_path=source_path,
                document_id=source_path.stem,
                chunk_index=idx,
            )

            chunks.append(chunk)

        return chunks
