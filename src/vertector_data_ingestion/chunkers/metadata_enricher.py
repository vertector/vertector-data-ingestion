"""Utilities for enriching chunks with metadata."""

from typing import Any, Dict, List, Optional

from vertector_data_ingestion.models.chunk import DocumentChunk


class MetadataEnricher:
    """Utilities for enriching document chunks with additional metadata."""

    @staticmethod
    def add_custom_metadata(
        chunk: DocumentChunk, metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """
        Add custom metadata to chunk.

        Args:
            chunk: Document chunk
            metadata: Custom metadata dictionary

        Returns:
            Chunk with updated metadata
        """
        chunk.metadata.update(metadata)
        return chunk

    @staticmethod
    def enrich_with_keywords(
        chunks: List[DocumentChunk], keywords: List[str]
    ) -> List[DocumentChunk]:
        """
        Add keyword matches to chunks.

        Args:
            chunks: List of chunks
            keywords: Keywords to match

        Returns:
            Chunks with keyword metadata
        """
        for chunk in chunks:
            matched_keywords = [kw for kw in keywords if kw.lower() in chunk.text.lower()]
            if matched_keywords:
                chunk.metadata["matched_keywords"] = matched_keywords

        return chunks

    @staticmethod
    def add_embeddings(
        chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> List[DocumentChunk]:
        """
        Add precomputed embeddings to chunks.

        Args:
            chunks: List of chunks
            embeddings: List of embedding vectors

        Returns:
            Chunks with embedding metadata
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata["embedding"] = embedding

        return chunks

    @staticmethod
    def add_summary(chunks: List[DocumentChunk], summaries: List[str]) -> List[DocumentChunk]:
        """
        Add summaries to chunks.

        Args:
            chunks: List of chunks
            summaries: List of chunk summaries

        Returns:
            Chunks with summary metadata
        """
        if len(chunks) != len(summaries):
            raise ValueError("Number of chunks and summaries must match")

        for chunk, summary in zip(chunks, summaries):
            chunk.metadata["summary"] = summary

        return chunks

    @staticmethod
    def filter_by_metadata(
        chunks: List[DocumentChunk],
        key: str,
        value: Any,
        match_exact: bool = True,
    ) -> List[DocumentChunk]:
        """
        Filter chunks by metadata value.

        Args:
            chunks: List of chunks
            key: Metadata key to filter on
            value: Value to match
            match_exact: If True, require exact match; if False, check containment

        Returns:
            Filtered list of chunks
        """
        filtered = []

        for chunk in chunks:
            if key in chunk.metadata:
                chunk_value = chunk.metadata[key]

                if match_exact:
                    if chunk_value == value:
                        filtered.append(chunk)
                else:
                    if value in str(chunk_value):
                        filtered.append(chunk)

        return filtered
